#!/usr/bin/env python3
"""
Run SmolVLA policy in an Isaac Lab SO-101 env (lift-cube or reach).
Uses the isaac_so_arm101 extension in-repo (SO-101 URDF and tasks).
Run from Isaac Lab: ./isaaclab.sh -p /path/to/Isaac/run_smolvla_isaac.py --task <task_id> [options]
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

# #region agent log
def _dbg(msg: str, data: dict, hypothesis_id: str = ""):
    import json
    try:
        _logpath = Path(__file__).resolve().parent / ".cursor" / "debug.log"
        _logpath.parent.mkdir(parents=True, exist_ok=True)
        with open(_logpath, "a") as f:
            f.write(json.dumps({"message": msg, "data": data, "hypothesisId": hypothesis_id, "timestamp": __import__("time").time()}) + "\n")
    except Exception:
        pass
# #endregion

# Project root (directory containing this script and isaac_so_arm101/)
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
# Register SO-101 envs from in-repo extension (reach + lift)
_EXTENSION_SRC = _SCRIPT_DIR / "isaac_so_arm101" / "src"
if _EXTENSION_SRC.exists() and str(_EXTENSION_SRC) not in sys.path:
    sys.path.insert(0, str(_EXTENSION_SRC))
_ext_import_ok = False
_ext_import_err = None
try:
    import isaac_so_arm101.tasks.reach  # noqa: F401  # registers Isaac-SO-ARM*-Reach-v0
    import isaac_so_arm101.tasks.lift   # noqa: F401  # registers Isaac-SO-ARM*-Lift-Cube-v0
    _ext_import_ok = True
except ImportError as e:
    _ext_import_err = f"{type(e).__name__}: {e}"
# #region agent log
_dbg("extension_path_and_import", {"script_dir": str(_SCRIPT_DIR), "extension_src": str(_EXTENSION_SRC), "extension_src_exists": _EXTENSION_SRC.exists(), "ext_import_ok": _ext_import_ok, "ext_import_err": _ext_import_err}, "H1")
# #endregion

import gymnasium as gym
import numpy as np

from adapters import isaac_obs_to_policy_frame, policy_action_to_env
from env_wrapper import IsaacEEWrapper


def parse_args():
    p = argparse.ArgumentParser(description="SmolVLA inference on Isaac Lab SO-101 env")
    p.add_argument("--task", type=str, default="Isaac-SO-ARM101-Lift-Cube-v0", help="Gym task id (e.g. Isaac-SO-ARM101-Lift-Cube-v0, Isaac-SO-ARM101-Reach-v0)")
    p.add_argument("--headless", action="store_true", help="Run without GUI")
    p.add_argument("--num_envs", type=int, default=1, help="Number of parallel envs")
    p.add_argument("--policy", type=str, default="lerobot/smolvla_base", help="HuggingFace policy repo or path")
    p.add_argument("--instruction", type=str, default="Pick the cube.", help="Language instruction for the policy")
    p.add_argument("--max_steps", type=int, default=500, help="Max steps per episode")
    p.add_argument("--episodes", type=int, default=3, help="Number of episodes to run")
    p.add_argument("--robot_name", type=str, default="robot", help="Name of robot articulation in scene (isaac_so_arm101 uses 'robot')")
    p.add_argument("--ee_link_name", type=str, default="gripper_link", help="End-effector link name (SO-101: gripper_link)")
    p.add_argument("--no_ee_in_obs", action="store_true", help="Do not add ee_pos/ee_quat/delta to obs dict")
    return p.parse_args()


def main():
    args = parse_args()

    # Load policy and processors (LeRobot)
    try:
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    except ImportError as e:
        print("LeRobot/SmolVLA not installed. Install with: pip install 'lerobot[smolvla]'", file=sys.stderr)
        raise SystemExit(1) from e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading policy from {args.policy} on {device} ...")
    policy = SmolVLAPolicy.from_pretrained(args.policy).to(device).eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        args.policy,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    # Create Isaac Lab env (must be run via isaaclab.sh so that Isaac Lab and SO-101 task are available)
    # #region agent log
    _reg = getattr(getattr(gym, "envs", None), "registry", None)
    _ids = [k for k in (_reg.keys() if _reg else []) if "SO" in k or "Isaac" in k][:20]
    _dbg("gym_make_call", {"task": args.task, "num_envs": args.num_envs, "headless": args.headless, "task_in_registry": args.task in (_reg or {}), "sample_registry_ids": _ids}, "H2")
    # #endregion
    try:
        env = gym.make(args.task, num_envs=args.num_envs, headless=args.headless)
    except Exception as e:
        # #region agent log
        _dbg("gym_make_failed", {"exc_type": type(e).__name__, "exc_msg": str(e), "traceback": traceback.format_exc()}, "H3")
        # #endregion
        print(
            "Failed to create env. Run this script with Isaac Lab: ./isaaclab.sh -p /path/to/Isaac/run_smolvla_isaac.py ...",
            file=sys.stderr,
        )
        print("Ensure isaac_so_arm101 is on PYTHONPATH (in-repo: isaac_so_arm101/src is added automatically).", file=sys.stderr)
        raise SystemExit(1) from e

    env = IsaacEEWrapper(
        env,
        robot_name=args.robot_name,
        ee_link_name=args.ee_link_name,
        add_ee_to_obs=not args.no_ee_in_obs,
    )

    # Infer action space shape for adapter
    action_space = env.action_space
    if hasattr(action_space, "shape"):
        action_shape = tuple(action_space.shape)
    else:
        action_shape = (getattr(action_space, "n", 7),)

    for ep in range(args.episodes):
        obs, info = env.reset()
        ee_state = info.get("ee_state", {})
        print(f"Episode {ep + 1}/{args.episodes} started. EE pos: {ee_state.get('ee_pos', 'N/A')}")
        step = 0
        while step < args.max_steps:
            # Use first env if batched (num_envs > 1) for policy input
            if isinstance(obs, dict):
                single_obs = {}
                for k, v in obs.items():
                    if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == args.num_envs:
                        single_obs[k] = v[0]
                    else:
                        single_obs[k] = v
            else:
                single_obs = {"obs": obs[0] if (hasattr(obs, "shape") and len(obs.shape) > 0 and obs.shape[0] == args.num_envs) else obs}
            # Build frame for policy (Isaac obs -> LeRobot-style frame)
            frame = isaac_obs_to_policy_frame(single_obs, language_instruction=args.instruction)
            # Preprocess -> select_action -> postprocess
            batch = preprocess(frame)
            with torch.inference_mode():
                action = policy.select_action(batch)
            action = postprocess(action)
            # Convert to env action (numpy, shape/clip)
            if hasattr(action, "cpu"):
                action = action.cpu().numpy()
            action = np.asarray(action)
            env_action = policy_action_to_env(action, env_action_space_shape=action_shape, clip=True)
            if args.num_envs > 1 and len(env_action.shape) == 1:
                env_action = np.tile(env_action[np.newaxis, :], (args.num_envs, 1))
            obs, reward, terminated, truncated, info = env.step(env_action)
            ee_state = info.get("ee_state", {})
            if step % 50 == 0:
                r = reward[0] if hasattr(reward, "shape") and reward.size > 1 else reward
                print(f"  step {step} reward={r} ee_pos={ee_state.get('ee_pos')}")
            step += 1
            term = terminated if not hasattr(terminated, "any") else terminated.any()
            trunc = truncated if not hasattr(truncated, "any") else truncated.any()
            if term or trunc:
                break
        print(f"Episode {ep + 1} finished after {step} steps.")
    env.close()
    print("Done.")


if __name__ == "__main__":
    main()
