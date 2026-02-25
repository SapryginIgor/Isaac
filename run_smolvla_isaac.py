#!/usr/bin/env python3
"""
Run SmolVLA policy in an Isaac Lab SO-101 env (lift-cube or reach).
Uses the isaac_so_arm101 extension in-repo. Follows Isaac Lab pattern: launch app first, then import.
Run from Isaac Lab: ./isaaclab.sh -p /path/to/Isaac/run_smolvla_isaac.py --task Isaac-SO-ARM101-Lift-Cube-v0
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import re
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# Project root (directory containing this script and isaac_so_arm101/)
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
_EXTENSION_SRC = _SCRIPT_DIR / "isaac_so_arm101" / "src"
if _EXTENSION_SRC.exists() and str(_EXTENSION_SRC) not in sys.path:
    sys.path.insert(0, str(_EXTENSION_SRC))

# add argparse arguments
parser = argparse.ArgumentParser(description="SmolVLA inference on Isaac Lab SO-101 env")
parser.add_argument("--task", type=str, default="Isaac-SO-ARM101-Lift-Cube-v0", help="Gym task id")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel envs")
parser.add_argument("--policy", type=str, default="lerobot/smolvla_base", help="HuggingFace policy repo or path")
parser.add_argument("--instruction", type=str, default="Pick the cube.", help="Language instruction for the policy")
parser.add_argument("--max_steps", type=int, default=500, help="Max steps per episode")
parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run")
parser.add_argument("--robot_name", type=str, default="robot", help="Robot articulation name in scene")
parser.add_argument("--ee_link_name", type=str, default="gripper_link", help="End-effector link name (SO-101: gripper_link)")
parser.add_argument("--no_ee_in_obs", action="store_true", help="Do not add ee_pos/ee_quat/delta to obs dict")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app (must run before any isaaclab/pxr imports)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np

import isaac_so_arm101.tasks.reach  # noqa: F401
import isaac_so_arm101.tasks.lift   # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from adapters import isaac_obs_to_policy_frame, policy_action_to_env
from env_wrapper import IsaacEEWrapper


def main():
    # Load policy and processors (LeRobot)
    try:
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    except ImportError as e:
        print("LeRobot/SmolVLA not installed. Install with: pip install 'lerobot[smolvla]'", file=sys.stderr)
        raise SystemExit(1) from e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading policy from {args_cli.policy} on {device} ...")
    policy = SmolVLAPolicy.from_pretrained(args_cli.policy).to(device).eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        args_cli.policy,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    # Resolve task id (extension registers with -v0 suffix)
    _reg = getattr(getattr(gym, "envs", None), "registry", None) or {}
    task_id = args_cli.task
    if task_id not in _reg and not re.match(r".*-v\d+$", task_id):
        _with_v0 = f"{task_id.rstrip('-v0')}-v0"
        if _with_v0 in _reg:
            task_id = _with_v0

    env_cfg = parse_env_cfg(task_id, device=args_cli.device, num_envs=args_cli.num_envs)
    try:
        env = gym.make(task_id, cfg=env_cfg)
    except Exception as e:
        print("Failed to create env.", file=sys.stderr)
        print("Ensure isaac_so_arm101 is on PYTHONPATH (in-repo: isaac_so_arm101/src is added automatically).", file=sys.stderr)
        raise SystemExit(1) from e

    env = IsaacEEWrapper(
        env,
        robot_name=args_cli.robot_name,
        ee_link_name=args_cli.ee_link_name,
        add_ee_to_obs=not args_cli.no_ee_in_obs,
    )

    action_space = env.action_space
    action_shape = tuple(action_space.shape) if hasattr(action_space, "shape") else (getattr(action_space, "n", 7),)

    for ep in range(args_cli.episodes):
        obs, info = env.reset()
        ee_state = info.get("ee_state", {})
        print(f"Episode {ep + 1}/{args_cli.episodes} started. EE pos: {ee_state.get('ee_pos', 'N/A')}")
        step = 0
        while step < args_cli.max_steps:
            if isinstance(obs, dict):
                single_obs = {k: (v[0] if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == args_cli.num_envs else v) for k, v in obs.items()}
            else:
                single_obs = {"obs": obs[0] if (hasattr(obs, "shape") and len(obs.shape) > 0 and obs.shape[0] == args_cli.num_envs) else obs}
            frame = isaac_obs_to_policy_frame(single_obs, language_instruction=args_cli.instruction)
            batch = preprocess(frame)
            # SmolVLA expects all tensors on the same device; convert numpy and move existing tensors
            if isinstance(batch, dict):
                out = {}
                for k, v in batch.items():
                    if isinstance(v, np.ndarray):
                        out[k] = torch.as_tensor(v, device=device)
                    elif isinstance(v, torch.Tensor) and v.device != device:
                        out[k] = v.to(device)
                    else:
                        out[k] = v
                batch = out
            with torch.inference_mode():
                action = policy.select_action(batch)
            action = postprocess(action)
            action = action.cpu().numpy() if hasattr(action, "cpu") else np.asarray(action)
            env_action = policy_action_to_env(action, env_action_space_shape=action_shape, clip=True)
            # Action manager expects (num_envs, action_dim); ensure 2D
            if env_action.ndim == 1:
                env_action = np.tile(env_action[np.newaxis, :], (args_cli.num_envs, 1))
            # Isaac Lab env expects a tensor; get inner env device and convert
            inner_env = env
            while hasattr(inner_env, "env"):
                inner_env = inner_env.env
            env_device = getattr(inner_env, "device", device)
            env_action_t = torch.as_tensor(env_action, device=env_device, dtype=torch.float32)
            obs, reward, terminated, truncated, info = env.step(env_action_t)
            ee_state = info.get("ee_state", {})
            if step % 50 == 0:
                r = reward.item()
                print(f"  step {step} reward={r} ee_pos={ee_state.get('ee_pos')}")
            step += 1
            term = terminated.any() if hasattr(terminated, "any") else terminated
            trunc = truncated.any() if hasattr(truncated, "any") else truncated
            if term or trunc:
                break
        print(f"Episode {ep + 1} finished after {step} steps.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
