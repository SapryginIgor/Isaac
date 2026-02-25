#!/usr/bin/env python3
"""
Run SmolVLA policy in an Isaac Lab SO-101 env (lift-cube or reach).
Requires: Isaac Sim + Isaac Lab installed, and a community SO-101 env (e.g. isaac_so_arm101) registered.
Run from Isaac Lab: ./isaaclab.sh -p /path/to/run_smolvla_isaac.py --task <task_id> [options]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root so env_wrapper and adapters can be imported when run via isaaclab.sh -p script.py
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import gymnasium as gym
import numpy as np

from adapters import isaac_obs_to_policy_frame, policy_action_to_env
from env_wrapper import IsaacEEWrapper


def parse_args():
    p = argparse.ArgumentParser(description="SmolVLA inference on Isaac Lab SO-101 env")
    p.add_argument("--task", type=str, default="SO-ARM100-Reach-v0", help="Gym task id (e.g. SO-ARM100-Reach-v0 or SO-101-Lift-Cube-v0)")
    p.add_argument("--headless", action="store_true", help="Run without GUI")
    p.add_argument("--num_envs", type=int, default=1, help="Number of parallel envs")
    p.add_argument("--policy", type=str, default="lerobot/smolvla_base", help="HuggingFace policy repo or path")
    p.add_argument("--instruction", type=str, default="Pick the cube.", help="Language instruction for the policy")
    p.add_argument("--max_steps", type=int, default=500, help="Max steps per episode")
    p.add_argument("--episodes", type=int, default=3, help="Number of episodes to run")
    p.add_argument("--robot_name", type=str, default=None, help="Name of robot articulation in scene (for EE wrapper)")
    p.add_argument("--ee_link_name", type=str, default=None, help="End-effector link name (for EE wrapper)")
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
    try:
        env = gym.make(args.task, num_envs=args.num_envs, headless=args.headless)
    except Exception as e:
        print(
            "Failed to create env. Ensure you run this script with Isaac Lab: ./isaaclab.sh -p run_smolvla_isaac.py ...",
            file=sys.stderr,
        )
        print("Also ensure a SO-101 task is installed (e.g. isaac_so_arm101) and the task id is correct.", file=sys.stderr)
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
