"""
LeRobot / EnvHub compatibility: expose make_env() so LeRobot training and data
pipelines can use this Isaac Lab SO-101 environment.

Use this env from within Isaac Lab's Python (run your script with isaaclab.sh).
The simulation app (AppLauncher) must be started by the caller before calling make_env().
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Ensure project root and isaac_so_arm101 are on path when loaded as EnvHub or from LeRobot
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
_EXTENSION_SRC = _SCRIPT_DIR / "isaac_so_arm101" / "src"
if _EXTENSION_SRC.exists() and str(_EXTENSION_SRC) not in sys.path:
    sys.path.insert(0, str(_EXTENSION_SRC))


def make_env(
    n_envs: int = 1,
    use_async_envs: bool = False,
    cfg: Any = None,
    *,
    task: str = "Isaac-SO-ARM101-Lift-Cube-v0",
    device: str | None = None,
    robot_name: str = "robot",
    ee_link_name: str = "gripper_link",
    add_ee_to_obs: bool = True,
):
    """
    Build the Isaac Lab SO-101 env (Lift-Cube or Reach) wrapped for EE state and
    LeRobot-friendly use.

    Must be called from a process where the Isaac Sim app is already running
    (e.g. script launched via isaaclab.sh). LeRobot/EnvHub entry point.

    Args:
        n_envs: Number of parallel environments (Isaac runs them in one process).
        use_async_envs: Ignored; Isaac is inherently vectorized in-process.
        cfg: Optional EnvConfig-like object; if provided, can override task/device/robot_name/etc.
        task: Gym task id (e.g. Isaac-SO-ARM101-Lift-Cube-v0, Isaac-SO-ARM101-Reach-v0).
        device: Device for simulation (e.g. "cuda:0"). Default from isaaclab.
        robot_name: Robot articulation name in the scene.
        ee_link_name: End-effector link name for EE pose (e.g. gripper_link).
        add_ee_to_obs: Whether to add ee_pos, ee_quat, ee_pos_delta to observations.

    Returns:
        A VectorEnv-compatible wrapper around the Isaac env so LeRobot does not
        try to clone it into multiple processes. The underlying env already
        runs n_envs in parallel.
    """
    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg

    import isaac_so_arm101.tasks.reach  # noqa: F401
    import isaac_so_arm101.tasks.lift   # noqa: F401

    from env_wrapper import IsaacEEWrapper
    from env_lerobot import IsaacAsVectorEnv

    if cfg is not None:
        task = cfg.task
        device = cfg.device
        robot_name = cfg.robot_name
        ee_link_name = cfg.ee_link_name
        add_ee_to_obs = cfg.add_ee_to_obs

    reg = gym.envs.registry
    if task not in reg and not task.endswith("-v0"):
        alt = f"{task.rstrip('-v0')}-v0"
        if alt in reg:
            task = alt

    env_cfg = parse_env_cfg(task, device=device, num_envs=n_envs)
    env = gym.make(task, cfg=env_cfg)
    env = IsaacEEWrapper(
        env,
        robot_name=robot_name,
        ee_link_name=ee_link_name,
        add_ee_to_obs=add_ee_to_obs,
    )
    return IsaacAsVectorEnv(env, num_envs=n_envs)
