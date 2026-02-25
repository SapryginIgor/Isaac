"""
Adapters between Isaac Lab env observations/actions and SmolVLA policy I/O.
Maps Isaac obs (images + robot state with EE) to the format expected by LeRobot preprocessor,
and policy output to env action (with optional scale/clip for SO-101).
"""

from __future__ import annotations

import numpy as np
from typing import Any


# Common keys used by LeRobot/SmolVLA for observations (dataset-dependent)
IMAGE_KEYS = ("observation.images.top", "observation.images.wrist", "observation.images.front", "observation.images.side")
STATE_KEY = "observation.state"
LANGUAGE_KEY = "language_instruction"


def isaac_obs_to_policy_frame(
    obs: dict[str, Any],
    language_instruction: str = "Pick the cube.",
    image_key_map: dict[str, str] | None = None,
    state_keys: list[str] | None = None,
) -> dict[str, Any]:
    """
    Build a frame-like dict for SmolVLA preprocessor from Isaac Lab env observation.

    Isaac Lab may return obs as a flat dict with keys like "joint_pos", "rgb_image", etc.,
    or from a ManagerBased env with concatenated terms. We produce keys that match
    typical LeRobot datasets: observation.images.<view>, observation.state, language_instruction.

    Args:
        obs: Raw observation dict from wrapped Isaac env (may include ee_pos, ee_quat, images).
        language_instruction: Task instruction string.
        image_key_map: Optional mapping from Isaac key -> LeRobot key, e.g. {"rgb": "observation.images.top"}.
        state_keys: Optional list of keys to concatenate into observation.state (default: joint + ee).
    """
    frame = {LANGUAGE_KEY: language_instruction}
    image_key_map = image_key_map or {}
    # Default: look for common Isaac Lab / SO-101 image keys
    for isaac_key in ("rgb", "image", "cameras.rgb", "observation.images.top", "observation.images.wrist"):
        if isaac_key in obs:
            key = image_key_map.get(isaac_key, "observation.images.top")
            frame[key] = obs[isaac_key]
            break
    # Second view if present
    for isaac_key in ("rgb_2", "image_2", "observation.images.wrist", "observation.images.side"):
        if isaac_key in obs and isaac_key != frame.get("observation.images.top"):
            key = image_key_map.get(isaac_key, "observation.images.wrist")
            frame[key] = obs[isaac_key]
            break
    # State: concatenate joint positions, EE pose, and optionally deltas (SmolVLA often uses proprio)
    state_parts = []
    if state_keys:
        for k in state_keys:
            if k in obs:
                v = np.asarray(obs[k]).flatten()
                state_parts.append(v)
    else:
        for k in ("joint_pos", "joint_positions", "obs", "observation.state", "proprio"):
            if k in obs:
                state_parts.append(np.asarray(obs[k]).flatten())
        for k in ("ee_pos", "ee_quat", "ee_pos_delta"):
            if k in obs:
                state_parts.append(np.asarray(obs[k]).flatten())
    if state_parts:
        frame[STATE_KEY] = np.concatenate(state_parts).astype(np.float32)
    else:
        frame[STATE_KEY] = np.zeros(0, dtype=np.float32)
    return frame


def policy_action_to_env(
    action: np.ndarray,
    env_action_space_shape: tuple[int, ...] | None = None,
    clip: bool = True,
    scale: float | None = None,
) -> np.ndarray:
    """
    Map SmolVLA policy output to env action. SO-101 typically matches policy (7D: 6 DOF + gripper).
    Optionally clip to [-1, 1] or env bounds and scale.
    """
    action = np.asarray(action).flatten()
    if env_action_space_shape is not None:
        target_len = int(np.prod(env_action_space_shape))
        if action.shape[0] > target_len:
            action = action[:target_len]
        elif action.shape[0] < target_len:
            action = np.pad(action, (0, target_len - action.shape[0]), mode="constant", constant_values=0)
    if clip:
        action = np.clip(action, -1.0, 1.0)
    if scale is not None:
        action = action * scale
    return action.astype(np.float32)
