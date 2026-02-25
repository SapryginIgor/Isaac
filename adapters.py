"""
Adapters between Isaac Lab env observations/actions and SmolVLA policy I/O.
Maps Isaac obs (images + robot state with EE) to the format expected by LeRobot preprocessor,
and policy output to env action (with optional scale/clip for SO-101).
"""

from __future__ import annotations

import numpy as np
from typing import Any


# SmolVLA policy expects these image keys; shape (B, C, H, W) with B=1 for single-step
CAMERA_KEYS = ("observation.images.camera1", "observation.images.camera2", "observation.images.camera3")
IMAGE_SHAPE = (3, 256, 256)  # CHW per image
BATCH_IMAGE_SHAPE = (1, 3, 256, 256)  # BCHW for policy
STATE_KEY = "observation.state"
LANGUAGE_KEY = "language_instruction"
TASK_KEY = "task"  # tokenizer_processor expects this in complementary_data


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "cpu"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _resize_to_chw(img: np.ndarray, target_hw: tuple[int, int] = (256, 256)) -> np.ndarray:
    """Resize image to target (H, W) and return as (C, H, W) float in [0, 1] or uint8."""
    img = _to_numpy(img)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    # (H, W, C) -> (C, H, W)
    if img.ndim == 3 and img.shape[-1] in (1, 3):
        img = np.transpose(img, (2, 0, 1))
    c, h, w = img.shape[0], img.shape[1], img.shape[2]
    if (h, w) == target_hw:
        return img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img.astype(np.float32)
    # simple numpy resize: index with linear sampling
    y = np.linspace(0, h - 1, target_hw[0]).astype(np.int32)
    x = np.linspace(0, w - 1, target_hw[1]).astype(np.int32)
    out = img[:, y, :][:, :, x]  # (C, target_hw[0], target_hw[1])
    return out.astype(np.float32) / 255.0 if out.dtype == np.uint8 else out.astype(np.float32)


def _gather_images(obs: dict[str, Any]) -> list[np.ndarray]:
    """Collect up to 3 images from obs (keys that look like image data: 2D or 3D with H,W > 1)."""
    images = []
    for k in (
        "observation.images.camera1", "observation.images.camera2", "observation.images.camera3",
        "rgb", "image", "cameras.rgb", "observation.images.top", "observation.images.wrist",
    ):
        if k not in obs:
            continue
        v = obs[k]
        v = _to_numpy(v)
        if v.ndim == 2:
            pass
        elif v.ndim == 3 and (v.shape[-1] in (1, 3) or v.shape[0] in (1, 3)):
            if min(v.shape) < 2:
                continue
        else:
            continue
        if v.size == 0:
            continue
        images.append(v)
        if len(images) >= 3:
            break
    return images


def isaac_obs_to_policy_frame(
    obs: dict[str, Any],
    language_instruction: str = "Pick the cube.",
    image_key_map: dict[str, str] | None = None,
    state_keys: list[str] | None = None,
) -> dict[str, Any]:
    """
    Build a frame-like dict for SmolVLA preprocessor from Isaac Lab env observation.
    Policy expects observation.images.camera1, camera2, camera3 with shape (3, 256, 256).
    """
    frame = {
        LANGUAGE_KEY: language_instruction,
        TASK_KEY: language_instruction,
    }
    image_key_map = image_key_map or {}
    images = _gather_images(obs)
    if not images:
        images = [np.zeros((3, 256, 256), dtype=np.float32)]
    # Resize to (3, 256, 256); duplicate first image if we have fewer than 3 views
    resized = []
    for i in range(3):
        img = _resize_to_chw(images[min(i, len(images) - 1)], (IMAGE_SHAPE[1], IMAGE_SHAPE[2]))
        if img.shape[0] == 1:
            img = np.repeat(img, 3, axis=0)
        resized.append(img.astype(np.float32))
    # Policy expects (B, C, H, W); add batch dim so each is (1, 3, 256, 256)
    for key, img in zip(CAMERA_KEYS, resized):
        frame[key] = np.expand_dims(img, axis=0)
    for src, dst in image_key_map.items():
        if src in frame and dst != src:
            frame[dst] = frame.pop(src)
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
        state = np.concatenate(state_parts).astype(np.float32)
    else:
        state = np.zeros(0, dtype=np.float32)
    # Policy expects (batch, state_dim); add batch dim
    frame[STATE_KEY] = np.expand_dims(state, axis=0)
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
