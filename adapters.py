"""
Adapters: Isaac Lab obs/actions <-> SmolVLA policy I/O.
"""

from __future__ import annotations

import numpy as np
from typing import Any

CAMERA_KEYS = ("observation.images.camera1", "observation.images.camera2", "observation.images.camera3")
IMAGE_SHAPE = (3, 256, 256)
STATE_KEY = "observation.state"
LANGUAGE_KEY = "language_instruction"
TASK_KEY = "task"


def _to_numpy(x: Any) -> np.ndarray:
    return x.detach().cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def _resize_to_chw(img: np.ndarray, target_hw: tuple[int, int] = (256, 256)) -> np.ndarray:
    img = _to_numpy(img)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    if img.ndim == 3 and img.shape[-1] in (1, 3):
        img = np.transpose(img, (2, 0, 1))
    h, w = img.shape[1], img.shape[2]
    if (h, w) != target_hw:
        y = np.linspace(0, h - 1, target_hw[0]).astype(np.int32)
        x = np.linspace(0, w - 1, target_hw[1]).astype(np.int32)
        img = img[:, y, :][:, :, x]
    return img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img.astype(np.float32)


def _gather_images(obs: dict[str, Any]) -> list[np.ndarray]:
    keys = ("observation.images.camera1", "observation.images.camera2", "observation.images.camera3",
            "rgb", "image", "observation.images.top", "observation.images.wrist")
    images = []
    for k in keys:
        if k not in obs or len(images) >= 3:
            continue
        v = _to_numpy(obs[k])
        if (v.ndim == 2 or (v.ndim == 3 and min(v.shape) >= 2)) and v.size > 0:
            images.append(v)
    return images


def isaac_obs_to_policy_frame(
    obs: dict[str, Any],
    language_instruction: str = "Pick the cube.",
    image_key_map: dict[str, str] | None = None,
    state_keys: list[str] | None = None,
) -> dict[str, Any]:
    frame = {LANGUAGE_KEY: language_instruction, TASK_KEY: language_instruction}
    image_key_map = image_key_map or {}
    images = _gather_images(obs)
    if not images:
        images = [np.zeros((3, 256, 256), dtype=np.float32)]
    resized = []
    for i in range(3):
        img = _resize_to_chw(images[min(i, len(images) - 1)], (IMAGE_SHAPE[1], IMAGE_SHAPE[2]))
        if img.shape[0] == 1:
            img = np.repeat(img, 3, axis=0)
        resized.append(img.astype(np.float32))
    for key, img in zip(CAMERA_KEYS, resized):
        frame[key] = np.expand_dims(img, axis=0)
    for src, dst in image_key_map.items():
        if src in frame and dst != src:
            frame[dst] = frame.pop(src)
    default_state_keys = ("joint_pos", "joint_positions", "obs", "observation.state", "proprio",
                          "ee_pos", "ee_quat", "ee_pos_delta")
    keys = state_keys if state_keys else default_state_keys
    state_parts = [np.asarray(obs[k]).flatten() for k in keys if k in obs]
    state = np.concatenate(state_parts).astype(np.float32) if state_parts else np.zeros(0, dtype=np.float32)
    frame[STATE_KEY] = np.expand_dims(state, axis=0)
    return frame


def policy_action_to_env(
    action: np.ndarray,
    env_action_space_shape: tuple[int, ...] | None = None,
    clip: bool = True,
    scale: float | None = None,
) -> np.ndarray:
    action = np.asarray(action).flatten()
    if env_action_space_shape is not None:
        n = int(np.prod(env_action_space_shape))
        action = action[:n] if action.shape[0] >= n else np.pad(action, (0, n - action.shape[0]))
    if clip:
        action = np.clip(action, -1.0, 1.0)
    if scale is not None:
        action = action * scale
    return action.astype(np.float32)
