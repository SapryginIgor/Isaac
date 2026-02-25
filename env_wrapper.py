"""
Gymnasium wrapper that adds end-effector position and deltas to an Isaac Lab SO-101 env.
Exposes get_ee_state() and optionally adds ee_pos, ee_quat, ee_pos_delta to observations.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


def _get_articulation_ee_pose(env, robot_name: str | None, ee_link_name: str | None):
    """
    Get current end-effector pose (position, quat) from Isaac Lab env.
    Tries env.unwrapped.scene (ManagerBasedRLEnv) or similar.
    """
    unwrapped = env
    while hasattr(unwrapped, "unwrapped") and unwrapped.unwrapped is not unwrapped:
        unwrapped = unwrapped.unwrapped
    scene = getattr(unwrapped, "scene", None)
    if scene is None:
        return None, None
    # Try common patterns: scene.articulations or scene
    articulations = getattr(scene, "articulations", None) or getattr(scene, "arms", None)
    if articulations is None:
        return None, None
    # articulations can be dict-like or list; robot_name can be key or first articulation
    if robot_name and hasattr(articulations, "get"):
        art = articulations.get(robot_name)
    elif robot_name and hasattr(articulations, "__getitem__"):
        try:
            art = articulations[robot_name]
        except (KeyError, TypeError):
            art = None
    else:
        art = articulations[0] if (hasattr(articulations, "__len__") and len(articulations) > 0) else None
    if art is None:
        return None, None
    # Get body pose: Articulation has body_names / link state; get world pose of EE link
    data = getattr(art, "data", None)
    if data is None:
        return None, None
    # Isaac Lab articulation: body_pos_w, body_quat_w for each body
    body_pos = getattr(data, "body_pos_w", None) or getattr(data, "root_pos_w", None)
    body_quat = getattr(data, "body_quat_w", None) or getattr(data, "root_quat_w", None)
    if body_pos is None or body_quat is None:
        return None, None
    # If batched (num_envs, num_bodies, 3), EE is last body or resolved by name (e.g. gripper_link for SO-101)
    if hasattr(body_pos, "shape") and len(body_pos.shape) >= 2:
        names = list(getattr(art, "body_names", [])) or list(getattr(data, "body_names", []))
        if ee_link_name and names and ee_link_name in names:
            idx = names.index(ee_link_name)
        else:
            idx = -1
        pos = body_pos[..., idx, :] if body_pos.shape[-2] > 1 else body_pos[..., 0, :]
        quat = body_quat[..., idx, :] if body_quat.shape[-2] > 1 else body_quat[..., 0, :]
        # Squeeze to (3,) and (4,) for single env
        pos = np.asarray(pos).reshape(-1, 3)[0]
        quat = np.asarray(quat).reshape(-1, 4)[0]
    else:
        pos = np.asarray(body_pos).reshape(-1)[:3]
        quat = np.asarray(body_quat).reshape(-1)[:4]
    return pos.astype(np.float64), quat.astype(np.float64)


class IsaacEEWrapper(gym.Wrapper):
    """
    Wraps an Isaac Lab (SO-101) env and exposes end-effector position and deltas.
    """

    def __init__(
        self,
        env: gym.Env,
        robot_name: str | None = None,
        ee_link_name: str | None = None,
        add_ee_to_obs: bool = True,
    ):
        super().__init__(env)
        self._robot_name = robot_name
        self._ee_link_name = ee_link_name
        self._add_ee_to_obs = add_ee_to_obs
        self._prev_ee_pos: np.ndarray | None = None
        self._prev_ee_quat: np.ndarray | None = None
        self._ee_pos: np.ndarray | None = None
        self._ee_quat: np.ndarray | None = None
        self._ee_pos_delta: np.ndarray | None = None
        self._ee_quat_delta: np.ndarray | None = None

        if add_ee_to_obs and hasattr(env, "observation_space"):
            os = env.observation_space
            if isinstance(os, spaces.Dict):
                spaces_dict = dict(os.spaces)
                spaces_dict["ee_pos"] = spaces.Box(-np.inf, np.inf, (3,), dtype=np.float64)
                spaces_dict["ee_quat"] = spaces.Box(-1, 1, (4,), dtype=np.float64)
                spaces_dict["ee_pos_delta"] = spaces.Box(-np.inf, np.inf, (3,), dtype=np.float64)
                self.observation_space = spaces.Dict(spaces_dict)
            else:
                self.observation_space = os

    def get_ee_state(self):
        """
        Return current end-effector state: position, quaternion, and deltas.
        """
        return {
            "ee_pos": self._ee_pos.copy() if self._ee_pos is not None else np.zeros(3, dtype=np.float64),
            "ee_quat": self._ee_quat.copy() if self._ee_quat is not None else np.array([0, 0, 0, 1], dtype=np.float64),
            "ee_pos_delta": self._ee_pos_delta.copy() if self._ee_pos_delta is not None else np.zeros(3, dtype=np.float64),
            "ee_quat_delta": self._ee_quat_delta.copy() if self._ee_quat_delta is not None else np.zeros(4, dtype=np.float64),
        }

    def _update_ee(self):
        pos, quat = _get_articulation_ee_pose(self.env, self._robot_name, self._ee_link_name)
        if pos is not None and quat is not None:
            self._ee_pos = pos
            self._ee_quat = quat
            if self._prev_ee_pos is not None:
                self._ee_pos_delta = pos - self._prev_ee_pos
                self._ee_quat_delta = np.array(quat) - np.array(self._prev_ee_quat)
            else:
                self._ee_pos_delta = np.zeros(3, dtype=np.float64)
                self._ee_quat_delta = np.zeros(4, dtype=np.float64)
            self._prev_ee_pos = pos.copy()
            self._prev_ee_quat = quat.copy()
        else:
            self._ee_pos = np.zeros(3, dtype=np.float64)
            self._ee_quat = np.array([0, 0, 0, 1], dtype=np.float64)
            self._ee_pos_delta = np.zeros(3, dtype=np.float64)
            self._ee_quat_delta = np.zeros(4, dtype=np.float64)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._update_ee()
        if self._add_ee_to_obs and isinstance(obs, dict):
            obs = dict(obs)
            obs["ee_pos"] = self._ee_pos
            obs["ee_quat"] = self._ee_quat
            obs["ee_pos_delta"] = self._ee_pos_delta
        info["ee_state"] = self.get_ee_state()
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._prev_ee_pos = None
        self._prev_ee_quat = None
        self._update_ee()
        if self._add_ee_to_obs and isinstance(obs, dict):
            obs = dict(obs)
            obs["ee_pos"] = self._ee_pos
            obs["ee_quat"] = self._ee_quat
            obs["ee_pos_delta"] = self._ee_pos_delta
        info["ee_state"] = self.get_ee_state()
        return obs, info
