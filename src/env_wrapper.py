"""
Gymnasium wrapper: adds end-effector pose and deltas to Isaac Lab SO-101 env.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


def _get_articulation_ee_pose(env, robot_name: str | None, ee_link_name: str | None):
    unwrapped = env
    while hasattr(unwrapped, "unwrapped") and unwrapped.unwrapped is not unwrapped:
        unwrapped = unwrapped.unwrapped
    scene = unwrapped.scene
    if scene is None:
        return None, None
    articulations = scene.articulations
    if articulations is None:
        return None, None
    art = articulations.get(robot_name) if hasattr(articulations, "get") and robot_name else None
    if art is None and hasattr(articulations, "__len__") and len(articulations) > 0:
        art = articulations[0]
    if art is None:
        return None, None
    data = art.data
    if data is None:
        return None, None
    body_pos = data.body_pos_w
    body_quat = data.body_quat_w
    if body_pos is None or body_quat is None:
        return None, None
    names = list(art.body_names)
    idx = names.index(ee_link_name) if ee_link_name and ee_link_name in names else -1
    if body_pos.shape and len(body_pos.shape) >= 2 and body_pos.shape[-2] > 1:
        pos = body_pos[..., idx, :]
        quat = body_quat[..., idx, :]
    else:
        pos = body_pos.reshape(-1, 3)[0] if body_pos.size >= 3 else body_pos
        quat = body_quat.reshape(-1, 4)[0] if body_quat.size >= 4 else body_quat
    pos = np.asarray(pos.cpu() if hasattr(pos, "cpu") else pos).reshape(3).astype(np.float64)
    quat = np.asarray(quat.cpu() if hasattr(quat, "cpu") else quat).reshape(4).astype(np.float64)
    return pos, quat


class IsaacEEWrapper(gym.Wrapper):

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
