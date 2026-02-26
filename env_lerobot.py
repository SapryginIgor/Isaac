"""
Thin adapter so the Isaac Lab env (one process, n_envs in parallel) presents
the gym.vector.VectorEnv interface. This prevents LeRobot from wrapping our env
in SyncVectorEnv([clone, clone, ...]), which would break (Isaac cannot be cloned).
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


def _unbatch_space(space: gym.Space, num_envs: int) -> gym.Space:
    """Derive single-env space from a possibly batched space (for VectorEnv API)."""
    if isinstance(space, spaces.Box):
        shape = space.shape
        if shape and len(shape) >= 1 and shape[0] == num_envs:
            low = np.asarray(space.low)
            high = np.asarray(space.high)
            if low.size > 1 and low.shape[0] == num_envs:
                low, high = low[0], high[0]
            elif low.size > 1 and low.shape != shape:
                low = np.reshape(low, shape)[0]
                high = np.reshape(high, shape)[0]
            return spaces.Box(low=low, high=high, shape=shape[1:], dtype=space.dtype)
        return space
    if isinstance(space, spaces.Dict):
        return spaces.Dict({
            k: _unbatch_space(v, num_envs) for k, v in space.spaces.items()
        })
    return space


class IsaacAsVectorEnv(gym.vector.VectorEnv):
    """
    Wraps a single Isaac Lab env (which already runs n_envs in parallel) so it
    looks like a gym.vector.VectorEnv. LeRobot then uses it as-is without cloning.
    """

    def __init__(self, env: gym.Env, num_envs: int):
        self._env = env
        self._num_envs = num_envs
        obs_space = env.observation_space
        act_space = env.action_space
        single_obs = _unbatch_space(obs_space, num_envs)
        single_act = _unbatch_space(act_space, num_envs)
        super().__init__(num_envs, single_obs, single_act)
        # Override batched spaces with the inner env's (Isaac may already expose batched)
        self._observation_space = obs_space
        self._action_space = act_space

    @property
    def observation_space(self) -> gym.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.Space:
        return self._action_space

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        return self._env.reset(seed=seed, options=options)

    def step(self, action):
        return self._env.step(action)

    def close(self, **kwargs):
        return self._env.close(**kwargs)

    @property
    def unwrapped(self) -> gym.Env:
        return self._env.unwrapped if hasattr(self._env, "unwrapped") else self._env
