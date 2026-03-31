from __future__ import annotations

from typing import Any

import numpy as np

from aca_distill.config import RewardConfig
from aca_distill.data.antmaze import flatten_antmaze_observation, shaped_reward


class FlattenObservationWrapper:
    def __init__(self, env: Any, observation_mean: np.ndarray | None = None, observation_std: np.ndarray | None = None) -> None:
        self.env = env
        self.observation_mean = observation_mean
        self.observation_std = observation_std
        self.action_space = env.action_space
        self.observation_space = getattr(env, "observation_space", None)

    def _transform(self, obs: Any) -> np.ndarray:
        flat = flatten_antmaze_observation(obs)
        if self.observation_mean is not None and self.observation_std is not None:
            flat = (flat - self.observation_mean) / self.observation_std
        return flat.astype(np.float32)

    def reset(self, *args: Any, **kwargs: Any) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(*args, **kwargs)
        return self._transform(obs), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._transform(obs), reward, terminated, truncated, info

    def __getattr__(self, item: str) -> Any:
        return getattr(self.env, item)


class RewardShapingWrapper:
    def __init__(self, env: Any, cfg: RewardConfig) -> None:
        self.env = env
        self.cfg = cfg
        self._last_obs: Any | None = None
        self.action_space = env.action_space
        self.observation_space = getattr(env, "observation_space", None)

    def reset(self, *args: Any, **kwargs: Any) -> tuple[Any, dict]:
        obs, info = self.env.reset(*args, **kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped = shaped_reward(self._last_obs, obs, reward, self.cfg)
        self._last_obs = obs
        return obs, shaped, terminated, truncated, info

    def __getattr__(self, item: str) -> Any:
        return getattr(self.env, item)

