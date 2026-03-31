from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from aca_distill.config import DatasetConfig, RewardConfig


def flatten_antmaze_observation(obs: Any) -> np.ndarray:
    if isinstance(obs, dict):
        parts = [
            np.asarray(obs["observation"], dtype=np.float32).reshape(-1),
            np.asarray(obs["achieved_goal"], dtype=np.float32).reshape(-1),
            np.asarray(obs["desired_goal"], dtype=np.float32).reshape(-1),
        ]
        return np.concatenate(parts, axis=0)
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def _extract_goal(obs: Any, key: str) -> np.ndarray | None:
    if isinstance(obs, dict) and key in obs:
        return np.asarray(obs[key], dtype=np.float32)
    return None


def antmaze_success(obs: Any, next_obs: Any, env_reward: float) -> float:
    achieved = _extract_goal(next_obs, "achieved_goal")
    desired = _extract_goal(next_obs, "desired_goal")
    if achieved is not None and desired is not None:
        return float(np.linalg.norm(achieved - desired) <= 0.5)
    return float(env_reward > 0.0)


def shaped_reward(obs: Any, next_obs: Any, env_reward: float, cfg: RewardConfig) -> float:
    if cfg.mode == "raw":
        return float(env_reward)

    success = antmaze_success(obs, next_obs, env_reward)
    achieved = _extract_goal(obs, "achieved_goal")
    desired = _extract_goal(obs, "desired_goal")
    next_achieved = _extract_goal(next_obs, "achieved_goal")
    next_desired = _extract_goal(next_obs, "desired_goal")

    progress_term = 0.0
    if achieved is not None and desired is not None and next_achieved is not None and next_desired is not None:
        current_distance = np.linalg.norm(achieved - desired)
        next_distance = np.linalg.norm(next_achieved - next_desired)
        progress_term = cfg.distance_scale * float(current_distance - next_distance)

    reward = float(env_reward) + progress_term + cfg.success_bonus * success - cfg.step_penalty
    if cfg.clip_value is not None:
        reward = float(np.clip(reward, -cfg.clip_value, cfg.clip_value))
    return reward


def index_observation(obs_seq: Any, index: int) -> Any:
    if isinstance(obs_seq, dict):
        return {key: value[index] for key, value in obs_seq.items()}
    return obs_seq[index]


@dataclass
class OfflineReplayBuffer:
    obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_obs: torch.Tensor
    done: torch.Tensor
    success: torch.Tensor
    observation_mean: torch.Tensor | None
    observation_std: torch.Tensor | None

    @property
    def size(self) -> int:
        return self.obs.shape[0]

    @property
    def obs_dim(self) -> int:
        return self.obs.shape[1]

    @property
    def action_dim(self) -> int:
        return self.action.shape[1]

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        indices = torch.randint(0, self.size, (batch_size,))
        return {
            "obs": self.obs[indices].to(device),
            "action": self.action[indices].to(device),
            "reward": self.reward[indices].to(device),
            "next_obs": self.next_obs[indices].to(device),
            "done": self.done[indices].to(device),
            "success": self.success[indices].to(device),
        }


def load_antmaze_dataset(cfg: DatasetConfig, reward_cfg: RewardConfig) -> OfflineReplayBuffer:
    try:
        import minari
    except ImportError as exc:
        raise ImportError("Minari is required to load AntMaze datasets. Install with `pip install -e \".[rl]\"`.") from exc

    try:
        dataset = minari.load_dataset(cfg.dataset_id, download=cfg.download)
    except TypeError:
        dataset = minari.load_dataset(cfg.dataset_id)

    obs_list: list[np.ndarray] = []
    action_list: list[np.ndarray] = []
    reward_list: list[float] = []
    next_obs_list: list[np.ndarray] = []
    done_list: list[float] = []
    success_list: list[float] = []

    for episode_idx, episode in enumerate(dataset.iterate_episodes()):
        if cfg.max_episodes is not None and episode_idx >= cfg.max_episodes:
            break
        horizon = len(episode.actions)
        for step in range(horizon):
            obs = index_observation(episode.observations, step)
            next_obs = index_observation(episode.observations, step + 1)
            action = np.asarray(episode.actions[step], dtype=np.float32)
            env_reward = float(episode.rewards[step])
            reward = shaped_reward(obs, next_obs, env_reward, reward_cfg) if cfg.reward_mode == "shaped" else env_reward
            done = float(bool(episode.terminations[step]) or bool(episode.truncations[step]))
            success = antmaze_success(obs, next_obs, env_reward)

            obs_list.append(flatten_antmaze_observation(obs))
            action_list.append(action)
            reward_list.append(reward)
            next_obs_list.append(flatten_antmaze_observation(next_obs))
            done_list.append(done)
            success_list.append(success)

    obs_array = np.asarray(obs_list, dtype=np.float32)
    next_obs_array = np.asarray(next_obs_list, dtype=np.float32)
    action_array = np.asarray(action_list, dtype=np.float32)
    reward_array = np.asarray(reward_list, dtype=np.float32).reshape(-1, 1)
    done_array = np.asarray(done_list, dtype=np.float32).reshape(-1, 1)
    success_array = np.asarray(success_list, dtype=np.float32).reshape(-1, 1)

    mean_tensor: torch.Tensor | None = None
    std_tensor: torch.Tensor | None = None
    if cfg.normalize_observations:
        mean = obs_array.mean(axis=0, keepdims=True)
        std = obs_array.std(axis=0, keepdims=True) + 1e-6
        obs_array = (obs_array - mean) / std
        next_obs_array = (next_obs_array - mean) / std
        mean_tensor = torch.from_numpy(mean.squeeze(0))
        std_tensor = torch.from_numpy(std.squeeze(0))

    return OfflineReplayBuffer(
        obs=torch.from_numpy(obs_array),
        action=torch.from_numpy(action_array),
        reward=torch.from_numpy(reward_array),
        next_obs=torch.from_numpy(next_obs_array),
        done=torch.from_numpy(done_array),
        success=torch.from_numpy(success_array),
        observation_mean=mean_tensor,
        observation_std=std_tensor,
    )

