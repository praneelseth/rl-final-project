from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

from aca_distill.data.antmaze import flatten_antmaze_observation
from aca_distill.envs.wrappers import FlattenObservationWrapper


def recover_antmaze_env(dataset_id: str, observation_mean: np.ndarray | None = None, observation_std: np.ndarray | None = None) -> Any:
    try:
        import minari
    except ImportError as exc:
        raise ImportError("Minari is required for evaluation. Install with `pip install -e \".[rl]\"`.") from exc

    dataset = minari.load_dataset(dataset_id)
    env = dataset.recover_environment(eval_env=True)
    return FlattenObservationWrapper(env, observation_mean=observation_mean, observation_std=observation_std)


def evaluate_policy(
    env: Any,
    action_fn: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
    episodes: int = 5,
    max_steps: int = 1000,
) -> dict[str, float]:
    returns: list[float] = []
    successes: list[float] = []

    for _ in range(episodes):
        obs, info = env.reset()
        total_reward = 0.0
        done = False
        steps = 0
        while not done and steps < max_steps:
            obs_tensor = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(device)
            action = action_fn(obs_tensor).squeeze(0).detach().cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            total_reward += float(reward)
            steps += 1

        returns.append(total_reward)
        successes.append(float(info.get("success", total_reward > 0.0)))

    return {
        "eval/return_mean": float(np.mean(returns)),
        "eval/success_rate": float(np.mean(successes)),
    }


def collect_rollout_artifacts(
    env: Any,
    action_fn: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
    episodes: int = 3,
) -> dict[str, np.ndarray]:
    positions: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for _ in range(episodes):
        raw_obs, info = env.unwrapped.reset()
        obs = flatten_antmaze_observation(raw_obs)
        done = False
        while not done:
            action = action_fn(torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(device))
            action_np = action.squeeze(0).detach().cpu().numpy()
            raw_obs, reward, terminated, truncated, info = env.unwrapped.step(action_np)
            obs = flatten_antmaze_observation(raw_obs)
            positions.append(np.asarray(raw_obs["achieved_goal"], dtype=np.float32))
            actions.append(action_np)
            done = bool(terminated or truncated)
    return {
        "positions": np.asarray(positions, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
    }
