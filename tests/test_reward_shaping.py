import numpy as np

from aca_distill.config import RewardConfig
from aca_distill.data.antmaze import antmaze_success, flatten_antmaze_observation, shaped_reward


def test_flatten_antmaze_observation_concatenates_goal_fields():
    obs = {
        "observation": np.array([1.0, 2.0, 3.0]),
        "achieved_goal": np.array([4.0, 5.0]),
        "desired_goal": np.array([6.0, 7.0]),
    }
    flat = flatten_antmaze_observation(obs)
    assert flat.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]


def test_progress_reward_increases_when_agent_moves_toward_goal():
    cfg = RewardConfig(mode="progress", distance_scale=1.0, success_bonus=1.0, step_penalty=0.0, clip_value=None)
    obs = {
        "observation": np.zeros(3),
        "achieved_goal": np.array([0.0, 0.0]),
        "desired_goal": np.array([2.0, 0.0]),
    }
    next_obs = {
        "observation": np.zeros(3),
        "achieved_goal": np.array([1.0, 0.0]),
        "desired_goal": np.array([2.0, 0.0]),
    }
    assert shaped_reward(obs, next_obs, 0.0, cfg) > 0.0
    assert antmaze_success(obs, next_obs, 0.0) == 0.0

