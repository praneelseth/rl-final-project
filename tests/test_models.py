import torch

from aca_distill.models.critic import NoiseLevelCritic
from aca_distill.models.student import StudentActor


def test_critic_outputs_scalar_per_sample():
    critic = NoiseLevelCritic(obs_dim=31, action_dim=8)
    obs = torch.randn(5, 31)
    action = torch.randn(5, 8)
    timestep = torch.randint(0, 21, (5,))
    q = critic(obs, action, timestep)
    assert q.shape == (5,)


def test_student_action_is_bounded():
    actor = StudentActor(obs_dim=31, action_dim=8)
    obs = torch.randn(5, 31)
    action = actor(obs)
    assert action.shape == (5, 8)
    assert torch.all(action <= 1.0)
    assert torch.all(action >= -1.0)
