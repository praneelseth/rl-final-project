import torch

from aca_distill.algos.distillation import StudentDistillation
from aca_distill.config import StudentConfig
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


def test_student_update_supports_bc_only_warmstart():
    actor = StudentActor(obs_dim=6, action_dim=2)
    learner = StudentDistillation(actor=actor, cfg=StudentConfig())
    obs = torch.randn(4, 6)
    dataset_action = torch.randn(4, 2).clamp(-1.0, 1.0)
    metrics = learner.update(
        obs=obs,
        teacher_action=None,
        dataset_action=dataset_action,
        behavior_cloning_coef=1.0,
    )
    assert metrics["student/distill_loss"] == 0.0
