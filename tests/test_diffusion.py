import torch

from aca_distill.algos.aca_teacher import ACATeacher
from aca_distill.algos.diffusion import DiffusionSchedule, add_noise
from aca_distill.config import DiffusionConfig, TeacherConfig
from aca_distill.models.critic import NoiseLevelCritic


def test_diffusion_schedule_shapes():
    schedule = DiffusionSchedule(steps=20, beta_start=1e-4, beta_end=2e-2, device=torch.device("cpu"))
    assert schedule.betas.shape[0] == 21
    assert schedule.alpha_bars.shape[0] == 21
    assert torch.isclose(schedule.alpha_bars[0], torch.tensor(1.0))


def test_add_noise_preserves_shape():
    schedule = DiffusionSchedule(steps=10, beta_start=1e-4, beta_end=2e-2, device=torch.device("cpu"))
    action = torch.zeros(4, 8)
    timestep = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    noisy, noise = add_noise(action, timestep, schedule)
    assert noisy.shape == action.shape
    assert noise.shape == action.shape


def test_teacher_update_supports_dataset_bootstrap_targets():
    device = torch.device("cpu")
    critic = NoiseLevelCritic(obs_dim=5, action_dim=2)
    schedule = DiffusionSchedule(steps=5, beta_start=1e-4, beta_end=2e-2, device=device)
    teacher = ACATeacher(
        critic=critic,
        schedule=schedule,
        cfg=TeacherConfig(conservative_actions=2, dataset_target_warmstart_steps=10),
        diffusion_cfg=DiffusionConfig(steps=5, batch_action_samples=2),
        action_dim=2,
    )
    batch = {
        "obs": torch.randn(4, 5),
        "action": torch.randn(4, 2).clamp(-1.0, 1.0),
        "next_action": torch.randn(4, 2).clamp(-1.0, 1.0),
        "reward": torch.randn(4),
        "next_obs": torch.randn(4, 5),
        "done": torch.zeros(4),
        "success": torch.zeros(4),
    }
    metrics = teacher.update(batch, student_action=None, use_teacher_targets=False)
    assert "teacher/loss" in metrics
