import torch

from aca_distill.algos.diffusion import DiffusionSchedule, add_noise


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

