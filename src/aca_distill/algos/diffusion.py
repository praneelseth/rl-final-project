from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DiffusionSchedule:
    steps: int
    beta_start: float
    beta_end: float
    device: torch.device

    def __post_init__(self) -> None:
        betas = torch.linspace(self.beta_start, self.beta_end, self.steps, device=self.device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1, device=self.device), alpha_bars[:-1]], dim=0)
        sigmas = torch.sqrt(((1.0 - alpha_bars_prev) / (1.0 - alpha_bars)) * betas)

        self.betas = torch.cat([torch.zeros(1, device=self.device), betas], dim=0)
        self.alphas = torch.cat([torch.ones(1, device=self.device), alphas], dim=0)
        self.alpha_bars = torch.cat([torch.ones(1, device=self.device), alpha_bars], dim=0)
        self.sigmas = torch.cat([torch.zeros(1, device=self.device), sigmas], dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    def gather(self, values: torch.Tensor, timestep: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        gathered = values[timestep].view(-1, *([1] * (like.ndim - 1)))
        return gathered.expand_as(like)


def add_noise(
    action: torch.Tensor,
    timestep: torch.Tensor,
    schedule: DiffusionSchedule,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if noise is None:
        noise = torch.randn_like(action)
    sqrt_alpha_bar = schedule.gather(schedule.sqrt_alpha_bars, timestep, action)
    sqrt_one_minus = schedule.gather(schedule.sqrt_one_minus_alpha_bars, timestep, action)
    noisy = sqrt_alpha_bar * action + sqrt_one_minus * noise
    return noisy, noise

