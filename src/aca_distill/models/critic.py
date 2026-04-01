from __future__ import annotations

import copy

import torch
from torch import nn

from aca_distill.models.common import MLP, SinusoidalTimeEmbedding


class NoiseLevelCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        hidden_layers: int = 3,
        time_embedding_dim: int = 32,
        activation: str = "mish",
    ) -> None:
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_embedding_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_embedding_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )
        self.q_net = MLP(
            input_dim=obs_dim + action_dim + hidden_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation=activation,
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        time_features = self.time_proj(self.time_embed(timestep))
        inputs = torch.cat([obs, action, time_features], dim=-1)
        return self.q_net(inputs).squeeze(-1)

    def make_target(self) -> "NoiseLevelCritic":
        target = copy.deepcopy(self)
        target.requires_grad_(False)
        return target


class DoubleNoiseLevelCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        hidden_layers: int = 3,
        time_embedding_dim: int = 32,
        activation: str = "mish",
    ) -> None:
        super().__init__()
        self.q1 = NoiseLevelCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            time_embedding_dim=time_embedding_dim,
            activation=activation,
        )
        self.q2 = NoiseLevelCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            time_embedding_dim=time_embedding_dim,
            activation=activation,
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor, timestep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(obs, action, timestep), self.q2(obs, action, timestep)

    def min_q(self, obs: torch.Tensor, action: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(obs, action, timestep)
        return torch.minimum(q1, q2)

    def make_target(self) -> "DoubleNoiseLevelCritic":
        target = copy.deepcopy(self)
        target.requires_grad_(False)
        return target
