from __future__ import annotations

import torch
from torch import nn

from aca_distill.models.common import MLP


class StudentActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        hidden_layers: int = 3,
        activation: str = "mish",
    ) -> None:
        super().__init__()
        self.backbone = MLP(
            input_dim=obs_dim,
            output_dim=action_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation=activation,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.backbone(obs))

