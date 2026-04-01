from __future__ import annotations

import torch
import torch.nn.functional as F

from aca_distill.config import PriorConfig
from aca_distill.models.student import StudentActor


class BehaviorCloningPrior:
    def __init__(self, actor: StudentActor, cfg: PriorConfig) -> None:
        self.actor = actor
        self.cfg = cfg
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.learning_rate)

    def update(self, obs: torch.Tensor, action: torch.Tensor) -> dict[str, float]:
        prediction = self.actor(obs)
        loss = F.mse_loss(prediction, action)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip_norm)
        self.optimizer.step()
        return {
            "prior/loss": loss.item(),
        }
