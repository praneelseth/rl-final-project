from __future__ import annotations

import torch
import torch.nn.functional as F

from aca_distill.config import StudentConfig
from aca_distill.models.student import StudentActor


class StudentDistillation:
    def __init__(self, actor: StudentActor, cfg: StudentConfig) -> None:
        self.actor = actor
        self.cfg = cfg
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.learning_rate)

    def update(
        self,
        obs: torch.Tensor,
        teacher_action: torch.Tensor,
        dataset_action: torch.Tensor,
    ) -> dict[str, float]:
        prediction = self.actor(obs)
        distill_loss = F.mse_loss(prediction, teacher_action)
        bc_loss = F.mse_loss(prediction, dataset_action)
        total = self.cfg.distill_coef * distill_loss + self.cfg.behavior_cloning_coef * bc_loss

        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        self.optimizer.step()
        return {
            "student/loss": total.item(),
            "student/distill_loss": distill_loss.item(),
            "student/bc_loss": bc_loss.item(),
        }

