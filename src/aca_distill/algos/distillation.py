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
        teacher_action: torch.Tensor | None,
        dataset_action: torch.Tensor,
        *,
        behavior_cloning_coef: float | None = None,
    ) -> dict[str, float]:
        prediction = self.actor(obs)
        if teacher_action is None:
            distill_loss = torch.zeros((), device=obs.device)
        else:
            distill_loss = F.mse_loss(prediction, teacher_action)
        bc_loss = F.mse_loss(prediction, dataset_action)
        bc_coef = self.cfg.behavior_cloning_coef if behavior_cloning_coef is None else behavior_cloning_coef
        total = self.cfg.distill_coef * distill_loss + bc_coef * bc_loss

        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        if self.cfg.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip_norm)
        self.optimizer.step()
        return {
            "student/loss": total.item(),
            "student/distill_loss": distill_loss.item(),
            "student/bc_loss": bc_loss.item(),
        }
