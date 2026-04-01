from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from aca_distill.algos.diffusion import DiffusionSchedule, add_noise
from aca_distill.config import DiffusionConfig, TeacherConfig
from aca_distill.models.critic import NoiseLevelCritic


@dataclass
class TeacherLosses:
    total: torch.Tensor
    td: torch.Tensor
    consistency: torch.Tensor
    conservative: torch.Tensor
    action_l2: torch.Tensor


class ACATeacher:
    def __init__(
        self,
        critic: NoiseLevelCritic,
        schedule: DiffusionSchedule,
        cfg: TeacherConfig,
        diffusion_cfg: DiffusionConfig,
        action_dim: int,
    ) -> None:
        self.critic = critic
        self.target_critic = critic.make_target()
        self.schedule = schedule
        self.cfg = cfg
        self.diffusion_cfg = diffusion_cfg
        self.action_dim = action_dim
        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.learning_rate)

    def _normalized_gradient(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        timestep: torch.Tensor,
        critic: NoiseLevelCritic,
    ) -> torch.Tensor:
        action = action.detach().requires_grad_(True)
        q_value = critic(obs, action, timestep)
        gradient = torch.autograd.grad(q_value.sum(), action, retain_graph=False)[0]
        norm = gradient.norm(dim=-1, keepdim=True).clamp_min(self.diffusion_cfg.gradient_epsilon)
        return gradient / norm

    @torch.no_grad()
    def _select_best(self, obs: torch.Tensor, candidates: torch.Tensor, critic: NoiseLevelCritic) -> torch.Tensor:
        batch_size, num_candidates, action_dim = candidates.shape
        tiled_obs = obs[:, None, :].expand(batch_size, num_candidates, obs.shape[-1]).reshape(-1, obs.shape[-1])
        flat_actions = candidates.reshape(-1, action_dim)
        timesteps = torch.zeros(flat_actions.shape[0], dtype=torch.long, device=obs.device)
        values = critic(tiled_obs, flat_actions, timesteps).view(batch_size, num_candidates)
        best_idx = values.argmax(dim=1)
        return candidates[torch.arange(batch_size, device=obs.device), best_idx]

    def sample_actions(
        self,
        obs: torch.Tensor,
        *,
        num_candidates: int | None = None,
        critic: NoiseLevelCritic | None = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        critic = critic or self.critic
        num_candidates = num_candidates or self.diffusion_cfg.batch_action_samples
        batch_size = obs.shape[0]
        tiled_obs = obs[:, None, :].expand(batch_size, num_candidates, obs.shape[-1]).reshape(-1, obs.shape[-1])
        action = torch.randn(batch_size * num_candidates, self.action_dim, device=obs.device)

        for step in range(self.schedule.steps, 0, -1):
            timestep = torch.full(
                (batch_size * num_candidates,),
                fill_value=step,
                dtype=torch.long,
                device=obs.device,
            )
            with torch.enable_grad():
                gradient = self._normalized_gradient(tiled_obs, action, timestep, critic)

            alpha = self.schedule.gather(self.schedule.alphas, timestep, action)
            alpha_bar = self.schedule.gather(self.schedule.alpha_bars, timestep, action)
            beta = self.schedule.gather(self.schedule.betas, timestep, action)
            sigma = self.schedule.gather(self.schedule.sigmas, timestep, action)
            if deterministic or step == 1:
                noise = torch.zeros_like(action)
            else:
                noise = torch.randn_like(action)

            action = (
                action
                + beta / torch.sqrt((1.0 - alpha_bar).clamp_min(1e-6))
                * (self.diffusion_cfg.guidance_scale * sigma)
                * gradient
            ) / torch.sqrt(alpha)
            action = (action + sigma * noise).clamp(-1.0, 1.0).detach()

        candidates = action.view(batch_size, num_candidates, self.action_dim)
        return self._select_best(obs, candidates, critic)

    def loss(
        self,
        batch: dict[str, torch.Tensor],
        student_action: torch.Tensor | None = None,
        *,
        use_teacher_targets: bool = True,
    ) -> TeacherLosses:
        obs = batch["obs"]
        action = batch["action"]
        next_action_from_dataset = batch["next_action"]
        reward = batch["reward"]
        next_obs = batch["next_obs"]
        done = batch["done"]

        zero_timestep = torch.zeros(obs.shape[0], dtype=torch.long, device=obs.device)
        q_data = self.critic(obs, action, zero_timestep)

        with torch.no_grad():
            if use_teacher_targets:
                next_action = self.sample_actions(
                    next_obs,
                    critic=self.target_critic,
                    deterministic=self.cfg.deterministic_target_sampling,
                )
            else:
                next_action = next_action_from_dataset
            next_q = self.target_critic(next_obs, next_action, zero_timestep)
            td_target = reward + self.cfg.discount * (1.0 - done) * next_q

        td_loss = F.mse_loss(q_data, td_target)

        timestep = torch.randint(
            low=1,
            high=self.schedule.steps + 1,
            size=(obs.shape[0],),
            device=obs.device,
        )
        noisy_action, _ = add_noise(action, timestep, self.schedule)
        noisy_q = self.critic(obs, noisy_action, timestep)
        consistency_loss = F.mse_loss(noisy_q, q_data.detach())

        random_actions = torch.empty(
            obs.shape[0],
            self.cfg.conservative_actions,
            self.action_dim,
            device=obs.device,
        ).uniform_(-1.0, 1.0)
        tiled_obs = obs[:, None, :].expand(-1, self.cfg.conservative_actions, -1)
        random_q = self.critic(
            tiled_obs.reshape(-1, obs.shape[-1]),
            random_actions.reshape(-1, self.action_dim),
            torch.zeros(obs.shape[0] * self.cfg.conservative_actions, dtype=torch.long, device=obs.device),
        ).view(obs.shape[0], self.cfg.conservative_actions)

        sampled_action = self.sample_actions(obs, num_candidates=1, deterministic=True).detach()
        sampled_q = self.critic(obs, sampled_action, zero_timestep).unsqueeze(1)
        candidate_q = torch.cat([random_q, sampled_q], dim=1)
        if student_action is not None:
            student_q = self.critic(obs, student_action.detach(), zero_timestep).unsqueeze(1)
            candidate_q = torch.cat([candidate_q, student_q], dim=1)
        conservative_loss = (torch.logsumexp(candidate_q, dim=1) - q_data).mean()

        action_l2 = action.pow(2).mean()
        total = (
            td_loss
            + self.cfg.consistency_coef * consistency_loss
            + self.cfg.conservative_coef * conservative_loss
            + self.cfg.action_l2_coef * action_l2
        )
        return TeacherLosses(
            total=total,
            td=td_loss,
            consistency=consistency_loss,
            conservative=conservative_loss,
            action_l2=action_l2,
        )

    def update(
        self,
        batch: dict[str, torch.Tensor],
        student_action: torch.Tensor | None = None,
        *,
        use_teacher_targets: bool = True,
    ) -> dict[str, float]:
        self.optimizer.zero_grad(set_to_none=True)
        losses = self.loss(batch, student_action=student_action, use_teacher_targets=use_teacher_targets)
        losses.total.backward()
        if self.cfg.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.grad_clip_norm)
        self.optimizer.step()

        with torch.no_grad():
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.mul_(1.0 - self.cfg.tau).add_(param, alpha=self.cfg.tau)

        return {
            "teacher/loss": losses.total.item(),
            "teacher/td_loss": losses.td.item(),
            "teacher/consistency_loss": losses.consistency.item(),
            "teacher/conservative_loss": losses.conservative.item(),
            "teacher/action_l2": losses.action_l2.item(),
        }
