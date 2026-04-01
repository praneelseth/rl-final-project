from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from aca_distill.algos.diffusion import DiffusionSchedule, add_noise
from aca_distill.config import DiffusionConfig, TeacherConfig
from aca_distill.models.critic import DoubleNoiseLevelCritic
from aca_distill.models.student import StudentActor


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
        critic: DoubleNoiseLevelCritic,
        prior_actor: StudentActor,
        schedule: DiffusionSchedule,
        cfg: TeacherConfig,
        diffusion_cfg: DiffusionConfig,
        action_dim: int,
    ) -> None:
        self.critic = critic
        self.target_critic = critic.make_target()
        self.prior_actor = prior_actor
        self.schedule = schedule
        self.cfg = cfg
        self.diffusion_cfg = diffusion_cfg
        self.action_dim = action_dim
        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.learning_rate)

    def _min_q(
        self,
        critic: DoubleNoiseLevelCritic,
        obs: torch.Tensor,
        action: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        return critic.min_q(obs, action, timestep)

    def _normalized_gradient(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        timestep: torch.Tensor,
        critic: DoubleNoiseLevelCritic,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action = action.detach().requires_grad_(True)
        q_value = self._min_q(critic, obs, action, timestep)
        q_gradient = torch.autograd.grad(q_value.sum(), action, retain_graph=False)[0]
        q_norm = q_gradient.norm(dim=-1, keepdim=True).clamp_min(self.diffusion_cfg.gradient_epsilon)
        q_gradient = q_gradient / q_norm

        with torch.no_grad():
            prior_action = self.prior_actor(obs)
        prior_gradient = (prior_action - action.detach()) / max(self.cfg.prior_noise_scale**2, 1e-4)
        prior_norm = prior_gradient.norm(dim=-1, keepdim=True).clamp_min(self.diffusion_cfg.gradient_epsilon)
        prior_gradient = prior_gradient / prior_norm
        return q_gradient, prior_gradient

    @torch.no_grad()
    def _select_best(self, obs: torch.Tensor, candidates: torch.Tensor, critic: DoubleNoiseLevelCritic) -> torch.Tensor:
        batch_size, num_candidates, action_dim = candidates.shape
        tiled_obs = obs[:, None, :].expand(batch_size, num_candidates, obs.shape[-1]).reshape(-1, obs.shape[-1])
        flat_actions = candidates.reshape(-1, action_dim)
        timesteps = torch.zeros(flat_actions.shape[0], dtype=torch.long, device=obs.device)
        values = self._min_q(critic, tiled_obs, flat_actions, timesteps).view(batch_size, num_candidates)
        best_idx = values.argmax(dim=1)
        return candidates[torch.arange(batch_size, device=obs.device), best_idx]

    def sample_actions(
        self,
        obs: torch.Tensor,
        *,
        num_candidates: int | None = None,
        critic: DoubleNoiseLevelCritic | None = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        critic = critic or self.critic
        num_candidates = num_candidates or self.diffusion_cfg.batch_action_samples
        batch_size = obs.shape[0]
        tiled_obs = obs[:, None, :].expand(batch_size, num_candidates, obs.shape[-1]).reshape(-1, obs.shape[-1])

        with torch.no_grad():
            prior_action = self.prior_actor(tiled_obs)
        if deterministic:
            action = prior_action.clone()
        else:
            action = (prior_action + self.cfg.prior_noise_scale * torch.randn_like(prior_action)).clamp(-1.0, 1.0)

        for step in range(self.schedule.steps, 0, -1):
            timestep = torch.full(
                (batch_size * num_candidates,),
                fill_value=step,
                dtype=torch.long,
                device=obs.device,
            )
            with torch.enable_grad():
                q_gradient, prior_gradient = self._normalized_gradient(tiled_obs, action, timestep, critic)

            alpha = self.schedule.gather(self.schedule.alphas, timestep, action)
            alpha_bar = self.schedule.gather(self.schedule.alpha_bars, timestep, action)
            beta = self.schedule.gather(self.schedule.betas, timestep, action)
            sigma = self.schedule.gather(self.schedule.sigmas, timestep, action)
            if deterministic or step == 1:
                noise = torch.zeros_like(action)
            else:
                noise = torch.randn_like(action)

            combined_gradient = self.diffusion_cfg.guidance_scale * q_gradient + self.cfg.prior_guidance_coef * prior_gradient
            action = (
                action
                + beta / torch.sqrt((1.0 - alpha_bar).clamp_min(1e-6)) * sigma * combined_gradient
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
        reward = self.cfg.reward_scale * batch["reward"]
        next_obs = batch["next_obs"]
        done = batch["done"]

        zero_timestep = torch.zeros(obs.shape[0], dtype=torch.long, device=obs.device)
        q1_data, q2_data = self.critic(obs, action, zero_timestep)

        with torch.no_grad():
            if use_teacher_targets:
                next_action = self.sample_actions(
                    next_obs,
                    critic=self.target_critic,
                    deterministic=self.cfg.deterministic_target_sampling,
                )
            else:
                next_action = self.prior_actor(next_obs).detach()
                mix = 0.25
                next_action = (1.0 - mix) * next_action + mix * next_action_from_dataset
                next_action = next_action.clamp(-1.0, 1.0)
            next_q = self.target_critic.min_q(next_obs, next_action, zero_timestep)
            td_target = reward + self.cfg.discount * (1.0 - done) * next_q

        td_loss = F.mse_loss(q1_data, td_target) + F.mse_loss(q2_data, td_target)

        timestep = torch.randint(
            low=1,
            high=self.schedule.steps + 1,
            size=(obs.shape[0],),
            device=obs.device,
        )
        noisy_action, _ = add_noise(action, timestep, self.schedule)
        noisy_q1, noisy_q2 = self.critic(obs, noisy_action, timestep)
        min_data = torch.minimum(q1_data, q2_data).detach()
        consistency_loss = F.mse_loss(noisy_q1, min_data) + F.mse_loss(noisy_q2, min_data)

        random_actions = torch.empty(
            obs.shape[0],
            self.cfg.conservative_actions,
            self.action_dim,
            device=obs.device,
        ).uniform_(-1.0, 1.0)
        tiled_obs = obs[:, None, :].expand(-1, self.cfg.conservative_actions, -1)
        zero_many = torch.zeros(obs.shape[0] * self.cfg.conservative_actions, dtype=torch.long, device=obs.device)
        random_q = self.critic.min_q(
            tiled_obs.reshape(-1, obs.shape[-1]),
            random_actions.reshape(-1, self.action_dim),
            zero_many,
        ).view(obs.shape[0], self.cfg.conservative_actions)

        prior_action = self.prior_actor(obs).detach()
        prior_q = self.critic.min_q(obs, prior_action, zero_timestep).unsqueeze(1)
        sampled_action = self.sample_actions(obs, num_candidates=1, deterministic=True).detach()
        sampled_q = self.critic.min_q(obs, sampled_action, zero_timestep).unsqueeze(1)
        candidate_q = torch.cat([random_q, prior_q, sampled_q], dim=1)
        if student_action is not None:
            student_q = self.critic.min_q(obs, student_action.detach(), zero_timestep).unsqueeze(1)
            candidate_q = torch.cat([candidate_q, student_q], dim=1)
        conservative_target = torch.minimum(q1_data, q2_data)
        conservative_loss = (torch.logsumexp(candidate_q, dim=1) - conservative_target).mean()

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
