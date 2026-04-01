from __future__ import annotations

from pathlib import Path

import torch
from tqdm import trange

from aca_distill.algos.aca_teacher import ACATeacher
from aca_distill.algos.behavior_cloning import BehaviorCloningPrior
from aca_distill.algos.distillation import StudentDistillation
from aca_distill.config import ProjectConfig
from aca_distill.data.antmaze import OfflineReplayBuffer
from aca_distill.eval.metrics import measure_latency_ms
from aca_distill.eval.rollout import evaluate_policy, recover_antmaze_env
from aca_distill.utils.checkpoint import save_checkpoint
from aca_distill.utils.logging import JsonlLogger, MetricAverager


class OfflineAntMazeTrainer:
    def __init__(
        self,
        cfg: ProjectConfig,
        replay: OfflineReplayBuffer,
        prior: BehaviorCloningPrior,
        teacher: ACATeacher,
        student: StudentDistillation,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.replay = replay
        self.prior = prior
        self.teacher = teacher
        self.student = student
        self.device = device
        self.work_dir = Path(cfg.training.work_dir)
        self.logger = JsonlLogger(self.work_dir / "metrics.jsonl")

    def _checkpoint_payload(self, step: int) -> dict:
        return {
            "step": step,
            "config": self.cfg,
            "teacher": {
                "critic": self.teacher.critic.state_dict(),
                "target_critic": self.teacher.target_critic.state_dict(),
                "optimizer": self.teacher.optimizer.state_dict(),
            },
            "prior": {
                "actor": self.prior.actor.state_dict(),
                "optimizer": self.prior.optimizer.state_dict(),
            },
            "student": {
                "actor": self.student.actor.state_dict(),
                "optimizer": self.student.optimizer.state_dict(),
            },
            "stats": {
                "observation_mean": self.replay.observation_mean,
                "observation_std": self.replay.observation_std,
            },
        }

    def _teacher_action_fn(self):
        return lambda obs: self.teacher.sample_actions(obs, deterministic=True)

    def _prior_action_fn(self):
        return lambda obs: self.prior.actor(obs)

    def _student_action_fn(self):
        return lambda obs: self.student.actor(obs)

    def _maybe_eval(self, step: int) -> dict[str, float]:
        if step % self.cfg.training.eval_every != 0:
            return {}
        try:
            mean = None if self.replay.observation_mean is None else self.replay.observation_mean.numpy()
            std = None if self.replay.observation_std is None else self.replay.observation_std.numpy()
            env = recover_antmaze_env(self.cfg.dataset.eval_dataset_id or self.cfg.dataset.dataset_id, mean, std)
        except Exception as exc:
            return {"eval/skipped": 1.0, "eval/error": 1.0}

        prior_metrics = evaluate_policy(
            env,
            self._prior_action_fn(),
            self.device,
            episodes=self.cfg.training.eval_episodes,
            max_steps=self.cfg.training.max_eval_steps,
        )
        teacher_metrics = evaluate_policy(
            env,
            self._teacher_action_fn(),
            self.device,
            episodes=self.cfg.training.eval_episodes,
            max_steps=self.cfg.training.max_eval_steps,
        )
        student_metrics = evaluate_policy(
            env,
            self._student_action_fn(),
            self.device,
            episodes=self.cfg.training.eval_episodes,
            max_steps=self.cfg.training.max_eval_steps,
        )
        sample_obs = self.replay.obs[:32].to(self.device)
        latency_prior = {
            key.replace("latency/", "prior_latency/"): value
            for key, value in measure_latency_ms(self._prior_action_fn(), sample_obs, repeats=20).items()
        }
        latency_teacher = measure_latency_ms(self._teacher_action_fn(), sample_obs, repeats=20)
        latency_student = {
            key.replace("latency/", "student_latency/"): value
            for key, value in measure_latency_ms(self._student_action_fn(), sample_obs, repeats=20).items()
        }
        return {
            **{key.replace("eval/", "prior_eval/"): value for key, value in prior_metrics.items()},
            **teacher_metrics,
            **{key.replace("eval/", "student_eval/"): value for key, value in student_metrics.items()},
            **latency_prior,
            **latency_teacher,
            **latency_student,
        }

    def train(self) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        averager = MetricAverager()

        for step in trange(1, self.cfg.prior.pretrain_steps + 1, desc="prior"):
            batch = self.replay.sample(self.cfg.training.batch_size, self.device)
            averager.update(self.prior.update(batch["obs"], batch["action"]))
            if step % self.cfg.training.log_every == 0:
                self.logger.log({"prior_step": step, **averager.compute()})
                averager.reset()

        for step in trange(1, self.cfg.training.total_steps + 1):
            batch = self.replay.sample(self.cfg.training.batch_size, self.device)
            use_teacher_targets = step > self.cfg.teacher.dataset_target_warmstart_steps
            if step > self.cfg.student.warmstart_behavior_cloning_steps:
                student_action = self.student.actor(batch["obs"]).detach()
            else:
                student_action = None
            for _ in range(self.cfg.training.teacher_updates_per_step):
                averager.update(
                    self.teacher.update(
                        batch,
                        student_action=student_action,
                        use_teacher_targets=use_teacher_targets,
                    )
                )

            if step <= self.cfg.student.warmstart_behavior_cloning_steps:
                teacher_action = None
                bc_coef = self.cfg.student.warmstart_behavior_cloning_coef
            else:
                with torch.no_grad():
                    teacher_action = self.teacher.sample_actions(batch["obs"], deterministic=True)
                bc_coef = self.cfg.student.behavior_cloning_coef
            for _ in range(self.cfg.training.student_updates_per_step):
                averager.update(
                    self.student.update(
                        batch["obs"],
                        teacher_action=teacher_action,
                        dataset_action=batch["action"],
                        behavior_cloning_coef=bc_coef,
                    )
                )

            if step % self.cfg.training.log_every == 0:
                payload = {"step": step, **averager.compute()}
                self.logger.log(payload)
                averager.reset()

            if step % self.cfg.training.eval_every == 0:
                self.logger.log({"step": step, **self._maybe_eval(step)})

            if step % self.cfg.training.checkpoint_every == 0:
                save_checkpoint(self.work_dir / "checkpoints" / f"step_{step}.pt", self._checkpoint_payload(step))
                save_checkpoint(self.work_dir / "checkpoints" / "latest.pt", self._checkpoint_payload(step))
