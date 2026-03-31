from __future__ import annotations

import argparse

import torch

from aca_distill.algos.aca_teacher import ACATeacher
from aca_distill.algos.diffusion import DiffusionSchedule
from aca_distill.algos.distillation import StudentDistillation
from aca_distill.config import load_config
from aca_distill.data.antmaze import load_antmaze_dataset
from aca_distill.eval.metrics import measure_latency_ms
from aca_distill.eval.rollout import evaluate_policy, recover_antmaze_env
from aca_distill.models.critic import NoiseLevelCritic
from aca_distill.models.student import StudentActor
from aca_distill.trainers.offline_trainer import OfflineAntMazeTrainer
from aca_distill.utils.checkpoint import load_checkpoint
from aca_distill.utils.seeding import seed_everything


def build_system(config_path: str):
    cfg = load_config(config_path)
    seed_everything(cfg.training.seed)
    device = torch.device(cfg.training.resolved_device())
    replay = load_antmaze_dataset(cfg.dataset, cfg.reward)

    critic = NoiseLevelCritic(
        obs_dim=replay.obs_dim,
        action_dim=replay.action_dim,
        hidden_dim=cfg.teacher.hidden_dim,
        hidden_layers=cfg.teacher.hidden_layers,
        time_embedding_dim=cfg.teacher.time_embedding_dim,
    ).to(device)
    actor = StudentActor(
        obs_dim=replay.obs_dim,
        action_dim=replay.action_dim,
        hidden_dim=cfg.student.hidden_dim,
        hidden_layers=cfg.student.hidden_layers,
    ).to(device)
    schedule = DiffusionSchedule(
        steps=cfg.diffusion.steps,
        beta_start=cfg.diffusion.beta_start,
        beta_end=cfg.diffusion.beta_end,
        device=device,
    )
    teacher = ACATeacher(
        critic=critic,
        schedule=schedule,
        cfg=cfg.teacher,
        diffusion_cfg=cfg.diffusion,
        action_dim=replay.action_dim,
    )
    student = StudentDistillation(actor=actor, cfg=cfg.student)
    return cfg, replay, teacher, student, device


def train_command(args: argparse.Namespace) -> None:
    cfg, replay, teacher, student, device = build_system(args.config)
    trainer = OfflineAntMazeTrainer(cfg, replay, teacher, student, device)
    trainer.train()


def eval_command(args: argparse.Namespace) -> None:
    cfg, replay, teacher, student, device = build_system(args.config)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    teacher.critic.load_state_dict(checkpoint["teacher"]["critic"])
    teacher.target_critic.load_state_dict(checkpoint["teacher"]["target_critic"])
    student.actor.load_state_dict(checkpoint["student"]["actor"])

    mean = None if replay.observation_mean is None else replay.observation_mean.numpy()
    std = None if replay.observation_std is None else replay.observation_std.numpy()
    env = recover_antmaze_env(cfg.dataset.eval_dataset_id or cfg.dataset.dataset_id, mean, std)

    teacher_action_fn = lambda obs: teacher.sample_actions(obs, deterministic=True)
    student_action_fn = lambda obs: student.actor(obs)
    teacher_metrics = evaluate_policy(
        env,
        teacher_action_fn,
        device,
        episodes=cfg.training.eval_episodes,
        max_steps=cfg.training.max_eval_steps,
    )
    student_metrics = evaluate_policy(
        env,
        student_action_fn,
        device,
        episodes=cfg.training.eval_episodes,
        max_steps=cfg.training.max_eval_steps,
    )
    latency = measure_latency_ms(teacher_action_fn, replay.obs[:32].to(device), repeats=20)
    student_latency = measure_latency_ms(student_action_fn, replay.obs[:32].to(device), repeats=20)

    print("Teacher metrics:", teacher_metrics)
    print("Student metrics:", student_metrics)
    print("Teacher latency:", latency)
    print("Student latency:", student_latency)


def main() -> None:
    parser = argparse.ArgumentParser(description="ACA distillation for AntMaze")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", required=True)
    train_parser.set_defaults(func=train_command)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--config", required=True)
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.set_defaults(func=eval_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
