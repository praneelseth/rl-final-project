from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml


@dataclass
class DatasetConfig:
    dataset_id: str = "medium-diverse-v1"
    eval_dataset_id: str | None = None
    normalize_observations: bool = True
    reward_mode: str = "shaped"
    max_episodes: int | None = None
    download: bool = True


@dataclass
class RewardConfig:
    mode: str = "progress"
    distance_scale: float = 1.0
    success_bonus: float = 1.0
    step_penalty: float = 0.0
    clip_value: float | None = 5.0


@dataclass
class DiffusionConfig:
    steps: int = 20
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    guidance_scale: float = 30.0
    batch_action_samples: int = 8
    gradient_epsilon: float = 1e-6


@dataclass
class TeacherConfig:
    hidden_dim: int = 256
    hidden_layers: int = 3
    time_embedding_dim: int = 32
    learning_rate: float = 1e-3
    discount: float = 0.99
    tau: float = 0.005
    consistency_coef: float = 1.0
    conservative_coef: float = 0.1
    conservative_actions: int = 8
    action_l2_coef: float = 1e-4


@dataclass
class StudentConfig:
    hidden_dim: int = 256
    hidden_layers: int = 3
    learning_rate: float = 3e-4
    distill_coef: float = 1.0
    behavior_cloning_coef: float = 0.25


@dataclass
class TrainingConfig:
    seed: int = 7
    device: str = "auto"
    batch_size: int = 256
    total_steps: int = 20_000
    teacher_updates_per_step: int = 1
    student_updates_per_step: int = 1
    eval_every: int = 2_000
    eval_episodes: int = 5
    log_every: int = 100
    checkpoint_every: int = 2_000
    work_dir: str = "runs/antmaze-medium"

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ProjectConfig:
    experiment_name: str = "antmaze-medium"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @property
    def work_dir(self) -> Path:
        return Path(self.training.work_dir)


def _merge_dataclass(dc_type: type[Any], values: dict[str, Any] | None) -> Any:
    values = values or {}
    return dc_type(**values)


def load_config(path: str | Path) -> ProjectConfig:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw = raw or {}
    return ProjectConfig(
        experiment_name=raw.get("experiment_name", "antmaze-medium"),
        dataset=_merge_dataclass(DatasetConfig, raw.get("dataset")),
        reward=_merge_dataclass(RewardConfig, raw.get("reward")),
        diffusion=_merge_dataclass(DiffusionConfig, raw.get("diffusion")),
        teacher=_merge_dataclass(TeacherConfig, raw.get("teacher")),
        student=_merge_dataclass(StudentConfig, raw.get("student")),
        training=_merge_dataclass(TrainingConfig, raw.get("training")),
    )

