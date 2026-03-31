# ACA Distillation for AntMaze

This repo sets up the project from the proposal as a clean, Colab-friendly research codebase:

- an ACA-style teacher that samples actions by critic-guided denoising
- a distilled student MLP actor that imitates the teacher in one forward pass
- offline-first AntMaze training on Minari/D4RL-style datasets
- light reward shaping for training while preserving sparse-goal evaluation
- evaluation helpers for success rate, latency, heatmaps, and action embeddings

## What is implemented

The original ACA paper is an online MuJoCo method, not an AntMaze code release. I therefore implemented a faithful ACA-style teacher from the paper equations and wrapped it in an offline AntMaze pipeline that is practical for your proposal:

- `NoiseLevelCritic`: timestep-conditioned critic `Q(s, a_t, t)`
- `ACATeacher`: action sampling via critic-guided denoising, based on ACA Definition 1
- `StudentActor`: single-pass actor trained with MSE distillation to the teacher
- conservative offline training glue: optional CQL-style regularization plus BC anchoring
- Minari dataset loader for AntMaze with flattening, normalization, and shaped-reward preprocessing
- evaluation utilities for latency, success rate, heatmaps, and t-SNE-ready action dumps

This gives us the full end-to-end skeleton and the core learning logic. It is intentionally offline-first because AntMaze datasets are the simplest stable starting point for the proposal, and the code is organized so online fine-tuning can be added later.

## Project plan

1. Start with `D4RL/antmaze/medium-diverse-v1` and `D4RL/antmaze/large-diverse-v1` Minari AntMaze datasets.
2. Train the ACA teacher offline with:
   - TD loss at `t = 0`
   - noisy consistency loss across diffusion steps
   - optional conservative Q regularization
3. Distill the teacher into a student MLP with:
   - teacher action regression
   - optional BC anchor to dataset actions
4. Evaluate teacher vs student on:
   - sparse success rate
   - action inference latency
   - ant position heatmaps
   - action embeddings for t-SNE
5. Extend later with:
   - online fine-tuning
   - more rigorous baselines
   - multi-seed IQM / confidence intervals

## Repo layout

- `configs/`: YAML configs for medium and large AntMaze runs
- `src/aca_distill/`: training code, models, env wrappers, and evaluation
- `scripts/train_antmaze.py`: simple CLI wrapper
- `notebooks/antmaze_colab.ipynb`: Colab entry notebook
- `tests/`: unit tests for diffusion math, reward shaping, and model shapes

## Install

```bash
pip install -e ".[dev]"
pip install -e ".[rl,viz]"
```

For Colab, the notebook includes the exact install cell.

## Dataset choices

The default configs use the newer Minari AntMaze datasets rather than the old D4RL Python package. According to the Minari docs, AntMaze datasets like `medium-diverse-v1` and `large-diverse-v1` recover the corresponding Gymnasium Robotics environments and preserve sparse goal-reaching evaluation semantics.

## Quick start

```bash
python scripts/train_antmaze.py train --config configs/antmaze_medium.yaml
```

Evaluate a checkpoint:

```bash
python -m aca_distill.cli eval \
  --config configs/antmaze_medium.yaml \
  --checkpoint runs/antmaze-medium/checkpoints/latest.pt
```

## Key assumptions

- The ACA paper did not appear to ship a public AntMaze codebase that we could directly transplant.
- AntMaze offline training benefits from conservative regularization, so this repo includes an optional lightweight CQL-style penalty.
- Reward shaping is used only as a preprocessing/training convenience; evaluation defaults to sparse success.

## Next extensions

- add SAC and diffusion-policy baselines
- add online fine-tuning on recovered AntMaze environments
- add multi-seed experiment sweeps and aggregate statistics
