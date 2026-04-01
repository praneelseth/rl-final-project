# ACA Distillation for AntMaze

https://github.com/praneelseth/rl-final-project

This repository implements the project proposed in the RL milestone document: adapt the 2025 Actor-Critic without Actor (ACA) idea into an AntMaze setting, then distill the teacher's iterative action-selection process into a fast single-pass student policy.

The central question is:

Can we keep the good decision quality of an ACA-style critic-guided teacher while replacing its expensive multi-step denoising procedure with a lightweight actor that runs in one forward pass?

That makes this both an RL project and a systems-style inference-efficiency project. We care about:

- whether the teacher learns useful goal-reaching behavior in AntMaze
- whether the student imitates that behavior well enough
- how much inference latency is reduced by distillation

## Experiment Summary

The experiment is organized around three policies:

- Prior: a behavior cloning policy trained directly on the offline dataset
- Teacher: an ACA-style implicit policy that chooses actions by prior-guided critic denoising
- Student: an explicit MLP actor trained to imitate the teacher's final denoised action

The default setup is offline-first:

- training data comes from Minari AntMaze datasets
- a BC prior is pretrained first to stay close to dataset support
- the teacher is trained from static transitions
- the student is trained from teacher-generated supervision plus dataset actions
- evaluation is done in a recovered AntMaze environment using sparse success

This is not a verbatim transplant of an official ACA AntMaze code release, because the ACA paper is an online MuJoCo paper and I did not find a public AntMaze implementation to reuse directly. Instead, this repo implements the ACA core equations faithfully and then adapts them into an offline AntMaze pipeline that matches the proposal's objective.

## What ACA Means In This Repo

In standard actor-critic RL, there are two learned objects:

- a critic `Q(s, a)`
- an actor `pi(a | s)`

ACA removes the explicit actor. Instead, the policy is induced by the gradient field of a timestep-conditioned critic:

- the critic is `Q(s, a_t, t)`
- `a_t` is a noisy action at diffusion step `t`
- actions are sampled by starting from Gaussian noise and repeatedly refining the action using `grad_a Q(s, a_t, t)`

In the original paper, the online version removes the behavior prior and uses critic-only guidance. In this repo, the offline AntMaze version keeps an explicit learned behavior prior so teacher sampling stays closer to dataset support.

So in this repo:

- the prior is a normal behavior-cloned actor
- the teacher is not a normal actor network
- the teacher is an implicit policy defined by denoising under both the prior and the critic gradient
- the student is the fast distilled replacement for that implicit policy

The main teacher implementation lives in [aca_teacher.py](/Users/pseth/Documents/GitHub/aca-distillation/src/aca_distill/algos/aca_teacher.py), and the timestep-conditioned critic lives in [critic.py](/Users/pseth/Documents/GitHub/aca-distillation/src/aca_distill/models/critic.py).

## Main Hypothesis

The project hypothesis is:

1. An ACA-style teacher can represent multi-modal action choices better than a simple unimodal actor because it samples through an iterative denoising process.
2. That teacher is too slow for real-time control because each action requires many denoising steps.
3. A student actor can distill the teacher's final action and recover much of its performance with much lower inference latency.

For AntMaze, this matters because valid navigation behavior can be multi-modal:

- there may be several corridors or turning strategies that eventually reach the goal
- the policy must coordinate locomotion and navigation
- sparse rewards make pure actor learning difficult

The intended deliverable is a teacher-student tradeoff curve:

- teacher: slower, possibly stronger
- student: faster, hopefully close in success rate

## RL Setup

The current implementation is offline-first, not online-first.

That means:

- we train from an existing AntMaze dataset instead of collecting fresh interaction during training
- we use AntMaze transitions from Minari
- we evaluate learned behavior in the environment after training

This choice was made because it is the simplest stable path to an end-to-end milestone in AntMaze while still matching the proposal's core distillation idea.

The training pipeline is:

1. Load a Minari AntMaze dataset.
2. Flatten observations into a single vector.
3. Optionally normalize observations using dataset mean and standard deviation.
4. Build an ACA-style timestep-conditioned critic.
5. Train the teacher critic on offline transitions.
6. Sample teacher actions for dataset states.
7. Train the student actor to imitate those teacher actions.
8. Evaluate teacher and student in the recovered AntMaze environment.

The trainer is in [offline_trainer.py](/Users/pseth/Documents/GitHub/aca-distillation/src/aca_distill/trainers/offline_trainer.py), and the dataset preprocessing is in [antmaze.py](/Users/pseth/Documents/GitHub/aca-distillation/src/aca_distill/data/antmaze.py).

## Observation Design

AntMaze observations are goal-conditioned. In Gymnasium Robotics style, an observation usually contains:

- `observation`: low-level robot state
- `achieved_goal`: the ant's current task-space position
- `desired_goal`: the target position

This repo flattens those into one vector:

`[observation, achieved_goal, desired_goal]`

This is implemented by `flatten_antmaze_observation(...)` in [antmaze.py](/Users/pseth/Documents/GitHub/aca-distillation/src/aca_distill/data/antmaze.py).

Why flatten it:

- it keeps the baseline simple
- both teacher and student receive the full state-goal context
- it is easy to normalize and feed into MLPs

## Reward Design

There is an explicit reward design in this repo.

The project proposal asked for sparse-goal evaluation with light shaping during training if that is simpler. That is exactly what the current code does.

There are two reward notions in the codebase:

- evaluation reward / success semantics
- training reward shaping

### Evaluation reward

Evaluation is intended to stay aligned with AntMaze's sparse goal-reaching objective:

- success is 1 if the ant reaches the goal region
- otherwise success is 0
- when possible, success is computed from `achieved_goal` and `desired_goal`
- otherwise it falls back to checking whether the environment reward is positive

This logic is in `antmaze_success(...)` in [antmaze.py](/Users/pseth/Documents/GitHub/aca-distillation/src/aca_distill/data/antmaze.py).

This means the milestone metric is still fundamentally:

- did the agent reach the goal?

### Training reward shaping

For training, the default configs use a light progress-based shaped reward and then scale the reward inside the critic target. The shaping function is:

`shaped_reward = env_reward + progress_term + success_bonus - step_penalty`

where:

- `env_reward` is the dataset reward
- `progress_term` is proportional to how much closer the ant gets to the goal between consecutive states
- `success_bonus` adds extra reward when the goal is reached
- `step_penalty` can discourage dithering, though it defaults to `0.0`

In code:

- `progress_term = distance_scale * (current_distance - next_distance)`
- if the ant moves closer to the goal, this is positive
- if it moves away from the goal, this is negative

This reward is then optionally clipped.

The default reward settings are in:

- [antmaze_medium.yaml](/Users/pseth/Documents/GitHub/aca-distillation/configs/antmaze_medium.yaml)
- [antmaze_large.yaml](/Users/pseth/Documents/GitHub/aca-distillation/configs/antmaze_large.yaml)

and the reward implementation is in `shaped_reward(...)` inside [antmaze.py](/Users/pseth/Documents/GitHub/aca-distillation/src/aca_distill/data/antmaze.py).

### Why this reward design

The reward design was chosen to be simple and easy to modify later:

- it preserves sparse-goal evaluation
- it makes offline value learning less brittle than pure sparse rewards
- it avoids committing to a heavily engineered maze-specific reward
- it is compatible with later ablations

The intended interpretation is:

- training gets a dense hint about progress
- evaluation still measures actual navigation success

### How to change the reward later

You can change:

- `mode`
- `distance_scale`
- `success_bonus`
- `step_penalty`
- `clip_value`

in the YAML configs.

Natural ablations later would be:

- raw sparse reward only
- progress shaping without success bonus
- stronger step penalty
- different success thresholds

## Teacher Objective

The teacher follows the ACA paper structure, but adapts it to the offline setting with stronger behavior regularization:

1. A TD loss at timestep `t = 0`
2. A noisy consistency loss across diffusion steps
3. A lightweight conservative offline regularizer
4. A pretrained BC prior that anchors teacher sampling

### 1. TD loss at the denoised endpoint

At timestep `0`, the critic learns a Bellman-style target:

- current input: `(s, a_0, 0)`
- target: `r + gamma * Q_target(s', a'_0, 0)`

where:

- `a'_0` is sampled by the teacher from the next state
- the target critic is a slowly updated copy of the critic

This lets the teacher learn the value of denoised actions.

### 2. Noisy consistency loss

ACA's key modeling choice is that the critic should also know how to value noisy actions `a_t` for `t > 0`.

So the code:

- samples a diffusion timestep `t`
- adds forward-process noise to a clean dataset action
- trains `Q(s, a_t, t)` to match the denoised value estimate `Q(s, a_0, 0)`

This is the mechanism that makes critic-guided denoising possible. Without it, gradients at noisy actions would be poorly behaved.

### 3. Conservative offline regularization

Pure offline Q-learning can overestimate actions that are not supported by the dataset. To reduce that problem, the code adds a simple CQL-style penalty:

- sample random actions
- optionally include student actions and teacher samples
- push down the critic on unsupported action candidates relative to dataset actions

This is intentionally lightweight rather than a full standalone CQL implementation. The goal is to stabilize the offline teacher enough for a first project milestone.

## Teacher Action Sampling

Action sampling for the teacher is iterative and prior-guided.

Given a state:

1. Start from a BC prior action plus small Gaussian noise.
2. For `t = T, T-1, ..., 1`, compute `grad_a Q(s, a_t, t)`.
3. Compute a prior-guidance direction that pulls actions back toward the BC prior.
4. Normalize the gradients.
5. Apply the ACA-style denoising update with both prior guidance and critic guidance.
5. Optionally sample several candidate actions.
6. Pick the candidate with the highest final `Q(s, a_0, 0)`.

This is implemented in `sample_actions(...)` in [aca_teacher.py](/Users/pseth/Documents/GitHub/aca-distillation/src/aca_distill/algos/aca_teacher.py).

This sampling procedure is what creates the latency bottleneck:

- the teacher does not produce an action in one pass
- it must do repeated gradient-based refinement

That is exactly why distillation is useful here.

## Student Distillation Setup

The student is a simple MLP actor with `tanh` output. It is intentionally much simpler than the teacher's iterative inference.

The student loss is:

- distillation MSE to the teacher action
- plus an optional behavior cloning MSE to the dataset action

So the student is trained to stay near:

- what the teacher would do
- while still being lightly anchored to dataset behavior

This is implemented in [distillation.py](/Users/pseth/Documents/GitHub/aca-distillation/src/aca_distill/algos/distillation.py) and [student.py](/Users/pseth/Documents/GitHub/aca-distillation/src/aca_distill/models/student.py).

Why include the BC anchor:

- it reduces wild drift early in training
- it is helpful in offline settings
- it gives the student a stabilizing signal if the teacher is still weak

## Why Offline Instead Of Online Right Now

The original ACA paper is online. The current repo is offline-first for practical reasons:

- AntMaze offline datasets are easy to reproduce in Colab
- the proposal specifically mentions D4RL-style AntMaze settings
- getting a working teacher-student distillation pipeline matters more than immediate online fine-tuning for the first milestone

This repo is therefore best understood as:

- ACA-inspired teacher mechanics from the paper
- adapted to an offline AntMaze milestone setting

The code structure leaves room for online fine-tuning later.

## Evaluation

The main evaluation question is whether the student preserves useful behavior while reducing inference cost.

The current built-in evaluation reports:

- prior return
- prior success rate
- teacher return
- teacher success rate
- student return
- student success rate
- prior latency in milliseconds
- teacher latency in milliseconds
- student latency in milliseconds

This is handled by:

- [rollout.py](/Users/pseth/Documents/GitHub/aca-distillation/src/aca_distill/eval/rollout.py)
- [metrics.py](/Users/pseth/Documents/GitHub/aca-distillation/src/aca_distill/eval/metrics.py)

### Primary milestone metrics

For a first project milestone, the most important outputs are:

- `eval/success_rate`
- `student_eval/success_rate`
- `latency/mean_ms`
- `student_latency/mean_ms`

These directly answer the proposal's core claim:

- does distillation preserve useful behavior?
- is the student faster than the teacher?

### Secondary milestone metrics

The codebase is also set up to support:

- training loss curves
- heatmaps of ant positions
- saved action vectors for t-SNE

The plotting itself is not yet fully productized in the repo, but the rollout helpers provide the data needed for those milestone visuals.

## Current Simplifications

This repo is a strong first-pass research scaffold, not a final benchmark reproduction. Important simplifications include:

- one critic instead of a twin-critic ensemble
- offline conservative regularization is lightweight, not a complete CQL reproduction
- no SAC baseline yet
- no diffusion-policy baseline yet
- no multi-seed experiment harness yet
- no full online fine-tuning loop yet

These simplifications were intentional so the repo would reach a working end-to-end state quickly.

## Configs

The main configs are:

- [antmaze_medium.yaml](/Users/pseth/Documents/GitHub/aca-distillation/configs/antmaze_medium.yaml)
- [antmaze_large.yaml](/Users/pseth/Documents/GitHub/aca-distillation/configs/antmaze_large.yaml)

Key parameters to know:

- diffusion steps
- guidance scale
- teacher learning rate
- student learning rate
- conservative regularization strength
- reward shaping parameters
- evaluation frequency

For a fast milestone run in Colab, reduce:

- `total_steps`
- `eval_every`
- `eval_episodes`

Then scale up once the pipeline works.

## Repo Layout

- `configs/`: YAML experiment configs
- `src/aca_distill/models/`: critic and student networks
- `src/aca_distill/algos/`: diffusion schedule, ACA teacher logic, distillation logic
- `src/aca_distill/data/`: dataset loading, observation flattening, reward shaping
- `src/aca_distill/eval/`: rollouts and latency metrics
- `src/aca_distill/trainers/`: offline training loop
- `scripts/train_antmaze.py`: entrypoint script
- `notebooks/antmaze_colab.ipynb`: Colab starter notebook
- `tests/`: unit tests for the core pieces

## Install

```bash
pip install -e ".[dev]"
pip install -e ".[rl,viz]"
```

## Quick Start

Train:

```bash
python scripts/train_antmaze.py train --config configs/antmaze_medium.yaml
```

Evaluate:

```bash
python -m aca_distill.cli eval \
  --config configs/antmaze_medium.yaml \
  --checkpoint runs/antmaze-medium/checkpoints/latest.pt
```

## Dataset Choice

The default datasets are:

- `D4RL/antmaze/medium-diverse-v1`
- `D4RL/antmaze/large-diverse-v1`

These are loaded through Minari rather than the older D4RL package.

Why Minari:

- it is the current Farama dataset path
- it integrates with Gymnasium environments
- it supports recovering the environment for evaluation

## Recommended First Milestone Story

If you are writing up an early project milestone, the clean story is:

1. We implemented an ACA-style teacher for AntMaze using a timestep-conditioned critic and critic-guided denoising.
2. We added a distilled student actor trained to imitate the teacher's final denoised action.
3. We used light progress-based reward shaping for training while keeping sparse goal-reaching success as the main evaluation target.
4. We measured both success rate and inference latency to quantify the teacher-student tradeoff.

That story accurately reflects what is currently implemented.

## Next Extensions

The most natural next steps are:

- add SAC and diffusion-policy baselines
- add online fine-tuning after offline pretraining
- add multi-seed evaluation with IQM and confidence intervals
- add full plotting scripts for heatmaps and t-SNE
- test reward-shaping ablations
- compare different denoising budgets for the teacher

## Important Caveat

This repository currently gives you a coherent experimental scaffold and an initial implementation, not a claim of state-of-the-art AntMaze performance. The right interpretation is:

- the reward is designed
- the ACA-style teacher is implemented
- the student distillation setup is implemented
- the evaluation path is implemented
- the benchmark-quality comparison suite is still future work

That is usually exactly the right balance for a project milestone.
