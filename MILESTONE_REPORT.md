# Milestone Report: ACA Distillation on AntMaze

## Project Goal

The goal of this project is to adapt Actor-Critic without Actor (ACA) to the AntMaze setting and study whether an ACA-style teacher can be distilled into a fast single-pass student policy.

The intended contribution has two parts:

- an RL contribution: test whether an ACA-style critic-guided policy can learn useful goal-reaching behavior in AntMaze
- a systems contribution: measure the performance-latency tradeoff between the iterative ACA teacher and the distilled student

## Current Experimental Setup

The current implementation is an offline-first AntMaze pipeline:

- dataset: `D4RL/antmaze/medium-diverse-v1` through Minari
- teacher: timestep-conditioned ACA-style critic `Q(s, a_t, t)` with critic-guided denoising
- student: MLP actor trained with distillation loss and behavior cloning regularization
- training reward: sparse dataset reward plus light progress-based shaping
- evaluation: sparse success rate and return in the recovered AntMaze environment

## Current Results

The most recent run used the default medium config with 40,000 training steps.

Final evaluation results:

- teacher return mean: `0.0`
- teacher success rate: `0.0`
- student return mean: `0.0`
- student success rate: `0.0`
- teacher mean latency: `11.35 ms`
- student mean latency: `0.076 ms`
- teacher-to-student latency ratio: approximately `150x`

Training metrics at the end of the run:

- teacher total loss: approximately `148.53`
- teacher TD loss: approximately `148.26`
- teacher consistency loss: approximately `0.175`
- teacher conservative loss: approximately `1.90`
- student total loss: approximately `0.545`
- student BC loss: approximately `0.271`
- student distillation loss: approximately `0.409`

## What These Results Mean

The main positive result is that the distilled student is dramatically faster than the teacher at inference time. The student is roughly two orders of magnitude faster, which supports the motivation for distillation from a deployment perspective.

However, the control performance is currently unsuccessful:

- neither the teacher nor the student achieves nonzero success rate
- neither model achieves nonzero return
- the teacher does not appear to learn a useful AntMaze control policy
- because the teacher is weak, the student does not receive useful supervision

This means the project currently demonstrates the latency side of the tradeoff, but not the performance side.

## Why BC Appears Better Than Distillation

In separate runs and from the training curves, behavior cloning appears to perform better than teacher distillation.

That is plausible for the current codebase because:

- the student BC loss is lower than the student distillation loss
- the dataset actions are likely better supervision than the current teacher actions
- if the teacher is not learning useful behavior, then distillation actively teaches the student the wrong target

In other words, the current teacher is not yet a good teacher. Under those conditions, behavior cloning is a stronger baseline than teacher imitation.

This is actually an informative result:

- distillation only helps if the teacher is better than the supervised baseline
- in offline sparse long-horizon AntMaze, that condition currently does not hold

## Main Problems Observed

### 1. The teacher is still not learning useful AntMaze behavior

The strongest evidence is simple:

- success rate stays at `0.0`
- return stays at `0.0`
- teacher TD loss remains very large late in training

The teacher loss is dominated by the TD term rather than the consistency or conservative terms. That suggests the core value-learning problem is not being solved.

### 2. Offline ACA is likely unstable in sparse long-horizon AntMaze

ACA was introduced as an online MuJoCo method. AntMaze is different in several important ways:

- it is sparse reward
- it is long horizon
- it is goal-conditioned
- offline bootstrapping is especially brittle

This means a direct ACA-style adaptation may be fundamentally much harder than expected, especially with a single critic and lightweight offline regularization.

### 3. The teacher targets may still be too poor for distillation

Even after stabilizing parts of the training loop, the student still appears to trust teacher actions that are worse than the dataset actions. The gap between BC loss and distillation loss reflects that.

The likely failure mode is:

- the critic does not provide a reliable gradient field
- denoising does not produce high-value actions
- the student receives noisy or misleading action targets

### 4. The current evaluation is showing failure honestly

This is important for interpretation. The current evaluation is not hiding the problem:

- there is no artificial success inflation
- sparse-goal evaluation remains at zero
- the run is correctly telling us that the current method is not yet solving the environment

That makes the result disappointing, but scientifically useful.

## Likely Technical Causes

The most likely causes of failure at this stage are:

- ACA was not originally designed for offline sparse long-horizon AntMaze
- the teacher uses a single critic rather than a more stable double-Q setup
- the offline regularization is lightweight and may not be strong enough
- the reward shaping may still be too weak to produce a learnable critic
- the teacher action sampling may be too noisy or too far out of the dataset support
- the student is being asked to imitate a teacher that has not surpassed behavior cloning

There may also be a representation issue:

- AntMaze combines navigation and locomotion
- a simple flattened observation MLP may be insufficient to learn a useful implicit policy through critic gradients alone

## What I Would Try Next

The next fixes and experiments I would prioritize are:

1. Add a pure behavior cloning baseline and report it explicitly alongside teacher and student.
2. Add stronger offline RL baselines such as IQL or CQL for comparison.
3. Replace the single critic with a double-critic architecture.
4. Increase the strength of dataset support constraints during teacher sampling.
5. Try a simpler environment first, such as Maze2D or smaller AntMaze variants, before medium-diverse.
6. Evaluate teacher denoising quality directly on dataset states rather than only through full environment rollout.
7. Separate the project into two questions:
   - can ACA learn useful actions offline in AntMaze at all?
   - if yes, does distillation preserve those actions?

The most important immediate step is to establish a strong baseline ladder:

- BC
- IQL or CQL
- ACA teacher
- distilled student

Without that ladder, it is hard to interpret whether ACA is underperforming because the implementation is weak or because the problem setting is a poor fit for ACA.

## Recommended Interpretation For The Milestone

A fair milestone interpretation is:

- the ACA-style teacher and distilled student pipeline has been implemented end to end
- the latency comparison works and shows a strong speedup for the student
- the current ACA adaptation does not yet achieve useful AntMaze performance
- behavior cloning currently appears stronger than distillation, which suggests the teacher has not yet become better than the dataset policy

This is a valid milestone result because it narrows the research question:

- the difficulty is no longer “can we code it?”
- the difficulty is now “is ACA actually well-suited for offline sparse long-horizon AntMaze?”

## Possible Project Pivot

Given the current results, a good pivot would be to turn the project into a study of **where ACA fails or succeeds outside the setting it was originally introduced for**.

That pivot is still closely related to ACA and arguably more interesting scientifically.

### Pivot Option 1: ACA on offline sparse long-horizon control

New question:

How does ACA behave when transferred from online MuJoCo benchmarks to offline sparse-reward goal-conditioned tasks such as AntMaze?

This is interesting because the original ACA paper focused on online continuous control, not offline goal-conditioned maze tasks. A negative result here would still be meaningful if it is benchmarked carefully.

Possible benchmark comparisons:

- BC
- IQL
- CQL
- ACA teacher
- distilled ACA student

This would turn the project into a transferability study of ACA.

### Pivot Option 2: ACA as a latency-performance baseline against diffusion policies

New question:

Does ACA provide a better latency-performance tradeoff than diffusion-based policies in offline goal-conditioned control?

This is attractive because ACA’s biggest conceptual advantage is efficiency relative to diffusion-style action generation. Even if absolute AntMaze performance is weak, it may still be interesting to compare:

- teacher latency
- student latency
- diffusion-policy latency
- BC latency

This could become a benchmarking project around inference cost rather than only raw return.

### Pivot Option 3: Distillation only after a stronger teacher

New question:

Can ACA distillation work if the teacher is replaced or augmented by a stronger offline teacher?

For example:

- use a stronger offline method as teacher
- distill that policy into a lightweight student
- compare the latency/performance tradeoff against ACA and BC

This would preserve the distillation theme while reducing dependence on ACA solving AntMaze by itself.

## Best Pivot Recommendation

The strongest and most honest pivot is:

**Benchmark ACA and distilled ACA against BC and modern offline RL baselines on offline AntMaze, with a focus on both performance and inference latency.**

Why this is a strong pivot:

- it stays close to the original proposal
- it uses the current codebase rather than discarding it
- it turns the current failure into a research question
- it produces a meaningful comparison even if ACA underperforms

The resulting paper question becomes:

Is ACA competitive outside its original online MuJoCo setting, and if not, where does it break down relative to offline baselines such as BC, IQL, and CQL?

That is a real and interesting question.

## Short Conclusion

At the current milestone, the project has succeeded technically but not yet empirically.

Succeeded:

- end-to-end ACA teacher implementation
- end-to-end student distillation implementation
- offline AntMaze training pipeline
- latency evaluation showing roughly `150x` faster student inference

Not yet succeeded:

- teacher learning useful AntMaze behavior
- student preserving useful teacher behavior
- outperforming behavior cloning

The most honest takeaway is that ACA adaptation to offline AntMaze is currently failing, and that failure points toward a more interesting next-stage question: whether ACA generalizes beyond its original benchmark regime, and how it compares against offline RL baselines in sparse long-horizon control.
