# Milestone Report: Prior-Guided ACA Distillation on AntMaze

## Project Goal

The goal of this project is to adapt Actor-Critic without Actor (ACA) to the AntMaze setting and study whether an ACA-style teacher can be distilled into a fast single-pass student policy.

The intended contribution has two parts:

- an RL contribution: test whether an ACA-style critic-guided policy can learn useful goal-reaching behavior in AntMaze
- a systems contribution: measure the performance-latency tradeoff between iterative ACA-style action generation and fast feedforward policies

## What Was Run In This Milestone

The latest notebook run used the updated prior-guided offline ACA pipeline rather than the earlier critic-only version.

The training recipe in the notebook was:

- dataset: `D4RL/antmaze/medium-diverse-v1`
- prior pretraining steps: `8000`
- teacher/student training steps: `20000`
- evaluation every `2000` steps
- evaluation episodes: `5`
- work directory: `runs/antmaze-medium-prior-guided-colab`

The current pipeline contains three learned policies:

- Prior: a behavior cloning policy trained directly on the dataset
- Teacher: a prior-guided ACA-style implicit policy with denoising and twin critics
- Student: an MLP policy distilled from the teacher while staying behavior-cloned to the data

## Current Results

Final evaluation at step `20000`:

- prior return mean: `0.0`
- prior success rate: `0.0`
- teacher return mean: `0.0`
- teacher success rate: `0.0`
- student return mean: `0.0`
- student success rate: `0.0`

Final latency:

- prior mean latency: `0.0758 ms`
- teacher mean latency: `17.49 ms`
- student mean latency: `0.0750 ms`
- teacher-to-student latency ratio: approximately `233x`

Final training metrics:

- teacher total loss: approximately `35.78`
- teacher TD loss: approximately `26.09`
- teacher consistency loss: approximately `9.29`
- teacher conservative loss: approximately `2.03`
- student total loss: approximately `0.120`
- student BC loss: approximately `0.109`
- student distillation loss: approximately `0.0111`

## What The Plots Show

The milestone notebook contains two main plot groups.

### 1. Prior / teacher / student training curves

The loss curves suggest that optimization is much healthier than in the earlier runs:

- the prior BC pretraining completes cleanly
- the teacher loss is now on the order of tens rather than exploding into the hundreds
- the student losses are low and stable
- the student distillation loss is much lower than the student BC loss by the end of training

This is an improvement over the earlier version of the project, where the teacher loss was much larger and more unstable.

### 2. Success / return / latency curves

The evaluation plots show a much less encouraging picture:

- prior success stays at `0.0`
- teacher success stays at `0.0`
- student success stays at `0.0`
- prior, teacher, and student return all remain `0.0`
- teacher latency is consistently far higher than either the prior or student latency

So the new method improved optimization behavior but did not improve actual task performance.

## What These Results Mean

The most honest summary is:

- the optimization problem looks more stable
- the control problem is still unsolved

This matters because it separates two questions:

1. Is the training pipeline numerically healthier now?
2. Does the learned policy actually solve AntMaze?

The answer appears to be:

- yes for the first question
- no for the second question

That is still a useful research outcome. It suggests that the original failure was not only due to a coding bug or training instability. Even after making the method more offline-correct and behavior-regularized, AntMaze medium-diverse still appears too difficult for the current ACA adaptation.

## Comparison To The Earlier Version

Compared with the earlier milestone attempt:

- the project now includes a BC prior
- the teacher is prior-guided rather than pure critic-guided from Gaussian noise
- the teacher uses twin critics instead of a single critic
- reward scaling is closer to ACA
- the teacher loss is much smaller and more stable
- the student distillation loss is now very low

However:

- all three policies still achieve zero return and zero success
- the prior itself is not solving the task
- the teacher is not outperforming the prior
- the student is not outperforming either one

This means the new result is not “the method works.” The new result is:

the updated method trains more stably, but stability alone is not enough to produce useful AntMaze behavior.

## Why The Student Distillation Loss Is Low But Performance Is Still Bad

One interesting feature of the new run is that the student distillation loss is very small:

- student distillation loss: about `0.011`
- student BC loss: about `0.109`

This likely means the student is doing a good job matching the teacher's action outputs.

But that does not imply the student is good at the task. It only implies:

- teacher and student are behaviorally similar in action space
- both may still be producing poor actions for goal-reaching

This is an important lesson for the project:

- low distillation error is not enough
- the teacher must actually be better than the baseline policy before distillation becomes valuable

## Main Problems Observed

### 1. The prior itself does not solve the task

The BC prior has:

- return `0.0`
- success `0.0`

That means the dataset policy induced by simple BC is already failing on this evaluation setting. If the prior is weak, then the prior-guided teacher starts from a poor base policy.

### 2. The teacher is not improving over the prior

The teacher was supposed to refine the prior through critic-guided denoising. But in this run:

- teacher return remains `0.0`
- teacher success remains `0.0`

So the teacher is not converting the prior into a stronger controller.

### 3. The student is faithfully imitating a weak teacher

The student distillation loss is small, which means the student is learning the teacher's action outputs. But because the teacher itself is not good, this gives no control benefit.

### 4. AntMaze medium-diverse may simply be too hard for this setup

This task is:

- sparse reward
- long horizon
- offline
- high-dimensional
- goal-conditioned

That is a very demanding regime. ACA was introduced primarily in online MuJoCo settings, not offline sparse long-horizon maze navigation.

## Likely Technical Reasons It Is Still Failing

The most likely reasons are:

- the BC prior is too weak to serve as a useful anchor
- the critic is still not learning a value field that meaningfully improves actions
- reward shaping is still not enough to turn AntMaze into an easy offline value-learning problem
- the current observation encoder and MLPs may be too simple for the navigation + locomotion structure
- medium-diverse may require stronger offline RL methods than this ACA adaptation currently provides

Another important possibility is evaluation mismatch:

- BC on offline AntMaze often needs careful implementation choices to show success
- if the prior cannot solve the task, the teacher has very little chance to do better

## What Worked In This Milestone

Even though the task performance is still zero, some parts of the project did succeed:

- the codebase now supports a prior-guided offline ACA variant
- the three-policy comparison is now explicit: prior vs teacher vs student
- the training curves are much healthier than before
- the student preserves the teacher's action outputs closely
- the latency comparison is very strong

The latency result is especially clear:

- teacher: `17.49 ms`
- student: `0.075 ms`

So if a stronger teacher can eventually be learned, distillation would likely provide a real systems benefit.

## What I Would Try Next

The next experiments I would prioritize are:

1. Benchmark pure BC explicitly and carefully as a main baseline.
2. Add strong offline RL baselines such as IQL and CQL.
3. Move to an easier environment before AntMaze medium-diverse:
   - Maze2D
   - smaller AntMaze variants
   - or AntMaze umaze
4. Test whether the teacher improves over the prior on dataset states before requiring full rollout success.
5. Try online fine-tuning after prior initialization instead of pure offline learning.
6. Improve representations:
   - better goal-conditioning
   - larger networks
   - potentially separate encoders for state and goal

The most important next question is:

Can ACA improve a prior policy at all in a simpler or partially online setting?

Right now, the answer on offline AntMaze medium-diverse appears to be no.

## Recommended Interpretation For Submission

A fair milestone interpretation is:

- the project successfully implemented a prior-guided ACA teacher, BC prior, and distilled student
- the updated method is significantly more stable than the earlier version
- despite better optimization behavior, none of the three policies solved AntMaze medium-diverse in the current offline setting
- the student achieves essentially the same latency as the BC prior and is over `200x` faster than the teacher
- the core unresolved issue is not distillation itself, but the inability of the teacher to outperform the prior

That is a useful milestone because it sharpens the real research question.

## Possible Project Pivot

Based on these results, the most interesting pivot is:

study whether ACA can improve a behavior prior in settings outside the online MuJoCo regime where it was originally proposed.

That can be framed in several ways.

### Pivot Option 1: ACA as a prior-improvement method

New question:

Can ACA improve over a BC prior on offline control tasks?

This is a cleaner question than “can distilled ACA solve AntMaze?” because it isolates the key missing ingredient:

- is the teacher actually better than the prior?

### Pivot Option 2: ACA benchmarking against offline RL baselines

New question:

How does ACA compare with BC, IQL, and CQL on offline sparse long-horizon tasks?

This is scientifically useful even if ACA underperforms, because ACA has not been primarily evaluated in this regime.

### Pivot Option 3: ACA latency benchmarking

New question:

What is the latency-performance tradeoff of prior, ACA teacher, and distilled student compared with other policy classes such as diffusion policies or standard offline RL actors?

This leverages the strongest result currently available in the project:

- the teacher is much slower
- the feedforward policies are much faster

## Best Pivot Recommendation

The strongest pivot is:

Benchmark whether ACA can improve over a BC prior, and compare that to standard offline RL baselines such as IQL and CQL.

Why this is the best pivot:

- it stays close to the current codebase
- it uses the new prior-guided formulation directly
- it turns the current negative result into a concrete research question
- it avoids overclaiming that distillation is the main bottleneck when the teacher itself is still weak

## Short Conclusion

At this milestone, the project improved substantially at the implementation and optimization level, but still did not achieve task success.

Succeeded:

- prior-guided ACA variant implemented
- prior / teacher / student evaluation implemented
- much more stable training behavior
- very strong latency gap between teacher and feedforward policies

Not yet succeeded:

- prior solving AntMaze medium-diverse
- teacher improving over the prior
- student delivering useful control performance

The most honest takeaway is:

the project now has a better ACA adaptation and a much clearer experimental structure, but offline AntMaze medium-diverse still appears to be too hard for the current method.

That makes the next-stage project question more precise and more interesting: whether ACA can improve a prior policy at all in offline or simpler goal-conditioned control, and how that compares to established offline RL baselines.
