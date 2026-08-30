---
title: "Learning-Rate Schedulers"
description: "why constant LR is suboptimal (condition-number oscillation, `η·σ²` noise ball, Robbins-Monro two-sum conditions); schedulers (Step/MultiStep/Exponential/Cosine/SGDR/CLR/OneCycle/ReduceLROnPlateau); warmup & why transformers need it (Adam 2nd-moment init + softmax saturation); LR finder, linear-vs-√k batch scaling; PyTorch mechanics & common mistakes (per-epoch vs per-batch, ordering, constructor implicit step, save state); practical guidelines by model type"
category: "Training Dynamics & Optimization"
order: 23
updatedDate: "2026-08-25T11:48:05.159Z"
---
A fixed LR forces a tradeoff: **large** → fast early progress but oscillation near minima; **small** →
precise convergence but wasted compute early. **Scheduling** resolves this by varying the rate over
training — big imprecise steps when they're affordable, small fine-grained steps when precision
matters. (LR *basics* and the SGD/Adam update rules: [[learning-rate]].)

---

## Why a constant LR is suboptimal

1. **Oscillation in narrow valleys (the condition-number problem).** When the loss has high curvature
   in one direction and low in another, a constant rate **bounces across the valley walls** while
   crawling along the floor. The rate must be small enough for the high-curvature direction → painfully
   slow progress in the low-curvature one. (See [[optimization]] on condition number.)
2. **Overshooting near convergence.** Mini-batch **gradient noise** (variance `σ²`) never vanishes.
   With constant LR, the iterates converge to a **noise ball** whose size ∝ `η·σ²` — a *neighborhood*
   of the minimum, not the minimum itself. Decaying `η` shrinks the ball.
3. **Robbins–Monro convergence conditions.** For SGD to converge to the actual minimum (not just a
   neighborhood), the step sizes must satisfy **two sums**:

$$\sum_t \eta_t = \infty \qquad \text{and} \qquad \sum_t \eta_t^2 < \infty$$

   The **first** (diverges) → η can't decay *too fast* (you must be able to travel any distance). The
   **second** (squares converge) → η must decay *fast enough* to damp the noise. `η_t ∝ 1/t` is the
   canonical schedule satisfying both; roughly `η_t ∝ 1/t^p` with `½ < p ≤ 1`. (Decaying *slower* than
   `1/t`, e.g. `1/√t`, satisfies the first but **violates the second**.)

---

## The schedulers

### StepLR — piecewise-constant decay
Hold LR constant for `s` epochs, then multiply by `γ ∈ (0,1)`. After `e` epochs there have been
`floor(e/s)` decay events → `LR = LR₀ · γ^⌊e/s⌋`. **+** simple, predictable. **−** abrupt jumps at step
boundaries can cause temporary instability.

### MultiStepLR — arbitrary milestones
Generalizes StepLR: multiply by `γ` at explicit **milestone** epochs. Use when the good schedule is
known empirically but irregular (e.g. milestones at 30%/60%/80% of training).

### ExponentialLR — smooth multiplicative decay
Multiply by `γ` **every** epoch (≡ StepLR with `s=1`) → smooth curve, no abrupt jumps. **−** can decay
too aggressively; `γ` needs calibration (too close to 1 → no decay; too small → premature convergence).
Common in older recipes; modern practice prefers cosine / OneCycle.

### CosineAnnealingLR — the safe default
LR follows a **cosine** from `LR_max` (t=0) to `LR_min` (t=T). Decay is **non-uniform**: slow at the
start (cosine near its flat peak), fastest through the middle, slow again near the end. This matches
training: high rate early for fast progress, steady decrease mid-training, gentle annealing to settle
into a precise minimum. **Few hyperparameters, robust across architectures/datasets.**

### CosineAnnealingWarmRestarts (SGDR)
Repeat the cosine cycle: after each cycle, **reset to `LR_max`** and decay again. The sudden increase
helps **escape poor local minima**; cycles can be **lengthened** over time for longer refinement later.

### Cyclical LR (CLR)
Oscillate between `LR_min` and `LR_max`; periodic high rates help escape saddle points / sharp minima.
Three policies: **triangular** (linear up half-cycle, linear down half-cycle), **triangular2** (max
halved each cycle → shrinking oscillations), **exp_range** (max decays exponentially → smoother
shrink). Reduces the need to tune LR precisely — the sweep naturally spends time at whatever rate
suits the current phase.

### OneCycleLR — super-convergence
A **single** cycle over the whole run: **Phase 1** (~30% of steps) aggressive warmup `LR_min → LR_max`;
**Phase 2** anneal `LR_max → LR_min` (often below the start). The aggressive-warmup high rate acts as a
**regularizer** — it prevents settling into sharp minima and pushes toward **flatter** regions that
generalize better. Per-**batch** scheduler.

### ReduceLROnPlateau — adaptive
Monitors a **metric** (usually val loss); if it doesn't improve for `patience` epochs, multiply LR by
`γ` (often 0.1). Data-driven — reacts to actual dynamics rather than a fixed curve. **Requires the
metric passed to `step(metric)`.**

---

## Warmup

Start at a **very small** LR and ramp to the target over `W` steps, then hand off to the decay schedule
(cosine/linear/constant). **Critical for transformers and large models.**

### Why transformers need warmup
- **Adam second-moment init.** Adam's second-moment estimate starts at 0; bias correction compensates,
  but the corrected estimate is **unreliable in the first few steps** → excessively large/erratic
  updates.
- **Attention softmax saturation.** Self-attention softmaxes over Q·K dot products. Early on these are
  random; a large LR can push softmax into **saturation** (nearly all weight on one position) →
  **sparse gradients** that destabilize training.
- **Warmup** lets the model calibrate attention with small, stable updates before ramping to full rate.

---

## Choosing / tuning the rate

### LR Finder (range test)
Increase LR **exponentially** over ~one epoch: `LR_i = LR_start · (LR_end/LR_start)^(i/N)` (from ~1e-7).
Plot **loss vs LR (log x-axis)** → three regions: **too low** (loss ~flat) → **good** (loss drops
fast) → **too high** (loss explodes). Set `LR_max` around the **steepest downward slope**, before the
loss turns up.

### Batch size ↔ LR (linear scaling rule)
Increase batch by factor `k` → scale LR by ≈`k` (**SGD**; Goyal et al., pair with warmup at large
batch). For **Adam**, the common heuristic is **√k** scaling instead, since Adam already normalizes by
the gradient's second moment.

### Does Adam need scheduling? (yes)
Adam keeps per-parameter first/second moment averages; the denominator gives **per-parameter** adaptive
rates (big-gradient params get smaller effective steps). But the LR is still a **master multiplier** on
all updates — scheduling it still reduces overall update noise and enables tighter convergence.

---

## PyTorch mechanics & common mistakes

- **Chaining (warmup + decay):** use `SequentialLR` to switch from a warmup scheduler to a decay
  scheduler at a milestone step. (`ChainedScheduler` composes multiplicatively.)
- **Wrong calling frequency.** `StepLR`/`ExponentialLR`/`CosineAnnealingLR` are **per-epoch**;
  `OneCycleLR`/`CyclicLR` are **per-batch**. Calling a per-epoch scheduler every batch **compresses the
  whole schedule into one epoch**.
- **Wrong ordering.** `optimizer.step()` **must precede** `scheduler.step()` each iteration — PyTorch
  warns if you reverse them (the params would update against a stale rate).
- **ReduceLROnPlateau needs the metric.** It's the odd one out: `scheduler.step(val_loss)`. Calling it
  bare errors / no-ops.
- **Constructor's implicit step.** Building a scheduler sets the **initial** LR (counter starts at 0).
  Calling `step()` again *before* the first epoch **skips the initial rate**.
- **Save scheduler state in checkpoints.** Save/load `scheduler.state_dict()` alongside model +
  optimizer, or on resume it **restarts from epoch 0** → wrong rates (see [[pytorch-nn-modules]] on
  `state_dict`).

---

## Practical guidelines

| Setting | Recommendation |
|---|---|
| **CNNs** | Cosine or StepLR with SGD+momentum; MultiStepLR milestones at 30/60/80% (γ=0.1) is a reliable baseline; OneCycleLR matches accuracy in fewer epochs if `LR_max` is well-chosen |
| **Transformers** | AdamW + **linear warmup (1–10% of steps)** → cosine or linear decay. Peak `1e-4`–`5e-4` pre-training; `1e-5`–`5e-5` fine-tuning |
| **Fine-tuning** | Small constant or gentle decay, **10–100× smaller** than original training LR; cosine + short warmup; sometimes **discriminative** rates (smaller for early layers) |
| **Default** | **Cosine annealing** — safest general choice (few hyperparameters, smooth, robust); add short warmup (5–10%) for a strong baseline |

**Always log the LR** (per param group, from `optimizer.param_groups`) alongside the loss — a loss
spike often means the rate is too high; a plateau may mean it should drop further ([[training-diagnostics]]).

---

Related: [[learning-rate]], [[optimization]], [[training-diagnostics]], [[pytorch-training-loop]],
[[normalization]], [[self-attention]]
