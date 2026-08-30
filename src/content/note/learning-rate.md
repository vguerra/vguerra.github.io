---
title: "Learning Rate: Effect on Convergence & How to Tune It"
description: "LR effect on convergence/stability, tuning (log-scale, loss-curve, LR range test); SGD vs SGD+momentum vs Adam update rules + when-preferred; batch-size-1 SGD; warmup + decay schedules"
category: "Training Dynamics & Optimization"
order: 22
updatedDate: "2026-07-18T14:04:37.187Z"
---
## How LR Affects Convergence / Stability

**Too high:**
- Early on you approach the minimum fast, but the model won't converge — steps **overshoot** the minimum, causing **oscillation** in the training loss → unstable.
- Two sub-regimes:
  - **Way too high** → loss **diverges**, produces **NaN**.
  - **Slightly too high** → loss **oscillates around** the minimum (converges to a noisy region, never settles).

**Too low:**
- Converges **very slowly** → needs many more passes over the data → wastes compute/resources.
- Upside: training is **very stable**, no oscillation.

The core tradeoff: **speed & instability (high) vs. slow but stable (low).** A too-low LR can even *mimic underfitting* (train loss stuck high) despite adequate model capacity — see [[overfitting-underfitting]].

---

## How to Tune

### 1. Log-scale search (baseline)
LR sensitivity is **multiplicative**, so search over **powers of 10**, not a linear grid:
```
1e-1, 1e-2, 1e-3, 1e-4, 1e-5   ✓
0.1, 0.2, 0.3, 0.4             ✗ (wasteful)
```
Coarse first, then refine around the best (3e-3, 1e-3, 3e-4). Typical sweet spots: **Adam ~1e-3–3e-4**, **SGD ~1e-1–1e-2**.

### 2. Read the loss curve
| Symptom | Diagnosis | Action |
|---|---|---|
| Diverges / NaN / spikes up | LR too high | decrease by an order of magnitude |
| Decreases very slowly | LR too low | increase |
| Oscillates | slightly high | decrease a bit |
| Smooth decrease then plateaus | roughly right | add a schedule |

### 3. LR range test (Leslie Smith)
Find a good LR in a single short run:
- Start with a tiny LR, **increase it exponentially every batch** over ~1 epoch.
- Plot **loss vs. LR** (log x-axis). Loss falls, hits a minimum, then rises.
- Pick an LR **~1 order of magnitude below** where the loss starts rising (steepest-descent region, not the minimum itself).

Cheaper and more principled than a full grid search.

---

## Use a Schedule, Not a Constant

Once you have a good *peak* LR, don't hold it fixed. The standard modern schedule is
**linear warmup → cosine decay**.

### Warmup — start small, ramp up (three reasons)

1. **Adaptive optimizers' variance estimates are unreliable early.** Adam divides by `√v̂` (RMS of
   recent gradients); early on `v` has seen few batches → **high-variance estimate** → dividing by
   it produces erratic/huge steps. Warmup keeps LR tiny until `v` has accumulated enough samples.
2. **The model is fragile at initialization** (applies even to plain SGD). Near a random init the
   loss surface can be sharp → early gradients are **large and noisy**; a full LR can throw weights
   into an unrecoverable region (or diverge to NaN). Small early steps let the model settle into a
   well-behaved region before big steps. *This is why warmup helps non-adaptive optimizers too — it's
   not only about Adam's statistics.*
3. **Large batches / high peak LRs** (transformers, LLMs). Aggressive peak LRs (often with the
   linear-scaling rule) are only survivable if you **ramp into** them rather than start there.

**Mechanics:** LR rises from ~0 to the peak **linearly** over the first N steps (hundreds to
thousands), then hands off to decay.

### Decay — reduce LR later (big steps early, small steps late)

- **Early:** far from any minimum → want **large steps** for fast progress across the landscape.
- **Late:** near a minimum → a large step **overshoots** and makes the loss bounce (the too-high-LR
  oscillation) → want **small steps** to settle precisely into the basin.

A *fixed* LR forces a bad compromise (big enough for early progress = too big to converge cleanly).
Decay resolves it: **high when you need speed, low when you need precision** — like annealing.

**Common shapes:**
- **Cosine decay** — smooth cosine curve from peak → ~0. Most popular today.
- **Linear decay** — straight line from peak → 0.
- **Step decay** — drop by a factor (e.g. ÷10) at fixed milestones. Classic older-CV recipe.
- **ReduceLROnPlateau** — *reactive*: drop LR when val loss stops improving (not a fixed schedule).

**Canonical transformer / LLM recipe: linear warmup + cosine decay.**

---

## Related Notes

- **Adaptive optimizers (Adam/AdamW)** are *less* sensitive to base LR than plain SGD, but still need a tuned base LR + warmup/decay. "Adam means no LR tuning" is a misconception.
- **Batch size coupling** — linear scaling rule: increase batch size by k → scale LR by ~k (with warmup to stabilize).

**Practical workflow:** read the loss curve → LR range test for a starting point → pick peak LR → add warmup + cosine decay → fine-tune.

---

## SGD vs SGD+Momentum vs Adam — Update Rules

### Plain SGD
Uses a **randomly-sampled mini-batch** (without replacement) to estimate the gradient, so each
step is cheap (no full pass) → more updates per epoch, faster progress, but a **noisier / less
smooth trajectory**. The stochasticity also helps escape sharp local minima. It doesn't truly
converge but hovers near the minimum.

$$\theta \leftarrow \theta - \alpha\, g_t \qquad (g_t = \text{grad on current batch})$$

LR is fixed; the **same** step scale applies to every parameter regardless of its gradient
magnitude — the core limitation adaptive methods fix.

#### SGD with batch size 1 (online / true SGD)

**Pros:**
- Each update is **O(d)** rather than **O(nd)** (d = params, n = dataset size) — no full-dataset pass per step.
- Can handle **streaming data** (process one example as it arrives).
- Noisy updates can **help generalization** (escape sharp minima).

**Cons:**
- Updates have **high variance**, so the LR often needs to be **small or decayed**.
- The loss **does not decrease monotonically** (bounces step to step).

Mini-batches are the practical middle ground: lower-variance gradient estimates than batch-1, while
still far cheaper than full-batch GD.

### SGD + Momentum (the bridge)
Dampens oscillation and smooths the path by blending the current gradient with the previous
step's direction (an EMA of gradients). Effective step **grows** when gradients stay aligned across
iterations, **shrinks** when the direction keeps flipping — accelerates through ravines.

$$m_{t+1} = \beta\, m_t + (1-\beta)\, g_t$$
$$\theta \leftarrow \theta - \alpha\, m_{t+1}$$

### Adam
Even with momentum, the loss can be far steeper in some directions than others, making a single LR
hard to pick. Adam gives each parameter its **own adaptive step size** by dividing by the RMS of
recent gradients (rescaling per-parameter). It tracks **two** EMAs, each with its own decay
constant, and **bias-corrects** both (both start at 0 → biased toward 0 early):

$$m_{t+1} = \beta_1 m_t + (1-\beta_1) g_t \qquad v_{t+1} = \beta_2 v_t + (1-\beta_2) g_t^2$$
$$\hat{m} = \frac{m_{t+1}}{1 - \beta_1^{\,t+1}} \qquad \hat{v} = \frac{v_{t+1}}{1 - \beta_2^{\,t+1}}$$
$$\theta \leftarrow \theta - \alpha\, \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}$$

- **β₁ ≈ 0.9** (first moment / momentum), **β₂ ≈ 0.999** (second moment / squared-grad, slower memory).
- Each β is used in **both** its own EMA and its own bias correction.
- **eps outside the sqrt** (PyTorch default), guards divide-by-zero early when `v̂ ≈ 0`.

### When to prefer each

| Axis | Winner | Why |
|---|---|---|
| **Generalization** | **SGD + momentum** | Often generalizes better, esp. in CV/CNNs (the "generalization gap"); SOTA vision with a good schedule. |
| **Ease / speed** | **Adam** | Converges faster with less LR tuning; default for transformers/NLP, sparse gradients, rapid prototyping. |
| **Memory** | **SGD** | Adam stores 2 extra buffers (m, v) per param → ~2–3× optimizer memory. Matters at billion-param scale. |

**Note:** "Adam for deep networks" is a misconception — plenty of very deep nets (ResNets, most
ImageNet training) use SGD+momentum. Depth isn't the deciding axis; the three above are.

Related: [[optimization]], [[overfitting-underfitting]]
