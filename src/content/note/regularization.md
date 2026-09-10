---
title: "Regularization: L2 vs Dropout vs Early Stopping"
description: "L2 vs dropout vs early stopping mechanics, L1 vs L2 (+ geometry & Elastic Net), L2-penalty vs decoupled weight decay (AdamW), inverted dropout, early-stopping↔L2, regularization-as-prior (MAP), implicit regularization (GD large-step / SGD stable-gradients), noise injection (input/weight/label smoothing), ensembling-as-regularization, reg term & validation loss"
category: "Generalization & Model Fitting"
order: 27
updatedDate: "2026-09-10T21:27:28.799Z"
---
Three common techniques to fight **overfitting**, each with a distinct mechanism.

---

## L2 Regularization (Weight Decay)

**Mechanism:** adds a penalty term to the loss — the sum of squared weights, scaled by λ:

```
L_total = L_data + λ · Σ w²
```

- The gradient of the penalty is `2λw` → **proportional to the weight itself**.
- Each gradient step shrinks a weight by an amount proportional to its current magnitude (compounds to roughly **exponential** decay over steps, not linear). Hence the name **weight decay**.
- Induces **shrinkage**: pushes weights toward 0 but **not exactly 0**.
- Smooth gradient everywhere.

**Effect / intuition:** controls model complexity by keeping weights small. Encourages the model to **use all features a little** rather than a few features a lot.

**Contrast with L1** (Σ|w|): L1 has a constant-magnitude gradient (`±λ`) that doesn't shrink as weights approach 0, so it drives many weights **exactly to 0** → **sparsity / feature selection**. L2 → shrinkage; L1 → sparsity.

### L2 penalty vs (decoupled) weight decay — *not* the same with Adam

"L2 regularization" and "weight decay" are treated as synonyms, but they only coincide for **plain
SGD**. They diverge once the optimizer is adaptive.

- **L2 penalty:** add `λ‖θ‖²` to the **loss** → the gradient gains a `+λθ` term → that term flows
  **through the optimizer's machinery** (momentum EMA, and crucially Adam's `1/√v` per-parameter
  scaling).
- **Decoupled weight decay:** shrink the weights **directly, as a separate step** —
  `θ ← (1 − lr·λ)θ` — *outside* the gradient update.

**Plain SGD → identical.** L2 gives `θ ← θ − lr(g + λθ) = (1 − lr·λ)θ − lr·g`, which *is* decoupled
weight decay. So the folklore holds here.

**Adam → substantially different (the AdamW point, Loshchilov & Hutter).** With L2, the `λθ` term rides
through the `1/√v` division, so the **effective decay becomes `∝ λθ/√v`** — inversely coupled to each
parameter's gradient history. Backwards from what you want: a large, actively-used weight (big `√v`)
gets its decay **divided down → barely shrinks**, while low-gradient weights get *more* decay.
Regularization strength becomes an artifact of optimization dynamics, not a clean prior.

**Why decoupling (AdamW) fixes it:** applying `θ ← (1 − lr·λ)θ` as a **separate step never touches
`√v`**, restoring a **uniform relative shrink** on every parameter (the intended prior) — the same
relative decay *regardless of gradient history*. It disentangles two jobs L2-in-Adam accidentally
fused: **`1/√v` adapts the loss-gradient step** (keep — helps optimization) while **weight decay
shrinks uniformly** (helps generalization). Bonus: `λ` becomes far more stable across architectures / LR
schedules. This is why **AdamW is the default for modern transformer training**, and why the "L2 =
weight decay" shorthand is silently wrong for Adam. (Momentum alone causes only a *mild* version of this
— the `λθ` term entering the velocity EMA; the `√v` scaling is the dominant effect. See [[momentum]],
[[learning-rate]].)

### L1 vs L2 geometry, and Elastic Net

**Why L1 → sparsity, L2 → shrinkage:** L1's `|w|` has a **kink at 0** (constant-magnitude gradient `±λ`
that doesn't shrink as `w→0`) → strong pull to **exact zeros**. L2's smooth gradient `2λw` shrinks
proportionally but **rarely reaches zero**. Geometrically, L1's **diamond** constraint region has
**corners** on the axes that the optimum tends to hit (→ zeros); L2's **circle** has no corners.

**Elastic Net** = both penalties: `L = L₀ + λ₁‖w‖₁ + λ₂‖w‖₂²` (or `λ(α‖w‖₁ + (1−α)‖w‖₂²)`). L1 gives
sparsity, L2 smooth small weights, `α` balances them. **Better than pure Lasso when features are
correlated** (Lasso arbitrarily picks one of a correlated group; Elastic Net keeps them). Cons: more
hyperparameters, noisier selection, slower.

> **Squared L2 vs L2 norm as a penalty:** use the **squared** L2 (`‖w‖²`) — it's smooth/differentiable
> everywhere (a polynomial), strictly convex, and its gradient is clean (`2λw`, or `λw` with the ½
> convention). The plain L2 norm has a `√` → not differentiable at 0 and costlier.

---

## Dropout

**Mechanism:** during training, randomly deactivate neurons; each forward pass uses a different random sub-network. This prevents **co-adaptation** — neurons can't rely on specific others always being present, so each learns more robust, independently useful features.

**Ensemble view:** training samples a different sub-network each step → approximates training an **ensemble** of many networks that share weights.

### Two formulations — mind the scaling

Dropout must keep the **expected activation** consistent between train and test. There are two equivalent ways:

| | Training | Test / Inference |
|---|---|---|
| **Classic dropout** | drop neurons | scale outputs by `p` (keep prob) |
| **Inverted dropout** (modern, PyTorch) | scale survivors by `1/p` (keep prob) | pass-through, no scaling |

- **Inverted dropout** is preferred in practice because inference needs **no special handling** — the network runs as-is. This is what `nn.Dropout` implements.
- Dropout is **only active during training** (`model.train()`), disabled at eval (`model.eval()`).

### Convention gotcha
- `p` sometimes means **keep** probability (classic literature) and sometimes **drop** probability.
- **PyTorch `nn.Dropout(p)`: `p` is the DROP probability.** So survivors are scaled by `1/(1-p)` during training.
- Always state which convention you mean — it flips the scaling factor.

**Note:** dropout does *not* permanently reduce capacity — the full network is used at test time. It subsamples a sub-network *per training step*.

See also [[pytorch-nn-modules]].

---

## Early Stopping

**Mechanism:** monitor validation loss during training; stop once it stops improving for a configurable number of steps (**patience**). Keep the checkpoint with the best val loss.

**Why it acts as a regularizer (connection to L2):**
- Weights start **small** (near-zero initialization).
- Gradient descent moves weights **away** from that small start, growing them to fit the data — the longer you train, the larger they get.
- Stopping early means weights **haven't traveled far from the origin** in weight space → they stay small.
- L2 *also* keeps weights small (via an explicit penalty). So both land you at **small-magnitude weights**, just by different routes.

**Formal correspondence** (for linear / quadratic loss): the **number of training steps** plays an *inverse* role to L2's λ:
- **Few steps** ≈ **large λ** (strong regularization, weights stay tiny)
- **Many steps** ≈ **small λ** (weak regularization, weights grow)

---

## Summary

| Technique | Mechanism | Keeps weights small? |
|---|---|---|
| **L2** | penalty term `λΣw²` in loss | yes — explicit shrinkage |
| **Dropout** | random sub-network per step; prevents co-adaptation | no — different mechanism (ensemble) |
| **Early stopping** | halt when val loss plateaus (patience) | yes — implicitly (limited travel from init) |

---

## Other Approaches to Overfitting

Regularization isn't the only option. Anything that **controls model complexity** is valid:

- **Remove uninformative features** — e.g. features with very low variance or low correlation with the target. Fewer inputs → less room to fit noise.
- **Stop early during iterative optimization** — a model that hasn't fully converged hasn't had time to memorize noise (see Early Stopping above).
- **Reduce dimensionality** — project features to a lower-dimensional space (e.g. PCA) before fitting, shrinking the effective parameter count.

The unifying principle: **any technique that limits the effective capacity of the model** can fight overfitting.

---

## Do You Include the Regularization Term in Validation Loss?

**No — report the raw (unregularized) data loss on validation.** The penalty is a
*training-time device*, not part of the metric you care about.

$$L_{train} = L_{data} + \lambda \Sigma w^2 \qquad L_{val} = L_{data}\ \text{only}$$

**Why exclude it:**
1. The penalty `λΣw²` depends on the **weights only**, not the data → it adds the *same constant*
   to any evaluation, telling you nothing about generalization.
2. Validation loss should answer *"how well does this predict unseen data?"* Adding the weight
   penalty conflates that with *"how big are the weights?"* — a training concern.

**Mental model:**
- **Regularization** shapes the *optimization* (biases training toward simpler solutions) — its job is done during gradient updates.
- **Validation loss** measures *generalization* — you want the pure data-fit term.

**Practical consequences:**
- For **early stopping / model selection**, monitor the **unregularized** val loss (or a task
  metric like accuracy / F1).
- Train loss (with penalty) and val loss (without) aren't directly comparable in *absolute* value,
  but the **gap trend** still diagnoses overfitting (see [[overfitting-underfitting]]).

**Related distinctions:**
- **Dropout is also off at validation** (`model.eval()`) — training-only too, but a *different*
  mechanism (stochastic forward pass, not an added loss term).
- **Decoupled weight decay (AdamW):** the penalty lives in the *optimizer step*, not the loss —
  so the computed loss never includes it on train *or* val.

---

## Explicit regularization as a prior (MAP)

Explicit regularization adds a term `λ·g(φ)` that's **larger for less-preferred parameters**:
`φ̂ = argmin Σ lᵢ + λ·g(φ)`. This term can be read as a **prior** `Pr(φ)` encoding beliefs about the
parameters *before* seeing data → minimizing the regularized loss is **maximum a posteriori (MAP)**
estimation (vs plain MLE for the unregularized loss). L2 ↔ a Gaussian prior on the weights.

## Implicit regularization (GD & SGD)

Regularization can arise from the **optimizer itself**, not an added term:
- **GD** — full-batch gradient descent generalizes better with **larger step sizes** (an implicit bias).
- **SGD** — implicitly favors regions where **gradients are stable** (all batches agree on the slope).
  This changes the *trajectory*, not the location of the global minimum. **SGD generalizes better than
  full-batch GD, and smaller batches often beat larger ones** — the noise lets it explore different
  parts of the loss surface. (The batch-size/LR ratio matters for generalization.)

## Adding noise as regularization

- **Input noise** — smooths the learned function.
- **Weight noise** — pushes toward **wide, flat minima** where individual weights don't matter much
  (robust).
- **Label noise / label smoothing** — train against a target where the true class has probability
  `1−p` and the rest share `p` equally. Stops the model becoming **over-confident** (over-confident
  models generalize worse, are poorly calibrated, and overfit noisy labels).

**Early stopping ≈ L2:** since weights start small, stopping early means they **don't have time to grow**
— similar effect to an explicit L2 penalty (already noted above).

## Ensembling as regularization

Average several models' predictions (mean of outputs for regression, mean of **pre-softmax** activations
for classification) — independent errors cancel. Get diversity from different **random inits**,
**bagging** (resample data with replacement), or different hyperparameters/model families ([[ensembles]]).

## Other approaches

Transfer learning, multi-task learning, data augmentation.

---

Related: [[overfitting-underfitting]], [[ensembles]], [[double-descent]], [[loss-functions]], [[learning-rate]]
