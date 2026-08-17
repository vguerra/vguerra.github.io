---
title: "Regularization: L2 vs Dropout vs Early Stopping"
description: "L2 vs dropout vs early stopping mechanics, L1 vs L2, inverted dropout, early-stopping↔L2, other approaches, reg term & validation loss"
category: "Generalization & Model Fitting"
order: 15
updatedDate: "2026-07-14T14:14:59.119Z"
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

Related: [[overfitting-underfitting]]
