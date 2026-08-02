---
title: "Diagnosing Overfitting vs Underfitting"
description: "diagnosing via train/val curves, bias-variance, fixes, val<train edge case, quick triage"
category: "Generalization & Model Fitting"
order: 12
updatedDate: "2026-07-14T09:46:31.172Z"
---
## Reading Train/Val Curves

The **train loss**, **val loss**, and the **gap** between them tell you which regime you're in.

| Regime | Train loss | Val loss | Gap | Where it plateaus |
|---|---|---|---|---|
| **Overfitting** | keeps ↓ toward 0 | ↓ then rises / diverges | grows wider over time | train low, val high |
| **Underfitting** | ↓ then plateaus | ↓ then plateaus | small, roughly constant | both **high** |
| **Good fit** | ↓ to low value | ↓ to low value | small, constant | both **low** |

### Overfitting
- Train loss decreases toward 0, val loss diverges upward at some point.
- The **gap widens** the longer you train.
- Model memorizes training data instead of learning generalizable signal.

### Underfitting
- Both train and val loss decrease but **plateau at a high value**.
- The gap stays **small and constant** — the model isn't even fitting the training data.
- Train loss itself being stuck high means the model lacks capacity to capture the signal.

**Key distinction:** underfitting and a *good fit* both show a small constant gap. The difference is **where they plateau** — high loss = underfitting, low loss = healthy.

---

## The Bias-Variance Framing

These two regimes are the **bias-variance tradeoff** in action:

- **Underfitting = high bias** — the model is too simple to capture the signal; it makes systematic errors regardless of the data it sees.
- **Overfitting = high variance** — the model is too sensitive to the specific training set; small changes in data produce large changes in the fit.

Naming this in an interview signals you understand the *underlying theory*, not just the symptoms on the curve. The goal of training is to balance the two — enough capacity to have low bias, enough constraint to keep variance in check.

---

## Actionable Fixes

**Overfitting** (low train, high val — reduce capacity / add constraints):
- Collect more / higher-quality data
- Regularization (L2, dropout)
- Data augmentation
- Reduce model size
- Early stopping

**Underfitting** (both high — increase capacity / train more):
- Increase model capacity (wider / deeper network, more parameters)
- Train longer (more epochs)
- Better / more expressive features
- Tune learning rate (too low → slow; too high → won't converge)
- Reduce regularization

---

## Edge Case: Val Loss Lower Than Train Loss

If val loss is *below* train loss, it's usually **not** a real "better than training" signal:
- **Dropout / augmentation** are active during training but off during validation → train is measured on a harder, noised version of the data.
- Train loss is often averaged *over* the epoch (while the model is still improving), while val is measured *after* the epoch.
- Val set may simply be easier / smaller.

---

## Quick Triage: Fixing Overfitting on a Real Model Fast

Order the fixes by **effort vs. impact** — and diagnose before you act.

**0. Diagnose first (before touching anything).**
Confirm it's actually overfitting — check the train/val gap. Rule out a **broken validation split** or **data leakage**, which can mimic or mask overfitting. Signals you don't blindly throw regularization at an undiagnosed problem.

**1. Early stopping** — cheapest. You're already monitoring val loss; just halt when it plateaus. No model changes.

**2. L2 penalty** — cheap. Add `λΣw²` to the loss; no architecture change, just a term.

**3. More / better-quality data** — highest leverage, but costly. When collecting or labeling is too expensive, use **data augmentation** on existing data as the cheaper proxy.

**4. Dropout** — requires modifying the model architecture, so it comes after the free/cheap options.

**5. Reduce model capacity** — last resort: fewer layers / less depth / narrower, depending on architecture. Structural change, so try it only after the above.

See [[regularization]] for the mechanics of each technique.

Related: [[normalization]], [[optimization]], [[regularization]]
