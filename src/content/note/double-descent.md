---
title: "Double Descent"
description: "test error down-up-down past interpolation, three regimes (under/critical/over-parameterized), why second descent (smooth interpolation = inductive bias), curse-of-dimensionality connection"
category: "Generalization & Model Fitting"
order: 26
updatedDate: "2026-09-10T21:27:14.732Z"
---
The classical bias-variance story says test error is U-shaped in model capacity. **Double descent**
(observed in deep nets) shows test error can go **down → up → down again** as capacity increases past
the point of interpolating the training data.

## The three regimes

Plotting test loss vs **capacity** (parameters):
1. **Classical / under-parameterized** — the usual U-curve; more capacity helps until it starts
   overfitting.
2. **Critical regime** — capacity is *just enough* to memorize the data; test performance is
   **temporarily worst** here.
3. **Modern / over-parameterized** — capacity **exceeds** what's needed to fit the data; test
   performance **improves again** and can surpass the classical sweet spot.

## Why the second descent happens

Once capacity drives **training loss to ~0**, the model fits every training point — extra capacity can't
change behavior *at* the data points, only **between** them. Which interpolating function the model
prefers is its **inductive bias**. As capacity grows, the model interpolates **more smoothly** between
points, and — absent information about what happens between training points — **smoothness is a sensible
prior** that generalizes well.

What encourages smoothness (hypotheses): network **initialization** may bias toward smooth functions,
and the **training algorithm (SGD)** may prefer converging to smooth solutions.

## Connections

- **Curse of dimensionality** — high-D volume overwhelms the number of training points; densely sampling
  the space needs **exponentially** more data ([[pca-svd]]), which is *why* smooth interpolation between
  sparse points matters.
- Contrast with the classical **bias-variance tradeoff** ([[overfitting-underfitting]]): double descent
  is why "bigger models overfit" is **not** the whole story for deep nets.

---

Related: [[overfitting-underfitting]], [[regularization]], [[deep-learning-theory]], [[pca-svd]]
