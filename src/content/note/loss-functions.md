---
title: "Loss Functions"
description: "MSE (Gaussian MLE, outlier-sensitivity), cross-entropy from logits (softmax → −log p_true, log-sum-exp stability, `torch.max` placement), Huber (δ, smooth L1), loss = MLE-under-noise unifying frame"
category: "Misc ML Concepts"
order: 48
updatedDate: "2026-08-11T21:29:58.604Z"
---
Quick map: **MSE** for numeric predictions, **cross-entropy** for choosing among classes,
**Huber** for numeric predictions that may contain large errors/outliers.

---

## MSE (Mean Squared Error)

Regression loss. Squaring makes every error a **non-negative** contribution (errors can't cancel)
and penalizes **large misses strongly**.

$$\mathcal{L} = \frac{1}{N}\sum_i (\hat{y}_i - y_i)^2$$

- **Outlier-sensitive:** one extreme prediction can dominate the mean even when everything else is
  close — the squared term amplifies it.
- **Gradient is linear in the error:** `∂/∂ŷ · ½(ŷ−y)² = (ŷ − y)` → bigger errors push
  proportionally harder.
- **Why squared and not, say, 4th power?** MSE is the **maximum-likelihood estimator under Gaussian
  noise**: assume `y = true + N(0, σ²)`, maximize likelihood → MSE falls out. That's the principled
  reason it's the regression default.

---

## Cross-Entropy (from logits)

Used when the model must pick **one class from several**. Each sample gives a **row of logits** —
unrestricted scores, **not probabilities**.

**What it computes:** softmax turns the logit row into a distribution, then the loss is the
**negative log-probability of the true class**:

$$\mathcal{L} = -\log p_{\text{true}}, \qquad p_j = \frac{e^{z_j}}{\sum_k e^{z_k}}$$

Minimizing CE = **maximizing the likelihood** of the correct label. `p_true` depends on **every**
logit through the softmax denominator — which is why you need the *whole* logit row, not just the
max.

### Why "from logits" — the numerical-stability point

`F.cross_entropy` / `nn.CrossEntropyLoss` take **raw logits** and fuse `log_softmax + NLL`
internally via the **log-sum-exp trick**:

$$\log\sum_j e^{z_j} = z_{\max} + \log\sum_j e^{z_j - z_{\max}}$$

Subtracting the max logit first means the largest exponent is `e^0 = 1` → **no overflow**; and
fusing avoids ever materializing raw probabilities (so no `log(0) = −∞`). Doing it manually
(`softmax` → `log` → NLL) risks `exp` overflow and `log(0)` underflow. Same stability principle as
the stable sigmoid and the mean/variance cancellation issue (see [[numpy-basics]], [[perplexity]]).

**General rule:** avoid taking `exp`/`log` of extreme intermediate values — shift by the max (or fuse
the ops) so intermediates stay `O(1)`.

**Gotcha:** **never apply softmax yourself and then pass it to `CrossEntropyLoss`** — you'd
double-count the softmax. Pass **raw logits**.

### Where `torch.max` legitimately appears (and where it must NOT)

- **Loss value:** ❌ do *not* use `torch.max` — CE needs the full distribution, not the top class.
- **Manual stable softmax/CE:** ✅ `torch.max(logits, dim=-1, keepdim=True)` for the log-sum-exp
  shift (`keepdim` so it broadcasts back against the logits).
- **Accuracy/predictions:** ✅ `torch.max(logits, dim=1)` → `(values, indices)`; the **indices** are
  the predicted class (argmax over logits = argmax over softmax, monotonic). Computed *separately*
  from the loss.

---

## Huber Loss

A **compromise between squared error and absolute error**, controlled by a threshold **δ**:

$$\mathcal{L}_\delta(e) = \begin{cases} \tfrac{1}{2}e^2 & |e| \le \delta \\ \delta(|e| - \tfrac{1}{2}\delta) & |e| > \delta \end{cases}, \quad e = \hat{y} - y$$

- **Small errors → quadratic:** smooth curve near the correct value (well-behaved gradients).
- **Large errors → linear:** outliers don't dominate the way they do under MSE.
- **δ is the transition point;** Huber is **continuous and differentiable at δ** (value *and* slope
  match where the pieces meet) → optimization-friendly.
- **Smooth L1 loss** is a close cousin (≈ Huber with δ=1), used in object detection (Faster R-CNN
  box regression) for the same outlier robustness.

---

## Unifying frame: loss = MLE under an assumed noise model

Each common loss is the **maximum-likelihood estimator** for a different assumed noise/label
distribution — a great way to answer "why this loss?":

| Loss | Assumed model | Note |
|---|---|---|
| **MSE** | Gaussian noise | outlier-sensitive (light tails) |
| **MAE (L1)** | Laplacian noise | heavier tails → more outlier-robust |
| **Cross-entropy** | categorical likelihood | classification |
| **Huber** | Gaussian core + Laplacian tails | robust in-between |

Related: [[perplexity]], [[numpy-basics]], [[regression-metrics]], [[pytorch-basics]]
