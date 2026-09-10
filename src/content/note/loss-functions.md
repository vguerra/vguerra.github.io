---
title: "Loss Functions"
description: "choosing-the-right-loss table (MSE/BCE/CCE/Hinge + output activations), MSE (Gaussian MLE, outlier-sensitivity), cross-entropy from logits (softmax → −log p_true, log-sum-exp stability, `torch.max` placement), CE↔KL divergence, clean gradient forms (`ŷ−y`, `p−1_true`) & exponential-family reason, why CE not MSE (sigmoid saturation), Hinge loss (SVM/margin vs calibrated probs), Huber (δ, smooth L1), **maximum-likelihood recipe** (choose distribution → predict its params → minimize NLL; Gaussian→MSE, Bernoulli→BCE, categorical→CE), heteroscedastic regression (predict μ & σ²), robust/quantile/focal/ranking losses, loss = MLE-under-noise unifying frame"
category: "Misc ML Concepts"
order: 81
updatedDate: "2026-09-10T21:26:12.257Z"
---
Quick map: **MSE** for numeric predictions, **cross-entropy** for choosing among classes,
**Huber** for numeric predictions that may contain large errors/outliers.

## Choosing the right loss

| Task | Loss | Output activation |
|---|---|---|
| Regression | **MSE** (or MAE) | None (linear) |
| Binary classification | **BCE** | Sigmoid |
| Multi-class classification | **CCE** (cross-entropy) | Softmax (or **fused** with loss) |
| Maximum-margin classifier | **Hinge** | None (linear) |

In practice frameworks **fuse** softmax + cross-entropy into one numerically stable op — PyTorch's
`CrossEntropyLoss` / `F.cross_entropy` take **raw logits**, not probabilities.

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

## Cross-entropy ↔ KL divergence

Cross-entropy decomposes as **the true label's entropy plus the KL divergence** to the prediction:

$$H(p, q) = -\sum p \log q = H(p) + D_{KL}(p \,\|\, q)$$

Since `H(p)` (the entropy of the *fixed* labels) is **constant** w.r.t. the model, **minimizing
cross-entropy ≡ minimizing `D_KL(p‖q)`** — you're pulling the predicted distribution `q` toward the
true distribution `p`. For a one-hot `p`, `H(p)=0`, so CE *equals* the KL divergence. This is the
information-theoretic reason CE is the natural classification loss.

---

## Clean gradient forms (and why)

All three "match to the output activation" losses have the **same tidy gradient** — prediction minus
target:

| Loss + activation | Gradient w.r.t. logits/pre-activation |
|---|---|
| **MSE** (linear) | `ŷ − y` |
| **BCE + sigmoid** | `ŷ − y` (same form as MSE!) |
| **CCE + softmax** | `p − 1_{true}` (softmax prob minus the one-hot target) |

**Why so clean — not a coincidence.** These pairings are the **canonical link functions** for
exponential-family distributions (Gaussian ↔ identity, Bernoulli ↔ sigmoid, categorical ↔ softmax).
Pairing cross-entropy with the matching activation makes the messy activation-derivative and
loss-derivative **cancel**, leaving `prediction − target`. Larger error → proportionally larger update,
with **no saturation term** to kill the gradient. (This is *why* you pair sigmoid with BCE and softmax
with CE — mismatched pairs, e.g. MSE-on-sigmoid, don't cancel and *do* saturate; see below.)

---

## Why cross-entropy, not MSE, for classification

Applying **MSE to sigmoid/softmax outputs** creates a **non-convex** loss surface with **slow gradients
when the model is confidently wrong**: the `σ'(z)` factor `→ 0` in the saturated region, so a
confidently-wrong prediction produces a **tiny** gradient → learning stalls. **Cross-entropy's gradient
(`p − target`) has no such saturation factor** — a confidently-wrong prediction produces a **large**
gradient → fast correction. That's the core reason CE is the classification default.

---

## Hinge Loss (max-margin)

$$\mathcal{L} = \max(0,\; 1 - y\cdot\hat{y}), \qquad y \in \{-1, +1\}$$

The loss of **SVMs**. It focuses on the **decision boundary**: once a point is correctly classified with
**enough margin** (`y·ŷ ≥ 1`), its loss is **exactly 0** and it's **ignored** — only boundary/violating
points contribute. Contrast with cross-entropy, which keeps pushing on *every* point (always a nonzero
gradient) to produce **calibrated probabilities**. So: **hinge** when you only care about the boundary /
the decision; **cross-entropy** when you need **confidence estimates / calibrated probabilities**.

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

## The maximum-likelihood recipe (constructing any loss)

The principled way to *derive* a loss (Prince, UDL). Treat the model as predicting a **probability
distribution** `Pr(y|θ)` over outputs, with the network computing its parameters `θ = f(x,φ)`:

1. **Choose a distribution** `Pr(y|θ)` over the output domain.
2. **Model predicts its parameters:** `θ = f(x,φ)`.
3. **Minimize negative log-likelihood** over the training set:
   `φ̂ = argmin −Σᵢ log Pr(yᵢ | f(xᵢ,φ))`.
4. **Inference:** return the full distribution, or its **argmax** (a point estimate `ŷ`).

Why NLL and not the raw likelihood: the product `∏ Pr(yᵢ|xᵢ)` (valid under **i.i.d.** data) underflows;
`log` is monotonic so `argmax ∏ = argmax Σ log = argmin −Σ log`. Each choice of distribution yields a
familiar loss:

| Distribution (→ output activation) | Model predicts | Loss that falls out |
|---|---|---|
| **Univariate Gaussian** (linear) | mean `μ` (fixed `σ²`) | **least squares / MSE** |
| **Gaussian, learned σ²** | `μ` and `σ²` | NLL with a variance term (**uncertainty estimate**) |
| **Bernoulli** (sigmoid) | `λ = σ(f)` | **binary cross-entropy** |
| **Categorical** (softmax) | `λₖ = softmax(f)` | **cross-entropy** |
| **Laplace** (linear) | median | **MAE** (robust regression) |

**Least squares from a Gaussian:** fixing `σ²`, dropping constants, `−log` of the Gaussian reduces to
`Σ(yᵢ − f(xᵢ,φ))²`. So the model predicts the **mean** `μ = f(x,φ)`, and `ŷ = argmax = μ = f(x,φ)`.

**Heteroscedastic regression:** if uncertainty **varies with x** (not constant/homoscedastic), predict
**two** outputs — `μ = f₁(x,φ)` and `σ² = f₂(x,φ)²` (squared to force positivity). The NLL now has both
a mean-error term (weighted by `1/σ²`) and a `log σ` term.

**Multiple outputs** — usually treat dimensions as **independent**: `Pr(y|f) = ∏_d Pr(y_d|f_d)`, so the
NLL becomes a **sum** over output dims.

**Cross-entropy = NLL = KL:** minimizing CE minimizes the **KL divergence** between the empirical data
distribution `q(y)` and the model's `Pr(y|θ)` — since the data's entropy is fixed (see the CE↔KL section
above).

## Beyond the basics

- **Robust regression** — Laplace assumption → **MAE**, estimates the **median** (outlier-robust).
- **Quantile regression** — predict a quantile (e.g. "true value < prediction 90% of the time") via
  **quantile / pinball loss**. Useful for risk models.
- **Focal loss** — for **class imbalance**: one extra parameter **down-weights well-classified**
  (easy/majority) examples so the model stops over-focusing on them ([[class-imbalance]]).
- **Learning to rank** — **pointwise** (score one item), **pairwise** (compare two), **listwise**
  (whole list at once, e.g. Plackett-Luce).

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
