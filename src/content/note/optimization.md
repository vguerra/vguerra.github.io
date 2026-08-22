---
title: "Optimization"
description: "Hessian eigenvalues, saddle points, condition number, adaptive optimizers, Newton's method, convergence, 2nd-order at LLM scale"
category: "Training Dynamics & Optimization"
order: 18
updatedDate: "2026-07-05T10:33:06.557Z"
---
## Convergence Stopping Criteria

Stop optimization when any of these are met:

| Criterion | Condition | Intuition |
|---|---|---|
| Gradient norm | `‖∇f(x)‖ < tol` | Near a critical point — gradient ≈ 0 at minimum |
| Step size | `‖xₖ₊₁ - xₖ‖ < tol` | Not moving anymore |
| Loss change | `\|f(xₖ₊₁) - f(xₖ)| < tol` | Loss not decreasing |
| Max iterations | `k > max_iter` | Safety fallback |

Gradient norm is the most principled — at a true minimum the gradient is exactly zero.

---

## Gradient Descent Update Rule

```
w ← w - η∇L(w)
```

- `η` = learning rate
- `∇L(w)` = gradient of loss w.r.t. weights

**Intuition:** the gradient points toward *steepest ascent* — subtracting it descends the loss landscape.

**Learning rate behavior:**
- Too large → overshoots, diverges
- Too small → converges but very slowly
- Just right → converges to a local minimum

**Convergence:** GD converges to a **local** minimum, not necessarily global. For convex functions (e.g. linear regression) local = global. For neural networks the landscape is non-convex — most critical points are saddle points.

Convergence is guaranteed when `η < 2/L` where `L` is the largest eigenvalue of the Hessian (Lipschitz constant of the gradient) — i.e., the learning rate must be small relative to the curvature.

---


## Why Second-Order Methods Are Rarely Used at LLM Scale

1. **Hessian size** — for `n` parameters, Hessian is `(n × n)`. GPT-3 has 175B params → `175B × 175B` matrix — physically impossible to store
2. **Inversion cost** — matrix inversion is O(n³) — catastrophically expensive even if storage were possible
3. **Mini-batch noise** — LLMs train with mini-batches; Hessian estimated on a mini-batch is a poor approximation, undermining second-order advantages
4. **Non-convex instability** — loss landscapes of deep nets are non-convex; second-order methods can diverge or converge to saddle points

**In practice:** Adam approximates second-order behavior cheaply using gradient moment estimates — no Hessian needed. This is why adaptive first-order methods dominate LLM training.

---

## Newton's Method

Makes a **quadratic approximation** of f around the current point:

```
f(x) ≈ f(xₖ) + ∇f(xₖ)ᵀ(x - xₖ) + ½(x - xₖ)ᵀ H(xₖ)(x - xₖ)
```

The **ᵀ** denotes **transpose**:
- `∇f(xₖ)ᵀ` — gradient is a column vector; transposed to row vector so `∇f(xₖ)ᵀ(x-xₖ)` is a scalar
- `(x-xₖ)ᵀ H(xₖ)(x-xₖ)` — **quadratic form**: transpose × matrix × vector → scalar

Both terms must be scalars since f(x) is scalar-valued.

Setting the gradient of the approximation to zero gives the Newton update:
```
xₖ₊₁ = xₖ - H(xₖ)⁻¹ ∇f(xₖ)
```

vs GD which uses `xₖ₊₁ = xₖ - η∇f(xₖ)` — Newton replaces the fixed learning rate with the inverse Hessian, encoding curvature information.

**How H⁻¹ rescales steps:**
- High curvature direction (large eigenvalue) → smaller step (avoid overshooting)
- Low curvature direction (small eigenvalue) → larger step (move faster)
- GD uses the same `η` in all directions → zigzags in elongated landscapes
- H⁻¹ effectively "rounds out" the landscape so steps go directly toward the minimum

---

## Hessian and Eigenvalues

The Hessian is the matrix of second-order partial derivatives. Its eigenvalues describe **curvature** at a critical point (where gradient = 0).

| Eigenvalues | Conclusion |
|---|---|
| All **positive** | Local **minimum** — curves up in all directions |
| All **negative** | Local **maximum** — curves down in all directions |
| **Mixed** (some +, some -) | **Saddle point** — neither min nor max |
| Any **zero** | Inconclusive — need higher-order analysis |

## Why It Matters for ML

- **True local minima are rare** in deep learning — most critical points are saddle points, especially in high dimensions
- The more parameters, the more likely eigenvalues are mixed
- **Gradient descent naturally escapes saddle points** over time due to noise

## Condition Number

Ratio of largest to smallest eigenvalue — measures how elongated the loss landscape is:

- **High condition number** → narrow valley → gradient descent oscillates, converges slowly
- This is why **adaptive optimizers** (Adam, RMSProp) outperform vanilla SGD — they effectively normalize curvature across dimensions
