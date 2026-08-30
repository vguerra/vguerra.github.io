---
title: "Momentum & Nesterov"
description: "why momentum (narrow-valley oscillation / ill-conditioning, SGD noise cancels while signal accumulates); velocity EMA `v=μv+g`, steady-state `g/(1−μ)` (10× at μ=0.9), `1/(1−μ)` window, noise reduction; dampening τ; Nesterov lookahead (conceptual + PyTorch reformulated forms, anticipatory braking); Adam connection"
category: "Training Dynamics & Optimization"
order: 21
updatedDate: "2026-08-27T20:21:47.036Z"
---
Vanilla gradient descent is **memoryless** — each step uses only the current gradient — which causes
two pathologies momentum fixes.

## Why momentum exists

1. **Narrow-valley oscillation (ill-conditioning).** With high curvature in one direction and low in
   another, GD **bounces across the steep walls** while crawling along the shallow floor. Raising the
   LR to speed the floor progress **amplifies** the oscillation; lowering it damps oscillation but
   slows convergence. This tension between directions of different curvature (the **condition-number**
   problem, see [[optimization]]) is what momentum resolves.
2. **Mini-batch noise (SGD).** Each mini-batch gradient is a noisy estimate of the true gradient. Key
   insight: the **noise is ~zero-mean and cancels** over time, while the **consistent descent
   direction accumulates**. Momentum exploits exactly this.

Result: **faster progress along the consistent direction, damped oscillation in inconsistent ones.**

---

## The mechanics

Momentum keeps a **velocity** `v_t` = exponential moving average of gradients (PyTorch convention,
dampening τ=0):

$$v_t = \mu\, v_{t-1} + g_t \qquad \theta_t = \theta_{t-1} - lr\cdot v_t$$

Unrolling the recurrence → a decaying weighted sum of all past gradients:

$$v_t = g_t + \mu g_{t-1} + \mu^2 g_{t-2} + \dots + \mu^{t-1} g_1$$

- **Coefficient `μ`** = fraction of previous velocity retained. `μ=0` → vanilla SGD; `μ→1` → velocity
  changes more slowly. `μ=0.9` retains 90% of the previous velocity.
- **Steady-state acceleration.** If the gradient is constant `g` for many steps,
  `v → g·(1 + μ + μ² + …) = g/(1−μ)`. At `μ=0.9` the effective step is **10× the raw gradient step** —
  momentum moves fast through consistent-gradient regions **without raising the LR** (which would
  destabilize the oscillatory directions).
- **Effective window** ≈ `1/(1−μ)` steps: `μ=0.9` → last ~10 gradients dominate; `μ=0.99` → ~100.
- **Noise reduction.** For the *normalized* EMA (`v = μv + (1−μ)g`) the stationary noise variance shrinks
  by `(1−μ)/(1+μ)`. For PyTorch's *unnormalized* form the signal grows `1/(1−μ)` and the noise std
  `1/√(1−μ²)`, so the **SNR improves ~√((1+μ)/(1−μ))**. Either way: momentum denoises.

---

## Dampening (PyTorch `dampening=τ`)

$$v_t = \mu\, v_{t-1} + (1-\tau)\, g_t$$

`τ=0` → standard momentum. `τ>0` **reduces the current gradient's contribution**, weighting accumulated
history more → smoother trajectory. (Nesterov **requires `τ=0`**.)

---

## Nesterov momentum (lookahead)

Instead of the gradient at the **current** point, step in the velocity direction **first**, then take
the gradient at that **lookahead** point.

**Conceptual form** — evaluate `∇f` at where momentum *would* carry you:

$$\theta_{\text{ahead}} = \theta_{t-1} - lr\cdot\mu\, v_{t-1}$$
$$v_t = \mu\, v_{t-1} + \nabla f(\theta_{\text{ahead}})$$
$$\theta_t = \theta_{t-1} - lr\cdot v_t$$

**PyTorch reformulated form** — algebraically rearranged so `∇f` is taken at the current params
(`g_t = ∇f(θ_{t-1})`):

```
b_t = μ·b_{t-1} + g_t                # velocity (dampening τ=0)
θ_t = θ_{t-1} − lr·(g_t + μ·b_t)     # step uses g_t + μ·b_t  (plain momentum would use b_t)
```

The extra **`μ·b_t`** term is the lookahead correction — plain momentum steps by `−lr·b_t`, Nesterov by
`−lr·(g_t + μ·b_t)`. The two forms are equivalent (a change of variables); PyTorch uses the second so it
never evaluates `∇f` anywhere but the current parameters.

**Why it helps — anticipatory braking.** Plain momentum only feels an overshoot *after* it has stepped
past the minimum (the old-position gradient still pointed forward → it kept accelerating). Nesterov
peeks at the lookahead point `θ − lr·μ·v`, which is already *past* the minimum, sees the gradient there
**pointing back**, and folds that corrective signal into the *same* update → it **brakes before
overshooting**, not after. Hence less oscillation and more responsiveness in curved regions.

**Flag:** `torch.optim.SGD(params, lr, momentum=0.9, nesterov=True)` — requires `momentum > 0` and
`dampening = 0`.

---

## Connection to Adam

Adam ≈ **momentum (1st moment)** + **RMSProp (2nd moment)** with bias correction — the momentum term
here is Adam's first-moment EMA. See [[learning-rate]] for the full SGD / SGD+momentum / Adam comparison.

---

Related: [[learning-rate]], [[optimization]], [[lr-schedulers]], [[pytorch-training-loop]],
[[training-diagnostics]]
