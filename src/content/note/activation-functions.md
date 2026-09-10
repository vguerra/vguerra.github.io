---
title: "Activation Functions"
description: "why non-linearity, ReLU/LeakyReLU/PReLU (dying ReLU), sigmoid/tanh (derivatives 0.25/1, zero-centering, `tanh=2σ(2x)−1`), GELU (`x·Φ(x)`, GPT/BERT), Swish/SiLU (self-gated, GELU≈SiLU cousins), SwiGLU/GeGLU gated FFN (LLaMA), practice table, interview Q&A"
category: "Misc ML Concepts"
order: 84
updatedDate: "2026-09-06T13:49:27.491Z"
---
Activations inject **non-linearity**. Without them, stacking linear layers **collapses to a single
linear transformation** regardless of depth (`W₂(W₁x) = (W₂W₁)x`). Non-linearity is what lets a network
approximate any continuous function (**universal approximation theorem**).

---

## ReLU and variants

**ReLU** — the default for most (non-transformer) nets: `f(x) = max(0, x)`.
- Computationally cheap; **doesn't saturate for positive inputs** → avoids vanishing gradients on the
  positive side (derivative exactly 1).
- **Not differentiable at 0** (subgradient 0 or 1 by convention).
- **Dying ReLU:** a neuron whose pre-activation is **always negative** (bad bias/init, or too-high LR)
  outputs 0 → gradient 0 → **never recovers** (self-locking). See [[training-diagnostics]].

**LeakyReLU** — small negative-side slope so gradient is never exactly 0:
`f(x) = x if x>0 else αx` (α≈0.01). **PReLU** makes α **learnable**. Both cure dying neurons.

---

## Sigmoid and tanh

**Sigmoid** `σ(x) = 1/(1+e^{-x})` → squashes to `(0,1)`. Derivative `σ(1−σ)`.
- **Max derivative 0.25** (at x=0) → each layer shrinks the gradient ≥4× → **vanishing gradients** in
  deep nets.
- **Outputs all-positive** (not zero-centered) → same-sign gradients → **zig-zag** updates.
- Today used only in **output layers** (binary classification → probability) and **gates** (LSTM/GRU,
  see [[rnn]]).

**Tanh** `(e^x − e^{-x})/(e^x + e^{-x})` → squashes to `(−1,1)`. Derivative `1 − tanh²`.
- **Zero-centered** → avoids sigmoid's zig-zag.
- **Max derivative 1** (at x=0) → better than sigmoid's 0.25, but **still saturates** at extremes →
  vanishing remains in very deep nets.
- **`tanh(x) = 2σ(2x) − 1`** — a rescaled, zero-centered sigmoid.

---

## GELU — Gaussian Error Linear Unit

Default in **GPT and BERT**. Gates the input by its **percentile under a Gaussian**:

$$\text{GELU}(x) = x\,\Phi(x)$$

(Φ = standard normal CDF.) Common **tanh approximation**:

$$\text{GELU}(x) \approx 0.5\,x\left(1 + \tanh\!\left[\sqrt{2/\pi}\,(x + 0.044715\,x^3)\right]\right)$$

- **Smooth everywhere** (differentiable at 0, unlike ReLU's hard corner).
- Large `+x` → approaches identity; large `−x` → approaches 0; small negatives **pass through slightly**
  (preserves a bit more information than ReLU's hard cutoff).

---

## Swish / SiLU

**Swish** `f(x) = x·σ(βx)`; derivative (β=1) `σ(x) + x·σ(x)(1−σ(x))`.
- **SiLU** is Swish with **β=1 fixed** (`x·σ(x)`); plain "Swish" has a **learnable β**.
- **Non-monotonic** — small dip below zero near `x ≈ −1.28`. **Self-gated** (σ acts as a soft gate on
  the linear input). **Bounded below, unbounded above.**
- Used in **EfficientNet** (Swish) and **LLaMA** (SiLU, inside SwiGLU).

**GELU ≈ SiLU (close cousins):** `GELU = x·Φ(x)`, `SiLU = x·σ(x)`, and the logistic **σ approximates the
Gaussian CDF Φ** → nearly identical curves. Both are **smooth, self-gated** ("multiply input by a soft
gate of itself") — the unifying idea of the GELU/Swish/SiLU family, vs ReLU's hard 0/1 gate.

---

## Gated FFN variants (modern LLMs)

LLaMA / PaLM / Mistral use **GLU variants** — **SwiGLU** (SiLU-gated), **GeGLU** (GELU-gated) — in the
feed-forward block: instead of `act(xW)`, compute **`act(xW) ⊙ (xV)`** with *two* input projections
(one activation branch, one learned gate). This is why LLaMA's FFN has **three** weight matrices, not
two. Common interview check: *"what activation does LLaMA use?"* → **SwiGLU** (not plain GELU).

---

## In practice

| Setting | Choice |
|---|---|
| Hidden layers (general) | **ReLU** default; **LeakyReLU/PReLU** if dying neurons |
| Transformers | **GELU** (GPT/BERT), **SwiGLU** (LLaMA/modern LLMs) |
| Output — binary | **Sigmoid** (probability) |
| Output — multi-class | **Softmax** (see [[loss-functions]]) |
| Gates | **Sigmoid** (LSTM/GRU gates) |
| Residual nets | ReLU or GELU (skips mitigate vanishing regardless — [[normalization]]) |

---

## Interview Q&A

- **Why not sigmoid everywhere?** Saturates at both extremes → near-zero gradients → vanishing in deep
  nets; not zero-centered → inefficient zig-zag updates. ReLU avoids positive-side saturation and is
  cheaper.
- **Dying ReLU & fix?** Always-negative pre-activation → 0 output, 0 gradient, never updates. Fixes:
  LeakyReLU, PReLU (learnable slope), He/Kaiming init ([[pytorch-nn-modules]]), lower LR.
- **Why GELU in transformers?** Smooth, probabilistic self-gating (vs ReLU's hard cutoff) empirically
  improves training; lets small negatives through, preserving information.
- **Why self-referencing derivatives (σ'=σ(1−σ), tanh'=1−tanh²)?** *Practical* reason: **reuse the
  forward value in backward** → skip recomputing `exp`. (They also happen to satisfy those ODEs, but the
  reuse is why it matters.)
- **Sigmoid ↔ tanh?** `tanh(x) = 2σ(2x) − 1` — tanh is a rescaled, zero-centered sigmoid; historically
  preferred in hidden layers for its zero-centered (balanced-gradient) outputs.

---

Related: [[normalization]], [[pytorch-nn-modules]], [[training-diagnostics]], [[rnn]],
[[loss-functions]], [[transformer-architecture]]
