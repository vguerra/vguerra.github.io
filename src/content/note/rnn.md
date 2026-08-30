---
title: "RNNs — Elman Cell, BPTT, and Gating"
description: "Elman cell (`h_t = tanh(W_ih x_t + W_hh h_{t-1} + b)`, shapes, why tanh), BPTT vanishing/exploding (two factors: `W_hh` spectral norm *and* `tanh'≤1`), weight sharing, how gating fixes it (additive cell state `c_t = f_t⊙c_{t-1}+i_t⊙g_t`, `∂c_t/∂c_{t-1}=f_t≈1` = gated residual highway / constant error carousel), LSTM vs GRU"
category: "Transformers & Sequence Models"
order: 32
updatedDate: "2026-08-30T13:59:21.722Z"
---
An RNN processes **sequential** data by maintaining a **hidden state** that carries information from
previous timesteps. Unlike feed-forward nets, it **shares the same parameters across all timesteps** —
so it handles **variable-length** sequences and generalizes to lengths unseen in training.

---

## The Elman RNN cell

$$h_t = \tanh(W_{ih}\, x_t + b_{ih} + W_{hh}\, h_{t-1} + b_{hh})$$

(matches PyTorch's `RNNCell`.)

- `x_t` = input at time `t`; `h_{t-1}` = previous hidden state.
- **`W_ih` shape `(h, d)`** — projects the input into hidden space.
- **`W_hh` shape `(h, h)`** — projects the previous hidden state into hidden space.
- **`h_t` is a compressed summary of all inputs up to `t`.** The *same* cell is applied repeatedly over
  the sequence.

**Why tanh:**
- Squashes activations to `[−1, 1]` → prevents **unbounded growth** of the recurrent state (ReLU in a
  recurrence can blow up).
- **Zero-centered** — outputs distributed around 0, better for gradient flow than sigmoid (whose
  all-positive outputs push same-sign gradients → zig-zag).
- Derivative `1 − tanh²(z)`: **≤ 1** (max at `z=0`), decaying to 0 for large `|z|` — which contributes
  to vanishing (below).

---

## BPTT — vanishing / exploding gradients (two factors)

In **backprop through time**, gradients flow backward through the chain of hidden states. Each step
contributes a multiplicative factor of roughly:

$$\frac{\partial h_t}{\partial h_{t-1}} \approx W_{hh}^\top \cdot \mathrm{diag}\big(\tanh'(\cdot)\big)$$

So there are **two** multiplicative culprits compounding over `T` steps:

1. **`W_hh` spectral norm** — `< 1` → product shrinks exponentially (**vanish**); `> 1` → **explode**.
2. **`tanh'` ≤ 1** — even a *perfectly conditioned* `W_hh` (spectral norm = 1) still vanishes because the
   **saturating activation derivatives compound**.

This is why vanilla RNNs struggle with **long-range dependencies**. (Exploding gradients are patched
with **gradient clipping**, Pascanu et al.; vanishing needs the architectural fix below.) See
[[training-diagnostics]].

---

## Weight sharing

`W_ih, W_hh, b_ih, b_hh` are **shared across all timesteps** → far fewer parameters than separate
weights per step, and the model **generalizes to sequence lengths unseen in training**.

---

## How gating fixes vanishing (LSTM / GRU)

The vanilla RNN's problem is that the hidden state is **fully rewritten** every step
(`h_t = tanh(W_hh h_{t-1} + …)`) — so the gradient is **forced** through a matrix-multiply-plus-squash
each step → structurally contractive → vanishes.

The **LSTM cell state** update is different:

$$c_t = f_t \odot c_{t-1} + i_t \odot g_t$$

Two properties make this a **gradient highway**:

1. **Additive** — `c_{t-1}` is *carried* (then new content added `+ i_t⊙g_t`), not destroyed and
   rebuilt through a matrix + nonlinearity.
2. **Element-wise gradient, not a matrix** — `∂c_t/∂c_{t-1} = f_t` (a diagonal/element-wise factor),
   **not** `W_hhᵀ·diag(tanh')`.

Compare the per-step gradient factor:

| | per-step `∂/∂` factor | behavior |
|---|---|---|
| **Vanilla RNN** | `W_hhᵀ · diag(tanh')` — full matrix × saturating derivative | structurally contractive → vanishes |
| **LSTM cell state** | `f_t` — a **learned** element-wise gate | can sit **≈ 1** → gradient flows undamped |

When the **forget gate learns `f_t ≈ 1`**, `∂c_t/∂c_{t-1} ≈ 1` → gradients cross many timesteps
**undamped**: the **constant error carousel**. It's exactly the **`+1` residual gradient highway**
([[normalization]]) — but **gated**: the network *chooses* when to keep the highway open vs. close it to
forget. The vanilla RNN has no opt-out; it *must* multiply by `W_hh·tanh'` every step.

**One-liner:** the LSTM replaces a *forced matrix-multiply-and-squash* with a *learned element-wise
gate that can approximate identity* — turning a structurally-vanishing path into a gated gradient
highway.

- **LSTM** — input / forget / output gates + a **separate cell state** `c_t`.
- **GRU** — simplifies to **update / reset** gates (merges cell & hidden state) → **fewer parameters**,
  retains most of the gating benefit.

The recurrent-vs-parallel training/inference duality (and how linear attention is a "fast-weight" RNN)
is in [[attention-free-architectures]].

---

Related: [[attention-free-architectures]], [[normalization]], [[training-diagnostics]],
[[self-attention]], [[momentum]]
