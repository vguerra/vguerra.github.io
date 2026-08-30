---
title: "Attention-Free / Sub-Quadratic Architectures"
description: "sub-quadratic landscape (linear attention, SSM/Mamba, Hyena, AFT), train/infer duality; linear attention deep-dive (kernel factorization, `(QKᵀ)V→Q(KᵀV)`, `d×d` recurrent state, feature maps, decay→RetNet/RWKV)"
category: "Transformers & Sequence Models"
order: 33
updatedDate: "2026-07-26T15:30:23.748Z"
---
Alternatives to softmax self-attention that avoid its **O(n²)** cost in sequence length (the `n×n`
score matrix at training, the linearly-growing **KV cache** at inference). All are trying to do
**sequence mixing** without the quadratic bill.

---

## The Landscape

| Family | Idea | Examples |
|---|---|---|
| **Linear attention** | drop softmax → factorize kernel → reassociate matmuls | Linear Transformer, Performer, RWKV |
| **State Space Models** | sequence as linear dynamical system `h_t = A h_{t-1} + B x_t` | S4, **Mamba** (selective/input-dependent) |
| **Long convolutions** | mix positions with learned long kernels + gating | Hyena |
| **Attention-Free Transformer (AFT)** | learned position bias + elementwise gating, no `n×n` | AFT (Apple 2021) |
| **Token-mixing MLPs** | MLP over the sequence axis | MLP-Mixer, gMLP (mostly vision) |

**The unifying win — train/infer duality.** The strong ones (linear attention, SSMs) have **two
equivalent forms**: a *parallel/convolutional* form for GPU-efficient training (whole sequence at
once) and a *recurrent* form for inference (fixed-size state, O(1)/token, no growing cache). Pure
RNNs had only the recurrent form (can't parallelize training); Transformers have only the parallel
form (growing KV cache). These architectures aim for both.

**The catch (why attention isn't dethroned).** A fixed-size recurrent state is **lossy compression
of history** — no exact lookup of an arbitrary past token → weak on precise long-range recall
(copying, induction heads, in-context retrieval). Hence **hybrids** (a few attention layers in a
Mamba stack, e.g. Jamba).

---

## Linear Attention (the core trick)

**Standard attention** for query `i`:

$$\text{out}_i = \sum_j \frac{\exp(q_i \cdot k_j / \sqrt{d})}{\sum_{j'} \exp(q_i \cdot k_{j'}/\sqrt{d})}\, v_j$$

The O(n²) comes from `exp(q_i·k_j)` for **every (i,j) pair** — softmax forces this because the
exponential of a dot product doesn't factor into a term for `i` times a term for `j`.

**The idea: replace the softmax similarity with a factorizable kernel** `sim(q_i, k_j) =
φ(q_i)·φ(k_j)`. Then:

$$\text{out}_i = \frac{\phi(q_i) \cdot \left(\sum_j \phi(k_j)\, v_j^\top\right)}{\phi(q_i) \cdot \sum_j \phi(k_j)}$$

Because `φ(q_i)` is independent of `j`, **pull it out of the sum**. The matrix
`S = Σ_j φ(k_j) v_jᵀ` is **`d×d`, independent of `i`** → compute once, reuse for all queries. Same
for normalizer `z = Σ_j φ(k_j)`.

**Complexity flip: O(n²d) → O(nd²).** It's just matmul associativity: `(QKᵀ)V` → `Q(KᵀV)`. Wins
whenever `n ≫ d`.

### The recurrent form (causal)

With causal masking (`j ≤ i`) the state is a prefix sum → a recurrence with **fixed `d×d` state**:

$$S_i = S_{i-1} + \phi(k_i) v_i^\top \qquad \text{out}_i = \frac{\phi(q_i)^\top S_i}{\phi(q_i)^\top z_i}$$

Inference is **O(1) in sequence length, no growing KV cache** — the duality: train in parallel, infer
as a linear RNN.

### Feature maps φ (must keep similarities ≥ 0)

- **Linear Transformer** (Katharopoulos 2020): `φ(x) = elu(x) + 1`.
- **Performer** (FAVOR+): random features so `φ(q)·φ(k) ≈ exp(q·k)` — an *unbiased approximation of
  softmax*, not a different kernel.

### Mental model: fast-weight associative memory

The recurrent state is a **matrix-valued RNN**: `S` accumulates key→value outer products
`φ(k)vᵀ`; reading is `φ(q)ᵀS` (query retrieves a blended value). This is Schmidhuber's "fast
weights."

### Why it underperforms softmax

1. **Fixed state = lossy memory** — all history compressed into one `d×d` matrix → poor exact
   long-range recall (copy/induction/lookup). Softmax can retrieve any exact past token.
2. **Less peaky weights** — softmax can put ~all mass on one token; positive-feature linear
   attention is smoother → weaker sharp selection.
3. **Normalization** `1/(φ(q)·z)` can be numerically fragile.

### Evolution: add decay/gating

Fix #1 by letting old associations fade: `S_i = γ S_{i-1} + φ(k_i)v_iᵀ`.
- **RetNet** — fixed exponential decay `γ`.
- **GLA / gated linear attention**, **RWKV** — data-dependent gating.
- Bridge to **Mamba** — a *content-dependent* (selective) version of this gated recurrence.

Related: [[self-attention]], [[transformer-architecture]], [[positional-encoding]]
