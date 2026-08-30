---
title: "Word Embeddings"
description: "embedding table `(vocab, d)`, differentiable lookup via one-hot-matmul view + sparse per-row gradient, why similar words cluster (distributional hypothesis, emergent not designed), static/context-free nature, weight tying, `padding_idx`; Bag-of-Words mean-pooling + sentiment pipeline, permutation-invariance limitation (negation/order-blindness), masked-mean padding fix"
category: "Transformers & Sequence Models"
order: 30
updatedDate: "2026-08-24T19:09:48.103Z"
---
Neural nets can't process raw text, so tokens are mapped to **dense vectors** where similar words end
up close together. This note covers what the embedding table *is*, why the lookup is differentiable,
and *why* similar words cluster.

---

## The embedding table

Just a **parameter matrix** `E` of shape `(vocab_size, embed_dim)`. Each row is the learned vector
for one token id. To get a token's embedding you **index into the matrix**:

```python
E[token_id]          # → the (embed_dim,) row for that token
E[token_ids]         # → (len, embed_dim) for a sequence of ids
```

`nn.Embedding(vocab_size, embed_dim)` wraps exactly this (with an efficient lookup).

---

## Why the lookup is differentiable — the one-hot-matmul view

Indexing *looks* non-differentiable, yet embeddings are **trained**. The key: the lookup is exactly a
**matrix multiply with a one-hot vector**:

$$E[\text{id}] = \text{onehot}(\text{id}) \; @ \; E$$

The one-hot row selects row `id` of `E`. So it's a **linear op** → gradients flow into `E` like any
matmul weight. `nn.Embedding` just **skips building the one-hot and the full matmul** — the indexing
is an *efficient implementation* of that product.

**Sparse gradient (falls straight out of the one-hot view).** In
`∂L/∂E = onehotᵀ @ (upstream grad)`, the one-hot is zero everywhere except the used position → every
**unused row gets zero gradient**. Only the rows for token ids **actually in the batch** update each
step. This is why million-row vocabularies are tractable (you touch a handful of rows per step) and
why `nn.Embedding` offers `sparse=True` gradients.

---

## Why similar words cluster — the distributional hypothesis

**Nothing in the loss says "put similar words close."** The objective only says "predict well."
Clustering is an **emergent side-effect**:

Words in **interchangeable contexts** ("the ___ sat on the mat," "I fed the ___") must lead the
downstream network to make **similar predictions**. Since the network sees a word *only through its
embedding*, the only way to produce the same required output for both is for their embeddings to feed
the downstream layers similarly. Across many examples, "cat" and "dog" receive **similar gradient
signals** and drift into the same region of latent space.

This is the **distributional hypothesis** — *"you shall know a word by the company it keeps"* (Firth)
— the principle behind word2vec/GloVe and the input embeddings of a transformer LM. Interview point:
the geometry is **emergent, not designed**.

---

## Two things worth naming

- **Static / context-free.** A given token id maps to the **same** vector regardless of surrounding
  context, so "bank" (river) and "bank" (money) share one embedding *here*. Context-dependence is
  introduced **later**, by [[self-attention]], which mixes token representations based on the whole
  sequence.
- **Weight tying.** In LMs the input embedding matrix is often **shared with the output projection**
  (the pre-softmax layer) — the same `(vocab, d)` matrix, transposed — saving a large number of
  parameters and usually improving quality. Ties into [[transformer-architecture]].

Other details: **`padding_idx`** reserves a row (usually zeros, no gradient) for the pad token so
variable-length sequences batch cleanly (see [[dataloader-and-batching]]).

---

## Bag-of-Words pooling & a sentiment pipeline

**Bag of Words (BoW)** collapses a variable-length sequence of embedding vectors into **one
fixed-size vector by averaging** (mean-pooling over the sequence). Word **order is discarded** (hence
"bag"), yet the representation carries enough signal for many classification tasks.

A minimal binary-sentiment pipeline:

```
token ids (batch, seq)
  → Embedding(V, 16)      → (batch, seq, 16)
  → mean over seq (dim=-2)→ (batch, 16)          # BoW pooling
  → Linear(16, 1)         → (batch, 1)           # = logistic regression on pooled features
  → Sigmoid               → probability
```

- **Pool over the sequence axis:** for `(batch, seq, embed)` that's `dim=1` **= `dim=-2`**. Prefer the
  **negative index (`-2`)** — it means "the sequence axis" regardless of extra leading dims
  (`-1`=embed, `-2`=seq). `mean(dim=-1)` would wrongly average the *features*.
- **Training-loss note:** use **`BCEWithLogitsLoss` on the raw logit** (drop the `Sigmoid` for the
  loss) — same log-sum-exp fusion/stability argument as cross-entropy ([[loss-functions]]). Apply the
  sigmoid only at inference.

### The permutation-invariance limitation

Averaging is a **permutation-invariant** operation, so any two sentences with the same *multiset* of
words pool to the **identical** vector → identical prediction. "man bites dog" ≡ "dog bites man";
**"not good"** can't be distinguished from "good." **Order information is destroyed at the pooling
step**, before the classifier sees it — no training can recover it. This is *the* motivation for
order-sensitive models: RNNs, and attention + [[positional-encoding]].

### The padding bug — masked mean

Batched sequences are **padded** to equal length, so a naive `mean(dim=-2)` averages the **pad-token
embeddings too**, diluting the signal (more for shorter sentences). Masking the numerator isn't
enough — the **denominator** must also exclude pads (divide by real-token count, not `seq_len`):

```
numerator   = (emb * mask.unsqueeze(-1)).sum(dim=-2)          # zero pads, sum over seq → (batch, embed)
denominator = mask.sum(dim=-1, keepdim=True).clamp(min=1)     # real-token count; clamp guards all-pad
pooled      = numerator / denominator
```

- `mask` is `(batch, seq)` with 1=real, 0=pad → **`unsqueeze(-1)`** so it broadcasts against
  `(batch, seq, embed)` (see [[broadcasting]]).
- `mask.sum(dim=-1)` = number of real tokens per sequence; **`clamp(min=1)`** avoids ÷0 on an
  all-padding row.

---

Related: [[self-attention]], [[transformer-architecture]], [[tokenization]], [[tensor-indexing]],
[[positional-encoding]], [[loss-functions]], [[broadcasting]]
