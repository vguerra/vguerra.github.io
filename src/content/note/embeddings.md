---
title: "Word Embeddings"
description: "embedding table `(vocab, d)`, differentiable lookup via one-hot-matmul view + sparse per-row gradient, why similar words cluster (distributional hypothesis, emergent not designed), static/context-free nature, weight tying, `padding_idx`"
category: "Transformers & Sequence Models"
order: 26
updatedDate: "2026-08-20T14:41:18.964Z"
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

Related: [[self-attention]], [[transformer-architecture]], [[tokenization]], [[tensor-indexing]],
[[positional-encoding]]
