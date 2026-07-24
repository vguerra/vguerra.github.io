---
title: "Positional Encoding"
description: "sinusoidal PE, even/odd indices, frequency intuition, RoPE vs learned vs sinusoidal"
category: "Transformers & Sequence Models"
order: 19
updatedDate: "2026-07-06T19:49:55.313Z"
---
## Sinusoidal Positional Encoding

Adds position information to token embeddings. Shape: `(seq_len, d_model)`.

```
PE[pos, 2i]   = sin(pos / 10000^(2i/d_model))   ← even column indices
PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))   ← odd column indices
```

**Indices refer to column indices (embedding dimensions)** — not row indices.

For each token position (row), sin/cos alternate across embedding dimensions (columns):
```
position 0: [sin, cos, sin, cos, ...]
position 1: [sin, cos, sin, cos, ...]
position 2: [sin, cos, sin, cos, ...]
```

Each sin/cos pair uses a different frequency:
- Low column indices → high frequency (oscillate fast)
- High column indices → low frequency (oscillate slowly)

This gives every position a unique pattern across the embedding dimensions.

## Why Positional Encoding?

Attention is **permutation-invariant** — same tokens in any order produce same output. Positional encoding injects order information so the model knows token positions.

---

## Three Approaches

### Sinusoidal (original Transformer)
- Fixed, no learned parameters
- Generalizes to unseen sequence lengths
- Encodes absolute position; relative distances recoverable via sin/cos identities
- Weakness: absolute not relative position

### Learned Embeddings (BERT, GPT-2)
- Position embeddings are learned parameters
- Simple to implement, works well in practice
- Weakness: fixed max sequence length — can't generalize beyond training length

### RoPE — Rotary Position Embeddings (LLaMA, Mistral, GPT-NeoX)
Encodes position by rotating Q and K vectors before the dot product:
```
dot(rotate(Q, pos_i), rotate(K, pos_j)) depends only on (pos_i - pos_j)
```
- Encodes **relative** position directly in attention scores
- No added parameters
- Generalizes better to longer sequences
- Extendable to very long contexts (YaRN, LongRoPE)
- **Dominant approach in modern LLMs**

### Comparison

| | Sinusoidal | Learned | RoPE |
|---|---|---|---|
| Parameters | None | Yes | None |
| Extrapolates | Yes (limited) | No | Yes |
| Encodes | Absolute | Absolute | Relative |
| Used in | Original Transformer | BERT, GPT-2 | LLaMA, Mistral |

---

## Index Range

`i` runs from `0` to `d_model/2 - 1`. For `d_model=8`:
```
i=0 → columns 0 (sin), 1 (cos)
i=1 → columns 2 (sin), 3 (cos)
i=2 → columns 4 (sin), 5 (cos)
i=3 → columns 6 (sin), 7 (cos)
```
4 pairs × 2 = 8 columns — fills the full embedding dimension exactly.

Each pair shares the same frequency — sin and cos are phase-shifted by 90°, encoding the same positional scale from two angles.

## Iterating Over Columns Instead of Pairs

```python
for col in range(d_model):
    if col % 2 == 0:
        PE[pos, col] = sin(pos / 10000 ** (col / d_model))
    else:
        PE[pos, col] = cos(pos / 10000 ** ((col - 1) / d_model))
```

`col - 1` for odd indices ensures columns 0,1 share the same frequency, columns 2,3 share the same frequency, etc.

Relationship between `col` and `i`:
```
col = 2i   → i = col / 2
col = 2i+1 → i = (col-1) / 2
```
