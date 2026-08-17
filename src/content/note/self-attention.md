---
title: "Self-Attention"
description: "scaled dot-product attention, QKV, softmax gotchas"
category: "Transformers & Sequence Models"
order: 19
updatedDate: "2026-07-05T14:30:04.454Z"
---
## Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
```

```python
import torch
import math

def compute_qkv(X, W_q, W_k, W_v):
    Q = torch.matmul(X, W_q)
    K = torch.matmul(X, W_k)
    V = torch.matmul(X, W_v)
    return Q, K, V

def self_attention(Q, K, V):
    d_k = K.shape[-1]
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    attn_weights = torch.softmax(attn_scores, dim=-1)
    return torch.matmul(attn_weights, V)
```

## Step-by-step

| Step | Operation | Shape | Purpose |
|------|-----------|-------|---------|
| 1 | `QKᵀ` | `(seq_len, seq_len)` | similarity score between every query-key pair |
| 2 | `/ √d_k` | same | dot product variance grows as `d_k` — dividing by `√d_k` restores unit variance, keeps softmax out of saturated regions |
| 3 | `softmax(dim=-1)` | same | converts scores to weights summing to 1 per row |
| 4 | `· V` | `(seq_len, d_v)` | weighted sum of values |

## Why √d_k Scaling

Dot products between Q and K vectors have variance proportional to `d_k`. Large scores push softmax into saturation:
- Distribution collapses to near one-hot (attends to only one token)
- Gradients through softmax become near-zero (vanishing gradients)

Dividing by `√d_k` restores unit variance — keeps softmax in a stable, informative regime.

**Note:** the cause is vector dimension `d_k`, not sequence length.

## Masking with np.where

```python
# general syntax
np.where(condition, value_if_true, value_if_false)

# causal mask — block future positions
seq_len = 4
mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
scores = np.where(mask, -np.inf, attention_scores)
```

Masked positions get `-inf` → after softmax `e^(-inf) = 0` → zero attention weight.

In PyTorch:
```python
scores = scores.masked_fill(mask, float('-inf'))
```

## Gotchas

- Use `dim=-1` (not `dim=1`) for softmax — works correctly for batched inputs
- Use `K.transpose(-2, -1)` (not `K.T`) — `.T` only works for 2D tensors
- `d_k = K.shape[-1]` — use last dim, not `shape[1]`, for batch compatibility
