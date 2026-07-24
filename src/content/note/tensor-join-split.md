---
title: "Joining & Splitting Tensors: cat, stack, split, chunk"
description: "`cat` (existing dim) vs `stack` (new dim), `chunk` (count) vs `split` (size), `unbind`"
category: "PyTorch — Tensors & Mechanics"
order: 4
updatedDate: "2026-07-16T19:43:47.556Z"
---
**The core distinction:** `cat` joins along an **existing** dimension; `stack` creates a **new**
one. Splitting is the inverse — `chunk` specifies a **count**, `split` specifies a **size**.

| Op | What it does | Rank change | Shape requirement |
|---|---|---|---|
| `cat` | join along **existing** dim | same | match in all non-join dims |
| `stack` | join along **new** dim | +1 | **all identical** |
| `chunk` | split into **N** pieces | same | you give the count |
| `split` | split into pieces of **size S** | same | you give size / explicit list |
| `unbind` | remove a dim → tuple | −1 | — |

---

## torch.cat — join along an existing dimension

Same number of dims out; only the concatenated dim grows. All other dims must match.

```python
a = torch.randn(2, 3)
b = torch.randn(4, 3)
torch.cat([a, b], dim=0).shape   # (6, 3) — join rows; dim-1 must match

c = torch.randn(2, 5)
torch.cat([a, c], dim=1).shape   # (2, 8) — join columns; dim-0 must match
```

**Rule:** every dim except the `cat` dim must be identical.

**Useful for:** merging batches, concatenating parallel-branch features (U-Net skip connections,
multi-head outputs), appending timesteps/tokens, combining Q/K/V. Extending along an axis that
**already exists**.

---

## torch.stack — join along a NEW dimension

All inputs must be the **exact same shape**. Result has **one more dim**.

```python
a = torch.randn(3); b = torch.randn(3); c = torch.randn(3)
torch.stack([a, b, c]).shape        # (3, 3) — new dim 0
torch.stack([a, b, c], dim=1).shape # (3, 3) — new dim at axis 1

x = torch.randn(2, 4); y = torch.randn(2, 4)
torch.stack([x, y]).shape           # (2, 2, 4) — new leading dim
```

**Useful for:** building a batch from a list of same-shape samples (DataLoader's default
collate), stacking per-timestep outputs, assembling ensemble predictions, adding a new
heads/models axis.

### cat vs. stack — critical difference
```python
# two tensors of shape (3,)
torch.cat([a, b])      # (6,)   — glued end-to-end, SAME rank
torch.stack([a, b])    # (2, 3) — new axis, rank +1
```
Ask: *longer tensor (cat) or extra dimension (stack)?* `stack` needs **identical** shapes; `cat`
only needs matching **non-join** dims.

---

## torch.chunk vs torch.split — the two ways to break apart

Both return a **tuple of views** (share storage, no copy) and take a `dim` argument. They differ
in what you specify:

**`torch.chunk(t, chunks, dim)`** — a given **number** of pieces (as equal as possible):
```python
x = torch.arange(10)
torch.chunk(x, 3)      # sizes (4, 4, 2) — last smaller
```

**`torch.split(t, size, dim)`** — pieces of a given **size** (or explicit list):
```python
torch.split(x, 4)          # sizes (4, 4, 2) — chunks of 4
torch.split(x, [2, 3, 5])  # explicit per-chunk sizes → (2, 3, 5)
```

- **`chunk`** → "give me *N* pieces" (specify the **count**)
- **`split`** → "give me pieces of *size S*" (specify the **size** / list)

**Useful for:** splitting a fused QKV projection, dividing a batch across devices, fixed-length
windows over a sequence, separating concatenated features.

```python
qkv = self.qkv_proj(x)             # (B, T, 3*d)
q, k, v = qkv.chunk(3, dim=-1)     # three (B, T, d) tensors
```

---

## Related operations

```python
torch.unbind(t, dim)          # remove a dim → tuple of slices (inverse of stack)
torch.tensor_split(t, n)      # split with well-defined behavior for uneven sizes
torch.hstack / torch.vstack   # convenience: cat along dim 1 / dim 0
```
`unbind` is the natural inverse of `stack`: `torch.unbind(torch.stack(xs)) == xs`.

---

## The Two Decisions That Pick the Tool

1. **Joining:** longer tensor (`cat`) or new axis (`stack`)?
2. **Splitting:** specify *how many* pieces (`chunk`) or *how big* each is (`split`)?

Related: [[tensor-memory-layout]], [[tensor-indexing]], [[self-attention]], [[pytorch-basics]]
