---
title: "PyTorch Tensor Indexing: Slicing, Masking, Fancy Indexing"
description: "slicing (view) vs boolean masking (flattens) vs integer/fancy indexing; `gather`, `index_select`, `where`, `masked_fill`"
category: "PyTorch — Tensors & Mechanics"
order: 2
updatedDate: "2026-07-16T19:13:40.323Z"
---
Three distinct indexing modes. Knowing *which* to reach for — and their view/copy/shape
behavior — is a core PyTorch skill.

| You want... | Use | Returns |
|---|---|---|
| A regular range / crop / stride | **Slicing** `t[1:, ::2]` | **view** (shares memory) |
| Elements matching a **condition** | **Boolean mask** `t[t>0]` | copy, **flattened to 1D** |
| Elements at **specific positions** (reorder/repeat) | **Integer array** `t[idx]` | copy, **shaped like the index** |

---

## 1. Slicing — contiguous / strided sub-regions

`t[start:stop:step]` per dimension. Selects a **regular range**. Returns a **view** (no copy).

```python
x = torch.arange(12).reshape(3, 4)

x[0]          # first row → (4,)
x[:, 1]       # second column → (3,)
x[1:, 2:]     # bottom-right block → (2, 2)
x[:, ::2]     # every other column → (3, 2)
x[..., -1]    # last element on final dim (ellipsis = all remaining dims)
```

**Useful for:** cropping images/feature maps, last timestep of a sequence (`out[:, -1, :]`),
splitting Q/K/V by column ranges, batch subsetting. Default for any *regular* selection.

**Gotcha:** it's a **view** — in-place edits propagate to the original. `.clone()` for an
independent copy.

---

## 2. Boolean Indexing (Masking) — select by condition

Index with a **bool tensor** of matching shape. Keeps `True` elements. Returns a **copy** and
**flattens** to 1D (can't guarantee a rectangular result survives).

```python
x = torch.tensor([-2.0, 3.0, -1.0, 5.0])

mask = x > 0
x[mask]              # tensor([3., 5.]) — 1D, positives only

x[x < 0] = 0         # in-place ReLU-by-hand
```

Row selection with a 1D mask:
```python
X = torch.randn(5, 3)
keep = torch.tensor([True, False, True, False, True])
X[keep]              # (3, 3) — selected rows
```

**Useful for:** filtering (drop padding, keep finite/valid entries), thresholding, per-element
loss masking, selecting positives/anchors in detection or contrastive learning. Anytime
selection depends on **values**, not fixed positions.

**Key relatives:**
```python
torch.where(mask, a, b)      # elementwise pick — KEEPS shape (doesn't flatten)
mask.nonzero(as_tuple=True)  # indices where True
x.masked_fill(mask, value)   # fill masked spots — e.g. attention fills -inf before softmax
```

Shape distinction: `x[mask]` **flattens**; `torch.where` **preserves shape**.

---

## 3. Integer Array (Fancy) Indexing — gather by explicit indices

Index with a **tensor/list of integer positions**. Arbitrary order, repeats allowed. Returns a
**copy**; output shape follows the **index tensor's** shape.

```python
x = torch.tensor([10, 20, 30, 40])
idx = torch.tensor([0, 2, 2, 3])
x[idx]               # tensor([10, 30, 30, 40]) — reorder + repeat
```

**Classic "one element per row" pattern** (true-class logit for NLL loss):
```python
logits = torch.randn(batch, n_classes)
labels = torch.tensor([2, 0, 1])                    # true class per sample
true_logits = logits[torch.arange(batch), labels]   # (batch,)
```
The two index tensors are **broadcast together** (paired row→label), NOT a cross-product.
Same shape used in [[perplexity]] / NLL notes.

**Useful for:** per-sample target gathering, embedding lookups (`emb[token_ids]`),
reordering/shuffling rows, negative sampling, top-k gather, one-hot construction.

**Structured cousins:**
```python
torch.gather(t, dim, index)     # index along one dim with a same-rank index tensor
torch.index_select(t, dim, idx) # simpler: select along a single dim
```

---

## The Three Distinctions to Remember

1. **Slicing returns a view** → in-place mutations leak to the original.
2. **Masking flattens** → result is 1D (use `torch.where` to keep shape).
3. **Fancy indexing follows the index shape** → allows repeats & reordering; multiple index
   tensors broadcast (paired), not cross-product.

Related: [[pytorch-basics]], [[numpy-basics]], [[perplexity]], [[self-attention]]
