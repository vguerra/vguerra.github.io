---
title: "Tensor Memory Layout: Storage, Strides, Contiguity, view/reshape/permute"
description: "storage/strides/`data_ptr`, contiguity, `view` vs `reshape`, `permute`/`transpose`, `.contiguous()`"
category: "PyTorch — Tensors & Mechanics"
order: 3
updatedDate: "2026-07-16T19:26:23.131Z"
---
These five topics are really **one** concept: PyTorch separates a tensor's **shape (metadata)**
from its underlying **memory (storage)**. Everything else follows.

---

## Foundation: Storage vs. Metadata

A tensor = two things:
1. **Storage** — a flat, 1D contiguous block of memory holding the actual numbers.
2. **Metadata** — `shape`, `stride`, and a storage `offset` describing how to *interpret* that
   flat block as N-D.

```python
x = torch.arange(6).reshape(2, 3)
x.storage()     # flat buffer: [0, 1, 2, 3, 4, 5]
x.shape         # (2, 3)
x.stride()      # (3, 1) → skip 3 elems to move a row, 1 to move a column
x.data_ptr()    # raw memory address of element [0,0]
```

- **`stride()`** — elements to skip in storage to advance one step along each dim.
- **`data_ptr()`** — raw address of the first element.

**Debugging trick — do two tensors share memory?**
```python
y = x.view(3, 2)
x.data_ptr() == y.data_ptr()   # True → same storage (a view); mutating one mutates the other
```

---

## Contiguous Tensors

**Contiguous** = elements laid out in storage in **row-major (C) order** matching the current
shape (walking the tensor logically = walking memory sequentially).

```python
x = torch.arange(6).reshape(2, 3)
x.is_contiguous()    # True

xt = x.t()           # transpose → (3, 2)
xt.is_contiguous()   # False!
```

Why is transpose non-contiguous? `transpose`/`permute` **don't move data** — they only **swap
strides** in metadata. `xt` shares the same storage `[0..5]` but reads it in a jumping pattern
(stride `(1, 3)`). Logically transposed, physically unchanged → **O(1), no copy.**

---

## permute — reorder dimensions (non-contiguous view)

Generalizes transpose to arbitrary reordering. Rewrites strides only → **view, non-contiguous,
no data movement.**

```python
x = torch.randn(8, 3, 224, 224)   # NCHW
x.permute(0, 2, 3, 1).shape       # NHWC → (8, 224, 224, 3)
```

**Useful for:** NCHW↔NHWC conversion, moving seq/batch/head axes in attention
(`(B,T,H,D) → (B,H,T,D)`), aligning dims before matmul/broadcast.

`permute` = arbitrary reorder; `transpose(a, b)` = swap only two dims (see [[numpy-basics]]).

---

## view() vs reshape()

Both change shape; they differ in the **view-or-copy guarantee**:

| | Behavior | Fails on non-contiguous? |
|---|---|---|
| **`view()`** | *Always* a view — shares storage, zero copy | **Yes** — raises error |
| **`reshape()`** | View if possible, else silently **copies** | No — always succeeds |

```python
x = torch.arange(6).reshape(2, 3)
x.view(3, 2)        # ✓ contiguous → view

xt = x.t()          # non-contiguous
xt.view(6)          # ✗ RuntimeError: view size incompatible with stride
xt.reshape(6)       # ✓ works — makes a contiguous copy under the hood
```

**When to use which:**
- **`view()`** when you *know* it's contiguous and want a **guaranteed no-copy** (perf-critical).
  Its failure is a *feature* — it flags that memory isn't laid out as you assumed.
- **`reshape()`** when you just want it to work regardless (safer default, esp. after
  transpose/permute).

---

## .contiguous() — force a contiguous copy

Returns a tensor with the same values reorganized into row-major memory for the current shape.
**No-op if already contiguous** (returns self, free).

```python
xt = x.t()                        # non-contiguous view
xtc = xt.contiguous()             # physically reorders memory
xtc.is_contiguous()               # True
xtc.data_ptr() == xt.data_ptr()   # False → real copy
```

**When needed — the canonical `permute`/`transpose` → `view` pattern:**
```python
# Multi-head attention reshaping (very common):
x = x.permute(0, 2, 1, 3)             # non-contiguous
x = x.contiguous().view(B, T, H*D)    # .contiguous() REQUIRED before view
```

Without `.contiguous()`, `.view()` throws the stride error. Either insert `.contiguous()` before
`view()`, or just use `reshape()` (does it for you). Some ops also internally require contiguous
inputs.

---

## Mental Model (ties all five together)

- **Storage** = the flat numbers. **Metadata (shape/stride/offset)** = how to read them.
- **`permute`/`transpose`** = rewrite strides only → cheap view, but **non-contiguous**.
- **Contiguity** = logical order matches physical memory order.
- **`view`** = reinterpret without copying → *requires* contiguity (errors otherwise).
- **`reshape`** = `view` if it can, else copy.
- **`.contiguous()`** = the bridge — pay one copy to make a permuted tensor `view`-able.

**Classic sequence:** `permute` → `.contiguous()` → `view`  (or just `permute` → `reshape`).

Related: [[tensor-indexing]], [[pytorch-basics]], [[numpy-basics]], [[self-attention]]
