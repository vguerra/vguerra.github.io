---
title: "Broadcasting in PyTorch"
description: "right-to-left alignment rules, stride-0 mechanism, (n,) vs (n,1) silent bug, `expand` vs `repeat`, `@` (matmul contracts) vs `*` (element-wise broadcasts)"
category: "PyTorch — Tensors & Mechanics"
order: 5
updatedDate: "2026-08-16T15:14:08.870Z"
---
Broadcasting combines tensors of **different shapes** in elementwise ops without copying memory.
It's behind most concise tensor code — and the source of some of the nastiest **silent** bugs.

---

## The Rules (two, applied right-to-left)

Align shapes **from the trailing (rightmost) dimension**, then check each pair:

1. Two dims are compatible if they are **equal**, OR **one is 1**, OR **one is missing** (a missing
   leading dim is treated as 1).
2. The output dim is the **max** of the two; a size-1 dim is *virtually stretched*.

```python
A = torch.randn(8, 1, 6, 1)
B = torch.randn(   7, 1, 5)
#  A: 8, 1, 6, 1
#  B:     7, 1, 5
#  →   8, 7, 6, 5    ✓
(A + B).shape         # (8, 7, 6, 5)

torch.randn(3, 4) + torch.randn(3, 5)   # ✗ 4 vs 5, neither is 1 → RuntimeError
```

---

## Mechanism: No Memory Is Copied

A broadcast dim of size 1 is **virtually repeated** by setting its **stride to 0** — the same
element is reread. No expanded tensor is materialized (see [[tensor-memory-layout]]).

```python
b = torch.randn(1, 5)
b.expand(3, 5).stride()   # (0, 1) — the 0 is the broadcast dim
```

---

## Where It Shows Up

**Bias / per-feature add over a batch:**
```python
X = torch.randn(32, 128)   # (batch, features)
b = torch.randn(128)       # (features,)
X + b                      # (128,)→(1,128) broadcast over 32 rows
```

**Per-channel scale/shift (norm affine params):**
```python
x = torch.randn(B, C, H, W)
gamma = torch.randn(C, 1, 1)
x * gamma                  # scales each channel
```

**Attention scores + mask:**
```python
scores = torch.randn(B, H, T, T)
mask   = torch.zeros(1, 1, T, T)   # broadcasts over batch & heads
scores + mask
```

**Pairwise / outer-product style:**
```python
a = torch.randn(n, d)
a[:, None, :] - a[None, :, :]      # (n,1,d)-(1,n,d) → (n,n,d) pairwise diffs
```

---

## The Classic Silent Bug — (n,) vs (n,1)

```python
pred   = torch.randn(100, 1)   # (n, 1)
target = torch.randn(100)      # (n,)
(pred - target).shape          # (100, 100)!!  — not (100,) or (100,1)
```
Align right: `(100,1)` vs `(100,)` → `(100,)` becomes `(1,100)`, then `1` broadcasts against `100`
on **both** axes → `(100,100)`. **No error** — just a huge wrong tensor and silently garbage loss.
(Same trap noted in [[pytorch-basics]] for loss functions.)

**Defenses:**
```python
assert pred.shape == target.shape   # fail loudly
target = target.unsqueeze(1)        # make (n,1)
pred   = pred.squeeze(1)            # or make (n,)
```
Match shapes explicitly at every elementwise op / loss.

---

## `@` (matmul) does NOT broadcast the core dims — it contracts them

Broadcasting is a property of **element-wise** ops (`+`, `-`, `*`, `/`). The matrix-multiply
operator `@` is a **different beast**: it *reduces* a shared dimension instead of stretching it.

```python
a = np.random.randn(3, 3)
w = np.random.randn(3)
a @ w      # (3,)   — matrix-vector product, NOT (3,3)
a * w      # (3,3)  — element-wise, w broadcast across rows
```

- **`a * w`** → element-wise. `w`=(3,) aligns right against `a`=(3,3) → treated as (1,3) → broadcast
  down the rows → **(3,3)**. This is the "stretch a size-1 dim" behavior.
- **`a @ w`** → matrix product. `w` is treated as a column; each row of `a` is **dot-producted**
  with `w`, summing over the shared length-3 axis:

  $$\text{result}[i] = \sum_j a[i,j]\,w[j]$$

  Row (3 numbers) · `w` (3 numbers) → **one scalar**; 3 rows → **(3,)**. The shared dim is
  **summed away**, not preserved.

**Shape rule for `@`:** `(m, k) @ (k,) → (m,)`. The last axis of the left contracts with the axis of
the right; the `k` **disappears**. Contrast `@`'s *contract-the-shared-dim* with broadcasting's
*stretch-the-size-1-dim* — opposite behaviors on the same shapes:

| shapes | `a * b` (element-wise, broadcasts) | `a @ b` (matmul, contracts) |
|---|---|---|
| `(4,3)`, `(3,)` | `(4,3)` — `b` stretched over rows | `(4,)` — 3-axis summed away |
| `(3,3)`, `(3,)` | `(3,3)` | `(3,)` |
| `(m,k)`, `(k,n)` | error (k≠n unless one is 1) | `(m,n)` — k contracted |

> `@` *does* broadcast the **leading batch dims** (e.g. `(B,m,k)@(B,k,n)→(B,m,n)`, and a missing
> batch dim broadcasts) — but the **last-two matrix dims always contract by the matmul rule**, never
> element-wise. Broadcasting applies only to the batch dims, not to the contracted core.

---

## Shaping Toolkit

```python
t.unsqueeze(dim)   # insert a size-1 dim (most common)
t[:, None]         # same via None-indexing
t.expand(shape)    # broadcast to shape — VIEW, stride-0, no copy
t.repeat(reps)     # COPIES data to tile — only when you need real writable copies
t.reshape(...)     # genuine restructure
```

**`expand` vs `repeat`** (common interview distinction):
- **`expand`** — free (view, stride-0); broadcast dims are virtual, not independently writable.
- **`repeat`** — allocates and copies real data. Use `expand` first; `repeat` only if you truly
  need independent copies.

---

## Two Things to Burn In

1. **Alignment is right-to-left**; a dim broadcasts only if **equal, 1, or missing**.
2. **`(n,)` ≠ `(n,1)`** — mixing them in an elementwise op silently yields `(n,n)`. Assert shapes.

Related: [[tensor-memory-layout]], [[tensor-indexing]], [[numpy-basics]], [[pytorch-basics]], [[normalization]]
