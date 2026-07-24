---
title: "Tensor Dtypes, Casting, and Type Promotion"
description: "dtype landscape, fp16 vs bf16, casting & truncation, type promotion, NumPy float64 trap"
category: "PyTorch — Tensors & Mechanics"
order: 6
updatedDate: "2026-07-16T20:01:47.813Z"
---
## 1. The Dtype Landscape

| dtype | Alias | Notes |
|---|---|---|
| `torch.float32` | `float` | **Default for floats.** Weights/activations. |
| `torch.float64` | `double` | Rare in DL; scientific precision. |
| `torch.float16` | `half` | Mixed precision. **Narrow range** → overflow/underflow. |
| `torch.bfloat16` | — | Mixed precision. **Same exponent range as float32**, fewer mantissa bits. |
| `torch.int64` | `long` | **Default for ints.** Indices, labels, embedding lookups. |
| `torch.int32` | `int` | Smaller int. |
| `torch.uint8` | `byte` | Images (0–255), historically masks. |
| `torch.bool` | — | Masks (boolean indexing). |

```python
torch.tensor([1.0, 2.0]).dtype   # float32 (default float)
torch.tensor([1, 2]).dtype       # int64   (default int)
```

**Two defaults:** floats → `float32`, ints → `int64`. Hence `nn.Embedding` / `F.cross_entropy`
demand `long` indices, and masks are `bool`.

### float16 vs bfloat16 (mixed-precision interview favorite)
Both 16-bit, different bit split:
- **float16** — more mantissa (precision), **tiny exponent range** → overflow/underflow; needs
  **loss scaling** to train stably.
- **bfloat16** — **same exponent range as float32** (truncated mantissa) → won't overflow where
  float32 wouldn't; far more robust for training. Standard on TPUs, A100+.

Trade: bf16 sacrifices *precision* to keep *range* — usually right for gradients.

---

## 2. Type Casting

All create a **copy** unless the dtype already matches:

```python
x.to(torch.float32)   # explicit, most general
x.float()             # → float32
x.double()            # → float64
x.half()              # → float16
x.long()              # → int64  (most common: indices/labels)
x.int()               # → int32
x.bool()              # → bool

x.type_as(other)      # match another tensor's dtype
x.to(device, dtype)   # cast dtype AND move device in one call
```

**Truncation gotcha** — float→int **truncates toward zero**, does NOT round:
```python
torch.tensor([1.9, -1.9]).int()    # [1, -1]  — not [2, -2]
torch.tensor([1.9]).round().int()  # [2]      — round first if you want rounding
```

**Why it matters:**
- Loss labels: `F.cross_entropy` needs `long` → `y.long()`
- Masks: boolean indexing needs `bool`
- Mixed precision: cast activations to `half`/`bfloat16`
- Overflow safety: cast to `float32`/`float64` before a sensitive reduction (summing many fp16)

---

## 3. Type Promotion in Operations

Mixing dtypes → PyTorch promotes to a common result type. Rules:

**Category ladder (low→high):** `bool` → int → float → complex. Higher category wins.
```python
(torch.tensor([1,2]) + torch.tensor([1.0,2.0])).dtype   # float32 (float > int)
(torch.tensor([True]) + torch.tensor([1])).dtype        # int64  (int > bool)
```

**Within a category, wider wins:**
```python
(torch.ones(3, dtype=torch.float16) +
 torch.ones(3, dtype=torch.float32)).dtype              # float32
```

**Scalar wrinkle** — Python scalars have *weaker* influence; they don't bump a tensor's
category/width:
```python
x = torch.ones(3, dtype=torch.float16)
(x * 2).dtype      # float16 — Python int does NOT promote
(x * 2.0).dtype    # float16 — Python float keeps tensor dtype
```
Deliberate: `x * 2` shouldn't silently upcast your fp16 tensor. Two *tensors* of different dtype
**do** promote.

**Where it bites — NumPy's float64:**
```python
np_arr = np.array([1.0, 2.0])     # float64!
t = torch.from_numpy(np_arr)      # inherits float64
(t + torch.randn(2)).dtype        # float64 — silently promoted, 2× memory
```
**Defense:** be explicit at boundaries — `torch.from_numpy(arr).float()` — and print `dtype` when
debugging numeric issues.

---

## Essentials

1. **Defaults:** floats→`float32`, ints→`int64`. Labels/indices = `long`; masks = `bool`.
2. **Casting copies**; **float→int truncates** (round first for rounding).
3. **Promotion:** higher category wins (bool<int<float<complex), wider wins within — but
   **Python scalars don't force promotion.**
4. **bf16 keeps float32's range** (robust); **fp16 needs loss scaling.**
5. **Watch NumPy `float64`** silently promoting your graph.

Related: [[pytorch-basics]], [[broadcasting]], [[numpy-basics]]
