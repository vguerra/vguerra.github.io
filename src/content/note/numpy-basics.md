---
title: "NumPy Basics"
description: "one-hot, argmax, rounding, type conversion, random matrices, `default_rng`, transpose, broadcasting, reshape, reductions, norms, ReLU, stable sigmoid"
category: "NumPy & Python"
order: 8
updatedDate: "2026-07-18T14:12:32.571Z"
---
## One-Hot Encoding

Use the identity matrix trick — clean and handles batches naturally:

```python
n_classes = 5
one_hot = np.eye(n_classes)[class_idx]           # single vector
one_hot_batch = np.eye(n_classes)[indices]        # batch, shape (N, n_classes)
```

For large `n_classes`, avoid the intermediate matrix:
```python
one_hot = np.zeros(n_classes)
one_hot[class_idx] = 1.0

# batch version
one_hot_batch = np.zeros((len(indices), n_classes))
one_hot_batch[np.arange(len(indices)), indices] = 1.0
```

---

## argmax

Returns the index of the maximum value:

```python
np.argmax(a)              # global max index
np.argmax(a, axis=1)     # per-row max index (batch of logits)
```

Complement: `np.argmin` for minimum index.

---

## Rounding Floats

```python
round(3.14159265, 6)          # Python built-in → 3.141593
f"{3.14159265:.6f}"           # string formatting
np.round(arr, 6)              # numpy array
```

---

## Converting Array Types

```python
arr.tolist()        # numpy array → Python list of floats (for JSON etc.)
arr.astype(float)   # stays ndarray, ensures float64
arr.astype(int)     # float → int
arr.astype(np.int32)
```

---

## Array ↔ Python List

**Array → list:**
```python
arr.tolist()   # ✓ recursive; casts elements to NATIVE Python types (int/float)
list(arr)      # ✗ iterates only the first axis; elements stay numpy scalars (np.float64)
```
Use `.tolist()` — handles any dimensionality and produces native types (needed for `json.dumps`,
which chokes on numpy scalars):
```python
np.array([[1, 2], [3, 4]]).tolist()   # [[1, 2], [3, 4]] — nested
list(np.array([1.0, 2.0]))            # [np.float64(1.0), np.float64(2.0)] — not Python floats
```

**List → array:**
```python
np.array([1, 2, 3])               # infers dtype (int64)
np.array([[1, 2], [3, 4]])        # nested → 2D
np.array([1, 2, 3], dtype=float)  # force dtype
np.asarray(my_list)               # skips copy if already an ndarray
```

- **`np.array`** always **copies** by default; **`np.asarray`** avoids the copy when the input is
  already an ndarray of the right dtype — prefer it for "array-like" function inputs.
- **Ragged lists** (unequal sublengths) → error in modern numpy (older: object array — avoid):
  ```python
  np.array([[1, 2], [3]])   # ERROR
  ```

---

## Initializing a 2D Matrix with Random Values

**Standard normal:**
```python
np.random.randn(rows, cols)    # legacy
rng = np.random.default_rng(seed=42)
W = rng.standard_normal((rows, cols))   # modern
```

Scale for neural network weights:
```python
W = np.random.randn(rows, cols) * 0.01
```

**Uniform distribution:**
```python
np.random.rand(rows, cols)              # [0, 1) — legacy
np.random.uniform(0, 1, (rows, cols))   # explicit range — legacy

# Modern approach
rng = np.random.default_rng(seed=42)
rng.uniform(0, 1, (rows, cols))         # [0, 1)
rng.uniform(-1, 1, (rows, cols))        # [-1, 1)
rng.random((rows, cols))                # [0, 1) — shorthand
```

---

## np.random.default_rng

Modern random number generator (NumPy 1.17+). Preferred over legacy `np.random.*`.

```python
rng = np.random.default_rng(seed=42)   # reproducible
rng = np.random.default_rng()          # random each run
```

**Why preferred:**
- Explicit seed on the object — no global state interference
- Higher quality algorithm (PCG64 vs Mersenne Twister)

**Common methods:**
```python
rng.standard_normal((3, 4))
rng.random((3, 4))             # uniform [0, 1)
rng.integers(0, 10, (3, 4))
rng.choice([1, 2, 3], size=5)
rng.shuffle(arr)
```

---

## Transpose / Permute Dimensions

```python
A.transpose(0, 2, 1, 3)          # pass new order of axes
np.transpose(A, (0, 2, 1, 3))    # same
```

`(0, 2, 1, 3)` means: keep axis 0, swap axes 1 and 2, keep axis 3.

PyTorch equivalent:
```python
A.permute(0, 2, 1, 3)   # arbitrary reordering
A.transpose(1, 2)        # swaps only two dims at a time
```

---

## Applying a Function Element-wise

```python
vf = np.vectorize(my_func)
vf(A)   # applies my_func to each scalar element
```

`np.vectorize` is syntactic sugar over a Python loop — not truly fast. Prefer native numpy ops:
```python
A ** 2 + 1          # C-level vectorized — much faster
np.maximum(A, 0)
```

Use `np.vectorize` only when the function can't be expressed with native numpy ops.

If the function needs the **row index**, build an index matrix or loop over rows:
```python
# index matrix approach
rows, cols = A.shape
i_matrix = np.repeat(np.arange(rows).reshape(-1, 1), cols, axis=1)
vf(A, i_matrix)

# loop approach — cleaner
result = np.stack([my_func(row, i) for i, row in enumerate(A)])
```

---

## Broadcasting

NumPy broadcasts automatically when dimensions are **equal** or **1**:

```python
row = np.ones((1, d))   # (1, d)
A   = np.ones((m, d))   # (m, d)
A + row                 # (1, d) → repeated m times automatically
```

Explicit broadcasting:
```python
np.broadcast_to(row, (m, d))   # read-only view
```

PyTorch:
```python
row + A                # same automatic broadcasting
row.expand(m, d)       # explicit
```

---

## Vector (i,) → Column Vector (i,1)

```python
v.reshape(-1, 1)        # numpy and pytorch — most portable
v[:, None]              # slicing with None adds a dim
v[:, np.newaxis]        # numpy — same as None, explicit
v.unsqueeze(1)          # pytorch
np.expand_dims(v, 1)    # numpy function form
```

---

## Reshape

```python
A.reshape(3, 4)      # explicit dims — total elements must stay the same
A.reshape(3, -1)     # -1 infers the missing dim automatically
A.reshape(-1)        # flatten to 1D

A.flatten()          # always returns a copy
A.ravel()            # returns view when possible (faster)
```

PyTorch equivalent:
```python
A.reshape(3, 4)      # same API
A.view(3, 4)         # faster but requires contiguous memory
```

---

## Mean and Variance

```python
np.mean(x)                            # global mean
np.var(x)                             # variance (1/N by default)
np.std(x)                             # standard deviation

np.mean(x, axis=-1, keepdims=True)    # across features — keepdims for broadcasting
np.var(x, axis=-1, keepdims=True)
```

PyTorch:
```python
x.mean(dim=-1, keepdim=True)
x.var(dim=-1, keepdim=True, unbiased=False)  # unbiased=False → 1/N (matches LayerNorm)
```

`np.var` uses `1/N` (population variance) by default. PyTorch `torch.var` defaults to `unbiased=True` (`1/(N-1)`) — set `unbiased=False` to match numpy/LayerNorm.

---

## Reduction Along an Axis

```python
np.max(A, axis=0)              # max along rows → shape (cols,)
np.max(A, axis=1)              # max along cols → shape (rows,)
np.max(A)                      # global max → scalar

np.max(A, axis=1, keepdims=True)   # shape (rows,1) — preserves dim for broadcasting
```

Same API applies to: `np.min`, `np.sum`, `np.mean`, `np.argmax`, `np.argmin`.

**NumPy vs PyTorch:** NumPy uses `axis`, PyTorch uses `dim` — same concept, different keyword.
```python
np.max(A, axis=1)     # numpy
torch.max(A, dim=1)   # pytorch
```

---

## Multiplication Operations

```python
a * b              # element-wise
np.multiply(a, b)  # same

np.dot(a, b)       # 1D vectors: inner product → scalar
a @ b              # dot product or matmul depending on shape (preferred)

np.matmul(A, B)    # matrix multiplication (m,k)@(k,n) → (m,n)
np.outer(a, b)     # outer product (m,)×(n,) → (m,n)
```

| Operation | Inputs | Output |
|---|---|---|
| Element-wise | `(n,)` × `(n,)` | `(n,)` |
| Dot product | `(n,)` × `(n,)` | scalar |
| Matrix × vector | `(m,n)` × `(n,)` | `(m,)` |
| Matrix × matrix | `(m,k)` × `(k,n)` | `(m,n)` |
| Outer product | `(m,)` × `(n,)` | `(m,n)` |
| Batched matmul | `(b,m,k)` × `(b,k,n)` | `(b,m,n)` |

`@` is preferred for matrix/vector multiplication — works for 2D and batched cases. Use `*` only for element-wise.

---

## Computing Norms

**NumPy:**
```python
np.linalg.norm(v)              # L2 norm (default)
np.linalg.norm(v, ord=1)       # L1 norm
np.linalg.norm(v, ord=np.inf)  # L∞ norm
```

**PyTorch:**
```python
torch.linalg.norm(v)           # L2 norm (default)
torch.norm(v)                  # older API, same result
torch.linalg.norm(v, ord=1)
```

---

## ReLU

No built-in in NumPy — use:
```python
np.maximum(x, 0)     # recommended
x * (x > 0)          # alternative
np.clip(x, 0, None)  # alternative
```

PyTorch:
```python
torch.relu(x)
torch.nn.functional.relu(x)
```

---

## Numerically Stable Sigmoid

NumPy has no built-in sigmoid. Options:

```python
from scipy.special import expit   # recommended
expit(x)

# Manual stable implementation:
def sigmoid(x):
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))
```
