---
title: "Tensor Shape Typing — jaxtyping (and torchtyping)"
description: "annotating tensor shape+dtype with jaxtyping (`Float[Tensor, \"batch seq dim\"]`, axis grammar `*`/`#`/`...`, the `@jaxtyped` binding-scope gotcha, enforce-at-boundaries) and legacy torchtyping (`TensorType`, `patch_typeguard`)"
category: "PyTorch — Tensors & Mechanics"
order: 11
updatedDate: "2026-08-20T20:07:53.517Z"
---
Annotate tensor **shape + dtype** in type hints so mismatches are caught at a function boundary
instead of surfacing as a cryptic broadcast/matmul error several ops later. Executable-quality docs
that beat a rotting `# (B, T, D)` comment.

> **Status:** use **`jaxtyping`** for new code. `torchtyping` is in maintenance mode; jaxtyping is its
> successor (same author), with cleaner syntax and — despite the name — **full PyTorch support**
> (also numpy / jax / TF).

---

## jaxtyping (recommended)

**Core form:** `DType[TensorClass, "shape spec"]`

```python
from jaxtyping import Float, Int, Bool
from torch import Tensor

def attention(
    q: Float[Tensor, "batch seq dim"],
    k: Float[Tensor, "batch seq dim"],
    v: Float[Tensor, "batch seq dim"],
) -> Float[Tensor, "batch seq dim"]:
    ...
```

- **dtype = the wrapper:** `Float`, `Int`, `Bool`, `Complex`, `UInt`, `Shaped` (any), `Num`.
  `Float[Tensor, ...]` = any float precision; `Float32`/`Float16` pin one.
- **The `Tensor` slot** is the array class (`torch.Tensor`); same API across frameworks.
- **shape** = space-separated axis names in the string.

### Axis-spec mini-language

| Syntax | Meaning |
|---|---|
| `"batch seq dim"` | **named** dims — checked for consistency across a call |
| `"3 h w"` | **fixed** integer literal (e.g. 3 channels) |
| `"*batch seq dim"` | **variadic** `*name` — any number of leading dims, bound consistently |
| `"..."` | **anonymous variadic** — any number of dims, not name-bound |
| `"#batch"` | **broadcastable** axis — allows size-1 broadcasting |
| `"_"` | single **anonymous** dim (any size, not bound) |

```python
def layernorm(x: Float[Tensor, "*batch dim"]) -> Float[Tensor, "*batch dim"]:
    ...   # works for (B,D), (B,T,D), any leading dims — *batch absorbs them
```

### Runtime enforcement (jaxtyping = types only; pair with a checker)

**Per-function:**
```python
from jaxtyping import jaxtyped
import typeguard

@jaxtyped(typechecker=typeguard.typechecked)
def forward(x: Float[Tensor, "batch dim"]) -> Float[Tensor, "batch"]:
    ...
```

**Whole-module import hook** (cleanest for a project — scope it to *your* package, not third-party):
```python
from jaxtyping import install_import_hook
with install_import_hook("my_package", "typeguard.typechecked"):
    import my_package
```

---

## The #1 gotcha: `@jaxtyped` is what links the names

**Always wrap with `@jaxtyped` (or use the import hook).** It establishes a **per-call binding
scope**: the first tensor to bind `dim=768` locks it, and every later `dim` in that *same call* must
equal 768.

**Without `@jaxtyped`, typeguard checks each annotation independently** — it verifies each tensor is
*some* 3-D float tensor but **never links axis names across arguments**, so a `batch` mismatch between
`q` and `k` sails through silently. The cross-tensor consistency guarantee — the whole reason to use
shape typing — **exists only inside the `@jaxtyped` scope.**

---

## Best practices (both libraries)

- **Types alone = documentation; types + checker = enforcement.** Choose per function.
- **Static checkers (mypy/pyright) never verify shapes** — jaxtyping types are `Annotated[Tensor,…]`,
  accepted but not shape-checked. **All shape checking is runtime.**
- **Enforce at boundaries and in tests, not hot inner loops** — runtime checks cost per call. Import
  hook on your package is a good default; disable on perf-critical paths.
- **Prefer named dims over integer literals** unless a size is genuinely fixed — named dims are what
  buy the consistency check.
- **Use `*batch` for rank-flexible fns**, `#` when you *intend* broadcasting (so size-1 dims aren't
  rejected).
- **Annotate dtype meaningfully** — `Int[Tensor, "batch seq"]` for token ids, `Bool[Tensor, "seq
  seq"]` for masks — catches the float-where-`long`-expected bug (e.g. `nn.Embedding` needs `long`
  indices, see [[embeddings]]).

---

## torchtyping (legacy — for reading old code)

```python
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()          # call ONCE at startup — enables the typeguard hooks

@typechecked
def forward(x: TensorType["batch", "seq", "dim"]) -> TensorType["batch", "dim"]:
    ...
```

- **String** = named dim; **int** = fixed; **`-1`** = any; **`...`** = arbitrary leading dims; dtype
  as `TensorType["batch", float]`.
- Named-dim consistency works the same, but requires the global **`patch_typeguard()`** step
  (forgetting it = "why aren't my checks firing?"). jaxtyping drops this in favor of explicit
  `@jaxtyped(typechecker=...)`.

**Migration:** `TensorType["batch", "seq", "dim"]` → `Float[Tensor, "batch seq dim"]` (move dtype to
the wrapper, dims into one space-separated string).

---

Related: [[broadcasting]], [[tensor-memory-layout]], [[tensor-dtypes]], [[pytorch-basics]],
[[embeddings]]
