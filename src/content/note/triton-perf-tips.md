---
title: "Triton — Kernel Authoring & Perf Tips"
description: "reusable Triton micro-optimizations: prefer multiply-by-precomputed-reciprocal over divide-by-constant (e.g. GELU's `/√2`, `√(2/π)`); constant vs runtime divisor caveat"
category: "GPU / Kernels"
order: 83
updatedDate: "2026-09-13T20:09:57.895Z"
---
Small, reusable micro-optimizations for writing fast Triton kernels. (Kernel fundamentals:
[[triton-vector-add]], [[triton-relu]].)

## Prefer multiply-by-reciprocal over divide-by-constant

Division is **more expensive** than multiplication on GPU hardware (a divide expands to several
instructions / uses the reciprocal unit; a multiply is one cheap op). When dividing by a **compile-time
constant** `c`, **precompute `1/c` as a Python float** and multiply:

```python
# instead of:  y = x / math.sqrt(2.0)
INV_SQRT2 = 0.7071067811865476        # = 1/sqrt(2), computed once at compile time (free)
y = x * INV_SQRT2                       # one multiply per lane, no divide
```

- The reciprocal is a **compile-time constant** → costs nothing at runtime; you just move the work from
  a per-lane divide to a per-lane multiply.
- **Where it shows up: GELU.** The erf form has `x/√2` and the tanh approximation has `√(2/π)` — fold
  these into precomputed multiplier constants rather than dividing per lane. Same for any activation with
  a fixed scale factor.
- With aggressive fast-math the compiler *may* do this automatically, but writing the multiply form
  **guarantees** it and is the idiomatic Triton style.

**Caveat:** this is for **constant** divisors. Dividing by a *runtime tensor* (e.g. the softmax
normalization sum) is a genuine element-wise reciprocal — you can't precompute it; there, use it as-is
(or `tl.math`'s reciprocal if you want the fast approximate instruction).

---

Related: [[triton-vector-add]], [[triton-relu]], [[activation-functions]]
