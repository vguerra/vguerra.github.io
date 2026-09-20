---
title: "Triton — Kernel Authoring & Perf Tips"
description: "reusable Triton micro-optimizations: prefer multiply-by-precomputed-reciprocal over divide-by-constant (e.g. GELU's `/√2`, `√(2/π)`); fuse elementwise into reduction `tl.sum(x*x)` (don't materialize intermediates → halve register pressure → higher occupancy, unless reused); constant vs runtime divisor caveat"
category: "GPU / Kernels"
order: 86
updatedDate: "2026-09-18T06:39:02.801Z"
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

## Fuse elementwise ops into the reduction (don't materialize intermediates)

Pass the expression **directly** to the reduction so the compiler folds the elementwise op into the
reduction tree instead of materializing a full intermediate tile:

```python
s2 = tl.sum(x * x, axis=0)         # ✅ square folded in — only x is live
# xsq = x * x; s2 = tl.sum(xsq)    # ⚠️ materializes a second full tile → ~2× register footprint
```

Holding `x` **and** `x²` doubles the reduction's register footprint; `tl.sum(x * x)` keeps only `x`
resident. **Fewer registers → higher occupancy → more warps in flight to hide HBM latency** — exactly
what a memory-bound reduction ([[triton-fused-mean-variance]], [[triton-sum-reduction]]) wants.
Generalizes to any *reduce-of-elementwise*: `tl.sum(x * w)`, `tl.max(x - m)`, `tl.sum(tl.exp(x - m))`
(the softmax denominator). Only materialize an intermediate when you **reuse** it (e.g. softmax needs
`exp(x-m)` for both the sum *and* the final divide — then storing it once is right).

---

Related: [[triton-vector-add]], [[triton-relu]], [[activation-functions]]
