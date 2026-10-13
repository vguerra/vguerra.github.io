---
title: "Triton — Kernel Authoring & Perf Tips"
description: "reusable Triton micro-optimizations: prefer multiply-by-precomputed-reciprocal over divide-by-constant (e.g. GELU's `/√2`, `√(2/π)`); fuse elementwise into reduction `tl.sum(x*x)` (don't materialize intermediates → halve register pressure → higher occupancy, unless reused); **pointer-construction discipline** (logical indices → masks in logical coords → strides once; double-offset/relative-mask bugs, hand-check); FlashAttention-debugging gotchas (`tl.arange` constexpr in signature, no scalar tile indexing, `&`-vs-`<` precedence, sentinels on scores not loaded K, `tl.dot` constraints, accumulator shapes); constant vs runtime divisor caveat"
category: "GPU / Kernels"
order: 93
updatedDate: "2026-10-05T11:44:52.954Z"
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

## Pointer-construction discipline: logical coordinates first, strides once

Build every pointer tile in three steps, in this order:

1. **Logical indices.** Absolute row index `i` and column index `j` as index vectors. A block start is added in **row units** (`pid * BLOCK_M + arange`), never in elements.
2. **Masks in logical coordinates.** Compare `i` and `j` with the row and column bounds. The index and the bound are in the same units.
3. **Strides applied exactly once, last.** `ptr = base + i * stride_row + j * stride_col`.

The bugs this prevents all compile, run, and give wrong numbers or out-of-bounds reads:

- **Element offset where a row index belongs.** Folding a block start already scaled by the row stride into the row index, then multiplying by the stride again, scales it twice. E.g. `pid = 1`, `BLOCK_M = 16`, `D = 4`: row 0 of the block lands at element 256 instead of 64.
- **Block start added without the stride.** The offset is in rows but is added like elements.
- **Chunk offset applied twice in a loop.** A K/V loop that both adds `chunk * BLOCK_N * D` and advances the row index by `BLOCK_N` per iteration jumps too far. Keep **one** mechanism and make the mask follow it.
- **Mask against a block-relative index.** `arange < N` compares a within-block position, so on the last partial block it lets non-existent rows through. The mask must use the absolute index.

**Hand check:** compute the first element of a non-zero `pid`'s block, and the first element past the last valid row. If both land where expected, the pointer and the mask are almost certainly right. The same structure covers a transposed load (swap which index is broadcast on which axis, strides unchanged), 4-D tensors (batch and head are extra `index * stride` terms), and non-contiguous inputs (take strides from the tensor).

## Other authoring gotchas (from FlashAttention debugging)

- **`tl.arange` needs a constexpr extent.** Both ends must be known at compile time, and the extent must be a power of two. Annotate the parameter in the **kernel signature** as `tl.constexpr`; passing it as a keyword at the launch site is not enough. Each distinct constexpr value is a separate compiled variant.
- **No scalar indexing into a tile.** A tile is distributed across threads, so `x[i]` isn't available. Use mask plus `tl.where` plus a reduction, `tl.gather` (newer versions), or a pointer load from memory. Usually the real fix is to keep per-row statistics like the running max as `(BLOCK_M,)` vectors and use them through broadcasting.
- **Operator precedence in masks.** `&` binds tighter than `<`, so `a < N & b` parses as `a < (N & b)`. Parenthesize each comparison: `(a < N) & (b < D)`.
- **Sentinels go on scores, not on the loaded K.** Load masked K rows with `0.0` so the dot product stays finite, then set the invalid score columns to `-inf` after the dot. Loading K with `-inf` produces `NaN` through `0 * inf`. V's masked rows also get `0.0`.
- **`tl.dot` constraints.** Operand dtypes must match what the MMA supports (fp32 needs an `input_precision` choice), every dot dimension has a hardware minimum (commonly 16), and the `tl.trans` signature differs across Triton versions. Without error text, localize by the `line:col` prefix and bisect one change at a time.
- **Accumulator shape in attention.** Per-row max and denominator are `(BLOCK_M,)`; the output accumulator is `(BLOCK_M, BLOCK_DMODEL)`. Rescale the output accumulator with the per-row correction broadcast along the column axis, before adding the new `P @ V` contribution.

---

Related: [[triton-vector-add]], [[triton-relu]], [[triton-tiled-matmul]], [[triton-fused-softmax]], [[activation-functions]]
