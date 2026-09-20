---
title: "Triton: Row-Wise LogSumExp"
description: "row-wise LSE `m+log(Σexp(x−m))`: same row-parallel shape/stability as softmax but emits 1 scalar/row (asymmetric output bandwidth 4MN in / 4M out), fusion ~2× (or ~4× if naive spills the exp tile), pitfalls (add-max-back canonical bug, `−∞` sentinel, log-on-scalar-not-per-lane)"
category: "GPU / Kernels"
order: 80
updatedDate: "2026-09-20T16:37:08.944Z"
---
LogSumExp (LSE) emits **one scalar per row** instead of a normalized row — but it shares the **entire
stability story and program shape** with fused softmax ([[triton-fused-softmax]]). Same shift-and-exp;
the *divide* is replaced by a **`log` applied once to the row sum after the reduction**, plus an
**additive correction** that adds the subtracted max back.

For each row `i` of an `(M, N)` matrix:

$$\text{out}[i] = m_i + \log\!\Big(\sum_j \exp(x[i,j] - m_i)\Big), \quad m_i = \max_j x[i,j]$$

Direct `log(Σ exp(x))` overflows fp32 above ~88, so we use the identity **`log(eᵐ·S) = m + log(S)`** —
the row-max shift makes the largest exponent `e⁰ = 1`, then the `+m` restores the true value. Same
log-sum-exp principle as [[loss-functions]].

## Same row-parallel shape, asymmetric output

Identical **row-parallel** decomposition as softmax (grid of M programs, one row per program, whole row
in one register tile, no cross-program combine — [[triton-fused-softmax]]). The **only** difference is at
the output: softmax writes **N** values per row, LSE writes **1**. So the bandwidth is **asymmetric**:
- **Input** unchanged — reads each row exactly once → `4MN` bytes.
- **Output** drops from `4MN` to **`4M`** total → negligible for long rows.

## Bandwidth vs the unfused form

Fused: **`4MN` in + `4M` out** (≈ `4MN`). The naive multi-kernel form depends on whether it spills the
`exp` tile:
- **Reads x twice, no `(M,N)` intermediate** (K1: row max; K2: read x, fused exp+sum → per-row sum; K3:
  tiny log+add) → ≈ `8MN` → **~2× the fused traffic**.
- **Materializes the full `(M,N)` `exp` intermediate** (K2 writes it, K3 reads it back) → ≈ `16MN` →
  **~4×**.

Either way, on a memory-bound kernel the traffic ratio ≈ the runtime ratio, for identical arithmetic.

## Roofline

Per input element: read 4 B, ~2 FLOPs (one subtract, one exp — the row max, log, and add amortize over
the row), write ~nothing → **2 FLOP / 4 B = 0.5 FLOP/byte** → **memory-bound**.

## Pitfalls

- **Forgetting to add `row_max` back** — emitting `log(Σ exp(x − m))` alone is off by exactly `m`. The
  full identity is `m + log(Σ exp(x − m))`. **The canonical LSE bug** — the un-corrected value still
  *looks* like a finite log-probability, so it passes a smell test while being silently wrong.
- **Masked lanes with `0.0`** — `exp(0) = 1`, so each masked lane adds **1** to the sum → inflated LSE.
  Sentinel must be **`−∞`** (lose the max comparison → exp to 0). Same double-duty sentinel as softmax.
- **`tl.log` per lane** — the log must be taken on the **single scalar row sum**, not lane-wise:
  `log(exp(xⱼ − m)) = xⱼ − m`, which sums to something entirely different from `log Σ exp(·)`. (Mirrors
  the "nonlinear op after the combine, not before" rule from [[triton-l2-norm]].)
- **Hardcoding the row stride** (`row_idx * N`) — assumes row-contiguous, no padding. Pass `x.stride(0)`
  from the host for transposed/non-contiguous inputs.

---

Related: [[triton-fused-softmax]], [[triton-max-reduction]], [[triton-sum-reduction]], [[loss-functions]], [[perplexity]]
