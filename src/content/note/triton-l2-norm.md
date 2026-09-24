---
title: "Triton: L2 Norm — Reduction with an In-Tile Fused Transform"
description: "reduction with in-tile fused transform: `√(Σx²)` = fold square before tree + host sqrt after; reduce-of-transform generalization (L2 square / L1 abs / logsumexp exp / count predicate), sentinel must respect transform dynamic range (`0.0` = identity + overflow-safe), 4N vs 12N bandwidth, non-linearity pitfalls (square before / sqrt after combine)"
category: "GPU / Kernels"
order: 85
updatedDate: "2026-09-18T07:06:28.977Z"
---
L2 norm = `√(Σ xᵢ²)`. It's a **sum reduction** ([[triton-sum-reduction]]) with a **square folded in
before the tree** and a **sqrt applied on the host after** the combine. Fusion makes the square a **free
in-register op** while the reduction shape stays identical to a plain sum.

## The pattern: reduce-of-transform

Read `x` once → **square in registers** (`tl.sum(x * x, axis=0)`, folded into the reduction operand, no
`x²` tile — [[triton-perf-tips]]) → atomic-add per-program partials into a scratch scalar → **host takes
`torch.sqrt`** and writes the user-visible output.

This is a **per-array reduction with an in-tile fused transform** — a reusable shape where the transform
varies:

| Kernel | In-tile transform | Reduction |
|---|---|---|
| **L2 norm** | square `x²` | sum, then host `√` |
| **L1 norm** | abs `|x|` | sum |
| **logsumexp** | `exp(x − m)` | sum, then host `log` + `m` |
| **count non-zeros** | predicate `x != 0` | sum |

**Launch contract:** (1) host allocates + **zeros** the scratch scalar; (2) after the kernel returns,
host applies the final nonlinear op (`√` here).

## Sentinel must respect the transform's dynamic range

The load sentinel (`other=`) must match **both** the reduction's additive identity **and** be safe under
the pre-reduction transform. For L2, **`0.0`** is ideal: `0² = 0` (additive identity) *and* immune to
overflow under squaring. A non-zero or extreme tail value would get **squared into** the sum.

## Bandwidth economics

Fused kernel: reads **4N** bytes, writes nothing. Naive two-kernel form (kernel 1 writes `x²`, kernel 2
reduces it): **8N + 4N = 12N** bytes (read x, write x², read x²) → **3× the traffic**. Atomics are the
same as plain sum: `G` atomics into one L2-hot cache line, serialized.

## Roofline

Per element: read 4 B, 2 FLOPs (one multiply + one add into the tree) → **2 FLOP / 4 B = 0.5 FLOP/byte**
→ **memory-bound**. Note the accumulator holds **squared** magnitudes, so it's larger → **rounding error
compounds more aggressively** than a plain sum (tighten tolerances at large N).

## Pitfalls (the non-linearity ones matter most)

- **Squaring after the reduction** — `tl.sum(x) * tl.sum(x)` computes `(Σx)² ≠ Σx²`. Fold the square into
  the operand: `tl.sum(x * x, axis=0)`.
- **`tl.sqrt` inside the kernel before the atomic** — `√a + √b ≠ √(a+b)`, so per-program sqrt-then-sum is
  wrong. The sqrt is the **final** op, applied **once** on the fully combined sum (same "nonlinear step
  after the cross-program combine" rule as the variance kernel — [[triton-fused-mean-variance]]).
- **Wrong/missing sentinel** — use `other=0.0` (additive identity *and* overflow-safe under squaring).
- **Reusing scratch without re-zeroing** — atomics accumulate stale state; first call works, later calls
  drift (call-order-dependent). Allocate via `torch.zeros` or `zero_()` before each launch.

---

Related: [[triton-sum-reduction]], [[triton-fused-mean-variance]], [[triton-perf-tips]], [[triton-max-reduction]], [[linear-algebra-basics]]
