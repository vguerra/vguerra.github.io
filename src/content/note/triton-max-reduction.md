---
title: "Triton: Max Reduction — Identity, Bit-Exactness, Single-Tile Pattern"
description: "max reduction: `−∞` identity (discard-by-comparison), whole-array-in-one-tile (no pre-zero, `next_power_of_2`), **bit-exact under reordering** (unlike sum → strict `==`), roofline 0.25 op/byte + launch-overhead-dominated, two-stage scratch fallback beyond register cap, `atomic_max` is integer-typed caveat, sentinel table (−∞/+∞/0/1)"
category: "GPU / Kernels"
order: 84
updatedDate: "2026-09-16T20:44:49.264Z"
---
Max is a reduction like sum ([[triton-sum-reduction]]) but with **two key differences**: a natural
identity element **`−∞`** (masked lanes are *discarded by comparison*, not added), and **bit-exactness
under reordering** (max is exactly associative; float sum isn't). It's the building block for **softmax /
logsumexp**, where the row-wide max is the first step of the numerical-stability pipeline.

## Whole-array-in-one-tile pattern

The simplest form: **one program** loads the entire array into a register tile, calls `tl.max` once,
stores one scalar.
- **No pre-zeroing needed** — with a single program the combine is a plain `tl.store`, not an atomic
  (contrast the sum kernel, which *must* `out.zero_()` because it accumulates via `atomic_add`).
- **`BLOCK_SIZE` = `triton.next_power_of_2(N)`** — the whole array must fit one tile.

**Operation-specific sentinel:** masked (tail) lanes load **`other=−inf`** (the identity of max), so they
lose every comparison. Matching the sentinel to the op: **`−∞` max, `+∞` min, `0` sum, `1` product.**

## Bit-exactness — the contrast with sum

`max` is **associative in exact, infinite-precision arithmetic**, and comparison has no rounding — so
**any order of pairwise reductions gives the same result**. Parallel and serial max are **guaranteed to
match exactly**, and two GPU runs are bit-identical. This is the opposite of the float **sum**, whose
tree-order and cross-program commit order make it non-reproducible. One of the few cases where GPU/CPU
agreement is exact — you can test max with a strict `==`.

## Roofline

Per element: read 4 B, one comparison → **0.25 op/byte** → **memory-bound** (crossover ~10, two orders
of magnitude away). For small `N` fitting one block, **launch overhead dominates**: a 4096-lane fp32 tile
is 16 KB (~128×128 B coalesced transactions) → ~**16 ns** through HBM at 1 TB/s, vs ~**5–10 µs** launch
overhead. Zero reuse; operands live in registers from `tl.load` through `tl.max`.

## Why power-of-two blocks

`tl.max` (and every Triton reduction) lowers to a **balanced tree of register-shuffle** instructions,
which needs power-of-two tile extents. A runtime/non-power-of-two shape forces a **scalar fallback** on
the trailing group and disables most of the optimization → still compiles, loses most of its speed.

## When N outgrows one block (~65536 lanes)

A tile of `2²⁰` lanes doesn't fit in registers on any current GPU. Switch to a **two-stage scratch**
form (no atomics):
1. **Stage 1:** launch `cdiv(N, BLOCK_SIZE)` programs; each reduces its tile with `tl.max` and writes its
   partial to `scratch[pid]` with a plain `tl.store`.
2. **Stage 2:** a single program loads `scratch` and reduces it the same way.

Alternative: **`tl.atomic_max`** — but it's **integer-typed**, so fp32 needs a **sign-aware
float→int reinterpret** (order-preserving only for non-negative inputs) *and* it serializes on one
address across `G` programs. The two-stage scratch kernel is the cleaner cross-program combine for
floats.

## Pitfalls

- **`other=0.0` instead of `−∞`** — silently breaks for any input whose true max is **negative** (masked
  lanes return 0, which wins). Match the sentinel to the reduction.
- **Non-power-of-two `BLOCK_SIZE`** — pass `N` through `triton.next_power_of_2` on the host before the
  `tl.constexpr`, else scalar fallback.
- **`tl.atomic_max` on fp32** — it's integer-typed; needs a sign-aware reinterpret. Prefer the two-stage
  scratch kernel for floats.
- **Assuming the single-program form scales arbitrarily** — it caps at the max register-resident block;
  beyond that, go two-stage.

---

Related: [[triton-sum-reduction]], [[triton-vector-add]], [[triton-relu]], [[loss-functions]], [[self-attention]]
