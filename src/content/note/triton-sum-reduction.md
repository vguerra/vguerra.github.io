---
title: "Triton: Sum Reduction — Atomics, Contention, Reproducibility"
description: "first reduction kernel: reduce-then-atomic pattern (tile→`tl.sum` in registers→one `tl.atomic_add`/program), must `out.zero_()` (atomic accumulates), operation-specific sentinel (`other=0.0` for sum vs −inf for max), atomic contention (G serialized L2 round-trips vs N per-lane), roofline 0.25 FLOP/byte, `tl.sum` internals (warp-shuffle + shared-mem), float non-associativity → non-reproducible (atol+rtol)"
category: "GPU / Kernels"
order: 82
updatedDate: "2026-09-15T14:45:37.007Z"
---
A **reduction** collapses an N-element vector into a **single scalar** — the first kernel where the
scalar must be **assembled from contributions of independent programs**. This is the canonical
per-array reduction pattern, and the conceptual jump from pointwise kernels ([[triton-vector-add]],
[[triton-relu]]): programs are **no longer independent at the output** — they all write the same address.

## The two-level pattern (reduce-then-combine)

1. Each program reduces its own `BLOCK_SIZE` tile to a scalar **in registers** with `tl.sum(x,
   axis=0)`.
2. Each program does **one** `tl.atomic_add` of its partial into `out[0]`.

**The output must be zeroed before launch** (`out.zero_()`) — the kernel reaches it via **atomic add**,
not a store, so a stale value accumulates into the result.

`BLOCK_SIZE = 1024` (power of two): wide vector loads + an in-tile reduction that lowers to a balanced
**log₂(1024) = 10-step tree**; large enough to amortize the per-program atomic, small enough for a
moderate register footprint.

**Operation-specific sentinel:** masked (tail) lanes must contribute the **additive identity `0`** →
`tl.load(..., mask=mask, other=0.0)`. Any other value pollutes `tl.sum`. (For a *max* reduction the
sentinel would be `−inf`, not 0 — the sentinel is the identity of *that* operation.)

## Atomics & contention — why reduce first

Every program emits **exactly one** atomic to the **same** HBM address; the hardware **serializes**
these through the **L2 atomic unit**. For a grid of `G` programs:
- **Atomic path:** `G` serialized round-trips to one cache line (each an **L2** round-trip).
- **Data path:** `4N/G` bytes per program, **fully parallel**, streamed **cold from HBM**.

The kernel is balanced when the **atomic stream is dwarfed by the data stream** — which is *exactly why*
each program reduces its tile first. **One atomic per lane** would replace `G` serialized round-trips
with `N` of them — same answer, a fraction of the bandwidth.

## Roofline

Per element: read 4 B (one fp32), do 1 add (the single scalar output + `G` atomics are negligible
bytes) → **arithmetic intensity = 1 FLOP / 4 B = 0.25 FLOP/byte** → **memory-bound**. Runtime is set by
how fast you can stream the input through HBM, not by the adds.

## What `tl.sum` does inside (1024 tile, `num_warps=4`)

Tile sharded **256 lanes per warp**. Each warp reduces its lanes with **register-shuffle** instructions
(`log₂(32) = 5` steps within a 32-lane warp) → one partial per warp. The **4 warp partials** are
exchanged through a small **shared-memory** staging (compiler inserts the barriers) → final scalar.
~`log₂(BLOCK_SIZE) = 10` logical steps. The author never names a warp, calls `__syncthreads`, or
declares shared memory — the compiler lowers all of it.

## Reproducibility — float sums aren't associative

Mathematical sum is commutative/associative; **fp32 is neither**. Additions are reordered in **two**
places: the in-tile reduction **tree**, and the **cross-program order the L2 atomic unit commits**. So:
- Two runs on the **same input are not bit-exact**.
- A **serial CPU sum almost never matches** the GPU sum.

Test with **both `atol` and `rtol`, scaled with N** (error accumulates with more terms).

## Pitfalls

- **Skipping `out.zero_()`** — `atomic_add` accumulates onto whatever's there; a stale value from the
  previous launch corrupts the result. *Worst failure mode:* the first call may work (fresh zeroed
  buffer), later calls drift — **call-order-dependent**.
- **Missing `other=0.0`** on the masked load — masked lanes return implementation-defined values that
  poison `tl.sum`. The sentinel is the only correct way to make them contribute the identity.
- **One atomic per lane** — serializes `N` atomics into one address; correct but a fraction of
  achievable bandwidth.
- **Expecting bit-exact CPU agreement** — parallel float reorders additions; use `atol`+`rtol`.

---

Related: [[triton-vector-add]], [[triton-relu]], [[triton-perf-tips]], [[numpy-basics]], [[self-attention]]
