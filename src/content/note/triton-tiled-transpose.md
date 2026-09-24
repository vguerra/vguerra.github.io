---
title: "Triton: Tiled Transpose — Coalescing Is the Whole Game"
description: "pure permutation (no arithmetic/reduction) → coalescing is the only lever: scalar transpose writes column-strided (1 elem/cache line), tiled loads to registers + stores transposed; 2-D tile map without reduction, transpose via store stride-swap, one shared mask; tiling changes **achieved bandwidth** (4×) not intensity; cache-line-matched blocks (32), shared-mem padded variant for peak"
category: "GPU / Kernels"
order: 80
updatedDate: "2026-09-24T20:00:05.836Z"
---
`A (M,N) → out (N,M)` with `out[i,j] = A[j,i]`. **No arithmetic, no reduction** — a pure permutation of
indices. Every element moves exactly once: `4·(MN + MN) = 8MN` bytes for fp32. **Hard memory-bound**,
which makes it the cleanest demonstration in the series that **memory access pattern alone** decides
performance.

## The coalescing problem

A scalar transpose reads **row-contiguous** from `A` and writes **column-strided** into `out` — adjacent
stores land `M` fp32s apart, so each store touches a **different cache line**: one useful element per
transaction instead of a full line. That runs at a **fraction** of HBM bandwidth.

**The tiled fix:** load a full `BLOCK_M × BLOCK_N` tile into **registers** in row layout, then store it
back from the *same* register tile in **transposed layout** — so each issued store still produces a
**contiguous burst along the destination's fast axis**. Both ends coalesce.

## 2-D tile map without reduction

Grid is 2-D; each program owns one `BLOCK_M × BLOCK_N` tile. **No K-loop, no reduction, no cross-program
coordination** — every output tile depends on exactly **one** input tile. Each program does: **one load,
one in-register layout reinterpretation, one store.** (The simplest 2-D decomposition — contrast
[[triton-tiled-matmul]], which adds a K-loop reduction per tile.)

**The transpose happens in the stride math on the store, not in a data shuffle:**
```
a_ptrs   = a_ptr   + offs_m[:, None]*stride_am + offs_n[None, :]*stride_an
out_ptrs = out_ptr + offs_n[None, :]*stride_om + offs_m[:, None]*stride_on
```
Note the swap: on the destination, the **column** index `offs_n` multiplies `out`'s **row** stride and
the **row** index `offs_m` multiplies `out`'s **column** stride.

**One shared mask serves both** load and store: `(offs_m[:,None] < M) & (offs_n[None,:] < N)`. Validity
is symmetric under transpose — if lane `(i,j)` is in-bounds in `A`, then `(j,i)` is in-bounds in `out`.

## Why tiling wins (without changing intensity)

The tile lives in registers for the program's lifetime. A scalar element-by-element transpose issues
`BLOCK_M × BLOCK_N` separate stores, each hitting a different cache line (stride `M` fp32s between
adjacent stores) — terrible per-store amortization.

**Key framing:** tiling does **not** change the arithmetic intensity (there's no arithmetic) — it
changes the **achieved bandwidth**, sometimes by **4× or more** between the scalar and tiled forms. On
this kernel, optimization is *exclusively* about effective bandwidth utilization: coalescing on both
ends, blocks large enough to amortize launch overhead, and enough programs to saturate the SMs.

**Compiler-handled:** vector-width selection for load and store — **128-bit** when tile shape and stride
allow, **64-bit** under weaker alignment, scalar fallback only as a last resort.

## Pitfalls

- **Forgetting to swap the stride axes on the store** — writing `offs_m[:,None]*stride_om +
  offs_n[None,:]*stride_on` produces a **copy of A, not its transpose**. It passes on square *symmetric*
  inputs and fails everywhere else. (Same silent-on-symmetric-tests family as the matmul `AᵀB` stride bug
  — [[triton-tiled-matmul]].)
- **Two separate masks for load and store** — `(offs_m < M) & (offs_n < N)` and its reordering are
  **identical predicates**; computing both just doubles the broadcast cost in emitted PTX. Reuse one.
- **Assuming this hits peak bandwidth** — the in-register tile transpose is a big win over scalar but
  still loses some bandwidth to imperfectly-coalesced stores when `out` is row-major and the tile shape
  interacts badly with the cache line. True peak needs the **shared-memory padded** variant (padding to
  break bank conflicts) — a separate kernel.
- **Block sizes that mismatch the cache line** — a `BLOCK_N` smaller than the fp32s per cache line
  (typically 16 or 32) makes loads **sub-line**, halving or quartering effective bandwidth.
  **`BLOCK_M = BLOCK_N = 32`** matches one 128-B cache line of fp32 on most hardware: smaller starves
  bandwidth, larger raises register pressure with no further load-side gain.

---

Related: [[triton-tiled-matmul]], [[triton-vector-add]], [[tensor-memory-layout]], [[triton-perf-tips]]
