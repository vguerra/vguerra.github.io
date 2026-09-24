---
title: "Triton: Tiled Matrix Multiply — Reuse Crosses the Roofline"
description: "first **compute-bound** kernel: reuse via tiling crosses the roofline; 2-D grid mirrors output (private per-tile K-reduction, no cross-program combine), SRAM operand staging + register accumulator + `num_stages` pipelining, L2 super-grouping, AI = `BM·BN/2(BM+BN)` (=16 at 64×64, BLOCK_K cancels), `tl.dot`→tensor-core MMA + swizzle, pitfalls (fp32 accumulator, K mask, stride-order = AᵀB bug)"
category: "GPU / Kernels"
order: 82
updatedDate: "2026-09-21T19:50:05.595Z"
---
`A (M,K) @ B (K,N) → C (M,N)`. In a **naive** matmul every element of A is read **N** times (once per
output column) and every element of B is read **M** times → memory-bound. **Tiling** is the fix: a **2-D
output tile owned by one program**, a **K-loop that accumulates into a register tile**, and one
**`tl.dot`** per iteration lowering to a tensor-core MMA. **Reuse is what turns matmul from memory-bound
into compute-bound**, and the tile is the unit that materializes that reuse. This is the first
**compute-bound** kernel in the series (contrast the memory-bound reductions).

`C[i,j] = Σₖ A[i,k]·B[k,j]` — total work **`2MNK` FLOPs** against a best-case **`4(MK + KN + MN)`** bytes
of HBM (read A, read B, write C, fp32).

## 2-D grid mirrors the 2-D output

Grid: `cdiv(M, BLOCK_M)` × `cdiv(N, BLOCK_N)`. Each program reads `(pid_m, pid_n)` from
`program_id(0/1)` and owns one **`BLOCK_M × BLOCK_N`** output tile. The tiles **partition** C — no two
programs write the same element, and **no cross-program reduction along K** (this version). So the kernel
is **embarrassingly parallel at the tile level**: each program does a **private** dot-product over K and
stores its tile; the only synchronization is the launcher's. (Like the row-parallel softmax
[[triton-fused-softmax]] — disjoint output → no atomics/scratch — but 2-D.)

`BLOCK_M, BLOCK_N, BLOCK_K` are `tl.constexpr` so the compiler sizes registers, unrolls the K-loop, and
picks tensor-core MMA instructions of the right shape. Common start: **`BLOCK_M = BLOCK_N = 64`,
`BLOCK_K = 32`** — powers of two (clean vectorization), large enough to keep tensor cores busy, small
enough to fit registers + SRAM.

## The memory hierarchy at work

Per K-iteration: one `tl.load` for A's `(BLOCK_M, BLOCK_K)` slab + one for B's `(BLOCK_K, BLOCK_N)` slab.
The compiler **stages both into on-chip SRAM** and feeds them to the MMA. The **accumulator lives in
registers** for the whole K-loop and hits HBM only at the **final masked store** (cast to storage dtype
there).

- **L2 reuse across programs:** adjacent `pid_n` programs load the **same A slab**; if their lifetimes
  overlap on an SM cluster, L2 serves the second without going to HBM. The naive **row-major** program-ID
  order exploits this poorly → the standard fix is a **grouped program-ID remap** ("L2-friendly
  schedule" / super-grouping) that interleaves `pid_m`/`pid_n` so neighbors share operands. (Separate
  optimization; not this kernel.)
- **Software pipelining (`num_stages`):** with `num_stages=2` the compiler issues the **next**
  iteration's A/B loads while the current `tl.dot` is in flight (**double-buffering** the SRAM staging) →
  tensor cores never stall on HBM. Cost: 2× SRAM footprint; benefit: operand-load latency hidden behind
  MMA latency.

## Arithmetic intensity — why it's compute-bound

Per K-iteration: load `BM·BK + BK·BN` fp32 values, do `2·BM·BN·BK` FLOPs:

$$\text{AI} = \frac{2 \cdot BM \cdot BN \cdot BK}{4(BM \cdot BK + BK \cdot BN)} = \frac{BM \cdot BN}{2(BM + BN)} \ \text{FLOP/byte}$$

**`BLOCK_K` cancels** — intensity scales with the **output-tile dimensions** `BM, BN`, not the K-depth.
For `BM = BN = 64` → **16 FLOP/byte**, past the ~10 crossover → **compute-bound**. Bigger output tiles →
more reuse → higher intensity; `BLOCK_K` is a **scheduling/occupancy** knob (SRAM size, pipeline depth),
not an intensity knob.

## `tl.dot` — all compiler

`acc += tl.dot(a, b)` on a `(BM,BK)×(BK,BN)` pair → the compiler lowers to tensor-core MMA of the target
shape, allocates SRAM staging, inserts the warp synchronization, and **swizzles the operand layout** to
avoid the shared-memory bank conflicts that would otherwise serialize SRAM access. Author never writes
any of it.

## Production optimizations (not in the basic kernel)

- **Autotune** over tile shape + `num_stages`.
- **Grouped program-ID remap** for L2 reuse (super-grouping).
- **`tl.make_block_ptr` + `tl.advance`** to express operand stride math symbolically once → cleaner
  K-loop pointer arithmetic.

## Pitfalls

- **Accumulator dtype too narrow** — `acc` as `tl.float16` silently loses precision once K passes a few
  dozen (the running sum's exponent outpaces fp16's 11-bit mantissa). **`acc` must be `tl.float32`** even
  with fp16 inputs/outputs; cast to storage dtype only on the final store. *The* matmul numerics rule.
- **Missing the K mask** — when K isn't a multiple of `BLOCK_K`, the last iteration overshoots the
  operands. Need `k_mask = (k + offs_k) < K` on **both** the A and B loads, or garbage past the operand
  end gets accumulated → breaks every non-aligned shape.
- **Wrong stride order** — putting `offs_m` on `stride_ak` and `offs_k` on `stride_am` silently computes
  **`AᵀB`** instead of `AB`. It compiles, runs, and may even pass square/symmetric random tests. Pull
  `A.stride(0), A.stride(1)` and pair them with the **row, then column** index expressions.
- **Block sizes not `constexpr`** — a runtime block forces a generic loop instead of the unrolled
  tensor-core MMA → correct but often an **order of magnitude slower**.

---

Related: [[triton-vector-add]], [[triton-fused-softmax]], [[convolution]], [[self-attention]], [[tensor-dtypes]]
