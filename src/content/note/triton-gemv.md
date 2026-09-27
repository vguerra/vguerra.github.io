---
title: "Triton: GEMV (Matrix-Vector Product)"
description: "matrix-vector: per-row reduction with row-block grouping (BLOCK_M trades `x`-reuse vs SM occupancy), asymmetric reuse (`x` used BLOCK_M×, **A not reused**), AI → 0.5 FLOP/byte vs matmul's 16 → **why LLM decoding is bandwidth-bound but training is compute-bound** (and why batching restores GEMM), broadcast-direction & fp32-accumulator pitfalls"
category: "GPU / Kernels"
order: 81
updatedDate: "2026-09-27T20:08:49.589Z"
---
`A (M,N) @ x (N,) → out (M,)`, with `out[i] = Σⱼ A[i,j]·x[j]`. A rectangular `(BLOCK_M, BLOCK_N)` tile
that one program **walks along the N axis in chunks**, accumulating into a `BLOCK_M`-wide fp32 vector.

## Per-row reduction with row-block grouping

Grid is **1-D**: `cdiv(M, BLOCK_M)` programs. Each owns a slab of `BLOCK_M` consecutive **output rows**,
holds a `BLOCK_M`-wide register accumulator, and walks A's N axis together with the matching chunk of
`x` in `BLOCK_N` steps. Output is **partitioned** — no two programs write the same element.

Row reductions are **independent** (each output row could be its own program), so why group? **Grouping
`BLOCK_M` consecutive rows lets one loaded chunk of `x` be reused across all `BLOCK_M` rows** every
iteration. `BLOCK_M` therefore trades **reuse of `x`** against **number of programs available to fill
the SMs**.

**Standard shape: `BLOCK_M = 32`, `BLOCK_N = 64`.** `BLOCK_M` is a small multiple of the warp width so
the row dimension distributes cleanly across warps; `BLOCK_N` is wider so each iteration brings in
enough `x` to amortize over the `BLOCK_M` rows.

## Memory: the asymmetric reuse

The accumulator lives in **registers** for the whole N-loop and hits HBM exactly **once**. There's **no
SRAM staging** — with no `tl.dot`, the `tl.sum` along the column axis is a **register-level** reduction.

Inside one iteration the `BLOCK_N` chunk of `x` multiplies **all `BLOCK_M` rows** of the A tile, so each
fp32 of `x` is used **`BLOCK_M` times**. **The A tile has no reuse at all** — that asymmetry is the whole
story of GEMV.

Per program: `BLOCK_M · N` fp32s of A and `N` fp32s of `x`. Across `cdiv(M, BLOCK_M)` programs, **`x` is
read `cdiv(M, BLOCK_M)` times in total** — the larger `BLOCK_M`, the fewer redundant re-reads. Programs
run concurrently, so the `x` chunks one program reads tend to stay in **L2** long enough for the next to
hit them cached.

## Roofline — and why it explains LLM inference

Per inner iteration: read `BLOCK_M·BLOCK_N` (A) + `BLOCK_N` (x) fp32s, do `2·BLOCK_M·BLOCK_N` FLOPs:

$$\text{AI} = \frac{2 \cdot BM \cdot BN}{4(BM \cdot BN + BN)} = \frac{BM}{2(BM+1)} \ \text{FLOP/byte}$$

As `BM` grows this **asymptotes to 0.5 FLOP/byte** — far below the ~10 crossover, so **GEMV is
memory-bound for any practical block shape**.

**Versus matmul:** a matmul reuses **both** operands across the perpendicular tile axis; GEMV reuses
**only the vector**. Concretely, a 64×64 matmul tile gives **16 FLOP/byte** ([[triton-tiled-matmul]])
against GEMV's **≈0.49** — roughly **30×**, i.e. on the order of the tile dimension.

> **This is the structural reason transformer *inference* is bandwidth-bound while *training* is
> compute-bound on the same GPU.** Decoding is dominated by **GEMV-shaped** ops (one token at a time →
> the weight matrix multiplies a *vector*, so every weight is loaded and used once), while training is
> dominated by **matmul-shaped** ops (a full batch/sequence → weights are reused across many rows). It's
> also why **batching** decode requests helps so much: it turns GEMV back into GEMM and restores operand
> reuse. See [[model-serving]].

## Pitfalls

- **Accumulator dtype too narrow** — an fp16 accumulator silently loses precision for `N ≥ 128` (the
  running dot product's exponent outpaces fp16's 11-bit mantissa). **Accumulator must be `tl.float32`**
  even when A and x are fp16; cast to storage dtype on the final store. (Same rule as
  [[triton-tiled-matmul]].)
- **Forgetting `other=0.0` on the masked loads** — without an explicit additive identity, masked lanes
  yield whatever the compiler picks (usually zero, **not guaranteed across versions**) and `tl.sum` can
  pick up garbage on the N-tail iteration. Pass `other=0.0` explicitly on **both** the A-tile and x-chunk
  loads.
- **Wrong broadcast direction** — `x_chunk[:, None]` stretches the vector across the **row** axis instead
  of the column axis → completely wrong product. The shape contract: **`x_chunk` aligns with the
  `BLOCK_N` columns** of the A tile, so it's `x_chunk[None, :]`.
- **Forgetting the row mask on the store** — when M isn't a multiple of `BLOCK_M`, the last program's
  accumulator covers rows past the end of `out`; without `m_mask` the kernel writes past the buffer.

---

Related: [[triton-tiled-matmul]], [[triton-sum-reduction]], [[triton-vector-add]], [[model-serving]], [[self-attention]]
