---
title: "Triton: Fused Row-Wise Softmax"
description: "capstone: fuse max-subtract/exp/sum/divide, one program per row; **row-parallel vs per-array** (disjoint output rows → no atomics/scratch/combine), `−∞` sentinel double-duty (lose max + exp→0), load-row-once-reuse-in-registers (2.5× less traffic than 3-kernel), online-softmax cap → FlashAttention, max-subtract stability, store-mask silent-corruption pitfall"
category: "GPU / Kernels"
order: 84
updatedDate: "2026-09-20T16:04:17.030Z"
---
The capstone of the reduction series — fuse **max-subtract → exp → sum → divide** into one program per
row. In CUDA this needs an explicit block-per-row layout, a shared-memory reduction tree, and at least
one `__syncthreads`; in Triton it's almost trivial, which is *why* row-level programming is the right
abstraction for this shape. It's also the direct on-ramp to **FlashAttention**.

For an `(M, N)` matrix, produce an `(M, N)` output whose rows sum to 1:

$$\text{out}[i,j] = \frac{\exp(x[i,j] - \max_k x[i,k])}{\sum_{k} \exp(x[i,k] - \max_k x[i,k])}$$

## Row-parallel reduction — structurally different from per-array

**Grid = M programs, one per row.** Each reads `row_idx = tl.program_id(0)`, offsets into `x` by
`row_idx * x_row_stride`, and holds the **whole row as one `BLOCK_SIZE` register tile**. **No
cross-program communication.**

| | **Per-row** (softmax) | **Per-array** (sum/max/L2) |
|---|---|---|
| Output | each program owns a **disjoint** output row | one **shared** scalar |
| Atomics / scratch / combine | **none needed** | all required |
| Parallelism | across rows (grid); reduction within one program | tiles across programs + cross-program combine |

Because each program owns a disjoint output slice, softmax needs **no atomics, no scratch, no
cross-program combine** — the opposite of [[triton-sum-reduction]]. **Cap:** the whole row must fit in
one register tile (fine for `N` up to tens of thousands). For **very long rows** (hundreds of thousands
of columns) it degenerates → switch to **online softmax**: stream the row in chunks maintaining a
**running max and running sum** (rescaling the accumulator when the max updates). That online form is
exactly the FlashAttention inner loop.

## The `−∞` sentinel doing double duty

`BLOCK_SIZE = triton.next_power_of_2(N)` (constexpr). Mask `cols < N` on **both** load and store
(`cols = tl.arange(0, BLOCK_SIZE)`). Crucially, the load uses **`other = -float('inf')`**, not 0:
- Masked lanes load `−∞` → **lose every `tl.max`** comparison → row max uncorrupted.
- Under subtract-and-exp they become `exp(−∞ − row_max) = exp(−∞) = 0` → **contribute nothing to
  `tl.sum`**.

One sentinel value handles both reductions correctly. (Contrast a plain sum, where the sentinel is `0`.)

## The whole point: load the row once, reuse in registers

The row tile is loaded **once** and consumed by **every** subsequent op — max, subtract, exp, sum,
divide — with **no intermediate spilling to HBM**. This is the "materialize/keep-resident **because you
reuse it**" case from [[triton-perf-tips]]: `exp(x − m)` feeds *both* the sum *and* the final divide, so
it stays in registers. Traffic: **read N + write N** fp32 — the bandwidth lower bound for any softmax
that emits the full normalized output.

**vs the unfused 3-kernel pipeline** (A: row max → length-M buffer; B: read x + max → write `exp(x−max)`
to a fresh `(M,N)` intermediate; C: read intermediate → sum rows → write output): that moves **~5·4MN**
bytes (read x twice, write + read the `(M,N)` intermediate, write output) vs the fused **2·4MN** →
**~2.5× less HBM traffic** for the same arithmetic.

## Roofline

Per element: read 4 B, write 4 B, ~3 FLOPs (subtract, exp, divide — the reductions amortize over the
row) → **3 FLOP / 8 B = 0.375 FLOP/byte** → **memory-bound**.

## Author vs compiler

**Author:** grid of M programs, `constexpr BLOCK_SIZE = next_power_of_2(N)`, row-stride arithmetic, masks
on load *and* store, the `−∞` sentinel, and the deliberate in-register pipeline (the fusion). **Compiler:**
coalesced wide loads/stores, `tl.max`/`tl.sum` → warp-shuffle trees + shared-mem cross-warp staging,
`tl.exp` → fast fp32 approximation, and scheduling exp to overlap with the next load's latency.

## Pitfalls

- **Skipping max-subtract** — bare `exp(x)` overflows fp32 above ~88 → `inf/inf = NaN` for any row with
  a large positive logit. A **correctness** fix (one `tl.max` + subtract), not a perf one. Same
  log-sum-exp stability principle as [[loss-functions]].
- **`other=0.0` on the row mask** — 0 feeds `tl.max` and corrupts the row max on any **all-negative** row;
  post-exp it adds `exp(0 − row_max)` to the sum. Must be **`−∞`**.
- **Hardcoding the row stride** (`row_idx * N`) — only works if `x` is row-contiguous with no padding;
  pass `x_row_stride` from the host to stay correct on transposed/sliced inputs.
- **Forgetting the store mask** — writes garbage past column `N` into the output buffer. The load mask is
  usually remembered (it's first); the **store mask is the one that silently corrupts** neighboring
  memory.

---

Related: [[triton-sum-reduction]], [[triton-max-reduction]], [[triton-l2-norm]], [[triton-perf-tips]], [[loss-functions]], [[self-attention]]
