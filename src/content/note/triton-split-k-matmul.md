---
title: "Triton: Split-K Matmul"
description: "split-K matmul: parallelism along K when few output tiles (grid × SPLIT_K, pid_k decode, chunk_k in elements vs loop count in tiles), K mask derivation from absolute coordinates, empty/partial chunks, `/` vs `//` float-pointer error, `tl.minimum`, atomic_add (zeroing, nondeterminism → tight tolerance, fp16 output caveat), when it helps"
category: "GPU / Kernels"
order: 86
updatedDate: "2026-10-19T10:25:33.234Z"
---
The tiled matmul ([[triton-tiled-matmul]]) gives each output tile to one program, which loops over the **whole** K dimension. When M and N are small but K is large, there are few output tiles, so few programs, and most of the GPU's SMs sit idle. **Split-K** adds parallelism along K: the K range is cut into `SPLIT_K` chunks, each handled by its own program, and the partial results are summed.

## Decomposition

- **Grid:** 1-D, `cdiv(M,BLOCK_M) * cdiv(N,BLOCK_N) * SPLIT_K` programs.
- **Pid decode:** `pid_k = pid % SPLIT_K`, `pid_mn = pid // SPLIT_K`, then `pid_m, pid_n` from `pid_mn` (`// num_pids_n` and `% num_pids_n`). Putting `pid_k` in the fastest-varying position makes the programs that share an output tile adjacent.
- **K chunk:** `chunk_k = cdiv(K, SPLIT_K)` in K **elements**. Program `pid_k` covers `[k_start, k_end)` with `k_start = pid_k * chunk_k` and `k_end = minimum(k_start + chunk_k, K)`. The last chunk may be short, and for small K, trailing chunks may be empty.
- **Mental model:** program `pid_k` takes the `M × chunk` column slab of A and the matching `chunk × N` row slab of B. Their product is a full-size `M × N` **partial C**. The true C is the sum of the `SPLIT_K` partials.
- **Loop:** `cdiv(chunk_k, BLOCK_K)` iterations, each advancing `BLOCK_K` along K, with an fp32 accumulator and `tl.dot(..., input_precision="ieee")` for fp32 inputs.
- **Combine:** `tl.atomic_add(c_ptrs, acc, mask=c_mask)` into an output that the host **zeroes first** (`out.zero_()`).

## Units: elements vs tiles

The loop count is in **tiles**; `k_start`, `k_end`, `chunk_k` are in **K elements**. Mixing them was the main bug: `range(k_start, k_end)` iterates once per K *element*, not once per `BLOCK_K` step. Name quantities with their unit (`chunk_k` for elements, a tile count for iterations) and check with a toy case: K = 256, `SPLIT_K` = 4, `BLOCK_K` = 32 gives a 64-element chunk and 2 iterations.

## The K mask

`offs_k` is computed once and the pointers advance each iteration, so iteration `i` loads the absolute coordinates `offs_k + i*BLOCK_K`. The condition "coordinate `< k_end`" rearranges to `offs_k < k_end - i*BLOCK_K`, which is the mask used (on the `[None, :]` axis for A, `[:, None]` for B). Because `k_end` is clipped to both the chunk end and `K`:

- a partial last tile of a chunk cannot read into the next program's chunk,
- an empty chunk (`k_start >= K`) gives an all-false mask, so loads return `0.0`, the dot adds zeros, and the program adds zeros to C. This is safe, just wasted work.

Derive a mask by asking which absolute coordinate each lane holds this iteration and comparing that with the bound.

## Pitfalls hit

- **`/` is true division in Triton.** `pid / SPLIT_K` yields a float, so `pid_m`, `offs_m`, and everything built on them become float and the pointer expression fails with `IncompatibleTypeErrorImpl('invalid operands of type pointer<fp32> and triton.language.float32')`. The error points at the pointer arithmetic, but the cause is upstream, at the first `/`. Use `//` for indices. Debugging method: follow each operand's dtype backward to the first float-producing operator ([[triton-debugging]]).
- **Python `min` on runtime scalars** worked in the author's version but isn't guaranteed across Triton versions. Use `tl.minimum`.
- **Redundant cast:** `acc.to(tl.float32)` on an accumulator already fp32 does nothing.

## Atomics and numerics

- **Nondeterminism.** Floating-point addition isn't associative, and the order in which the `SPLIT_K` partials reach an output element varies between runs. Results are therefore not bitwise reproducible. The effect is a few ulps (about `1e-6` relative in fp32), so compare with a **tight** tolerance, found by measuring run-to-run differences or the error against an fp64 reference. A tolerance as loose as `1e-2` would accept a dropped partial tile or a wrong tail mask.
- **Zeroing.** `atomic_add` accumulates onto whatever is in memory, so `out` must be zeroed before launch. That costs an extra pass over C. Forgetting it adds the old contents.
- **Output dtype.** The accumulator is fp32. Atomically adding into an fp16/bf16 output rounds at every one of the `SPLIT_K` adds and may not be supported. Common alternatives: an fp32 workspace followed by a final cast, or a second reduction kernel that sums partials in a fixed order, which also makes the result deterministic.
- **Atomic traffic.** Each output tile receives `SPLIT_K` atomic updates, so contention and extra memory traffic are the price of the extra parallelism.

## When it helps

Split-K pays off when the number of output tiles is smaller than, or comparable to, the number of SMs while K is large (a tall-skinny or short-fat problem, a small-batch linear layer). There, extra programs fill idle SMs. When there are already many tiles per SM, splitting K adds atomics and zeroing overhead for nothing. `SPLIT_K` is a tunable ([[triton-autotuned-matmul]]); an alternative with the same goal is the stream-K style of scheduling.

## Tests worth running

- K not a multiple of `SPLIT_K * BLOCK_K` (for example K = 200: chunk of 50, partial second tile).
- K smaller than `SPLIT_K * BLOCK_K` (K = 40 or 10): some programs have short or empty chunks. Trace one by hand.
- M, N not multiples of the block size and M ≠ N, for the `% M` / `% N` wrap on loads against the unwrapped store mask.
- Non-constant, non-symmetric data, and several runs to measure nondeterminism.

---

Related: [[triton-tiled-matmul]], [[triton-grouped-matmul]], [[triton-autotuned-matmul]], [[triton-sum-reduction]], [[triton-debugging]]
