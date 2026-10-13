---
title: "Triton: FlashAttention-2 Forward Pass"
description: "FlashAttention-2 forward: one program per query block, online-softmax state shapes (`m`,`l` `(BLOCK_M,)`, `acc` `(BLOCK_M,D)`), update order, **rescale multiplies both `l` and `acc`** (paper's printed `diag(·)⁻¹` inconsistent — derive, don't copy), `diag(v)` = row broadcast, first-iteration/`-inf` cases, mask & sentinel placement, `tl.dot` minimums (head-dim padding), cross-attention `N_q ≠ N_kv`, verification checklist, the unfused 3-kernel baseline (scores → row softmax → weights·V) and why fusing wins (`N²` HBM traffic, row-in-one-tile limit) with its bugs (flat-offset-vs-column mask, `+` vs `&` masks, `(N_q,N_kv)` buffer)"
category: "GPU / Kernels"
order: 81
updatedDate: "2026-10-09T13:09:52.445Z"
---
The synthesis of the kernel series: **tiled matmul** (the compute), **online softmax** (the cross-row trick), and **fusion** (never materialize the `N×N` scores in HBM). One program owns a block of query rows, streams over key/value blocks, and keeps a running softmax state in registers.

## Decomposition

- **Grid:** one program per block of `BLOCK_M` query rows (`cdiv(N_q, BLOCK_M)` programs). Programs are independent: disjoint output rows, so no atomics, scratch, or cross-program combine (same row-parallel structure as [[triton-fused-softmax]]).
- **Per program:** load the `Q` block once, scale it once (applying `1/sqrt(D)` to `Q` before the loop is cheaper than scaling the scores every iteration), then loop over K/V blocks of `BLOCK_N` rows.
- **Per iteration:** load a K block and V block, form the scores `S = Q·Kᵀ`, update the running state, accumulate `P·V`.
- **After the loop:** divide by the denominator and store (row mask on the store).

## The online-softmax state and its shapes

| Quantity | Shape | Initial value | Role |
|---|---|---|---|
| `m` (running row max) | `(BLOCK_M,)` | `-inf` | stability shift |
| `l` (running denominator) | `(BLOCK_M,)` | `0` | sum of `exp(s - m)` |
| `acc` (unnormalized output) | `(BLOCK_M, BLOCK_DMODEL)` | `0`, fp32 | sum of `exp(s - m)·V` |
| `S`, `P` | `(BLOCK_M, BLOCK_N)` | n/a | scores / exponentiated scores |

Per iteration, in this order:
1. Scores `S = Q·Kᵀ` (K loaded normally, transposed in registers with `tl.trans`).
2. Set the score **columns** of invalid key positions to `-inf`.
3. `m_new = max(m, rowmax(S))`: reduction over axis 1, elementwise max with the old vector.
4. Correction factor `c = exp(m - m_new)`, shape `(BLOCK_M,)`.
5. `P = exp(S - m_new[:, None])`.
6. `l = c·l + rowsum(P)`.
7. `acc = c[:, None]·acc + P·V` (**rescale the old accumulator first, then add the new block**).
8. `m = m_new`.

Finalize: `out = acc / l[:, None]`. Optionally `L = m + log(l)` is the row log-sum-exp, saved for the backward pass ([[triton-logsumexp]]).

## The rescale direction (verified, not assumed)

The old accumulator holds terms `exp(s - m_old)·v`. You want `exp(s - m_new)·v = exp(s - m_old)·v · exp(m_old - m_new)`. So **both** `l` and `acc` are **multiplied** by `c = exp(m_old - m_new)`, a factor `≤ 1`.

**A note on the paper's pseudocode:** the FlashAttention-2 paper's Algorithm 1, as printed, writes the output update with `diag(e^{m_old - m_new})⁻¹` while the denominator update uses `e^{m_old - m_new}` with no inverse. These two are inconsistent: `l` and `acc` hold sums over the same exponentials and must be rescaled by the same factor, and an inverse of `0` on the first step (`m_old = -inf`) is undefined. The version that matches a reference softmax multiplies both, which is what was confirmed here by derivation and against the platform's tests. Lesson: **derive the update direction; don't copy a formula you haven't checked**.

**`diag(v)` in these formulas is notation for row scaling:** `diag(v)·M` multiplies row `i` of `M` by `v[i]`, so in a kernel it is just `v[:, None]` broadcast against the tile. The diagonal matrix is never built. `diag(l)⁻¹` is likewise a per-row division by `l`.

**First iteration:** `m_old = -inf`, so `c = exp(-inf - finite) = 0`, and `0 × (zero accumulator) = 0` with no `NaN`. A **fully masked row** would give `-inf - (-inf) = NaN`; this cannot occur without a causal mask because every K block here contains at least one valid key. Adding causal masking requires a guard for that case.

## Masks and sentinels (where the bugs lived)

- **Load K and V with `other=0.0`.** `-inf` in K would make the dot product `NaN` via `0·inf`.
- **Put `-inf` on the scores, after the dot,** for key positions beyond the key length. That makes them lose the max and `exp` to exactly 0, so they contribute nothing to `l` or to `P·V`.
- **The score mask must vary along the key axis** (a `[None, :]` predicate on `S`). Using the K/V load mask on the scores is wrong: its axes are key rows and head-dim columns.
- **Head-dim padding is harmless:** padded columns of Q, K, V load as 0, contribute 0 to dot products, and the output store masks them off.
- **Cross-attention:** the query length and the key/value length are different quantities. The grid, the query-row mask, and the store use the **query** length; the loop count, the K/V load mask, and the score mask use the **key/value** length. Passing a single `N` taken from `Q` silently ignores the extra key rows (a test with 1 query and 2 keys returned 5 instead of 7.5). Keep the roles named and separate.

## `tl.dot` constraints that bit

- Each dot dimension has a **hardware minimum** (commonly 16), including the contraction extent `BLOCK_DMODEL`. A host that rounds the head dimension up from 1 gives `BLOCK_DMODEL = 2` for `D = 2`, and the dot cannot compile. Fix: `BLOCK_DMODEL = max(next_power_of_two(D), 16)`; the masks make the padding exact. Do not hard-code 16, which truncates `D > 16`.
- fp32 operands need an explicit `input_precision` choice (`"ieee"` for exact fp32).
- The dot's failure gave no readable message on the platform, only a `line:col`. Localization by bisection: comment out pieces, one change per submission.
- `tl.trans`'s signature varies by Triton version.

## Verification checklist

Compare against PyTorch softmax-attention on: partial last query block, partial last key block, tiny `D`, larger `D`, **`N_q ≠ N_kv`**, **`BLOCK_M ≠ BLOCK_N`** (square tiles hide broadcast-axis bugs), and a case with several key chunks.

## The unfused three-kernel baseline (and why fusing wins)

The straightforward implementation launches three kernels:

1. **Scores:** a 2-D grid of `(BLOCK_M, BLOCK_N)` output tiles computes `S = scale·Q·Kᵀ` and stores it to a scores buffer in HBM, shape **`(N_q, N_kv)`**.
2. **Softmax:** one program per row loads the full row of scores, subtracts the row max, exponentiates, normalizes, and stores the weights back (in place).
3. **Weights times V:** a grid over query blocks loops over key chunks, `acc += W_tile · V_tile` with an fp32 accumulator (the tiled matmul pattern), then stores.

**Why it costs more:** the scores matrix is `O(N_q·N_kv)` of **HBM memory**, written once, read and rewritten by softmax, read again by the last kernel. That is several passes over an `N²`-sized tensor, plus three launches. The fused kernel keeps the scores tile in registers and never writes it, so its HBM traffic is just `Q`, `K`, `V`, and the output. The unfused version also limits sequence length by memory, and the softmax kernel requires an **entire row to fit in one register tile** (`next_power_of_two(N_kv)` lanes), which breaks for long rows, exactly the case online softmax exists for ([[triton-fused-softmax]]).

**Bugs found while building it** (all compile and run, with wrong or zero output and no error text):
- **Flat offset compared with a length in the softmax mask.** `offs = row·N + col` was compared with `N` instead of comparing the **column index** with the row length, so every row except row 0 was entirely masked and left unnormalized. Same class as the pointer-discipline bugs in [[triton-perf-tips]]: compare logical coordinates with bounds, and apply strides separately.
- **`+` instead of `&` when combining boolean masks.** Adding two boolean tensors acts like an OR, so the mask admitted every position where either condition held, which allowed out-of-bounds reads of `V`.
- **Wrong scores-buffer shape and lengths for cross-attention.** The buffer must be `(N_q, N_kv)`. The scores grid's second axis, the softmax row length, the `S` row stride, the `S`-column mask, and the `V` row mask all use the **key/value** length; only the grid's first axis and the output masks use the query length. Allocating `(N, N)` and using one `N` everywhere hides the bug until the lengths differ.
- **Uninitialized buffer.** `torch.empty` leaves unwritten entries with garbage, so any mismatch in which entries get written shows up as noise.
- **Debugging without prints:** write an intermediate stage into `out` when the shapes allow it, add one stage at a time, and check that each softmax row sums to 1 (rows after the first failing points at row indexing).

## Not covered here

Causal masking, saving `L`, multiple heads and batches (extra `index·stride` terms; see the pointer discipline in [[triton-perf-tips]]), autotuning ([[triton-autotuned-matmul]]), the backward pass, and the transposed-K load.

---

Related: [[triton-fused-softmax]], [[triton-tiled-matmul]], [[triton-logsumexp]], [[triton-fused-matmul-bias-relu]], [[triton-perf-tips]], [[self-attention]]
