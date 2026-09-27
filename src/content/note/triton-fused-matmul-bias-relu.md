---
title: "Triton: Fused Matmul + Bias + ReLU (Epilogue Fusion)"
description: "**epilogue fusion** (the reason to write Triton kernels): 20MN→4MN activation traffic (5×) + 3 launches→1, bias broadcast is free, epilogue ops raise intensity, **the fusion boundary** (tile-local ok / cross-row needs separate kernel), variants (residual add, quantized output), pitfalls (op order, `bias[None,:]`, bias mask, load-after-K-loop)"
category: "GPU / Kernels"
order: 80
updatedDate: "2026-09-27T20:32:09.531Z"
---
**Epilogue fusion is the canonical optimization that makes writing Triton kernels worth it for
transformer inference.** An unfused linear layer does the matmul, **writes** the full `(M,N)` output to
HBM, **reads** it back to add bias, **writes** again, **reads** a third time for ReLU, and **writes**
the final result — one matmul plus **four extra HBM round-trips** on the activation tensor.

This kernel collapses all of it into **one launch**: the accumulator **never leaves registers** between
the last `tl.dot` and the masked store; the bias is loaded once and broadcast across rows in-place; and
`tl.maximum` applies the ReLU before the single store.

Given `A (M,K)`, `B (K,N)`, `bias (N,)`:

$$\text{out}[i,j] = \max\!\Big(0,\ \Big(\sum_k A[i,k]\,B[k,j]\Big) + \text{bias}[j]\Big)$$

The bias is **per column**, broadcast across rows. Note the bias is added **once, after** the K-loop —
not inside the sum.

## Structure

Grid is **identical to plain tiled matmul** ([[triton-tiled-matmul]]): 2-D, `cdiv(M,BLOCK_M) ×
cdiv(N,BLOCK_N)` programs. Pattern: **per-tile reduction with a fused epilogue** — everything between
the end of the K-loop and the store happens in registers.

## The HBM traffic — the whole point

| | Activation traffic |
|---|---|
| **Unfused** (write matmul → read+write bias → read+write ReLU) | **20MN** bytes (5 passes × 4MN) |
| **Fused** (one store) | **4MN** bytes |

A **5× reduction** on the activation tensor — which in a transformer linear layer is typically the
**largest tensor in the operation**, making this the single biggest available optimization.

**Launch overhead amplifies it.** Each unfused stage costs ~**5–10 µs** of launch latency *plus* its
HBM-bound execution. **Three launches vs one** means 3× the launch overhead, and on small `MN` that
overhead can exceed the actual data movement. Fusion recovers **both** bandwidth and launch latency in
one move.

**Concretely** — `torch.relu(A @ B + bias)` dispatches **three** kernels (cuBLAS matmul, elementwise bias
add, elementwise ReLU). Each elementwise stage is HBM-bound, moving `8MN` bytes (read+write). For a
`(4096,4096)×(4096,4096)` layer at 1 TB/s that's ~**134 MB → ~130 µs each**, against a matmul that takes
a few hundred µs at a good fraction of peak tensor-core throughput. **Fusion deletes both elementwise
stages outright.**

## Roofline

The matmul sets the floor: `BLOCK_M·BLOCK_N / 2(BLOCK_M+BLOCK_N)` = **16 FLOP/byte** at `(64,64)` →
compute-bound. The **bias** adds `BM·BN` FLOPs and only `4·BLOCK_N` bytes per program; the **ReLU** adds
`BM·BN` FLOPs and **zero** bytes. Both **raise** intensity slightly — more FLOPs on the same byte budget
— pushing further into compute-bound territory. (Epilogue ops are essentially free once you're already
holding the tile.)

## The broadcast is free

Triton follows NumPy semantics: a `(BLOCK_N,)` vector indexed `[None, :]` has shape `(1, BLOCK_N)` and
broadcasts against the `(BLOCK_M, BLOCK_N)` accumulator. The codegen does **not** materialize a
replicated `(BLOCK_M, BLOCK_N)` bias tile in registers — it emits an MMA-shaped add that reads the bias
**once per output column** and adds it to all rows, **identical in cost** to the add-into-accumulator
already running every K-loop iteration ([[broadcasting]]).

## The fusion boundary (what you *can't* fuse)

Fusion works only while the epilogue is **tile-local**. Once it needs **cross-tile or cross-row**
information — a row-wise softmax, a column-wise reduction, a LayerNorm needing the **full row's
variance** — the operation no longer fits inside one tile and can't be done in registers.

- **Fusable (tile-local):** bias add, scaling, dropout, ReLU/GELU, **residual add** against a same-shape
  tensor.
- **Not fusable directly:** anything requiring cross-tile state → a separate kernel, or a more elaborate
  **fused-attention-style** construction (the online/streaming trick from [[triton-fused-softmax]]).

## Variants (same structure, different epilogue window)

- **Matmul + residual add** — load a same-shape residual tile *after* the K-loop and add it to the
  accumulator before the activation; it **reuses the store-mask predicate** exactly.
- **Matmul + quantized output** — apply a scale and clamp to the accumulator and cast to **int8/fp8**
  before the store. This is the **production path for quantized transformer inference**
  ([[model-compression]]).

The canonical pattern is fixed; the variants differ only in **what runs in the epilogue window between
the K-loop and the store**.

## Pitfalls

- **Wrong operation order** — ReLU before bias gives `ReLU(AB) + bias`, a **different function**: any
  negative pre-bias entry is zeroed before the bias can rescue it, and a positive bias gets added to a
  clamped zero instead of a negative number. **Bias first, then ReLU.**
- **Wrong broadcast direction** — `bias[:, None]` broadcasts along the **column** axis, adding `bias[i]`
  instead of `bias[j]` to `out[i,j]`. It runs and can pass square/symmetric tests, failing everywhere
  else. The contract is **`bias[None, :]`**.
- **Missing the bias load mask** — the rightmost N-tile reads past the end of the bias vector when N
  isn't a multiple of `BLOCK_N`. Without `offs_n < N` and `other=0.0`, garbage contaminates the
  accumulator **before** the ReLU → wrong boundary outputs.
- **Loading the bias before the K-loop** — that extends its live range across the *entire* loop, raising
  **register pressure** and risking **accumulator spills**. Load it only **after** the K-loop, in the
  tight epilogue window just before the store.

---

Related: [[triton-tiled-matmul]], [[triton-relu]], [[triton-fused-softmax]], [[broadcasting]], [[model-compression]], [[model-serving]]
