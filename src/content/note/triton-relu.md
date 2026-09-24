---
title: "Triton: ReLU — Branchless Clamp, Fusion, Tile-Level Ops"
description: "Triton ReLU: branchless clamp `tl.maximum(x,0.0)` vs warp divergence from `if`, `tl.where` bool-tile cost, `0.0`-literal dtype gotcha, mask protects the store, roofline (0.125 ops/byte → fused not standalone), cross-kernel intensity ladder, `num_warps`, tile-level-ops philosophy"
category: "GPU / Kernels"
order: 90
updatedDate: "2026-09-12T21:48:38.760Z"
---
ReLU is a pure pointwise map — no cross-lane dependency, no reduction, no shared memory. Same
tile/mask/coalescing skeleton as [[triton-vector-add]]; this note focuses on what's **ReLU-specific**:
the **branchless clamp**, why it's almost never its own kernel, and the tile-level-op philosophy.

## The branchless clamp (the key idea)

Express the clamp as a **tile-level max**, not a per-lane conditional:

```python
out = tl.maximum(x, 0.0)          # ✅ one FMAX per lane, branchless
# out = tl.where(x > 0, x, 0.0)   # ⚠️ semantically identical but materializes a bool tile
```

**Why not an `if`:** a naive CUDA `if (x>0) out=x; else out=0` triggers **warp divergence** — all 32
lanes of a warp issue **both** sides of the branch, mask off the lanes that didn't take each path, and
**serialize** the two paths in time. Even when one side is trivial, the divergence machinery has
overhead. `tl.maximum` lowers to a **single FMAX per lane** — no branch, no divergence. A Python `if x >
0:` on a tile is either rejected by the compiler or rewritten to `tl.where` anyway (no benefit).

**Tile-level primitives are the language the compiler wants:** `tl.maximum/minimum/where/exp/sigmoid`
each lower to one straight-line instruction per lane, and the compiler **fuses** them into long
FMA-and-FMAX sequences. Per-lane Python conditionals break that composition.

## The `0.0` literal gotcha

`tl.maximum(x, 0)` with an **integer** literal can **promote the tile to int** and silently corrupt the
output. Use `tl.maximum(x, 0.0)` — the **float** literal is required for an fp32 input.

## Mask discipline (even when arithmetic is harmless)

For ReLU an unmasked over-read would feed **garbage into the clamp harmlessly** — but the matching
`tl.store` would **write garbage past the output buffer**. So the mask **primarily protects the store**;
read-protection on `tl.load` is the second-order benefit. Mask `offs < N` on **both** load and store,
always. (E.g. `N=257` or `N=1025` overshoot `BLOCK_SIZE=1024` in the last program → scrambled output
without the mask.)

## Roofline — why ReLU is fused, not standalone

Per output element: read 4 B, one `max`, write 4 B → **arithmetic intensity = 1 op / 8 B = 0.125
ops/byte** → firmly **memory-bound**. A 1M-element fp32 ReLU moves **8 MB** through HBM → ~**8 µs** at
1 TB/s — **comparable to kernel-launch overhead**. So ReLU is **almost never its own kernel**: it's
**fused** into the preceding matmul or the following add, paying for itself by **removing an HBM
round-trip** (the fusion principle from [[triton-vector-add]]).

**Cross-kernel intensity ladder** (all memory-bound; runtime tracks HBM traffic ~linearly):

| Kernel | Intensity (ops or FLOPs / byte) |
|---|---|
| Vector add | 0.083 |
| ReLU | 0.125 |
| FMA | 0.17 |
| SiLU | ~0.4 |
| GELU (erf counted) | ~0.4–0.6 |

## Author vs compiler (ReLU specifics)

**Author:** grid shape, `constexpr` block size, offset arithmetic, mask predicate, and the **form of the
clamp** (`tl.maximum`, not an `if`). **Compiler:** wide vector loads, register allocation, and sharing
the 1024-lane tile across **`num_warps` (default 4)** in groups of **256 lanes** — while one warp waits
on an HBM load, the others issue their FMAX, hiding latency. The author never names a warp, never calls
`__syncthreads`, never declares shared memory.

## Pitfalls

- **Python `if` on a tile** → rejected or rewritten to `tl.where`, no benefit. Use `tl.maximum(x, 0.0)`.
- **Integer literal `0`** → dtype promotion to int, corrupt output. Use `0.0`.
- **Missing tail mask** → writes past the output buffer (test sees scrambled bytes).
- **Non-`constexpr` block size** → no register sizing / unrolling / wide loads → loses bandwidth
  headroom.

---

Related: [[triton-vector-add]], [[activation-functions]], [[convolution]], [[self-attention]]
