---
title: "Triton: Vector Add — Tile/Mask, Coalescing, Roofline"
description: "Triton tile-and-mask model via vector add: author-vs-compiler contract (+ warp latency hiding), 1-D grid / `program_id` / `offs`, `constexpr` BLOCK_SIZE, block-size↔program-count & launch-overhead (~5–10 μs fixed), tail mask (`offs < N`, correctness not perf), memory coalescing (128 B transactions, stride penalty, free in Triton), L2 incidental (runs at HBM speed), roofline & arithmetic intensity (vector add ~0.083 FLOP/byte memory-bound, ridge point ~10, matmul `O(BLOCK_K)` reuse → compute-bound), pitfalls (mask, constexpr, power-of-two, fusion removes HBM round-trips)"
category: "GPU / Kernels"
order: 80
updatedDate: "2026-09-10T21:05:08.399Z"
---
Elementwise vector add is the **canonical pointwise map**: every output element depends on exactly one
element of each input, nothing else. **No reduction, no cross-program communication, no shared memory,
no synchronization.** The whole kernel is **load → add → store** at the tile level, repeated across the
launch grid. Simple enough to expose the fundamentals: the tile-and-mask model, memory coalescing, and
the roofline.

---

## The compiler contract (author vs compiler)

Triton's pitch: the compiler hides the **mechanical, error-prone** parts of GPU programming while
leaving the parts that need **kernel-level intent** to the author.

| **Author chooses** | **Compiler handles** |
|---|---|
| grid shape `cdiv(N, BLOCK_SIZE)` | lowering tile loads/stores to wide PTX of the right vector width |
| `constexpr` block size | register allocation for the tile |
| offset arithmetic (`offs`) | how the tile is shared across warps inside a program |
| mask predicate (`offs < N`) | inserting pipelining |
| the entry-function signature | emitting the actual machine code |

The author **never** names a warp, writes a coalescing rule, declares shared memory, or inserts a sync
barrier. **Latency hiding is automatic:** different warps load different HBM slices in parallel, and
while one warp is stalled on a load, others issue arithmetic — the hardware's SMT hides memory latency
with no author effort.

---

## Program decomposition

- **1-D launch grid** of `cdiv(N, BLOCK_SIZE)` programs. The grid is a **callable of the meta-params**:
  `grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)` — so it adapts when `BLOCK_SIZE` is
  autotuned.
- Each program is identified by **`tl.program_id(0) ∈ [0, cdiv(N, BLOCK_SIZE))`** (half-open: `0 …
  cdiv−1`) and owns **one contiguous tile** of `BLOCK_SIZE` consecutive elements.
- Program `p` starts at `p·BLOCK_SIZE`; the lane offsets are
  **`offs = p*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)`**.
- **Embarrassingly parallel:** no two programs touch the same element → any order, concurrent, **no
  barriers, no atomics**.

---

## Shape & masking

Tile shapes are **compile-time constants** (`tl.constexpr`). Standard 1-D choice **`BLOCK_SIZE = 1024`**:
power of two (wide vector loads), large enough to amortize per-program launch overhead, small enough to
keep the **register footprint** within what the hardware exposes. The compiler uses the constexpr to
size registers, unroll the inner sequence, and emit the right PTX vector instructions.

**Block size ↔ program count & launch overhead.** Smaller block → **more** programs; larger block →
**fewer** programs (and wider vector loads). Fewer/larger programs help when **per-program overhead is
large relative to per-program work**. Launch cost is ~**5–10 μs** — but note it's a **fixed
per-kernel-launch** cost that *doesn't shrink* with fewer programs; what a bigger block actually buys is
**wider loads + per-program dispatch amortized over more work**. When it matters: a 1M-element fp32
vector moves `12 MB` through HBM (2 reads + 1 write) → at 1 TB/s that's ~**12 μs**, *comparable* to the
5–10 μs launch cost. So **small `N` is overhead-dominated** — exactly the regime where a bigger
`BLOCK_SIZE` (and not launching a swarm of tiny programs) pays back.

**The tail mask.** `BLOCK_SIZE` is fixed at compile time but `N` is runtime, so the **last program
overshoots**. E.g. `N=1,000,000`, `BLOCK_SIZE=1024` → `cdiv = 977` programs; the last (`p=976`) starts
at `999,424` and covers `999,424 … 1,000,447` — the final **448 lanes are past the end**. The mask
**`offs < N`** disables out-of-range lanes on every `tl.load` (don't read garbage) and `tl.store` (don't
write past the buffer).

> **The mask is a correctness tool, not a performance one.** Without it the kernel touches memory it
> doesn't own → undefined behaviour. (It may *appear* to work if the buffer happens to be padded — a
> classic silent bug.)

---

## Memory hierarchy & coalescing

Vector add has the **simplest possible memory pattern**: each input element loaded from **HBM exactly
once**, each output stored **exactly once**, **nothing reused**. So:

- **No shared memory / SRAM needed** — the tile is consumed and discarded; no other program wants those
  bytes. The compiler keeps operands **in registers** between load, add, store.
- **The one thing it needs: coalesced HBM access.** Contiguous lane offsets (`p·BLOCK_SIZE + 0,1,2,…`)
  let the compiler emit a **few wide memory transactions** instead of `BLOCK_SIZE` separate ones — the
  difference between **peak HBM bandwidth and a fraction of it**.
- **Coalescing is free in Triton.** Unlike CUDA (where you hand-write coalescing rules), you express a
  **contiguous tile** and trust the compiler to lower the load to the right vector width.

**Transaction arithmetic.** Accelerators serve HBM in **32 B or 128 B transactions**. A `BLOCK_SIZE=1024`
fp32 tile = `4096 B` per tensor = **32 transactions of 128 B** when perfectly coalesced. If access were
**strided by 4**, the same 1024 logical loads cost **4× the transactions** (each carries useful data for
only 1 of 4 lanes).

**L2 is incidental here.** No reuse → no two programs request the same line → a line L2 happens to
capture is never queried again. L2 isn't hurting (free to have) but isn't helping — **the kernel runs at
HBM speed, not L2 speed.** Any benchmark reporting **faster than peak HBM** is almost certainly measuring
a **warm cache** or a tensor small enough to fit entirely in L2.

---

## Memory-bound vs compute-bound (the roofline)

Per output element: read `4 + 4 = 8 B` (two fp32), write `4 B`, do **1 add**. So **arithmetic
intensity**:

$$\text{AI} = \frac{1\ \text{FLOP}}{12\ \text{bytes}} \approx 0.083\ \text{FLOP/byte}$$

**Roofline:** `performance = min(peak_FLOPs, AI × peak_bandwidth)`. The **ridge point** (where the two
meet) is `peak_FLOPs / peak_bandwidth` ≈ **~10 FLOP/byte** on modern accelerators (tens of TFLOP/s ÷ a
few TB/s). Vector add at `0.083` sits **far left of the ridge → firmly memory-bound.** The only
optimizations that change its runtime affect **bandwidth**: coalescing, enough programs to saturate the
SMs, and a block size large enough that launch overhead is negligible. (Tuning knobs — `@triton.autotune`
over `BLOCK_SIZE`, plus `num_warps` / `num_stages` — affect occupancy / latency-hiding, not the
intensity.)

**Where other kernels sit on the roofline:**

| Kernel | Arithmetic intensity | Regime |
|---|---|---|
| **Vector add** | ~0.083 FLOP/byte | memory-bound |
| **Pointwise activations** (ReLU/GELU/SiLU) | ~0.1 | memory-bound |
| **Fused softmax** (per-row: max, exp, normalize) | ~0.3 | memory-bound |
| **Tiled matmul** | **O(BLOCK_K)** — tunable | compute-bound once BLOCK_K ~ tens |

The matmul is the interesting case: each loaded operand tile of depth `BLOCK_K` is **reused** across
`BLOCK_M`/`BLOCK_N` output lanes, so **intensity scales with `BLOCK_K`** — data reuse is what pushes it
across the ridge into compute-bound territory. (This is the same "reuse → arithmetic intensity" idea
behind im2col-as-GEMM in [[convolution]].)

---

## Pitfalls

- **Forgetting the tail mask** — may *appear* to work if the buffer is padded, but in general the kernel
  reads garbage and writes past the output. Always mask `offs < N` on load *and* store.
- **Block size not `tl.constexpr`** — a *runtime* block size prevents the compiler from sizing
  registers, unrolling the load-add-store sequence, and picking the vector width. It still compiles but
  **loses most of its speed** and may fall back to a **scalar loop**.
- **Non-power-of-two block size** — the compiler vectorizes cleanly only for powers of two; other values
  force **scalar fallbacks** for partial vectors. Standard: **128 / 256 / 512 / 1024 / 2048 / 4096**.
- **Treating the kernel as compute-bound** — vector add is at the **roofline floor**, so adding *more*
  arithmetic per element is **essentially free** up to a point (HBM traffic dominates). This is the
  motivation for **fusion**: fused add+ReLU, fused bias+activation, fused FMA all pay for themselves by
  **removing extra round-trips through HBM**, *not* by doing arithmetic faster. **The lever on a
  memory-bound kernel is bytes moved, not FLOPs done.**

---

Related: [[convolution]], [[activation-functions]], [[self-attention]], [[scaling-laws]],
[[tensor-memory-layout]]
