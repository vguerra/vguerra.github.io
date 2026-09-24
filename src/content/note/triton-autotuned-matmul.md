---
title: "Triton: Autotuned Matmul"
description: "`@triton.autotune` over small config space (compile+bench all on first call, cache winner by shape); **callable grid** lambda over meta-dict (block sizes unknown until sweep), config = tile+num_warps+num_stages, mask discipline makes search robust, register-pressure binding constraint (spills), shape-sensitivity (tile coverage + wave quantization), pitfalls (space-too-wide hangs, static-grid NameError, missing kwargs → re-tune, per-process cache)"
category: "GPU / Kernels"
order: 81
updatedDate: "2026-09-23T14:21:12.898Z"
---
Same tiled matmul kernel ([[triton-tiled-matmul]]) — the **only** change is a `@triton.autotune`
decorator over a small config space; the kernel body is untouched. The **optimal tile shape shifts with
the input dimensions**: a `(64,64,32)` tile excellent on a `(4096,4096,4096)` square matmul leaves
performance on the floor for a `(128,8192,8192)` skinny-tall problem, and vice versa — an order of
magnitude apart.

## How autotune works

On the **first call for a given signature**, Triton **compiles every config, benchmarks each on the
actual inputs, caches the winner keyed by problem shape**, and reuses it for matching subsequent calls.
It does **not** change the parallel decomposition — it changes which `BLOCK_M`/`BLOCK_N` fill the grid,
hence **how many programs are in flight**.

**The grid must be a callable, not a tuple.** Because `BLOCK_M`/`BLOCK_N` aren't known until the sweep
picks a config, the launcher passes `grid = lambda meta: (cdiv(M, meta['BLOCK_M']), cdiv(N,
meta['BLOCK_N']))`. Triton invokes the lambda **after** the sweep with the winner's block sizes
substituted in. The grid is a **function of the meta-dict**, not a closed-over Python value.

Each `triton.Config` pins `(BLOCK_M, BLOCK_N, BLOCK_K)` + `num_warps` + `num_stages`. A representative
small space:
- `(64,64,32)` — balanced square
- `(128,64,32)` — tall-skinny (wider M tile)
- `(64,128,32)` — short-fat

**Keep it to ~3–5 configs** inside a test harness — the first call pays full compile+benchmark for
*every* candidate.

**Mask discipline makes the search robust:** configs where `BLOCK_K > K` are still valid (the K mask
zeros out-of-range columns → one K-loop iteration); configs where `BLOCK_M/N > M/N` are valid (the store
mask discards the overshoot). So the same masks that make the plain kernel correct let the autotuner
safely measure wildly mismatched block shapes and rank them by wall time.

## What the knobs trade

- **Larger `BLOCK_M`/`BLOCK_N`** → more register footprint per program, higher arithmetic intensity
  (reuse scales with tile size — an A slab is reused `BLOCK_N` times, a B slab `BLOCK_M` times).
- **Larger `BLOCK_K`** → more SRAM staging + longer per-iteration MMA.
- **`num_stages`** → software-pipeline depth: at depth `s`, load `s−1` iterations ahead of the current
  `tl.dot`, double/triple-buffering the SRAM staging to hide HBM latency. Multiplies SRAM by depth.
- **`num_warps`** → how the tile is sharded across warps: more warps → thinner per warp, higher
  occupancy; fewer → concentrated, less intra-program coordination.

Both `num_warps`/`num_stages` trade **resources (registers, SRAM) for latency hiding**, and the balance
is shape-dependent.

**Register pressure is the binding constraint at the top of the space.** A `(128,128)` fp32 accumulator
is `128·128·4 = 65,536 B` per program — a large fraction of an SM's register file. Overshoot → **spill to
local memory** → throughput collapses. The autotuner just *measures* the collapse and ranks that config
last — it doesn't reason about it analytically.

## Why the winner is shape-sensitive

Autotune doesn't move the kernel across the roofline; it moves it **closer to the compute ceiling** by
maximizing tensor-core utilization for the specific `(M,N,K)`. Two effects:
- **Tile coverage** — when the matrix is small enough that the tile count is comparable to the SM count.
  A `(64,64)` block on a `(128,128)` output gives **4 tiles** — far too few to saturate 100+ SMs — so a
  `(32,32)` block giving **16 tiles** wins *despite worse per-tile reuse*.
- **Wave quantization** — when the tile count is just over a multiple of the SM count, a small extra
  "wave" of programs runs behind the main wave's tail. The winner is the config whose tile count is
  closest to a clean multiple of the SM count. (Same tile/wave-quantization idea as picking batch/hidden
  dims that are multiples of the tensor-core tile.)

## Cache & production

The autotune cache is **per-process** (in process memory) — a fresh Python process repeats the full
sweep. **Persistent caches** (`cache_results` / env vars pinning a config) serialize the winner across
restarts. **`prune_configs_by`** drops obviously dominated configs (e.g. `BLOCK_K > K`) *before*
benchmarking, cutting first-call cost without losing coverage. You can also gate per-accelerator configs
by including them and letting benchmarking choose.

## Pitfalls

- **Search space too wide for the budget** — 20+ configs on a 60 s subprocess timeout exhausts the time
  *compiling*, never benchmarks, and the first call appears to **hang**. Keep 3–5 in-harness; use a
  separate **offline tuning pass** for larger spaces.
- **Static grid tuple instead of a callable** — `grid = (cdiv(M, BLOCK_M), …)` at the call site raises
  `NameError` (`BLOCK_M` isn't resolved yet). The grid **must** be a lambda over the meta-dict.
- **Missing key arguments** — if `M`/`N`/`K` aren't passed as kwargs, the autotuner can't match the
  cached entry by name → **every call looks like a fresh shape → re-tunes**. Pass explicit `M=M, N=N,
  K=K`.
- **Per-process cache surprise** — a harness spawning a subprocess per test pays the full sweep **every
  test** even when the shape repeats. Use a persistent cache file or warm the cache once in a long-lived
  process.

---

Related: [[triton-tiled-matmul]], [[triton-vector-add]], [[triton-fused-softmax]], [[tensor-dtypes]]
