---
title: "Triton: Grouped Program-ID Matmul (L2-Friendly Schedule)"
description: "grouped program-id remap for L2 reuse: why launch order matters (wave working set), 1-D grid → `(pid_m,pid_n)` mapping (group, `group_size_m`, walk down a tile column), the last-group divisor bug (constant `GROUP_SIZE_M` vs actual `group_size_m`) and the enumeration hand-check, `% M`/`% N` wrap trick"
category: "GPU / Kernels"
order: 85
updatedDate: "2026-10-16T13:02:29.743Z"
---
The tiled matmul ([[triton-tiled-matmul]]) is correct under any order in which output tiles are scheduled, but the **order determines cache reuse**. The standard fix is a **grouped program-id remap**, also called super-grouping.

## Why the order matters

Programs with neighbouring ids run at about the same time. A tile `(pid_m, pid_n)` needs the A row-slab for `pid_m` and the B column-slab for `pid_n`. With a **row-major** order over the tile grid, the first wave of programs shares one A slab but touches a *different* B slab each time, so across a wave the working set of B is large and L2 reuse is poor. With **grouped ordering**, a wave covers a compact block of `GROUP_SIZE_M` tile rows by several tile columns, so the same few A and B slabs serve many programs. (The Triton tutorial's illustration: for a 9×9 tile grid the first 9 programs load about 90 operand blocks in row-major order versus about 54 grouped. Treat the exact numbers as illustrative.) The arithmetic is unchanged; only **where the data comes from** (L2 instead of HBM) changes.

## The remap

The grid stays **1-D** with `cdiv(M,BLOCK_M) * cdiv(N,BLOCK_N)` programs. Each program converts its linear `pid` into `(pid_m, pid_n)` with these steps:

1. `num_pids_in_group = GROUP_SIZE_M * num_pids_n`: a group is `GROUP_SIZE_M` tile **rows** by all tile columns.
2. `group_id = pid // num_pids_in_group`, `first_pid_m = group_id * GROUP_SIZE_M`.
3. `group_size_m = min(num_pids_m - first_pid_m, GROUP_SIZE_M)`: the **last group may be smaller**.
4. Within the group, with `r = pid % num_pids_in_group`: `pid_m = first_pid_m + r % group_size_m` and `pid_n = r // group_size_m`.

Programs within a group walk **down a tile column first** (all rows of the group), then step to the next column. So consecutive programs share the same B slab and a small set of A slabs.

## The bug that hides until the edge

Using the constant `GROUP_SIZE_M` instead of the **actual `group_size_m`** as the divisor in `pid_n`. When every group is full they are equal and everything passes. When `num_pids_m` isn't a multiple of `GROUP_SIZE_M`, the last group is smaller and the mapping is no longer a bijection: some tiles are computed twice and others **never**.

**Hand check (the enumeration exercise):** `num_pids_m = 3`, `num_pids_n = 2`, `GROUP_SIZE_M = 2`. Pids 0-3 form a full group. For the last group, `group_size_m = 1` and pids 4 and 5 should give tiles `(2,0)` and `(2,1)`. With the constant divisor they both give `(2,0)` and tile `(2,1)` is never written. Because the loads wrap with `%` and the store is masked, **nothing crashes**; part of the output is just wrong. Always list `pid → (pid_m, pid_n)` for a small grid and confirm each tile appears exactly once.

**Tests that expose it:** use sizes where `cdiv(M, BLOCK_M)` is **not** a multiple of `GROUP_SIZE_M`, and a case where it is smaller than the group size (a single partial group). Compare the full output against `A @ B`.

## The `% M` / `% N` wraparound trick

The load offsets for rows and columns are taken modulo `M` and `N`, so edge tiles read valid (wrapped) memory with **no row/column load mask**; only the K dimension needs a mask. The **store** uses the unwrapped offsets with a `(row < M) & (col < N)` mask, so wrapped garbage is discarded. Corollary: a wrong tile mapping computes the wrong block silently, since nothing goes out of bounds.

## Other points

- Everything else is the standard tiled matmul: fp32 accumulator, K-loop with a K mask, `tl.dot` with an explicit `input_precision` for fp32 inputs.
- `GROUP_SIZE_M` is a tunable alongside the block sizes and `num_stages` ([[triton-autotuned-matmul]]); typical values are around 8.
- The benefit grows with matrix size, when the operands don't fit in L2 and consecutive programs would otherwise thrash it.

---

Related: [[triton-tiled-matmul]], [[triton-autotuned-matmul]], [[triton-debugging]], [[triton-perf-tips]]
