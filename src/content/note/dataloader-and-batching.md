---
title: "DataLoader & Batching"
description: "`DataLoader` params (batch_size, shuffle & gradient bias, drop_last, num_workers, pin_memory), throughput vs statistics knobs, keeping the GPU fed"
category: "Training Dynamics & Optimization"
order: 16
updatedDate: "2026-08-19T15:20:22.836Z"
---
The data-feeding half of the training loop. Key `torch.utils.data.DataLoader` parameters and *why*
each one matters.

---

## Key parameters

- **`batch_size`** — how many samples per optimization step. Affects **gradient variance** (bigger
  batch → lower-variance gradient estimate), **memory** (activation memory scales with it, see
  [[training-memory]]), and **training dynamics** (bigger batch often needs a larger LR / warmup).
- **`shuffle=True`** — **critical for SGD convergence.** Without shuffling, each batch's gradient is a
  **biased** estimate (consecutive samples may share structure — sorted labels, temporal order), and
  the model sees the same batch sequence every epoch. Shuffling makes each batch **roughly represent
  the full dataset**, so the mini-batch gradient is an unbiased estimate of the full gradient. Shuffle
  the **training** set; leave validation/test unshuffled (order doesn't matter for a metric).
- **`drop_last=True`** — discard the final **incomplete** batch. Useful for **consistent BatchNorm**
  behavior (a size-1 tail batch breaks BN — see [[normalization]]) and **clean loss averaging** (every
  batch has equal weight). Costs you a few samples per epoch.
- **`num_workers`** — number of parallel **data-loading subprocesses**. Loading/augmenting on the CPU
  in parallel keeps the **GPU from starving** while it waits for the next batch. Too high → CPU/RAM
  contention; tune it (often 4–8 per GPU).
- **`pin_memory=True`** — stages tensors in **CUDA pinned (page-locked) memory**, enabling
  **asynchronous** host→device transfers (overlap copy with compute via `.to(device,
  non_blocking=True)`). Details in [[tensor-devices]].

---

## The mental model

The DataLoader exists to **keep the GPU fed**. Training throughput is bottlenecked whenever the GPU
finishes a step before the next batch is ready — so `num_workers` (parallel loading) + `pin_memory`
(async transfer) are about **hiding data-loading latency behind compute**, while `batch_size` /
`shuffle` / `drop_last` are about **gradient quality and consistency**.

- **Throughput knobs:** `num_workers`, `pin_memory`, `prefetch_factor`.
- **Statistics knobs:** `batch_size`, `shuffle`, `drop_last`.

If GPU utilization is low and spiky, suspect the **input pipeline** (raise `num_workers`, enable
`pin_memory`) before blaming the model — a classic profiling finding (see [[training-diagnostics]]).

---

Related: [[pytorch-training-loop]], [[training-memory]], [[normalization]], [[tensor-devices]],
[[learning-rate]]
