---
title: "Distributed Training (Multi-GPU)"
description: "data parallelism & effective batch size (linear/√k LR scaling), sync vs async SGD (straggler vs staleness), three parallelisms (data/pipeline/tensor + 3-D), gradient checkpointing & micro-batching, DDP vs DataParallel, DistributedSampler sharding, SyncBatchNorm, optimizer-state consistency"
category: "ML Systems / Production"
order: 77
updatedDate: "2026-09-10T21:26:22.532Z"
---
Scaling training across GPUs — how gradients sync, and the pitfalls of going from 1 → many GPUs.

## Data parallelism & effective batch size

With data parallelism, **global batch = per-GPU batch × #GPUs**. A per-GPU batch that works on 1 GPU may
give a **too-large global batch** on 8 GPUs → worse generalization / divergence.
- **Fix:** scale LR with global batch (**linear scaling rule**: batch ×k → LR ×k; or **√k** for Adam,
  see [[lr-schedulers]]), *or* keep global batch constant by reducing per-GPU batch. Pair large-batch
  with **warmup**.

## Synchronous vs asynchronous SGD

| | **Sync SGD** | **Async SGD** |
|---|---|---|
| Gradients | computed on different batches, **averaged**, then one update | each worker updates **independently**, no wait |
| Barrier | yes (after each step) | no |
| Consistency | consistent gradients, stable/predictable | may use **stale** weights → slower/diverges |
| Weakness | **one slow worker stalls everyone** (straggler) | robust to slow nodes |

Async is used in parameter-server / edge setups; sync (all-reduce) is the modern default for tightly-coupled GPUs.

## The three kinds of parallelism

| Kind | What's split | Requirement / cost |
|---|---|---|
| **Data parallelism** | the **batch** — each node has a **full model copy**, runs a subset of the batch; gradients aggregated (all-reduce) & redistributed (synchronous) | model must fit in **each** node's memory |
| **Pipeline (model) parallelism** | **different layers** on different nodes; node 1 does the first layers → passes activations to node 2, etc.; gradients flow back in reverse | model needn't fit on one node, but nodes sit **idle** waiting (pipeline bubble) — mitigated by micro-batching |
| **Tensor (model) parallelism** | the computation **within a single layer** split across nodes | for layers too big for one node; heavy communication |

Real large-model training combines all three (**3-D parallelism**).

## Fitting big models in memory

- **Gradient checkpointing** — store activations only every N layers on the forward pass; **recompute**
  the missing ones during backward (trade compute for memory — [[autograd-and-autodiff]], [[training-memory]]).
- **Micro-batching** — subdivide a batch into smaller parts, **accumulate** their gradients before one
  update (= gradient accumulation; also fills the pipeline-parallel bubble).

## Common multi-GPU pitfalls

- **Improper gradient sync** — use **`DistributedDataParallel` (DDP)**, *not* `DataParallel`; ensure
  `.backward()`/`.step()` are called correctly; don't mix manual and auto grad sync. DDP all-reduces
  gradients across GPUs so updates stay consistent.
- **Data sharding** — each GPU must see a **different** shard, else the model overfits / converges
  poorly. Use **`DistributedSampler`** in the DataLoader ([[dataloader-and-batching]]) + shuffle.
- **BatchNorm** — computes stats **per-GPU** by default → broken with small per-GPU batches (< 8).
  Replace with **GroupNorm/LayerNorm** ([[normalization]]) or use **SyncBatchNorm** (expensive).
- **Optimizer state** — Adam's per-parameter state must stay consistent across GPUs; watch broken
  gradient scaling in mixed precision.

---

Related: [[lr-schedulers]], [[dataloader-and-batching]], [[normalization]], [[pytorch-training-loop]], [[model-compression]]
