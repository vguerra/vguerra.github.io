---
title: "Modern GPU Programming for MLSys — Reading Index"
description: "reading index for the book: chapter map, note names, and how it connects to the rest of these notes"
category: "GPU — Hardware & Execution Model"
order: 39
updatedDate: "2026-08-17T20:24:50.784Z"
---
Notes taken while working through
[Modern GPU Programming for MLSys](https://mlc.ai/modern-gpu-programming-for-mlsys/)
(MLC Community). The book goes from hardware, to programming model, to complete
state-of-the-art kernels, targeting the **NVIDIA Blackwell** architecture and using
the **TIRx** Python DSL for runnable examples.

Each chapter gets its own note. Unlinked entries below are chapters not written up yet.

## Part I — Understanding the GPU

| Ch | Topic | Note |
|---|---|---|
| 1 | GPU execution model | `gpu-execution-model` |
| 2 | What makes a kernel fast | `gpu-kernel-performance` |
| 3 | Data layout and its notation | `gpu-data-layout` |
| 4 | Evolution of tensor core data layouts | `gpu-tensor-core-layouts` |
| 5 | Async data movement: TMA | `gpu-tma-async-copy` |
| 6 | Blackwell tensor core: `tcgen05.mma` | `gpu-blackwell-tensor-core` |
| 7 | Tensor memory (TMEM) | `gpu-tensor-memory-tmem` |
| 8 | Async coordination: mbarrier | `gpu-mbarrier` |
| 9 | Advanced scheduling: cluster launch control | `gpu-cluster-launch-control` |

## Part II — TIRx

| Ch | Topic | Note |
|---|---|---|
| 1 | Introduction to TIRx | `tirx-basics` |
| 2 | TIRx layout API | `tirx-layout-api` |

## Part III — GEMM: tiled to SOTA

| Ch | Topic | Note |
|---|---|---|
| 1 | Building a tiled GEMM | `gpu-gemm-tiled` |
| 2 | Pipelining GEMM with TMA | `gpu-gemm-pipelining` |
| 3 | Scaling GEMM with warp specialization and clusters | `gpu-gemm-warp-specialization` |

## Part IV — Flash Attention 4

| Ch | Topic | Note |
|---|---|---|
| 1 | Flash Attention 4 | `flash-attention-4` |

## Reference

Language reference, compiler internals, and debugging warp-specialized kernels —
`gpu-debugging-kernels` for anything worth keeping.

## Why this book connects to the rest of these notes

The kernels it builds are the ones underneath everyday training and inference:
GEMM is the workhorse of [[transformer-architecture]], and Part IV is the fused
implementation of [[self-attention]]. The memory-layout material extends the
host-side view in [[tensor-memory-layout]] down to what the hardware actually does.
