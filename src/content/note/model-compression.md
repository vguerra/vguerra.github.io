---
title: "Model Compression — Pruning, Distillation, Quantization"
description: "pruning (magnitude/structured/gradient, structured→real speedup, Lottery Ticket), knowledge distillation (soft targets/dark knowledge, why student beats training-small-directly), quantization/mixed-precision (fp16 vanish/overflow, master fp32 + loss scaling, Kahan sum)"
category: "ML Systems / Production"
order: 78
updatedDate: "2026-09-10T21:13:12.788Z"
---
Making models smaller/faster/cheaper for deployment, ideally with minimal accuracy loss.

## Pruning

Remove or zero out **less-important** weights/neurons/filters. Benefits: fewer params (smaller model),
faster inference, sparsity, a mild **regularization** effect.
- **What to prune:** individual weights, neurons, conv **filters/channels**, whole layers.
- **How to choose:** **magnitude-based** (smallest |weight| — assumed least important); **structured**
  (whole channels/filters by L1 norm / average activation / sensitivity); **gradient/Hessian-based**
  (rank by impact on loss); learned pruning.
- **Structured vs unstructured:** structured pruning (whole filters/channels) gives **real runtime
  speedups**; unstructured (individual weights) needs **sparse-matrix support** to benefit.
- **Lottery Ticket Hypothesis:** a dense net contains a sparse subnetwork ("winning ticket") that, when
  retrained **from the original init**, matches the full net.

## Knowledge distillation

Train a small **student** to mimic a large **teacher**. Why distill from a big model rather than train
small directly: the small model lacks capacity to learn rich representations from scratch, but the
teacher provides **soft targets** carrying **"dark knowledge"** (class similarities), and approximating
the teacher's already-good function is an **easier** task than learning from labels alone → better
generalization + efficient deployment.

## Quantization & reduced precision

Use **fewer bits** (fp16/bf16/int8) → faster arithmetic, **half the memory** (2 B vs 4 B for params,
gradients, activations), more efficient data movement, bigger batches, lower energy.
- **Problems:** fp16's limited range → small gradients **vanish**, large values **overflow**; only
  ~10 mantissa bits → inaccurate updates.
- **Mixed-precision recipe:** compute in fp16, keep a **master fp32 copy** of weights for updates,
  **loss-scaling** during backward. Keep sensitive ops (softmax/exp/normalization) in fp32
  ([[tensor-dtypes]]).
- **Kahan summation** for summing many floats with minimal precision loss (vs naive `np.mean`).

---

Related: [[tensor-dtypes]], [[distributed-training]], [[model-serving]], [[numpy-basics]], [[training-diagnostics]]
