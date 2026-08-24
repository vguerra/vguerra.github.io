---
title: "Memory Management During Training"
description: "where GPU memory goes (params/gradients/optimizer-state Adam×4/activations/graph), 16 bytes-per-param rule, the accumulate-loss-tensor OOM pitfall (`.item()`/`.detach()`), levers to reduce memory"
category: "Training Dynamics & Optimization"
order: 18
updatedDate: "2026-08-19T15:20:37.833Z"
---
Where GPU memory goes during training, and the graph-lifetime pitfall that silently OOMs you.

---

## The major consumers

For a model with `N` parameters:

| Consumer | Size | Notes |
|---|---|---|
| **Parameters** | `N` × (4 bytes fp32 / 2 bytes fp16/bf16) | the weights themselves |
| **Gradients** | same as parameters (`N`) | `.grad` for every param → roughly **doubles** param memory |
| **Optimizer state** | **Adam:** 2×`N` (1st + 2nd moment); **SGD+momentum:** 1×`N` | Adam ⇒ params + grads + 2 moments ≈ **4× param memory** (the "×3 extra" over the weights) |
| **Activations** | scales with `batch × depth × width × seq_len` | intermediates cached for backward (see [[autograd-and-autodiff]]); **often the dominant term** |
| **Graph metadata** | proportional to live activations | while the graph exists, all its intermediate tensors stay pinned |

**Rule of thumb (Adam, fp32):** parameters + gradients + optimizer state ≈ **16 bytes/param** before
activations even enter — `4 (param) + 4 (grad) + 8 (Adam moments)`. Activations are then added on top
and are what actually scales with batch size.

---

## The graph-lifetime pitfall — accumulating loss tensors

**The bug:** summing loss *tensors* across batches without extracting the scalar:

```python
total_loss += loss          # ⚠️ loss still carries grad_fn → keeps its WHOLE graph alive
```

Because `loss` is a graph node, holding a reference keeps **its entire computation graph** (and all
cached activations) from being freed. Over many batches these chain together → memory grows every
step → **OOM that appears partway through the epoch** (a strong tell that *something is accumulating*).

**The fix — sever the graph by extracting a Python float:**

```python
total_loss += loss.item()        # scalar → no graph reference
# or, if you need a tensor: loss.detach()
```

`.item()` (or `.detach()`) cuts the link so the graph can be garbage-collected after each backward.
Same principle as not stashing raw `output` in a forward hook ([[pytorch-hooks]]).

---

## Levers to reduce memory

- **`torch.no_grad()`** for validation/inference — no graph built, activations freed immediately
  (see [[pytorch-training-loop]]).
- **Gradient checkpointing** — recompute activations in backward instead of storing them; ~√N stored
  instead of N, ~33% extra compute ([[autograd-and-autodiff]]).
- **Lower precision** (fp16/bf16) — halves parameter/gradient/activation bytes (see [[tensor-dtypes]]).
- **Smaller batch / gradient accumulation** — trade per-step activation memory for more steps.
- **`.item()` / `.detach()`** on anything you log or accumulate — never hold graph nodes you don't
  need.

---

Related: [[autograd-and-autodiff]], [[pytorch-training-loop]], [[pytorch-hooks]], [[tensor-dtypes]],
[[dataloader-and-batching]]
