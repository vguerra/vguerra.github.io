---
title: "Autograd & Automatic Differentiation"
description: "dynamic (define-by-run) vs static graphs, `grad_fn`; reverse-mode autodiff; forward-vs-reverse choice rule (n≫m → reverse, Jacobian column/row framing); the memory tradeoff (must cache activations → `no_grad` frees, gradient checkpointing recomputes); backward ≈ 2× forward → `C≈6ND`"
category: "PyTorch — Tensors & Mechanics"
order: 8
updatedDate: "2026-08-19T15:12:12.458Z"
---
How PyTorch computes gradients: the dynamic graph, reverse-mode autodiff, the forward-vs-reverse
choice rule, and the memory tradeoff that shapes large-model training.

---

## Dynamic (define-by-run) computation graph

PyTorch uses a **dynamic** graph: the graph is **constructed anew during each forward pass**, so it
can **change from iteration to iteration**. This allows ordinary Python **control flow** (if/loops,
data-dependent branching) *inside* models — the graph is just whatever ops actually ran.

- Contrast: **static** graphs (define-then-run, e.g. TF1) build the graph once, then execute it
  repeatedly — faster to optimize/deploy, but rigid and harder to debug.
- Each node stores a reference to a **`grad_fn`** — the function that computes the node's **local
  gradient** during backward. (Leaf tensors created by the user have `grad_fn = None`.)
- The graph is **freed after `backward()`** unless you pass `retain_graph=True`.

---

## Backpropagation = chain rule via reverse-mode autodiff

Backprop is the chain rule applied systematically. **"Reverse-mode"** = the **direction of traversal**:
gradients flow from the scalar output **backward** through the graph to the parameters.

**Key property:** it computes the gradient of **one scalar output w.r.t. many parameters in a single
pass**. Forward-mode would need **one pass per parameter** — prohibitive for models with millions of
params.

---

## Forward-mode vs reverse-mode: the choice rule

Think of differentiating a map `f : ℝⁿ → ℝᵐ`, whose derivative is the **m×n Jacobian**:

| Mode | Builds the Jacobian | Cost scales with | Wins when |
|---|---|---|---|
| **Forward** | one **column** at a time (push one *input* direction through) | **n** (inputs) | `m ≫ n` — few inputs, many outputs |
| **Reverse** | one **row** at a time (pull one *output* direction back) | **m** (outputs) | `n ≫ m` — many inputs, few outputs |

Pick the mode that matches the **cheaper dimension**. A neural net is the extreme `n ≫ m` case:
**millions of parameters** (inputs to the loss), **one scalar loss** (`m = 1`) → reverse-mode is
*maximally* favorable — **one** backward pass yields every gradient. Forward-mode would need a million
passes.

---

## The tradeoff: reverse-mode costs memory

Reverse-mode is **not free**. To apply the chain rule at each node, backward needs that node's **local
Jacobian**, which depends on the node's **inputs**. So the forward pass must **cache the intermediate
values / activations** and hold them until backward consumes them.

> **Reverse-mode = cheap gradients, expensive memory. Forward-mode = expensive gradients, cheap
> memory.**

Consequences (all fall out of "must cache activations"):

- **Activation memory dominates training.** It scales with `batch × depth × width × seq_len` and often
  exceeds the memory of parameters + optimizer state — this is *why* you OOM at large batch/sequence.
- **`torch.no_grad()` saves memory** because it declares "no backward is coming" → intermediates aren't
  cached → freed immediately (the validation-loop point in [[pytorch-training-loop]]).
- **Gradient checkpointing** is the explicit lever: don't store most activations; **recompute them
  during backward** by re-running forward on demand. Trades ~33% extra compute for a large memory
  reduction (store ~√N checkpoints instead of all N activations). How large models fit in limited VRAM.

---

## Cost of backward, and the link to `C ≈ 6ND`

The backward pass costs roughly **~2× the forward** — each forward matmul needs **two** matmuls in
backward: one for the **input gradient** (to keep propagating) and one for the **weight gradient**.

So a full training step ≈ **3× a forward pass** (1× forward + 2× backward). Per token, with `N`
parameters, that's:
- **Forward:** ~`2ND` FLOPs
- **Backward:** ~`4ND` FLOPs
- **Total:** ~`6ND` FLOPs → the **`C ≈ 6ND`** estimate in [[scaling-laws]] (`N` params, `D` tokens).

(Inference is forward-only → ~`2ND`, which is why the training/inference FLOP ratio is ~3×.)

---

Related: [[pytorch-training-loop]], [[pytorch-hooks]], [[pytorch-basics]], [[scaling-laws]],
[[training-diagnostics]]
