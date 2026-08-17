---
title: "PyTorch Hooks & Iterating a Module's Layers"
description: "iterating layers (`named_modules`/`children`/`named_parameters`, leaf filter), forward/backward/tensor hooks, `handle.remove()` lifecycle, `p.grad` (param) vs `grad_output` (activation) grad, why to `detach().std().item()` instead of stashing outputs"
category: "PyTorch — Tensors & Mechanics"
order: 8
updatedDate: "2026-08-16T16:40:01.168Z"
---
Hooks let you **observe or modify** the inputs/outputs/gradients of a module *without editing its
`forward`*. General-purpose: activation/gradient diagnostics ([[training-diagnostics]]), feature
extraction, Grad-CAM, gradient surgery, activation caching.

---

## Iterating over a Module's Layers (the prerequisite)

Two axes: **modules vs parameters**, and **recursive vs immediate-children**.

```python
model.modules()          # ALL modules, recursively (depth-first). FIRST item is `model` itself.
model.named_modules()    # same + names: ("encoder.layer.0.attn", module)  — dotted path
model.children()         # IMMEDIATE children only (one level), no recursion
model.named_children()   # same, with names

model.parameters()       # all learnable tensors, recursively
model.named_parameters() # ("encoder.layer.0.attn.q_proj.weight", tensor)
```

- **`modules()` flattens the whole tree** and yields `model` first — use for a global "hook every
  layer" sweep.
- **`children()`** gives only top-level blocks — use to iterate stages of a `Sequential` without
  descending.
- **Gotcha:** `named_modules()` **includes container modules** (`Sequential`, `ModuleList`, the whole
  model), not just leaf ops. A container's output = its last child's output → hooking everything gives
  **redundant entries**.

**Filter to leaf layers only** (actual `Linear`/`Conv`/`ReLU`, not containers) — a leaf has **no
children**:

```python
for name, module in model.named_modules():
    if not list(module.children()):     # empty list is falsy → leaf
        ...                             # register hook here
```

**Which iterator for which diagnostic:**
- **Activation** stats → hook **modules** (activations are module *outputs*).
- **Gradient / weight** stats → iterate **`named_parameters()`**, read `p.grad` / `p.data` after
  `backward()` — simpler than backward hooks.

---

## The Three Hook Registration Points

### 1. Forward hook — module outputs (activations)

```python
handle = module.register_forward_hook(fn)

def fn(module, input, output):
    # `output` is the tensor this layer produced
    # (returning a tensor here REPLACES the output — that's how you modify activations)
    ...
```

There is also `register_forward_pre_hook(fn)` with signature `fn(module, input)` — fires *before*
the forward, lets you inspect/modify **inputs**.

### 2. Full backward hook — gradients flowing *through a module*

```python
handle = module.register_full_backward_hook(fn)

def fn(module, grad_input, grad_output):
    # grad_output[0] = dL/d(this layer's OUTPUT) — the ACTIVATION gradient
    ...
```

Use **`register_full_backward_hook`**, *not* the deprecated `register_backward_hook` (buggy for
modules with multiple inputs).

### 3. Tensor hook — gradient of one specific tensor

```python
handle = some_tensor.register_hook(fn)   # fn(grad) -> optionally a modified grad
```

Fires when that tensor's gradient is computed; return a tensor to **replace** the gradient
(gradient clipping/surgery on a single tensor).

---

## Lifecycle: always `remove()`

Every `register_*` returns a **handle**. You **must** call `handle.remove()` when done:

```python
handles = []
handles.append(module.register_forward_hook(fn))
...
for h in handles:
    h.remove()
```

Otherwise hooks stay attached forever, fire on **every** subsequent forward/backward, and **leak
memory** (they hold references to activations).

---

## Two Design Gotchas (the crux)

### `p.grad` (parameter grad) vs backward-hook `grad_output` (activation grad)

They are **different quantities**:

- **`p.grad`** after `backward()` = the **parameter** gradient `∂L/∂W`, `∂L/∂b` → magnitude of the
  weight update. Get it with `named_parameters()`; **no hook needed**.
- **backward hook `grad_output[0]`** = the **activation** gradient `∂L/∂(layer output)` → the thing
  that actually **flows backward through the network**. This is the right signal for spotting where
  vanishing/exploding sets in across depth. `p.grad` is downstream of it (roughly
  `grad_output` × input), so use the hook when you specifically want flow-through activation grads.

### Never stash the raw `output` — detach and reduce to a scalar

`output` in a forward hook is **still attached to the autograd graph** (`requires_grad=True`, has a
`grad_fn`). Doing `activations[name] = output`:

- **Pins the entire computation graph** so it can't be freed after `backward()` → memory grows every
  step → OOM.
- Holds far more than you need (you only want a scalar statistic).

Fixes, best last:

```python
activations[name] = output.detach()               # minimal: cut off the graph
activations[name] = output.detach().std().item()  # better: store a Python float, no tensor held
```

`.item()` pulls a plain float off the GPU → no GPU memory pinned. Same principle in the backward
hook: reduce `grad_output[0]` to a scalar norm inside the hook, don't stash the tensor.

---

## Putting it together (instrumentation shape — not the fill-in)

The wiring for a per-layer health check ([[training-diagnostics]]):

1. Loop `named_modules()`, filter to leaves (`if not list(module.children())`).
2. Register a **forward hook** recording `output.detach().std().item()` into a dict keyed by `name`
   (+ dead-neuron fraction for ReLU outputs).
3. Run one forward + `backward()`.
4. Read per-parameter `p.grad.norm()` from `named_parameters()`.
5. `handle.remove()` for every handle.

> **Dead-neuron fraction — the subtle averaging point:** "fraction of units that never activate"
> means a neuron is dead only if it's ≤0 for **every example in the batch** — so reduce the
> `(batch, features)` activation over the **batch dim first** (does this unit fire for *any* sample?),
> *then* take the fraction across features. Averaging the wrong way conflates "off for this sample"
> with "permanently dead."

---

Related: [[training-diagnostics]], [[pytorch-nn-modules]], [[tensor-memory-layout]],
[[pytorch-basics]], [[normalization]]
