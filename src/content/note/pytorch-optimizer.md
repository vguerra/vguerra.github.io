---
title: "The PyTorch `Optimizer` Class"
description: "`Optimizer` anatomy: constructor & `param_groups` (per-group LR/weight-decay overrides, scheduler writes `lr` into group dict), lazy `self.state`, index-based `state_dict` (resume-order gotcha), `zero_grad`, `step` (closure/L-BFGS, skip None-grad, `no_grad` + in-place `mul_`/`add_`)"
category: "PyTorch — Tensors & Mechanics"
order: 11
updatedDate: "2026-08-27T20:47:02.024Z"
---
Anatomy of `torch.optim.Optimizer` — how it stores parameters, per-group hyperparameters, and internal
state, and what a well-structured `step()` does. (Update rules themselves: [[momentum]], [[learning-rate]].)

---

## Constructor

Takes two args: **`params`** (an iterable of parameters *or* parameter-group dicts) and **`defaults`**
(a dict of default hyperparameter values). It normalizes `params` into a **list of parameter-group
dictionaries**, each with a `'params'` key (the tensors) plus a key per hyperparameter:

- `params` = a plain iterable of tensors → **one** parameter group.
- `params` = a list of dicts (each with a `'params'` key + optional hyperparameter overrides) → **one
  group per dict**, missing hyperparameters filled from `defaults`.

---

## `param_groups` — per-group hyperparameter overrides

A **list of dicts**, enabling different hyperparameters for different parts of the model:

- **Different LRs per model part** (transfer learning): small LR for the pretrained **backbone**, larger
  for the newly-added **head**.
- **Selective weight decay:** apply WD to weight matrices but **not** to biases or BatchNorm params —
  put them in separate groups with `weight_decay=0`.

```python
optim.SGD([
    {'params': backbone.parameters(), 'lr': 1e-4},
    {'params': head.parameters(),     'lr': 1e-3},
], lr=1e-3, momentum=0.9)          # lr in the dicts overrides the default
```

**Scheduler interaction (clean separation of concerns):** LR schedulers work by **writing the new
`lr` into each group dict** when they step. The optimizer **reads `lr` from the group dict on every
`step()`** rather than caching it as an instance variable — which is what lets the scheduler and
optimizer stay decoupled ([[lr-schedulers]]). Always log `lr` from `optimizer.param_groups`.

---

## Internal state (`self.state`)

A **`defaultdict(dict)` keyed by parameter tensor identity**, holding each parameter's optimizer state
(momentum buffer, Adam's 1st/2nd moments, step count, …). **Lazily initialized** — created the first
time a parameter's gradient is processed (memory-efficient and robust to params that never get a
gradient).

---

## State serialization (`state_dict` / `load_state_dict`)

Save/restore the optimizer's complete state — **essential for resuming from a checkpoint** (skip it and
momentum/moment buffers reset, corrupting the resumed trajectory).

- `state_dict()` maps **parameter *indices*** (not the tensors) → their state, plus the `param_groups`
  (with hyperparameters).
- On load, parameters are matched **by index within each group**.

> **Gotcha:** because matching is positional, you must rebuild the optimizer over the params in the
> **same order and the same `param_groups` structure** on resume. Reorder params or split groups
> differently and `load_state_dict` silently maps buffers to the **wrong** parameters. Same
> "reconstruct identically" discipline as model `state_dict` ([[pytorch-nn-modules]]). Save model +
> optimizer (+ scheduler) state together.

---

## `zero_grad`

Clears gradients before each backward (PyTorch **accumulates** by default). `set_to_none=True` (the
modern default) is faster/lighter — see the zero-vs-None behavioral table in [[pytorch-training-loop]].

---

## `step` — the update

Structure of a well-formed `step()`:

1. **Handle the optional `closure`** — a callable that re-evaluates the loss; needed by algorithms that
   evaluate it multiple times per step (e.g. **L-BFGS**).
2. **Iterate `param_groups`, then params within each group.**
3. For each param **with a gradient**:
   - **skip if `p.grad is None`** (frozen param, or nothing flowed to it),
   - read the gradient `p.grad`,
   - look up / lazily init its `state`,
   - apply the update rule, **modifying the parameter in place**.
4. **Read hyperparameters from the *current group dict*** (`group['lr']`, etc.) to respect per-group
   overrides.

**Two mechanics that matter:**
- **Runs under `@torch.no_grad()`.** The whole `step()` is wrapped so the update isn't recorded on the
  autograd graph. (The older idiom used `p.data` to bypass tracking; modern reference impls use the
  `no_grad` context and operate on `p` directly.)
- **In-place ops only (`mul_`, `add_`, `addcdiv_`).** Creating a *new* tensor (`p = p - lr*g`) would
  **rebind the name and orphan the leaf**, severing the model's reference to its parameter — the exact
  bug behind `w = w - lr*g` vs `w.sub_(lr*g)` ([[pytorch-basics]]). In-place mutation keeps the
  parameter object identity intact.

---

Related: [[momentum]], [[learning-rate]], [[lr-schedulers]], [[pytorch-training-loop]],
[[pytorch-nn-modules]], [[pytorch-basics]]
