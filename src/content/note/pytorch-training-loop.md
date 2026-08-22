---
title: "Anatomy of a PyTorch Training Step"
description: "anatomy of a training step (zero_grad → forward → loss → backward → step); two accumulations (within-backward chain-rule sum vs across-step design choice), `set_to_none` + zero-vs-None table (memory/speed/momentum·weight-decay behavior), mean-vs-sum reduction ↔ effective LR, why step-after-backward (stale-grad bug), `model.eval()` vs `torch.no_grad()` (orthogonal switches for validation)"
category: "PyTorch — Tensors & Mechanics"
order: 9
updatedDate: "2026-08-21T19:28:00.037Z"
---
The five stages of one optimization step, the constraints that fix their order, and the two
easy-to-forget concerns (gradient accumulation semantics, train/eval mode).

```python
for x, y in loader:
    optimizer.zero_grad()          # 1. clear old gradients
    pred = model(x)                # 2. forward pass (builds the graph)
    loss = criterion(pred, y)      # 3. loss → scalar root of the graph
    loss.backward()                # 4. reverse-mode autodiff → fills .grad
    optimizer.step()               # 5. update params from .grad
```

---

## 1. Clearing old gradients — `zero_grad()`

`backward()` **accumulates** into `.grad` (`+=`), it doesn't overwrite. So without an explicit reset,
gradients from previous iterations pile up. **This is a design choice**, not something the chain rule
forces — and it's what enables **gradient accumulation** across mini-batches (simulate a large batch
by running several forward/backward passes before one `step()`).

**Two *different* accumulations — don't conflate them:**
- **Within one `backward()`:** if a parameter is used along multiple paths in the graph, its gradient
  is the **sum of contributions over those paths** — this *is* the multivariable chain rule, automatic
  and always correct.
- **Across successive `backward()` calls:** `.grad` keeps adding to whatever was there — the **design
  choice** that requires manual `zero_grad()`.

`zero_grad()` is the reason for the *second*, not the first.

**`set_to_none=True`** (the **modern default**): sets `.grad = None` instead of writing a zero tensor.
Slightly less memory, and *faster* (skips a memory write). **Behavioral consequence:** a param that
received no gradient ends up with `.grad = None` rather than a zero tensor — which is why sweeps over
gradients need a `if p.grad is not None` guard (see [[pytorch-hooks]], [[pytorch-basics]]).

### `zero_grad`: zero vs None — three differences

| | `set_to_none=False` (zero) | `set_to_none=True` (**default**) |
|---|---|---|
| **Memory** | grad tensors stay allocated (no realloc next step) | grad tensors deallocated → lower peak memory |
| **Speed** | must `memset` every grad to 0 each step | skips the write → **faster** |
| **Behavior on a param that gets no grad this step** | optimizer still processes it: **momentum keeps applying, weight decay still shrinks it** (`grad += wd·param` is nonzero even when grad was 0) | most optimizers **skip the param** — no momentum step, no weight-decay step |

The third row is the subtle one: **zero vs None is not purely memory/speed** — for a parameter that
intermittently receives no gradient (sparse embeddings, a conditional branch), it changes *whether the
param keeps updating*. In normal dense training every param gets a gradient every step, so it rarely
bites — but it's a real behavioral difference. (This is also why `set_to_none` needs the
`p.grad is not None` guard when you sweep gradients.)

---

## 2. Forward pass — evaluate + build the graph

Runs the model on the batch and **simultaneously constructs the computation graph** autograd will
traverse in backward. Every op involving a gradient-tracked parameter creates a **node in a DAG**,
recording (a) which function was applied, (b) which tensors were inputs, and (c) enough information to
compute the **local Jacobian** during backward. See activation/graph mechanics in [[pytorch-hooks]].

---

## 3. Loss — collapse to a scalar

The loss reduces the batch of predictions/targets to a **single scalar** — the **root** of the graph
from which backprop begins (backward needs a scalar to seed `dL/dL = 1`).

**Reduction matters:**
- **`mean`** (default) divides by the number of elements → loss scale is **independent of batch
  size** → gradient magnitude stable across batch sizes.
- **`sum`** makes gradient magnitude **scale with batch size** → acts like a **larger effective
  learning rate**. If you switch reductions or change batch size, you may need to retune LR.

---

## 4. Backward pass — reverse-mode autodiff

`loss.backward()` triggers **reverse-mode automatic differentiation**: starting from the scalar loss,
PyTorch walks **backward** through the DAG, applying the chain rule at each node to propagate gradients
to every **leaf** parameter, filling `p.grad`. (Non-leaf tensors don't retain grad unless you ask;
the graph is freed after backward unless `retain_graph=True`.)

---

## 5. Optimizer step — update parameters

The optimizer uses `.grad` to update parameters. It **never touches the computation graph** — it
operates purely on parameter tensors and their `.grad` (the update runs under `no_grad` internally).

- **SGD:** `param ← param − lr · grad`
- **Adam:** running averages of the 1st and 2nd moments of the gradient, with bias correction →
  per-parameter adaptive step sizes (see [[learning-rate]]).

---

## Why the order is forced

- **`step()` must come after `backward()`.** Call `step()` first and: on iteration 1 `.grad` is `None`
  → with `set_to_none` default, `step()` **silently skips** those params (a no-op, no error); on every
  later iteration you update with the **previous** batch's gradient → a silent **off-by-one / stale
  gradient** bug. The model still "trains," just wrongly.
- **`zero_grad()` placement is flexible** — start-of-loop or end-of-loop both work, as long as it sits
  **after `step()` has read the grads and before the next `backward()`**. What's *not* allowed is
  zeroing between `backward()` and `step()` (you'd wipe the gradients before using them).

---

## The train/eval concern (two orthogonal switches)

A correct loop also handles **mode** and **graph construction** for validation — these are
**independent knobs, both needed**:

| Switch | Controls | Forget it → |
|---|---|---|
| **`model.train()` / `model.eval()`** | *layer behavior*: dropout on/off, BatchNorm batch-stats vs running-stats | wrong metrics at eval (dropout still dropping, BN using batch stats) |
| **`torch.no_grad()`** | *graph construction*: whether autograd builds the DAG + stores activations | val still numerically correct, but **builds a useless graph** → wasted memory (can OOM), slower |

**What each mode actually does to the two layers:**
- **Dropout** — *train:* randomly zeros a fraction `p` of activations and **scales the survivors by
  `1/(1−p)`** so the expected activation is unchanged (inverted dropout, see [[regularization]]).
  *eval:* identity — all activations pass through, nothing zeroed. Leaving train mode on during
  validation injects noise into the loss estimate.
- **BatchNorm** — *train:* normalizes using **batch** statistics and updates its **running averages**.
  *eval:* uses the **stored running averages** for deterministic output. Forget eval and your
  validation normalization depends on batch composition, not the learned statistics.

`eval()` alone can't save the memory — it only changes layer behavior and has no idea whether you
intend to call `backward()`, so it keeps building the graph. `no_grad()` alone doesn't fix
dropout/BN. Validation needs **both** (and switch back to `train()` before resuming):

```python
model.eval()
with torch.no_grad():
    for x, y in val_loader:
        val_loss += criterion(model(x), y)
model.train()          # switch back before resuming training
```

---

Related: [[pytorch-basics]], [[pytorch-hooks]], [[learning-rate]], [[training-diagnostics]],
[[normalization]], [[loss-functions]], [[training-memory]], [[dataloader-and-batching]],
[[regularization]], [[autograd-and-autodiff]]
