---
title: "Tensor Devices & Device Management (CPU / GPU)"
description: "CPU/GPU/MPS, create-on-device, `model.to` (in-place) vs `tensor.to` (copy), device consistency, transfer overhead, `pin_memory`/`non_blocking`"
category: "PyTorch — Tensors & Mechanics"
order: 7
updatedDate: "2026-07-16T21:05:30.278Z"
---
## Devices and Availability

```python
torch.cuda.is_available()          # CUDA GPU present?
torch.backends.mps.is_available()  # Apple Silicon (Metal)

torch.device('cpu')
torch.device('cuda')       # default GPU
torch.device('cuda:0')     # specific GPU by index
torch.device('mps')        # Apple Silicon
```

**Device-agnostic pattern** (write once, run anywhere):
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

---

## Creating Tensors on a Device

```python
x = torch.randn(1000, 1000, device=device)   # (A) directly on device — PREFERRED
x = torch.randn(1000, 1000).to(device)       # (B) create on CPU then move — wasteful
```
(A) skips the CPU allocation + CPU→GPU transfer. Most constructors accept `device=`.

---

## Moving Tensors

```python
x = x.to(device)   # general (device and/or dtype)
x = x.cuda()       # shorthand → GPU
x = x.cpu()        # shorthand → CPU
```

**Two critical facts:**
1. `.to()` on a tensor returns a **NEW tensor — must reassign**:
   ```python
   x.to(device)       # ❌ result discarded, no effect
   x = x.to(device)   # ✓
   ```
2. Already on target device → `.to()` is a **no-op** (free).

---

## The Big Asymmetry: Models vs Tensors (top bug source)

```python
x = x.to(device)      # TENSORS: .to() returns a new tensor → MUST reassign
model.to(device)      # MODULES: .to() moves params/buffers IN-PLACE → reassign optional
```

`nn.Module.to()` mutates parameters/buffers **in place**; `tensor.to()` produces a **copy**.
So `model.to(device)` alone works, but `x.to(device)` alone silently does nothing.
**Mnemonic: models move in place, tensors don't.**

---

## Device Consistency — the #1 Runtime Error

All tensors in an op must share a device, else:
```
RuntimeError: Expected all tensors to be on the same device, found cuda:0 and cpu!
```

**Discipline:** move the **model once**, move **each batch (inputs AND targets)** in the loop:
```python
model.to(device)
for xb, yb in loader:
    xb, yb = xb.to(device), yb.to(device)   # BOTH — forgetting targets is classic
    out = model(xb)
    loss = criterion(out, yb)
```

---

## Transfer Overhead — Minimize Crossings

CPU↔GPU transfers cross PCIe and are **slow vs. on-device compute**.

- **Don't shuttle data in hot loops** — keep it on GPU through forward/backward.
- **Avoid `.item()` / `.cpu()` inside the training step** — each forces a **sync** (GPU is async;
  pulling a value to Python blocks until kernels finish). Accumulate on-device, pull once/epoch:
  ```python
  total += loss.item()     # ❌ syncs every batch
  total += loss.detach()   # ✓ accumulate on-GPU; .item() after the epoch
  ```
- **Overlap transfer with compute** — `pin_memory=True` (DataLoader) + `non_blocking=True`:
  ```python
  loader = DataLoader(ds, pin_memory=True)
  xb = xb.to(device, non_blocking=True)   # async copy, overlaps GPU work
  ```
  Pinned (page-locked) host memory is what enables the async copy.

---

## Moving Models — Extra Notes

- `model.to(device)` moves **all parameters and registered buffers** (e.g. BatchNorm running
  stats). Plain tensor attributes are **NOT** moved unless registered:
  `self.register_buffer('const', t)` — otherwise a manual constant stays on CPU (bug).
- **Optimizer:** built from `model.parameters()`; `.to()` moves params in place (same objects), so
  refs stay valid. Loading optimizer state from a checkpoint may need moving to device too.
- **Checkpoints:** `torch.load(path, map_location=device)` controls where it lands.

---

## Essentials

1. **Device-agnostic:** `device = 'cuda' if cuda.is_available() else 'cpu'`; target it everywhere.
2. **Create on-device** (`device=`) rather than create-then-move.
3. **Models move in place** (`model.to(device)`); **tensors don't** (`x = x.to(device)`).
4. **Consistency:** all operands same device — model once, every batch (inputs **and** targets).
5. **Minimize transfers:** no `.item()`/`.cpu()` in hot loops (they sync); `pin_memory` +
   `non_blocking` to overlap.

Related: [[pytorch-basics]], [[tensor-dtypes]], [[tensor-memory-layout]]
