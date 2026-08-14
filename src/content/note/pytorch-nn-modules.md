---
title: "PyTorch nn Modules"
description: "`nn.Linear`, `nn.Dropout`, custom `nn.Module`, manual weight init, `nn.Parameter` (vs `register_buffer`), `kaiming_uniform_` (Kaiming vs Xavier), `state_dict`/`load_state_dict`"
category: "PyTorch — Tensors & Mechanics"
order: 1
updatedDate: "2026-08-08T18:56:39.876Z"
---
## nn.Linear

```python
import torch.nn as nn

linear = nn.Linear(in_features=64, out_features=32)

x = torch.randn(batch_size, 64)
out = linear(x)   # shape: (batch_size, 32)
```

Computes: `output = input @ W.T + b`

- `W` shape: `(out_features, in_features)`
- `b` shape: `(out_features,)`
- Weights initialized from `U(-√k, √k)` where `k = 1/in_features`

**Learnable parameters:** `(in_features × out_features) + out_features`

**Disable bias:**
```python
linear = nn.Linear(64, 32, bias=False)
```

**Manual weight initialization:**
```python
# Uniform [-limit, limit] where limit = 1/sqrt(input_size)
# This is Glorot/Xavier-like initialization

# NumPy
import numpy as np
input_size = 64
limit = 1.0 / np.sqrt(input_size)
W = np.random.uniform(-limit, limit, (input_size, 32))
b = np.zeros(32)

# PyTorch
import torch
limit = 1.0 / np.sqrt(input_size)
W = torch.nn.init.uniform_(torch.empty(input_size, 32), -limit, limit)
b = torch.zeros(32)
```

Keeps activations stable across layers by scaling with fan-in.


---

## nn.Dropout

```python
dropout = nn.Dropout(p=0.5)   # p = probability of zeroing an element
out = dropout(x)
```

- **Training**: zeros elements with probability `p`, scales survivors by `1/(1-p)` to preserve expected value
- **Eval**: pass-through — no dropout applied

```python
model.train()   # dropout active
model.eval()    # dropout disabled
```

Common values: `p=0.1` to `p=0.5` — higher for larger models.

---

## nn.Module — Custom Model

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()   # required — sets up parameter tracking, hooks, etc.
        self.linear = nn.Linear(64, 32)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.linear(x)
        x = self.dropout(x)
        return x
```

`super().__init__()` is mandatory — without it, PyTorch can't track parameters and calls like `model.parameters()` or `model.to(device)` will break.

---

## nn.Parameter — declaring learnable weights

A **Tensor subclass** that, when assigned as an attribute of an `nn.Module`, is **automatically
registered** as a learnable parameter.

```python
self.weight = nn.Parameter(torch.empty(out_features, in_features))
```

Registration means it automatically: appears in `model.parameters()` (**optimizer updates it**),
appears in `model.state_dict()` (**saved/loaded**), moves with `model.to(device)`, and has
`requires_grad=True` by default.

**The three ways to attach a tensor to a module** (common interview probe):

| | in `parameters()`? | optimized? | in `state_dict`? | use for |
|---|---|---|---|---|
| `nn.Parameter` | ✅ | ✅ | ✅ | learnable weights |
| `register_buffer(...)` | ❌ | ❌ | ✅ | non-learned state (BN running stats, PE, masks) |
| plain `self.x = tensor` | ❌ | ❌ | ❌ | ⚠️ usually a bug |

**Gotcha:** a raw tensor assigned as an attribute is **not** registered → won't move to GPU with the
model and won't be in the checkpoint. Learnable → `Parameter`; fixed state (e.g. a causal mask) →
`register_buffer`.

---

## Weight init — `nn.init.kaiming_uniform_`

**Trailing underscore = in-place.** Overwrites the tensor's values, returns the same tensor. Call it
*on* an existing parameter:

```python
w = nn.Parameter(torch.empty(out_features, in_features))
nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
```

**What:** Kaiming/He init — samples `U(−bound, +bound)` with `bound` derived from the layer's **fan**
so the **variance of activations stays roughly constant across depth** (else they explode/vanish).

- **`mode`** — `'fan_in'` (default) preserves variance in the **forward** pass; `'fan_out'` in the
  **backward** (gradient) pass.
- **`nonlinearity`** — `'relu'` / `'leaky_relu'` sets the *gain*; `a` = leaky-ReLU negative slope.

**Why Kaiming vs Xavier/Glorot:** Kaiming is for **ReLU-family** activations — ReLU zeros ~half the
inputs (halving variance), so Kaiming uses a larger gain to compensate. **Xavier/Glorot** assumes a
symmetric zero-centered activation (tanh/sigmoid) and doesn't account for ReLU's variance loss →
wrong family = poor early training.

**Where the factor of 2 comes from (the quantitative core):** for a zero-mean symmetric input,
ReLU zeros the entire negative half → `E[ReLU(x)²] = ½·E[x²]`. That's a **½ variance loss per
layer**; over `L` layers the signal (and gradients) shrink by **(½)^L** → exponential vanishing.
Xavier sets `Var(W) ≈ 1/fan` (assumes the activation *preserves* variance) so it under-scales for
ReLU. Kaiming sets `Var(W) = 2/fan_in` — the **2 is exactly the reciprocal of ReLU's ½**, restoring
unit variance per layer (`gain = √2`). So *Kaiming = Xavier + "undo ReLU's halving."*

**`fan_in` vs `fan_out`:** `fan_in` stabilizes the **forward** activation variance; `fan_out` the
**backward** gradient variance. You can't have both unless `fan_in = fan_out`.

*Caveat:* the (½)^L catastrophe is the **plain deep-net** story. **Residual connections +
normalization** make modern nets far less init-sensitive (signal has other paths, gets
re-normalized) — init still affects stability/speed, but "wrong init → guaranteed failure" is
softened.

*(Trivia: `nn.Linear`'s default is `kaiming_uniform_(a=√5)` — a historical quirk, not necessarily
optimal, hence people often re-init explicitly.)*

---

## Checkpointing — `state_dict` / `load_state_dict`

A **`state_dict`** is an `OrderedDict` of **name → tensor** for every registered Parameter and
buffer — the **serializable snapshot of the numbers** (architecture NOT included).

```python
torch.save(model.state_dict(), "ckpt.pt")          # save

model = MyModel(...)                                # 1. rebuild SAME architecture
sd = torch.load("ckpt.pt", map_location="cpu")     # 2. load dict (map_location → device)
model.load_state_dict(sd)                          # 3. copy tensors into the model
```

- **You load *into* an existing model** — `state_dict` holds only tensors, so build the architecture
  first, then pour weights in. Preferred over `torch.save(model)` (pickles the whole object,
  fragile across code changes).
- **`strict=True` by default** — keys must match exactly or it raises. `strict=False` allows partial
  loads and returns `(missing_keys, unexpected_keys)` — e.g. loading a pretrained backbone into a
  model with a new head. `missing_keys` = params in the model but not the checkpoint → they keep
  their **init values** (not loaded).
- **Optimizer has its own `state_dict`** (momentum buffers, step counts) — save/load separately when
  resuming training, or momentum resets.

**How the three connect:** `Parameter` *declares* what's in the `state_dict`; `kaiming_uniform_`
*fills* those parameters when training from scratch; `load_state_dict` *overwrites* them when
resuming/fine-tuning. Init and loading are the two mutually-exclusive ways a parameter gets its
starting values.

Related: [[pytorch-basics]], [[tensor-devices]], [[regularization]]
