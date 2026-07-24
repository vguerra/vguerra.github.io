---
title: "PyTorch nn Modules"
description: "`nn.Linear`, `nn.Dropout`, custom `nn.Module`, manual weight init"
category: "PyTorch — Tensors & Mechanics"
order: 1
updatedDate: "2026-07-11T20:44:52.244Z"
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
