---
title: "ML Concepts"
description: "Conv2d parameters, sigmoid, log"
category: "Misc ML Concepts"
order: 29
updatedDate: "2026-07-02T19:13:05.240Z"
---
## Conv2d Learnable Parameters

```
params = (kernel_h × kernel_w × C_in × C_out) + C_out
          └─────────────── weights ───────────────┘   └─ biases ┘
```

**Example:** 3×3 conv, 32 input channels, 64 output channels:
```
(3 × 3 × 32 × 64) + 64 = 18,496
```

Key points:
- Each output filter is a `(kernel_h × kernel_w × C_in)` 3D tensor
- There are `C_out` such filters
- Parameter count is **independent of input spatial dimensions (H, W)**
- Bias can be disabled with `bias=False`

---

## Sigmoid Function

Squashes any real number to (0, 1):

```
σ(x) = 1 / (1 + e⁻ˣ)
```

- Output range: (0, 1)
- σ(0) = 0.5
- Saturates near 0 and 1 at extremes
- Used for binary classification output, gates in LSTMs

---

## Natural Log Function

```
log(x), defined only for x > 0
```

- log(1) = 0
- log(x) → -∞ as x → 0⁺
- Grows slowly (logarithmically)
- Used in NLL/cross-entropy loss — requires clipping probabilities to avoid log(0)
