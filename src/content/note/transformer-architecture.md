---
title: "Transformer Architecture"
description: "FFN role, attention vs FFN, memory view of FFN"
category: "Transformers & Sequence Models"
order: 22
updatedDate: "2026-07-06T19:51:02.505Z"
---
## What FFN Adds That Attention Cannot

### What Attention Does
Attention is a **routing mechanism** — mixes information across positions via weighted average of values. This is essentially a linear operation over the sequence (softmax aside).

### What FFN Adds
```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂   # two linear layers + ReLU
```

Applied **pointwise** — each token independently:

1. **Nonlinearity** — without FFN, stacking attention layers collapses to a single linear operation. FFN introduces the nonlinear transformations needed to learn complex functions.

2. **Per-token computation** — attention computes relationships *between* tokens. FFN transforms *each token's representation in place*, independently of other positions.

### The "Memory" View
FFN layers act as **key-value memories** (Geva et al. 2021):
- First layer detects patterns (keys)
- Second layer outputs associated content (values)
- Much of the factual knowledge in LLMs is stored in FFN weights, not attention

### Together
```
Attention → WHERE to get information (routing across positions)
FFN       → WHAT to do with it (nonlinear transformation per token)
```
