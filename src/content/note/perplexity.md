---
title: "Perplexity"
description: "definition, stable log-prob implementation, why not to multiply probs"
category: "Transformers & Sequence Models"
order: 25
updatedDate: "2026-07-02T19:59:16.201Z"
---
## Definition

```
PP = exp(-(1/N) * Σ log p(wᵢ))
```

Measures how well a probability model predicts a sequence — lower is better.

## Stable Implementation

Always use log-probabilities — the product form underflows to 0 for long sequences.

```python
# From log-probs (most stable)
def perplexity(log_probs):
    return torch.exp(-torch.mean(log_probs))

# From raw probs
def perplexity(probs):
    log_probs = torch.log(torch.clamp(probs, min=1e-9))
    return torch.exp(-torch.mean(log_probs))
```

## Why Not Multiply Probabilities

```python
# Unstable — underflows to 0.0 for long sequences
pp = (prod(p_i for p_i in probs)) ** (-1/N)
```

Each probability is < 1, so multiplying hundreds together → `0.0` (float underflow).

**Key insight:** log turns products into sums — summing log-probs stays numerically safe even for thousands of tokens.
