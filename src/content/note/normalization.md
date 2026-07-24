---
title: "Normalization"
description: "LayerNorm formula, variance, LayerNorm vs BatchNorm"
category: "Transformers & Sequence Models"
order: 21
updatedDate: "2026-07-06T19:22:51.479Z"
---
## LayerNorm

```
y = γ · (x - μ) / √(σ² + ε) + β
```

- `μ` = mean of input vector across features
- `σ²` = variance of input vector across features: `(1/d) Σᵢ (xᵢ - μ)²`
- `ε` = small constant for numerical stability (e.g. 1e-5)
- `γ, β` = learnable scale and shift parameters

**Key:** variance is computed **per sample across features** — not across the batch.

Steps:
1. Center: `x - μ`
2. Normalize to unit variance: `/ √(σ² + ε)`
3. Rescale and shift: `γ · (...) + β`

## LayerNorm on a Batch

For input shape `(batch, features)`, LayerNorm computes **per sample** — samples never interact:

```python
x = np.random.randn(4, 8)                       # (batch=4, features=8)
μ  = np.mean(x, axis=-1, keepdims=True)         # (4, 1) — one mean per sample
σ² = np.var(x, axis=-1, keepdims=True)          # (4, 1) — one variance per sample
x_norm = (x - μ) / np.sqrt(σ² + 1e-5)          # (4, 8)
```

BatchNorm computes across the batch dimension (axis=0) per feature — samples do interact.

## Role of Residuals + LayerNorm for Deep Stack Trainability

**Residuals** add a skip connection: `output = F(x) + x`

During backprop:
```
∂L/∂x = ∂L/∂output · (∂F(x)/∂x + 1)
```
The `+1` term creates a direct gradient highway — even if `∂F(x)/∂x` vanishes, gradients still flow. Also: at init with weights near zero, `F(x) ≈ 0` → block acts as identity → network starts effectively shallow and learns depth gradually.

**LayerNorm** keeps activations well-conditioned at every layer regardless of what earlier layers are doing — prevents internal covariate shift in deep stacks. Per-sample normalization means it's stable at any batch size, including batch size 1 at inference.

---

## LayerNorm vs BatchNorm

| | LayerNorm | BatchNorm |
|---|---|---|
| Normalizes over | features (per sample) | batch (per feature) |
| Depends on batch size | No | Yes |
| Works at inference | Always | Needs running stats |
| Used in | Transformers | CNNs |
