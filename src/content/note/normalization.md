---
title: "Normalization"
description: "LayerNorm formula, variance, LayerNorm vs BatchNorm, why γ/β exist, ICS caveat (loss-smoothing), 3 BN problems LN fixes, Pre-LN vs Post-LN, RMSNorm, BN-in-residual-nets (variance control) + benefits (stable fwd/higher LR/regularization), norm variants table (Instance/Ghost/Batch-Renorm), weight norm, group norm"
category: "Transformers & Sequence Models"
order: 47
updatedDate: "2026-09-10T21:27:42.828Z"
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

## Why the learnable γ, β exist

Pure normalization forces every layer to zero-mean / unit-variance, which **removes
representational capacity** — sometimes the network *wants* a non-zero mean or a different scale.
The learnable **scale γ** and **shift β** let the network *learn to undo* the normalization when
that's optimal (it can even recover the identity mapping). So LayerNorm standardizes for
conditioning, then hands control of the final scale/offset back to the model.

## Why it works — and a caveat on "internal covariate shift"

The original motivation was reducing **internal covariate shift** (later layers constantly chasing
the shifting distribution of earlier ones). That explanation is **contested**: Santurkar et al.
(2018) showed the real benefit is that normalization **smooths the loss landscape** (more
predictable gradients), enabling higher learning rates and faster convergence — *not* covariate
shift per se. Safe interview framing: "keeps activations well-conditioned and smooths optimization."

## The three BatchNorm problems LayerNorm fixes

1. **Mini-batch dependency.** BN computes stats across the *batch* → small batches give noisy
   estimates → unstable training; and BN is **fundamentally broken at batch size 1** (variance
   undefined). Forces large batches.
2. **RNN / variable-length incompatibility.** Sequence lengths vary → BN would need separate stats
   per timestep, and later positions (seen in fewer sequences) get noisier estimates.
3. **Train/inference mismatch.** BN uses *batch* stats in training but *running* stats at inference
   → the model can behave differently in `eval` mode (a subtle-bug source).

**LayerNorm sidesteps all three:** it normalizes each sample across its own features, so it works
**identically regardless of batch size, sequence length, or train-vs-eval mode** — no running stats,
no batch coupling.

## LayerNorm vs BatchNorm

| | LayerNorm | BatchNorm |
|---|---|---|
| Normalizes over | features (per sample) | batch (per feature) |
| Depends on batch size | No | Yes |
| Works at inference | Always | Needs running stats |
| Used in | Transformers | CNNs |

## Modern variants

**Pre-LN vs Post-LN.** The *original* Transformer applied LN **after** the residual add
(**Post-LN**); modern LLMs apply it **inside** the residual branch, **before** each sublayer
(**Pre-LN**) — Pre-LN has better-behaved gradients and trains stably **without warmup**. Either way
a transformer block uses LN **twice** (around attention and around the FFN); only its *position*
relative to the residual changed.

**RMSNorm** (LLaMA and most current LLMs). Drops mean-centering entirely — only rescales by the
root-mean-square, no `μ` subtraction and no `β`:

$$y = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \cdot \gamma$$

Cheaper (one fewer reduction, fewer params) and works as well — evidence that the **re-scaling, not
the re-centering, is the part that matters**.

## BatchNorm in residual networks & its benefits

Beyond the covariate-shift/loss-smoothing story, BN plays a specific role in **residual networks**
([[residual-networks]]): residual blocks make activation variance grow **exponentially** at init (each
block adds its input back → variance doubles); BN re-centers/rescales so variance grows only
**linearly** (with `δ=0, γ=1`, each block adds ~1 unit of variance). Three benefits:
- **Stable forward prop** — unit-variance activations at init.
- **Higher learning rates** — BN smooths the loss surface (reduces shattered gradients) → more
  predictable gradients → larger stable LR → better performance.
- **Regularization** — normalization stats depend on the **batch**, so each example's activation is
  perturbed differently each iteration (injected noise).

**Mechanism detail:** standardize `(h − m_h)/(s_h + ε)` then scale/shift `γh + δ` (learned) — output has
mean `δ`, std `γ` per hidden unit. `K` layers × `D` units → `KD` learned offsets + `KD` scales. At test
time, `m_h, s_h` are computed over the **whole training set** and frozen. BN makes the net **invariant to
weight/bias rescaling** (double the weights → stats double → normalization cancels it). Caveat: it
**causes gradient explosion in ReLU nets *without* skip connections**.

## Normalization variants (what stats they use)

| Norm | Normalizes over | Batch-dependent? |
|---|---|---|
| **BatchNorm** | batch (per channel) | **yes** |
| **LayerNorm** | all channels/features of **one example** | no |
| **GroupNorm** | groups of channels, per example | no |
| **InstanceNorm** | **each channel** separately, per example (GroupNorm with groups = channels) | no |
| **GhostNorm** | a **sub-batch** → noisier stats → more regularization (large batches) | yes (partial) |
| **Batch Renorm** | keeps a **running average** to correct unreliable small-batch stats | yes (corrected) |

---

## Weight Normalization

A different axis from activation normalization: **reparameterize each weight vector** to decouple its
**direction** from its **magnitude** — `w = g · v/‖v‖`, learning direction `v` and scalar norm `g`
separately (so `‖w‖ = g`). Controls scaling explicitly (helps avoid exploding/vanishing activations),
**no mini-batch statistics needed** (unlike BatchNorm) → works for small batches / RNNs / online. It
normalizes *parameters*, not *activations*.

## Group Normalization

Divides channels into **groups** and normalizes within each group, **independent of batch size** →
consistent with small batches. Good for object detection / segmentation / mobile models; you must choose
the **number of groups**. Sits between LayerNorm (all channels = 1 group) and InstanceNorm (1 channel
per group).

---

Related: [[transformer-architecture]], [[self-attention]], [[learning-rate]], [[numpy-basics]], [[distributed-training]], [[rnn]]
