---
title: "Convolution (Conv2d)"
description: "Conv2d: weight sharing & translation equivariance, cross-correlation vs true convolution (framework gotcha), multi-channel formula & shapes `(C_out,C_in,k,k)`, output-size formula, im2col (conv as GEMM), receptive field `1+L(k−1)` (why 2×3×3 beats 5×5), params & MACs-vs-FLOPs"
category: "Misc ML Concepts"
order: 63
updatedDate: "2026-08-28T14:28:05.362Z"
---
The core operation of CNNs. Instead of connecting every input to every output (fully-connected), a
convolution slides a small **kernel/filter** across the spatial dims, computing a weighted sum at each
position. This **cuts parameters** and **exploits spatial structure**.

---

## Why convolution — weight sharing & equivariance

Mapping an `H×W` grayscale image to an `H×W` output with a **fully-connected** layer needs **`(H·W)²`**
weights. A convolution with a `k×k` kernel needs only **`k² + 1`** (bias) — because the **same kernel is
applied at every position** (**weight sharing**).

- **Translation equivariance:** if the input **shifts**, the output **shifts by the same amount** (not
  *invariance* — the output isn't unchanged, it moves along). This is the right **inductive bias** for
  vision: an edge should be detectable **regardless of where** it appears.

---

## ⚠️ It's cross-correlation, not true convolution

The CNN formula (below) does **not flip** the kernel — that's **cross-correlation**:

$$y_{ij} = \sum_{m}\sum_{n} x_{i+m,\,j+n}\, w_{m,n} + b$$

*True* mathematical convolution flips the kernel (`w_{m,n} → w_{k-1-m,\,k-1-n}`). **Every framework's
`Conv2d` computes cross-correlation** and calls it convolution. **Why it doesn't matter:** the kernel
is **learned**, so it would simply learn the flipped weights — the flip is absorbed into training. (It
*would* matter for a fixed, hand-designed kernel.) Classic interview gotcha: *"does PyTorch Conv2d do a
true convolution?"* → **no, cross-correlation.**

---

## Shapes & the multi-channel formula

**Single channel:** input `(H, W)`, kernel `(k, k)`, **valid** convolution (kernel only where it fits
fully, no padding) → output `(H−k+1, W−k+1)`.

**Multi-channel:** input `(C_in, H, W)`. **Each output channel** has its own kernel `(C_in, k, k)` (it
looks at **all** input channels) and its own bias → bias vector `(C_out,)`:

$$y_{c,i,j} = \sum_{c'=0}^{C_{in}-1}\sum_{m}\sum_{n} x_{c',\,i+m,\,j+n}\, w_{c,c',m,n} + b_c$$

Full weight tensor: **`(C_out, C_in, k, k)`**.

**Batched:** input `(N, C_in, H, W)` → output `(N, C_out, H_out, W_out)`. Applied independently per
sample; **weights shared across the batch**.

---

## Output-size formula (padding `p`, stride `s`)

$$H_{out} = \left\lfloor \frac{H + 2p - k}{s} \right\rfloor + 1$$

(Valid conv, `p=0, s=1` → `H_out = H − k + 1`.) The **floor** matters when the stride doesn't divide
evenly. Same for `W_out`. Dilation `d` replaces `k` with `k_eff = d(k−1)+1`.

---

## Implementations

### Sliding window (naive)
For each output position `(i,j)`: extract patch `x[:, :, i:i+k, j:j+k]` → flatten → matmul with the
weights → `y[:, :, i, j]`. Loops over spatial positions, vectorizes across batch/channels. Fine for
small inputs/kernels.

### im2col (image-to-column) — conv as one GEMM
The production trick: **extract all patches at once** and turn the whole convolution into a **single
matrix multiply**:

1. Extract all `H_out·W_out` patches; reshape each `(C_in, k, k)` → a column of length `C_in·k²`.
2. Stack columns → **`X_col` of shape `(C_in·k², H_out·W_out)`**.
3. Flatten weights → **`W_flat` of shape `(C_out, C_in·k²)`**.

$$Y_{\text{flat}} = W_{\text{flat}}\, X_{\text{col}} + b \quad\to\quad (C_{out},\, H_{out}\,W_{out})$$

Reshape `Y_flat` back to `(C_out, H_out, W_out)`. **Trades memory for speed:** `X_col` **duplicates**
input data (overlapping patches share elements, ~`k²`× blow-up) but the op maps directly to
**optimized BLAS/GEMM** — why it's fast. (This is why a matmul understanding underpins conv perf.)

---

## Receptive field

The region of the **original input** that influences one output neuron. One layer with kernel `k` → each
output sees a `k×k` patch. Stacking `L` layers (stride 1) grows it **linearly**:

$$\text{RF} = 1 + L\,(k-1)$$

(3×3 over 5 layers → RF = 11.) Linear growth is *why* **dilated convs** (exponential RF) and
**stride/pooling** (downsampling) exist — to reach a large RF without hundreds of layers. It's also why
**two stacked 3×3 convs beat one 5×5**: same RF (5), **fewer params** (2·9 vs 25 per channel-pair) and
an extra nonlinearity.

---

## Parameters & compute

- **Learnable params:** `C_out · C_in · k² + C_out` (weights + biases).
- **Compute:** `C_out · C_in · k² · H_out · W_out` — this counts **MACs** (multiply-accumulates).
  **FLOPs ≈ 2×** that if counting the multiply and add separately. State which you mean.

---

Related: [[ml-concepts]], [[broadcasting]], [[tensor-memory-layout]], [[normalization]],
[[pytorch-nn-modules]]
