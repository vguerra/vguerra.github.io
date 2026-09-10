---
title: "Convolution (Conv2d)"
description: "Conv2d: weight sharing & translation equivariance, cross-correlation vs true convolution (framework gotcha), multi-channel formula & shapes `(C_out,C_in,k,k)`, output-size formula, im2col (conv as GEMM), receptive field `1+L(k−1)` (why 2×3×3 beats 5×5), CNN building blocks (filter sizes/1×1/FC↔conv, padding, max-vs-avg pooling, strided-conv, upsampling, CNN-for-text), equivariance-vs-invariance, stride/dilation, spatial-dropout/cutout, applications (YOLO/segmentation), params & MACs-vs-FLOPs"
category: "Misc ML Concepts"
order: 86
updatedDate: "2026-09-10T21:29:10.513Z"
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

## CNN building blocks

**Why "locally connected":** each neuron connects to a small local region, not the whole input →
**fewer parameters** (faster, less overfitting), exploits **spatial locality** (nearby pixels
correlated), and weight sharing gives **translation invariance**.

**Filter sizes:**
- **1×1** — channel mixing / dimensionality reduction (combine channels, no spatial mixing); adds
  non-linearity without changing spatial size; pointwise projection. (A **fully-connected layer = a 1×1
  conv** over a 1×1 spatial map — `nn.Linear(512,1000)` ≡ `nn.Conv2d(512,1000,kernel_size=1)`.)
- **3×3** — default for local patterns; efficient, expressive, stackable (two 3×3 ≻ one 5×5).
- **5×5 / 7×7** — larger receptive field per layer, more params; usually only in **early** layers on
  high-res input. Match filter size to resolution (MNIST → 1×1/3×3; ImageNet → 7×7 early like ResNet).

**Padding** — add a border (usually zeros) to **control output size** / preserve spatial dims. Without
it, border pixels are underrepresented and you can't keep dims constant across layers (needed for
**residual** skip connections that require matching shapes). Variants: **zero** (common), **reflective**
(mirror), **replicate** (edge value).

**Pooling** — downsample feature maps (translation invariance, fewer params, less overfitting):
- **Max pooling** — keeps the strongest activation → sparse, sensitive to *presence* of a feature; best
  for **classification / detection** (most discriminative features).
- **Average pooling** — smooths/blends all activations → denser; better for **segmentation / denoising /
  texture**.
- **Remove pooling** → higher spatial resolution (good for segmentation/generation) but more
  compute/memory, more overfitting risk, less translation invariance.
- **Strided conv (stride 2) vs 2×2 max-pool:** strided conv is **learnable** (more params) and produces
  smoother feature maps; max-pool is parameter-free.

**Upsampling** — increase spatial resolution (segmentation, super-resolution, generative decoders):
nearest-neighbor (duplicate), bilinear/bicubic (interpolate, smoother), **transpose conv** (learned
upsampling), pixel-shuffle.

**CNNs for text (1-D):** input is `(sequence_length × embedding_dim)` — treat sequence length as
**width** and **embedding dim as the input channels**. A 1-D conv slides over positions, mixing across
the embedding channels.

---

## Parameters & compute

- **Learnable params:** `C_out · C_in · k² + C_out` (weights + biases).
- **Compute:** `C_out · C_in · k² · H_out · W_out` — this counts **MACs** (multiply-accumulates).
  **FLOPs ≈ 2×** that if counting the multiply and add separately. State which you mean.

---

## More CNN mechanics & applications

**Equivariance vs invariance:** conv layers are **translation-equivariant** (shift input → output
shifts). Classification networks want **invariance** (same label regardless of translation/rotation/flip)
— achieved by pooling + architecture, not the conv alone. **Why CNNs beat FCNs:** superior **inductive
bias** — an FCN must learn each pattern at *every* position; the CNN **shares** weights across positions.

**Stride / dilation:** **stride** = how far the kernel shifts (stride 2 → ~half the outputs,
downsampling). **Dilation** = spread kernel taps with gaps (dilation 2 makes a size-5 kernel act like a
sparse size-3) → larger receptive field, no extra params.

**Downsampling** = strided conv, **max pooling** (retains max of each region — invariance to small
shifts), or average pooling. **Upsampling** = duplicate, **max unpooling** (send values back to the
argmax positions, rest 0), bilinear, or **transposed convolution** (learned). Applied **per channel**.

**Spatial dropout / cutout:** plain dropout is **weak for conv layers** — neighboring pixels are
correlated, so a dropped unit's info survives via adjacent positions. Fixes: **spatial dropout** (drop
whole **feature maps**) and **cutout** (mask a square patch of the input).

**Applications:** **image classification** (reshape to fixed size); **object detection** — **YOLO**
outputs, per grid cell, class + bounding boxes (`x,y,w,h,confidence`), then non-max suppression removes
low-confidence/duplicate boxes; **semantic segmentation** — label **every pixel** (encoder-decoder /
U-Net, [[residual-networks]]). Conv-specific init: Xavier/He, or **ConvolutionOrthogonal** (trains 10k+
layers without residuals).

---

Related: [[ml-concepts]], [[broadcasting]], [[tensor-memory-layout]], [[normalization]],
[[pytorch-nn-modules]], [[residual-networks]]
