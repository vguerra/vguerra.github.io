---
title: "Residual Networks"
description: "degradation problem, residual blocks (`x+f(x)`, additive change), shattered gradients & shorter gradient paths, order of ops (activation-first), exploding-variance (why BN needed), why ResNets work (ensembles of shallow nets, smoother surface, wider>deeper), architectures (ResNet/bottleneck/DenseNet/U-Net)"
category: "Misc ML Concepts"
order: 85
updatedDate: "2026-09-10T21:25:52.325Z"
---
Increasing depth indefinitely **stops helping** — past a point deep plain nets become **hard to train**
(the degradation problem). Residual connections fix this.

## Residual blocks

Each layer computes an **additive change** to the representation rather than transforming it directly:
`out = x + f(x)`. The skip branch adds the layer's input back to its output, so **each function learns an
additive change** and **each layer contributes directly to the network output**.

**Why it helps — shorter gradient paths.** Changes in early-layer params reach the output both directly
(via the skip) and indirectly (through chains of derivatives of varying lengths). Gradients through the
**shorter paths are better behaved**, so residual nets suffer less from **shattered gradients**.

**Shattered gradients:** in *shallow* nets nearby gradients are correlated; in *deep* plain nets that
correlation **drops to ~0** (early-layer changes modify the output in increasingly complex ways with
depth). Residual links keep gradients correlated → trainable.

## Order of operations

If ReLU is the *last* op in a block, the block's output is non-negative → a residual block could only
**increase** the input. So the convention is **activation first, then linear**: `x → ReLU → Linear →
(…) → + x`. Blocks usually **end in a linear** transform. Because a block starting with ReLU zeros a
negative input entirely, **start the network with a linear layer**, not a residual block.

## The exploding-variance problem

Residuals roughly **double** the trainable depth before degradation — but not arbitrarily, because the
**variance of activations grows exponentially** at initialization. He init keeps variance stable across
a *linear+ReLU*, but since each block **adds its input back**, the variance **doubles per block** →
exponential growth. This is why residual nets pair with **BatchNorm** (below), which re-centers/rescales
so variance grows only **linearly** (each block adds ~1 unit of variance).

## Why ResNets work

- **Ensembles of shallow nets** — an unraveled residual net is a sum over many paths; gradients don't
  propagate well through the *longest* paths, so a very deep resnet behaves like a **combination of
  shallower** nets.
- **Smoother loss surface** near minima (vs the same net with skips removed) → easier optimization.
- **Wider can beat deeper** — more channels per layer sometimes outperforms more layers.

## Architectures

- **ResNet** — each block: BatchNorm → ReLU → conv, twice, then add to input.
- **Bottleneck block** — `1×1` (reduce channels) → `3×3` → `1×1` (restore channels); more
  parameter-efficient.
- **DenseNet** — **concatenates** (not adds) earlier layer outputs to later ones → direct contribution
  from all earlier layers.
- **U-Net / hourglass** — encoder-decoder where **earlier representations are concatenated to later
  ones** (used in segmentation).

---

Related: [[normalization]], [[convolution]], [[training-diagnostics]], [[rnn]], [[pytorch-nn-modules]]
