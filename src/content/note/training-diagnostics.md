---
title: "Training Diagnostics — Debugging Why a Model Won't Learn"
description: "debugging why a model won't learn: reading loss-curve shapes (flat/plateau/spiky/NaN, `log C` random baseline), the 3 silent failures (dead ReLU, vanishing/exploding gradients + causes/fixes), residuals + norm as gradient highways, per-layer health check & log-scale gradient-norm plot, parameter-update-ratio ≈1e-3 heuristic, debugging by `.grad` state (None/zero/NaN/params-not-changing symptom→cause table), Karpathy's recipe (overfit one batch first)"
category: "Training Dynamics & Optimization"
order: 16
updatedDate: "2026-08-21T19:23:49.724Z"
---
Training diagnostics catch the bugs that keep a model from learning well. In practice we
**inspect activations, gradients, and loss curves** to figure out *why* a net isn't learning.

**Core premise: neural nets fail silently — they don't crash.** A bug rarely throws an exception;
it just quietly produces a model that doesn't learn (or learns badly). So you need to *actively look*
at internal signals rather than wait for an error.

---

## Reading the Loss Curve

The loss curve is the first thing to look at. Its **shape** localizes the problem:

| Shape | Likely meaning | What to check / do |
|---|---|---|
| **Flat from step 0** | Not learning at all | LR is 0 or way too small; broken symmetry (all neurons init'd equally); broken data pipeline (shuffled labels, all-zero inputs, wrong loss reduction) |
| **Flat, but at the random baseline** | Forward path is *fine*, optimization isn't | Compare to `log(C)` (see below). If flat ≈ baseline → LR/symmetry/data. If flat ≫ baseline → loss/data wiring |
| **Diverges to ∞ / NaN** | Exploding gradients | LR too high, bad init; apply grad clipping / normalization |
| **Plateau after an initial drop** | *Ambiguous* — see below | Could be genuine convergence (fine!), underfitting (bad), or LR-limited |
| **Spiky / noisy but trending down** | Noisy gradient estimate | Batch too small (high-variance gradient); LR a touch too high (overshoot near min); occasional tall spikes → a few bad/mislabeled batches |
| **Train ↓ while val ↑** | Overfitting | Regularization, early stopping, more data (see [[overfitting-underfitting]]) |

### The random-baseline sanity check (cross-entropy)

An untrained net making **uniform** predictions puts `p_true = 1/C`, so its cross-entropy loss is:

$$\mathcal{L}_{\text{random}} = -\log\!\frac{1}{C} = \log C$$

(**natural log**; ≈ 2.30 for 10 classes, ≈ 6.91 for 1000). **Watch the sign:** it's `−log(1/C) = log C`,
a **positive** number — CE loss is never negative.

- First-step loss ≈ `log C` → forward path and loss are wired correctly; the net just hasn't learned.
- First-step loss ≫ `log C` (e.g. 15 when `log C ≈ 2.3`) → something is wrong *before* training:
  overconfident-wrong logits from bad init/scaling, mislabeled data, or a mis-reduced loss.

### The plateau trap

A plateau after the initial drop has **two very different meanings** — don't reflexively "adjust LR":

- Plateau at a **low/reasonable** loss → the model may have simply **converged**. Nothing wrong.
- Plateau at a **high** loss → *now* it's a problem: underfitting (too little capacity), optimization
  stuck, or LR too high to settle.

When LR *is* the culprit, **lower it**: a too-high LR near a minimum makes the optimizer overshoot
and **bounce around the basin** instead of descending into it. Decaying the step size lets it settle
— which is exactly why LR schedules (step/cosine decay) exist (see [[learning-rate]]).

**Confirmation test:** if decaying the LR produces a sudden **fresh drop** (the classic staircase at
each decay step), the plateau *was* LR-limited. If decaying does nothing, it was real convergence or
an underfitting/capacity issue.

---

## The Three Silent Failure Modes

### Dead neurons (dead ReLU)

ReLU outputs 0 for negative inputs. If a neuron's **pre-activation** `Wx+b` is negative for **every**
example in the data, it outputs 0 everywhere → its gradient is **permanently 0** → no update can ever
revive it. It's a **self-locking** failure. With bad initialization you can lose a large fraction of
the network this way.

**Causes:**
- **Bad initialization** (weights/bias put the neuron in the negative region from the start).
- **LR too high** — a single bad update can shove a *healthy* neuron so its pre-activation goes
  negative for all inputs, killing it mid-training.

**Fixes:**
- **Non-saturating-on-the-negative-side activations:** LeakyReLU, PReLU, ELU, GELU — they have a
  **nonzero slope (hence nonzero gradient) for negative inputs**, so a neuron in the negative region
  still gets gradient and can climb back out. LeakyReLU's small negative slope (~0.01) is the minimal
  version of this idea.
- **Lower the learning rate**; **better init** (Kaiming, see [[pytorch-nn-modules]]).

### Vanishing gradients

Gradient norms shrink **exponentially** as they flow backward; early layers barely learn.

**Cause 1 — saturating activations (by construction, independent of init).** Sigmoid/tanh flatten
for large |input|, so their derivative → 0 there. Sigmoid's derivative `σ(x)(1−σ(x))` **maxes at
0.25** (at x=0), so *even at best* each sigmoid layer shrinks the gradient ≥4×:
`0.25¹⁰ ≈ 10⁻⁶` over 10 layers before saturation even kicks in. This is why ReLU (positive-side
derivative = exactly 1, no attenuation) displaced them for deep hidden layers. tanh is somewhat
better (derivative maxes at 1, zero-centered) but still saturates.

**Cause 2 — poor initialization** (variance not preserved layer-to-layer).

**Fixes:** better init (**Kaiming** for ReLU), non-saturating activations, and the two architectural
features below.

### Exploding gradients

Gradient norms grow exponentially; weights diverge toward infinity and eventually **overflow to NaN**.

**Fixes:** **gradient clipping** (`torch.nn.utils.clip_grad_norm_`), lower LR, proper init,
normalization.

---

## Two Architectural Features That Keep Gradients Flowing

Beyond init and activation choice, deep stacks stay trainable because of:

1. **Skip / residual connections** (`out = F(x) + x`). Backprop gives
   `∂L/∂out · (∂F/∂x + 1)` — the **`+1` is a gradient highway**: even if `∂F/∂x` vanishes, gradient
   passes straight through the identity path, so it can't fully vanish across a block. Bonus: at init
   `F(x)≈0` → block ≈ identity → a deep net starts effectively shallow and grows depth as it learns.
2. **Normalization** (Layer/Batch/RMS). Re-centers/re-scales activations at each layer to stay
   well-conditioned (~unit variance) → keeps them out of the saturating/exploding regimes and smooths
   the loss landscape. See [[normalization]].

Together, these are *the* reason 100+ layer nets and deep transformers train at all.

---

## Health Check — What to Track (and How to Localize)

For each layer, track:
- **Activation std** — collapsing to 0 or exploding → fix initialization.
- **Gradient norm** — near 0 → vanishing; very large → exploding.
- **Dead-neuron fraction** — fraction of ReLU units that never activate over a batch/epoch.

*How to instrument these checks in PyTorch (forward/backward hooks, leaf-layer filtering, storing
detached scalars): see [[pytorch-hooks]].*

**Localizing *where* it breaks:** plot **per-layer gradient norm** (one number per layer) with the
**y-axis on a log scale**, layer index on x. A steady **downward slope from output → input layers**
*is* vanishing gradients made visible; a sharp upward spike is where things explode. This turns "the
net isn't learning" into "layers 0–3 get no gradient."

---

## The parameter-update-ratio heuristic (≈ 1e-3)

A quantitative health check beyond "is the gradient norm sane": track the **relative update size**
per parameter tensor:

$$\text{ratio} = \frac{\eta \, \lVert \nabla_\theta \mathcal{L} \rVert}{\lVert \theta \rVert}$$

i.e. *how big is the step relative to the weights it's changing.* It should sit around **`1e-3`**
(roughly: weights change by ~0.1% per step).

- **≫ 1e-3** (e.g. 1e-1) → learning rate **too high** — updates are large relative to the weights.
- **≪ 1e-3** (e.g. 1e-6) → learning rate **too low** — weights barely move, training crawls.

Compute it per layer/parameter after `backward()` (`p.grad.norm()` and `p.norm()`, guarding
`p.grad is not None` — see [[pytorch-hooks]]). This is Karpathy's rule and a fast way to sanity-check
an LR without a full sweep.

**Other quick instruments:**
- **Overfit one batch** (also in the recipe below) — the single best bug test.
- **Set random seeds** while debugging (weight init, shuffling, dropout masks) so a code change's
  effect is isolated from run-to-run randomness.
- **Extract scalars for logging** — accumulate `loss.item()`, never the loss *tensor*, or you leak the
  graph and OOM ([[training-memory]]).

---

## Debugging by reading the `.grad` state

Inspecting `p.grad` after `backward()` localizes the problem by **symptom**:

| `.grad` symptom | Cause |
|---|---|
| **`None`** *(after a backward that should fill it)* | parameter **not involved** in computing the loss — layer unused in forward, path **detached**, or gradient tracking **disabled** (`no_grad`). |
| **exactly `0`** | parameter **doesn't affect the loss** at the current point — classic **dead ReLU** (neuron's inputs all negative → output 0 → zero gradient → can't recover). |
| **`NaN` / `Inf`** | numerical instability (÷0, `log(0)`, overflow) — **propagates through the update and corrupts all subsequent steps**. |
| **loss not decreasing** | LR wrong (too high → oscillation, too low → plateau), **or** forgot `zero_grad()` → gradients accumulate across steps → effective grad grows → divergence. **Monitor gradient norm per step.** |
| **params not changing** | update not landing — likely **out-of-place** `w = w - lr*g` (rebinds the name to a *new* tensor; any holder of the original keeps the stale one) instead of **in-place** `w.sub_(lr*g)` (mutates the object everyone references). **Snapshot values before/after to verify** (see [[pytorch-basics]]). |

**`None` is not always a bug:** it's legitimately `None` **before the first `backward()`** and **right
after `zero_grad(set_to_none=True)`** (the modern default). Only `None` *after* a backward you
expected to populate it indicates the unused-param/detached/`no_grad` problem.

---

## Karpathy's "Recipe for Training Neural Networks" — Debugging Checklist

Andrej Karpathy's blog post lays out a systematic debugging process. Condensed:

1. **Look at the loss curve.** Is it going down? If not, check learning rate and gradients.
2. **Check activations.** Print the mean and std of activations at each layer. If they collapse to 0
   or explode, fix initialization.
3. **Check gradients.** Are they vanishing (norm near 0) or exploding (norm very large)? Use gradient
   clipping or better init.
4. **Look for dead neurons.** If many ReLU neurons never activate, your learning rate might be too
   high or init too extreme.
5. **Overfit one batch first.** Before training on the full dataset, make sure the model can
   *memorize a single batch* (drive its loss to ~0). If it can't, there's a bug — not a tuning issue.
6. **Add complexity gradually.** Start with a small model, verify it works, then scale up.

### Common failure modes (Karpathy)

- **Vanishing gradients:** gradients shrink to zero in early layers; the network stops learning.
  *Fix:* better init (Kaiming), skip connections, or layer normalization.
- **Exploding gradients:** gradients grow exponentially; loss becomes NaN.
  *Fix:* gradient clipping (`torch.nn.utils.clip_grad_norm_`), lower learning rate.
- **Dead ReLU:** neurons that always output 0 because their input is always negative; they stop
  learning because ReLU's gradient is 0 for negative inputs.
  *Fix:* LeakyReLU, lower learning rate, or better init.

> **The single most useful first move:** *overfit one batch.* A correct model+loss+optimizer should
> memorize a handful of examples to near-zero loss in seconds. If it can't, stop tuning and go find
> the bug — the problem is wiring, not hyperparameters.

---

## References

**Start here (practical):**
- **Karpathy, "A Recipe for Training Neural Networks"** (2019, karpathy.github.io) — the best
  practical debugging guide; source of the checklist above.
- **Karpathy, CS231n notes** — "Setting up the data and loss" / "Learning" sections: gradient checks,
  sanity checks (the `log C` baseline), babysitting loss/activation curves.

**Why the failure modes exist:**
- **Glorot & Bengio, "Understanding the difficulty of training deep feedforward neural networks"**
  (2010) — Xavier init; named and *measured* vanishing/exploding activations & gradients per layer.
  Origin of the "track activation/gradient variance per layer" mindset.
- **He et al., "Delving Deep into Rectifiers"** (2015) — Kaiming init, the ReLU-specific fix.
- **Bengio et al., "Learning long-term dependencies with gradient descent is difficult"** (1994) /
  **Hochreiter (1991)** — original vanishing-gradient analyses.

**The architectural fixes:**
- **He et al., "Deep Residual Learning"** (ResNet, 2015) — skip connections; the plain-net
  *degradation* figure is the canonical "gradients don't flow" evidence.
- **Ioffe & Szegedy, "Batch Normalization"** (2015) — original ICS motivation.
- **Santurkar et al., "How Does Batch Normalization Help Optimization?"** (2018) — shows it's
  **loss-landscape smoothing**, not ICS.

**Clipping / RNN-specific:**
- **Pascanu, Mikolov, Bengio, "On the difficulty of training RNNs"** (2013) — introduces **gradient
  clipping**; rigorous exploding-gradient analysis.
- **Frankle & Carbin, "The Lottery Ticket Hypothesis"** (2019) — tangential, sharpens init intuition.

*If you read just two: Karpathy's recipe (the* how*) and Glorot & Bengio (the* why the numbers blow
up*).*

---

Related: [[normalization]], [[learning-rate]], [[pytorch-nn-modules]], [[overfitting-underfitting]],
[[loss-functions]], [[optimization]]
