---
title: "Class Imbalance & Weighted Sampling"
description: "why accuracy misleads + metrics (precision/recall/F1/AUROC/AUPRC/MCC/κ), sampling strategies (over/under/SMOTE), `WeightedRandomSampler` math (`w_c=1/n_c` derivation, construction pattern, replacement/0.632), class-weighted & focal loss, sampling↔loss-weighting equivalence, stratified vs balanced, calibration caveat, BN interaction"
category: "Generalization & Model Fitting"
order: 29
updatedDate: "2026-08-26T10:29:12.465Z"
---
Skewed class distributions (fraud 99.9% legit, rare-disease imaging, spam) break naive training:
uniform sampling produces gradients **dominated by the majority class**, so the model minimizes
average loss by neglecting the minority and shifts the decision boundary toward it. A model that always
predicts the majority class gets high **accuracy** but is useless.

> **950 / 50 example:** always-predict-class-0 → **95% accuracy**, but **0 recall, 0 precision, F1=0**
> on the positive class.

---

## Why accuracy misleads — better metrics

$$\text{Accuracy} = \frac{TP+TN}{TP+TN+FP+FN}$$

The denominator is dominated by the majority class → disproportionate credit for majority-correct
predictions. Use instead:

- **Precision** `TP/(TP+FP)` — of predicted-positive, how many are right (few false alarms).
- **Recall / sensitivity** `TP/(TP+FN)` — of actual-positive, how many caught (few misses).
- **F1** `2·P·R/(P+R)` — harmonic mean; punishes sacrificing one for the other.
- **AUROC** — ability to rank positives above negatives across all thresholds; 0.5 random, 1.0 perfect;
  threshold-independent. Can look **deceptively high** under heavy imbalance.
- **AUPRC** — precision-recall area; directly measures the **rare-positive** operating region → the
  metric to trust when the positive class is very rare.

---

## Sampling strategies

| Strategy | Idea | Risk / note |
|---|---|---|
| **Oversampling** | draw minority more often (with replacement); in 950/50 each minority sample seen ~19×/epoch | **overfits** minority (memorization) → mitigate with **augmentation** on the repeats |
| **Undersampling** | drop majority to match minority (950/50 → 50/50, 100 total) | fast, but **discards 900 majority samples** → info loss; OK if majority is dense |
| **SMOTE** | synthesize minority points by interpolating: `x_new = x + λ(x' − x)`, `x'` a k-NN minority neighbor, `λ ~ U(0,1)` | enriches feature space vs repeating; **fails in high-dim** (images/text) — interpolation leaves the data manifold; a **preprocessing** step, not in the DataLoader |

---

## Weighted random sampling — the math

`C` classes, class `c` has `n_c` samples, `N = Σ n_c`. Give every sample in class `c` weight `w_c`.
Probability of drawing sample `i` in one draw: `P(i) = w_i / Σ_j w_j`. Expected class-`c` count in `N`
draws:

$$\mathbb{E}[\text{count}_c] = N \cdot \frac{n_c\, w_c}{\sum_k n_k\, w_k}$$

For **equal representation**, set **`w_c = 1/n_c`**: then `n_c · w_c = 1` for every class, `Σ_k n_k w_k
= C`, so each class gets `N/C` samples/epoch. In the 950/50 case `w_0 = 1/950 ≈ 0.00105`, `w_1 = 1/50 =
0.02` → minority drawn ~**19×** more often, exactly cancelling the 19:1 imbalance.

### Alternative weight schemes
- **Temperature-scaled:** `w_c = 1/n_c^α`, `α ∈ [0,1]`. `α=0` → uniform, `α=1` → full inverse-freq,
  in-between = smooth interpolation. Useful when the minority is *tiny* and full balancing would
  over-oversample.
- **Effective number** (Cui et al. 2019): each extra same-class sample adds diminishing info →
  `E_n = (1 − β^n)/(1 − β)`, weight `w_c = 1/E_{n_c}`, `β ∈ [0,1)`. `β=0` → inverse-freq; `β→1` →
  uniform.

---

## `WeightedRandomSampler` in PyTorch

Params: **`weights`** (N non-negative floats, one **per sample**; needn't sum to 1 — normalized
internally), **`num_samples`** (indices drawn per epoch; `=N` keeps epoch length), **`replacement`**
(**must be `True`** for oversampling — you draw minority samples more times than they exist). Internally
uses **multinomial** sampling.

**Construction pattern** (matches the Balanced DataLoader exercise):
```python
label_counts  = torch.bincount(labels)      # per-class counts
class_weights = 1.0 / label_counts          # w_c = 1/n_c   (guard against count 0 → inf)
sample_weights = class_weights[labels]       # KEY STEP: per-class → per-sample by fancy-indexing
sampler = WeightedRandomSampler(sample_weights, num_samples=len(labels), replacement=True)
DataLoader(dataset, batch_size=..., sampler=sampler)   # NOT shuffle=True (sampler owns ordering)
```
The `class_weights[labels]` fancy-index is the crux — it broadcasts per-class weights out to every
sample. See [[dataloader-and-batching]] (sampler hierarchy) and [[tensor-indexing]].

### Replacement implications
Drawing `N` with replacement from `N`: `E[unique] = N(1 − (1−1/N)^N) ≈ 0.632N` → **~37% of samples
missed each epoch** (more for low-weight majority, fewer for high-weight minority). **A feature, not a
bug** — the goal is balance, not full coverage; every sample is seen over many epochs, and the
stochasticity is a mild regularizer. `replacement=False` → each index at most once → **can't
oversample** (useful for stratified/proportional draws).

---

## Per-sample vs per-class weights

The sampler is fundamentally **per-sample**; class-balancing just gives all same-class samples the same
weight. Per-sample weighting also enables: **difficulty** (upweight often-wrong samples — hard-example
mining / curriculum), **data quality** (downweight noisy/mislabeled), **recency** (upweight recent for
distribution shift). Class→sample conversion is one indexing op.

---

## Alternative: reweight the loss instead of the sampler

**Class-weighted loss:** `L = −w_c · log ŷ_c`. Inverse-freq weights give ~the same expected per-class
gradient as balanced sampling **over a uniform epoch** — but only **approximately** equivalent:
weighted loss **scales each sample's gradient magnitude**, weighted sampling **changes which samples are
seen**. They interact differently with **BatchNorm, momentum, and regularization**. Both work; can be
combined.

**Focal loss** — weight by *confidence*, not frequency:

$$L_{\text{focal}} = -(1 - \hat{y}_c)^\gamma \log \hat{y}_c$$

Confident (`ŷ_c → 1`) → `(1−ŷ_c)^γ → 0` → tiny loss; uncertain → factor ≈ 1 → full loss. Auto-downweights
**easy** examples (mostly majority), upweights **hard** ones (often minority/ambiguous). Combine with
class weights: `L = −α_c (1−ŷ_c)^γ log ŷ_c` (RetinaNet). See [[loss-functions]].

### Sampling ↔ loss-weighting relationship
- Uniform sampling + weighted loss: `E[∇L] = Σ_c Σ_{i∈c} w_c ∇ℓ_i`.
- Weighted sampling (`w_c=1/n_c`) + unweighted loss: `E[∇L] ∝ Σ_c (1/n_c) Σ_{i∈c} ∇ℓ_i = Σ_c \overline{∇ℓ}_c`.

For **single-sample SGD** the expected gradients are **proportional**. For **minibatch** SGD, batch
**composition** differs → different gradient **variance** → different dynamics.

---

## Stratified ≠ balanced

- **Balanced** sampling → **equal** class representation per epoch (fixes imbalance).
- **Stratified** sampling → **preserves natural proportions** per batch (95/5 stays 95/5).

Stratified doesn't *fix* imbalance — it preserves it — but guarantees every batch has *some* minority
samples, stabilizing gradient estimates. Best for **evaluation** (stable metrics) and
**train/val splitting**. PyTorch has no built-in stratified sampler; implement via per-class index
lists interleaved.

---

## Practical considerations

**Use balanced sampling when:** moderate–severe imbalance (worse than ~1:10); the minority is the class
you care about; dataset large enough that repeats aren't memorized.

**Avoid it when:** the imbalance is the **true deployment distribution** and you need **calibrated
probabilities** — balancing **distorts probability estimates** (model expects 50/50, meets 95/5 in
prod). Alternative: train on natural distribution + **recalibrate** (Platt / temperature scaling).

- **Verify it works:** iterate one epoch, tally labels — counts should be ~equal (up to noise).
- **BatchNorm interaction:** balanced batches → batch stats reflect the *balanced* distribution, but
  inference uses running stats → train/infer mismatch. Minor for moderate imbalance; for extreme cases
  prefer **Group/Layer norm** ([[normalization]]).
- **Multi-label:** "class count" is ill-defined → sampling-based balancing is hard; prefer loss
  weighting (BCE **`pos_weight`**).
- **Always validate on the *natural* distribution** — rebalance train only, so hyperparameters (α, γ,
  oversampling ratio) optimize real-world performance.

---

## Deeper evaluation metrics

- **Confusion matrix** `C×C` — entry `(i,j)` = true `i` predicted `j`; diagonal correct, off-diagonal
  reveals systematic confusions.
- **Macro** average — per-class metric, unweighted mean → every class equal regardless of size.
- **Micro** average — aggregate all classes then compute; micro-precision = accuracy for multi-class.
- **Weighted** average — weight each class metric by its support (macro/micro compromise).
- **Cohen's κ** — agreement corrected for chance (0 = chance, 1 = perfect).
- **MCC** (Matthews) — uses all four confusion cells, `∈ [−1,1]`, symmetric in pos/neg → one of the
  most balanced binary metrics.

---

## Combining approaches (best on severe imbalance)

1. **Balanced sampling / moderate oversampling** so the model sees minority often.
2. **Augment** the oversampled minority to cut memorization.
3. **Class-weighted or focal loss** to further emphasize hard/minority samples in the gradient.
4. **Evaluate with class-balanced metrics** (macro-F1, AUPRC), not accuracy.
5. **Tune** α (weight scheme), γ (focal), oversample ratio on a **naturally-distributed** val set.

---

Related: [[dataloader-and-batching]], [[loss-functions]], [[tensor-indexing]], [[normalization]],
[[preprocessing-fit-transform]], [[regression-metrics]]
