---
title: "Evaluation Metrics"
description: "precision/recall/F1, accuracy-under-imbalance, macro/micro/weighted F1, confusion-matrix worked example + raising recall, κ/MCC, regression (MAE/RMSE/MAPE/R²), distribution divergences (KL/JS/TV/Wasserstein/NLL)"
category: "Generalization & Model Fitting"
order: 33
updatedDate: "2026-09-10T21:16:33.396Z"
---
## Classification metrics

From the confusion matrix (TP/FP/FN/TN):
- **Precision** `TP/(TP+FP)` — of predicted positives, how many are right (few false alarms).
- **Recall / sensitivity** `TP/(TP+FN)` — of actual positives, how many caught (few misses).
- **F1** `2·P·R/(P+R)` — harmonic mean; punishes trading one for the other.

**Accuracy is misleading under imbalance** — a 99.99%-accurate "always predict majority" model can have
**zero recall** (see [[class-imbalance]]). Use F1 / AUROC / AUPRC instead.

**F1 for multi-class** (one-vs-all per class, then combine):
- **Macro** — unweighted mean over classes (every class equal, even tiny ones).
- **Micro** — pool all TP/FP/FN then compute once (= accuracy for single-label multi-class).
- **Weighted** — macro weighted by class **support** (imbalance-aware).
- **Per-class** — report each separately.

**Worked example:** confusion matrix `TP=30, FN=20, FP=5, TN=40` → Precision `30/35=0.857`,
Recall `30/50=0.6`, F1 `≈0.705`. "Precise but misses positives" → to **raise recall**: lower the
decision threshold, penalize false negatives more in the loss, resample, add features, use a stronger
model. Other metrics: **Cohen's κ** (agreement corrected for chance), **MCC** (uses all four cells,
`∈[−1,1]`, balanced for binary).

## Regression metrics

- **MAE** `(1/n)Σ|y−ŷ|` — linear penalty, **robust to outliers**, median-like, interpretable.
- **RMSE** `√((1/n)Σ(y−ŷ)²)` — **penalizes large errors more** (L2), same units as target; use when
  outliers matter.
- **MAPE** `(1/n)Σ|y−ŷ|/|y| ·100` — average **percentage** error (watch division by small `y`).
- **R²** — fraction of variance explained (see [[regression-metrics]]).

**Choosing loss for the task:** log-loss over MSE for logistic regression (probabilities, not
continuous — [[logistic-regression]]); cross-entropy over MSE for multi-class (sharper gradients, no
saturation — [[loss-functions]]).

## Comparing distributions

How close is the model's `Q` to the data's `P`?
- **KL divergence** `D_KL(P‖Q)` — info lost approximating `P` with `Q`; **asymmetric**.
- **Jensen-Shannon** — symmetric, bounded version of KL.
- **Total variation distance**.
- **Wasserstein (earth-mover)** — cost to transform one distribution into the other; well-behaved even
  for non-overlapping supports (used in WGANs, drift detection).
- **Negative log-likelihood** — how well `Q` explains observed data.

---

Related: [[class-imbalance]], [[loss-functions]], [[regression-metrics]], [[cross-validation]], [[classification-vs-regression]]
