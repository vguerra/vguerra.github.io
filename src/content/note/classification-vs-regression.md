---
title: "Classification vs Regression"
description: "output/goal/loss/metric differences, measuring confidence (softmax/entropy/MC-dropout vs prediction intervals), converting between (threshold vs binning + risks)"
category: "Generalization & Model Fitting"
order: 31
updatedDate: "2026-09-10T21:16:06.240Z"
---
| | Classification | Regression |
|---|---|---|
| **Output** | discrete label (finite classes) | continuous value |
| **Goal** | which category? | what numeric value? |
| **Loss** | cross-entropy, hinge | MSE, MAE |
| **Metrics** | accuracy, F1, AUC | RMSE, R², MAE |

## Measuring confidence

- **Classification** — `max(softmax)` as confidence; **entropy** of the output (lower → more confident);
  **MC dropout** (dropout at inference to estimate uncertainty).
- **Regression** — **confidence/prediction intervals** (the range the true value likely falls in).

## Converting between them

**Classification → regression:** predict a continuous score + threshold (sigmoid → threshold at 0.5 for
binary; regress toward 0/1 targets, argmax for multi-class). Useful when you want **uncertainty
estimates**, regression-based models, or to combine losses (multi-task). Risk: regression may **ignore
class boundaries** → poor accuracy.

**Regression → classification:** **bin** the continuous output into buckets (e.g. age → 0–18/19–35/…),
turning it into ordinal/multi-class. Useful when exact values are noisy/irrelevant, you care about
**ranges**, or to handle outliers/imbalance. Risk: **discretization loses information**/resolution, and
**ordinal** ordering can be lost in standard classification.

---

Related: [[loss-functions]], [[logistic-regression]], [[evaluation-metrics]], [[class-imbalance]]
