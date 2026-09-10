---
title: "ML Models in Production — Drift, Degradation & Adaptation"
description: "why models degrade (covariate/concept/label drift, training-serving skew, feedback loops, staleness), drift detection (PSI/KS/KL/Wasserstein/chi-sq/classifier-based), breaking feedback loops (explore-exploit, counterfactual eval), domain adaptation (feature alignment, MMD, pseudo-labeling, fine-tuning), \"great on test poor in prod\" diagnosis"
category: "ML Systems / Production"
order: 76
updatedDate: "2026-09-10T21:10:36.270Z"
---
Models degrade after deployment because the world changes out from under them. The failure modes, how to
detect them, and how to adapt.

## Why models degrade

- **Data / covariate shift** — the input distribution `P(X)` changes (new user behavior, new sensors,
  evolving slang). Model sees unfamiliar inputs → worse predictions.
- **Concept drift** — the **relationship** `P(Y|X)` changes (fraud tactics evolve, user preferences
  shift). The mapping the model learned is now wrong.
- **Label drift** — the label distribution `P(Y)` changes.
- **Data-quality issues** — noisier/incomplete/misformatted production data; pipeline mismatches; API
  changes.
- **Training–serving skew** — features computed **differently** at inference than in training (different
  preprocessing code, mishandled time-based features). The subtle, common killer.
- **Feedback loops** — the model's own predictions change future inputs (recommender promotes items →
  only those get clicked → reinforced; fraud model blocks txns before they're labeled → biased training
  data).
- **Latency/resource constraints** — a simplified/quantized model is deployed, hurting accuracy.
- **No retraining/monitoring** — the model simply goes stale.

**Diagnosing "great on test, poor in prod":** compare train/test vs prod distributions (PSI, KS test);
check for a too-easy or leaky test set; log & compare prod features to the training pipeline
(training-serving skew); check whether the *deployed* (quantized/simplified) model differs from the
evaluated one.

---

## Drift detection

| Drift type | What changes |
|---|---|
| **Covariate** | input features `P(X)` |
| **Label** | label distribution `P(Y)` |
| **Concept** | relationship `P(Y|X)` |

**Process:** log features/predictions/labels → pick a **reference** (training or golden-validation set)
→ compare new data to reference with statistical tests → threshold + alert → act (retrain/fine-tune).

**Tests:** **KL divergence** & **Wasserstein distance** (distribution shift), **PSI** (Population
Stability Index, bin-based), **KS test** (univariate continuous), **Chi-squared** (categorical),
**classifier-based** (train a model to tell old vs new data — if it can, they differ).

---

## Breaking feedback loops

The danger: predictions influence behavior → behavior retrains the model → **bias amplifies**. Mitigations:
- **Explore/exploit** — ε-greedy, Thompson sampling, UCB: occasionally show non-top items to collect
  unbiased data.
- **Counterfactual / off-policy eval** — inverse propensity scoring to reweight logged data.
- **Randomized A/B tests** on small segments; **log the serving policy** (which model made each decision).
- **Diversification / novelty boosts**; model long-term preferences, not just clicks.
- **Delay online learning** — buffer recent data so the model doesn't instantly reinforce its own outputs.

---

## Domain adaptation

Adapt a model trained on a **source** distribution to a different **target** distribution (reduces need
for labels in every domain):
- **Feature alignment** — learn representations where source & target look similar (Domain-Adversarial
  NNs).
- **Distribution matching** — minimize Maximum Mean Discrepancy / Wasserstein between domains.
- **Self-training / pseudo-labeling** — predict on target, use confident predictions as labels.
- **Fine-tuning** — pretrain on source, fine-tune on a small labeled target set.
- **Augmentation / simulation** — make source data look more like target.

---

## Best practices

Monitor performance + drift metrics; validate incoming data (reject bad inputs); unify training &
serving feature pipelines (kills skew); retrain on fresh data regularly; use online/active/continual
learning where labels are scarce. **Always validate on the natural distribution**, not a rebalanced one
([[class-imbalance]]).

---

Related: [[class-imbalance]], [[preprocessing-fit-transform]], [[model-serving]], [[overfitting-underfitting]],
[[dataloader-and-batching]]
