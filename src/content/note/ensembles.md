---
title: "Ensembles — Bagging, Boosting, Stacking"
description: "bagging (variance ↓, Random Forest = bagging + feature randomness), boosting (bias ↓, sequential, Gradient Boosting/XGBoost), stacking, averaging/voting, bagging-vs-boosting contrast"
category: "Classical ML"
order: 40
updatedDate: "2026-09-10T21:12:12.460Z"
---
Combining models generally improves performance by **reducing variance**, **compensating individual
errors**, and **generalizing better** — provided the models are **diverse** (uncorrelated errors),
independently trained, and each better than random.

## Bagging (Bootstrap Aggregating)

**Reduces variance** / avoids overfitting. Sample the training set **with replacement** → many
independent **bootstraps** → train a model on each → aggregate:
- **Classification** → majority vote.
- **Regression** → average.

Helps **unstable** learners (deep trees, NNs); can *hurt* stable ones (KNN). **Random Forest** = bagging
of decision trees **+ feature randomness** (each split considers a random subset of features → more
decorrelated trees). Handles imbalance and noise well; used in fraud/disease/finance.

## Boosting

**Reduces bias** (and variance). Train learners **sequentially**, each focusing on the examples the
previous ones **got wrong** (reweighted or fit to residuals):
1. Train weak learner 1.
2. Reweight samples (up-weight misclassified).
3. Train learner 2 on the reweighted data; add to the ensemble.
4. Repeat.

**Gradient Boosting** (XGBoost/LightGBM) fits each new tree to the **residual/gradient** of the current
ensemble. Strengths: mixed data types, non-linear relations, little feature engineering, handles
missing/outliers; strong on **tabular/structured** data, imbalanced sets, ranking. Can **overfit** if
too aggressive.

## Stacking

Combine models using **another model** (a meta-learner) trained on the base models' outputs.

## Averaging / voting

Simple average (regression) or majority vote (classification).

---

## Bagging vs boosting (the contrast)

| | Bagging | Boosting |
|---|---|---|
| Target | **variance** ↓ | **bias** ↓ (and variance) |
| Training | **parallel**, independent | **sequential**, dependent |
| Risk | can underfit if bases too simple | can **overfit** (too aggressive) |
| Example | Random Forest | XGBoost / Gradient Boosting |

Connects to the [[overfitting-underfitting]] bias-variance framing: **bagging lowers variance without
raising bias much; boosting can reduce both.**

---

Related: [[classical-ml-models]], [[overfitting-underfitting]], [[feature-selection]], [[class-imbalance]]
