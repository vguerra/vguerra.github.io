---
title: "Preprocessing: Fit on Train, Apply Everywhere"
description: "fit on train / apply everywhere, fit vs transform, data leakage, pipelines, CV"
category: "Generalization & Model Fitting"
order: 28
updatedDate: "2026-07-14T14:17:10.320Z"
---
One of the most important (and most tested) principles in applied ML:

> **Any transformation you fit on training data must be frozen and reapplied *identically*
> to validation, test, and inference data.**

Feature selection, scaling, imputation, encoding — all are **preprocessing transformations**
subject to this rule.

---

## The Rule

Once a transformation's parameters are learned from training data, that decision becomes a
**fixed part of the pipeline**:

- **Validation** → apply the same transformation
- **Test** → apply the same transformation
- **Inference / production** → apply the same transformation

If you don't, the model receives inputs with the wrong shape / feature ordering / scale — it was
trained to expect a specific set of features in a specific form. Mismatch → broken or meaningless
predictions.

---

## The Critical Part: WHERE You Compute the Statistics

**Compute all statistics on the training set only. Never recompute on val/test.**

Example — variance-threshold feature selection: decide *which columns to drop* using **training
data variances**, then drop those *same* columns everywhere.

Two reasons you must not recompute on val/test:

1. **Consistency** — recomputing on val might select a *different* column set than train, leaving
   mismatched features. The model breaks or its inputs become meaningless.
2. **Data leakage** — computing statistics on val/test lets evaluation-set information influence
   your pipeline, making val/test scores **optimistically biased** (they no longer simulate truly
   unseen data). At inference you often have a **single sample**, where a variance can't even be
   computed.

---

## Mental Model: fit vs transform

sklearn splits every preprocessor into two methods for exactly this reason:

```python
selector.fit(X_train)          # LEARN parameters (which columns to drop) — TRAIN ONLY
selector.transform(X_train)    # APPLY
selector.transform(X_val)      # APPLY the same decision — no re-fitting
selector.transform(X_test)     # APPLY the same decision
```

- **`fit`** — learn parameters (columns to drop, mean/std, fill values). **Train only.**
- **`transform`** — apply the learned decision. **Everywhere.**
- **`fit_transform`** — both combined; use **only on the training set**.

Same principle for `StandardScaler` (mean/std from train), imputation (fill values from train),
encoders (categories from train), etc.

---

## The Safe Way: Pipelines

Wrap all steps in a single pipeline so you *cannot* accidentally re-fit on val/test:

```python
from sklearn.pipeline import Pipeline
pipe = Pipeline([
    ("var", VarianceThreshold(0.0)),
    ("scale", StandardScaler()),
    ("model", ...),
])
pipe.fit(X_train, y_train)   # every step fits on TRAIN only
pipe.predict(X_val)          # every step transforms consistently
```

With cross-validation, put the pipeline *inside* the CV loop so preprocessing is re-fit on each
fold's training portion only — otherwise stats leak across folds.

---

## Interview one-liner

*"Preprocessing parameters are learned from training data and frozen; validation, test, and
inference only apply them. Recomputing on evaluation data causes leakage and inconsistent
features."*

Related: [[feature-selection]], [[regression-metrics]], [[overfitting-underfitting]]
