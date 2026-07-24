---
title: "Regression: OLS and R²"
description: "OLS (normal equations), R² definition/interpretation, negative R², adjusted R²"
category: "Generalization & Model Fitting"
order: 15
updatedDate: "2026-07-14T12:04:52.856Z"
---
## OLS — Ordinary Least Squares

The standard method for fitting linear regression: find coefficients that **minimize the sum of squared residuals**.

$$\hat{\beta} = \arg\min_{\beta} \sum_i (y_i - x_i^T\beta)^2$$

- **"Ordinary"** distinguishes it from variants:
  - **WLS** (weighted least squares) — weights per observation
  - **GLS** (generalized least squares) — correlated / heteroscedastic errors
  - **Ridge** (L2-regularized), **Lasso** (L1-regularized)
- **Closed-form solution** (normal equations) — no iterative optimization needed:

$$\hat{\beta} = (X^TX)^{-1}X^Ty$$

- Because OLS *minimizes* SS_res, it is literally the fit that **maximizes R²** on the training data.

---

## R² — Coefficient of Determination

Measures the fraction of variance in the target that the model explains, **relative to a mean-predictor baseline**.

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$

- **SS_res** (residual sum of squares) — error your model makes
- **SS_tot** (total sum of squares) — error of just predicting the mean `ȳ` (= variance × n)

Intuition: *"What fraction of the baseline's error did I eliminate?"*

### Interpretation by value

| R² | Meaning |
|---|---|
| **1.0** | Perfect fit — explains all variance (SS_res = 0) |
| **0.7** | Explains 70% of the variance in the target |
| **0.0** | No better than predicting the mean |
| **< 0** | *Worse* than predicting the mean (SS_res > SS_tot) |

### Key points interviewers probe

1. **R² can be negative** — if the model fits worse than the horizontal line `ȳ`. Common on a *test* set when the model doesn't generalize.
2. **Baseline-relative, not absolute** — R² doesn't tell you if predictions are good in real units. Pair it with **RMSE / MAE** for absolute error.
3. **R² never decreases when adding features** (on training data), even useless ones → use **adjusted R²**, which penalizes predictors that don't earn their keep.
4. **Regression-only**, and assumes the mean is a sensible baseline.

### Diagnostic uses

- **R² high on train, low on test** → overfitting (see [[overfitting-underfitting]]).
- **High R² isn't automatically good** — always check residual plots for structure (patterns in residuals mean the model is missing something, e.g. nonlinearity), and watch for outliers inflating/deflating the score.

---

## Adjusted R²

$$R^2_{adj} = 1 - (1 - R^2)\frac{n - 1}{n - p - 1}$$

where `n` = samples, `p` = number of predictors. Penalizes added features that don't improve fit enough — can **decrease** when a useless feature is added, unlike plain R².

Related: [[overfitting-underfitting]], [[regularization]]
