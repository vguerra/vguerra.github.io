---
title: "Regression: OLS and R²"
description: "OLS (normal equations), R² definition/interpretation, negative R², adjusted R², polynomial fitting (`np.polyfit` / `Polynomial.fit`)"
category: "Generalization & Model Fitting"
order: 16
updatedDate: "2026-07-24T21:10:30.978Z"
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

---

## OLS for Polynomial Fitting (nonlinear-in-x, linear-in-coefficients)

Fitting `y = a·x² + b·x + c` is **still OLS**. The key insight: OLS requires linearity in the
**coefficients**, not in `x`. So a parabola (curved in `x`) is a *linear* model in `(a, b, c)` —
you just build a design matrix whose columns are the powers of `x`:

$$X = \begin{bmatrix} x_1^2 & x_1 & 1 \\ x_2^2 & x_2 & 1 \\ \vdots & \vdots & \vdots \\ x_n^2 & x_n & 1 \end{bmatrix}, \quad \beta = \begin{bmatrix} a \\ b \\ c \end{bmatrix}$$

then solve the same normal equations `β̂ = (XᵀX)⁻¹Xᵀy`.

### The tool: `np.polyfit`

```python
coeffs = np.polyfit(x, y, deg=2)      # returns [a, b, c] — HIGHEST power first
a, b, c = coeffs

p = np.poly1d(coeffs)                  # callable polynomial
y_hat = p(x_new)                       # or np.polyval(coeffs, x_new)
```

- It builds the Vandermonde matrix `[x², x, 1]` and solves via **SVD/`lstsq`** — numerically safer
  than literally forming `(XᵀX)⁻¹` (avoids squaring the condition number).
- **Coefficient order is descending** (`[a, b, c]`), a common off-by-order bug.

### Modern replacement: `numpy.polynomial.Polynomial`

```python
from numpy.polynomial import Polynomial
p = Polynomial.fit(x, y, deg=2)        # coeffs ASCENDING here; rescales x internally
```

`Polynomial.fit` rescales the `x` domain before fitting → better conditioning when x spans a wide
range (e.g. compute in FLOPs, or log-compute for a local scaling-law fit — see [[scaling-laws]]).

### Special cases (conditioning intuition)

- **`n` = 3 points, `deg` = 2** (3 unknowns, 3 equations): the parabola passes through all points
  **exactly** → residual sum of squares = 0. "Least squares" degenerates to exact interpolation;
  `XᵀX` is square and (if x-values are distinct) invertible.
- **Duplicated x-values** with different y: two identical rows in `X` → `X` is **rank-deficient**,
  `XᵀX` singular. `polyfit`/`lstsq` still return the minimum-norm solution (via SVD) but you can't
  interpolate contradictory targets — OLS instead averages them in the least-squares sense.
- This is the same singular-matrix failure mode as zero-variance / perfectly-collinear features in
  plain linear regression (see [[feature-selection]]).

### Why `a > 0` gives a unique minimum

For the fitted `L(x) = ax² + bx + c`: `dL/dx = 2ax + b = 0 ⟹ x* = -b/(2a)`, and the second
derivative `2a > 0` confirms it's a minimum (strict convexity). Useful when fitting a local
quadratic to loss-vs-(log-)compute to locate an optimal operating point.

Related: [[overfitting-underfitting]], [[regularization]], [[feature-selection]], [[scaling-laws]]
