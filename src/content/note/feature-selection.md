---
title: "Feature Selection: Removing Low-Variance / Uncorrelated Features"
description: "low-variance / uncorrelated feature removal, interpreting `np.var`, univariate caveats, mutual info, model-based selection"
category: "Generalization & Model Fitting"
order: 27
updatedDate: "2026-07-15T08:29:46.641Z"
---
Filtering out uninformative features controls model complexity → helps fight overfitting
(see [[regularization]] "Other Approaches to Overfitting").

---

## 1. Low-Variance Features

A feature that barely varies carries almost no information. Compute variance **per column**
(axis 0) and keep those above a threshold.

```python
variances = X.var(axis=0)      # shape (n_features,)
mask = variances > threshold    # boolean mask
X_filtered = X[:, mask]         # keep passing columns
```

**Critical caveat — variance is scale-dependent.** A feature in millimeters has 10⁶× the
variance of the same feature in kilometers. **Standardize first** (or use a relative
threshold), otherwise the threshold is meaningless across features with different units.

**sklearn:**
```python
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)
X_filtered = selector.fit_transform(X)
```

### Caveat: low variance ≠ uninformative

Variance filtering is **unsupervised** — it never looks at the target `y`. It removes features
that don't *vary*, not features that don't *predict*. Those are different questions.

- **Low variance can still be highly predictive.** A binary feature that's 1 for only 2% of
  samples has tiny variance but may perfectly identify a rare class → variance filtering would
  wrongly discard it.
- **High variance can be pure noise.** A maximally-spread feature with zero predictive value
  survives the filter.

**Mental model:**
- **Variance filter** → "does this feature carry *any* signal at all?" (unsupervised, cheap, crude)
- **Correlation / MI / model-based** → "does this feature help predict *the target*?" (supervised — what you actually care about)

**Use variance thresholding as a cheap first-pass cleanup** to drop obviously-dead features
(constants, near-constants), *not* as your primary feature selector. For selecting *predictive*
features, use a supervised method (below).

### The only universally safe threshold: exactly-zero variance

Dropping features with **zero variance** (constants) is the one variance-based filter with **no
risk** of discarding signal:
- A constant feature carries **zero predictive information** by definition.
- It can **break models** — e.g. a constant column makes `XᵀX` singular, so the OLS normal
  equations `(XᵀX)⁻¹` blow up (see [[regression-metrics]]).

The moment you raise the threshold *above* zero ("drop *low* variance"), you're making a
**judgment call** — a low-but-nonzero-variance feature might be a rare-but-predictive signal.
Validate against the target / via cross-validation before dropping those.

```python
mask = X.var(axis=0) > 0          # always safe: drop only exact constants
mask = X.var(axis=0) > 0.01       # judgment call: might discard useful rare signals
```

**Float subtlety:** a truly constant column may compute to a tiny nonzero variance (e.g. `1e-17`)
due to rounding — use a small tolerance instead of literal `== 0`:
```python
mask = X.var(axis=0) > 1e-10      # treat near-zero as constant
```

**Rule of thumb:** zero (~zero) variance → always drop, no judgment needed. Low-but-nonzero
variance → judgment call, validate against the target first.

---

## 2. Features Uncorrelated with the Target

Needs the target `y`. Compute each feature's correlation with `y`, drop the weak ones.

```python
corrs = np.array([np.corrcoef(X[:, j], y)[0, 1] for j in range(X.shape[1])])
mask = np.abs(corrs) > threshold    # abs — strong negative correlation counts too
X_filtered = X[:, mask]
```

**Two gotchas:**
- Use `np.abs()` — correlation of **−0.9** is highly informative, not weak.
- `np.corrcoef` measures **linear** correlation only. A feature with a strong *nonlinear*
  relationship to `y` (e.g. quadratic) can have near-zero correlation yet be very useful.
  For nonlinear dependence use **mutual information**
  (`sklearn.feature_selection.mutual_info_regression`).

**sklearn:**
```python
from sklearn.feature_selection import SelectKBest, f_regression
selector = SelectKBest(f_regression, k=10)   # top-k by correlation-based F-score
X_filtered = selector.fit_transform(X, y)
```

---

## Important Caveat: Univariate Filtering Misses Interactions

Filtering features *one at a time* by correlation with the target is a **univariate** method —
it ignores **feature interactions**:
- A feature useless alone can be valuable *combined* with another.
- Two features individually correlated with `y` may be **redundant with each other**.

For feature-feature **redundancy**, inspect the correlation matrix and drop one of each
highly-correlated pair:
```python
corr_matrix = np.corrcoef(X, rowvar=False)   # rowvar=False → columns are variables
```

For interaction-aware selection, prefer **model-based** methods (e.g. L1/Lasso coefficients,
tree feature importances, or recursive feature elimination) over univariate filters.

---

## Interpreting Variance Values (np.var)

Variance = **average squared deviation from the mean** — how spread out the values are.

$$\text{Var}(x) = \frac{1}{N}\sum_i (x_i - \bar{x})^2$$

| Value | Meaning |
|---|---|
| **0** | all values identical — constant feature, zero information, safe to drop |
| **small** | values cluster tightly around the mean |
| **large** | values widely spread |

**Trap 1 — units are squared.** A feature in meters has variance in **meters²**, which is hard
to reason about. **Standard deviation** (`np.std` = √variance) is more interpretable — it's back
in the original units ("typical deviation from the mean").

```python
np.var(x)   # meters²  — hard to interpret
np.std(x)   # meters   — same units as the data
```

**Trap 2 — raw variance is NOT comparable across features.** It depends on scale *and* units:

```python
# height in meters:   var ≈ 0.01          (tiny)
# income in dollars:  var ≈ 25_000_000    (huge)
```

Income only looks "more variable" because dollars are a larger-magnitude unit — not because it's
genuinely more spread relative to its own scale. This is exactly why variance-threshold selection
requires **standardizing first**.

**Scale-free alternative — coefficient of variation** (std relative to mean, dimensionless):
```python
cv = np.std(x) / np.mean(x)   # only meaningful for positive-mean data
```

**numpy gotcha — population vs sample:**
```python
np.var(x)           # 1/N     — population (default)
np.var(x, ddof=1)   # 1/(N-1) — sample, unbiased (Bessel's correction)
```
Mirror of the PyTorch quirk (torch defaults to *unbiased*; numpy defaults to *population*) — see
[[numpy-basics]].

Related: [[regularization]], [[regression-metrics]], [[numpy-basics]]
