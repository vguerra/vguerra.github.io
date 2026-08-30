---
title: "Linear Algebra Basics"
description: "singular matrices (equivalent characterizations, why they break OLS), near-singular / condition number, detecting rank-deficiency"
category: "Math Foundations"
order: 40
updatedDate: "2026-07-25T19:50:03.824Z"
---
## Singular Matrices

A square matrix is **singular** when it is **not invertible** — there is no `A⁻¹` with
`A A⁻¹ = I`. In OLS this is exactly when `(XᵀX)⁻¹` doesn't exist and the closed-form breaks.

### Equivalent characterizations

For a square matrix `A`, these are **all true at once** (any one implies the rest):

| Condition | Meaning |
|---|---|
| **`det(A) = 0`** | Determinant is zero |
| **Rank-deficient** | Columns (or rows) are **linearly dependent** — one is a combination of others |
| **Non-trivial null space** | There is a nonzero `v` with `Av = 0` |
| **Has a zero eigenvalue** | At least one eigenvalue equals 0 |
| **Not full rank** | rank < number of columns |

**Intuition:** the matrix collapses space onto a lower dimension. It sends some direction to zero
(the null-space vector `v`), destroying information → the map can't be undone → no inverse.

**Why `Av = 0` rules out an inverse:** if `A⁻¹` existed, left-multiply `Av = 0` by it →
`v = A⁻¹·0 = 0`, contradicting `v ≠ 0`. So a nonzero null-space vector and invertibility can't
coexist.

### Why this bites OLS

`XᵀX` is singular exactly when `X`'s columns are linearly dependent:
- **Duplicate / redundant features** — a feature and an exact copy, or perfectly correlated columns
  ("height in cm" vs "height in m"). Same as the duplicate-x case in polynomial fitting
  (see [[regression-metrics]]).
- **A constant (zero-variance) feature** alongside the intercept column → two proportional columns
  (see [[feature-selection]]).
- **More features than samples** (`d > n`) → rank ≤ n < d, guaranteed rank-deficient.

---

## Near-Singular & Condition Number

Being *exactly* singular is rare in floating point; being **near-singular** is the practical
problem. A matrix can be technically invertible but **ill-conditioned** — tiny (but nonzero)
determinant, huge **condition number** (ratio of largest to smallest singular value).

- `inv` on an ill-conditioned matrix returns numbers dominated by floating-point error, amplified
  by the condition number.
- This is the real reason to prefer `np.linalg.solve` (LU) over forming `inv`, and
  `np.linalg.lstsq` (SVD) over both when `X` may be rank-deficient / ill-conditioned. Forming
  `XᵀX` itself **squares** the condition number — another reason SVD directly on `X` (lstsq) is
  safer than the normal equations.
- Related second-order view: condition number = ratio of Hessian eigenvalues (see
  [[optimization]]).

### Detecting it in code

```python
np.linalg.matrix_rank(X)      # < d ⟹ rank-deficient
np.linalg.cond(A)             # large ⟹ ill-conditioned
U, S, Vt = np.linalg.svd(X)   # near-zero singular values in S flag the problem
```

Related: [[regression-metrics]], [[feature-selection]], [[optimization]]
