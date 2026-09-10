---
title: "Linear Algebra Basics"
description: "vectors (dot/outer product, L0–L∞ norms, independence), matrix fundamentals (linear transforms, inverse, determinant, derivative/gradient/Jacobian/Hessian hierarchy); singular matrices (equivalent characterizations, why they break OLS), near-singular / condition number, detecting rank-deficiency"
category: "Math Foundations"
order: 53
updatedDate: "2026-09-10T21:09:48.293Z"
---
## Vectors

### Dot product
`a·b = Σ aᵢbᵢ` — measures **how much one vector points along another** (`a·b = ‖a‖‖b‖cosθ`).
- **> 0** if angle < 90°, **= 0** if orthogonal (90°), **< 0** if > 90°.
- The unit vector `v` maximizing `u·v` is **`v = u/‖u‖`** (same direction, cosθ=1).
- Basis of QKᵀ attention scores, cosine similarity, projections.

### Outer product
Column × row → a **matrix**: `a bᵀ`, with `(a bᵀ)ᵢⱼ = aᵢbⱼ`. Always **rank-1**. Uses: building rank-1
matrices, expressing a matrix as a **sum of outer products** (SVD/PCA/matrix factorization), the `QKᵀ`
similarity matrix in attention ([[self-attention]]), feature-interaction modeling.

### Norms (vector length/magnitude)
Properties every norm satisfies: non-negativity, definiteness (only `0` has norm 0), scaling
(`‖αv‖=|α|‖v‖`), triangle inequality (`‖u+v‖≤‖u‖+‖v‖`).

| Norm | Formula | Notes |
|---|---|---|
| **L0** (pseudo-norm) | # non-zero elements | sparse optimization / compressed sensing |
| **L1** (Manhattan) | `Σ|xᵢ|` | total abs deviation; **promotes sparsity** (Lasso) |
| **L2** (Euclidean) | `√(Σxᵢ²)` | standard distance/length |
| **Lp** | `(Σ|xᵢ|ᵖ)^{1/p}` | general form |
| **L∞** (max) | `max|xᵢ|` | largest-magnitude coordinate |

A **norm** applies to a single vector; a **metric** is a distance between two points. **Every norm
induces a metric** (L2→Euclidean, L1→Manhattan, L∞→max-coordinate). Vectors are **linearly independent**
if neither is a scalar multiple of the other. See [[regularization]] for L1/L2 as penalties.

---

## Matrix Fundamentals

- **Linear transformation:** `A ∈ ℝ^{m×n}` applied to `x ∈ ℝⁿ` gives `Ax ∈ ℝᵐ` — a matrix *is* a linear
  map between spaces.
- **Inverse:** `A⁻¹` with `AA⁻¹ = I`. Exists iff `A` is **square, full-rank** (linearly independent
  rows/cols), **non-zero determinant** (see Singular Matrices below).
- **Determinant:** the **volume-scaling factor** of the transform (area in 2-D, volume in 3-D).
  **Positive** → orientation preserved; **negative** → orientation flipped; **`det=0`** → **singular**
  (squashes space into a lower dimension → not invertible).

### Derivative → Gradient → Jacobian → Hessian (the hierarchy)
| Object | Applies to | Shape |
|---|---|---|
| **Derivative** | scalar → scalar | scalar |
| **Gradient** | scalar of **many** vars | vector of partials |
| **Jacobian** | **vector**-valued of many vars | matrix (each row = gradient of one output) |
| **Hessian** | scalar of many vars, 2nd order | matrix of 2nd partials (curvature — see [[optimization]]) |

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
