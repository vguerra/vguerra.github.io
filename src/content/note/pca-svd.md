---
title: "PCA, SVD & Eigendecomposition"
description: "dimensionality reduction (curse of dimensionality), covariance matrix, PCA (standardize → eigendecompose covariance → top-k), eigendecomposition vs SVD, matrix as sum of rank-1 outer products, applications (LSA, compression, LoRA)"
category: "Math Foundations"
order: 54
updatedDate: "2026-09-10T21:10:04.821Z"
---
Dimensionality reduction finds a smaller set of features that **summarize the data** — capturing as much
**variation** as possible so the original data can be approximately reconstructed.

## Why reduce dimensionality
- **Curse of dimensionality:** high-D spaces are **sparse**, distances lose meaning, algorithms degrade
  and need exponentially more data.
- **Less overfitting** (fewer inputs → simpler model → better generalization), especially on small data.
- **Noise reduction**, **visualization**, **compression**, faster training.

---

## Covariance matrix

Captures how features **vary together**. For data with rows = samples, cols = features: diagonal =
per-feature **variances**, off-diagonal `(i,j)` = **covariance** of features `i,j`.
- **> 0** → increase together (positive correlation); **< 0** → one up one down; **= 0** → no *linear*
  relationship.

---

## PCA (Principal Component Analysis)

1. **Standardize features first** (critical — else PCA is biased toward large-scale features). Use
   **z-score** (`(x−μ)/σ` → mean 0, variance 1), **not** min-max (min-max ignores variance → bad for
   PCA).
2. Compute the **covariance matrix**.
3. **Eigendecompose** it: **eigenvectors = directions of maximum variance** (the principal components);
   **eigenvalues = how much variance** each captures.
4. Keep the top-`k` components (largest eigenvalues) → project data onto them.

---

## Eigendecomposition vs SVD

- **Eigendecomposition** — **square** matrices only: `A = V Λ V⁻¹`, `V` = eigenvectors, `Λ` = diagonal of
  eigenvalues. "Breaking a square matrix into its eigenvectors/eigenvalues."
- **SVD (Singular Value Decomposition)** — **any** `m×n` matrix: `A = U Σ Vᵀ` (`U` left-singular vectors,
  `Σ` diagonal singular values, `V` right-singular vectors). A **generalization** of eigendecomposition;
  PCA can be computed via SVD of the (centered) data matrix directly.

**A matrix as a sum of outer products:** both decompositions express `A = Σ σᵢ uᵢ vᵢᵀ` — a weighted sum
of **rank-1** outer products ([[linear-algebra-basics]]), truncating to the top terms gives the best
low-rank approximation.

---

## Applications

- **PCA** — variance-max directions for compression/denoising/visualization.
- **LSA (Latent Semantic Analysis)** — SVD on a term-document matrix to extract NLP semantics.
- **Image compression** — drop small singular values / low-variance directions.
- **Deep learning** — weight matrices have dominant modes; low-rank structure underpins **LoRA** and
  model compression.

---

Related: [[linear-algebra-basics]], [[feature-selection]], [[preprocessing-fit-transform]],
[[embeddings]], [[optimization]]
