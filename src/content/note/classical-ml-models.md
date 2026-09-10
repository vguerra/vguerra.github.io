---
title: "Classical ML Models — KNN, K-Means, GMM, Trees, SVM"
description: "parametric vs non-parametric, KNN (choosing k, bias-variance), K-Means (init sensitivity, choosing k) vs GMM (soft/overlapping), decision trees (high variance), kernel methods / SVM (kernel trick, hinge loss)"
category: "Classical ML"
order: 39
updatedDate: "2026-09-10T21:11:59.413Z"
---
## Parametric vs non-parametric

- **Parametric** — fixed functional form, **fixed # parameters** regardless of data size (logistic
  regression, linear models). Efficient, interpretable, fast, constant memory; risk **underfitting**.
- **Non-parametric** — no fixed form, **complexity grows with data** (KNN, decision trees, kernel SVM).
  Flexible, fits complex patterns; slower, may **overfit**, can require storing the whole dataset.

Use **parametric** for small data / strong prior / real-time inference; **non-parametric** for lots of
data / unknown complex patterns / flexibility over speed.

---

## KNN (k-nearest neighbors)

Non-parametric. For a query, look at its **k nearest training points**:
- **Classification** → majority class among the k.
- **Regression** → average value of the k.

**Choosing k:** cross-validation (highest CV accuracy); **odd k** for binary (avoid ties); rule of thumb
`k ≈ √n`; plot accuracy vs k. **Low k** → low bias, high variance (flexible, noise-sensitive); **high k**
→ high bias, low variance (smoother, may underfit). Uses: anomaly detection, search, recommenders.

---

## K-Means clustering

Partition into **k clusters**, each point to the **nearest cluster mean**; minimizes within-cluster
variance (squared Euclidean). **No global-optimum guarantee** — result depends on initialization → run
multiple times / use k-means++. **Choosing k:** elbow, silhouette, gap statistic, domain knowledge.
Assumes **equal-size, spherical** clusters; sensitive to outliers.

**K-Means vs GMM:** K-Means = hard assignment, fast, spherical clusters. **GMM** (Gaussian Mixture Model)
= **soft** assignment (probabilistic), handles **overlapping/elongated** clusters and complex
distributions.

---

## Decision trees

Nested if-else conditions from features → target at the leaves.
- **Classification tree** — discrete target, leaf = class.
- **Regression tree** — continuous target, leaf = mean of the values landing there.

Easy to interpret/visualize, but **high variance** — small data changes → very different trees (→
motivates ensembles: [[ensembles]]).

---

## Kernel methods / SVM

A class of pattern-analysis algorithms; the best-known is the **SVM (Support Vector Machine)** —
maximum-margin classifier. The **kernel trick** maps data into a higher-dimensional space (implicitly,
via a kernel function) to make it linearly separable, without computing the mapping explicitly.
**Hinge loss** is the SVM's loss ([[loss-functions]]).

---

Related: [[ensembles]], [[logistic-regression]], [[pca-svd]], [[feature-selection]], [[loss-functions]], [[overfitting-underfitting]]
