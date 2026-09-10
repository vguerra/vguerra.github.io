---
title: "Types of Learning & Sampling"
description: "supervised/unsupervised/weakly/semi (self-training, consistency reg, pseudo-labeling)/active learning (uncertainty/margin/entropy/QBC/diversity), LM supervised-vs-unsupervised, sampling (with/without replacement, bootstrapping, MCMC)"
category: "Learning Paradigms & Workflow"
order: 35
updatedDate: "2026-09-10T21:13:37.890Z"
---
## Learning paradigms

- **Supervised** — every input has a ground-truth label; learn `f(x)→y`. Classification, regression,
  sequence prediction.
- **Unsupervised** — no labels; find structure. Clustering (k-means), dimensionality reduction (PCA),
  anomaly detection. Good for pretraining / feature extraction.
- **Weakly supervised** — noisy/limited/imprecise labels: **inexact** (coarse, e.g. image-level not
  boxes), **inaccurate** (errors, e.g. crowdsourced), **incomplete** (only some labeled).
- **Semi-supervised** — small labeled + large unlabeled set; leverage the unlabeled data to improve
  generalization when labels are expensive. Methods: **self-training**, **consistency regularization**
  (penalize different predictions for the same input under noise → robustness), **pseudo-labeling**
  (model labels unlabeled data, trains on its confident predictions as if ground truth).
- **Active learning** — the model **selects the most informative** unlabeled samples and queries an
  oracle (human) for labels → high performance with fewer labels. Selection strategies: **uncertainty
  sampling** (lowest confidence), **margin sampling** (top-2 class probs close), **entropy-based**
  (highest output entropy), **query-by-committee** (max disagreement across models), **diversity/core-set**
  (represent the data distribution). Used in labeling tools / human-in-the-loop.

> **Is a language model supervised or unsupervised?** *Technically unsupervised* (no manual labels — it
> learns from raw text). *Mechanistically supervised* — each token becomes an (input, target) pair
> trained with cross-entropy. The distinction is about the **source of labels**, not the training
> mechanism. See [[embeddings]].

---

## Sampling & creating training data

- **With replacement** — items can repeat, samples independent, infinite outcomes. Used in
  **bootstrapping** (bagging / random forests — [[ensembles]]).
- **Without replacement** — each item at most once, samples dependent, limited to dataset size. Used in
  **train/test splits** (no overlap) and **epochs** (see [[pytorch-training-loop]]).
- **MCMC (Markov Chain Monte Carlo)** — explore a complex, high-dimensional distribution by a guided
  random walk: a **Markov chain** (each state depends only on the previous) + **Monte Carlo** (random
  sampling to estimate integrals/expectations). Used for Bayesian inference and sampling from
  intractable distributions.

---

Related: [[ensembles]], [[class-imbalance]], [[embeddings]], [[pytorch-training-loop]], [[ml-in-production]]
