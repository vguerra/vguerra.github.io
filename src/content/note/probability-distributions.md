---
title: "Probability Distributions"
description: "Gaussian/Bernoulli/Categorical/Binomial/Multinomial, the generalization ladder (single-vs-n-trial × binary-vs-K-way), links to CE/BCE/softmax"
category: "Math Foundations"
order: 55
updatedDate: "2026-09-10T21:10:15.299Z"
---
A **probability distribution** describes the possible outcomes of a random variable and their
probabilities. The ones that matter most in ML:

| Distribution | Describes | Parameters | ML connection |
|---|---|---|---|
| **Gaussian / Normal** | continuous real values | mean `μ`, variance `σ²` | MSE = Gaussian-noise MLE ([[loss-functions]]); weight init; noise models |
| **Bernoulli** | single binary trial (success/fail) | `p` | binary classification, dropout mask |
| **Categorical / Multinoulli** | one outcome of **K** categories | `p₁…p_K` | softmax output; multi-class classification |
| **Binomial** | **# successes in n** independent Bernoulli trials | `n, p` | count of heads in n flips |
| **Multinomial** | counts across **K** categories over n trials | `n, p₁…p_K` | generalization of binomial; word counts |

**The generalization ladder:**
- **Bernoulli → Categorical** (2 outcomes → K outcomes, single trial).
- **Bernoulli → Binomial** (single trial → n trials, still binary).
- **Binomial → Multinomial** (binary → K outcomes, n trials).
- **Categorical → Multinomial** (single trial → n trials, K outcomes).

So: *single-trial* vs *n-trial* on one axis, *binary* vs *K-way* on the other.

**Cross-entropy** is the **categorical** likelihood; **BCE** the **Bernoulli** likelihood — which is why
they're the natural classification losses ([[loss-functions]]). The **softmax** output layer
parameterizes a categorical distribution over classes.

---

Related: [[loss-functions]], [[perplexity]], [[pca-svd]], [[numpy-basics]]
