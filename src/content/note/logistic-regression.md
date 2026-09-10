---
title: "Logistic Regression, Odds & Odds Ratios"
description: "probability vs odds vs odds-ratio, log-odds/`e^β` interpretation, interaction terms, why log-loss not MSE (sigmoid saturation), assumptions & feature scaling"
category: "Classical ML"
order: 38
updatedDate: "2026-09-10T21:11:44.568Z"
---
Logistic regression models the **probability** of a binary outcome. Its interpretability rests on
**odds** and **log-odds**.

## Probability vs odds

Split trials into **successes** and **failures** (successes + failures = trials):
- **Probability** `P(success) = #successes / #trials` — ranges `[0,1]`; `0.5` = equal; `>0.5` success
  more likely.
- **Odds** `Odds(success) = #successes / #failures` — ranges `[0, ∞)`; `1:1` = equal; `>1` success more
  likely.
- **Link:** `Odds = P/(1−P)`.

Odds are easier to compare near the extremes (0 or 1), where differences in **order of magnitude** are
more intuitive than tiny probability differences. In logistic regression, the constant effect of a
continuous predictor is expressed as an **odds ratio**.

## Odds ratio

Compares the odds of an event under two conditions. As a predictor increases:
- **OR > 1** → event **more** likely.
- **OR < 1** → event **less** likely.

A coefficient `β` in logistic regression means: a one-unit increase in the predictor multiplies the odds
by `e^β` (the odds ratio).

## Interaction terms

An **interaction** between two predictors is represented by their **product** (`x₁·x₂`) and captures a
**non-additive** effect — the effect of one predictor depends on the value of the other.

---

## Why log-loss, not MSE

Logistic regression models **probabilities**, not continuous values, so it uses **log-loss** (binary
cross-entropy):

$$\mathcal{L} = -[\,y\log\hat{y} + (1-y)\log(1-\hat{y})\,]$$

- **MSE penalizes over/under-confidence equally** — not what you want for classification.
- **MSE + sigmoid saturates** → flat gradients near 0/1 → slow/broken training at the decision boundary.
- MSE assumes **Gaussian** errors, false for binary labels. Log-loss = the **Bernoulli** MLE
  ([[loss-functions]], [[probability-distributions]]).

## Practical notes

- **Assumptions (linear regression cousin):** linear relationship (in log-odds for LR), **no
  multicollinearity** (predictors not strongly correlated).
- **Feature scaling matters:** unscaled features → uneven gradients → slow/unstable convergence; and with
  L1/L2 the penalty is **biased** toward large-scale features (they dominate the loss). Standardize first
  ([[preprocessing-fit-transform]]).
- **Overfitting caveat:** adding more predictors to a fixed dataset usually improves *fit* but risks
  **overfitting** → poor generalization ([[overfitting-underfitting]]).

---

Related: [[loss-functions]], [[probability-distributions]], [[regularization]], [[preprocessing-fit-transform]], [[overfitting-underfitting]]
