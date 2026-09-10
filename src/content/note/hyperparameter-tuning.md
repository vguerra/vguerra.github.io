---
title: "Hyperparameters & Tuning"
description: "parameters vs hyperparameters, key HPs table, tuning methods (grid/random/Bayesian/Hyperband/evolutionary), why random beats grid"
category: "Learning Paradigms & Workflow"
order: 37
updatedDate: "2026-09-10T21:15:57.401Z"
---
**Parameters** are learned from data (weights/biases, updated by the optimizer). **Hyperparameters** are
**set before training** and control *how* learning happens (tuned manually or by search). *Parameters =
what the model learns; hyperparameters = what you choose to help it learn.*

## Key hyperparameters

| Hyperparameter | Role |
|---|---|
| Learning rate | step size (too high → diverge; too low → slow/stuck) |
| Batch size | gradient stability vs memory |
| Epochs | when to stop |
| Dropout rate | regularization strength |
| Optimizer | update behavior (Adam vs SGD) |
| Architecture | depth, width, kernel sizes |

They control **model complexity** and the **bias-variance tradeoff** (regularization, depth/width,
dropout — [[overfitting-underfitting]]). Bad choices waste compute (exploding/vanishing gradients, hours
with no improvement); good ones give faster convergence, better accuracy, stable training, better
generalization.

## Tuning methods

| Method | Idea | Trade-off |
|---|---|---|
| **Grid search** | try every combo from a predefined set | thorough on small spaces; **exponential**, not scalable |
| **Random search** | sample configs randomly | more efficient than grid; may miss optima, doesn't learn |
| **Bayesian optimization** | probabilistic model of the objective picks promising configs next | efficient for expensive models, **learns** from past; complex, slower per trial |
| **Successive halving / Hyperband** | train many configs briefly, keep the best, give them more budget | great under limited compute (early stopping); needs well-defined resource limits |
| **Evolutionary** | population of configs, mutate/combine top performers | good for dynamic/online tuning; expensive |

Random search beats grid because good configs usually depend on **a few** important hyperparameters, and
random sampling covers those dimensions better per trial.

---

Related: [[learning-rate]], [[lr-schedulers]], [[overfitting-underfitting]], [[regularization]], [[deep-learning-theory]]
