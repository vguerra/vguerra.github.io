---
title: "Adaptive Optimizers — AdaGrad, RMSProp, Adam"
description: "why adaptive (sparse gradients), AdaGrad (sum-of-squares → monotonic decay flaw), RMSProp (EMA fix), Adam (momentum + RMSProp + bias correction), Adam vs SGD table, AdamW pointer"
category: "Training Dynamics & Optimization"
order: 21
updatedDate: "2026-09-10T21:17:59.393Z"
---
Optimizers that give each parameter its **own** effective learning rate based on its gradient history.
(Momentum/Nesterov: [[momentum]]; LR schedules: [[lr-schedulers]].)

## Why adaptive — sparse gradients

Sparse gradients arise in **NLP (one-hot/embedding lookups)**, recommenders, sparse features. A fixed LR
(vanilla SGD) is **too small for infrequent features**, so they learn slowly. Adaptive methods give rare
features a **larger** effective step and frequent ones a smaller one.

## AdaGrad

Scales each parameter's step by the **inverse sqrt of the sum of all past squared gradients**:
- Frequent params accumulate large history → **smaller** updates (stabilize); rare params → **larger**
  updates. Great for sparse settings.
- **Flaw:** the accumulator only **grows**, so the effective LR **monotonically decays to zero** and
  can't recover — training stalls even when there's more to learn.

## RMSProp

Fixes AdaGrad by using an **exponential moving average** of squared gradients instead of a growing sum →
the effective LR can **increase or decrease** (recent gradients dominate, old ones decay). No permanent
decay-to-zero.

## Adam (Adaptive Moment Estimation)

**Marries momentum + RMSProp:** keeps EMAs of both the **first moment** (mean of gradients — momentum)
and the **second moment** (mean of squared gradients — RMSProp scaling), with **bias correction** for
the zero-initialized estimates (the cold-start issue behind transformer warmup — [[lr-schedulers]]).
The `1/√v` denominator gives per-parameter adaptive rates.

## Adam vs SGD

| | SGD (+momentum) | Adam |
|---|---|---|
| LR | one global rate | **per-parameter** adaptive |
| Momentum | optional (add manually) | built in |
| Convergence | slower, needs careful LR tuning | fast, works out of the box |
| Memory | low | higher (2 extra tensors/param) |
| Generalization | often **better** (vision) | may overfit / find sharper minima |

**Use SGD** when generalization matters (vision) and you can afford tuning; **use Adam** for fast
convergence, deep/complex models, or **sparse gradients (NLP)**. Note **AdamW** decouples weight decay
from the adaptive scaling ([[regularization]]).

---

Related: [[momentum]], [[learning-rate]], [[lr-schedulers]], [[regularization]], [[optimization]], [[pytorch-optimizer]]
