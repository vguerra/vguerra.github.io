---
title: "Cross-Validation & Data Splits"
description: "train/val/test roles, methods (holdout/k-fold/stratified/LOOCV/time-series/group), why CV rare in DL (early stopping + ensembling), split hygiene & leakage, test<val red flag"
category: "Generalization & Model Fitting"
order: 32
updatedDate: "2026-09-10T21:16:17.910Z"
---
Using the **same data** to train and test gives a **misleadingly optimistic** estimate — the model is
judged on data it memorized. Split roles: **train** (fit), **validation** (tune hyperparameters),
**test/holdout** (final unbiased evaluation).

## Methods

| Method | Idea | Notes |
|---|---|---|
| **Holdout** | single split (e.g. 80/20) | fine for large data; high variance between splits |
| **k-fold** | split into k parts, train on k−1, validate on the rest, repeat k× | robust estimate; k× slower |
| **Stratified k-fold** | k-fold preserving **class proportions** per fold | important for **imbalanced** classification |
| **Leave-one-out (LOOCV)** | k = n (one sample out each time) | very thorough, expensive, high variance |
| **Leave-p-out** | leave out p samples | even more exhaustive |
| **Time-series split** | train on past, validate on future (rolling/expanding window) | no leakage for temporal data; **not** for i.i.d. data |
| **Group k-fold** | all samples of a group (user/patient) in one fold | prevents leakage when samples within a group are correlated |

## Why CV is rare in deep learning

Data is usually large enough, training is expensive, and **early stopping on a validation set** already
acts as a built-in evaluation loop. Instead of CV for stability, people train multiple models and
**ensemble**. Consider CV in DL only for **small datasets** or **transfer learning**.

## Split hygiene (avoiding leakage)

- Shuffle; keep splits **representative**; ensure **no group leaks** across train/val (same
  user/patient/session in both).
- **Test loss lower than validation** is a red flag: validation set harder/unbalanced, test set too
  small/biased, or hyperparameters were tuned on validation (making test feel easier).

---

Related: [[overfitting-underfitting]], [[class-imbalance]], [[evaluation-metrics]], [[preprocessing-fit-transform]], [[ml-in-production]]
