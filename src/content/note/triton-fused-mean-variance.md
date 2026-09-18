---
title: "Triton: Fused Mean & Variance — Linearity & the Host Combine"
description: "one-pass mean+variance via `σ²=E[x²]−E[x]²`: two atomics/program into two scratch buffers, **only linear stats combine across programs** (accumulate Σx/Σx², apply nonlinear identity on host — variance isn't additive), fusion economics (4MB vs 8MB, plain-sum throughput vs half), 0.75 FLOP/byte, catastrophic-cancellation caveat + Welford (doesn't parallelize)"
category: "GPU / Kernels"
order: 80
updatedDate: "2026-09-17T18:53:22.374Z"
---
Computing **population mean and variance in one pass** is the right move on a bandwidth-bound device.
The identity `σ² = E[x²] − E[x]²` lets you reduce **two linear accumulators** (`Σx`, `Σx²`) in a single
sweep, saving an entire HBM round-trip vs. two separate reductions. Builds on [[triton-sum-reduction]].

## The pattern — two accumulators, one grid

Same grid as a single-statistic reduction, but each program contributes **two atomics** into **two
distinct addresses** (`sum_buf`, `sumsq_buf`). Load/mask/sentinel mechanics are identical to the sum
kernel (`other=0.0`, the additive identity). Host contract:
1. **Zero both scratch buffers** (reached via `atomic_add`).
2. **Finalize on the host** after the kernel returns: divide each buffer by `N`, then apply the variance
   formula.

## ⚠️ The central rule: only linear statistics combine across programs

**Variance is not linear in the input** — `Var(A ∪ B) ≠ Var(A) + Var(B)` — so **per-program variances
cannot be summed** across programs. Only `Σx` and `Σx²` are linear, so **those** are what the atomics
accumulate; the nonlinear identity `E[x²] − μ²` is applied **on the host, after** the cross-program
combine is complete. Computing `σ²` inside the kernel and atomic-adding it is **wrong**.

(This generalizes: it's *why* LayerNorm/BatchNorm reductions accumulate sum and sum-of-squares, not
variance — [[normalization]].)

## Fusion economics

At `N = 10⁶`: the fused kernel **reads 4 MB, writes 8 bytes** (two fp32 scalars); the naive two-kernel
form **reads 8 MB**. At 1 TB/s → ~**4 µs fused vs ~8 µs naive**. Launch overhead also matters for tiny
`N` — the naive form **pays it twice**. Atomic traffic doubles (**2G** atomics for `G` programs) but both
addresses stay **hot in L2** and are independent. **Net: the fused kernel sits at ~plain-sum throughput;
the naive two-kernel form at half that.**

## Roofline

Per element: read 4 B, do **3 FLOPs** — one multiply (`x·x`), one add into the `Σx` tree, one add into
the `Σx²` tree → **arithmetic intensity = 3 FLOP / 4 B = 0.75 FLOP/byte**. That's **3× a plain sum** but
still well under the ~10 FLOP/byte fp32 crossover → **memory-bound**. Fusion raises the *intensity* but
the kernel still approaches the HBM ceiling, now at full (plain-sum) throughput instead of half.

## Numerical caveat — catastrophic cancellation

`E[x²] − μ²` loses precision when **`variance ≪ mean²`** (data tightly clustered around a large mean) —
the two large near-equal terms cancel ([[numpy-basics]]). Production alternative: **Welford's algorithm**
(running mean + running sum of squared deviations, numerically stable recurrence). Welford is one-pass
like the identity, but its recurrence **doesn't parallelize cleanly across tiles**, so for a GPU tile
reduction the identity form is the pragmatic tradeoff — simple and fast. Switch to Welford for
near-constant data.

## Pitfalls

- **Two separate kernels** (one for `Σx`, one for `Σx²`) — doubles HBM traffic and pays two launches.
  The whole point is to **fuse**, not to be tidy.
- **Forgetting to zero either scratch buffer** — both are reached via atomics; skipping the zero drifts
  on the **second and later** calls (first call passes → call-order-dependent, the worst failure mode).
- **Computing variance in the kernel and atomic-adding it** — variance isn't linear; per-program
  variances can't be summed. Accumulate `Σx`/`Σx²`, apply the identity on the host.
- **Too-tight tolerance at large `N`** — the sum-of-squares accumulator drifts with `N` *and* the
  subtraction can cancel; use combined `atol=1e-2, rtol=1e-2` for the variance check at large `N`.

---

Related: [[triton-sum-reduction]], [[triton-max-reduction]], [[numpy-basics]], [[normalization]], [[triton-vector-add]]
