---
title: "Scaling Laws & Chinchilla"
description: "Chinchilla (C≈6ND, 20 tokens/param), Kaplan-era under-training, fitted loss model, inference-cost correction (overtraining), optimal-operating-point via local-quadratic fit"
category: "Transformers & Sequence Models"
order: 33
updatedDate: "2026-08-03T07:45:25.017Z"
---
**"Training Compute-Optimal Large Language Models"** (Hoffmann et al., DeepMind, 2022) — the
**Chinchilla** paper. One of the most consequential results in LLM training strategy.

---

## The Question

Given a **fixed compute budget C**, how do you split it between:
- **N** — model size (parameters)
- **D** — training data (tokens)

Training compute scales roughly as:

$$C \approx 6\,N\,D \quad \text{(FLOPs)}$$

With C fixed, **N·D is constant** → the tradeoff is **zero-sum**: every FLOP spent on a bigger
model is a FLOP not spent on more tokens.

### C ≈ 6ND is accounting, NOT an optimality statement

A common conflation worth keeping straight — these are two separate things:

1. **The identity (bookkeeping):** C ≈ 6ND says what *any* run costs. It holds for every (N, D)
   combination — good and terrible ones alike.
2. **The optimality result (Chinchilla's contribution):** among **all (N, D) pairs costing the
   same C** (a hyperbola: N·D = C/6), which achieves the **lowest loss**?
   - (huge N, tiny D) → under-trained → bad loss *(Gopher's corner)*
   - (tiny N, huge D) → model too small to absorb the data → bad loss
   - in between → **minimum loss** ← N ∝ C^0.5, D ∝ C^0.5, ~20 tokens/param

> **At budget C, your loss is the best achievable *only if* you allocate Chinchilla-optimally.**
> C determines the *floor*; the (N, D) split determines whether you *reach* it.

Formally: minimize the fitted L(N, D) subject to 6ND = C → the solution traces the
**compute-optimal frontier L(C)** (lower envelope of what each budget can buy). Gopher and
Chinchilla sat at the same C — Gopher *above* the frontier, Chinchilla *on* it. Same budget,
different split, better model: proof the split (not the budget) was being wasted.

### Visualizing the budget curve (the hyperbola)

Only **two axes**: x = N, y = D. There is no third axis — **C/6 is just a constant** (the value
the product must equal). Fixing C imposes `N·D = C/6 = k`, same shape as `x·y = k` → a
**hyperbola** in the N–D plane:

```
 D (tokens)
 │ ●  (tiny N, huge D — model too small)
 │  \
 │   \
 │    ●  ← Chinchilla-optimal point (D/N ≈ 20)
 │     `--.
 │         `---.
 │              `----●  (huge N, tiny D — Gopher's corner)
 └──────────────────────── N (params)
        every point on the curve: N·D = C/6  (cost exactly C)
```

- Slide **right** → bigger model, fewer tokens (the zero-sum trade). Slide **left** → smaller
  model, more tokens.
- **Bigger budget** → a hyperbola further out; budgets form a family of **nested hyperbolas**.
- **Loss is a third quantity** — picture L(N, D) as contour lines drawn *over* this plane. The
  Chinchilla question: *walk along your budget's hyperbola; where is the loss lowest?* That
  tangency point is (N_opt, D_opt), at D/N ≈ 20.
- Gopher and Chinchilla are on the **same hyperbola** (same C); Gopher sits far down-right where
  loss is worse.

### Worked example: GPT-3's budget, Chinchilla-optimal split

GPT-3 sat at (N, D) = (175B, 300B tokens) → ratio ≈ 1.7 tokens/param. Where should it have been
on the **same hyperbola**?

$$N \cdot D = 175\text{B} \times 300\text{B} = 5.25\times10^{22} \qquad D = 20N$$
$$\Rightarrow\; 20N^2 = 5.25\times10^{22} \;\Rightarrow\; N = \sqrt{2.625\times10^{21}} \approx 51\text{B}, \quad D = 20N \approx 1.02\text{T}$$

**Same compute → a 51B model on ~1T tokens**: 3.4× smaller, 3.4× more data, lower loss, and 3.4×
cheaper at inference. (Both factors equal because moving from ratio 1.7 to 20 shifts each axis by
√(20/1.7) ≈ 3.4.)

**Method + sanity checks:** substitute the ratio into the budget constraint → quadratic in N →
don't forget the square root. Then verify the candidate (N, D) against BOTH original equations
(product = budget, ratio = 20) — catches algebra slips instantly.

**Ratio arithmetic (interview check):** double N at fixed C → D halves → the tokens-per-parameter
ratio D/N drops **4×** (numerator halves AND denominator doubles):
$$\frac{D/2}{2N} = \frac{1}{4}\cdot\frac{D}{N}$$
e.g. 20 tokens/param → **5** tokens/param → under-trained model.

---

## Before Chinchilla: Kaplan et al. (2020)

The earlier OpenAI scaling laws said: as compute grows, scale **model size faster than data**.
This produced the era of giant under-trained models:
- **GPT-3**: 175B params, ~300B tokens → ~1.7 tokens/param
- **Gopher**: 280B params, ~300B tokens → ~1 token/param

---

## Chinchilla's Result

From a careful sweep (400+ models, three analysis methods): the loss-optimal split scales
**N and D equally**:

$$N_{opt} \propto C^{0.5} \qquad D_{opt} \propto C^{0.5}$$

**Rule of thumb: D ≈ 20·N — about 20 tokens per parameter** (that's just the ratio D/N ≈ 20;
tokens per parameter *is* D/N).

**Validation:** Chinchilla (70B params × 1.4T tokens = 20:1) trained with the *same compute* as
Gopher (280B × 300B) **beat Gopher across the board** — and is 4× cheaper at inference.

### The fitted loss model

$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}$$

- **E** — irreducible loss (entropy of language)
- **A/N^α** — finite model-capacity error (α ≈ 0.34)
- **B/D^β** — finite data error (β ≈ 0.28)
- α ≈ β → roughly symmetric → balanced scaling is optimal.

---

## Why It Mattered

1. **Reset training recipes** — post-2022 models train on far more tokens per parameter.
2. **Canonical methodology** — fit scaling laws on small runs, extrapolate to choose (N, D) for
   the big run.

## Beyond Chinchilla: the Inference-Cost Correction

Chinchilla optimizes **training** compute only. Every serving request costs ∝ N (forward pass),
and a deployed model serves billions of requests — so **total cost of ownership** favors training
a **smaller model far longer** than the 20:1 optimum ("overtraining"):

- **LLaMA-3 8B**: ~15T tokens → ~1,875 tokens/param — deliberately ~90× past Chinchilla-optimal.
- You pay extra training FLOPs once to save inference FLOPs forever.

**Key nuance:** 20:1 is the *training-compute-optimal* ratio — a fitted constant, not a law of
nature. The deployment-optimal ratio is much higher.

---

## D and Training Time

$$\text{steps} = \frac{D}{\text{batch size} \times \text{seq length}}$$

$$\text{time} \approx \frac{6\,N\,D}{\text{throughput} \times \text{utilization (MFU)}}$$

- Time per step is ~constant (fixed model + hardware) → **at fixed N, wall-clock time is linear
  in D**. Double the tokens = double the GPU-hours.
- "Train longer" in the Chinchilla discussion = "increase D" = more steps = more wall-clock.
- C isn't abstract: at fixed hardware, **C ∝ wall-clock × cluster size**. Overtraining a small
  model (LLaMA-3 style) means literally paying a much longer training run once, to serve cheaper
  forever.

## D Counts Tokens SEEN, Not Unique Tokens

$$D = \text{dataset size} \times \text{epochs}$$

Two ways to raise D: **more unique data** (grow corpus) or **more passes** (repeat data). Same
compute per token — but not equal value:

- **Fresh tokens** — full signal; the regime Chinchilla was fit in (~1 epoch).
- **Repeated tokens** — **diminishing returns**; heavy repetition → memorization, can hurt.
- Muennighoff et al. 2023 ("Scaling Data-Constrained LMs"): **up to ~4 epochs, repeats are almost
  as good as fresh**; value decays fast beyond, ~nothing after dozens of epochs.

At frontier scale (15T+ tokens), **unique high-quality data — not compute — becomes the binding
constraint** (the "data wall") → hence data filtering, controlled repetition, synthetic data.

---

## Scaling-Law Extrapolation for Capability / Safety Evals

Same move as compute-optimal planning, pointed at a different question: **forecast what a model
will be capable of before/while training → plan evals and safeguards ahead of the capability
arriving.**

**Four uses:**
1. **Forecasting → proactive eval scheduling.** Extrapolate where the next run lands; commit *in
   advance* to dangerous-capability evals at that scale. Backbone of **responsible scaling
   policies** (Anthropic RSP, OpenAI Preparedness): capability thresholds + eval obligations
   triggered at forecasted scales.
2. **Checkpoint evals mid-run.** Capabilities improve predictably with tokens seen → eval
   intermediate checkpoints. If a dangerous-capability eval moves at 30% of training, pause and
   assess *before* the run finishes.
3. **Anomaly detection.** The fit predicts expected loss for your (N, D). A model coming in
   substantially *better* than predicted (data/algo improvement) flags that capability forecasts
   and eval plans may be stale.
4. **Compute governance.** Compute is measurable and tracks capability → FLOP thresholds as
   regulatory tripwires (e.g. EU AI Act 10²⁵ FLOP). Only coherent *because* scaling laws make
   compute a capability proxy.

**Critical caveat: loss extrapolates smoothly; capabilities may not.**
- Cross-entropy loss extrapolation is reliable; **downstream task performance can jump**
  ("emergent abilities") — exactly the capabilities safety cares about.
- Counterpoint: emergence is partly a **metric artifact** — smooth metrics (log-likelihood of
  correct answer vs. exact-match) often turn the jump into a smooth curve.
- Practical consequence: **design evals with continuous/graded metrics** to see capabilities
  *approaching* rather than *appearing*; treat extrapolation as a planning tool with error bars.

**Why graded metrics fix it:** capability evals are usually **pass/fail** — flat at 0% until a
sudden jump, no warning. Loss forecasts smoothly *because loss is already a continuous, graded
metric*. Design safety evals the same way (log-likelihood, partial credit, calibrated
probability) and they become extrapolable **just like loss** — a rising trend you can forecast
toward a threshold, instead of a cliff you discover after the fact.

**Not in the original paper.** Hoffmann et al. (2022) is narrowly about training-compute-optimal
(N, D) — it does not discuss capability evals, RSPs, or compute governance. This application is
later field work (Anthropic RSP, OpenAI Preparedness, Wei et al. 2022 on emergent abilities,
Schaeffer et al. 2023 on emergence as a metric artifact) that *uses* Chinchilla-style scaling laws
as one input.

---

## Finding the Optimal Operating Point by Fitting a Local Quadratic

A practical scaling-law move: you have a handful of `(size, loss)` measurements and want the
**loss-minimizing size**. Loss vs **log-compute (or log-size)** is locally U-shaped, so fit a
parabola in `log10(size)` and read off its vertex.

**Recipe:**
1. Transform: `x = log10(size)` (the parabola is U-shaped in log-space, not raw size).
2. Fit `L(x) = a·x² + b·x + c` by OLS — design matrix `[x², x, 1]`, solve the normal equations
   (see [[regression-metrics]] for the polynomial-OLS setup).
3. Vertex (the minimum, valid **because `a > 0`**):
   $$x^* = -\frac{b}{2a} \qquad L_{\min} = c - \frac{b^2}{4a}$$
   (`L_min` = substitute `x*` back in; or just `np.polyval([a,b,c], x_opt)`.)
4. Undo the transform: `optimal_size = 10**x*`.

**Robustness caveats (the interview follow-ups):**
- **`a > 0` is load-bearing.** With noise or few points the fit can return `a ≤ 0` → `x*` is a
  *maximum* or explodes (`a ≈ 0` → divide-by-~0). Guard it; the vertex is only a minimum when
  `a > 0` (see [[linear-algebra-basics]] for why the fit can also be unstable).
  - **Misconception:** fitting a degree-2 polynomial does *not* force `a > 0`, and neither does
    having all-positive sizes/losses. The sign of `a` is **curvature** — how the *slope* changes —
    determined entirely by the data's shape, not by the model choice or coordinate signs. Collinear
    points → `a = 0`; concave (slope decreasing) → `a < 0`. `a > 0` holds **only if the sampled
    sizes straddle the loss minimum** (loss goes down *and back up*). If every sample is still on
    the descending branch (common — you haven't trained a big enough model to bottom out), the fit
    gives `a ≈ 0` or `< 0` and `x*` is meaningless. The guard is really checking "did my data
    capture the bottom of the curve?"
- **Extrapolation.** Nothing forces `x*` inside the observed size range — a vertex beyond your
  sampled points is an extrapolated guess. Consider clipping to `[min, max]` of observed log-sizes.
- **Need ≥ 3 distinct x-values** or `XᵀX` is singular (3 coefficients). Fewer → `LinAlgError`.
- `np.linalg.lstsq` (SVD) degrades more gracefully than `solve` if `X` is ill-conditioned;
  log-transforming the sizes usually keeps conditioning fine.

Related: [[optimization]], [[learning-rate]], [[perplexity]], [[tokenization]], [[regression-metrics]], [[linear-algebra-basics]]
