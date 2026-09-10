---
title: "Beam Search Decoding"
description: "breadth-limited decoding: beam/width/log-prob score, algorithm (top-k of all extensions then split completed/active), greedy vs beam, not globally optimal, length bias & normalization (`length^α`, the raw-score short-sequence bias), beam vs sampling (likelihood trap)"
category: "Transformers & Sequence Models"
order: 51
updatedDate: "2026-09-04T20:13:00.429Z"
---
A **breadth-limited** search for sequence generation — a middle ground between **exhaustive** search
(`O(V^T)`, intractable) and **greedy** decoding (`k=1`, myopic). At each step it keeps the `k`
highest-scoring partial sequences.

---

## Vocabulary

- **Beam** — a partial sequence of tokens paired with its **cumulative log-probability** score.
- **Beam width `k`** — how many top-scoring candidates are retained each step.
- **Score** — sum of per-token log-probs (= log joint probability):

$$\text{score}(y) = \sum_{t=1}^{T} \log P(y_t \mid y_1,\dots,y_{t-1})$$

**Why log-probs:** summing logs avoids the **numerical underflow** of multiplying many small
probabilities; and since `log` is **monotonic**, maximizing `Σ log P` = maximizing `∏ P`:

$$\arg\max_y \prod_t P(y_t\mid y_{<t}) = \arg\max_y \sum_t \log P(y_t\mid y_{<t})$$

(Same stability principle as [[perplexity]] / [[loss-functions]].)

---

## Algorithm

1. Initialize one beam: `[start_token]`, score `0`.
2. At each step, for every active beam:
   - compute log-probs over the **full vocab**,
   - generate all one-token extensions, adding `log P` to the cumulative score.
3. Keep the **top `k`** of **all** extensions (EOS and non-EOS together).
4. Move any top-k beam ending in **EOS** to a `completed` list; the rest stay active.
5. Terminate when active beams are exhausted **or** `max_len` is reached.
6. Return the highest-scoring beam from `completed + active`.

**Key correctness point:** take the top-k of *all* extensions, then split into completed/active — don't
save EOS extensions *unconditionally* (that lets non-competitive completions leak in, and lets non-EOS
beams keep slots an EOS beam should have taken). Note active can dip **below `k`** when several top-k
beams complete in one step; it replenishes next step (each active beam expands to `V`, top-k refills).

---

## Greedy vs beam

- **Greedy (`k=1`)** picks the single most probable token each step → can get trapped in **locally
  optimal, globally suboptimal** paths.
- **Beam (`k>1`)** explores multiple hypotheses in parallel → often better sequences.
- **Cost `O(T·k·V)`** — linear in `k`, vocab `V`, and length `T`. Larger `k` → better quality, more
  compute.
- **Not globally optimal.** Beam search is still a **heuristic** (greedy at the beam level) — it can
  prune the true argmax if that sequence has a weak-scoring *prefix*. Only `k = V^T` (full search)
  guarantees the optimum.

---

## Length bias & normalization (the classic gotcha)

Since every `log P ≤ 0`, each additional token **adds a negative number** → **longer sequences score
lower**. So raw-score beam search is **biased toward short sequences** (it prefers to emit EOS early),
producing truncated outputs in translation/summarization.

**The bias bites specifically when comparing sequences of *different lengths*** — i.e. the final
`completed + active` merge (beams finished at different steps). *Within* a single step all active beams
have the same length, so raw scores compare fine there.

**Fix — length normalization** (GNMT length penalty, Wu et al.):

$$\text{score}(y) = \frac{1}{|y|^\alpha}\sum_t \log P(y_t\mid y_{<t}), \quad \alpha \in [0,1]$$

Decide: normalize by full length (incl. start token) or generated-token count — be consistent with your
indexing. Typically applied at **final comparison**; normalizing during per-step pruning changes which
beams survive.

> **Note:** some *exercises* deliberately use **raw (non-normalized) scores** and **include the start
> token** — correct for the assignment, but know that production systems normalize because of this
> short-sequence bias.

---

## When (not) to use beam search

Beam search **maximizes likelihood** — great for tasks with a single correct-ish answer
(**translation, summarization, ASR**). For **open-ended** generation (dialogue, story writing),
likelihood-maximization yields **bland, repetitive, degenerate** text (the "likelihood trap") → use
**sampling** instead: temperature, top-k, nucleus/top-p.

---

Related: [[perplexity]], [[loss-functions]], [[self-attention]], [[rnn]], [[tokenization]]
