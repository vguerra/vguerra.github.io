---
title: "RLHF — Reinforcement Learning from Human Feedback"
description: "RLHF pipeline (pretraining → SFT → reward model + PPO), SFT model & its 3 roles, reward model from pairwise preferences (Bradley-Terry, architecture, margin, shift-invariance), PPO + KL penalty, reward hacking, DPO/RLAIF"
category: "Reinforcement Learning"
order: 37
updatedDate: "2026-07-26T15:11:19.650Z"
---
Aligning a pretrained LLM to human preferences. Standard three-stage pipeline:
**Pretraining → SFT → (Reward Model + PPO)**.

---

## Stage 1: Pretraining

Base model trained to predict the next token on web-scale text. Can complete text but doesn't
reliably follow instructions — prompt "Explain gravity" and it may continue with *more questions*,
because that mimics its training text.

---

## Stage 2: SFT — Supervised Fine-Tuning

Fine-tune the base model on curated **(prompt, ideal response)** pairs written/vetted by humans.
Plain supervised learning: **cross-entropy on the target response tokens**. Teaches "given a prompt
like this, respond like this." Output = the **SFT model**, an instruction-follower — but only as
good as its demonstrations.

**The SFT model appears three times later in RLHF:**
1. **Initialization for the reward model** (replace its LM head with a scalar head).
2. **Initialization for the PPO policy** (the thing being optimized).
3. **KL-reference anchor** — PPO penalizes drift away from the SFT distribution (prevents collapse
   into reward-gaming gibberish).

**Why not just do more SFT?** Humans find it far easier to **judge** ("which response is better?")
than to **produce** a perfect demonstration. SFT is capped at the quality of human-written answers;
preferences let the model exceed that ceiling and cover cases no one wrote demos for.

---

## Stage 3a: Reward Model (from pairwise preferences)

Collect **preference pairs**: same prompt, two responses, a human marks which is preferred
(chosen `w` vs rejected `l`). The RM learns to assign a **scalar score** `r(prompt, response)`, higher
for preferred responses.

### Architecture

- **Initialize from the SFT model** (or base) — it already understands language & the task.
- **Replace the LM head** (vocab-sized softmax) with a **linear layer → single scalar**.
- Read the scalar from the **last token's** final hidden state.
- One forward pass per response → one number.

### Objective: Bradley-Terry

Probability the chosen response is preferred, given the two rewards:

$$P(w \succ l) = \sigma(r_w - r_l)$$

Minimize the negative log-likelihood of observed preferences:

$$\mathcal{L} = -\log \sigma(r_w - r_l)$$

### Margin variant (Llama-2)

$$\mathcal{L} = -\log \sigma(r_w - r_l - m)$$

Subtracting a margin *inside* the sigmoid means the model isn't satisfied until the gap **exceeds
`m`** (not just 0). Llama-2 sets `m` **proportional to the annotator's stated preference strength** —
bigger margin for "much better" pairs than "slightly better" ones. A natural source for `m` is a
rating/confidence label you already collected.

### Key subtlety: shift-invariance

The loss depends **only on the difference** `r_w − r_l`, never on absolute values. Adding a constant
to every reward leaves the loss unchanged → **the absolute scale is arbitrary; only *relative*
scores are meaningful.** Consequence: RLHF pipelines **normalize/center** rewards before feeding them
to PPO.

### Why pairwise instead of absolute 1–10 scores?

Absolute human ratings are **noisy and poorly calibrated** — different annotators (and the same
annotator over time) anchor the scale differently. Pairwise comparisons are more consistent and
easier to elicit reliably.

---

## Stage 3b: PPO — Optimize the Policy Against the RM

The SFT model (now the **policy**) generates responses; the reward model scores them; **PPO** updates
the policy to increase reward. Uses:

- **GAE** for advantage estimation (see [[rl-fundamentals]]).
- A **clipped importance ratio** `π_new/π_old` to stay near the data-collection policy (PPO is
  on-policy-ish — see [[rl-fundamentals]]).
- A **KL penalty** to the **SFT reference** — the total reward is roughly
  `r(x,y) − β·KL(π ‖ π_SFT)`. This keeps the policy from drifting into text that games the RM.

### Reward hacking / over-optimization

Because the RM is a *learned imperfect proxy* for human preference, optimizing hard against it
eventually finds inputs where the RM is wrong — reward climbs while true quality drops (Goodhart's
law: "when a measure becomes a target, it ceases to be a good measure"). The **KL penalty** and
**early stopping** on a held-out human eval are the standard guards.

---

## Alternatives worth naming

- **DPO (Direct Preference Optimization)** — skips the explicit RM + PPO loop; derives a loss that
  optimizes the policy *directly* on preference pairs, with the reward implicit. Simpler, more
  stable, no separate RM or RL rollout.
- **RLAIF** — replace human preference labels with an AI labeler (Constitutional AI style).

Related: [[rl-fundamentals]], [[optimization]]
