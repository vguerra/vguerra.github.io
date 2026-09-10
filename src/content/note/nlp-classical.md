---
title: "Classical NLP — TF-IDF, n-grams, BLEU, Softmax Approximations"
description: "TF-IDF (TF·IDF, limitations), n-gram LMs + smoothing (Laplace/add-k/Good-Turing/Kneser-Ney), large-vocab softmax approximations (hierarchical/sampled/adaptive/tied), BLEU (+ METEOR/ROUGE/BERTScore/COMET)"
category: "Transformers & Sequence Models"
order: 49
updatedDate: "2026-09-10T21:17:45.264Z"
---
## TF-IDF (Term Frequency – Inverse Document Frequency)

Scores a word's importance in a document relative to a corpus: `TF-IDF(t,d,D) = TF(t,d) · IDF(t,D)`.
- **TF** `= (# times t in d) / (total terms in d)` — how often a word appears in one doc.
- **IDF** `= log(N / (1 + df(t)))` — `N` = # docs, `df(t)` = # docs containing `t`; **penalizes common
  words**, boosts rare ones.

**Limitations:** no semantics (tokens independent), ignores **word order** (bag-of-words), no context
awareness (same weight regardless of usage), unigram (can't capture phrases), **sparse & high-dim**,
can't handle OOV words, fixed (doesn't learn), and rare words/typos get **high IDF** (noise-sensitive).

## n-gram language models

Estimate `P(next | previous n−1 words)`. Increasing `n` captures more context (bigram → trigram) but:
counts **grow exponentially**, most sequences become **rare/unseen**, storage explodes, and returns
**diminish**. Prefer n-grams over neural LMs for **small corpora** (easier/faster to train, less
overfitting, interpretable).

**Smoothing** handles unseen n-grams (a single zero probability zeros the whole sentence):
- **Laplace (add-1)** — add 1 to all counts.
- **Add-k** — add small `k` (e.g. 0.01), less aggressive.
- **Good-Turing** — adjust counts using how many events share each frequency.
- **Kneser-Ney** — backoff + discounting; considers *how many different contexts* a word appears in
  (best for low-frequency/unseen bigrams/trigrams).

## Softmax over a large vocabulary

A full softmax over 50k–100k words needs a huge `(hidden × vocab)` matrix and touches **every** logit
each step → slow, memory-heavy, poor on rare words. Approximations:
- **Hierarchical softmax** — organize words in a binary (Huffman) tree → `O(log V)` instead of `O(V)`.
- **Sampled softmax / negative sampling** — compute over the true word + a few sampled negatives.
- **Adaptive softmax** — partition vocab into frequent/rare groups, cheaper approximation for rare.
- **Tied embeddings** — share input & output embedding weights ([[embeddings]]).

## BLEU (translation metric)

Precision-based n-gram overlap vs reference translations. **Pros:** language-agnostic, fast, supports
multiple references. **Cons:** ignores **semantics** (penalizes valid rewording), **no recall**
(doesn't penalize missing content), meaningful at **corpus** not sentence level, insensitive to word
order. Alternatives: **METEOR** (recall + synonyms/stemming), **ROUGE** (recall, summarization),
**BERTScore** (semantic via embeddings), **COMET** (neural, human-aligned).

---

Related: [[embeddings]], [[perplexity]], [[tokenization]], [[beam-search]], [[self-attention]]
