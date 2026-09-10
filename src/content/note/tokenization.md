---
title: "Tokenization"
description: "token granularity levels (word/subword/char/byte table), vocabulary & special tokens, subword algorithms (BPE/WordPiece/Unigram), how BPE/WordPiece handles rare words, numbers, code"
category: "Transformers & Sequence Models"
order: 48
updatedDate: "2026-09-10T21:28:56.993Z"
---
## Token granularity levels

A **token** is an arbitrarily-defined unit of text. The level trades vocabulary size against sequence
length:

| Level | Unit | Pros | Cons |
|---|---|---|---|
| **Word** | each word | intuitive, fewer tokens (fast inference) | huge vocab, **OOV** words unhandled |
| **Subword** (BPE/WordPiece) | word pieces | smaller vocab, captures roots, handles OOV | more complex, longer input |
| **Character** | each char | tiny vocab, no OOV | very long input, embeddings less meaningful |
| **Byte** | raw byte | any language/emoji, tiny vocab | very long input, hard to interpret |

**Subword wins** in practice: it leverages **root meanings** (run/runner/running share a root) and
minimizes **OOV** by recognizing small variations. Finer tokens → smaller vocab; multilingual corpora →
larger vocab.

**Vocabulary** = the fixed set of tokens, including **special tokens**: `[PAD]` (pad to max length),
`[UNK]` (out-of-vocabulary). A **tokenizer** converts text ↔ tokens (**encode** / **decode**) and is
either **rule-based** (no training) or **learned** (trained on data). **Normalization** first handles
inconsistencies: lowercasing, accent removal, unusual Unicode.

## Subword algorithms

- **BPE (Byte-Pair Encoding)** — start from characters; repeatedly **merge the most frequent adjacent
  pair** into a new token, recording merge rules, until the target vocab size. Encoding applies the merge
  rules in learned order. **Frequency-based → fast.**
- **WordPiece** — like BPE but merges the pair giving the biggest **language-model likelihood**
  improvement (not raw frequency). Needs expensive scoring passes; with lots of data, quality ≈ BPE.
- **Unigram** — starts from a **large** vocab and **iteratively discards** the tokens that hurt the
  corpus likelihood least (EM to score each subword). Encoding picks the **highest-probability
  segmentation** of a word. Assumes token probabilities are independent.

---

## How Tokenization Affects Rare Words, Numbers, and Code

BPE/WordPiece builds vocabulary from frequency — infrequent patterns get split into smaller subword pieces.

### Rare Words
```
"serendipitous" → ["ser", "end", "ip", "itious"]   # split into subwords
"cat"           → ["cat"]                            # common, stays whole
```
Rare words consume more tokens and lose semantic coherence — the model reconstructs meaning from fragments.

### Numbers
```
"12345" → ["123", "45"]   or   ["1", "23", "45"]   # inconsistent splits
"42"    → ["42"]                                     # common number may stay whole
```
Same value can tokenize differently depending on context — makes arithmetic and numerical reasoning hard.

### Code
```
"self.attention" → ["self", ".", "attention"]
"forward_pass"  → ["forward", "_", "pass"]
"nn.Linear"     → ["nn", ".", "Linear"]
```
Code suffers because:
- Identifiers split at underscores, dots, camelCase boundaries
- Whitespace/indentation is significant but tokenized inconsistently
- Code-specific tokens (`->`, `::`, `=>`) may split or merge unpredictably

### Why It Matters
Each split token costs context window space and forces the model to learn relationships across fragments rather than treating the concept as a unit.

---

## Tokenization for Code LLMs

Code-focused LLMs use **BPE** (same algorithm as text LLMs) but with a **code-aware vocabulary**:

- Trained on large code corpora (GitHub, Stack Overflow)
- Common identifiers (`def`, `class`, `self`, `return`, `torch`), operators (`->`, `::`, `**`), and indentation patterns become single tokens
- Larger vocabularies (50k–100k) to cover both natural language and code

**Examples:**
- **Codex / GPT-4** — BPE with code-heavy training data
- **Code Llama** — extends Llama's BPE tokenizer with more code tokens
- **DeepSeek-Coder** — BPE with 32k vocab, heavily weighted toward code

**Key insight:** it's less about a different algorithm, more about what data the tokenizer was trained on — BPE trained on code naturally learns that `def `, `self.`, `__init__` are high-frequency and assigns them single tokens.

Some models treat **whitespace/indentation explicitly** — since Python indentation is semantic, each indent level may get its own token.

---

## Why a Bad Tokenizer Raises Effective Sequence Length and Cost

A small vocabulary (e.g. character-level) splits sequences into many more tokens:
```
"transformer" → 11 tokens (char-level) vs 1-2 tokens (good BPE)
```

**Compute cost — O(n²):** the attention matrix `QKᵀ` is `(seq_len × seq_len)` — both compute and memory scale quadratically with sequence length. FFN layers are O(n), so attention is the bottleneck.

**Context window cost:** longer sequences consume more of the fixed context window, leaving less room for useful content.

**Memory cost (KV cache):** keys and values stored during generation grow linearly with sequence length — longer tokenizations increase memory pressure.

**Concrete example:** at 1000 characters, character-level → 1000² vs BPE → ~200² attention operations = **25× more compute**.
