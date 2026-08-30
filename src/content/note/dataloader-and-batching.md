---
title: "Data Loading & Batching (PyTorch)"
description: "Dataset/Sampler/DataLoader architecture, map vs iterable style, eager/lazy/mmap loading, dtypes, feature/label `(N,)` vs `(N,1)` collate gotcha, DataLoader params (statistics vs throughput knobs), collate function, sampler hierarchy, `pin_memory` two-step transfer, batching strategies (bucket/dynamic/grad-accum), performance tuning, pitfalls"
category: "Training Dynamics & Optimization"
order: 18
updatedDate: "2026-08-26T09:21:59.100Z"
---
Feeding data to a model in an **organized, efficient, repeatable** way. PyTorch splits this into three
abstractions:

- **Dataset** — knows how to read a **single sample** from underlying storage.
- **Sampler** — decides the **order** samples are accessed, by producing a sequence of **indices**.
- **DataLoader** — the **orchestrator**: consults the Sampler for indices, fetches samples from the
  Dataset (optionally in parallel across workers), **collates** them into batches, and yields batches
  to the training/validation loop.

---

## Dataset protocol: map-style vs iterable-style

**Map-style** (most common) — supports **random access by index**. Implements:
- `__len__` → total sample count (used to compute #batches/epoch and validate indices — **must be
  fast**).
- `__getitem__(i)` → the i-th sample, usually a `(features, label)` tuple of tensors.

Use when data fits in memory or is stored in a **seekable** on-disk format.

**Iterable-style** — implements `__iter__` (the iterator protocol), **no `__len__`**. Produces samples
in a **streaming** fashion, one at a time. Use when random access is impossible/impractical: network
streams, log files tailed in real time, huge shards.

---

## Dataset constructor: eager vs lazy vs memory-mapped

Choice depends on **data size** and **cost of reading one sample**:

| Strategy | Constructor does | `__getitem__` does | Trade-off |
|---|---|---|---|
| **Eager** | read + process **everything** upfront, store as tensors in RAM | O(1) tensor index (view / small copy) | fast access, but **whole dataset must fit in RAM** |
| **Lazy** | store only **metadata / file paths** | reads + processes the sample **each call** | minimal memory (one sample at a time), but every access pays I/O + compute — **standard for image datasets** |
| **Memory-mapped** | open an mmap over an on-disk array | index into the array; **OS pages in on demand** | looks like eager (index an array) but memory is governed by the **OS page cache** — great for **tens-of-GB tokenized NLP corpora** |

With mmap, frequently-accessed regions stay hot in the page cache; the dataset code reads like eager
indexing but doesn't need the data to fit in RAM.

---

## Dtypes & memory

- **`float32`** — default for features; enough precision for almost all NN compute.
- **`float64`** — rarely needed; doubles memory and most GPU ops are much slower.
- **`float16`/`bfloat16`** — 2 bytes, for **mixed precision**. Store the dataset in **float32** and
  cast to low precision **during the forward pass**, *not* in the dataset itself (see [[tensor-dtypes]]).
- **`long` (int64)** — standard for **classification labels**.

---

## Separating features from labels (and the `(N,)` vs `(N,1)` gotcha)

A tabular matrix of `N × (D+1)` splits into features `(N, D)` and labels `(N,)` **or** `(N, 1)` — and
the choice **matters at collate time**:

- labels `(N,)` → each label is a **scalar** tensor `()` → collate **stacks** them into **`(B,)`**.
- labels `(N, 1)` → each label has shape `(1,)` → collate stacks into **`(B, 1)`**.

Both are valid, but must **match what the loss expects** — mixing `(B,)` and `(B,1)` triggers the
silent broadcasting blow-up to `(B, B)` (see [[broadcasting]]). Be consistent.

---

## The DataLoader: batching, shuffling, parallelism

Two families of knobs — **statistics** (gradient quality) and **throughput** (keeping the GPU fed):

**Statistics knobs**
- **`batch_size`** — how many samples per batch; the default collate stacks `B` tensors of shape `(D,)`
  into `(B, D)`. Affects gradient variance, memory, and dynamics.
- **`shuffle`** — `True` → `RandomSampler` internally; `False` → `SequentialSampler`. Without shuffling,
  batch gradients are **biased** (consecutive samples share structure); shuffle makes each batch
  ~represent the full dataset. **Can't set `shuffle=True` *and* a custom `sampler`.** Shuffle train
  only.
- **`drop_last`** — `True` discards the final incomplete batch → consistent BatchNorm ([[normalization]])
  and clean loss averaging; **don't** use for evaluation (you want to score every sample).

**Throughput knobs**
- **`num_workers`** — `0` = load in the **main process** (data prep and compute serialize → GPU idles
  during prep). `>0` spawns worker **subprocesses** that fetch batches in the background while the GPU
  trains on the current one. Fine to leave at 0 for pre-loaded tabular tensors (indexing is instant);
  essential for image/disk pipelines where loading dominates.
- **`prefetch_factor`** — batches prepared ahead per worker (default 2). Raise if workers are fast but
  there are occasional stalls.
- **`pin_memory`** — see below.

---

## The collate function

Receives a **list of samples**, returns a **batch**. Default collate **recursively stacks tensors**
(adds a leading batch dim), converts+stacks numpy arrays, turns numbers into tensors, and groups
tuples/lists element-wise — so a dataset of `(features, label)` tuples yields
`(batched_features, batched_labels)`.

Write a **custom collate** when:
- variable-length sequences must be **padded** (and attention masks built),
- images of **different sizes** must be handled,
- non-tensor data needs special handling.

---

## Sampler hierarchy

Samplers are iterators that **yield indices**; the DataLoader passes them to `__getitem__`.

| Sampler | Yields |
|---|---|
| **SequentialSampler** | `0,1,…,N-1` (default when `shuffle=False`) |
| **RandomSampler** | a random permutation (default when `shuffle=True`); supports replacement |
| **SubsetRandomSampler** | random permutation of a **given index list** — handy for train/val splits |
| **WeightedRandomSampler** | draws by **per-sample weights** — the primary tool for **class imbalance** |
| **BatchSampler** | wraps a sampler and groups its indices into batches |
| **DistributedSampler** | partitions the dataset across processes so each GPU sees a **disjoint** subset (multi-GPU) |

---

## `pin_memory` & GPU transfer

Normally CPU tensors live in **pageable** memory. Copying to GPU is then a **two-step** process: CUDA
first copies into a temporary **pinned** buffer, then DMAs to the GPU. With **`pin_memory=True`**, the
DataLoader allocates batch tensors **directly in pinned memory**, eliminating the first copy and
letting the DMA start immediately. Pair with `.to(device, non_blocking=True)` for async transfer (see
[[tensor-devices]]). Negligible overhead on GPU training — a consistent win.

---

## Batching strategies beyond uniform

- **Length-based / bucket batching** — group similar-length sequences to **minimize padding** waste
  (important in NLP). A bucket sampler sorts by length, partitions into buckets, batches within each.
- **Dynamic batching** — size the batch by **total tokens/pixels** rather than sample count, keeping
  GPU memory ~constant across variable sequence/image sizes.
- **Gradient accumulation** — when the desired effective batch exceeds GPU memory, process several
  small batches and accumulate gradients before one step. The DataLoader stays oblivious — it yields
  small batches; the training loop handles accumulation (see [[training-memory]], [[pytorch-training-loop]]).

---

## Performance tuning — never let data loading bottleneck

The GPU should always have a batch ready. Levers:

- **Raise `num_workers`** until GPU utilization stops improving. Start near #CPU cores; SSDs saturate
  more workers than spinning disks.
- **`pin_memory=True`** on GPU — consistent DMA speedup, negligible cost.
- **Raise `prefetch_factor`** if workers are fast but stalls remain.
- **Pre-process into efficient formats** — LMDB / TFRecord / WebDataset (fewer small-file opens),
  pre-tokenized **memory-mapped** arrays (no tokenization at train time).
- **Profile the pipeline.** If 80% of step time is data loading, no model optimization helps
  ([[training-diagnostics]]).

---

## Common pitfalls

- **Non-tensor data + default collate** → errors; write a custom collate.
- **Dtype mismatch with the loss** — CE wants `long` labels, MSE wants `float32` (see [[loss-functions]]).
- **Assuming a fixed batch size** — the last incomplete batch causes shape mismatches unless handled
  (or `drop_last`).
- **Expensive work in `__getitem__` without workers** → GPU starves.
- **Inconsistent train/eval transforms** → silent accuracy degradation.
- **Unseeded workers with random augmentation** → non-reproducible runs (seed each worker).

---

Related: [[pytorch-training-loop]], [[training-memory]], [[tensor-devices]], [[tensor-dtypes]],
[[normalization]], [[broadcasting]], [[learning-rate]], [[training-diagnostics]], [[loss-functions]]
