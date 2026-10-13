---
title: "Debugging Triton Kernels (esp. without error messages)"
description: "debugging kernels when the platform shows no error text: symptom→cause table (`map::at`, `tl.arange`, wrong numbers, NaN), bisection with one change per submission (beware dead-code elimination), hand-computing tiny cases, asymmetric/unequal-size tests, derive formulas don't copy, keep similar quantities named separately"
category: "GPU / Kernels"
order: 80
updatedDate: "2026-10-05T21:15:42.322Z"
---
A workflow distilled from debugging the tiled matmul and FlashAttention kernels on a platform that shows only a failing test and, for compile errors, just a `line:col` position. Related authoring gotchas: [[triton-perf-tips]]. Worked example: [[triton-flash-attention]].

## 1. Find out what kind of failure it is

| Symptom | Likely class | First move |
|---|---|---|
| Compile error, no readable text, `line:col` only | lowering / constraint failure | bisect (section 2) |
| Compile error `map::at` | compiler failed to lower an op (version-specific), often `tl.dot` with fp32 operands | give the dot an `input_precision`; check operand dtypes |
| `tl.arange` complaint | extent not constexpr, or not a power of two | annotate `tl.constexpr` in the kernel **signature**; check the extent |
| Runs, wrong numbers | indexing, mask, broadcast-axis or formula bug | hand-compute a tiny case (section 3) |
| Runs, correct on some tests only | something that depends on sizes: tail masks, unequal lengths, small dims | add asymmetric tests (section 4) |
| `NaN` or `inf` in outputs | first-iteration state, `0 * inf`, `0 / 0`, `-inf - (-inf)` | trace the first iteration by hand |
| "Undefined name" style failures | renamed or commented-out variables still used later | read the whole kernel for stale names |

The reported position of a compile error is not always the faulty line: an undefined name later in the kernel can surface elsewhere, so fix the obvious errors first.

## 2. Bisect when you can't read the error

Treat each hypothesis as an experiment and change **one thing per submission**, noting the error position each time. If the position moves or the error disappears, you learned something.

- **Strip the kernel down:** zeros to the output, then add back loads, then the dot, then the rest. Stop at the piece that brings the failure back.
- **Beware dead-code elimination.** An unused result (`s` never consumed) can be removed by the compiler, which also removes the ops that fed it. "It compiles when I comment out the dot" does not clear the operations that only fed the dot. Keep the intermediate alive by making it affect something that is stored.
- **Remove loops** (run one iteration) to separate loop-carried-state bugs from per-iteration ones.
- **Substitute a dot-free equivalent** with the same output shape (a broadcast multiply plus a reduction) to isolate the dot.

## 3. Hand-compute a tiny case

The most efficient debugging step in this whole workflow. Pick inputs where the right answer is obvious by hand: all-zero queries make softmax uniform (a failing test had `Q = 0`, two keys, and the expected output was the plain average, 7.5, while the kernel gave 5, which immediately showed only one key row was used). A first iteration with `m_old = -inf` is a good tiny case for any online algorithm: write down each intermediate and check for `0/0`, `0 * inf`, `-inf - (-inf)`.

- Check the **first element of a non-zero program id's block** and the **first element past the last valid row**: do the pointer and the mask agree?
- For pointer tiles, derive the offsets on a toy matrix on paper (a 4x3 row-major example) before touching code.

## 4. Test with asymmetric inputs

Bugs that survive symmetric data: a transposed `AᵀB` or a stride swap, a broadcast along the wrong axis, a mask built for a different tensor. They compile, run, and pass tests where dimensions coincide.

- Use shapes where **no two dimensions are equal**: `M ≠ N ≠ K`, `BLOCK_M ≠ BLOCK_N` (temporarily set one to 32), `D ≠ block size`.
- Use sizes that are **not multiples of the block**, so tail masks run.
- Use **several chunks** in any loop, with the last one partial.
- Test **tiny** sizes (`D = 2`) and larger ones (`D = 64`): padding and minimum-extent bugs live at the extremes.
- Test **unequal lengths** wherever two quantities could differ (`N_q ≠ N_kv`).
- Use non-symmetric, non-constant data.

## 5. Derive formulas, don't copy them

A formula transcribed from a paper may be misprinted or have a different convention. For anything with a direction, a sign, or an inverse (an online-softmax rescale factor, a normalization): derive it from what the accumulator holds and what you need it to hold, then check the first-iteration case and one numerical example. The FlashAttention-2 pseudocode as printed has an update with an inverse on the output but not on the denominator; the two must use the same factor.

## 6. Keep similar quantities named separately

Many bugs were two things that looked alike being treated as one:

- query length vs key/value length (`N_q` vs `N_kv`),
- logical size vs tile extent (`D` vs `BLOCK_DMODEL`),
- the load mask vs the score mask (different axes),
- row indices vs element offsets,
- a runtime value vs a `constexpr`.

Name each role distinctly, annotate each tensor's shape in a comment, and when passing positional integer arguments keep the kernel signature and the launch call in the same order.

## 7. Process habits

- **One change per submission** and record the outcome.
- Compare against PyTorch on the same inputs and find the **first row** that disagrees, then ask whether it is in the first block or the last partial one.
- Remove debug prints from the host code before submitting.
- When a fix works, ask **why** (for example, the dot's minimum extent), and write the reason down. The mental model matters more than the patch.

---

Related: [[triton-perf-tips]], [[triton-flash-attention]], [[triton-tiled-matmul]], [[training-diagnostics]], [[pytorch-hooks]]
