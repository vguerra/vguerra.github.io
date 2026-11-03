# Vendored grammars

## `mlir.json`: MLIR TextMate grammar

Syntax highlighting for fenced `mlir` code blocks. Shiki, which Expressive
Code uses for highlighting, does not bundle an MLIR grammar, so it is
registered in `ec.config.mjs`.

- **Source:** https://github.com/llvm/llvm-project/blob/main/mlir/utils/textmate/mlir.json
- **Upstream commit:** `9e087a6ed0c0` (2026-01-18), "[mlir][vscode] Update to capture angle brackets in types/attrs (#176665)"
- **License:** Apache License v2.0 with LLVM Exceptions
  (https://github.com/llvm/llvm-project/blob/main/LICENSE.TXT)
- **Modifications:** none. The file is byte-for-byte upstream.

To update, re-download the file from the URL above and bump the commit here.

LLVM IR needs no vendored file: Shiki bundles it as `llvm`.
