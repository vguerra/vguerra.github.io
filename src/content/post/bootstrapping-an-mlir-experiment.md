---
title: "Bootstrapping a tiny out-of-tree MLIR experiment"
description: "Build the smallest useful out-of-tree MLIR project: one pass, one mlir-opt-style driver, one test, and one build script."
publishDate: "2026-10-14"
tags: ["mlir", "compilers"]
---

Most MLIR tutorials start inside the LLVM monorepo: add a directory, touch a
`CMakeLists.txt`, rebuild a few thousand files. That is the right setup if you
are upstreaming a dialect. It is a heavy setup if you just want to try a
rewrite pattern on a Tuesday evening.

This post builds the smallest useful **out-of-tree** MLIR project: one pass,
one custom `mlir-opt`-style driver, one test file, and a build script `build.sh`,
that goes from nothing to a runnable binary. It links against an existing LLVM 
installation you have already, so a build takes seconds in your machine.

The pass we use is rather trivial: It rewrites `x + 0` and `0 + x` to
`x`. The point is rather to understand the scaffolding around it,
because the scaffolding is what you will be reusing for future experiments.

This is the structure of the experiment:

## What we are building

```
fold-add-zero/
├── CMakeLists.txt              # project root: finds MLIR, wires subdirs
├── build.sh                    # configure + build, one command
├── fold-add-zero.mlir          # test input with FileCheck expectations
├── lib/
│   ├── CMakeLists.txt          # the pass, as a library
│   ├── fold-add-zero-rewrite.h
│   └── fold-add-zero-rewrite.cpp
└── tools/
    ├── CMakeLists.txt          # the driver executable
    └── my-opt.cpp
```

We want to build two targets:

| target | what it is |
|---|---|
| `FoldAddZeroPass` | a static library containing the pass |
| `my-opt` | an executable: `mlir-opt`, plus our pass registered |

Splitting the pass into a **library** and the driver into an **executable** is
the structural idea worth taking away. The pass knows nothing about command
lines. The driver knows nothing about what the pass does. Adding a second pass
later means adding a second library and one more line in `my-opt.cpp`.

## Why a custom driver at all?

`mlir-opt` can only run passes that were compiled into it. Ours was not. So we
build our own `mlir-opt`: a `main()` that calls the same `MlirOptMain` entry
point the real tool uses, but registers our pass first. You get every standard
flag (`--verify-each`, `--mlir-print-ir-after-all`, `-o`, `--split-input-file`)
for free.

## The root `CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.20.0)

project(zero-rewrite-pattern LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})
include(AddLLVM)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddMLIR)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_BINARY_DIR})

add_subdirectory(lib)
add_subdirectory(tools)
```

Line by line, since this is the part that looks like boilerplate until it
breaks:

**`CMAKE_CXX_STANDARD 17`.** LLVM's headers require at least C++17. This also
pins you to it: `std::format` and other C++20 facilities do not exist in this
project, which is why LLVM-style code reaches for `llvm::formatv` instead.

**`CMAKE_EXPORT_COMPILE_COMMANDS ON`.** Writes `compile_commands.json` into
the build directory. Your editor's language server (clangd) reads it to learn
include paths. Without it, every `#include "mlir/..."` in the editor is red
even though the build succeeds. `build.sh` symlinks it to the project root for
exactly this reason. Note that this line is optional, you can still compile
without it.

**`find_package(MLIR REQUIRED CONFIG)`.** The important line. `CONFIG` mode
means "do not search for a `FindMLIR.cmake`; find the `MLIRConfig.cmake` that
the MLIR installation ships." That file defines the variables and CMake
targets the rest of the file uses. It also pulls in LLVM's equivalent, which
is why `LLVM_DIR` gets set without us asking.

If CMake cannot find it, you set `MLIR_DIR` to the directory *containing*
`MLIRConfig.cmake` (for Homebrew: `<prefix>/lib/cmake/mlir`). That is the one
knob you will actually touch, and `build.sh` is where it gets set.

**`include(AddLLVM)` and `include(AddMLIR)`.** These bring in the helper
functions `add_llvm_executable` and `add_mlir_library`, which wrap
`add_executable`/`add_library` with LLVM's compiler flags and, in the library's
case, the extra bookkeeping MLIR expects. `AddMLIR` lives in `MLIR_CMAKE_DIR`,
so that directory has to be added to `CMAKE_MODULE_PATH` first, which is why
the `list(APPEND ...)` sits between the two `include`s.

**`set(MLIR_BINARY_DIR ...)`.** Inherited from MLIR's own standalone example.
In `AddMLIR.cmake` the only reader is `add_mlir_doc`, which writes generated
documentation under it. This project never calls that function, so you can
delete the line; I kept it so the file stays recognizable next to upstream's.

**The four `include_directories`.** LLVM and MLIR headers, plus your own
source and build directories. `PROJECT_SOURCE_DIR` is what lets `my-opt.cpp`
write `#include "lib/fold-add-zero-rewrite.h"` rather than a relative path
that breaks when you move the file.

**`add_subdirectory`.** Hands control to `lib/` and `tools/`. Order matters
here only loosely: `tools/` links against a target defined in `lib/`, but CMake
resolves target names after reading everything, so it would work either way.
Keeping the dependency order is just easier to read.

## The library: `lib/CMakeLists.txt`

```cmake
add_mlir_library(FoldAddZeroPass
    fold-add-zero-rewrite.cpp

    LINK_LIBS PUBLIC
    MLIRArithDialect
    MLIRFuncDialect
    MLIRPass
    MLIRTransforms
)
```

`add_mlir_library` takes a name, the sources, and the MLIR libraries the code
uses. `LINK_LIBS PUBLIC` means anything that links `FoldAddZeroPass` also
inherits these.

The list is not arbitrary. Each entry is there because the `.cpp` calls into it:

| library | used for |
|---|---|
| `MLIRArithDialect` | `arith::AddIOp` |
| `MLIRFuncDialect` | `func::FuncOp` (the pass is anchored on it) |
| `MLIRPass` | `PassWrapper`, `PassRegistration` |
| `MLIRTransforms` | `applyPatternsGreedily` |

**This list is where you will get stuck next.** MLIR is split into dozens of
small static libraries. Forget one and the compile succeeds but the *link*
fails with `Undefined symbols ... referenced from ...`. The fix is always the
same: find which library defines the missing symbol (`nm -g
.../llvm/lib/libMLIR*.a | grep <symbol>`) and add it here.

## The driver: `tools/`

```cmake
set (LIBS
    FoldAddZeroPass
    MLIROptLib
    MLIRArithDialect
    MLIRFuncDialect
)

add_llvm_executable(my-opt my-opt.cpp)
llvm_update_compile_flags(my-opt)
target_link_libraries(my-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(my-opt)
```

`MLIROptLib` is the library that provides `MlirOptMain`. `FoldAddZeroPass` is
our own library from `lib/`, referenced by target name; CMake wires up both the
link and the build order.

`mlir_check_all_link_libraries` is a safety net with one specific job. It
prints a warning if the target links both the monolithic `LLVM` shared library
and an individual `LLVM*` component library. That combination can load the
same symbols twice and fail strangely at runtime. It matters here because
Homebrew's LLVM ships `libLLVM.dylib`, and naming an `LLVM*` target by hand in
`LIBS` is an easy way to trigger it.

And the whole driver:

```cpp
#include "lib/fold-add-zero-rewrite.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  registry.insert<
      mlir::arith::ArithDialect,
      mlir::func::FuncDialect
  >();

  mlir::registerFoldAddZeroPass();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "My OPT Pass Driver", registry));
}
```

Two registrations, two different jobs:

- **`registry.insert<...>`** tells the parser which *dialects* exist. Leave one
  out and any file using it fails to parse. I removed `ArithDialect` to check:

  ```
  fold-add-zero.mlir:7:13: error: Dialect `arith' not found for custom op 'arith.constant'
  ```

  Note the message blames the input file, not your driver. When you add a new
  dialect to a test and get "not found", this is the line to check.
- **`registerFoldAddZeroPass()`** tells the pass manager which *passes* exist,
  so that `-fold-add-zero` on the command line resolves to ours.

Everything else, argument parsing, file reading, verification, printing, is
inside `MlirOptMain`.

## The pass, briefly

The header exposes one function, so the driver does not need to see the pass's
internals:

```cpp
// lib/fold-add-zero-rewrite.h
namespace mlir { void registerFoldAddZeroPass(); }
```

The rewrite itself is an `OpRewritePattern` rooted at `arith.addi`:

```cpp
struct FoldAddZero final : OpRewritePattern<arith::AddIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AddIOp op,
                                PatternRewriter &rewriter) const override {
    if (matchPattern(op.getLhs(), m_Zero())) {
      // 0 + x  ->  x
      rewriter.replaceOp(op, op.getRhs());
      return success();
    }
    if (matchPattern(op.getRhs(), m_Zero())) {
      // x + 0  ->  x
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }
    return rewriter.notifyMatchFailure(op, "neither operand is zero");
  }
};
```

`m_Zero()` comes from `mlir/IR/Matchers.h` and matches a value defined by a
constant zero. All mutation goes through `rewriter`; `replaceOp` redirects
every use of the add's result to the surviving operand and erases the add.

A pattern does nothing on its own. It needs a **driver** to apply it, which is
what the pass wrapper provides:

```cpp
struct FoldAddZeroPass final
    : PassWrapper<FoldAddZeroPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldAddZeroPass)

  StringRef getArgument() const final { return "fold-add-zero"; }
  StringRef getDescription() const final {
    return "Replace integer additions with zero";
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldAddZero>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

void mlir::registerFoldAddZeroPass() {
  static PassRegistration<FoldAddZeroPass> registration;
}
```

`getArgument()` is the command-line name. `OperationPass<func::FuncOp>` says
the pass runs once per function, which lets MLIR run it on functions in
parallel. `applyPatternsGreedily` keeps applying patterns until nothing
changes.

## `build.sh`, step by step

Here is the script as I first wrote it:

```sh
#!/bin/sh

if [ -d "./build" ]; then
    echo "removing build directory"
    rm -r build
fi
mkdir -p build

pushd build

MLIR_DIR=$(brew --prefix llvm) cmake .. -G Ninja
cmake --build . --target FoldAddZeroPass
cmake --build . --target my-opt

popd

ln -sf ./build/compile_commands.json
```

What each part does:

1. **Delete and recreate `build/`.** A clean configure every time. CMake caches
   settings like `MLIR_DIR` in `build/CMakeCache.txt`, so a stale directory
   can keep using an old LLVM after you have fixed your script. For a project
   this small, a clean build costs seconds, and it removes a whole category of
   "works after I deleted build/" confusion.
2. **`pushd build`.** An out-of-source build: generated files stay in `build/`
   and your source tree stays clean (one `build/` entry in `.gitignore`).
3. **`MLIR_DIR=... cmake .. -G Ninja`.** The *configure* step. CMake reads the
   `CMakeLists.txt` files, runs `find_package(MLIR)`, and writes a Ninja build
   graph. Nothing is compiled yet. `-G Ninja` picks the generator; Ninja is
   faster than Makefiles for incremental builds and is what LLVM itself uses.
4. **`cmake --build . --target FoldAddZeroPass`.** Compiles `lib/` into
   `build/lib/libFoldAddZeroPass.a`.
5. **`cmake --build . --target my-opt`.** Compiles the driver and links it
   against that library, producing `build/tools/my-opt`.
6. **`popd`, then `ln -sf ... compile_commands.json`.** Symlinks the compilation
   database from `build/` into the project root so the editor finds it.

Note that here, **building two targets explicitly is redundant.** `my-opt` depends on
`FoldAddZeroPass`, so asking for `my-opt` builds both, in order, I just wanted
to show you how to specify the build targets explicitly.

### A little note for Homebrew users on MacOS

You can ask Homebrew, where your LLVM installation lives:

```sh
$(brew --prefix llvm)/lib/cmake/mlir
```

`brew --prefix llvm` returns the stable `/opt/homebrew/opt/llvm` symlink, which
follows upgrades. Passing the directory that actually contains
`MLIRConfig.cmake`, via `-DMLIR_DIR=`, is also more direct than the
environment-variable form I started with.

Two smaller issues while I was in there:

- **`pushd`/`popd` are not POSIX.** The script has a `#!/bin/sh` shebang, and
  it happens to work on macOS because `sh` there is bash in disguise. On a
  system where `sh` is `dash` (Debian, Ubuntu), `pushd` does not exist. CMake
  has a cleaner answer anyway: `-S` and `-B` choose the source and build
  directories without changing directory at all.

### A shortere version of the build script

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Override with MLIR_DIR=... if you built LLVM yourself.
MLIR_DIR="${MLIR_DIR:-$(brew --prefix llvm)/lib/cmake/mlir}"

rm -rf build
cmake -S . -B build -G Ninja -DMLIR_DIR="$MLIR_DIR"
cmake --build build --target my-opt
ln -sf build/compile_commands.json compile_commands.json
```

`set -euo pipefail` makes the script stop at the first failure instead of
ploughing on to a confusing second error, and `cd "$(dirname "$0")"` means it
works no matter where you invoke it from. 

## Running it

```
$ ./build/tools/my-opt --help | grep fold-add-zero
      --fold-add-zero      -   Replace integer additions with zero
```

Our pass is in the list. Run it on:

```mlir
func.func @rhs_zero(%x: i32) -> i32 {
  %zero = arith.constant 0 : i32
  %sum = arith.addi %x, %zero : i32
  return %sum : i32
}
```

```
$ ./build/tools/my-opt fold-add-zero.mlir -fold-add-zero
module {
  func.func @rhs_zero(%arg0: i32) -> i32 {
    return %arg0 : i32
  }
  ...
}
```

The `addi` is gone, and `return` now uses the function argument directly:

```
%x ─────────┐                           %x ──────────────→ func.return
            ├─ arith.addi → %sum →  ⇒
%zero ──────┘
```

The `arith.constant` disappeared too, because once `replaceOp` removes its only
user the greedy driver erases the now-dead constant.

## Where to go from here

You can find the code in this github repository: [Tiny OOT MLIR example](https://github.com/vguerra/tiny-oot-mlir-experiment).

You now have a skeleton where each new experiment is a new library in `lib/`,
one registration line in `my-opt.cpp`, and a `.mlir` file. The same skeleton
carries a type converter and conversion pass, a dialect with TableGen, or a
transform-dialect extension. For those you add libraries to `LINK_LIBS`, and
for TableGen you add the `mlir_tablegen` and `add_public_tablegen_target`
calls, but the shape stays the same: library, driver, test file, build script.
