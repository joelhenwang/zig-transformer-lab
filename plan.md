# `zig-transformer-lab` — Hand-off Plan for the CUDA-Ready Implementation Agent

**Document version:** 1.0 (final, approved by user)
**Audience:** A coding agent running on a CUDA-ready Linux machine, with no prior context.
**Scope:** Everything required to implement, test, document, and ship the `zig-transformer-lab` package without further clarification from the user.

> This document is self-contained. The receiving agent should not need to re-research the Zig/CUDA/ML ecosystem, should not need to ask clarifying questions about scope, and should not need to guess at conventions. All foundational decisions are locked. Anything not locked is a local implementation choice the agent is authorized to make.

---

## Table of Contents

1. [Mission and Success Criteria](#1-mission-and-success-criteria)
2. [Locked Decisions (D1 through D14)](#2-locked-decisions-d1-through-d14)
3. [Ecosystem Map (research performed on the user's behalf)](#3-ecosystem-map-research-performed-on-the-users-behalf)
4. [Pre-Flight Checklist](#4-pre-flight-checklist)
5. [Final Repository Layout](#5-final-repository-layout)
6. [Build System Specification](#6-build-system-specification)
7. [Coding and Documentation Conventions](#7-coding-and-documentation-conventions)
8. [Staged Execution Plan (Stages 1 through 9)](#8-staged-execution-plan-stages-1-through-9)
9. [Python Oracle Design](#9-python-oracle-design)
10. [Testing Matrix](#10-testing-matrix)
11. [Error-Handling Policy](#11-error-handling-policy)
12. [Working Agreements for the Receiving Agent](#12-working-agreements-for-the-receiving-agent)
13. [Risks and Mitigations](#13-risks-and-mitigations)
14. [First Actions (Literal Checklist)](#14-first-actions-literal-checklist)
15. [Appendix A — `AGENTS.md` Template](#appendix-a--agentsmd-template)
16. [Appendix B — `.gitignore` Contents](#appendix-b--gitignore-contents)
17. [Appendix C — `LICENSE` Contents (MIT)](#appendix-c--license-contents-mit)
18. [Appendix D — `README.md` Skeleton](#appendix-d--readmemd-skeleton)
19. [Appendix E — `docs/00_overview.md` Outline](#appendix-e--docs00_overviewmd-outline)
20. [Appendix F — Row-Major cuBLAS GEMM Derivation](#appendix-f--row-major-cublas-gemm-derivation)
21. [Appendix G — Kernel Launch Recipes and Shape Contracts](#appendix-g--kernel-launch-recipes-and-shape-contracts)
22. [Appendix H — Checkpoint File Format](#appendix-h--checkpoint-file-format)

---

## 1. Mission and Success Criteria

### 1.1 Mission

Build a small, heavily commented, pedagogical personal Zig 0.16.0 library that trains a **tiny one-block, one-head, word-level transformer from scratch** on CPU first, then accelerates it with CUDA (cuBLAS plus a minimal set of custom `.cu` kernels). The package is learning infrastructure: every line is written to teach a beginner how PyTorch-like systems work internally.

The user's context:
- Knows Python well, is a complete beginner in Zig.
- Near-zero current ML/DL knowledge; studied some math/stats in university and forgot most of it.
- Zero GPU/kernel programming background.
- Wants explanations that assume learning-from-near-zero, code that is heavily commented and pedagogical, and extensive design notes explaining why each implementation works.

### 1.2 Hardware and environment

- Linux.
- NVIDIA RTX 4060 Ti 16 GB (compute capability `sm_89`).
- Zig 0.16.0 (exact).
- CUDA Toolkit 12 or newer (nvcc, `libcuda`, `libcudart`, `libcublas`).
- Python 3 with NumPy and PyTorch for the oracle.

### 1.3 Success definition (end state after Stage 9)

1. `zig build test` is green on CPU-only machines (no GPU required for most tests).
2. `zig build test -Dcuda=true` is green on the target machine.
3. `zig build run-example -Dexample=04_overfit_one_batch` drives loss to near zero on a single batch within 100 steps.
4. `zig build run-example -Dexample=05_train_tiny` decreases loss on an ultra-tiny crafted corpus and prints a coherent continuation.
5. `zig build run-example -Dexample=06_train_shakespeare -Dcuda=true` runs end-to-end on GPU, at least 30x faster than CPU, and generates Shakespeare-like word sequences.
6. `zig build run-example -Dexample=08_cuda_vs_cpu -Dcuda=true` reports max absolute diff less than `5e-5` between CPU and CUDA forward pass for the full model, and less than `1e-4` between gradients after one step.
7. Ten `docs/0X_*.md` chapters exist and form a readable book covering every subsystem from Zig primer through PyTorch parallels.

### 1.4 Non-goals

- Not a production ML framework.
- No mixed precision (no `f16`/`bf16`).
- No distributed or multi-GPU training.
- No ONNX, safetensors, or HuggingFace interop.
- No pure-Zig GPU kernel code (kernels are CUDA C compiled with `nvcc`).
- No lazy tensors, no graph compiler, no MLIR, no XLA.

---

## 2. Locked Decisions (D1 through D14)

These are final. Do not revisit without explicit user approval.

| # | Decision | Value |
|---|----------|-------|
| D1 | CUDA binding strategy | Hand-written minimal wrapper in `src/backend/cuda/bindings.zig` that dynamically loads `libcuda.so.1`, `libcudart.so`, and `libcublas.so` via `dlopen` plus `dlsym`. **No external Zig-CUDA dependency.** |
| D2 | CPU BLAS | **None.** Path is: CPU-naive f32 matmul (correctness oracle) then directly to CUDA with cuBLAS. |
| D3 | CUDA kernel build | Offline `nvcc -arch=sm_89 -ptx` at Zig build time. `.ptx` artifacts emitted under `zig-out/ptx/`. Loaded at runtime with `cuModuleLoadData`. No NVRTC, no JIT. |
| D4 | Python oracle | `tools/oracle.py` using NumPy plus PyTorch (PyTorch only for full-model autograd ground truth). |
| D5 | Model shape during Stages 2 through 7 | **Hard-coded 1 block / 1 head.** Generalization to configurable `n_layer`, `n_head` is an explicit Stage 8 refactor exercise with a guided solution. |
| D6 | Documentation | Book-level `docs/0X_*.md` chapter per stage (500 to 1500 lines each) plus heavy inline comments in every `.zig` file. **Raw markdown only.** No static-site generator, no mdbook. ASCII art for diagrams. |
| D7 | Environment | Linux; system Zig 0.16.0 on `$PATH`; system CUDA Toolkit (nvcc plus `libcuda`, `libcudart`, `libcublas`); RTX 4060 Ti (`sm_89`). |
| D8 | Training corpora | Both: `data/tiny.txt` (~5 KB crafted, for overfit tests) and `data/tinyshakespeare.txt` (~1 MB, for realistic run). |
| D9 | Float type | `f32` only throughout. No `f16`/`bf16`/mixed precision. |
| D10 | License | **MIT only.** No `NOTICE` file. Third-party architectural inspirations are credited inline in the relevant file's header (short phrase: `// Inspired by <project> (link); no code copied.`) and once in `docs/00_overview.md`. |
| D11 | Zig version | Pinned to **0.16.0** (exact). Record in `build.zig.zon` as `.minimum_zig_version = "0.16.0"`. |
| D12 | Autograd style | Tape-based reverse-mode, dynamic graph, not generic. Covers only the operations needed for this transformer. |
| D13 | Attention variant | Pre-norm causal self-attention. Post-norm tradeoff documented. |
| D14 | Randomness | `std.Random.Xoshiro256` seeded from a config-provided `u64`; every random operation takes an explicit seed. Runs must be deterministic given the same seed. |

### 2.1 Additional locked policies (finalized alongside D1-D14)

- **P1. Artifacts.** `docs/pre_flight.md`, `zig-out/`, `.zig-cache/`, loss CSVs, checkpoints, and Python venvs are all gitignored. See Appendix B.
- **P2. Agent brief file.** `AGENTS.md` at the repository root is committed in Stage 1 and mirrors Section 12 of this plan. See Appendix A.
- **P3. Stage boundaries.** Each stage ends with (a) passing tests, (b) a runnable example, (c) the matching `docs/0X_*.md` chapter committed. Do not proceed to the next stage until acceptance criteria are met.
- **P4. Commits.** One commit per stage with message `stage(N): <summary>`. Acceptance-criterion outputs pasted into the commit body.

### 2.2 What the receiving agent is allowed to decide locally

- Exact Zig idioms (`std.ArrayList(T)` vs `std.BoundedArray(T, N)`, etc).
- Private struct field ordering, private helper names.
- Test fixture values for hand-computed checks.
- Whether a given op's backward is fused or split, as long as results match the oracle within tolerance.
- Additional inline comments and examples beyond those required here.

### 2.3 What requires the user's explicit approval

- Changing any of D1 through D14 or P1 through P4.
- Introducing any third-party Zig dependency in `build.zig.zon`.
- Skipping, merging, or reordering stages.
- Using `f16`/`bf16` anywhere.
- Writing GPU kernel code in any language other than CUDA C.
- Adding Python dependencies beyond `numpy` and `torch` to `tools/oracle.py`.

---

## 3. Ecosystem Map (research performed on the user's behalf)

The following scan was performed in May 2026. The receiving agent should treat these projects as **design references only**. No code is to be copied from any of them. Re-derive cleanly.

| Project | URL | Role | License | Notes |
|---------|-----|------|---------|-------|
| `Marco-Christiani/zigrad` | github.com/Marco-Christiani/zigrad | Design reference for autograd, graph tracing, parameter iteration. | **LGPL-3.0** | Under active rewrite through ~mid 2026. **Do not copy code** (license incompatibility). |
| `CogitatorTech/zigformer` | github.com/CogitatorTech/zigformer | Closest educational analogue in pure Zig. Tokenizer and training-loop shape reference. | MIT | Zig 0.15.2. Read, do not port verbatim. |
| `karpathy/nanoGPT`, `karpathy/llm.c` | github.com/karpathy/nanoGPT, github.com/karpathy/llm.c | Canonical minimal-transformer references. | MIT | Our math must match these. |
| Rust `cudarc` | github.com/coreylowman/cudarc | Three-layer `sys`/`result`/`safe` architecture idea for the CUDA wrapper. | MIT/Apache-2.0 | Re-implement cleanly in Zig. |
| `aiurion/zigCUDA` | github.com/aiurion/zigCUDA | Existing pure-Zig bindings to CUDA Driver API. Zig 0.16.0. | MIT | **Not used as dependency** (D1). Source may be consulted for how to `dlopen`/`dlsym` cleanly. |
| `akhildevelops/cudaz` | github.com/akhildevelops/cudaz | Most mature Zig CUDA wrapper; driver + NVRTC. | MIT-like | **Not used as dependency** (D1). |
| `coderonion/zcuda` | github.com/coderonion/zcuda | Broad binding surface (cuBLAS, cuBLASLt, cuRAND, cuDNN, etc). | MIT | **Not used as dependency** (D1). Too large for scope. |
| `gwenzek/cudaz` | github.com/gwenzek/cudaz | Toy wrapper using Zig's LLVM PTX backend. | Apache-2.0 | **Incompatible with policy** (no pure-Zig GPU kernels). |
| `zml/zml` | github.com/zml/zml | Production inference stack (XLA/MLIR/Bazel). | Apache-2.0 | Off-limits as reference for this pedagogical scope. |
| `zillama/ggml-zig` | github.com/zillama/ggml-zig | Pure-Zig `ggml` port. Closest to the name "zgml". | varies | Not used; ggml's quantization is off-topic for f32 training. |

### 3.1 Reference classification summary

- **Use as design reference (study, then rewrite):** `zigrad` (architecture only, not code), `zigformer`, `nanoGPT`, `llm.c`, `cudarc`.
- **Use as optional dependency:** None. D1 locks us to a hand-written wrapper.
- **May adapt small mechanical patterns (with file-header attribution):** cuBLAS handle lifecycle, `Dim3`/`LaunchConfig` struct shape, tokenization helpers. These are idiomatic and trivially re-derived.
- **Avoid:** `zml`, `gwenzek/cudaz`, `ggml-zig` for the reasons above.

The `docs/00_overview.md` chapter must include this table and rationale.

---

## 4. Pre-Flight Checklist

Before starting Stage 1, the receiving agent verifies the machine and records the output to `docs/pre_flight.md` (gitignored, local-only).

```bash
zig version                                 # expect 0.16.0
nvcc --version                              # expect CUDA >= 12.0
nvidia-smi                                  # expect RTX 4060 Ti, driver >= 550
ls /usr/local/cuda/lib64/libcuda.so* \
   /usr/lib/x86_64-linux-gnu/libcuda.so* 2>/dev/null
ldconfig -p | grep -E 'libcuda|libcudart|libcublas'
python3 -c 'import numpy, torch; print(numpy.__version__, torch.__version__, torch.cuda.is_available())'
echo $CUDA_HOME
uname -a
```

**Stop condition:** If `zig version` is not exactly `0.16.0`, or `nvcc --version` reports CUDA < 12, or `torch.cuda.is_available()` returns `False`, or none of the CUDA libraries are discoverable, the agent stops and reports to the user rather than improvising.

**CUDA toolkit path detection.** `build.zig` uses, in order: `-Dcuda_home=<path>` flag, `$CUDA_HOME` env var, then falls back through `/usr/local/cuda`, `/opt/cuda`, `/usr/lib/cuda`.

---

## 5. Final Repository Layout

```
zig-transformer-lab/
|-- build.zig
|-- build.zig.zon
|-- README.md
|-- LICENSE                         # MIT
|-- AGENTS.md                       # Agent brief (Appendix A)
|-- .gitignore
|-- docs/
|   |-- 00_overview.md
|   |-- 01_zig_primer.md
|   |-- 02_tensors.md
|   |-- 03_autograd.md
|   |-- 04_nn.md
|   |-- 05_transformer_math.md
|   |-- 06_tokenizer_data.md
|   |-- 07_cpu_training.md
|   |-- 08_backends_cuda.md
|   |-- 09_debugging.md
|   `-- 10_pytorch_parallels.md
|-- src/
|   |-- root.zig                    # package entry, re-exports
|   |-- core/
|   |   |-- allocator.zig
|   |   |-- dtype.zig
|   |   |-- device.zig
|   |   |-- errors.zig
|   |   `-- rng.zig
|   |-- tensor/
|   |   |-- shape.zig
|   |   |-- tensor.zig
|   |   |-- print.zig
|   |   `-- ops/
|   |       |-- create.zig
|   |       |-- elementwise.zig
|   |       |-- reduce.zig
|   |       |-- matmul.zig
|   |       |-- unary.zig
|   |       |-- softmax.zig
|   |       `-- loss.zig
|   |-- autograd/
|   |   |-- node.zig
|   |   |-- tape.zig
|   |   |-- backward.zig
|   |   `-- gradcheck.zig
|   |-- nn/
|   |   |-- module.zig
|   |   |-- linear.zig
|   |   |-- embedding.zig
|   |   |-- layernorm.zig
|   |   |-- activations.zig
|   |   |-- attention.zig
|   |   |-- mlp.zig
|   |   |-- block.zig
|   |   `-- model.zig
|   |-- optim/
|   |   |-- optimizer.zig
|   |   |-- sgd.zig
|   |   `-- adamw.zig
|   |-- tokenizer/
|   |   |-- vocab.zig
|   |   `-- word.zig
|   |-- data/
|   |   |-- dataset.zig
|   |   |-- windowing.zig
|   |   `-- batcher.zig
|   |-- backend/
|   |   |-- backend.zig             # vtable interface
|   |   |-- cpu_naive/
|   |   |   `-- dispatch.zig
|   |   `-- cuda/
|   |       |-- bindings.zig
|   |       |-- context.zig
|   |       |-- mem.zig
|   |       |-- module.zig
|   |       |-- gemm.zig
|   |       |-- dispatch.zig
|   |       `-- kernels/
|   |           |-- elementwise.cu
|   |           |-- softmax.cu
|   |           |-- layernorm.cu
|   |           |-- gelu.cu
|   |           |-- embedding.cu
|   |           |-- causal_mask.cu
|   |           |-- ce_loss.cu
|   |           `-- adamw.cu
|   |-- debug/
|   |   |-- assert_shape.zig
|   |   |-- finite.zig
|   |   `-- compare.zig
|   `-- lab/
|       `-- train.zig
|-- examples/
|   |-- 01_tensor_playground.zig
|   |-- 02_autograd_scalar.zig
|   |-- 03_autograd_tensor.zig
|   |-- 04_overfit_one_batch.zig
|   |-- 05_train_tiny.zig
|   |-- 06_train_shakespeare.zig
|   |-- 07_generate.zig
|   `-- 08_cuda_vs_cpu.zig
|-- tests/
|   |-- unit_all.zig                # imports every src/**/*.zig so tests run
|   |-- integration_cpu.zig
|   `-- integration_cuda.zig        # only compiled when -Dcuda=true
|-- tools/
|   |-- requirements.txt            # numpy, torch
|   |-- oracle.py
|   |-- compare_outputs.py
|   `-- plot_loss.py
`-- data/
    |-- tiny.txt
    `-- tinyshakespeare.txt
```

---

## 6. Build System Specification

The `build.zig` is authored in Stage 1 and extended incrementally. It must expose the following user-facing surface.

### 6.1 Build options

- `-Doptimize=Debug|ReleaseSafe|ReleaseFast` (standard Zig option).
- `-Dtarget=...` (standard; default native).
- `-Dcuda=bool` (default `false`) — enables CUDA backend compilation, kernel build, and integration tests.
- `-Dcuda_arch=string` (default `"sm_89"`) — nvcc `-arch` flag.
- `-Dcuda_home=string` (default: autodetect from `$CUDA_HOME` then `/usr/local/cuda`, `/opt/cuda`, `/usr/lib/cuda`).
- `-Dexample=string` — name of an example file under `examples/` (without extension) for `run-example`.
- `-Dseed=u64` (default `1337`) — passed into examples as a build-options constant.

### 6.2 Build steps

- `zig build` — builds the library module and all CPU-only examples. CUDA targets skipped unless `-Dcuda=true`.
- `zig build test` — runs all unit tests in `src/**/*.zig` plus `tests/integration_cpu.zig`. CUDA integration tests only included when `-Dcuda=true`.
- `zig build run-example -Dexample=NAME` — runs the named example.
- `zig build kernels` (only when `-Dcuda=true`) — invokes `nvcc` for each `.cu` under `src/backend/cuda/kernels/` and writes `zig-out/ptx/<name>.ptx`. Must be a dependency of every CUDA-enabled artifact.
- `zig build docs` — prints a table of `docs/*.md` chapter line counts (sanity check).

### 6.3 nvcc invocation

```
nvcc -O3 \
     -arch=${cuda_arch} \
     -ptx \
     --use_fast_math \
     -Xcompiler -fPIC \
     -o zig-out/ptx/<kernel>.ptx \
     src/backend/cuda/kernels/<kernel>.cu
```

### 6.4 Linking

- When `-Dcuda=true`: the bindings module opens libraries dynamically at runtime, so the executable must link `libc` (and on older glibc systems, explicitly `linkSystemLibrary("dl")`). Do **not** `linkSystemLibrary("cuda")`.
- The CUDA toolkit is only required at **build time** (for `nvcc`). At runtime only the NVIDIA driver plus `libcublas.so` are required.

### 6.5 Zig module graph

- Single library module `zig_transformer_lab` from `src/root.zig`.
- Each example imports only that module.
- `tests/unit_all.zig` `@import`s every `.zig` file under `src/` so `zig build test` runs every embedded `test { ... }` without manual registration. The receiving agent maintains this file as new source files are added.

---

## 7. Coding and Documentation Conventions

Applied uniformly to every `.zig` file.

### 7.1 File header block (first 20 to 60 lines)

Every source file begins with a block comment containing:

1. Purpose (one paragraph).
2. Shape contract for any tensor-shaped entry points.
3. Math formulas (forward and, where applicable, backward) in LaTeX-flavored ASCII.
4. Memory ownership contract (who allocates, who frees).
5. Error conditions.
6. TODO markers for future optimization.
7. Credits where an architectural idea came from outside.

### 7.2 Public function doc comments

- Minimum one worked shape example.
- State memory ownership for any returned value that allocates.

### 7.3 Simplicity rules

- No clever abstractions in the first pass. Inline loops are preferred where they teach.
- Generics used only when duplication is clearly worse than learning cost.
- No `@panic` outside tests. Use the library error set.

### 7.4 Memory and errors

- Every allocating function takes an explicit `std.mem.Allocator`. No hidden global allocators.
- Every `Tensor` records whether it owns its data buffer.
- Single library-wide error set in `src/core/errors.zig`:

```
pub const LabError = error{
    ShapeMismatch,
    OutOfMemory,
    InvalidArgument,
    CudaError,
    IoError,
    NotImplemented,
    NumericalError,
};
```

Library functions return `!T` using this error set.

### 7.5 Shape assertions

- Every op calls `debug.assertShape(...)` at entry.
- Active in `Debug` and `ReleaseSafe`; compiled out in `ReleaseFast`.

### 7.6 Tests

- Tests co-located via Zig's `test "..."` blocks inside the source files they test.
- Any test using randomness seeds the RNG explicitly.
- No test depends on wall-clock time or sleeps.

### 7.7 Formatting

- `zig fmt` clean. No tabs. Trailing newline at EOF.

### 7.8 Documentation chapters

- Raw markdown. ASCII diagrams only.
- Per-stage `docs/0X_*.md` chapter delivered in the same commit as its code.
- Minimum 500 lines; target 500 to 1500.
- Each chapter ends with "Common mistakes" and 2 to 4 "Exercise + Solution" pairs.

---

## 8. Staged Execution Plan (Stages 1 through 9)

Each stage is a complete unit of work. When a stage closes, its tests pass, its example runs, its docs chapter is committed. **Do not proceed to the next stage until its acceptance criteria are met.**

---

### Stage 1 — Project scaffold and overview doc

**Goal.** Buildable skeleton with no ML code yet, plus `docs/00_overview.md`.

**Files to create.**

- `build.zig`, `build.zig.zon`
- `README.md` (see Appendix D for skeleton)
- `LICENSE` (MIT; see Appendix C)
- `AGENTS.md` (see Appendix A)
- `.gitignore` (see Appendix B)
- `src/root.zig` (empty public API with re-export scaffolding)
- `src/core/errors.zig` (the error set from 7.4)
- `tests/unit_all.zig` (stub `@import` list; empty is acceptable until Stage 2)
- `docs/00_overview.md` (see Appendix E for outline)

**Commit.** `stage(1): scaffold, overview, agent brief`.

**Acceptance.**

- `zig build` succeeds.
- `zig build test` runs zero tests without error.
- If `-Dcuda=true`, `nvcc` is detected and the kernels step exists (no kernels yet, so it is a no-op).

---

### Stage 2 — CPU tensor foundation

**Goal.** Implement `Tensor` and every required op on CPU in pure Zig f32, with tests and a playground example.

**Files.**

- `src/core/{allocator,dtype,device,rng}.zig`
- `src/tensor/{shape,tensor,print}.zig`
- `src/tensor/ops/{create,elementwise,reduce,matmul,unary,softmax,loss}.zig`
- `examples/01_tensor_playground.zig`
- `docs/01_zig_primer.md`, `docs/02_tensors.md`

**Concepts implemented.**

- `Shape` (up to rank 4), `Strides`, row-major indexing, contiguous check, `view` vs `owned`.
- `Tensor` struct: `{ data: []f32, shape: Shape, strides: Strides, device: Device, owned: bool }`.
- Creation: `zeros`, `ones`, `full`, `randn`, `randu`, `arange`.
- Elementwise with NumPy-style broadcasting (rank <= 3): `add`, `sub`, `mul`, `div`, plus scalar variants and in-place forms.
- Reduction with axis: `sum`, `mean`, `max`.
- Linear algebra: `matmul` naive triple loop (rank-2), `matmul_batch` (rank-3 batched), `transpose_2d` (view), `reshape` (view when possible else copy).
- Unary: `exp`, `log`, `neg`, `relu`, `gelu_exact`.
- Numerically stable `softmax` and `log_softmax` along the last axis.
- `cross_entropy(logits, targets)` returning mean scalar loss.
- `print.debugSummary(tensor)` — shape, strides, dtype, min/mean/max, NaN/Inf count.

**Tests.** One unit test per op with a hand-computed ground-truth value on a tiny tensor (for example `softmax([[1, 2, 3]])`). Numerical tolerance `1e-6` for f32.

**Example.** `01_tensor_playground.zig` demonstrates shape arithmetic, broadcasting a `(1, 3)` into `(2, 3)`, and prints a softmax.

**Docs.**

- `01_zig_primer.md` — allocators, slices, comptime basics, error unions, `defer`, ownership — tuned to what this library uses.
- `02_tensors.md` — row-major derivation, stride math, broadcasting rules with ASCII diagrams, numerical stability of softmax/log_softmax.

**Commit.** `stage(2): cpu tensor foundation`.

**Acceptance.**

- `zig build test` green with at least 30 unit tests.
- `01_tensor_playground.zig` runs; output matches a printed reference saved in the docs chapter.
- NumPy oracle comparison (`tools/oracle.py tensor-op --op softmax ...`) shows max abs diff < `1e-6` on the playground inputs.

---

### Stage 3 — Autograd engine

**Goal.** Tape-based reverse-mode autograd, restricted to the ops needed for the transformer.

**Files.**

- `src/autograd/{node,tape,backward,gradcheck}.zig`
- Extend `src/tensor/tensor.zig` with `requires_grad: bool`, `grad: ?*Tensor`, hidden `tape_node: ?NodeId`.
- `examples/02_autograd_scalar.zig`, `examples/03_autograd_tensor.zig`
- `docs/03_autograd.md`

**Design.**

- A single global-per-step `Tape` created and destroyed by the user. Ops on tensors with `requires_grad=true` append a `Node` recording: op kind, parent `NodeId`s, saved tensors/scalars required for backward, and a function pointer to the backward closure.
- `backward(loss)` performs iterative (not recursive) post-order traversal from the loss node, accumulating gradients into each leaf's `.grad` tensor.
- Non-leaf intermediate grads are freed after use unless `retain_graph = true`.
- Broadcasting backward: broadcasted dims are summed out via `sum_to_shape`.
- Gradient accumulation: `.grad` writes are always `+=`, never `=`.
- `zero_grad(params)` resets all leaf grads.

**Ops with registered backward.**

- Elementwise add, sub, mul, div (plus scalar forms).
- Matmul: `dA = dC @ B^T`, `dB = A^T @ dC`; batched variant included.
- Transpose, reshape, view (no-op backward modulo contiguity).
- Sum, mean along axis.
- Exp, log, neg, relu, gelu.
- Softmax, log_softmax (standalone versions provided for teaching; usually fused with CE at the call site).
- Cross-entropy (fused `log_softmax + NLL` backward: `dlogits = softmax(logits) - one_hot(targets)`).
- Embedding (forward = gather, backward = scatter-add into weight grad).

**`gradcheck.zig`.** Central-difference comparator with configurable `eps` (default `1e-3`) and `tol_rel` (`1e-2`). Samples a subset of parameter indices for speed.

**Examples.**

- `02_autograd_scalar.zig`: micrograd-style hello world, compute `(a * b + c)^2`, print grads for `a, b, c`.
- `03_autograd_tensor.zig`: small tensor graph `loss = mean(softmax(X @ W)^2)`, gradient check against finite diff.

**Docs.** `03_autograd.md`:

- Why tapes work (worked dependency DAG).
- Derivation of matmul backward.
- Derivation of fused softmax-CE backward.
- Broadcasting-sum-reduction trick.
- Gotchas: aliasing grads, `retain_graph`, double backward explicitly not supported.

**Commit.** `stage(3): tape-based autograd`.

**Acceptance.**

- At least 10 gradcheck tests pass (every backward formula has a dedicated check).
- With `retain_graph=false`, no memory leaks (Zig leak-checker clean in Debug).
- PyTorch oracle on `03_autograd_tensor.zig` matches grads within `1e-4` rel.

---

### Stage 4 — `nn` module and optimizers

**Goal.** All layers needed for the transformer plus SGD and AdamW.

**Files.**

- `src/nn/{module,linear,embedding,layernorm,activations,attention,mlp,block,model}.zig`
- `src/optim/{optimizer,sgd,adamw}.zig`
- `docs/04_nn.md`, `docs/05_transformer_math.md`

**Module protocol.** Each layer exposes:

- `init(allocator, cfg, rng) !Self`
- `forward(self, input, tape) !Tensor`
- `parameters(self, list: *std.ArrayList(*Tensor)) !void`
- `deinit(self)`

**Layers.**

- `Linear(in, out)` — Kaiming-uniform init; bias optional (default `true`).
- `Embedding(vocab, d_model)` — standard lookup; scatter-add backward.
- `LayerNorm(d_model, eps=1e-5)` — affine gamma/beta, forward with Welford, vectorized backward.
- `GELU` — exact via `0.5 * x * (1 + erf(x / sqrt(2)))`; tanh approximation commented but not default.
- `CausalSelfAttention(d_model)` — `num_heads=1` hard-coded. Forward: `Q=XW_q, K=XW_k, V=XW_v`, scores `= Q @ K^T / sqrt(d_k)`, causal mask add, softmax, `attn @ V`, output projection.
- `MLP(d_model, d_ff=4*d_model)` — `Linear -> GELU -> Linear`.
- `TransformerBlock` — pre-norm: `x = x + Attn(LN(x))`, then `x = x + MLP(LN(x))`.
- `TinyWordTransformer(cfg)` — `tok_embed + pos_embed -> Block -> final LN -> lm_head`. Weight tying between `tok_embed` and `lm_head` off by default; commented as an exercise.

**Optimizers.**

- `SGD(lr, momentum=0, weight_decay=0)`.
- `AdamW(lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)` — decoupled weight decay, bias-corrected first and second moment estimates.
- Shared `Optimizer` vtable with `step(params)` and `zero_grad(params)`.

**Checkpointing.** `src/nn/model.zig::save(path)` / `load(path)` — binary format defined in Appendix H.

**Docs.**

- `04_nn.md` — module abstraction, parameter iteration, initialization strategies.
- `05_transformer_math.md` — full shape trace for `B=2, T=8, V=64, D=32` with ASCII art for attention.

**Commit.** `stage(4): nn layers and optimizers`.

**Acceptance.**

- Forward on a random batch produces `(B, T, V)` logits with correct shape and finite values.
- One optimizer step on a constant input decreases loss.
- PyTorch oracle reproducing the same init reports matching logits within `1e-4`.

---

### Stage 5 — Tokenizer and dataset pipeline

**Goal.** Word-level tokenizer that round-trips, plus batching.

**Files.**

- `src/tokenizer/{vocab,word}.zig`
- `src/data/{dataset,windowing,batcher}.zig`
- `docs/06_tokenizer_data.md`

**Tokenizer spec.**

- Lowercase-only by default (config option to keep case).
- Whitespace split, peel leading/trailing punctuation from each token: `. , ! ? ; : " ' ( )`. Punctuation becomes its own token.
- Collapse runs of whitespace.
- Specials: `<unk>=0`, `<pad>=1`, `<bos>=2`, `<eos>=3` (IDs reserved in that order).
- Vocab frequency cutoff: keep top `max_vocab - 4` words by count; the rest map to `<unk>`.
- Serialization: one `id\tword\n` per line, UTF-8, stable across runs (sorted by id).

**Dataset.**

- Load a `.txt` file into `[]u32` token stream.
- Windowing: produce `(input, target)` pairs where `target[t] = input[t+1]`. Generate all valid windows of length `T`.
- Batcher: Xoshiro-seeded deterministic shuffle of window indices, drop-last batching.
- Debug: `print_batch(batch, vocab)` shows IDs and decoded words side by side.

**Commit.** `stage(5): word-level tokenizer and dataset`.

**Acceptance.**

- Round-trip encode/decode of `data/tiny.txt` differs from the original only in whitespace collapse.
- On `data/tinyshakespeare.txt` with `max_vocab=2000`, OOV rate <= 5%.

---

### Stage 6 — End-to-end CPU training

**Goal.** Fully working CPU training loop.

**Files.**

- `src/lab/train.zig` — top-level trainer.
- `examples/04_overfit_one_batch.zig`
- `examples/05_train_tiny.zig`
- `examples/06_train_shakespeare.zig` (CPU path; will be re-used in Stage 7 with `-Dcuda=true`).
- `examples/07_generate.zig` — load checkpoint, sample.
- `docs/07_cpu_training.md`

**Training loop skeleton.**

```
for step in 1..max_steps:
    batch = batcher.next()
    tape = Tape.init(alloc)
    logits = model.forward(batch.input, &tape)
    loss = cross_entropy(logits, batch.target)
    tape.backward(loss)
    optim.step(params)
    optim.zero_grad(params)
    tape.deinit()
    if step % log_every == 0:   log(step, loss)
    if step % sample_every == 0: generate_sample()
    if step % ckpt_every == 0:   model.save(path)
```

**Generation.** Greedy plus top-k (k default 5) plus temperature (default `1.0`). Runs on whichever backend the model is on.

**Docs.** `07_cpu_training.md` with full shape trace *through* `train.zig`, a line-by-line explanation of `optim.step()`, and commentary on what residual connections and LayerNorm do for training stability.

**Commit.** `stage(6): end-to-end cpu training`.

**Acceptance.**

- Overfit test: single batch, loss < `0.05` in <= 100 steps across 5 seeds.
- Tiny corpus: loss decreases monotonically over 1000 steps.
- Shakespeare on CPU: loss < `5.0` within 500 steps at `T=16, D=32, V=2000`.

---

### Stage 7 — CUDA backend

**Goal.** Move forward plus backward onto the GPU via cuBLAS GEMM and a small set of custom `.cu` kernels. Cross-validate against CPU.

**Files.**

- `src/backend/backend.zig` — vtable.
- `src/backend/cpu_naive/dispatch.zig` — wraps Stage 2 ops.
- `src/backend/cuda/{bindings,context,mem,module,gemm,dispatch}.zig`
- `src/backend/cuda/kernels/*.cu`
- `examples/08_cuda_vs_cpu.zig`
- `tests/integration_cuda.zig`
- `docs/08_backends_cuda.md`

#### 7.A Bindings (`bindings.zig`)

Dynamically load three shared libraries and resolve only the symbols we need:

```
libcuda.so.1:
    cuInit, cuDeviceGet, cuCtxCreate_v2, cuCtxDestroy_v2,
    cuCtxSetCurrent, cuModuleLoadData, cuModuleGetFunction,
    cuLaunchKernel, cuStreamCreate, cuStreamDestroy_v2,
    cuStreamSynchronize, cuMemAlloc_v2, cuMemFree_v2,
    cuMemcpyHtoD_v2, cuMemcpyDtoH_v2, cuGetErrorString

libcudart.so:
    cudaGetLastError, cudaDeviceSynchronize  (for friendlier errors)

libcublas.so:
    cublasCreate_v2, cublasDestroy_v2, cublasSgemm_v2,
    cublasSgemmStridedBatched, cublasSetStream_v2,
    cublasGetStatusString
```

All wrappers convert non-success codes into `error.CudaError`. In Debug builds, the numeric code and the symbol name are logged, and the error message from `cuGetErrorString` or `cublasGetStatusString` is stored thread-locally for `debug.lastCudaError()` retrieval.

#### 7.B Context (`context.zig`)

```
pub const CudaContext = struct {
    device: CUdevice,
    context: CUcontext,
    stream: CUstream,
    cublas: cublasHandle_t,
    ptx_modules: std.StringHashMap(CUmodule),
    allocator: std.mem.Allocator,
};
```

`init(alloc, device_id)`:

1. `cuInit(0)`
2. `cuDeviceGet` on `device_id` (default 0)
3. `cuCtxCreate_v2`
4. `cuStreamCreate`
5. `cublasCreate_v2`
6. `cublasSetStream_v2`
7. Load every `.ptx` under `zig-out/ptx/` via `cuModuleLoadData` and cache by file stem in `ptx_modules`.

#### 7.C Memory (`mem.zig`)

`DeviceBuffer` RAII over `cuMemAlloc_v2`:

- `DeviceBuffer.from_host(ctx, slice_f32) !Self`
- `self.to_host(alloc) ![]f32`
- `self.deinit()`
- `Tensor.to_cuda(ctx)` and `Tensor.to_cpu(alloc)` preserve shape/strides.

#### 7.D Row-major GEMM wrapper (`gemm.zig`)

**This is the single most error-prone spot in the codebase.** See Appendix F for the full diagrammatic derivation. The wrapper presents a row-major `C = alpha * A @ B + beta * C` API and internally calls cuBLAS (column-major) with swapped operands.

Batched matmul uses `cublasSgemmStridedBatched` for `(B, M, K) @ (B, K, N)`.

A dedicated unit test multiplies two known `(2, 3) @ (3, 4)` matrices and compares against the CPU result to within `1e-5`.

#### 7.E Kernels (`src/backend/cuda/kernels/*.cu`)

All kernels are small, single-purpose, f32-only, and have a matching host launcher in `dispatch.zig`. Each `.cu` file starts with a comment block: purpose, grid/block recipe, memory access pattern, backward counterpart.

See Appendix G for launch recipes and shape contracts. Minimum set:

- `elementwise.cu` — add, sub, mul, div, scalar ops, in-place residual add.
- `softmax.cu` — row-wise with max-subtract trick; block per row, shared memory for max + sum.
- `layernorm.cu` — online mean/variance per row (Welford), gamma/beta affine, backward included.
- `gelu.cu` — exact forward and backward.
- `embedding.cu` — forward gather; backward `atomicAdd` scatter.
- `causal_mask.cu` — adds -infinity above diagonal into scores.
- `ce_loss.cu` — fused `log_softmax + NLL + grad` w.r.t. logits.
- `adamw.cu` — per-parameter step with bias correction.

Every kernel:

- Starts with `if (idx >= n) return;` bounds check.
- Never issues out-of-bounds global memory reads or writes.
- Has a matching unit test with tiny fixed inputs and a CPU reference.
- Is callable from `dispatch.zig` via `cuLaunchKernel` with a parameter-packing helper.

#### 7.F Backend dispatch

`src/backend/backend.zig`:

```
pub const Backend = struct {
    ctx: ?*anyopaque,
    vtable: *const VTable,
};

pub const VTable = struct {
    matmul:        *const fn(*Backend, *Tensor, *const Tensor, *const Tensor) anyerror!void,
    matmul_batch:  *const fn(*Backend, *Tensor, *const Tensor, *const Tensor) anyerror!void,
    add:           ...
    softmax_row:   ...
    layernorm:     ...
    gelu:          ...
    embedding_fwd: ...
    embedding_bwd: ...
    causal_mask:   ...
    ce_loss:       ...
    adamw_step:    ...
    to_device:     ...
    to_host:       ...
};
```

Autograd calls `backend.vtable.matmul(...)` instead of `ops.matmul(...)` directly. CPU and CUDA provide identical semantics.

#### 7.G Cross-validation (`08_cuda_vs_cpu.zig`)

- Full-model forward on CPU and CUDA on an identical random batch: max abs diff < `5e-5`.
- One training step on both: gradient max abs diff < `1e-4`, parameter diff after `optim.step` < `2e-4`.

**Docs.** `08_backends_cuda.md`:

- Why matmul dominates transformer compute (FLOP-count derivation).
- Why cuBLAS.
- What cuBLAS hides (algorithm selection, tensor cores; not used here since f32, but mentioned).
- How PyTorch dispatches high-level ops to cuBLAS/cuDNN (ATen dispatcher overview).
- Row-major/column-major derivation with diagrams (copied from Appendix F and expanded).
- PTX loading lifecycle and why we precompile offline.
- Kernel-by-kernel walkthrough.

**Commit.** `stage(7): cuda backend`.

**Acceptance.**

- Every kernel has a unit test.
- `08_cuda_vs_cpu.zig` passes tolerances above.
- `06_train_shakespeare.zig -Dcuda=true` is at least 30x faster than CPU and produces a similar loss trajectory (final loss within 10% of the CPU run at matched steps).

---

### Stage 8 — Debugging discipline and N-block refactor

**Goal.** Make the project debuggable and generalize from `(n_layer=1, n_head=1)` to arbitrary small `N`.

**Files.**

- Expand `src/debug/*` with utilities proven useful during Stages 2 through 7.
- Refactor `src/nn/model.zig` and `src/nn/block.zig` to read `n_layer` and `n_head` from `Config`.
- `docs/09_debugging.md`.

**Debug utilities (final form).**

- `assertShape(tensor, expected)` with rich error messages including both shapes.
- `assertFinite(tensor)` — scans for NaN/Inf, prints the first offending index.
- `compare(cpu_tensor, cuda_tensor, tol)` — device-aware diff helper.
- `dump(tensor, path)` — writes `.bin` dump for Python-side inspection.

**`docs/09_debugging.md` must cover.**

- Shape-assert driven development (example: a wrong transpose producing a downstream shape error three functions away; how to read the chain).
- NaN/Inf detection workflow: when it happens, where to look first (softmax overflow, `log(0)`, large `lr`).
- Gradient checking a new op.
- CPU vs GPU output comparison workflow.
- Overfit-one-batch smoke test.
- `compute-sanitizer --tool memcheck zig-out/bin/train_shakespeare` walkthrough, with an intentionally broken kernel variant (in a sidebar file not compiled by default) and the exact output the user should see.
- Nsight Compute beginner pass on `softmax_rowwise`: launch, inspect warp occupancy, memory throughput.
- Common CUDA errors catalog with repros:
  1. Illegal memory access — dropped bounds check.
  2. Wrong grid/block — off-by-one on `T`.
  3. Non-coalesced access — wrong stride in `layernorm_rowwise`.
  4. Missed `cudaStreamSynchronize` — reading host data before copy completes.
  5. Host/device pointer mixup.
  6. Row-major vs column-major in cuBLAS.

**Multi-block refactor.** `Config { n_layer: u8, n_head: u8, d_model: u32, d_ff: u32, block_size: u32, vocab_size: u32, dropout: f32 }` (dropout field present for future use; not implemented now). Model holds a slice of blocks. Attention supports `n_head > 1` by reshaping `(B, T, D) -> (B, n_head, T, D/n_head)` and batched matmul. Guided solution lives in `docs/09_debugging.md`.

**Commit.** `stage(8): debugging tools and n-block refactor`.

**Acceptance.**

- `compute-sanitizer` clean run on `06_train_shakespeare -Dcuda=true`.
- 2-block, 2-head, `D=64` smoke run completes in < 2x the time of the 1/1/32 model.

---

### Stage 9 — Documentation finalization

**Goal.** Complete `docs/10_pytorch_parallels.md` and polish.

**Deliverables.**

- PyTorch parallels mapping:
  - `Tensor` vs `torch.Tensor` (storage + sizes + strides).
  - Autograd tape vs `torch.autograd.Function` / dynamic graph.
  - `nn.Module` protocol vs `torch.nn.Module`.
  - Our optimizers vs `torch.optim.SGD`/`AdamW`.
  - Backend vtable vs ATen dispatcher + device-specific backends.
  - Row-major cuBLAS wrapper vs `at::cuda::blas::gemm`.
  - Custom CUDA kernels vs fused kernels in `aten/src/ATen/native/cuda/`.
- README per subfolder (short, one to two paragraphs, pointing at the relevant chapter).
- "Common mistakes" sidebar and 2 to 4 "Exercise + Solution" pairs per chapter.
- Final inline-comment polish across `src/`.

**Commit.** `stage(9): documentation finalization`.

**Acceptance.**

- `zig build docs` shows every chapter at >= 500 lines.
- `grep -r TODO src/` returns only future-optimization TODOs, not missing-implementation TODOs.

---

## 9. Python Oracle Design

### 9.1 `tools/requirements.txt`

```
numpy>=1.24
torch>=2.0
```

Install into `tools/.venv/` with `python3 -m venv tools/.venv && tools/.venv/bin/pip install -r tools/requirements.txt`. The venv is gitignored. Zig code has no torch dependency.

### 9.2 `tools/oracle.py`

CLI subcommands:

- `oracle.py tensor-op --op <softmax|matmul|layernorm|...> --in input.bin --shape 2,3 --out expected.bin` — reads raw f32 binary written by `debug.dump`, runs NumPy equivalent, writes expected output.
- `oracle.py gradcheck --model tiny_config.json --seed 1337 --in batch.bin --out grads.bin` — builds a PyTorch model with matching weights, runs forward+backward, dumps gradients for Zig to diff.

### 9.3 `tools/compare_outputs.py`

Takes two `.bin` files plus shape and tolerance, prints max abs diff, mean diff, first offending index.

### 9.4 `tools/plot_loss.py`

Reads a CSV produced by `lab/train.zig` (columns: `step,loss,lr`) and plots loss vs step. Optional; CPU-only.

---

## 10. Testing Matrix

| Test class | Runs on | Trigger |
|------------|---------|---------|
| Unit tests per op | CPU | `zig build test` |
| Autograd gradchecks | CPU | `zig build test` |
| Tokenizer round-trip | CPU | `zig build test` |
| Model forward shape | CPU | `zig build test` |
| Overfit-one-batch | CPU | `zig build run-example -Dexample=04_overfit_one_batch` |
| Kernel unit tests | GPU | `zig build test -Dcuda=true` |
| CUDA vs CPU parity | GPU | `zig build run-example -Dexample=08_cuda_vs_cpu -Dcuda=true` |
| Integration train_tiny | CPU | `zig build run-example -Dexample=05_train_tiny` |
| Integration train_shakespeare | CPU and GPU | `zig build run-example -Dexample=06_train_shakespeare [-Dcuda=true]` |

All tests seed their RNG; no test depends on wall-clock.

---

## 11. Error-Handling Policy

- All fallible library functions return the error set from `src/core/errors.zig` (Section 7.4).
- CUDA errors: wrapper returns `error.CudaError`; a thread-local `last_cuda_error_message` is populated from `cuGetErrorString` / `cublasGetStatusString`. Users inspect via `debug.lastCudaError()`.
- Out of memory is always `error.OutOfMemory`. Allocators are explicit.
- Shape mismatches: `error.ShapeMismatch`; the message contains both shapes in human-readable form.
- IO errors propagated as `error.IoError` with context.
- Tests that expect errors use `try std.testing.expectError(...)`.

---

## 12. Working Agreements for the Receiving Agent

These are the operating rules while implementing the library.

1. **Never skip a stage's acceptance criteria** to come back later. If a test is flaky, fix it before moving on.
2. **Commit after each stage** with message `stage(N): <summary>` and paste acceptance-criterion outputs into the commit body.
3. **Keep docs in lockstep with code.** A stage's docs chapter lands in the same commit as its code.
4. **No `@panic` escapes.** If you do not know how to handle a branch, raise it to the user; do not hide it.
5. **Treat the row-major / column-major cuBLAS wrapping as sacred.** If a CUDA test fails with a small but nonzero diff, suspect this first.
6. **If a Zig 0.16.0 stdlib API differs from what you expect**, trust the compiler error and read the diagnostic carefully before guessing.
7. **When stuck, ask the user** with a crisp options-style question: no more than 4 options, never ambiguous.
8. **Every new `.zig` file must be reachable** through `src/root.zig` or `tests/unit_all.zig`; otherwise its tests silently do not run.
9. **Every kernel `.cu` must have** a matching Zig launcher in `dispatch.zig`, a shape contract in its file header, and a corresponding unit test.
10. **Do not introduce dependencies.** Locked by D1. Ask first.
11. **Do not copy code from `zigrad`** (LGPL). Only architectural ideas, attributed.
12. **Assume offline mode.** Do not rely on network access during builds or tests.
13. **Prefer CPU-naive correctness first.** Every CUDA kernel is validated against its CPU counterpart before being trusted.
14. **Determinism.** Every run reproduces with the same seed.

A condensed version of these agreements lives in `AGENTS.md` at the repository root.

---

## 13. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Zig 0.16.0 stdlib churn (ArrayList, GPA, Random, etc). | Pin in `build.zig.zon`. Isolate stdlib-touching helpers in `src/core/`. |
| Row-major vs column-major confusion in cuBLAS. | Confined to `src/backend/cuda/gemm.zig`. Dedicated test. Documented with diagrams in `docs/08_backends_cuda.md` and Appendix F. |
| Word-level vocab explosion on Shakespeare. | Frequency cutoff default `max_vocab=2000`. OOV via `<unk>` tracked and logged. |
| Slow CPU matmul for Shakespeare. | Accepted. CPU is correctness path. Shakespeare is a GPU workload after Stage 7. |
| CUDA kernel out-of-bounds. | Stage 8 compute-sanitizer gate. Every kernel has bounds check. |
| License contamination. | No code copied from `zigrad` (LGPL). File headers credit inspirations only. |
| Checkpoint format drift. | Magic plus version tag. `load` rejects mismatched version with a clear error. |
| Determinism drift across backends. | Acceptance tolerances defined per test. Stage 7 cross-validation catches regressions. |
| NVIDIA driver or CUDA version changes. | Dlopen by name (`.so.1`). `build.zig` honors `-Dcuda_home` and `$CUDA_HOME`. |

---

## 14. First Actions (Literal Checklist)

1. Read this entire plan. Internalize Locked Decisions D1 through D14 and policies P1 through P4.
2. Run the Pre-Flight Checklist (Section 4) and write outputs to `docs/pre_flight.md` (local-only, gitignored).
3. If any pre-flight check fails, stop and report to the user.
4. Create the Stage 1 file set:
   - `build.zig`, `build.zig.zon`
   - `README.md` (Appendix D)
   - `LICENSE` (Appendix C)
   - `AGENTS.md` (Appendix A)
   - `.gitignore` (Appendix B)
   - `src/root.zig`
   - `src/core/errors.zig`
   - `tests/unit_all.zig`
   - `docs/00_overview.md` (Appendix E)
5. Run `zig build` and `zig build test`. Both must succeed.
6. Commit: `stage(1): scaffold, overview, agent brief`. Paste acceptance outputs into the commit body.
7. Proceed Stage 2 through Stage 9 in order. Do not interleave.

---

## Appendix A — `AGENTS.md` Template

The receiving agent creates this file in Stage 1. It mirrors Section 12.

```markdown
# Agent Brief — zig-transformer-lab

## One-sentence mission
Build a pedagogical Zig 0.16.0 library that trains a tiny 1-block 1-head word-level transformer on CPU, then on CUDA, with extensive documentation.

## Hard rules
- Do not violate Locked Decisions D1 through D14 or policies P1 through P4. Ask the user before changing any decision.
- No third-party Zig dependencies.
- Zig version pinned at 0.16.0 (exact).
- f32 only. No mixed precision.
- GPU kernels are CUDA C only. No pure-Zig GPU kernels.
- Kernels are compiled offline with nvcc to .ptx and loaded via cuModuleLoadData.
- Never copy code from LGPL sources. Architectural ideas only, attributed in file headers.

## Workflow
- Implement stages 1 through 9 in order. Do not interleave stages.
- Commit after each stage with `stage(N): <summary>`; paste acceptance outputs into the commit body.
- Docs chapter for a stage ships in the same commit as the stage's code.

## Style
- File header block: purpose, shape contract, math formulas, ownership, errors, TODOs, credits.
- Explicit allocators. No hidden globals. No @panic outside tests.
- `zig fmt` clean. Tests co-located via `test "..."` blocks.
- Every public function has a worked shape example in its doc comment.

## CUDA sacred spots
- Row-major to column-major wrapping in `src/backend/cuda/gemm.zig`. Dedicated tests.
- Bounds checks in every kernel.
- Offline .ptx only (no NVRTC).
- Bindings module opens libraries with dlopen at runtime.

## When stuck
- Ask the user with a crisp options-style question (max 4 options).
- Never guess at hardware, environment, or decisions.

## Full plan
See `plan.md` at the repository root (or the copy the user provides) and `docs/00_overview.md`.
```

---

## Appendix B — `.gitignore` Contents

```
# Zig
zig-out/
zig-cache/
.zig-cache/

# Python tools
tools/.venv/
.venv/
__pycache__/
*.pyc

# Pre-flight (local only)
docs/pre_flight.md

# Run artifacts
logs/
*.loss.csv
*.ckpt
*.bin
!data/*.bin         # placeholder; currently no binary data shipped

# Editor / OS
.vscode/
.idea/
*.swp
.DS_Store
Thumbs.db
```

---

## Appendix C — `LICENSE` Contents (MIT)

```
MIT License

Copyright (c) 2026 <USER NAME>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The receiving agent replaces `<USER NAME>` with the user's name before committing.

---

## Appendix D — `README.md` Skeleton

```markdown
# zig-transformer-lab

A small, heavily commented, pedagogical Zig 0.16.0 library for training a tiny
one-block, one-head, word-level transformer from scratch. Designed to teach
the internals of PyTorch-like systems, tensor libraries, autograd, training
loops, and CUDA acceleration.

## Hardware and software requirements
- Linux
- Zig 0.16.0 (exact)
- NVIDIA GPU with compute capability >= 8.0 recommended (developed on RTX 4060 Ti, sm_89)
- CUDA Toolkit >= 12 for GPU builds
- Python 3 with numpy and torch for the oracle (optional)

## Quickstart
```bash
git clone <url>
cd zig-transformer-lab
zig build test
zig build run-example -Dexample=01_tensor_playground
```

## CUDA build
```bash
zig build test -Dcuda=true
zig build run-example -Dexample=06_train_shakespeare -Dcuda=true
```

## Reading order
See `docs/00_overview.md`. Start with `docs/01_zig_primer.md` if you are new to Zig.

## License
MIT. See `LICENSE`.
```

---

## Appendix E — `docs/00_overview.md` Outline

Sections in order:

1. **Mission** — one paragraph matching Section 1.1 of this plan.
2. **How to read these docs** — recommended chapter order; prerequisites per chapter.
3. **Ecosystem map** — Section 3 table, with rationale for each classification.
4. **Locked decisions** — the D1-D14 table and P1-P4 policies.
5. **Repository layout** — Section 5 tree with brief notes per folder.
6. **Quickstart** — matches README.
7. **Glossary** — tensor, shape, stride, view, grad, tape, kernel, PTX, cuBLAS, etc.
8. **Where to go next** — pointer to each subsequent chapter.

---

## Appendix F — Row-Major cuBLAS GEMM Derivation

cuBLAS operates in **column-major** layout. Our tensors are **row-major**. A row-major matrix `M` of shape `(rows, cols)` stored contiguously has the exact same bytes as a column-major matrix `M^T` of shape `(cols, rows)`.

We want to compute (row-major):

```
C(M, N) = alpha * A(M, K) @ B(K, N) + beta * C(M, N)
```

Equivalently, re-interpreting every operand as column-major (i.e. transposed):

```
(C^T)(N, M) = alpha * (B^T)(N, K) @ (A^T)(K, M) + beta * (C^T)(N, M)
```

So the cuBLAS call we need is:

```
cublasSgemm(
    handle,
    CUBLAS_OP_N, CUBLAS_OP_N,   // neither transposed (we already transposed by reinterpretation)
    N, M, K,                    // col-major M, N, K for the C^T result
    &alpha,
    B_device_ptr, N,            // lda for B^T in col-major is N
    A_device_ptr, K,            // lda for A^T in col-major is K
    &beta,
    C_device_ptr, N             // lda for C^T in col-major is N
);
```

Swap `A` and `B` in the argument list, swap `M` and `N` in the dimensions, leading dimensions come from the row-count of the column-major transposed view. This single wrapper is the only place in the codebase that needs to know about column-major. Everywhere else we stay row-major.

**Unit test.** `A = [[1,2,3],[4,5,6]]` (2x3), `B = [[1,0,1],[0,1,0],[1,0,1]]` (3x3). Expected `C = [[4,2,4],[10,5,10]]`. Compute with the wrapper; assert max abs diff < `1e-5`.

Batched variant: `cublasSgemmStridedBatched` with strides `M*K`, `K*N`, `M*N`, operand swap identical to the single-GEMM case.

---

## Appendix G — Kernel Launch Recipes and Shape Contracts

All launchers live in `src/backend/cuda/dispatch.zig`. All kernels take `f32*` pointers and `i32` or `u32` sizes. No templates.

### G.1 `elementwise_add(out, a, b, n)`
- Inputs: `a, b: (n,)`; Output: `out: (n,)`.
- Grid: `ceil(n / 256)`, Block: `256`.
- Thread work: `i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return; out[i] = a[i] + b[i];`
- Backward: `elementwise_add_bwd` scatters `d_out` to both `d_a` and `d_b` (identity).

### G.2 `softmax_rowwise(out, x, rows, cols)`
- Inputs: `x: (rows, cols)`; Output: `out: (rows, cols)`.
- Grid: `rows`, Block: `min(next_pow2(cols), 256)`.
- Shared memory: one f32 for max, one for sum.
- Algorithm: parallel max across threads in block; subtract; parallel sum of exp; divide.
- Backward: `y = softmax(x); dx = y * (dy - sum(dy * y, axis=-1, keepdim))`.

### G.3 `layernorm_rowwise(out, x, gamma, beta, rows, cols, eps)`
- Inputs: `x: (rows, cols)`, `gamma, beta: (cols,)`; Output: `out: (rows, cols)`.
- Grid: `rows`, Block: `min(next_pow2(cols), 256)`.
- Shared memory for Welford mean/variance reduction.
- Output: `y = gamma * (x - mean) / sqrt(var + eps) + beta`.
- Backward kernel stores `mean` and `rstd = 1 / sqrt(var + eps)` to avoid recomputation; derives `dx`, `dgamma`, `dbeta`.

### G.4 `gelu(out, x, n)` and `gelu_bwd(dx, x, dy, n)`
- Forward: `out[i] = 0.5 * x[i] * (1 + erff(x[i] * 0.7071067811865475f));`
- Backward uses derivative of exact GELU.
- Grid/block: `ceil(n / 256)` / `256`.

### G.5 `embedding_fwd(out, weight, ids, B, T, D, V)`
- Inputs: `weight: (V, D)`, `ids: (B, T)` u32; Output: `out: (B, T, D)`.
- Grid: `(B, T)`, Block: `min(D, 256)`.
- Each block copies the row `weight[ids[b,t], :]` into `out[b, t, :]`.

### G.6 `embedding_bwd(d_weight, d_out, ids, B, T, D, V)`
- Grid: `(B, T)`, Block: `min(D, 256)`.
- Uses `atomicAdd` to scatter `d_out[b, t, :]` into `d_weight[ids[b, t], :]`.

### G.7 `causal_mask_add(scores, B, T)`
- Inputs/Output in place: `scores: (B, T, T)`.
- Grid: `(B, T)`, Block: `T`.
- Adds `-INFINITY` to positions `scores[b, i, j]` where `j > i`.

### G.8 `ce_loss_fwd_bwd(loss, dlogits, logits, targets, B, T, V)`
- Inputs: `logits: (B, T, V)`, `targets: (B, T)` u32; Outputs: `loss` scalar, `dlogits: (B, T, V)`.
- Grid: `(B, T)`, Block: `min(V, 256)`.
- Per-row: compute max, sum of exp, log_softmax, NLL at `targets[b, t]`, and `dlogits = softmax - one_hot(target)`. Accumulate loss into a block-atomic scalar.

### G.9 `adamw_step(param, grad, m, v, lr, beta1, beta2, eps, weight_decay, bias1, bias2, n)`
- In-place update of `param`, `m`, `v`. One thread per parameter.
- Grid: `ceil(n / 256)`, Block: `256`.
- `m = beta1 * m + (1 - beta1) * grad`
- `v = beta2 * v + (1 - beta2) * grad * grad`
- `m_hat = m / bias1`, `v_hat = v / bias2`
- `param = param - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * param)`

---

## Appendix H — Checkpoint File Format

```
offset  size   field
------  -----  -------------------------------------------
0       4      magic: ASCII "TWTL"
4       4      version: u32, little-endian, currently 1
8       4      num_params: u32, little-endian
12      ...    repeated num_params times:
                 4    name_len: u32
                 N    name: [name_len]u8, UTF-8, no NUL
                 1    rank: u8 (1 to 4)
                 16   dims: [4]u32 (unused dims are 0)
                 4    data_len_bytes: u32 (= prod(dims) * 4)
                 M    data: f32 little-endian, row-major
```

`load` verifies `magic == "TWTL"` and `version == 1`. Mismatch returns `error.IoError` with a message including the observed values.

---

**End of plan.md.** The receiving agent begins at Section 14 (First Actions) and proceeds sequentially through Stages 1 through 9.
