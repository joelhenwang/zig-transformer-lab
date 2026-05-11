# 00 — Overview

Welcome to **zig-transformer-lab**, a tiny educational deep-learning runtime built
from first principles in Zig 0.16.0.

## 1. Mission

Build a small, heavily commented, pedagogical personal library that trains a
**tiny one-block, one-head, word-level transformer from scratch** on CPU first,
then accelerates it with CUDA (cuBLAS plus a minimal set of custom `.cu` kernels).

The package is *learning infrastructure*: every line is written to teach a
beginner how PyTorch-like systems work internally.

### Who this is for

- Knows Python well, complete beginner in Zig
- Near-zero ML/DL knowledge; studied some math/stats in university
- Zero GPU/kernel programming background
- Wants explanations that assume learning-from-near-zero, code that is
  heavily commented, and extensive design notes explaining *why* each
  implementation works

### What you will understand at the end

1. What a tensor *actually is* in bytes, pointers, and metadata
2. How autograd records the forward pass and replays it in reverse
3. Why matmul dominates transformer compute
4. How a CUDA kernel is launched from a host program
5. Why row-major and column-major are different, and what that means for cuBLAS
6. How a training loop works at the level of individual array operations
7. What PyTorch hides behind `loss.backward()` and `optimizer.step()`

### What this is NOT

- Not a production ML framework
- No mixed precision (no f16/bf16)
- No distributed or multi-GPU training
- No ONNX, safetensors, or HuggingFace interop
- No pure-Zig GPU kernel code (kernels are CUDA C compiled with nvcc)
- No lazy tensors, no graph compiler, no MLIR, no XLA

## 2. How to read these docs

Recommended order:

1. **00_overview.md** (this file) — project scope, decisions, layout
2. **01_zig_primer.md** — Zig 0.16.0 concepts used throughout the library
3. **02_tensors.md** — row-major layout, strides, broadcasting, softmax stability
4. **02b_from_tensors_to_training.md** — bridges Stage 2 ops to ML/DL concepts, forward-pass trace, PyTorch equivalents
5. **02c_tensor_invariants.md** — the five structural invariants every Tensor must satisfy (Stage 6.5 / PR-γ)
6. **02d_storage_and_views.md** — storage/view split, the Storage union, why []f32 can't represent CUDA memory (Stage 6.5 / PR-δ)
7. **03_autograd.md** — tape-based reverse-mode autograd, matmul backward, fused CE
8. **03b_from_autograd_to_training.md** — bridges Stage 3 autograd to ML/DL, backward trace through transformer, residual gradient flow, optimizer preview
9. **03c_saved_tensors.md** — how the tape owns its own backward data; why keepAlive died (Stage 6.5 / PR-ε, flagship chapter)
10. **04_nn.md** — Module protocol, parameter iteration, initialization strategies
11. **04b_from_nn_to_training.md** — bridges Stage 4 nn/optim to ML/DL, full training step trace, optimizer math
12. **05_transformer_math.md** — full shape trace for one transformer block
13. **05b_from_tokenizer_to_training.md** — bridges Stages 5-6 data pipeline to ML/DL concepts, shift-by-1 supervision, end-to-end integration checklist (Stage 9)
14. **06_tokenizer_data.md** — word-level tokenizer, windowing, batching
15. **07_cpu_training.md** — end-to-end CPU training loop, generation, three silent bugs we fixed
16. **07b_learning_guide_training.md** — bridges Stage 6 code to ML/DL concepts
17. **07c_optimizer_state.md** — ParamId-keyed optimizer state (Stage 6.5 / PR-ζ)
18. **07d_checkpoint_format.md** — the ZTLC binary format (Stages 6.5 + 8)
19. **08_backends_cuda.md** — cuBLAS, PTX loading, kernel walk-throughs, row-major worksheet (Stage 7)
20. **08b_from_cuda_to_training.md** — bridges Stages 7-8 CUDA backend to what it means to train on GPU, per-step device residency walk, Trainer CUDA path (Stage 9)
21. **09_debugging.md** — NaN hunting, compute-sanitizer, Nsight Compute, shape-assert workflow
22. **10_pytorch_parallels.md** — mapping every concept back to PyTorch, bridging exercise (Stage 9)

Each chapter ends with a "Common mistakes" section. Read it before you hit the bug.

## 3. Ecosystem map

The following projects were studied as design references. **No code is copied.**
Everything is re-derived cleanly in Zig.

| Project | URL | Role | License | Notes |
|---------|-----|------|---------|-------|
| `Marco-Christiani/zigrad` | github.com/Marco-Christiani/zigrad | Design reference for autograd, graph tracing, parameter iteration | LGPL-3.0 | Under active rewrite. **Do not copy code** (license incompatibility). |
| `CogitatorTech/zigformer` | github.com/CogitatorTech/zigformer | Closest educational analogue in pure Zig. Tokenizer and training-loop shape reference. | MIT | Zig 0.15.2. Read, do not port verbatim. |
| `karpathy/nanoGPT`, `karpathy/llm.c` | github.com/karpathy/nanoGPT, github.com/karpathy/llm.c | Canonical minimal-transformer references. Our math must match these. | MIT | |
| Rust `cudarc` | github.com/coreylowman/cudarc | Three-layer sys/result/safe architecture idea for CUDA wrapper. | MIT/Apache-2.0 | Re-implement cleanly in Zig. |
| `aiurion/zigCUDA` | github.com/aiurion/zigCUDA | Existing pure-Zig bindings to CUDA Driver API. Zig 0.16.0. | MIT | **Not used as dependency** (D1). Source may be consulted for dlopen/dlsym patterns. |
| `akhildevelops/cudaz` | github.com/akhildevelops/cudaz | Most mature Zig CUDA wrapper; driver + NVRTC. | MIT-like | **Not used as dependency** (D1). |
| `coderonion/zcuda` | github.com/coderonion/zcuda | Broad binding surface (cuBLAS, cuBLASLt, cuRAND, cuDNN, etc). | MIT | **Not used as dependency** (D1). Too large for scope. |
| `gwenzek/cudaz` | github.com/gwenzek/cudaz | Toy wrapper using Zig's LLVM PTX backend. | Apache-2.0 | **Incompatible with policy** (no pure-Zig GPU kernels). |
| `zml/zml` | github.com/zml/zml | Production inference stack (XLA/MLIR/Bazel). | Apache-2.0 | Off-limits as reference for this pedagogical scope. |
| `zillama/ggml-zig` | github.com/zillama/ggml-zig | Pure-Zig ggml port. | varies | Not used; ggml's quantization is off-topic for f32 training. |

### Classification

- **Use as design reference (study, then rewrite):** zigrad (architecture only), zigformer, nanoGPT, llm.c, cudarc
- **Use as optional dependency:** None. D1 locks us to a hand-written wrapper.
- **May adapt small mechanical patterns (with file-header attribution):** cuBLAS handle lifecycle, Dim3/LaunchConfig struct shape, tokenization helpers
- **Avoid:** zml, gwenzek/cudaz, ggml-zig

## 4. Locked decisions (D1–D14)

These are final. Do not revisit without explicit user approval.

| # | Decision | Value |
|---|----------|-------|
| D1 | CUDA binding strategy | Hand-written minimal wrapper via dlopen/dlsym. No external Zig-CUDA dependency. |
| D2 | CPU BLAS | None. CPU-naive f32 matmul (correctness oracle) then directly to CUDA with cuBLAS. |
| D3 | CUDA kernel build | Offline nvcc -arch=sm_89 -ptx at Zig build time. .ptx under zig-out/ptx/. Loaded at runtime with cuModuleLoadData. No NVRTC, no JIT. |
| D4 | Python oracle | tools/oracle.py using NumPy plus PyTorch. |
| D5 | Model shape | Stages 2–7 hard-coded 1 block / 1 head. Stage 8 (Milestone 2) added `TransformerConfig.n_layer: u8`, `n_head: u8`, `dropout: f32` with defaults `(1, 1, 0.0)` preserving the Stage 2–7 behaviour. Milestones 3–4 wire them through to `TinyWordTransformer.blocks: []TransformerBlock` and multi-head `CausalSelfAttention`. |
| D6 | Documentation | Book-level docs/0X_*.md chapter per stage (500–1500 lines each) plus heavy inline comments. Raw markdown only. ASCII art for diagrams. |
| D7 | Environment | Linux; system Zig 0.16.0; system CUDA Toolkit; RTX 4060 Ti (sm_89). |
| D8 | Training corpora | data/tiny.txt (~5 KB crafted) and data/tinyshakespeare.txt (~1 MB). |
| D9 | Float type | f32 only throughout. No f16/bf16/mixed precision. |
| D10 | License | MIT only. No NOTICE file. Third-party inspirations credited inline. |
| D11 | Zig version | Pinned to 0.16.0 (exact). Recorded in build.zig.zon. |
| D12 | Autograd style | Tape-based reverse-mode, dynamic graph, not generic. Covers only operations needed for this transformer. |
| D13 | Attention variant | Pre-norm causal self-attention. Post-norm tradeoff documented. |
| D14 | Randomness | std.Random.Xoshiro256 seeded from a config-provided u64; every random operation takes an explicit seed. Runs must be deterministic given the same seed. |

### Additional locked policies (P1–P4)

- **P1. Artifacts.** docs/pre_flight.md, zig-out/, .zig-cache/, loss CSVs, checkpoints, and Python venvs are all gitignored.
- **P2. Agent brief file.** AGENTS.md at repository root, committed in Stage 1.
- **P3. Stage boundaries.** Each stage ends with (a) passing tests, (b) a runnable example, (c) the matching docs chapter committed.
- **P4. Commits.** One commit per stage with message `stage(N): <summary>`. Acceptance-criterion outputs pasted into the commit body.

## 5. Repository layout

```
zig-transformer-lab/
|-- build.zig                    # Build system (Zig 0.16.0 API)
|-- build.zig.zon                # Package manifest, Zig version pin
|-- README.md                    # Quick-start guide
|-- LICENSE                       # MIT
|-- AGENTS.md                     # Agent contract
|-- .gitignore
|-- docs/                         # Book chapters (one per stage)
|   |-- 00_overview.md            #   This file
|   |-- 01_zig_primer.md          #   Zig concepts used in the library
|   |-- 02_tensors.md             #   Tensors, strides, broadcasting
|   |-- 02b_from_tensors_to_training.md #   Stage 2 ops → ML/DL bridge, forward-pass trace
|   |-- 03_autograd.md            #   Tape-based autograd
|   |-- 03b_from_autograd_to_training.md #   Stage 3 autograd → ML/DL bridge, backward-pass trace
|   |-- 04_nn.md                  #   Module protocol, layers
|   |-- 04b_from_nn_to_training.md #   Stage 4 nn/optim → ML/DL bridge
|   |-- 05_transformer_math.md   #   Full shape trace
|   |-- 06_tokenizer_data.md     #   Tokenizer, dataset, batching
|   |-- 07_cpu_training.md       #   Training loop, generation
|   |-- 08_backends_cuda.md      #   CUDA backend, cuBLAS, kernels
|   |-- 09_debugging.md          #   Debug tools, compute-sanitizer
|   `-- 10_pytorch_parallels.md  #   PyTorch comparison
|-- src/
|   |-- root.zig                  # Package entry, re-exports
|   |-- core/                     # Foundational types
|   |   |-- errors.zig            #   Library-wide error set
|   |   |-- dtype.zig             #   f32 type tag
|   |   |-- device.zig            #   CPU/CUDA device enum
|   |   `-- rng.zig               #   Xoshiro256 seeded RNG
|   |-- tensor/                   # Tensor struct and operations
|   |   |-- shape.zig             #   Shape, Strides, rank up to 4
|   |   |-- tensor.zig            #   Tensor struct (data, shape, strides, device, owned)
|   |   |-- print.zig             #   debugSummary
|   |   `-- ops/                  #   All tensor operations
|   |       |-- create.zig        #   zeros, ones, full, randn, randu, arange
|   |       |-- elementwise.zig   #   add, sub, mul, div + broadcasting
|   |       |-- reduce.zig        #   sum, mean, max with axis
|   |       |-- matmul.zig        #   naive matmul, transpose, reshape
|   |       |-- unary.zig         #   exp, log, neg, relu, gelu
|   |       |-- softmax.zig       #   numerically stable softmax, log_softmax
|   |       `-- loss.zig          #   cross_entropy
|   |-- autograd/                 # Tape-based reverse-mode autograd
|   |   |-- node.zig              #   Node, NodeId, OpKind
|   |   |-- tape.zig              #   Tape (ArrayList of nodes)
|   |   |-- backward.zig          #   Iterative backward traversal
|   |   `-- gradcheck.zig         #   Central-difference gradient checker
|   |-- nn/                       # Neural network layers
|   |   |-- module.zig            #   Module protocol
|   |   |-- linear.zig            #   Linear layer, Kaiming init
|   |   |-- embedding.zig         #   Embedding lookup
|   |   |-- layernorm.zig         #   LayerNorm (Welford)
|   |   |-- activations.zig      #   GELU
|   |   |-- attention.zig        #   CausalSelfAttention (1 head)
|   |   |-- mlp.zig               #   Linear -> GELU -> Linear
|   |   |-- block.zig             #   Pre-norm transformer block
|   |   `-- model.zig             #   TinyWordTransformer + save/load
|   |-- optim/                    # Optimizers
|   |   |-- optimizer.zig         #   Optimizer vtable
|   |   |-- sgd.zig               #   SGD (momentum, weight decay)
|   |   `-- adamw.zig             #   AdamW (bias-corrected, decoupled)
|   |-- tokenizer/                # Word-level tokenizer
|   |   |-- vocab.zig             #   Vocab struct, serialize/deserialize
|   |   `-- word.zig              #   Encode/decode
|   |-- data/                     # Dataset pipeline
|   |   |-- dataset.zig           #   File -> token stream
|   |   |-- windowing.zig         #   (input, target) pairs
|   |   `-- batcher.zig           #   Deterministic shuffle + batching
|   |-- backend/                  # Backend abstraction
|   |   |-- backend.zig           #   VTable interface
|   |   |-- cpu_naive/
|   |   |   `-- dispatch.zig      #   Wraps Stage 2 CPU ops
|   |   `-- cuda/
|   |       |-- bindings.zig      #   dlopen/dlsym for libcuda, libcudart, libcublas
|   |       |-- context.zig       #   CudaContext (device, context, stream, cuBLAS handle)
|   |       |-- mem.zig           #   DeviceBuffer (HtoD, DtoH, deinit)
|   |       |-- module.zig        #   PTX module loading
|   |       |-- gemm.zig          #   Row-major cuBLAS GEMM wrapper
|   |       |-- dispatch.zig      #   Unified CUDA dispatcher
|   |       `-- kernels/          #   .cu kernel files
|   |           |-- elementwise.cu
|   |           |-- softmax.cu
|   |           |-- layernorm.cu
|   |           |-- gelu.cu
|   |           |-- embedding.cu
|   |           |-- causal_mask.cu
|   |           |-- ce_loss.cu
|   |           `-- adamw.cu
|   |-- debug/                    # Debug utilities
|   |   |-- assert_shape.zig      #   Shape assertions with rich errors
|   |   |-- finite.zig            #   NaN/Inf detection
|   |   `-- compare.zig           #   CPU/GPU diff helper
|   `-- lab/
|       `-- train.zig             #   Top-level training loop
|-- examples/                     # Runnable examples (one per stage)
|   |-- 01_tensor_playground.zig
|   |-- 02_autograd_scalar.zig
|   |-- 03_autograd_tensor.zig
|   |-- 04_overfit_one_batch.zig
|   |-- 05_train_tiny.zig
|   |-- 06_train_shakespeare.zig
|   |-- 07_generate.zig
|   `-- 08_cuda_vs_cpu.zig
|-- tests/
|   |-- unit_all.zig              #   Imports every src/**/*.zig
|   |-- integration_cpu.zig
|   `-- integration_cuda.zig      #   Only compiled when -Dcuda=true
|-- tools/
|   |-- requirements.txt          #   numpy, torch
|   |-- oracle.py                 #   Python oracle for cross-validation
|   |-- compare_outputs.py        #   Binary diff tool
|   `-- plot_loss.py              #   Loss curve plotter
`-- data/
    |-- tiny.txt                  #   ~5 KB crafted corpus
    `-- tinyshakespeare.txt       #   ~1 MB Shakespeare
```

## 6. Quickstart

```bash
# Build the library
zig build

# Run all CPU tests
zig build test

# Run a specific example
zig build run-example -Dexample=01_tensor_playground

# Run with CUDA
zig build test -Dcuda=true
zig build run-example -Dexample=06_train_shakespeare -Dcuda=true
```

## 7. Glossary

- **tensor** — a multidimensional array of f32 values, plus metadata (shape,
  strides, device). The fundamental data structure.
- **shape** — the size of each dimension, e.g. `(2, 3)` means 2 rows, 3 columns.
- **strides** — how many f32 elements to skip to move one step along each
  dimension. In row-major, the last dimension has stride 1.
- **row-major** — memory layout where the rightmost index varies fastest.
  `index(i, j) = i * cols + j`. This is our convention throughout.
- **view** — a tensor that shares another tensor's data buffer. Does not own
  the memory. Created by operations like `transpose` or `reshape` when possible.
- **owned** — a tensor that allocated its own data buffer and must free it
  in `deinit`.
- **grad** — the gradient of a loss with respect to a tensor. Stored as a
  tensor of the same shape, accumulated via `+=`.
- **tape** — a record of operations performed during the forward pass. Each
  operation appends a node. Backward replays the tape in reverse.
- **node** — a single entry on the tape: operation kind, parent node IDs, saved
  data needed for backward, and a function pointer to the backward computation.
- **kernel** — a function that runs on the GPU. Each kernel is compiled from
  CUDA C to PTX and loaded at runtime.
- **PTX** — NVIDIA Parallel Thread Execution: an intermediate representation
  for GPU code. Compiled offline by nvcc from .cu files.
- **cuBLAS** — NVIDIA's CUDA Basic Linear Algebra Subprograms library. Provides
  highly optimized matrix multiplication (GEMM) and other BLAS operations.
- **GEMM** — General Matrix Multiply: `C = alpha * A @ B + beta * C`. The
  single most important CUDA operation for transformers.
- **sm_89** — Compute capability 8.9, corresponding to RTX 4060 Ti (Ada Lovelace).
- **dlopen/dlsym** — POSIX functions for loading shared libraries at runtime
  and looking up symbol addresses. Used to load libcuda.so, libcudart.so,
  libcublas.so without linking them at compile time.

## 8. Where to go next

- New to Zig? Start with **docs/01_zig_primer.md**.
- Comfortable with Zig but new to tensors? Start with **docs/02_tensors.md**.
- Know tensors but want the ML/DL connection? Read **docs/02b_from_tensors_to_training.md**.
- Know autograd mechanics but want the transformer connection? Read **docs/03b_from_autograd_to_training.md**.
- Want the full picture? Read the chapters in order, 01 through 10.
