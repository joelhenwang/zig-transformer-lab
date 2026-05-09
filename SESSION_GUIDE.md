# zig-transformer-lab — Session Continuation Guide

> **Purpose:** Any AI agent starting a fresh session on this repository should
> read this file and AGENTS.md to become productive immediately, with zero
> loss of context from previous sessions.

---

## 1. Project Identity

**zig-transformer-lab** is a pedagogical Zig 0.16.0 library that trains a tiny
1-block, 1-head, word-level transformer from scratch on CPU, then accelerates
it with CUDA. The project follows a hybrid approach: implementation-agent mode
(plan.md's 9-stage structure) with pedagogical code (heavy comments, teach-then-implement flow).

- **Plan:** `plan.md` at repo root (1300 lines, self-contained, all decisions locked)
- **Agent contract:** `AGENTS.md`
- **Zig 0.16.0 skill:** `skills/modern-zig-0-16-tutor/SKILL.md` (+ 19 reference files, 12 recipes, 7 templates)

---

## 2. Environment

| Item | Value |
|------|-------|
| Zig 0.16.0 | `~/.local/bin/zig` (symlinked from `~/Downloads/zig/zig`), added to `~/.bashrc` PATH |
| CUDA 13.2 | `nvcc` at `/usr/local/cuda/bin/nvcc` |
| GPU | RTX 4060 Ti 16GB, sm_89, driver 595.58 |
| CUDA libs | `/usr/lib/x86_64-linux-gnu/libcuda.so.1`, `/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so`, `libcublas.so` |
| Python oracle | `tools/.venv/bin/python3` — numpy 2.4.4, torch 2.11.0+cu130, CUDA available |
| `CUDA_HOME` | Not set (autodetect from `/usr/local/cuda`) |

**Always run with:** `export PATH="/home/joelwang-rtx/.local/bin:$PATH"` or use the full path.

---

## 3. Implementation Progress

### Stage 1 — Project Scaffold ✅ COMPLETE
**Commit:** `153095b stage(1): scaffold, overview, agent brief`

Files created:
- `build.zig`, `build.zig.zon` — full spec with `-Dcuda`, `-Dcuda_arch`, `-Dcuda_home`, `-Dexample`, `-Dseed`
- `src/root.zig` — re-export scaffolding (all stages pre-planned as comments)
- `src/core/errors.zig` — `LabError` error set
- `tests/unit_all.zig` — module-based test aggregator
- `AGENTS.md` — hybrid mode agent brief
- `README.md`, `LICENSE` (MIT, Joel Wang), `.gitignore`, `tools/requirements.txt`
- `docs/00_overview.md` — 283 lines: mission, ecosystem, locked decisions D1-D14, repo layout, glossary
- `plan.md` — copied from user's Downloads

Acceptance: ✅ `zig build` succeeds, `zig build test` passes

### Stage 2 — CPU Tensor Foundation ✅ COMPLETE (uncommitted)

**Files completed (all compile and pass tests):**

| File | Contents | Tests |
|------|----------|-------|
| `src/core/dtype.zig` | DType enum (.f32 only), sizeInBytes(), label() | 2 |
| `src/core/device.zig` | Device enum (.cpu, .cuda), isCuda(), label() | 3 |
| `src/core/rng.zig` | Rng wrapping Xoshiro256, floatF32(), normalF32() (Box-Muller) | 5 |
| `src/tensor/shape.zig` | Shape/Strides structs, init1D-4D, computeStrides, totalElements, isContiguous, equals, toString, broadcastShapes, squeeze | ~24 |
| `src/tensor/tensor.zig` | Tensor struct (data, shape, strides, dtype, device, owned, autograd fields), init/deinit, at/atPtr, flatIndex, isContiguous, view, reshape, transpose2d, copyTo, fill | ~15 |
| `src/tensor/print.zig` | debugSummary (shape/strides/stats in 1 pass), printValues | ~4 |
| `src/tensor/ops/create.zig` | zeros, ones, full, randn, randu, arange, fromSlice | ~9 |
| `src/tensor/ops/elementwise.zig` | add, sub, mul, div (broadcast), addScalar, mulScalar, addInPlace, neg | ~13 |
| `src/tensor/ops/reduce.zig` | sum, mean, max (with axis), sumAll | ~7 |
| `src/tensor/ops/matmul.zig` | matmul (ikj cache-friendly), matmulBatch, transpose2d | ~7 |
| `src/tensor/ops/unary.zig` | exp, log, neg, relu, geluExact | ~5 |
| `src/tensor/ops/softmax.zig` | numerically stable softmax, logSoftmax (last axis) | ~5 |
| `src/tensor/ops/loss.zig` | crossEntropy (fused log_softmax + NLL) | ~4 |

**Total: ~103 tests passing**

**Examples:**
- `examples/01_tensor_playground.zig` — 13-section runnable example covering all Stage 2 ops

**Documentation:**
- `docs/01_zig_primer.md` — 803 lines: Zig 0.16 concepts, 15-entry gotcha table, 10-entry common mistakes
- `docs/02_tensors.md` — 973 lines: row-major strides, broadcasting, softmax stability, ikj matmul, views
- `docs/02b_from_tensors_to_training.md` — 861 lines: bridges Stage 2 ops to ML/DL, forward-pass trace, PyTorch equivalents

**Acceptance verified:** `zig build test` green, example runs, all docs 500+ lines

**Remaining:** Commit as `stage(2): cpu tensor foundation`

### Stage 3 — Tape-based Autograd 🔲 NOT STARTED

### Stage 4 — nn Module and Optimizers 🔲 NOT STARTED

### Stage 5 — Tokenizer and Dataset Pipeline 🔲 NOT STARTED

### Stage 6 — End-to-end CPU Training 🔲 NOT STARTED

### Stage 7 — CUDA Backend 🔲 NOT STARTED

### Stage 8 — Debugging and N-block Refactor 🔲 NOT STARTED

### Stage 9 — Documentation Finalization 🔲 NOT STARTED

---

## 4. Key Zig 0.16.0 API Patterns (Used in This Project)

These are the patterns that came up during implementation and caused compilation errors. Memorize them:

| Pattern | Correct (0.16.0) | Wrong (pre-0.16) |
|---------|-------------------|-------------------|
| Main signature | `pub fn main(init: std.process.Init) !void` | `pub fn main() !void` |
| Allocator from Init | `init.arena.allocator()` | Hand-rolled GPA |
| Shape init | `Shape.init2D(rows, cols)` | `Shape.init(&.{rows, cols})` |
| Build module | `b.createModule(.{...})` | `b.addExecutable({.root_source_file=...})` |
| Add import | `exe.root_module.addImport("x", m)` | `exe.addModule("x", m)` |
| Link lib | `exe.root_module.linkSystemLibrary("c", .{})` | `exe.linkSystemLibrary("c")` |
| ArrayList | `var list: ArrayList(T) = .empty; list.append(gpa, v)` | `ArrayList(T).init(a); list.append(v)` |
| Cast builtins | `@ptrFromInt`, `@intFromPtr`, `@intFromEnum` | `@intToPtr`, `@ptrToInt`, `@enumToInt` |
| C calling conv | `callconv(.c)` | `callconv(.C)` |
| std.fs.cwd() | `std.Io.Dir.cwd()` | `std.fs.cwd()` |
| Unmanaged API | Pass allocator to append/deinit | Managed style (no allocator) |
| `var` vs `const` | Zig enforces `const` for non-mutated locals | No enforcement |

**Full reference:** `skills/modern-zig-0-16-tutor/SKILL.md` has a 22-row stale-pattern table and compiler-error quick lookup.

---

## 5. Shape API Reference (This Project)

The `Shape` struct uses explicit rank-based constructors:

```zig
const s1 = Shape.init1D(10);          // rank 1: (10,)
const s2 = Shape.init2D(2, 3);        // rank 2: (2, 3)
const s3 = Shape.init3D(2, 3, 4);     // rank 3: (2, 3, 4)
const s4 = Shape.init4D(2, 3, 4, 5);  // rank 4: (2, 3, 4, 5)

// Access:
s2.ndim()           // 2
s2.dims[0]          // 2
s2.dims[1]          // 3

// Key free functions (import from shape.zig):
computeStrides(shape) -> Strides
totalElements(shape) -> usize
isContiguous(shape, strides) -> bool
equals(a, b) -> bool
broadcastShapes(a, b) -> !Shape
toString(shape, buf) -> []const u8
squeeze(shape, axis) -> Shape
```

---

## 6. Build Commands

```bash
# Must set PATH first:
export PATH="/home/joelwang-rtx/.local/bin:$PATH"

# Build (no-op if nothing changed):
zig build

# Run all tests:
zig build test

# Run a specific example (after creating it):
zig build run-example -Dexample=01_tensor_playground

# Format all source:
zig fmt src/

# Build with CUDA (Stage 7+):
zig build test -Dcuda=true

# Compile kernels only:
zig build kernels -Dcuda=true

# Docs line count check:
zig build docs
```

---

## 7. How to Resume Implementation

### Stage 2: Commit (if not yet done)

If Stage 2 is not yet committed, commit it now:

```bash
git add -A
git commit -m "stage(2): cpu tensor foundation"
```

Acceptance already verified:
- `zig build test` green with ~103 tests
- `01_tensor_playground.zig` runs with expected output
- All doc chapters 500+ lines

### To start Stage 3 (Autograd):

Read `plan.md` Section on Stage 3 carefully. Key files to create:
- `src/autograd/node.zig`, `tape.zig`, `backward.zig`, `gradcheck.zig`
- Extend `src/tensor/tensor.zig` (requires_grad, grad, tape_node fields already declared)
- `examples/02_autograd_scalar.zig`, `03_autograd_tensor.zig`
- `docs/03_autograd.md`

Build incrementally: implement simple backwards first (add/sub/mul/div), test each, then move to complex ones (matmul, softmax, CE). The tape-based design is in plan.md Stage 3.

### Stages 4-9:

Follow plan.md precisely. Each stage has: files to create, design notes, acceptance criteria, commit message format.

---

## 8. Repository Layout (Current)

```
zig-transformer-lab/
├── build.zig              ✅ Full spec with CUDA options
├── build.zig.zon          ✅ Zig 0.16.0 pinned
├── AGENTS.md              ✅ Hybrid mode agent brief
├── README.md              ✅
├── LICENSE                ✅ MIT, Joel Wang
├── .gitignore             ✅
├── plan.md                ✅ 1300-line implementation plan
├── skills/                ✅ modern-zig-0-16-tutor (copied from zig-transformer)
│   └── modern-zig-0-16-tutor/
│       ├── SKILL.md
│       ├── references/    (19 files: build, io, allocators, containers, C interop, CUDA, ML, testing, etc.)
│       ├── recipes/      (12 files: allocators, arrays, build-system, testing, numerical-code, etc.)
│       ├── templates/    (7 files: basic-main, build-exe, build-library, tensor2d-skeleton, etc.)
│       ├── examples/     (6 items: runnable snippets)
│       ├── scripts/      (4 files: stale pattern detector, version checker, validator)
│       └── tests/        (2 files: expected stale/modern patterns)
├── docs/
│   ├── 00_overview.md     ✅ 283 lines
│   ├── pre_flight.md      ✅ Local only (gitignored)
│   ├── 01_zig_primer.md   ✅ 803 lines (Stage 2)
│   ├── 02_tensors.md      ✅ 973 lines (Stage 2)
│   └── 02b_from_tensors_to_training.md ✅ 861 lines (Stage 2)
├── src/
│   ├── root.zig           ✅ Stage 1+2 wired
│   ├── core/              ✅ errors, dtype, device, rng
│   ├── tensor/             ✅ shape, tensor, print
│   │   └── ops/           ✅ create, elementwise, reduce, matmul, unary, softmax, loss
│   ├── autograd/          🔲 Stage 3
│   ├── nn/                🔲 Stage 4
│   ├── optim/             🔲 Stage 4
│   ├── tokenizer/         🔲 Stage 5
│   ├── data/              🔲 Stage 5
│   ├── backend/           🔲 Stage 7
│   ├── debug/             🔲 Stage 8
│   └── lab/               🔲 Stage 6
├── examples/
│   └── 01_tensor_playground.zig ✅ 13-section runnable example (Stage 2)
├── tests/
│   └── unit_all.zig       ✅ Module-based aggregator (dead code — see AGENTS.md)
├── tools/
│   ├── requirements.txt   ✅
│   └── .venv/             ✅ numpy+torch (gitignored)
└── data/                  🔲 Stage 5
```

---

## 9. Skill Usage Guide

The `skills/modern-zig-0-16-tutor/` directory is a comprehensive Zig 0.16.0 reference. When you encounter Zig API questions or compilation errors:

1. **Always read `SKILL.md` first** — it has the 22-row stale-pattern table and compiler-error quick lookup
2. **Load reference files by topic** (max 3 per turn to manage token budget):

| Task | Load |
|------|------|
| Build system errors | `references/08-build-system-0-16.md` |
| I/O (stdout, reader, writer) | `references/07-io-0-16.md` |
| Allocators, arena, leaks | `references/05-memory-allocators.md` |
| ArrayList / HashMap | `references/06-containers-0-16.md` |
| C/CUDA interop | `references/09-c-interop-0-16.md` + `references/17-zig-cuda-interop-notes.md` |
| Testing | `references/10-testing-debugging.md` |
| Code review | `references/15-code-review-checklist.md` |
| ML/tensor design | `references/16-zig-for-ml-runtime-projects.md` |

3. **Templates:** Use `templates/tensor2d-skeleton-0-16.zig` as a reference for tensor code style
4. **Stale pattern detector:** `scripts/grep_stale_patterns.py` can scan the codebase

---

## 10. Known Issues and Gotchas

1. **Shape API:** Uses `init1D/init2D/init3D/init4D`, NOT `init(&.{...})`. This caught multiple subagents.
2. **`std.fs.cwd()` doesn't exist** in Zig 0.16.0. Use `std.Io.Dir.cwd()` instead.
3. **`var` vs `const`:** Zig 0.16.0 strictly enforces that non-mutated locals must be `const`. The compiler will error if you use `var` and don't mutate.
4. **ArrayList is unmanaged:** `var list: ArrayList(T) = .empty; list.append(gpa, v); list.deinit(gpa)`. The old managed style `.init(a)` is stale.
5. **Tensor.transpose2d** returns a view (owned=false). Don't deinit it independently.
6. **broadcastShapes** returns `!Shape` (can fail with ShapeMismatch). Always use `try`.
7. **crossEntropy** expects targets stored as f32 values (rounds to int for indexing).
8. **The `seed` build option** is declared but not yet consumed by examples (placeholder for Stage 6).
9. **`std.Io.get()` does not exist in 0.16.0.** There is no global stdout writer. In examples, use `init.io.lockStderr(&buffer, null)` to get a locked stderr writer. The `Init` struct provides the `io: Io` field.

---

## 11. Locked Decisions Quick Reference

Full table in `docs/00_overview.md` or `plan.md` Section 2.

| Key | Decision |
|-----|----------|
| D1 | Hand-written dlopen/dlsym CUDA bindings. No external Zig-CUDA deps. |
| D2 | No CPU BLAS. Naive f32 matmul → directly to CUDA cuBLAS. |
| D3 | Offline nvcc -ptx, loaded via cuModuleLoadData. No NVRTC. |
| D5 | Hard-coded 1 block / 1 head until Stage 8. |
| D9 | f32 only. No f16/bf16. |
| D11 | Zig 0.16.0 exact. |
| D12 | Tape-based reverse-mode autograd (not generic). |
| D13 | Pre-norm causal self-attention. |

---

## 12. Session Start Checklist

When starting a new session, do these in order:

1. `cd /home/joelwang-rtx/Desktop/ai_lab/zig-transformer-lab`
2. Read `AGENTS.md` (agent contract)
3. Read this file (`SESSION_GUIDE.md`) for current state
4. Read `src/root.zig` to see what's wired vs commented
5. Check `git log --oneline` for completed stages
6. Run `export PATH="/home/joelwang-rtx/.local/bin:$PATH" && zig build test` to verify baseline
7. Consult `skills/modern-zig-0-16-tutor/SKILL.md` for any Zig API questions
8. Pick up where the last stage left off

---

## 13. Commit Convention

```
stage(N): <summary>

Acceptance criteria:
- <criterion 1>
- <criterion 2>

Files created/modified:
- <file list>
```

One commit per stage. Do NOT proceed to the next stage until acceptance criteria are met.
