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

### Stage 2 — CPU Tensor Foundation ✅ COMPLETE
**Commit:** `001c74e stage(2): cpu tensor foundation` — 7 files, 3082 insertions

### Stage 3 — Tape-based Autograd ✅ COMPLETE
**Commit:** `f8405e3 stage(3): tape-based autograd` — 15 files, 5112 insertions

### Stage 4 — NN Layers + Optimizers ✅ COMPLETE
**Commit:** `b02801b stage(4): nn layers and optimizers` — 18 files, 3638 insertions

### Stage 5 — Tokenizer + Data Pipeline ✅ COMPLETE
**Commit:** `d286c8a stage(5): word-level tokenizer and dataset`

### Stage 6 — End-to-end CPU Training ✅ COMPLETE
**Commit:** `015da3c stage(6): end-to-end cpu training` — 10 files, 1530 insertions
- `src/lab/train.zig` — Trainer, generate(), gradient clipping, grad norm logging
- `examples/06_train_shakespeare.zig`, `examples/07_generate.zig`
- `docs/07_cpu_training.md` — 492 lines
- Bug fixes: backwardCrossEntropy @round, reshapeTracked, NamedParam, dangling pointers, beta2=0.999
- All 215+ tests pass, 0 leaks

### Stage 6.5 — CPU Hardening and Backend Seam ✅ COMPLETE
**Commits:** `f9c1d3b` (refactor), `28e73e1` (teaching docs), `97b0aaa` (oracle), `3331801` (oracle expansion)
- Seven PRs (α–η): honesty pass, strided ops fix, invariants, storage union, operation-owned SavedTensor, ParamId optimizer state, strict checkpoint validation.
- Five teaching chapters: `docs/02c`, `02d`, `03c`, `07c`, `07d`.
- PyTorch oracle: 14 cases under `tests/fixtures/` exercising every CPU op family up to full-model forward parity.
- 263 tests pass + 14 oracle parity tests pass, on Windows and Linux.

### Stage 7-setup — Remote RTX workflow ✅ COMPLETE
**Commit:** `1e3b540 stage(7-setup): remote runner scripts and workflow docs`
- `run_remote_example.sh` + `sync_remote_example.sh` wrappers for the RTX box at `joelwang-rtx@192.168.1.197`.
- `.gitattributes` enforces LF on `*.sh`/`*.py` so rsync doesn't break remote execution.
- Toolchain verified: Ubuntu 24.04, RTX 4060 Ti 16 GB, CUDA 13.2, Zig 0.16.0. Both `zig build test` and `zig build test-oracle` pass on remote.

### Stage 7 — CUDA Backend ✅ COMPLETE
**Commit range:** `07bd274` .. `584160b` + docs `f5ecf37`. 14 PRs (α–ξ) + two
end-to-end examples landed. Final test count: 267 CPU + 73 CUDA on RTX 4060
Ti, compute-sanitizer memory-clean. Speedup measured at the Shakespeare
config: **30.59×** (CPU 143.7 ms/step, CUDA 4.7 ms/step). See
`docs/stage7_plan.md` (playbook) and `docs/stage7_endgame_plan.md`
(session-by-session landing log).

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

### To start Stage 8 (debugging discipline + N-block refactor):

**First action:** open `docs/stage8_plan.md`. It is a self-contained
8-milestone plan with per-commit file lists, acceptance criteria,
and the session-by-session execution schedule. Execute Milestone 1
first (four small commits under `src/debug/`).

Secondary reading: `docs/stage7_endgame_plan.md` for the style of
milestone cards and the cadence that worked well in Stage 7.

### Stage 7 reference (already COMPLETE):

Tag `stage-7-complete` on origin. 267 CPU + 73 CUDA tests,
compute-sanitizer clean, 30.59× speedup on Shakespeare config.
Reference docs: `docs/stage7_plan.md` (14-PR playbook as executed)
and `docs/stage7_endgame_plan.md` (session landing log).

### Stage 9:

Follow `plan.md` §873 — documentation finalization. Starts after
Stage 8 ships so the PyTorch parallels chapter can cover the real
multi-head attention code.

---

## 8. Repository Layout (Current)

```
zig-transformer-lab/
├── build.zig              ✅ Full spec with CUDA options
├── build.zig.zon          ✅ Zig 0.16.0 pinned
├── AGENTS.md              ✅ Agent brief (Stages 1-6 done, Stage 7 next)
├── README.md              ✅
├── LICENSE                ✅ MIT, Joel Wang
├── .gitignore             ✅
├── plan.md                ✅ 1300-line implementation plan
├── skills/                ✅ modern-zig-0-16-tutor
│   └── modern-zig-0-16-tutor/
│       ├── SKILL.md
│       ├── references/    (19 files: build, io, allocators, containers, C interop, CUDA, ML, testing, etc.)
│       ├── recipes/      (12 files: allocators, arrays, build-system, testing, numerical-code, etc.)
│       ├── templates/    (7 files: basic-main, build-exe, build-library, tensor2d-skeleton, etc.)
│       ├── examples/     (6 items: runnable snippets)
│       ├── scripts/      (4 files: stale pattern detector, version checker, validator)
│       └── tests/        (2 files: expected stale/modern patterns)
├── docs/
│   ├── 00_overview.md     ✅ 291 lines
│   ├── pre_flight.md      ✅ Local only (gitignored)
│   ├── 01_zig_primer.md   ✅ 803 lines (Stage 2)
│   ├── 02_tensors.md       ✅ 973 lines (Stage 2)
│   ├── 02b_from_tensors_to_training.md ✅ 861 lines (Stage 2)
│   ├── 03_autograd.md     ✅ (Stage 3)
│   ├── 03b_from_autograd_to_training.md ✅ (Stage 3)
│   ├── 04_nn.md           ✅ 858 lines (Stage 4)
│   ├── 04b_from_nn_to_training.md ✅ 1072 lines (Stage 4/5)
│   ├── 05_transformer_math.md ✅ (Stage 4)
│   ├── 06_tokenizer_data.md ✅ (Stage 5)
│   └── 07_cpu_training.md ✅ 492 lines (Stage 6)
├── src/
│   ├── root.zig           ✅ Stages 1-6 wired
│   ├── core/              ✅ errors, dtype, device, rng
│   ├── tensor/            ✅ shape, tensor, print
│   │   └── ops/           ✅ create, elementwise, reduce, matmul, unary, softmax, loss, shape_ops
│   ├── autograd/          ✅ node, tape, backward, gradcheck (Stage 3)
│   ├── nn/                ✅ module, linear, embedding, layernorm, activations, attention, mlp, block, model (Stage 4)
│   ├── optim/             ✅ optimizer, sgd, adamw (Stage 4)
│   ├── tokenizer/         ✅ vocab, word (Stage 5)
│   ├── data/              ✅ dataset, windowing, batcher (Stage 5)
│   ├── backend/           🔲 Stage 7 (backend.zig, cpu_naive/, cuda/)
│   ├── debug/             🔲 Stage 8
│   └── lab/               ✅ train.zig (Stage 6)
├── examples/
│   ├── 01_tensor_playground.zig ✅ (Stage 2)
│   ├── 02_autograd_scalar.zig  ✅ (Stage 3)
│   ├── 03_autograd_tensor.zig  ✅ (Stage 3)
│   ├── 04_overfit_one_batch.zig ✅ (Stage 4)
│   ├── 05_train_tiny.zig       ✅ (Stage 6 fixes)
│   ├── 06_train_shakespeare.zig ✅ (Stage 6)
│   └── 07_generate.zig         ✅ (Stage 6)
├── tests/
│   └── unit_all.zig       ✅ Dead code — see AGENTS.md
├── tools/
│   ├── requirements.txt   ✅
│   └── .venv/             ✅ numpy+torch (gitignored)
└── data/
    ├── tiny.txt           ✅ ~5 KB crafted corpus (Stage 5)
    └── tinyshakespeare.txt ✅ ~1 MB Shakespeare (Stage 5)
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
8. **Always use `reshapeTracked()` in training loop.** Untracked `reshape()` creates a VIEW sharing tape_node → silent gradient shape mismatch. (Stage 6 bug)
9. **Always use `@round` before `@intFromFloat` on indices.** `@intFromFloat` truncates towards zero. (Stage 6 bug)
10. **Don't store HashMap-containing structs (AdamW) as fields in structs returned by value.** Internal self-referential pointers corrupt on copy. Create locally in `train()`. (Stage 6 bug)
11. **Don't collect params in `init()` if model is copied into struct.** Pointers to local's fields dangle after `return Trainer{ .model = model }`. (Stage 6 bug)
12. **`collectNamedParams` needs pointer self (`*TinyWordTransformer`)**. By-value creates dangling pointers to stack-local copy. (Stage 6 bug)
13. **Gradient clipping is essential** for training stability. Default `grad_clip_norm=5.0`. (Stage 6)
14. **`beta2=0.999`** (not 0.95) is the correct default for AdamW. 0.95 causes instability. (Stage 6)

**Full 35-entry gotcha table:** See `AGENTS.md` "Zig 0.16.0 compilation gotchas" section.

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

1. `cd` into the repo (Windows: `C:\Users\...\zig-transformer-lab`; Linux: `/home/joelwang-rtx/Desktop/ai_lab/zig-transformer-lab`)
2. Read `AGENTS.md` (agent contract, progress, gotchas, current gate)
3. Read this file (`SESSION_GUIDE.md`) for current state
4. **If Stage 8:** read `docs/stage8_plan.md` in full before touching code. Milestones 1–8 with commit plans.
5. Check `git log --oneline -15` for completed stages (expect stages 1–7 done + tag `stage-7-complete`).
6. Run `zig build test` to verify baseline (267+ tests after Stage 7; 282+ after Stage 8 M1).
7. Run `zig build test-oracle` to verify oracle fixtures (14+ tests; 15+ after Stage 8 M5).
8. For CUDA work: verify remote access with `bash ./run_remote_example.sh "echo ok && zig version && nvcc --version | tail -1"`.
9. Load `skills/modern-zig-0-16-tutor/SKILL.md` for any Zig API questions.
10. Pick up at the next unchecked milestone in `docs/stage8_plan.md`.

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
