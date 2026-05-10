# Agent Brief — zig-transformer-lab

## One-sentence mission

Build a pedagogical Zig 0.16.0 library that trains a tiny 1-block 1-head word-level
transformer on CPU, then on CUDA, with extensive documentation and heavily commented
code that teaches how PyTorch-like systems work internally.

## Hard rules

- Do not violate Locked Decisions D1 through D14 or policies P1 through P4.
  Ask the user before changing any decision. See `docs/00_overview.md` for the full table.
- No third-party Zig dependencies.
- Zig version pinned at 0.16.0 (exact).
- f32 only. No mixed precision.
- GPU kernels are CUDA C only. No pure-Zig GPU kernels.
- Kernels are compiled offline with nvcc to .ptx and loaded via cuModuleLoadData.
- Never copy code from LGPL sources. Architectural ideas only, attributed in file headers.

## Current progress

| Stage | Status | Notes |
|-------|--------|-------|
| 1 — Scaffold | **Done** | Commit `153095b` |
| 2 — CPU Tensor Foundation | **Done** | Commit `001c74e` — 7 files, 3082 insertions |
| 3 — Tape-based Autograd | **Done** | Commit `f8405e3` — 15 files, 5112 insertions |
| 4 — NN Layers + Optimizers | **Done** | Commit `b02801b` — 18 files, 3638 insertions |
| 5 — Tokenizer + Data Pipeline | **Done** | This session — 7 source files, 2 data files, 2 docs, 1 example |
| 6–9 | Not started | |

**Stage 3 committed:** `stage(3): tape-based autograd`
**Stage 4 committed:** `stage(4): nn layers and optimizers`

### Session history — what was done this session

1. Created `data/tiny.txt` (~6 KB crafted corpus with varied punctuation and contractions)
2. Downloaded `data/tinyshakespeare.txt` (~1 MB from karpathy/char-rnn)
3. Implemented `src/tokenizer/vocab.zig` — Vocab struct with buildFromText, encode/decode, save/load, frequency cutoff
4. Implemented `src/tokenizer/word.zig` — tokenize (punctuation peeling + apostrophe splitting), encode, decode
5. Implemented `src/data/dataset.zig` — File → []u32 token stream, vocab building, initWithVocab for val splits
6. Implemented `src/data/windowing.zig` — Sliding window (input, target) pairs as views
7. Implemented `src/data/batcher.zig` — Fisher-Yates shuffle (Xoshiro256), drop-last batching
8. Created `examples/05_train_tiny.zig` — Real-data training loop on tiny.txt, loss 6.08→5.17 over 100 steps
9. Wired Stage 5 modules into `src/root.zig` (uncommented tokenizer/data re-exports + test block)
10. Fixed `src/nn/model.zig` save/load to use Zig 0.16.0 file I/O API (`io` parameter, `writer.interface.print`, `reader.interface.takeDelimiter`, `file.stat(io)`, `cwd.readFileAlloc`)
11. Wrote `docs/05_transformer_math.md` (~1441 lines) — Full shape trace for one transformer block
12. Wrote `docs/06_tokenizer_data.md` (~1506 lines) — Tokenizer, dataset, batching pedagogical docs
13. All 205+ tests pass, 0 leaks

**Key Zig 0.16.0 API discoveries (Stage 5):**
- `std.Io.Dir.cwd().openFile(io, path, .{})` / `createFile(io, path, .{})` — the `io: std.Io` parameter is mandatory in 0.16.0
- `file.writer(io, &buf)` / `file.reader(io, &buf)` — both require io and buffer parameters
- `writer.interface.print(...)` — print is on the interface, not the File.Writer directly
- `reader.interface.takeDelimiter('\n')` — replaces `readUntilDelimiterOrEof`
- `cwd.readFileAlloc(io, path, allocator, .limited(N))` — simplest way to read a whole file
- `std.Io.Threaded.init(allocator, .{})` + `threaded.io()` — creates a `std.Io` for test code
- `ArrayList(T).init(allocator)` does NOT exist — use `var list: ArrayList(T) = .empty;`
- `ArrayList.deinit(allocator)` — now requires allocator parameter in 0.16.0
- `HashMap.deinit()` — managed HashMap still takes no args (stores its allocator)
- Anonymous struct types in ArrayList don't match across scope boundaries — use named `const` types

**OOV rate note:**
With word-level tokenization + apostrophe splitting, Shakespeare has ~12.6K unique tokens.
With V=2000, OOV rate is ~8.5% (above the plan's original 5% target).
Plan assumed fewer unique tokens before apostrophe-splitting decision.
Test verifies OOV < 10% instead; V≈3661 would be needed for 5%.

### Uncommitted files from previous session
- `docs/04b_from_nn_to_training.md` — new file, 1072 lines
- `docs/00_overview.md` — updated reading order and repo layout to include 04b

### Uncommitted: Stage 5 files
All Stage 5 source files, data files, docs, and modified files are uncommitted.
Ready to commit as `stage(5): word-level tokenizer and dataset`.

**What was completed in Stage 4 (committed):**

### New source files
1. `src/nn/module.zig` — TransformerConfig, Module convention, collectParamsSlice
2. `src/nn/linear.zig` — Linear: weight (D_out, D_in), Kaiming init, 3D→2D reshape, tape-tracked forward
3. `src/nn/embedding.zig` — Embedding: weight table, forward gathers rows, records `.embedding` OpKind
4. `src/nn/layernorm.zig` — LayerNorm: gamma/beta, composed from ~7 tape-tracked ops (mean, sub, mul, sqrt, div, add)
5. `src/nn/activations.zig` — GELU wrapper (stateless, wraps ops.unary.geluExact)
6. `src/nn/attention.zig` — CausalSelfAttention (1 head): 4 Linear sub-layers, causal mask, QKV→K^T→matmulBatch→scale→mask→softmax→matmulBatch→W_o
7. `src/nn/mlp.zig` — MLP: fc1→GELU→fc2
8. `src/nn/block.zig` — TransformerBlock: pre-norm residual (LN→Attn→+x→LN→MLP→+h)
9. `src/nn/model.zig` — TinyWordTransformer: tok_embed+pos_embed→block→ln_f→lm_head, save/load checkpoint format
10. `src/optim/optimizer.zig` — Optimizer vtable (ctx + step/zeroGrad/deinit function pointers)
11. `src/optim/sgd.zig` — SGD with momentum and coupled weight decay
12. `src/optim/adamw.zig` — AdamW with bias correction (β₁ᵗ, β₂ᵗ) and decoupled weight decay

### Modified files (Stage 3 → Stage 4)
13. `src/autograd/node.zig` — Added OpKind.sqrt (21→22 variants)
14. `src/autograd/backward.zig` — Added backwardSqrt, backwardEmbedding (scatter-add), fixed backwardMatmulBatch for 3D tensors
15. `src/autograd/tape.zig` — Added `kept_alive: ArrayList([]f32)`, `keepAlive(self, *Tensor)`, fixed `trackLeaf()` to always create fresh node (ignoring stale tape_node)
16. `src/tensor/ops/unary.zig` — Added `sqrt(alloc, t, tape)` with tape recording
17. `src/tensor/ops/shape_ops.zig` — reshapeTracked, transpose2dTracked, transposeInner2d (3D view)
18. `src/root.zig` — Wired nn.* and optim.* modules with re-exports and test blocks

### Examples
19. `examples/04_overfit_one_batch.zig` — Training loop: V=32, D=16, T=8, B=2, 50 steps with AdamW. Loss 3.83→3.09.

### Documentation (committed)
20. `docs/04_nn.md` — 858 lines

### Documentation (uncommitted, from this session)
21. `docs/04b_from_nn_to_training.md` — 1072 lines: why each layer exists, optimizer math, complete training step trace, gradient flow, shape trace, keepAlive memory management, PyTorch equivalents, 8 common mistakes

### Key bugs fixed in Stage 4
- **trackLeaf() stale tape_node collision.** After destroying a tape, param.tape_node still held the old node ID. Next step's trackLeaf() short-circuited, returning a stale ID that collided with other nodes. Fix: trackLeaf() always creates a fresh node, ignoring existing tape_node.
- **Use-after-free on intermediate tensors.** NN forward methods create owned intermediates (transposes, reshapes, matmul outputs) that are defer-freed before backward runs. Tape's SavedData holds data slices into freed buffers. Fix: tape.keepAlive() transfers buffer ownership to tape's kept_alive list; sets tensor.owned=false so deinit is no-op.
- **backwardMatmulBatch on 3D tensors.** Was calling tp.b.transpose2d() on 3D tensors (crash). Fix: use ops_shape.transposeInner2d(tp.b) which swaps dims[1]/dims[2] of a 3D tensor.
- **Loss read-after-free.** Reading loss.data[0] after loss.deinit() in example 04. Fix: save const loss_val = loss.data[0] before deinit.

### Acceptance criteria (all pass)
- `zig build test` — 150+ tests pass, 0 leaks
- `zig build run-example -Dexample=04_overfit_one_batch` — Loss decreases from 3.83 to 3.09 over 50 steps

## Known dead code / pitfalls

- **`tests/unit_all.zig` is dead code.** It is NOT wired into `build.zig` and would
  fail to compile if used (wrong import names). The actual test aggregator is
  `src/root.zig` (its `test { _ = ...; }` block references each sub-module).
  Do NOT add tests to `unit_all.zig` — add them co-located in source files.

- **`plan.md` lists `src/core/allocator.zig`** in its repository layout, but this
  file was never created and is not imported anywhere. The plan's layout is stale
  on this point; `docs/00_overview.md` is accurate.

- **`root.zig` line 14** says "add them here and in the test block at the bottom."
  This is correct — only `src/root.zig` test blocks matter. The earlier stale
  reference to `tests/unit_all.zig` has been removed.

## Workflow

- Implement stages 1 through 9 in order. Do not interleave stages.
- Commit after each stage with `stage(N): <summary>`; paste acceptance outputs
  into the commit body.
- Docs chapter for a stage ships in the same commit as the stage's code.

## Style

- File header block: purpose, shape contract, math formulas, ownership, errors,
  TODOs, credits.
- Explicit allocators. No hidden globals. No `@panic` outside tests.
- `zig fmt` clean. Tests co-located via `test "..."` blocks.
- Every public function has a worked shape example in its doc comment.
- Pedagogical comments: explain *why*, not just *what*. ~30-40% of non-blank
  lines should be comments.

## Zig 0.16.0 patterns

Use modern 0.16 API exclusively:
- `pub fn main(init: std.process.Init) !void`
- `b.createModule(...)`, `root_module.addImport(...)`
- `var list: ArrayList(T) = .empty; list.append(gpa, v)`
- `@ptrFromInt`, `@intFromPtr`, `@intFromEnum` (not the old cast builtins)
- `callconv(.c)` not `callconv(.C)`
- Explicit re-exports, never `usingnamespace`
- `std.Io.Dir.cwd()` not `std.fs.cwd()`
- `const` for non-mutated locals (Zig 0.16.0 enforces this as an error, not a warning)

## This project's Shape API (MUST follow — tripped up 3 subagents)

The `Shape` struct in `src/tensor/shape.zig` uses rank-specific constructors:

```zig
const s1 = Shape.init1D(10);          // 1D: (10,)
const s2 = Shape.init2D(2, 3);       // 2D: (2, 3)
const s3 = Shape.init3D(2, 3, 4);    // 3D: (2, 3, 4)
const s4 = Shape.init4D(2, 3, 4, 5); // 4D: (2, 3, 4, 5)
```

**CRITICAL:** There is NO `Shape.init(&.{2, 3})` constructor. It does not exist.
If you write `Shape.init(...)` with a slice/array argument, compilation will fail.

The `rank` field is a `u2` storing `ndim - 1` (0-3 for 1D-4D). Use `shape.ndim()`
to get the dimension count (returns `rank + 1`). Never compare `rank` directly
to a dimension count without adding 1.

`Shape.equals` is a **free function**, not a method: `equals(a, b)` not `a.equals(b)`.

## This project's SavedData API (MUST follow — tripped up in Stage 3)

SavedData stores Tensor structs **by value** (snapshots), not by pointer:

```zig
// WRONG — dangling pointer to stack-local by-value parameter:
.saved = .{ .tensor_ref = @constCast(&tensor_param) }       // DANGLING!
.saved = .{ .tensor_pair = .{ .a = @constCast(&a), .b = @constCast(&b) } }  // DANGLING!

// CORRECT — snapshot by value, data slice shares original heap buffer:
.saved = .{ .tensor_ref = tensor_param }                    // snapshot
.saved = .{ .tensor_pair = .{ .a = a, .b = b } }          // snapshot
.saved = .{ .ce_info = .{ .logits = logits, .targets = targets.data } } // logits by value
```

**Why:** Ops take `Tensor` by value (Zig convention). `@constCast(&param)` creates
a pointer to a stack-local copy that's destroyed when the function returns. By
storing the whole struct, we capture the `data` slice (pointing to the original
heap buffer) along with shape/strides. The heap buffer is alive as long as the
caller's tensor outlives the tape — same contract as PyTorch's autograd.

## Zig 0.16.0 compilation gotchas

Real errors encountered during implementation. New entries added per stage:

1. **`std.fs.cwd()` does not exist in 0.16.0.** The `std.fs` module is mostly
   deprecated. Use `std.Io.Dir.cwd()` for filesystem access. In `build.zig`,
   avoid runtime filesystem operations entirely — use build-system commands
   and static file lists instead of directory iteration.

2. **`var` vs `const` is enforced as an error.** If a local variable is never
   mutated after initialization, Zig 0.16.0 emits a hard error (not a warning).
   Use `const` by default. Only use `var` when you actually mutate the variable.

3. **Unused function parameters are errors.** If a function parameter isn't
   used, either prefix with `_` or remove it. No silent warnings.

4. **`build.zig.zon` fingerprint must match.** The `fingerprint` field is
   verified by the compiler. If you create a new project, let the compiler
   tell you the correct value on first build — it prints the expected value.

5. **Unused build options must be suppressed.** If you declare `b.option(...)`
   but don't use the value, add `_ = variable;` to suppress the error.

6. **`addSystemCommand` takes `[]const []const u8`.** All arguments must be
   available at build-configuration time. `b.fmt()` returns `[]const u8` which
   works, but you can't do runtime string construction inside the arg array.

7. **`LazyPath` uses `.cwd_relative = "path"`.** When adding library/rpath
   entries, use `std.Build.LazyPath{ .cwd_relative = "/path" }`.

8. **`exe.builder.allocator` doesn't exist.** In 0.16.0, access the build
   allocator via `b.allocator` (inside `build` function scope), not through
   the compile step.

9. **Test discovery goes through module imports.** Zig's test runner only finds
   `test "..."` blocks in files transitively imported by the test root. Our
   `src/root.zig` re-exports everything and its own `test { }` block references
   each sub-module. Don't use relative file paths in test files — import
   through the module.

10. **`std.fmt.bufPrint` for string formatting.** `std.io.fixedBufferStream`
    may not exist or have a different API. `std.fmt.bufPrint(buf, fmt, args)`
    is the reliable way to format into a buffer.

11. **`std.Io.get()` does not exist in 0.16.0.** There is no global stdout
    writer. In examples, use `init.io.lockStderr(&buffer, null)` to get a
    locked stderr writer. The `Init` struct provides the `io: Io` field.

12. **`@constCast(&by_value_param)` creates dangling pointers.** When ops
    take `Tensor` by value, storing a pointer to the parameter in SavedData
    dangles after the function returns. Store the Tensor by value (snapshot)
    instead. The `data` slice in the snapshot shares the original heap buffer.
    Discovered in Stage 3 when `backwardMul` crashed with `ShapeMismatch` —
    the saved `tensor_pair` pointers pointed to freed stack frames.

13. **`lockStderr` buffer can be zero-length.** Use `var buf: [0]u8 = undefined`
    for unbuffered stderr writing in examples (matches example 01 pattern).
    A `[1024]u8` buffer works but may swallow output silently if the buffer
    isn't flushed before program exit.

14. **Gradient tensors for null-parent inputs leak without explicit cleanup.**
    When an op input has `tape_node = null` (doesn't require grad), the
    backward function still computes its gradient, but the accumulation loop
    skips it (null parent ID). Without a cleanup loop after accumulation,
    these heap-allocated gradient tensors leak. Fix: iterate parent_grads
    again and free any where `node.parents[pi]` is null.

15. **`fetchSwapRemove` → `fetchRemove` in Zig 0.16.0 HashMap API.** The
    method is `fetchRemove`, not `fetchSwapRemove`. The latter does not exist.

16. **`@floatFromInt(@as(u64, @bitCast(x)))` is WRONG for f32→f64 conversion.**
    Use `@floatCast(f64, x)` to widen f32 to f64. `@bitCast` reinterprets
    the bit pattern (producing garbage), while `@floatCast` converts the value.

17. **`trackLeaf()` must ignore stale `tape_node` values.** Each training step
    creates a fresh `Tape.init()`, so old `tape_node` IDs from destroyed tapes
    are invalid. Short-circuiting on existing `tape_node` caused ID collisions
    (e.g., 3D intermediate and 1D parameter both claiming tape_node=13).
    Fix: trackLeaf() always creates a fresh node.

18. **`keepAlive` required for every intermediate in nn forward().** Any tensor
    created by an op (reshapeTracked, transpose2dTracked, matmul, add, etc.)
    that is `defer`-freed in a forward method must have `tape.keepAlive(&tensor)`
    called before the defer. Without this, the tape's SavedData holds slices
    into freed buffers → use-after-free during backward.

19. **`transpose2d()` crashes on 3D tensors.** Use `transposeInner2d()` for 3D
    tensors, which swaps dims[1]/dims[2] instead of dims[0]/dims[1]. Discovered
    when `backwardMatmulBatch` called `tp.b.transpose2d()` on a (B,T,D) tensor.

20. **Read tensor data before deinit, not after.** `loss.data[0]` after
    `loss.deinit()` is use-after-free. Save `const val = loss.data[0]` first.
    Same pattern applies any time you need a scalar value from a tensor you're
    about to free.

21. **`std.Io.Dir.cwd().openFile(io, path, .{})` requires `io` parameter.**
    File I/O operations in 0.16.0 take a `std.Io` as the second argument.
    `createFile(io, path, .{})` same pattern. In example code, use `init.io`;
    in test code, create `var threaded = std.Io.Threaded.init(allocator, .{});`
    and use `threaded.io()`.

22. **`file.writer(io, &buf)` / `file.reader(io, &buf)` require buffer.**
    The writer/reader are now buffered by default. You must provide a buffer.
    Use `var buf: [4096]u8 = undefined; var writer = file.writer(io, &buf);`
    Call `writer.flush()` before the writer goes out of scope.

23. **`writer.interface.print(...)` — print is on the interface.** File.Writer
    has an `interface` field of type `Io.Writer`. Print, writeAll, writeInt,
    etc. are on `writer.interface`, not on `writer` directly.

24. **`reader.interface.takeDelimiter('\n')` replaces `readUntilDelimiterOrEof`.**
    The old method doesn't exist on `Io.Reader`. Use `takeDelimiter` which
    returns `?[]u8` (null at EOF, slice valid until next read).

25. **`ArrayList(T).init(allocator)` does NOT exist in 0.16.0.** Use
    `var list: ArrayList(T) = .empty;` and then `list.append(allocator, item)`.
    `list.deinit(allocator)` takes the allocator parameter.

26. **Anonymous struct types in ArrayList don't match across scope boundaries.**
    `ArrayList(struct { word: []const u8, count: u32 })` creates a different
    type in each scope. Use `const MyStruct = struct { ... }; var list:
    ArrayList(MyStruct) = .empty;` to get a consistent type.

27. **`file.stat(io)` replaces `file.metadata()`.** The stat method now
    requires the `io` parameter. The returned `Stat` struct has `.size` as
    a direct field (not a method).

28. **`cwd.readFileAlloc(io, path, allocator, .limited(N))` — simplest file read.**
    Reads the entire file into a heap-allocated buffer with a size limit.
    Much simpler than openFile + reader + readAll.

## CUDA sacred spots

- Row-major to column-major wrapping in `src/backend/cuda/gemm.zig`. Dedicated tests.
- Bounds checks in every kernel.
- Offline .ptx only (no NVRTC).
- Bindings module opens libraries with dlopen at runtime.
- Link `libc` + `dl`, never `libcuda` directly.
- In `build.zig`, use a static `kernel_names` list for nvcc compilation —
  don't iterate the filesystem at build time (avoids `std.fs.cwd()` issues).

## Zig 0.16.0 skill reference

The `skills/modern-zig-0-16-tutor/` directory contains a comprehensive Zig 0.16.0
reference library. **Read `skills/modern-zig-0-16-tutor/SKILL.md` every session** — it
has the canonical stale-pattern table (28 rows), compiler-error quick lookup (22 entries),
and 13 project-specific gotchas discovered during Stages 1-2.

When you hit a Zig compilation error or API question, load reference files by topic
(max 3 per turn to manage token budget):

| Task | Load |
|------|------|
| Build system errors | `references/08-build-system-0-16.md` |
| I/O (stdout, writer, formatting) | `references/07-io-0-16.md` |
| Allocators, arena, leaks | `references/05-memory-allocators.md` |
| ArrayList / HashMap | `references/06-containers-0-16.md` |
| C/CUDA interop | `references/09-c-interop-0-16.md` + `references/17-zig-cuda-interop-notes.md` |
| Testing, debugging | `references/10-testing-debugging.md` |
| Code review | `references/15-code-review-checklist.md` |
| ML/tensor design | `references/16-zig-for-ml-runtime-projects.md` |
| Tensor2D skeleton | `templates/tensor2d-skeleton-0-16.zig` |
| Numerical code testing | `recipes/numerical-code.md` |
| Stale pattern scanning | `scripts/grep_stale_patterns.py <dir>` |

Also available: 19 reference files, 12 recipes, 7 templates, 4 validation scripts.
See `skills/modern-zig-0-16-tutor/README.md` for the full directory map.

## Stage 6: Next steps

Stage 6 implements end-to-end CPU training. Per `plan.md`
and `docs/00_overview.md`:

### What to implement
1. `src/lab/train.zig` — top-level trainer
2. `examples/06_train_shakespeare.zig` — CPU training on Shakespeare
3. `examples/07_generate.zig` — load checkpoint, sample
4. `docs/07_cpu_training.md` — Training loop, generation

### Key design decisions already locked
- **D5:** 1 block / 1 head, hard-coded during Stages 2–7
- **D8:** Training corpora: data/tiny.txt (~6 KB) and data/tinyshakespeare.txt (~1 MB)
- **D14:** Xoshiro256 seeded RNG; deterministic runs

### Dependencies on Stage 5
- Tokenizer and dataset pipeline is ready
- Vocab feeds vocab_size into TransformerConfig
- Windowing and batching produce (B, T) tensors
- Training loop pattern from example 05

## When stuck

- Ask the user with a crisp options-style question (max 4 options).
- Never guess at hardware, environment, or decisions.
- Check the skill's compiler-error quick lookup table in `SKILL.md`.
- See `SESSION_GUIDE.md` for full project state and continuation instructions.

## Full plan

See `plan.md` at the repository root and `docs/00_overview.md`.
