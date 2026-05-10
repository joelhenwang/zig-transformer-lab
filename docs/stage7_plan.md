# Stage 7 Implementation Playbook

> **This document is the single source of truth for Stage 7 (CUDA
> backend).** A fresh agent session should be able to execute any
> `[ ]` PR below without rediscovering context. Read top-to-bottom
> once; then treat each PR card as a self-contained work order.

> Companion reading: `AGENTS.md` (contract, gotchas, hard rules),
> `plan.md` (original cross-stage plan and the D1–D14 locked
> decisions), `docs/00_overview.md` (project scope and decision
> table), `docs/02c`–`07d` (the Stage 6.5 architecture chapters).
> `docs/oracle.md` for the parity-test methodology.

---

## 0 — How to use this document (for the next session)

You are about to start (or resume) Stage 7 on a fresh session. The
following order produces minimum context-loss and fewest re-derivations:

1. Read `AGENTS.md` end-to-end. It contains:
   - Locked decisions D1–D14 and policies P1–P4.
   - The 35+ Zig 0.16.0 gotchas table.
   - Current progress table (Stage 6.5 and 7-setup are done).
   - Remote RTX workflow (script usage, CUDA 13.2 note, smoke test
     confirmed state).
2. Read this file in full.
3. Skim `docs/02c_tensor_invariants.md`, `02d_storage_and_views.md`,
   `03c_saved_tensors.md`, `07c_optimizer_state.md`,
   `07d_checkpoint_format.md` for architectural context.
4. Skim `docs/oracle.md` for parity-test methodology.
5. Execute the next `[ ]` PR in Section 4 using its PR card in
   Section 6.
6. After each PR lands: update the progress checkbox in Section 4
   and add an entry to Section 10.
7. Commit after every PR. Push after every commit unless user says
   otherwise. Commit style: `stage(7X): <scope>` where `X` matches
   the PR letter.

**Do not** re-derive decisions already locked in D1–D14 or this
document. If a decision seems wrong, surface it in a question with
max 4 options and wait.

**Do** ask the user when you hit an ambiguous state. The bar is
"crisp options-style question, max 4 options".

---

## 1 — Current state snapshot

As of commit `230097b`:

```
230097b docs: mark stage 6.5 and 7-setup done in progress table
1e3b540 stage(7-setup): remote runner scripts and workflow docs
3331801 feat(oracle): expand parity coverage to 14 cases
97b0aaa feat(oracle): PyTorch parity tests for CPU ops
28e73e1 docs(6.5): teaching chapters for the CPU hardening PRs
f9c1d3b stage(6.5): cpu hardening and backend seam
f29de59 stage 6 complete
```

### Tests & builds

```
zig build test         → 263/263 pass on Windows and Linux
zig build test-oracle  → 14/14 pass on Windows and Linux
zig build docs         → 13 chapters, 15,555 lines
zig build kernels      → no-op (kernel_names = [_][]const u8{})
```

### Training baseline

Example `04_overfit_one_batch` at step 99: **loss = 0.537453**, bit-
identical across every Stage 6.5 PR. Any Stage 7 change that
touches CPU paths must preserve this (CUDA path has its own tolerance
budget).

---

## 2 — Environment

### Local (Windows, where OpenCode runs)

| Item | Value |
|---|---|
| Shell | PowerShell 7 |
| Bash | MSYS 5.2.37 from Git for Windows, in PATH |
| Zig | 0.16.0 exact |
| Python | 3.12.7, torch 2.11.0+cpu, numpy 2.3.5 |
| OpenSSH | Built-in Windows client; key-based auth to remote |
| Git config | `core.autocrlf` enforced LF for `*.sh` / `*.py` via `.gitattributes` |

### Remote (`joelwang-rtx@192.168.1.197`, where CUDA lives)

| Item | Value |
|---|---|
| Hostname | `joelwang-rtx-MS-7C56` |
| OS | Ubuntu 24.04.4 LTS |
| Kernel | Linux 6.17.0-22-generic |
| CPU | AMD Ryzen 9 5900XT (32 threads) |
| GPU | NVIDIA GeForce RTX 4060 Ti, 16 GB VRAM, sm_89 |
| Driver | 595.58.03 |
| CUDA Toolkit | **13.2** (newer than the 12.x plan.md assumed) |
| nvcc | `/usr/local/cuda-13.2/bin/nvcc` |
| compute-sanitizer | `/usr/local/cuda-13.2/bin/compute-sanitizer` |
| libcublas | `libcublas.so.13` (not `.so.12`) |
| Zig | 0.16.0 exact |
| Python venv | `~/Desktop/ai_lab/.venv` with torch 2.11.0+cu130 |
| Repo | `~/Desktop/ai_lab/zig-transformer-lab` |

### Remote invocation via my shell tool

```
# Run any command on the remote, repo-rooted, venv active
bash ./run_remote_example.sh "<single-quoted command>"

# Rsync uncommitted working tree (primary workflow still prefers git)
bash ./sync_remote_example.sh
```

Both scripts embed the remote target. Update them if the target
changes. Scripts are LF-enforced by `.gitattributes` — rsync's
`--delete` means uncommitted-work iterations are safe.

---

## 3 — Locked decisions specific to Stage 7

These override anything in earlier docs if they conflict.

### From the D1–D14 table

- **D1** — Hand-written CUDA bindings. No external Zig dependency.
  Bindings load `libcuda.so.1`, `libcudart.so` (optional, see below),
  and `libcublas.so.13` via dlopen/dlsym.
- **D3** — Offline nvcc `-ptx`. No NVRTC, no JIT compilation.
- **D9** — f32 only. No mixed precision.
- **D10** — CUDA C kernels only. No pure-Zig GPU code.
- **D11** — Zig 0.16.0 exact.
- **D12** — Tape-based reverse-mode autograd, not generic.
- **D14** — Deterministic given a seed.

### Stage-7-specific policies

- **Target sm_89 only.** nvcc is always invoked with `-arch=sm_89`.
  Broadening targets is a Stage 9 concern.
- **No `--use_fast_math` in correctness mode.** PR-ν sets the
  correctness baseline; a future `-Dcuda_fast_math=true` option may
  land *after* the full-model parity test is green.
- **Every CUDA op ships with a CPU parity test** against an oracle
  fixture. New ops may require new fixtures in `tools/oracle.py`.
- **Row-major → cuBLAS column-major derivation must be written**
  into `docs/08_backends_cuda.md` **before** the GEMM wrapper is
  committed (PR-κ). Not optional. This is the single highest-risk
  subsystem.
- **No `linkSystemLibrary("cuda")`.** Libraries are dlopen'd at
  runtime. `build.zig` links `libc` (and `libdl` on Linux).
- **Every kernel starts with a bounds check.** Out-of-bounds reads
  or writes in kernels are treated as P0 bugs.
- **Every CUDA call is checked.** `bindings.check(result)` on every
  Driver API call; `bindings.checkCublas(status)` on cuBLAS. Missing
  checks are a review failure.
- **Debug builds synchronise after every kernel launch.**
  `cuCtxSynchronize` or `cuStreamSynchronize` in debug to surface
  async errors; skipped in release for throughput.
- **Forbid implicit CPU/CUDA transfers.** `requireSameDevice(a, b)`
  from PR-γ catches this. No op ever auto-copies.

---

## 4 — Stage 7 PR roadmap

Strict execution order. Dependency column shows what must be merged
first. Flip `- [ ]` to `- [x]` as PRs land, add the commit hash, and
append an entry in Section 10.

| # | PR | Scope (one line) | Depends on | Status |
|---|---|---|---|---|
| 1 | α | Dynamic loader smoke test (no kernels) | — | `[x]` 07bd274 |
| 2 | β | `CudaContext` lifecycle (device, context, stream, cuBLAS) | α | `[x]` 3c409a1 |
| 3 | γ | `DeviceBuffer` alloc/free/copy | β | `[x]` be977ea (+6b918e6 fix) |
| 4 | δ | Tensor `Storage.cuda` variant | γ | `[x]` fe269ef |
| 5 | ε | `Tensor.toCuda` / `toCpu` + roundtrip | δ | `[x]` be6e0f8 |
| 6 | ζ | PTX loader + vector-add smoke kernel | β | `[x]` d2b458f (+6c8b630, 3d73ef0 fixes) |
| 7 | η | Elementwise CUDA ops (same-shape) + parity | ε, ζ | `[x]` 849947c (forward) + cab742c (tape + add routing) + 2ad2ba7 (remaining 6 ops routed) |
| 8 | θ | Broadcasting + scalar CUDA ops + parity | η | `[x]` 809fd90 (forward; backward via PR-ι) |
| 9 | ι | CUDA reductions (sum, mean, sumAll) + parity | θ | `[x]` 43a2f1e (sum/sumAll/broadcastTo/sumToShape + full add_2d fwd+bwd oracle parity) |
| 10 | κ | cuBLAS row-major GEMM + docs/08 | γ | `[x]` 42e917f (+4e40453 transpose fix) |
| 11 | λ | Softmax + causal mask CUDA + parity | η | `[x]` 52abdf7 (softmax + log_softmax; causal mask reuses ops_elementwise.add) |
| 12 | μ | Embedding + cross-entropy + AdamW CUDA + parity | κ, λ | `[ ]` |
| 13 | ν | Full-model CUDA parity (uses `full_model_forward` fixture) | all prior | `[ ]` |
| 14 | ξ | Training speed benchmarks | ν | `[ ]` |

Estimated effort: 40–60 hours across 12–15 sessions.

---

## 5 — CUDA 13.2 gotchas (confirmed on the remote)

These were discovered during the 7-setup smoke test and must inform
every PR below.

### 5.1 Library name versioning

- **Driver library:** `libcuda.so.1` (unchanged across CUDA majors).
- **cuBLAS:** `libcublas.so.13` on CUDA 13, not `.so.12`. Our dlopen
  code tries `.13` first, falls back to `.12` with a warning. If
  both fail, return `error.CudaError` with a clear message.
- **Runtime (cudart):** typically `libcudart.so.13` on CUDA 13. We
  only need it for `cudaGetLastError` / `cudaDeviceSynchronize`;
  these can be replaced by Driver API equivalents
  (`cuCtxSynchronize` alone is enough for our uses). **Decision:**
  do not dlopen libcudart in PR-α; add only if a later PR actually
  needs it.

### 5.2 Symbol naming

- Use the `_v2` suffix for context, memory, and stream APIs:
  `cuCtxCreate_v2`, `cuCtxDestroy_v2`, `cuMemAlloc_v2`,
  `cuMemFree_v2`, `cuMemcpyHtoD_v2`, `cuMemcpyDtoH_v2`,
  `cuStreamDestroy_v2`, `cublasCreate_v2`, `cublasDestroy_v2`,
  `cublasSgemm_v2`, `cublasSetStream_v2`.
- `cuInit`, `cuDeviceGetCount`, `cuDeviceGet`, `cuStreamCreate`,
  `cuStreamSynchronize`, `cuModuleLoadData`, `cuModuleGetFunction`,
  `cuLaunchKernel`, `cuGetErrorString`: unversioned.
- `cuDeviceGetName_v2` is the canonical name on CUDA 12+. The
  unversioned `cuDeviceGetName` may still resolve, but the `_v2`
  form is the ABI we want to target.
- cuBLAS status-string: `cublasGetStatusString` (unversioned).

### 5.3 PTX format

- nvcc 13.2 emits `.target sm_89` and `.version 8.x` for our
  kernels. Driver 595.x accepts these fine (`cuModuleLoadData`
  returns `CUDA_SUCCESS`).
- If a kernel ever returns `CUDA_ERROR_NO_BINARY_FOR_GPU`
  (0x300 / 209), the `-arch=` in `build.zig` doesn't match the
  target device. Verify with `nvcc --help | grep arch`.

### 5.4 Removed APIs (CUDA 13 vs 12)

- `cuCtxAttach` / `cuCtxDetach`: removed. We use `cuCtxCreate_v2` /
  `cuCtxDestroy_v2`. Unaffected.
- `cuMemAllocManaged` signature changed around the flags argument;
  we don't use managed memory. Unaffected.

### 5.5 Runtime vs driver APIs

- We target the **Driver API** (cu*), not the Runtime API (cuda*),
  because the Driver API is what `libcuda.so.1` provides and
  doesn't require `libcudart` (which is CUDA-version-specific).
- One exception: `cublasGetStatusString` lives in `libcublas.so`.
  We dlopen that library for GEMM; getting the status string is
  free.

---

## 6 — PR cards

Each card is a complete work order. Follow it without re-designing.

---

### PR-α — CUDA dynamic loader smoke test

**Status:** `[ ]` Pending

**Purpose.** Prove that the build system, dlopen machinery, and
symbol resolution work end-to-end on the remote's CUDA 13.2 install.
No math, no kernels, no tensors on GPU. This is the cheapest
insurance against discovering at PR-η that a symbol was renamed.

**Scope.**

1. Add `src/backend/cuda/bindings.zig` — a pure-Zig module defining:
   - Opaque handle types (`CUdevice`, `CUcontext`, `CUstream`,
     `CUmodule`, `CUfunction`, `cublasHandle_t`).
   - The error enums (`CUresult`, `cublasStatus_t`) and their
     success constants.
   - A `Loader` struct holding function pointers for every symbol
     we will ever need during Stage 7 (~30 symbols).
   - `pub var loader: ?Loader = null;` — populated by `load()`.
   - `pub fn load() LabError!void` — dlopens `libcuda.so.1` and
     `libcublas.so.13` (fallback `.so.12`), dlsyms each entry point,
     surfaces any missing symbol as `error.CudaError` with the
     symbol name logged in debug.
   - `pub fn check(r: CUresult) LabError!void` — maps non-zero to
     `error.CudaError`; in debug builds logs the result code and
     the string from `cuGetErrorString` if available.
   - `pub fn checkCublas(s: cublasStatus_t) LabError!void` — same
     for cuBLAS.
2. Add `tests/integration_cuda.zig` with a single test:
   - Call `bindings.load()`.
   - `cuInit(0)`.
   - `cuDeviceGetCount(&n)`; assert `n >= 1`.
   - `cuDeviceGet(&dev, 0)`.
   - `cuDeviceGetName_v2(&buf[0], 128, dev)`; log the name, assert
     it contains "NVIDIA" or "GeForce".
   - `cuCtxCreate_v2(&ctx, 0, dev)` + `defer cuCtxDestroy_v2(ctx)`.
   - `cublasCreate_v2(&handle)` + `defer cublasDestroy_v2(handle)`.
3. The `build.zig` is already scaffolded to pick up
   `tests/integration_cuda.zig` when `-Dcuda=true`. Creating the
   file activates the path.

**Files to create.**

- `src/backend/cuda/bindings.zig` (new, ~400 LOC with comments)
- `tests/integration_cuda.zig` (new, ~60 LOC)

**Files to modify.**

- None directly. `build.zig` already has the `if (cuda) { ... }`
  branch for `tests/integration_cuda.zig`.
- `src/root.zig`: no change needed. CUDA types are backend-internal;
  they get exposed at the library root in PR-δ when Tensor acquires
  a `storage.cuda` variant.

**API surface to introduce.**

```zig
// src/backend/cuda/bindings.zig

pub const CUdevice = c_int;
pub const CUcontext = ?*opaque {};
pub const CUstream = ?*opaque {};
pub const CUmodule = ?*opaque {};
pub const CUfunction = ?*opaque {};
pub const CUdeviceptr = u64;  // 64-bit handle, NOT a host pointer
pub const cublasHandle_t = ?*opaque {};

pub const CUresult = c_int;
pub const CUDA_SUCCESS: CUresult = 0;

pub const cublasStatus_t = c_int;
pub const CUBLAS_STATUS_SUCCESS: cublasStatus_t = 0;

pub const Loader = struct {
    libcuda: *anyopaque,
    libcublas: *anyopaque,

    // Initialisation / device
    cuInit:             *const fn (flags: c_uint) callconv(.c) CUresult,
    cuDeviceGetCount:   *const fn (*c_int) callconv(.c) CUresult,
    cuDeviceGet:        *const fn (*CUdevice, c_int) callconv(.c) CUresult,
    cuDeviceGetName_v2: *const fn ([*]u8, c_int, CUdevice) callconv(.c) CUresult,
    cuGetErrorString:   *const fn (CUresult, *[*:0]const u8) callconv(.c) CUresult,

    // Context
    cuCtxCreate_v2:  *const fn (*CUcontext, c_uint, CUdevice) callconv(.c) CUresult,
    cuCtxDestroy_v2: *const fn (CUcontext) callconv(.c) CUresult,
    cuCtxSetCurrent: *const fn (CUcontext) callconv(.c) CUresult,
    cuCtxSynchronize: *const fn () callconv(.c) CUresult,

    // Stream
    cuStreamCreate:      *const fn (*CUstream, c_uint) callconv(.c) CUresult,
    cuStreamDestroy_v2:  *const fn (CUstream) callconv(.c) CUresult,
    cuStreamSynchronize: *const fn (CUstream) callconv(.c) CUresult,

    // Memory
    cuMemAlloc_v2:     *const fn (*CUdeviceptr, usize) callconv(.c) CUresult,
    cuMemFree_v2:      *const fn (CUdeviceptr) callconv(.c) CUresult,
    cuMemcpyHtoD_v2:   *const fn (CUdeviceptr, *const anyopaque, usize) callconv(.c) CUresult,
    cuMemcpyDtoH_v2:   *const fn (*anyopaque, CUdeviceptr, usize) callconv(.c) CUresult,
    cuMemcpyDtoD_v2:   *const fn (CUdeviceptr, CUdeviceptr, usize) callconv(.c) CUresult,

    // Modules / kernels
    cuModuleLoadData:    *const fn (*CUmodule, *const anyopaque) callconv(.c) CUresult,
    cuModuleUnload:      *const fn (CUmodule) callconv(.c) CUresult,
    cuModuleGetFunction: *const fn (*CUfunction, CUmodule, [*:0]const u8) callconv(.c) CUresult,
    cuLaunchKernel:      *const fn (
        f: CUfunction,
        gridX: c_uint, gridY: c_uint, gridZ: c_uint,
        blockX: c_uint, blockY: c_uint, blockZ: c_uint,
        sharedBytes: c_uint,
        stream: CUstream,
        params: ?*[*]?*anyopaque,
        extra: ?*[*]?*anyopaque,
    ) callconv(.c) CUresult,

    // cuBLAS
    cublasCreate_v2:             *const fn (*cublasHandle_t) callconv(.c) cublasStatus_t,
    cublasDestroy_v2:            *const fn (cublasHandle_t) callconv(.c) cublasStatus_t,
    cublasSetStream_v2:          *const fn (cublasHandle_t, CUstream) callconv(.c) cublasStatus_t,
    cublasSgemm_v2:              *const fn (
        handle: cublasHandle_t,
        transa: c_int, transb: c_int,
        m: c_int, n: c_int, k: c_int,
        alpha: *const f32,
        A: CUdeviceptr, lda: c_int,
        B: CUdeviceptr, ldb: c_int,
        beta: *const f32,
        C: CUdeviceptr, ldc: c_int,
    ) callconv(.c) cublasStatus_t,
    cublasSgemmStridedBatched:   *const fn (
        handle: cublasHandle_t,
        transa: c_int, transb: c_int,
        m: c_int, n: c_int, k: c_int,
        alpha: *const f32,
        A: CUdeviceptr, lda: c_int, strideA: i64,
        B: CUdeviceptr, ldb: c_int, strideB: i64,
        beta: *const f32,
        C: CUdeviceptr, ldc: c_int, strideC: i64,
        batchCount: c_int,
    ) callconv(.c) cublasStatus_t,
    cublasGetStatusString: *const fn (cublasStatus_t) callconv(.c) [*:0]const u8,
};

pub var loader: ?Loader = null;

pub fn load() LabError!void { ... }
pub fn check(r: CUresult) LabError!void { ... }
pub fn checkCublas(s: cublasStatus_t) LabError!void { ... }
```

Note: the full symbol list is defined in PR-α even though many
symbols are only *used* in later PRs. This is intentional — we load
everything once and never touch `bindings.zig` again except to add
CUDA 14 / future symbols.

**Acceptance criteria.**

On the remote:

```
bash run_remote_example.sh "git pull --ff-only && zig build test -Dcuda=true 2>&1 | tail -15"
```

Expected tail:

```
1/1 integration_cuda.test.CUDA loader smoke... device: NVIDIA GeForce RTX 4060 Ti
OK
All 1 tests passed.
```

`EXITCODE=0`.

**Run commands (remote).**

```
# Normal
bash run_remote_example.sh "zig build test -Dcuda=true"

# Debug with symbol-resolution noise
bash run_remote_example.sh "zig build test -Dcuda=true -Doptimize=Debug 2>&1 | tail -30"

# Direct binary for clean output
bash run_remote_example.sh 'find .zig-cache/o -name test -type f -executable -newer build.zig | head -1 | xargs -I{} {}'
```

**Oracle strategy.** None. PR-α has no numeric computation. First
parity test is in PR-η.

**Commit message template.**

```
stage(7a): CUDA dynamic loader smoke test

Introduces the dlopen-based bindings layer every later Stage 7 PR
depends on. No kernels, no memory allocation, no tensors on GPU
yet — this PR only verifies the build system, dlopen path, and
symbol resolution against the remote's CUDA 13.2 install.

Files

  src/backend/cuda/bindings.zig   (~400 LOC)
    Opaque handle types. Loader struct with one fn-ptr per symbol
    we will need across all Stage 7 PRs (~30 total). load() opens
    libcuda.so.1 and libcublas.so.13 (fallback .so.12), dlsyms
    every entry point, logs missing symbols in debug and surfaces
    them as error.CudaError. check() / checkCublas() wrap the
    return codes.

  tests/integration_cuda.zig      (~60 LOC)
    One test: load + cuInit + cuDeviceGetCount + cuDeviceGet +
    cuDeviceGetName_v2 + cuCtxCreate_v2 + cuCtxDestroy_v2 +
    cublasCreate_v2 + cublasDestroy_v2. Asserts device count >= 1
    and device name contains NVIDIA.

Verification on joelwang-rtx-MS-7C56 (RTX 4060 Ti, CUDA 13.2)

  zig build test -Dcuda=true   → 1/1 CUDA tests pass
                                 device 0: NVIDIA GeForce RTX 4060 Ti
```

**Dependencies.** None.

**Blocks.** PR-β, PR-γ, PR-ζ (everything that calls CUDA).

**Estimated effort.** 2–3 hours. Most of it is correct function-
pointer signatures and testing that the dlopen path works; the
test itself is intentionally minimal.

**Gotchas.**

- `_v2` suffixes are mandatory for several symbols; `cuCtxCreate`
  without `_v2` exists in old CUDA 2.x and has a different ABI.
- `cuDeviceGetName_v2` writes a nul-terminated string into the
  buffer; compute the length with `std.mem.sliceTo` rather than
  assuming the full 128 bytes are used.
- Library fallback: log exact `dlerror()` before trying `.so.12`.
  The log line is how future sessions diagnose deployment drift.
- Do **not** `linkSystemLibrary("cuda")` or `"cublas")` in
  `build.zig`. All CUDA libs are dlopened.

**Post-PR.** Update Section 4 checkbox and add an entry in Section
10 with the commit hash and date.

---

### PR-β — `CudaContext` lifecycle

**Status:** `[ ]` Pending

**Purpose.** Wrap the "create device + context + stream + cuBLAS
handle" boilerplate into one object with a clear lifetime. Every
later PR takes `*CudaContext` as a parameter.

**Scope.**

1. Add `src/backend/cuda/context.zig` with:
   - `CudaContext` struct holding `device`, `ctx`, `stream`, `cublas`,
     `ptx_modules: std.StringHashMap(CUmodule)`, `allocator`.
   - `pub fn init(alloc, device_id) LabError!CudaContext`.
   - `pub fn deinit(self: *CudaContext) void` — tears down in reverse
     order, logs any non-success codes but does not return errors
     (deinit is infallible by convention).
   - `pub fn synchronize(self: *CudaContext) LabError!void` — wraps
     `cuStreamSynchronize(self.stream)`.
2. Tests in the same file (or `tests/integration_cuda.zig` expansion):
   - `init(alloc, 0)` + `defer deinit()`.
   - Sync on an empty stream succeeds.
   - Create, sync, destroy with no leaks.
3. Optionally expose a `pub fn getContext() ?*CudaContext` process-
   global for convenience (tests still construct explicitly). Keep
   this optional; if it's not needed by PR-γ, don't add it.

**Files to create.**

- `src/backend/cuda/context.zig` (new, ~200 LOC)

**Files to modify.**

- `tests/integration_cuda.zig` — add a context round-trip test.

**API surface.**

```zig
pub const CudaContext = struct {
    allocator: std.mem.Allocator,
    device: bindings.CUdevice,
    ctx: bindings.CUcontext,
    stream: bindings.CUstream,
    cublas: bindings.cublasHandle_t,
    ptx_modules: std.StringHashMap(bindings.CUmodule),

    pub fn init(alloc: std.mem.Allocator, device_id: c_int) LabError!CudaContext { ... }
    pub fn deinit(self: *CudaContext) void { ... }
    pub fn synchronize(self: *CudaContext) LabError!void { ... }
};
```

**Init sequence.**

```zig
try bindings.load();
try bindings.check(L.cuInit(0));
try bindings.check(L.cuDeviceGet(&device, device_id));
try bindings.check(L.cuCtxCreate_v2(&ctx, 0, device));
try bindings.check(L.cuStreamCreate(&stream, 0));
try bindings.checkCublas(L.cublasCreate_v2(&cublas));
try bindings.checkCublas(L.cublasSetStream_v2(cublas, stream));
```

**Deinit sequence (reverse).**

```zig
_ = L.cublasDestroy_v2(self.cublas);
_ = L.cuStreamDestroy_v2(self.stream);
_ = L.cuCtxDestroy_v2(self.ctx);
var it = self.ptx_modules.iterator();
while (it.next()) |e| {
    _ = L.cuModuleUnload(e.value_ptr.*);
    self.allocator.free(e.key_ptr.*);
}
self.ptx_modules.deinit();
```

**Acceptance criteria.**

- `zig build test -Dcuda=true` passes (2/2 CUDA tests now).
- `compute-sanitizer` clean on the test binary.
- No leaks reported by `GeneralPurposeAllocator(.{ .safety = true })`.

**Run commands.**

```
bash run_remote_example.sh "zig build test -Dcuda=true 2>&1 | tail -10"
bash run_remote_example.sh 'find .zig-cache/o -name test -type f -executable -newer build.zig | head -1 | xargs -I{} compute-sanitizer --leak-check=full {}'
```

**Oracle strategy.** None yet.

**Commit template.**

```
stage(7b): CudaContext lifecycle

Wraps device/context/stream/cuBLAS-handle creation into one struct
with a clear deinit order. Every later Stage 7 PR takes
*CudaContext as a parameter instead of re-building the full lane.

Files

  src/backend/cuda/context.zig   CudaContext struct, init/deinit/
                                 synchronize; holds a StringHashMap
                                 of PTX modules for PR-ζ onward.
  tests/integration_cuda.zig     Extended with context round-trip
                                 test (init + sync + deinit; no
                                 leaks, no residual CUDA errors).

Verification

  zig build test -Dcuda=true                     → 2/2 CUDA tests
  compute-sanitizer --leak-check=full <test exe> → 0 leaks
```

**Dependencies.** PR-α.

**Blocks.** PR-γ, PR-ζ, everything that creates GPU memory or
launches kernels.

**Estimated effort.** 2 hours.

**Gotchas.**

- Store the HashMap's string keys in memory the context owns (dup
  them on insert, free on deinit). Don't assume caller-lifetime.
- `cuStreamCreate` flags argument: pass `0` for default stream
  (synchronous with the null stream on the device). Non-default
  flags are a Stage 9 concern.
- If the remote has multiple GPUs, `cuDeviceGet(&dev, 0)` picks
  device 0; we assume single-GPU for Stage 7.

---

### PR-γ — `DeviceBuffer` alloc/free/copy

**Status:** `[ ]` Pending

**Purpose.** Typed RAII wrapper around `cuMemAlloc_v2`. The backing
for every CUDA tensor in later PRs.

**Scope.**

1. Add `src/backend/cuda/mem.zig`:
   - `DeviceBuffer` struct: `ptr: CUdeviceptr`, `len: usize` (in
     f32 elements), `ctx: *CudaContext`, `owned: bool`.
   - `pub fn alloc(ctx, n) LabError!DeviceBuffer` — allocates
     `n * @sizeOf(f32)` bytes; returns owned buffer.
   - `pub fn fromHost(ctx, slice) LabError!DeviceBuffer` — alloc +
     HtoD copy.
   - `pub fn toHost(self, alloc) LabError![]f32` — alloc []f32 on
     host, DtoH copy; caller owns returned slice.
   - `pub fn copyFromHost(self, slice) LabError!void` — HtoD into
     existing buffer; sizes must match.
   - `pub fn copyToHost(self, dst_slice) LabError!void` — DtoH
     into caller slice; sizes must match.
   - `pub fn copyFromDevice(self, src) LabError!void` — DtoD.
   - `pub fn deinit(self: *DeviceBuffer) void` — `cuMemFree_v2` if
     owned.
2. Tests: HtoD → DtoH roundtrip; bytes match exactly (f32 is bit-
   exact under copy).

**Files to create.**

- `src/backend/cuda/mem.zig` (new, ~250 LOC)

**Files to modify.**

- `tests/integration_cuda.zig` — add roundtrip test.

**API surface.**

```zig
pub const DeviceBuffer = struct {
    ctx: *const CudaContext,
    ptr: bindings.CUdeviceptr,
    len: usize,      // in f32 elements
    owned: bool,

    pub fn alloc(ctx: *const CudaContext, n: usize) LabError!DeviceBuffer { ... }
    pub fn fromHost(ctx: *const CudaContext, src: []const f32) LabError!DeviceBuffer { ... }
    pub fn toHost(self: DeviceBuffer, alloc: std.mem.Allocator) LabError![]f32 { ... }
    pub fn copyFromHost(self: DeviceBuffer, src: []const f32) LabError!void { ... }
    pub fn copyToHost(self: DeviceBuffer, dst: []f32) LabError!void { ... }
    pub fn copyFromDevice(self: DeviceBuffer, src: DeviceBuffer) LabError!void { ... }
    pub fn deinit(self: *DeviceBuffer) void { ... }
};
```

**Acceptance criteria.**

- Roundtrip test: write `[1.0, -2.5, 1e6, 1e-6, NaN]` (yes, NaN
  round-trips fine through memcpy), read it back, assert
  byte-identical. Use `std.mem.eql` on raw byte slices.
- No leaks under `GeneralPurposeAllocator(.{ .safety = true })`.
- `compute-sanitizer --tool=memcheck` clean.

**Run commands.**

```
bash run_remote_example.sh "zig build test -Dcuda=true"
bash run_remote_example.sh 'find .zig-cache/o -name test -type f -executable -newer build.zig | head -1 | xargs -I{} compute-sanitizer --tool=memcheck {}'
```

**Oracle strategy.** None. Pure byte-level copy; the IEEE 754
representation is preserved under memcpy without any arithmetic.

**Commit template.**

```
stage(7c): DeviceBuffer alloc / free / copy

RAII wrapper around cuMemAlloc_v2 + the four memcpy variants we
need (HtoD, DtoH, DtoD, D->host-alloc). Every later Stage 7 PR
touches GPU memory through this type, not via raw bindings.

Files

  src/backend/cuda/mem.zig        DeviceBuffer with alloc, fromHost,
                                   toHost, copyFromHost/Device,
                                   copyToHost, deinit. `len` is
                                   tracked in f32 elements; byte
                                   sizes computed at the API edge.
  tests/integration_cuda.zig      Roundtrip test: five f32 values
                                   including NaN round-trip through
                                   HtoD + DtoH byte-identically.

Verification

  zig build test -Dcuda=true                      → 3/3 CUDA tests
  compute-sanitizer --tool=memcheck <test exe>    → clean
```

**Dependencies.** PR-β.

**Blocks.** PR-δ, PR-ε, every kernel.

**Estimated effort.** 2 hours.

**Gotchas.**

- `cuMemAlloc_v2` takes *bytes*, not elements. Compute
  `n * @sizeOf(f32)` at the boundary.
- `cuMemcpyHtoD_v2` signature: destination is a `CUdeviceptr`,
  source is `*const anyopaque`. Do not pass `*f32` — cast to
  `*const anyopaque` explicitly.
- On deinit, `cuMemFree_v2` of a buffer whose context has been
  destroyed is UB. `DeviceBuffer` holds a `*const CudaContext`; we
  rely on the caller to deinit buffers before deiniting the
  context. Document this in `context.zig`.

---

### PR-δ — Tensor `Storage.cuda` variant

**Status:** `[ ]` Pending

**Purpose.** Replace the placeholder `cuda: void` in the `Storage`
union (introduced by PR-δ of Stage 6.5) with a real `CudaStorage`
struct backed by `DeviceBuffer`.

**Scope.**

1. In `src/tensor/tensor.zig`:
   - Add `CudaStorage` struct `{ buffer: DeviceBuffer, ... }`. The
     inner `DeviceBuffer` already tracks `ptr`, `len`, `owned`,
     `ctx`. The outer struct may be nearly empty; consider directly
     using `DeviceBuffer` as the payload.
   - Change `Storage = union(Device) { cpu: CpuStorage, cuda: CudaStorage }`.
   - Update `Storage.len` to dispatch to the active variant.
   - Update `Storage.deinit` to call the CUDA variant's `deinit`.
   - Update `nonOwningStorage` to handle the CUDA branch.
2. Update `Tensor.checkInvariants`:
   - CUDA tensors have `data.len == 0` (the host-side compat slice
     is intentionally empty).
   - `storage.cpu` and `device == .cpu` must agree as before;
     `storage == .cuda` implies `device == .cuda`.
3. No Tensor methods that read `.data[i]` should be callable on
   a CUDA tensor. The empty compat slice makes such calls iterate
   zero times — a loud failure mode, not silent wrong answers.
4. Update the PR-γ tests in `src/tensor/tensor.zig` that manually
   construct `Tensor{ ... }` literals to use the new union shape.

**Files to modify.**

- `src/tensor/tensor.zig` (Storage union, Tensor literals in tests)
- `src/root.zig` — expose `backend.cuda` submodule for the CUDA
  types to be reachable from test files.

**Files to create.**

- None. `src/backend/cuda/context.zig` and `mem.zig` already exist.

**API surface change.**

```zig
pub const CudaStorage = struct {
    buffer: DeviceBuffer,   // owns the CUdeviceptr if buffer.owned == true

    pub fn len(self: CudaStorage) usize {
        return self.buffer.len;
    }

    pub fn deinit(self: *CudaStorage) void {
        self.buffer.deinit();
    }
};

pub const Storage = union(Device) {
    cpu: CpuStorage,
    cuda: CudaStorage,

    pub fn len(self: Storage) usize { ... }
    pub fn deinit(self: *Storage, allocator: Allocator) void { ... }
};
```

**Acceptance criteria.**

- All existing 263 tests still pass (the Tensor invariant changes
  shouldn't affect CPU-only code).
- A new test constructs a Tensor with `Storage.cuda` by hand,
  calls `checkInvariants()`, asserts it passes.
- A new test constructs a degenerate CPU/CUDA mismatch and asserts
  `error.DeviceMismatch` / `error.InvalidLayout`.

**Run commands.**

```
bash run_remote_example.sh "zig build test -Dcuda=true"
zig build test   # on Windows, without -Dcuda, must also be green
```

**Oracle strategy.** None yet.

**Commit template.**

```
stage(7d): Tensor Storage.cuda variant

Replaces the Stage 6.5 placeholder `cuda: void` with a real
CudaStorage wrapping DeviceBuffer. Every op that branches on
storage type can now allocate, free, and own CUDA memory through
the same invariant-checked path the CPU variant uses.

Files

  src/tensor/tensor.zig     Storage.cuda now carries DeviceBuffer.
                            Storage.len and deinit dispatch correctly.
                            nonOwningStorage handles the CUDA branch.
                            checkInvariants asserts CUDA tensors have
                            empty data compat slices.
  src/root.zig              Re-exports backend.cuda symbols.
  tests                     CPU-only invariants unaffected; two new
                            tests exercise the CUDA storage path.

Verification

  zig build test -Dcuda=true   → 263 + N CUDA tests pass
  zig build test               → 263/263 unchanged on Windows
```

**Dependencies.** PR-γ.

**Blocks.** PR-ε, every op that produces a CUDA tensor.

**Estimated effort.** 3 hours. Most risk is missing a call site;
`checkInvariants` will catch it in debug.

**Gotchas.**

- The union layout changes; any cached `.zig-cache/` must be
  invalidated. `zig build --watch` is fine; stale caches from before
  the union change can cause mysterious Debug-only crashes.
- The `CpuStorage.owned` bool and `Tensor.owned` compat alias must
  stay in sync as they did in Stage 6.5. Ensure the PR's changes
  preserve this.

---

### PR-ε — `Tensor.toCuda` / `toCpu` + roundtrip test

**Status:** `[ ]` Pending

**Purpose.** Transfer a Tensor between CPU and CUDA storage,
preserving shape, strides, and bytes.

**Scope.**

1. Add methods on `Tensor`:
   - `pub fn toCuda(self: Tensor, ctx: *const CudaContext) LabError!Tensor`
     — allocates a DeviceBuffer sized for the full shape, HtoD
     copies `self.data`, builds a new Tensor with `Storage.cuda`,
     same `shape`/`strides`/`offset`.
   - `pub fn toCpu(self: Tensor, alloc: Allocator) LabError!Tensor`
     — allocates host `[]f32`, DtoH copies, builds CPU Tensor.
2. Preserve strides and offset. A non-contiguous view on CPU
   becomes a non-contiguous view on CUDA with the same stride
   layout. (No "make contiguous" pass.)
3. Test: build a `(2, 3, 4)` CPU tensor with known values, send to
   CUDA, bring back; compare bytes.
4. Test: build a CPU tensor, transpose2d it, send to CUDA, bring
   back; compare with CPU equivalent — confirm strides survive.
5. Test: attempt to call `tensor.data[0]` on a CUDA tensor — since
   `.data` is length zero, this should either be a bounds panic in
   debug or just iterate zero times; document the observed
   behaviour in the test.

**Files to modify.**

- `src/tensor/tensor.zig`

**API surface.**

```zig
pub fn toCuda(self: Tensor, ctx: *const CudaContext) LabError!Tensor { ... }
pub fn toCpu(self: Tensor, alloc: std.mem.Allocator) LabError!Tensor { ... }
```

**Acceptance criteria.**

- Round-trip test passes for both contiguous and transposed inputs.
- `checkInvariants` passes on both sides of the transfer.
- No leaks.

**Run commands.**

```
bash run_remote_example.sh "zig build test -Dcuda=true"
```

**Oracle strategy.** None. Byte-level copy.

**Commit template.**

```
stage(7e): Tensor.toCuda / toCpu

Host ↔ device transfer methods. Preserve shape/strides/offset
across transfer so a non-contiguous CPU view becomes the same
non-contiguous layout on CUDA. No auto-contiguify — callers can
request a contiguous copy explicitly later if needed.

Files

  src/tensor/tensor.zig
    toCuda(ctx):   alloc DeviceBuffer sized for storage.len(),
                   HtoD copy, return Tensor with Storage.cuda.
    toCpu(alloc):  alloc []f32, DtoH copy, return Tensor with
                   Storage.cpu.

Verification

  zig build test -Dcuda=true   → N+2 CUDA tests (contig + transpose
                                 roundtrips)
```

**Dependencies.** PR-δ.

**Blocks.** PR-η onward.

**Estimated effort.** 2 hours.

**Gotchas.**

- Copy the whole storage, not just `totalElements(shape)`. For a
  non-contiguous view the storage may be larger than the reachable
  element count; copying only the reachable count and reconstructing
  with the same strides would OOB-read on the CUDA side when an
  op stride-walks past the element count.
- Reuse `DeviceBuffer.alloc` + `copyFromHost`; do not reimplement.
- `toCpu` should set `Storage.cpu.owned = true` on the returned
  tensor. The CUDA source is untouched.

---

### PR-ζ — PTX loader + vector-add smoke kernel

**Status:** `[ ]` Pending

**Purpose.** Prove the end-to-end "nvcc → PTX → load → launch"
pipeline with the smallest possible kernel.

**Scope.**

1. Add `src/backend/cuda/kernels/vector_add.cu`:
   ```cuda
   extern "C" __global__
   void vector_add(const float* a, const float* b, float* c, unsigned int n) {
       unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
       if (i >= n) return;
       c[i] = a[i] + b[i];
   }
   ```
2. Update `build.zig`:
   - Add `"vector_add"` to `kernel_names`.
   - The existing `buildKernels` loop already invokes nvcc on each
     entry. Verify it produces `zig-out/ptx/vector_add.ptx`.
3. Add `src/backend/cuda/module.zig`:
   - `pub fn loadPtxFromFile(ctx, file_stem) LabError!CUmodule` —
     reads `zig-out/ptx/<stem>.ptx` into memory, calls
     `cuModuleLoadData`, caches in `ctx.ptx_modules`.
   - `pub fn getKernel(ctx, module_stem, kernel_name) LabError!CUfunction`.
4. Add `src/backend/cuda/launch.zig` (or fold into `module.zig`):
   - A helper `pub fn launch(ctx, fn, grid, block, shared, args)`
     that packs Zig values into a `?[]?*anyopaque` array for
     `cuLaunchKernel`.
5. Test: allocate two `[N]f32` tensors on CUDA, call vector_add,
   copy result back, compare with CPU sum.

**Files to create.**

- `src/backend/cuda/kernels/vector_add.cu` (new, ~15 LOC)
- `src/backend/cuda/module.zig` (new, ~200 LOC)

**Files to modify.**

- `build.zig` — add `"vector_add"` to `kernel_names`.
- `tests/integration_cuda.zig` — vector_add smoke test.

**Acceptance criteria.**

- `zig build kernels -Dcuda=true` emits `zig-out/ptx/vector_add.ptx`.
- Vector_add test: N=1024 random f32 inputs → output matches
  CPU `a+b` exactly (integer sums may match bit-exact for non-denormal
  inputs; use `expectClose` with `abs_tol=0` if f32 rounding is zero,
  else `abs_tol=1e-7`).
- `compute-sanitizer` clean on the test.

**Run commands.**

```
bash run_remote_example.sh "zig build kernels -Dcuda=true && ls zig-out/ptx/"
bash run_remote_example.sh "zig build test -Dcuda=true"
bash run_remote_example.sh 'find .zig-cache/o -name test -type f -executable -newer build.zig | head -1 | xargs -I{} compute-sanitizer --tool=memcheck {}'
```

**Oracle strategy.** None; CPU reference is trivial (pure `a+b`).

**Commit template.**

```
stage(7f): PTX loader + vector-add smoke kernel

Completes the end-to-end nvcc → PTX → load → launch pipeline with
the smallest possible kernel. Every later op kernel follows the
same pattern: .cu file, build.zig entry, module.load + getKernel
+ launch.

Files

  src/backend/cuda/kernels/vector_add.cu
    Two-line kernel: bounds-checked c[i] = a[i] + b[i].
  src/backend/cuda/module.zig
    loadPtxFromFile, getKernel, launch helper. Caches modules in
    CudaContext.ptx_modules by file stem.
  build.zig
    kernel_names gains "vector_add".
  tests/integration_cuda.zig
    Smoke test: 1024-element vector_add round-trips, result matches
    CPU reference.

Verification

  zig build kernels -Dcuda=true         → zig-out/ptx/vector_add.ptx
  zig build test -Dcuda=true            → N+1 tests pass
  compute-sanitizer --tool=memcheck ... → clean
```

**Dependencies.** PR-β (context), PR-γ (DeviceBuffer). Does not
need PR-δ/ε because this PR works with raw `DeviceBuffer`s, not
Tensors.

**Blocks.** PR-η and every later kernel.

**Estimated effort.** 3–4 hours (first kernel; subsequent kernels
are faster).

**Gotchas.**

- `extern "C"` is mandatory on the kernel function or the name
  gets C++-mangled and `cuModuleGetFunction` returns
  `CUDA_ERROR_NOT_FOUND`.
- `__global__` returns void; return types are passed via pointers.
- `cuLaunchKernel`'s `params` array wants pointers to each
  argument, not the arguments themselves. Pack via Zig
  `[N]?*anyopaque` with `@constCast(&arg)`.
- `blockDim` of 256 is a safe default for a 1D elementwise kernel.
  Grid size `(n + 255) / 256`.
- Read the `.ptx` file at runtime; do not try to embed via
  `@embedFile` (binary would have to be present at compile time
  of the Zig code, but our build produces `.ptx` at build time;
  the ordering is awkward).

---

### PR-η — Elementwise CUDA ops (same-shape) + parity

**Status:** `[ ]` Pending

**Purpose.** First real op family on GPU. Same-shape-only; no
broadcasting yet (that's PR-θ).

**Scope.**

1. Add `src/backend/cuda/kernels/elementwise.cu`:
   - Kernels: `add`, `sub`, `mul`, `div`, `neg`, `add_scalar`,
     `mul_scalar`. Same-shape only (output is the shape of the
     input, no broadcast).
2. Add `src/backend/cuda/dispatch.zig`:
   - `pub fn add(ctx, a, b, tape) !Tensor` etc., each checking
     device agreement, allocating a CUDA output Tensor, launching
     the kernel, recording on the tape.
3. Extend autograd: the backward for CUDA elementwise ops reuses
   the same `OpKind` values — `backward.zig` already handles them;
   the only change is that saved tensors come from the CUDA path
   via `tape.record()` with CUDA-storage snapshots. PR-ε's
   `takeOwnershipOfSaved` needs a CUDA variant that does
   `cuMemcpyDtoD_v2` into a tape-owned device buffer.
4. Parity tests: fixtures `add_2d`. Load CPU fixture, `toCuda`,
   run CUDA add, `toCpu`, `expectClose` within tolerance.

**Files to create.**

- `src/backend/cuda/kernels/elementwise.cu` (~100 LOC)
- `src/backend/cuda/dispatch.zig` (~400 LOC for this PR's ops)

**Files to modify.**

- `src/autograd/tape.zig` — extend `takeOwnershipOfSaved` to handle
  CUDA-storage tensors (DtoD copy into a tape-owned device buffer).
- `build.zig` — add `"elementwise"` to `kernel_names`.
- `tests/integration_cuda.zig` — add parity tests.

**Acceptance criteria.**

- Parity with oracle `add_2d` within `rel_tol=1e-4, abs_tol=1e-5`.
- Backward parity: after running backward, gradients match the
  oracle's `grad_input_0.ztlt` / `grad_input_1.ztlt` within
  tolerance.
- `compute-sanitizer` clean.

**Run commands.**

```
bash run_remote_example.sh "zig build test -Dcuda=true"
```

**Oracle strategy.** Reuse `add_2d` for add. No new fixtures.

**Commit template.**

```
stage(7g): CUDA elementwise ops (same-shape) + parity

First real GPU op family. Same-shape add/sub/mul/div/neg/
add_scalar/mul_scalar. No broadcasting in this PR.

Files

  src/backend/cuda/kernels/elementwise.cu
  src/backend/cuda/dispatch.zig          (7 ops)
  src/autograd/tape.zig                  (takeOwnershipOfSaved
                                          handles CUDA storage via
                                          DtoD copy)
  build.zig                              (kernel_names + "elementwise")
  tests/integration_cuda.zig             (parity test vs add_2d)

Parity vs oracle add_2d:
  forward max_abs_diff=<X>e-7  max_rel_err=<X>e-5
  backward both grads within tolerance
```

**Dependencies.** PR-ε, PR-ζ.

**Blocks.** PR-θ, PR-λ (softmax), PR-μ (embedding).

**Estimated effort.** 5–7 hours.

**Gotchas.**

- `neg` has `saved = .nothing` in our tape; CUDA version must
  record the same way.
- Scalar ops pass the scalar via the kernel params array as a
  `f32` value (not a pointer). Pack with
  `@constCast(@as(*const anyopaque, @ptrCast(&scalar)))`.
- Backward's `takeOwnershipOfSaved` for a CUDA tensor allocates a
  new `DeviceBuffer`, does DtoD, and builds a saved Tensor whose
  `storage.cuda` points at the new buffer. The tape's `kept_alive`
  list must learn to free CUDA buffers; consider a sibling list
  `kept_alive_cuda: ArrayList(DeviceBuffer)`.

---

### PR-θ — Broadcasting + scalar CUDA ops + parity

**Status:** `[ ]` Pending

**Purpose.** Extend elementwise CUDA ops to handle NumPy-style
broadcasting (rank-3 max) matching `src/tensor/ops/elementwise.zig`
behaviour.

**Scope.**

1. New kernel `elementwise_broadcast.cu` (or extend the existing
   file) that takes input shape/stride info and an output shape,
   computes per-axis indices from the flat output index, and reads
   each input with its stride.
2. Or (simpler, what we'll do): for each broadcast axis, compute
   per-element source offset on the host, launch a kernel that
   uses pre-computed offset arrays. Same model as our CPU code.
3. Parity: fixtures `add_broadcast_2d_1d`, `mul_broadcast`.

**Files to modify.**

- `src/backend/cuda/kernels/elementwise.cu`
- `src/backend/cuda/dispatch.zig`
- `tests/integration_cuda.zig`

**Acceptance criteria.**

- Parity with oracle `add_broadcast_2d_1d` and `mul_broadcast`
  forward and backward within tolerance.

**Estimated effort.** 4–5 hours.

**Gotchas.**

- Broadcast backward on CUDA: the "sum back to input shape" step
  needs a reduction kernel. Implementing `sum_to_shape` on CUDA
  requires reductions, which is PR-ι. Order dependency: PR-θ's
  broadcast backward can land only after PR-ι. **Alternative:**
  land PR-ι first, then PR-θ. We keep the order as specified —
  PR-θ ships forward only; its backward is stubbed with a "call
  PR-ι's reduce" pattern that's wired live in PR-ι. Document this
  temporary state clearly.

---

### PR-ι — CUDA reductions + parity

**Status:** `[ ]` Pending

**Purpose.** Sum, mean, sum-all along an axis. Foundation for
LayerNorm backward, loss reductions, `sum_to_shape`.

**Scope.**

1. `src/backend/cuda/kernels/reduce.cu` with kernels
   `sum_axis_rowwise`, `mean_axis_rowwise`, `sum_all_block`.
   Single-block reductions first — not highly optimised; we care
   about correctness over throughput here.
2. Dispatch functions mirroring `src/tensor/ops/reduce.zig`:
   `sum(axis)`, `mean(axis)`, `sumAll`.
3. Parity tests: `sum_axis_3d`, `mean_axis_3d`.
4. Wire PR-θ's broadcast backward to use `sum_to_shape` built on
   these primitives.

**Estimated effort.** 5–6 hours. Reductions are subtle (shared
memory, warp shuffles) even in the non-optimised form.

**Gotchas.**

- Single-block reductions cap at `n <= 1024` threads per block
  (hardware limit). For larger axes, either use a two-pass reduction
  or a multi-block reduction. Our current model shapes
  (`B=4, T=16, D=32`) have axis lengths of 32 at most, so single-
  block is fine. Document the limit.

---

### PR-κ — cuBLAS row-major GEMM + `docs/08_backends_cuda.md`

**Status:** `[ ]` Pending

**Purpose.** Matmul. The highest-risk single subsystem in the
project. Every step of the row-major → column-major translation
must be written down *before* any code is committed.

**Scope.**

1. **Write `docs/08_backends_cuda.md` first.** Minimum content:
   - Why matmul dominates transformer compute (FLOP count).
   - Why cuBLAS (vs. hand-rolled tile kernels).
   - Row-major vs column-major storage definitions.
   - Full derivation: given row-major `A (M×K)`, `B (K×N)`,
     `C (M×N)`, compute `C = A @ B` using cuBLAS column-major
     `sgemm`. Answer: call cuBLAS with `transa=N, transb=N` on
     operands `B` (as col-major `N×K`) and `A` (as col-major `K×M`),
     getting `C` (as col-major `N×M` which is row-major `M×N`). Be
     explicit about every dim. Include ASCII diagrams.
   - Derivation for batched matmul with strided batches.
   - PTX loading lifecycle.
   - Kernel-by-kernel catalog (filled in as PRs land).
2. Add `src/backend/cuda/gemm.zig`:
   - `pub fn matmul(ctx, a, b, tape) !Tensor` — row-major 2D.
   - `pub fn matmulBatch(ctx, a, b, tape) !Tensor` — row-major 3D
     via `cublasSgemmStridedBatched`.
   - Each wraps the cuBLAS call with the dimension swap derived in
     the doc, so the public API is row-major.
3. Parity: `matmul_2d`, `matmul_batch_3d`. Asymmetric M/N/K in
   both fixtures catches the most common bug (calling `sgemm` with
   the wrong transpose flags gives a result with correct shape but
   wrong values).

**Files to create.**

- `docs/08_backends_cuda.md` (~800 LOC of exposition)
- `src/backend/cuda/gemm.zig` (~300 LOC)

**Files to modify.**

- `src/backend/cuda/dispatch.zig` — wire matmul/matmulBatch.
- `tests/integration_cuda.zig` — matmul parity tests.

**Acceptance criteria.**

- Parity within `rel_tol=1e-4, abs_tol=5e-5` on both fixtures,
  forward and backward.
- `docs/08` is at least 500 lines and contains the derivation with
  ASCII diagrams *before* gemm.zig is merged. Review the chapter
  as if it were the only source of the derivation — it is.

**Estimated effort.** 8–10 hours. Most of it is the derivation +
docs writing + verifying the test on non-square matrices.

**Gotchas.**

- Classic cuBLAS trap: a 2×2 matrix multiplied by itself matches
  even with wrong transpose flags. **Never test GEMM on square
  matrices only.** The oracle's `matmul_2d` is `(4,5) @ (5,3) →
  (4,3)`; all three dims are different. Keep that property in any
  new matmul test.
- `cublasSgemm_v2` expects leading-dimension (ld) arguments. For
  row-major input interpreted as column-major, `ld = <row_count
  in col-major view>`, which is the number of *columns* in the
  original row-major matrix. Derive this in the doc; don't memorise.
- Strided batched matmul: `strideA`, `strideB`, `strideC` are the
  offset between matrix i and matrix i+1 in each operand. For
  contiguous `(B, M, K)` row-major input, `strideA = M*K`.

---

### PR-λ — Softmax + causal mask CUDA + parity

**Status:** `[ ]` Pending

**Purpose.** Row-wise softmax (the last-axis softmax used by
attention and output logits) and the causal mask kernel.

**Scope.**

1. `src/backend/cuda/kernels/softmax.cu`:
   - Block per row, shared memory for max + sum.
   - Max-subtraction pass, exp pass, normalise pass.
   - Fused log_softmax variant.
2. `src/backend/cuda/kernels/causal_mask.cu`:
   - Adds `-infinity` above the diagonal. Simple two-index kernel.
3. Parity: `softmax_3d_last_axis`, `log_softmax_3d`.

**Estimated effort.** 4–5 hours.

**Gotchas.**

- Softmax backward: use the saved output `y`, gradient is
  `dy - y * sum(y * dy, axis=last, keepdim=true)`. Requires a
  row-wise reduction; reuse PR-ι's primitives.
- The existing oracle fixture has `D=4` rows only; stress-test
  with a larger ad-hoc test (D=32, D=64) before calling parity
  done. A shared-memory sizing bug may only appear at larger D.

---

### PR-μ — Embedding + cross-entropy + AdamW CUDA + parity

**Status:** `[ ]` Pending

**Purpose.** The three remaining ops needed for end-to-end
training on GPU.

**Scope.**

1. `src/backend/cuda/kernels/embedding.cu`:
   - Forward: gather row `ids[i]` from weight into `out[i]`.
   - Backward: `atomicAdd` into `weight_grad[ids[i]]`. The atomic
     is mandatory because multiple ids may point at the same row.
2. `src/backend/cuda/kernels/ce_loss.cu`:
   - Fused `log_softmax + NLL + grad` w.r.t. logits. One kernel
     emits both the scalar loss and the gradient.
3. `src/backend/cuda/kernels/adamw.cu`:
   - Per-parameter AdamW step with bias correction and decoupled
     weight decay. One kernel handles one parameter; host loops over
     params. (A one-kernel-for-all-params variant is a Stage 9
     optimisation.)
4. Parity: `embedding_3d`, `cross_entropy_3d`. Add a new fixture
   `adamw_step` to `tools/oracle.py` (a single parameter, a known
   gradient, run one AdamW step in PyTorch, save the expected
   param and m/v).

**Estimated effort.** 8–10 hours.

**Gotchas.**

- `atomicAdd` on f32 is native on sm_89 — no compatibility macros
  needed. If we ever support older archs, we'd need a CAS-based
  implementation.
- Cross-entropy reduction: the oracle uses `reduction='mean'`. The
  CUDA kernel must divide the loss by `B*T` before returning. Get
  the division on the forward side, not backward.
- AdamW's `state` on CUDA: per-ParamId, `m` and `v` live as CUDA
  tensors. On `step()`, the kernel updates `param`, `m`, `v` in
  place. The PR-ζ `ParamId`-keyed HashMap works unchanged — it's
  the storage that moves, not the key.

---

### PR-ν — Full-model CUDA parity

**Status:** `[ ]` Pending

**Purpose.** Run a complete `TinyWordTransformer` forward and one
training step on GPU. Compare against the oracle's
`full_model_forward` fixture (forward) and the CPU gradient trace
(one step).

**Scope.**

1. `examples/08_cuda_vs_cpu.zig`:
   - Build a CPU and CUDA model, load oracle weights into both.
   - Forward on both; max abs diff of logits < `5e-4`.
   - Backward on both; per-param max abs diff of gradients < `1e-3`.
   - One optimizer step on both; per-param max abs diff of
     parameters < `2e-3`.
2. Update `docs/08_backends_cuda.md` with the parity trace and the
   per-op tolerance contribution.

**Estimated effort.** 6–8 hours. Most of it is orchestration and
diagnosing tolerance drift.

**Acceptance criteria.**

- All three tolerances above pass simultaneously.
- `compute-sanitizer --leak-check=full` clean on one full training
  step.

**Gotchas.**

- Tolerance drift is expected as ops compose. Individual op
  tolerances are `5e-5`; a 10-op sequence (embedding → LN → attn
  → LN → MLP → LN → lm_head → CE) multiplies by ~√10 ≈ 3, giving
  ~`2e-4` for forward. `5e-4` has comfortable headroom.
- Gradient tolerance is looser because backward multiplies
  tolerances through the chain rule. `1e-3` is generous.

---

### PR-ξ — Training speed benchmarks

**Status:** `[ ]` Pending

**Purpose.** Confirm CUDA is meaningfully faster than CPU. Goal
from `plan.md` §1.3: ≥30× speedup on `06_train_shakespeare` at
matched step count.

**Scope.**

1. `examples/09_cuda_benchmark.zig`:
   - Runs 100 steps on CPU, measures wall-clock.
   - Runs 100 steps on CUDA, measures wall-clock.
   - Prints speedup ratio.
2. Extend `docs/08` with a "Performance" appendix summarising the
   result.

**Acceptance criteria.**

- Speedup ≥ 30× on the Shakespeare config (`V=2000, D=32, T=16, B=4`).
- Final loss within 10% of CPU run at matched step count.

**Estimated effort.** 2–3 hours.

**Gotchas.**

- Warm-up: first few iterations include JIT / driver cache
  effects. Measure steps 10–100, discard 0–9.
- Synchronise before measuring. `cuStreamSynchronize` after each
  step for timing; the release build's async launch won't count
  properly otherwise.

---

## 7 — Cross-cutting concerns

### 7.1 Testing strategy

Every CUDA op has a parity test in `tests/integration_cuda.zig`:

1. Load the CPU oracle fixture (`tests/fixtures/<case>/*.ztlt`).
2. `toCuda` the inputs.
3. Run the op on CUDA.
4. `toCpu` the output.
5. `expectClose` against the expected output with per-op tolerance.

For backward, repeat on the gradient tensors.

The oracle's `full_model_forward` fixture is reused in PR-ν for an
end-to-end check. No new PyTorch runs needed unless a new fixture
is genuinely required (e.g., `adamw_step` in PR-μ).

### 7.2 Tolerance policy

| Op class | rel_tol | abs_tol |
|---|---|---|
| Pure transfer (toCuda/toCpu) | 0 | 0 (bit-exact) |
| Elementwise | 1e-4 | 1e-5 |
| Reductions | 1e-4 | 1e-5 |
| cuBLAS GEMM | 1e-4 | 5e-5 |
| Softmax / log_softmax | 1e-4 | 5e-5 |
| GELU | 1e-4 | 5e-5 |
| Cross-entropy | 1e-4 | 1e-4 |
| Embedding forward | 1e-4 | 1e-5 |
| Embedding backward (scatter-add) | 1e-4 | 1e-5 |
| Full-model forward (PR-ν) | 1e-3 | 5e-4 |
| Full-model backward (PR-ν) | 1e-3 | 1e-3 |
| AdamW step (PR-μ) | 1e-3 | 5e-4 |

Assertion style (same as oracle): `max_rel_err < rel_tol OR
max_abs_diff < abs_tol`.

### 7.3 Memory discipline

- Every `DeviceBuffer` is `defer buf.deinit()`'d.
- The tape's `kept_alive_cuda: ArrayList(DeviceBuffer)` mirrors
  the CPU list. Backward reads tape-owned device buffers.
- `Tape.deinit` frees all tape-owned CUDA buffers before context
  deinit. **Order matters:** buffers first, then context.

### 7.4 Error handling

- Every bindings call: `try bindings.check(...)`.
- Debug builds: sync after each kernel launch
  (`ctx.synchronize()`) to surface async errors immediately.
- Release builds: skip the per-launch sync; only sync when the
  user needs a result (e.g., `toCpu`).

### 7.5 Commit hygiene per PR

Each PR commit body should include:

- A paragraph describing what was added and why (architecture
  rationale, not just "added files X and Y").
- The acceptance-test output (parity numbers, exit codes, leak-
  checker state).
- An ASCII summary table of fixtures exercised and tolerances
  observed.

Example from an imagined PR-η commit:

```
Parity vs oracle add_2d:
  forward  max_abs_diff=3.8e-07  max_rel_err=1.1e-05  PASS
  grad a   max_abs_diff=0        max_rel_err=0        PASS  (sum grad is 1.0 exact)
  grad b   max_abs_diff=0        max_rel_err=0        PASS
```

---

## 8 — OpenCode workflow notes

Written so a next session doesn't re-learn the plumbing.

### 8.1 Shell tool behaviour

- The `bash` tool runs PowerShell 7 on Windows. Default timeout is
  120 seconds. For long-running commands (full training,
  `compute-sanitizer`), pass `timeout: N` explicitly.
- Stderr appears red in PowerShell but does not imply failure.
  Always check `$LASTEXITCODE` (PowerShell) or capture the bash
  exit code server-side (`R=$?; echo EXITCODE=$R`).
- The MSYS `bash` is available in PATH; invoke `.sh` scripts with
  `bash ./script.sh "..."`.

### 8.2 Plan mode

- A `<system-reminder>` tag containing "plan mode" means the
  session is read-only. File edits, commits, and shell mutations
  are blocked. Use `question` tool or surface the plan textually;
  wait for the user to exit plan mode.
- Never attempt to bypass plan mode. The `write`, `edit`, and
  shell tools will no-op or error.

### 8.3 Commit discipline

- Use `stage(7X): <scope>` for all Stage 7 commits. `X` is the PR
  letter (α → a, β → b, γ → c, δ → d, ε → e, ζ → f, η → g, θ → h,
  ι → i, κ → j, λ → k, μ → l, ν → m, ξ → n). Why the letter
  mapping? Because Greek letters don't round-trip cleanly through
  PowerShell here-strings on Windows. The letter alias in commit
  subjects is a workaround; the PR name in *docs* remains Greek.
- Commit body in the format used in `f9c1d3b`, `97b0aaa`, `1e3b540`:
  headline → blank line → multi-section body with rationale.
- Commit after every coherent unit of work. Never batch multiple
  PRs in one commit.
- Push after every commit unless the user says otherwise.

### 8.4 Remote invocation patterns

- Always pass a single-quoted string as the argument:
  ```
  bash ./run_remote_example.sh "zig build test -Dcuda=true"
  ```
  Multiple unquoted args get word-split by the script's `$@`.
- Use `| tail -N` or `| head -N` server-side to cut noise before
  my local shell sees it. The SSH login banner is ~30 lines of
  noise on every call.
- For file-level debugging (`hexdump`, `stat`), prefer running
  server-side via the wrapper instead of pulling the file back.
- For timings, use `time` prefix; capture `real` from stderr.

### 8.5 Long-running commands

- Training runs: pass `timeout: 600000` (10 minutes) or split
  into a backgrounded `nohup` job + status check.
- `compute-sanitizer` on a full step: can take 5–15 minutes.
  Use explicit timeout.
- Benchmark steps (PR-ξ): measure 100 steps with warm-up.

---

## 9 — Rollback / checkpoint signals

| Symptom | Meaning | Action |
|---|---|---|
| Parity test fails with `max_rel_err > 0.1` | Real semantic bug in the kernel or dispatch | Revert the PR. Re-derive the math. Write a failing unit test first, then reopen. |
| Parity fails in `1e-3` to `1e-1` range | Tolerance under-scoped | First check if tolerance is too tight; re-read Section 7.2. Only blame the kernel if tolerance is generous. |
| `compute-sanitizer` reports uninitialised reads | Kernel bounds bug | DO NOT merge. Fix and re-run. |
| `compute-sanitizer` reports race condition | Missing `__syncthreads` in a reduce/softmax kernel | Fix the barrier placement. |
| `dlsym` returns null for a symbol | Symbol renamed or removed in CUDA 13 | `nm -D /usr/local/cuda-13.2/lib64/libcublas.so.13 | grep <sym>` to confirm the name. Update bindings. |
| `cuModuleLoadData` → `CUDA_ERROR_NO_BINARY_FOR_GPU` (0x209) | `.ptx` sm mismatch | Confirm `build.zig` invokes nvcc with `-arch=sm_89`. |
| Training loss diverges on CUDA | AdamW or accumulation bug | Compare per-param gradient against CPU *before* optimizer step. If grads match but params drift, bug is in AdamW kernel. |
| `CUDA_ERROR_ILLEGAL_ADDRESS` after some iterations | Host/device pointer mixup or stale pointer after realloc | Run the failing op under `compute-sanitizer --tool=memcheck` for a stack trace. |
| Driver API returns `CUDA_ERROR_INVALID_CONTEXT` | Context destroyed or not current | Check `cuCtxSetCurrent` calls; we rely on one context per process. |

---

## 10 — Progress log

Appended to as PRs land. Format: `- [x] PR-X — <scope> (commit HASH, YYYY-MM-DD)`.

```
- [x] PR-α — CUDA dynamic loader smoke test (07bd274, 2026-05-10) — 3/3 pass on RTX 4060 Ti, CUDA 13.2 (driver 13020)
- [x] PR-β — CudaContext lifecycle (3c409a1, 2026-05-10) — 6/6 pass on RTX 4060 Ti, compute-sanitizer clean
- [x] PR-γ — DeviceBuffer alloc/free/copy (be977ea + fix 6b918e6, 2026-05-10) — 12/12 pass on RTX 4060 Ti, compute-sanitizer clean (0 leaks, 0 errors)
- [x] PR-δ — Tensor Storage.cuda variant (fe269ef, 2026-05-10) — 267/267 CPU + 14/14 CUDA pass on RTX 4060 Ti, compute-sanitizer clean (0 leaks, 0 errors)
- [x] PR-ε — Tensor.toCuda / toCpu (be6e0f8, 2026-05-10) — 267 CPU + 19/19 CUDA pass on RTX 4060 Ti, compute-sanitizer clean (0 leaks, 0 errors)
- [x] PR-ζ — PTX loader + vector-add smoke kernel (d2b458f + null-term fix 6c8b630 + log-level fix 3d73ef0, 2026-05-10) — 267 CPU + 24/24 CUDA pass on RTX 4060 Ti (includes two real kernel launches with bit-identical CPU parity on 1024 elements); compute-sanitizer: 0 memory leaks, 0 memory-access errors (1 API-level error is the deliberate `expectError` on cuModuleGetFunction with a non-existent symbol)
- [x] PR-η — Elementwise CUDA ops (849947c forward + cab742c tape/routing + 2ad2ba7 remaining 6 routings, 2026-05-10) — 267 CPU + 31 CUDA pass after PR-η.2; `ops_elementwise.{add,sub,mul,div,neg,addScalar,mulScalar}` and `ops_unary.neg` route CUDA inputs to GPU with tape recording; oracle add_2d forward parity matches PyTorch within rel_tol=1e-4, abs_tol=1e-5. compute-sanitizer: 0 leaks, 0 memory errors.
- [x] PR-θ — Broadcasting elementwise CUDA (809fd90, 2026-05-10) — rank-4 stride-aware broadcast kernels for add/sub/mul/div; `ops_elementwise.*` picks fast path vs broadcast based on shape+layout; oracle add_broadcast_2d_1d forward parity matches within tolerance; 34/34 CUDA pass.
- [x] PR-ι — CUDA reductions (43a2f1e, 2026-05-10) — sumAll (atomicAdd), sumAxis (row-major contiguous), bcast_copy + broadcastTo + sumToShape; device-aware zerosLike/onesLike + tape.backward seeds on the same device as loss; **full oracle add_2d forward+backward parity end-to-end on GPU**; 40/40 CUDA pass; compute-sanitizer 0 leaks, 0 memory errors.
- [x] PR-κ — cuBLAS GEMM + docs/08 (42e917f + fix 4e40453, 2026-05-10) — docs/08_backends_cuda.md (330 lines) with full row-major ↔ col-major derivation; gemm.zig wraps cublasSgemm_v2 and cublasSgemmStridedBatched using the operand-swap trick; ops_matmul.matmul routes CUDA inputs; **oracle matmul_2d forward+backward parity and matmul_batch_3d forward parity within rel_tol=1e-4, abs_tol=5e-5**; 45/45 CUDA tests pass; compute-sanitizer 0 leaks, 0 memory errors.
- [x] PR-λ — Softmax + log-softmax CUDA (52abdf7, 2026-05-10) — softmax_last/log_softmax_last kernels (block-per-row, 3-pass shared-memory reduction); ops_softmax routes CUDA inputs; oracle softmax_3d_last_axis forward+backward parity + log_softmax_3d forward parity + large-C (D=64) stress test; 49/49 CUDA tests pass; compute-sanitizer 0 leaks, 0 memory errors. Causal mask reuses ops_elementwise.add (no dedicated kernel needed).
- [ ] PR-θ — Broadcasting / scalar CUDA ops + parity
- [ ] PR-ι — CUDA reductions + parity
- [ ] PR-κ — cuBLAS row-major GEMM + docs/08
- [ ] PR-λ — Softmax + causal mask CUDA + parity
- [ ] PR-μ — Embedding + cross-entropy + AdamW CUDA + parity
- [ ] PR-ν — Full-model CUDA parity
- [ ] PR-ξ — Training speed benchmarks
```

---

## 11 — Reading list (appendix)

### Essential before starting

- `AGENTS.md` — contract, gotchas, hard rules.
- `plan.md` Sections 7.A–7.I — the original Stage 7 design.
- `docs/00_overview.md` — locked decisions D1–D14.
- This file (Section 0 → Section 10).

### Per-PR reading

| PR | Docs to read first |
|---|---|
| α | `plan.md` §7.A (bindings), Zig C-interop notes. |
| β | `plan.md` §7.B (context). |
| γ | `plan.md` §7.C (memory). |
| δ | `docs/02d_storage_and_views.md` (the CPU-only seam this PR extends). |
| ε | `docs/02c_tensor_invariants.md` (invariants that must hold across transfer). |
| ζ | `plan.md` §7.E intro (PTX lifecycle). |
| η | `docs/03_autograd.md` + `03c_saved_tensors.md` (how tape copies interact with CUDA storage). |
| θ | `src/tensor/ops/elementwise.zig` (CPU broadcast implementation — mirror it). |
| ι | `src/tensor/ops/reduce.zig` (CPU reduction for parity target). |
| κ | `plan.md` Appendix F (row-major/col-major derivation) — copy into `docs/08`. |
| λ | `docs/02_tensors.md` softmax stability section. |
| μ | `docs/07c_optimizer_state.md` (ParamId-keyed optimizer, unchanged). |
| ν | `docs/oracle.md` (parity methodology), `docs/04_nn.md` (model layout). |
| ξ | `plan.md` §1.3 point 5 (speedup target). |

### Skill library

`skills/modern-zig-0-16-tutor/` has references for:

- `references/09-c-interop-0-16.md` — `extern fn`, `callconv(.c)`,
  pointer conversions. Use for every `bindings.zig` entry.
- `references/17-zig-cuda-interop-notes.md` — CUDA-specific
  patterns.
- `references/08-build-system-0-16.md` — `addSystemCommand` for
  nvcc invocations.
- `references/15-code-review-checklist.md` — pre-commit pass.

---

## 12 — Known anti-patterns (Stage 7-specific)

Written as "don't do X" so the next session can recognise and
avoid each.

### 12.1 Don't write `@as([*]f32, @ptrFromInt(some_CUdeviceptr))[0..len]`

This constructs a host slice from a device pointer address. The
CPU then dereferences an address that doesn't map to its address
space. On Linux you get a segfault; on Windows you might read
unrelated memory. The Storage union makes this structurally hard;
don't engineer around it.

### 12.2 Don't `linkSystemLibrary("cuda")` or `"cublas")`

Locked by D1. If `build.zig` needs a CUDA toolkit library at
*build* time (for nvcc invocation), that's fine — but runtime
linking goes through dlopen only.

### 12.3 Don't use NVRTC

Locked by D3. All kernels are compiled offline with `nvcc -ptx` at
Zig-build time.

### 12.4 Don't write pure-Zig GPU code

Locked by D10. GPU kernels are `.cu` only. Zig only drives launches
and manages memory.

### 12.5 Don't copy from `zigrad`

Locked by its LGPL license. Architectural ideas only, attributed
in file headers.

### 12.6 Don't test GEMM on square matrices only

The row/column-major mix is the #1 bug in CUDA wrappers. Square
matrices mask the bug. Every GEMM test must have distinct M, N, K.

### 12.7 Don't synchronise on the wrong stream

Our single-stream model: everything runs on `ctx.stream`.
`cuCtxSynchronize` syncs the entire context; `cuStreamSynchronize`
syncs one stream. In our setting they're equivalent, but
`cuStreamSynchronize(ctx.stream)` is preferred — it doesn't
serialise unrelated future streams.

### 12.8 Don't forget `cudaGetLastError` (when using cudart)

Our design avoids cudart. If PR-ν or later decides to pull cudart
for any reason (e.g., managed memory), remember to check
`cudaGetLastError()` after every runtime API call — errors are
sticky otherwise and surface on the next unrelated call.

### 12.9 Don't leave `-Dcuda_fast_math=true` as a default

Locked by D14. The `--use_fast_math` nvcc flag trades correctness
for throughput; we build correctness first. A future Stage 9 PR
may add the option; it is not a default.

### 12.10 Don't make `CudaContext` a global singleton

Tempting but wrong. A test that fails to deinit leaves a dead
context around; the next test gets `CUDA_ERROR_INVALID_CONTEXT`.
Explicit per-test `init`/`deinit` catches leaks immediately.

---

## 13 — What "done" looks like for Stage 7

- [ ] All 14 PRs in Section 4 are `[x]`.
- [ ] `docs/08_backends_cuda.md` is ≥ 500 lines and covers:
  - Why matmul dominates compute.
  - Row-major / column-major derivation with diagrams.
  - Kernel-by-kernel walk-through for every `.cu` file.
  - PTX loading lifecycle.
  - Performance appendix from PR-ξ.
- [ ] `zig build test` remains 263/263 on both platforms.
- [ ] `zig build test -Dcuda=true` passes all CUDA parity tests.
- [ ] `zig build test-oracle` remains 14/14 on both platforms.
- [ ] `zig build run-example -Dexample=06_train_shakespeare -Dcuda=true`
      completes a full training run and is ≥30× faster than CPU.
- [ ] `zig build run-example -Dexample=08_cuda_vs_cpu -Dcuda=true`
      shows forward diff < 5e-4, gradient diff < 1e-3, param diff
      after one step < 2e-3.
- [ ] AGENTS.md progress table marks Stage 7 Done with a commit
      range.
- [ ] A single `stage(7): cuda backend complete` summary commit
      (or the last sub-PR commit) closes the stage.

---

## 14 — When you finish Stage 7

Immediately after PR-ξ lands and the Stage 7 box above is all
checked:

1. Update `AGENTS.md` progress table: Stage 7 → Done.
2. Update `SESSION_GUIDE.md` §3 — flip Stage 7 to complete.
3. Update this document's progress log (Section 10) with all
   commit hashes.
4. Tag the commit: `git tag stage-7-complete` (optional).
5. Consider whether Stage 8 (N-block refactor) or Stage 9 (docs
   finalisation) is the next priority. `plan.md` §Stage 8 and
   §Stage 9 are the canonical specs; draft a Stage 8 playbook
   following this document's format if tackling multi-block next.

---

## 15 — Questions to raise before starting

If any of these don't resolve quickly on your own, ask the user
with a crisp options-style question (max 4 options).

- Do we need `libcudart.so.13` for any op we haven't enumerated?
  (Baseline answer: no, Driver API alone suffices.)
- Does the user want `-Dcuda_fast_math=true` as an *optional* flag
  in PR-ξ after correctness parity passes? (Baseline: defer to a
  Stage 9 PR.)
- Multi-GPU: the remote has one GPU. Do we hard-code device 0 or
  expose `-Dcuda_device=N`? (Baseline: hard-code device 0 until
  Stage 9.)
- Does the user want per-kernel timing output in release builds,
  or only under `-Dprofile=true`? (Baseline: profile flag only.)

---

*End of Stage 7 Implementation Playbook.*
