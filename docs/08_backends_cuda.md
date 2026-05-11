# 08 — Backends and CUDA (Stage 7)

This chapter explains the CUDA backend: how we compile kernels
offline, load them at runtime, dispatch tensor ops to GPU, and —
most importantly — how we translate row-major tensor math into the
column-major world of cuBLAS. The row/column-major translation is
the single highest-risk subsystem in the whole library; every
other piece of Stage 7 is downstream of it.

Read `docs/02d_storage_and_views.md` first for the host-side
Storage/Tensor model.

---

## 1. Why matmul dominates transformer compute

Every transformer block is roughly 80% matmul FLOPs. In our
TinyWordTransformer (V=2000, D=32, T=16, B=4, one layer, one head):

| Op | Cost (FLOPs) | Fraction |
|---|---|---|
| `Q = X · W_q` + K, V analogues (3 linears, D×D per token) | `3 · B · T · D² ≈ 6.1 K` | ~40% |
| `scores = Q · K^T` | `B · T² · D ≈ 32 K` | ~20% |
| `A · V` | `B · T² · D ≈ 32 K` | ~20% |
| `W_o` projection | `B · T · D² ≈ 2 K` | ~1% |
| MLP (fc1, fc2, typically 4×D wide) | `8 · B · T · D² ≈ 16 K` | ~10% |
| LayerNorm, softmax, add, residual, mask | `O(B · T · D)` | <5% |
| `lm_head` (D → V) | `B · T · D · V ≈ 4 M` | dominates! |

`lm_head` is the giant matmul that projects the D=32 hidden
representation into the V=2000 vocabulary at every position. On a
real-scale transformer (D=768, V=50k), it's even more dominant.

The takeaway: if matmul is slow, everything is slow. If matmul is
correct, everything else can be validated against it. This is why
PR-κ ships the GEMM wrapper before any more kernels.

---

## 2. Why cuBLAS, not a hand-rolled matmul

A GPU matmul done well is a 1000-line tiling kernel that carefully
balances shared memory, register pressure, warp scheduling, and
tensor-core dispatch. NVIDIA has been tuning this kernel for
fifteen years across every generation of their hardware. Our
options:

| Option | Perf (%peak) | LOC | Pedagogical value |
|---|---|---|---|
| Naive 3-loop kernel | 1–5% | 30 | high but wrong-feeling |
| Shared-memory tile, 16×16 | 10–30% | 80 | high |
| CUTLASS template | 60–90% | 1000+ | medium (mostly setup) |
| cuBLAS (sgemm) | 80–95% | 0 kernel, ~50 host | medium |

For correctness, cuBLAS wins: it's the only option guaranteed to
be bit-exact across RTX / Ampere / Hopper without re-tuning. For
throughput, cuBLAS is competitive with hand-rolled CUTLASS up to
the biggest frontier-model shapes.

Our design decision (D9 in the project charter): **use cuBLAS for
every GEMM in Stage 7**. A hand-rolled tile kernel is a potential
Stage 9 teaching chapter, but correctness comes first.

---

## 3. Row-major vs column-major: the one trap everyone falls into

### 3.1 What "row-major" means

PyTorch, NumPy, our Tensor type — all row-major. A 3×2 matrix

```
[ 1 2 ]
[ 3 4 ]
[ 5 6 ]
```

stored row-major is the flat buffer `[1, 2, 3, 4, 5, 6]`: element
`(i, j)` lives at offset `i * stride[0] + j * stride[1]`, with
`strides = [2, 1]` for a contiguous layout.

### 3.2 What "column-major" means

Fortran, Julia, LAPACK, BLAS — all column-major. The same 3×2
matrix stored column-major is `[1, 3, 5, 2, 4, 6]`: element
`(i, j)` lives at offset `i + j * stride[1]`, with `strides = [1, 3]`
for a contiguous layout.

### 3.3 Why cuBLAS is column-major

Historical: cuBLAS follows the BLAS specification from 1979, which
predates C's prevalence. LAPACK is Fortran-first, and BLAS is its
foundation. To stay ABI-compatible, cuBLAS inherited column-major.

### 3.4 The one identity that makes everything work

If you reinterpret the same bytes row-major vs column-major, you
get transposes:

```
  A stored row-major as shape (M, K)
= A^T stored column-major as shape (K, M)
  ^^^  same bytes, different interpretation
```

**Every cuBLAS call we make relies on this identity.** Writing
down the derivation once is cheaper than re-deriving it at each
call site, so we do it here.

### 3.5 The derivation for 2-D matmul

Given row-major tensors `A : (M, K)` and `B : (K, N)`, we want
`C : (M, N) = A @ B` also stored row-major.

Treating the bytes as column-major:

```
  A_rm ⟷  A_cm = A^T : (K, M) in col-major
  B_rm ⟷  B_cm = B^T : (N, K) in col-major
  C_rm ⟷  C_cm = C^T : (N, M) in col-major
```

What is `C_cm` as a function of `A_cm` and `B_cm`?

```
C_cm = C^T
     = (A · B)^T               (transposing the product)
     = B^T · A^T                (the standard identity)
     = B_cm · A_cm              (substituting our row-major view)
```

So in cuBLAS's world, **we compute `C_cm = B_cm · A_cm`**. Notice
the operand swap: cuBLAS multiplies `B` first and `A` second.

Now the cuBLAS sgemm signature:

```
cublasSgemm_v2(
    handle,
    transa, transb,        // CUBLAS_OP_N or CUBLAS_OP_T per operand
    m, n, k,               // column-major dimensions
    alpha,
    A_arg, lda,            // first operand (the "A" in op(A) · op(B))
    B_arg, ldb,            // second operand
    beta,
    C_arg, ldc,
)
// Computes (col-major): C_arg = alpha · op(A_arg) · op(B_arg) + beta · C_arg
// op(A_arg) has shape (m, k); op(B_arg) has shape (k, n); C_arg has shape (m, n).
```

To produce `C_cm = B_cm · A_cm` where `B_cm : (N, K)` and
`A_cm : (K, M)`:

```
  m := N              // output rows (col-major view of C)
  n := M              // output cols (col-major view of C)
  k := K              // contraction
  A_arg := B_rm_ptr   // because B_cm is the first operand
  B_arg := A_rm_ptr
  transa := N, transb := N   // we want op(·) = identity
  lda := N            // leading-dim of B_cm in col-major = rows of B_cm = N
  ldb := K            // leading-dim of A_cm in col-major = rows of A_cm = K
  ldc := N            // leading-dim of C_cm in col-major = rows of C_cm = N
```

That is the one call every row-major→cuBLAS matmul makes. Write it
once, test it hard, never touch it again.

### 3.6 Visual confirmation

Let's check with a tiny concrete example. `A : (2, 3)` and
`B : (3, 2)`, giving `C : (2, 2)`.

```
A (row-major) = [ 1 2 3 ]        A_cm (col-major view of same bytes) : (3, 2) =
                [ 4 5 6 ]         [ 1 4 ]
                                  [ 2 5 ]
                                  [ 3 6 ]

B (row-major) = [ 1 0 ]          B_cm : (2, 3) =
                [ 0 1 ]           [ 1 0 0 ]
                [ 1 1 ]           [ 0 1 1 ]

C_expected = A @ B = [ 1+0+3    0+2+3 ]   = [ 4 5 ]
                     [ 4+0+6    0+5+6 ]     [ 10 11 ]
```

Call `sgemm(N, N, m=2, n=2, k=3, α=1, B_ptr, ld=2, A_ptr, ld=3, β=0, C_ptr, ld=2)`:

`C_cm = B_cm · A_cm : (2, 2) = [ 4 10 ]`
                               `[ 5 11 ]`

Interpret `C_cm`'s bytes row-major: `[4, 10, 5, 11]` reads as
`C_rm = [ [4, 5], [10, 11] ]`, matching the expected `A @ B`. ✓

### 3.7 The #1 GEMM bug: square test matrices

```
A = [ 1 2 ]   B = [ 1 2 ]   A @ B = [ 7  10 ]
    [ 3 4 ]       [ 3 4 ]           [ 15 22 ]

A^T @ B^T   = [ 13 20 ]      B @ A   = [ 7  10 ]
              [ 20 29 ]                 [ 15 22 ]     ← SAME!
```

When `M == N`, certain wrong transpositions yield the right answer
by coincidence. Our oracle fixtures deliberately use asymmetric
dimensions: `matmul_2d` is `(4, 5) @ (5, 3) → (4, 3)`. A wrong
transa/transb choice gives the wrong shape, not a wrong value,
catching the bug instantly.

**Rule: never test GEMM with `M == N`.**

---

## 4. Leading dimensions (`lda`, `ldb`, `ldc`)

In column-major storage, the leading dimension is the stride in
elements between successive **columns** — i.e., the number of
rows in each column. For a contiguous col-major `(K, M)` matrix,
`lda = K`.

Counter-intuitive corollary: when we pass a row-major matrix to
cuBLAS without copying, its leading dimension in the col-major
view is its row-major **column count**. For contiguous row-major
`(M, N)`, `ld = N`.

Rule of thumb that always works: **ld = the innermost row-major
stride**, which equals the number of columns in the row-major
matrix (for contiguous inputs).

If we ever pass a strided row-major view (e.g. a sub-slice), the
leading dimension becomes `row_major_stride[0]`, not the shape. For
our current scope (PR-κ) we reject non-contiguous inputs and only
handle the contiguous case.

---

## 5. Batched matmul

Attention needs a batch of small matmuls: `scores = Q @ K^T` where
`Q : (B, T, D)` and `K^T : (B, D, T)`. cuBLAS provides
`cublasSgemmStridedBatched` which takes strides between successive
matrices in each operand.

For row-major inputs `A : (B, M, K)` and `B : (B, K, N)` producing
`C : (B, M, N)`:

- Each batch of `A` is `M * K` contiguous elements.
- Each batch of `B` is `K * N` contiguous elements.
- Each batch of `C` is `M * N` contiguous elements.

Applying the row-major→col-major derivation per batch:

```
m, n, k   = N, M, K                   (same as non-batched)
A_arg     = B_rm_ptr  (first operand in cuBLAS order)
B_arg     = A_rm_ptr  (second operand)
strideA   = K * N     (elements between consecutive B_rm batches)
strideB   = M * K     (elements between consecutive A_rm batches)
strideC   = M * N
batchCount = B
```

Leading dimensions are unchanged from the non-batched case
(`lda = N`, `ldb = K`, `ldc = N`) because each matrix within a
batch has the same layout as in the 2-D case.

---

## 6. GEMM error budget

f32 GEMM on RTX 4060 Ti without `--use_fast_math`:

- FADD and FMUL are IEEE 754 compliant per-op.
- FMA accumulation inside each dot product is not bit-identical
  to the CPU's sequential adds — the accumulation tree differs.
- For `K = 5` (our smallest fixture) the discrepancy is at most
  a few ULPs per output element.
- For `K = 768` (real-scale transformer) the discrepancy grows to
  ~1e-5 relative error on typical weight / activation magnitudes.

Our tolerance policy for cuBLAS GEMM (playbook §7.2):

```
rel_tol = 1e-4
abs_tol = 5e-5
```

Any single-op test within this band is considered bit-consistent
with the CPU reference for pedagogical purposes.

---

## 7. PTX lifecycle (brief; full detail in PR-ζ chapter)

The Zig build step compiles every `*.cu` under
`src/backend/cuda/kernels/` to `zig-out/ptx/*.ptx` via nvcc:

```
nvcc -O3 -arch=sm_89 -ptx -Xcompiler -fPIC -o out.ptx in.cu
```

At runtime, `src/backend/cuda/module.zig` reads each `.ptx` into
memory, null-terminates it (cuBLAS-internal JIT requires this),
and hands it to `cuModuleLoadData`. The result is a `CUmodule`
cached by stem in `CudaContext.ptx_modules`. Kernels are resolved
by their `extern "C"` symbol via `cuModuleGetFunction`.

Our cuBLAS GEMM wrapper does NOT go through this lifecycle —
cuBLAS kernels live in `libcublas.so.13` and are resolved by
`cublasSgemm_v2` / `cublasSgemmStridedBatched` via the dlsym'd
bindings. The PTX path is for our own kernels (elementwise,
reductions, softmax, etc.).

---

## 8. Kernel catalog as of PR-κ

| Kernel | File | Purpose |
|---|---|---|
| `vector_add` | `vector_add.cu` | Smoke test (PR-ζ) |
| `elw_add/sub/mul/div/neg/add_scalar/mul_scalar` | `elementwise.cu` | Same-shape forward (PR-η) |
| `elw_broadcast_{add,sub,mul,div}` | `elementwise.cu` | Rank-4 stride-aware (PR-θ) |
| `reduce_sum_all` | `reduce.cu` | atomicAdd whole-tensor sum (PR-ι) |
| `reduce_sum_axis` | `reduce.cu` | Row-major axis sum (PR-ι) |
| `bcast_copy` | `reduce.cu` | Rank-4 stride-aware gather (PR-ι) |
| cuBLAS `sgemm_v2` | `libcublas.so.13` | Row-major 2-D GEMM via swap trick (PR-κ) |
| cuBLAS `sgemmStridedBatched` | `libcublas.so.13` | Row-major 3-D batched GEMM (PR-κ) |

Every kernel beginning with `elw_` or `reduce_` or `bcast_` is
authored by this project under the Stage 7 policy (bounds check at
the top, `extern "C"`, compiled offline by nvcc). cuBLAS kernels
are third-party but count as a hard dependency per D1.

---

## 9. Host-side dispatch flow

For a row-major 2-D matmul `C = A @ B`:

1. `ops_elementwise` / `ops_matmul` detects that `a.device == .cuda`
   at the top of the op and routes to
   `backend.cuda.gemm.matmul(a, b)`.
2. `gemm.matmul` validates: both 2-D, both contiguous, same device,
   dims compatible.
3. Allocates a fresh `DeviceBuffer` for the output on `a`'s context.
4. Calls `cublasSgemm_v2` with the row-major→col-major argument
   swap (see §3.5).
5. Packages the output as a Tensor with contiguous strides and
   returns.
6. `ops_matmul.matmul` records a tape node if a tape is provided,
   using the existing `.matmul` OpKind — backward code runs
   identically on CPU and CUDA because the backward formulas
   reduce to more matmuls that also route through device
   dispatch.

For batched 3-D matmul the only additions are stride computation
and `cublasSgemmStridedBatched` in place of `cublasSgemm_v2`.

---

## 10. What's NOT in PR-κ

- Transposed operand GEMM (`a @ b^T` via `transa=T`) — deferred;
  the backward chain for attention currently materialises
  transposes as views and then calls matmul.
- Mixed precision (tf32, fp16) — Stage 9.
- Triangular / symmetric GEMM variants — not needed by
  transformers.
- `cublasLt` tensor-core-optimised paths — Stage 9 performance
  work.
- Non-contiguous input GEMM — rejected with `error.InvalidLayout`.

---

## 11. Sanity tests for a GEMM wrapper

When writing your own row-major GEMM over a col-major BLAS, test
every one of these at minimum:

1. `(4, 5) @ (5, 3) → (4, 3)` — asymmetric, catches transpose bugs.
2. `(1, K) @ (K, 1) → (1, 1)` — scalar output, catches dim confusion.
3. `(M, K) @ (K, N) → (M, N)` with `M ≠ N` at several sizes.
4. Identity matmul `(M, K) @ I_K = A` — trivial but verifies the
   multiplication is happening (not a memcpy).
5. Backward: `dL/dA = dL/dC @ B^T`, `dL/dB = A^T @ dL/dC`. Hidden
   transposes; if the forward is wrong the backward cascades.

Our oracle fixture covers (1), (3), and (5) via the `matmul_2d`
case. The remaining sanity cases are small crafted tests in the
integration harness.

---

## 12. From Dispatching an Op on CPU to on CUDA

A question new readers ask first: "when I call `add(a, b)`, how does
it know to run on the right device?" The answer lives in a handful
of switch statements inside the op dispatchers. This section walks
one op — `add` — through both backends side by side so you can see
the whole path in one glance.

### 12.1 The entry point

`src/tensor/ops/elementwise.zig` exposes the public `add`
entrypoint. Here's the sequence, abbreviated:

```zig
pub fn add(alloc: Allocator, a: Tensor, b: Tensor, tape: ?*Tape) !Tensor {
    // 1. Validate compatible shapes (broadcast-aware).
    // 2. Pick a backend based on a.device.
    // 3. Record the op in the tape (device-agnostic).
    // 4. Return the result.

    return switch (a.device) {
        .cpu => cpuAdd(alloc, a, b, tape),
        .cuda => cudaAdd(alloc, a, b, tape),
    };
}
```

The `device` field on `Tensor` is the *single* piece of runtime
state that decides which kernel runs. A tensor whose parameters
were fed through `moveToCuda` has `device = .cuda`; everything
downstream follows.

### 12.2 The CPU branch

The CPU implementation loops in Zig:

```zig
fn cpuAdd(...) !Tensor {
    var out = try Tensor.init(alloc, a.shape);
    for (0..a.data.len) |i| out.data[i] = a.data[i] + b.data[i];
    // ... tape record with SavedData capturing a and b snapshots ...
    return out;
}
```

Simple, obvious, slow for large tensors.

### 12.3 The CUDA branch

The CUDA implementation does three things: allocates a device
buffer, launches a kernel, records on the tape:

```zig
fn cudaAdd(...) !Tensor {
    const ctx = a.storage.cuda.ctx;           // pulled from the input
    var out_dev = try DeviceBuffer.alloc(ctx, a.storage.cuda.len);

    // Grid: ceil(N / block_dim). Block: 256 threads.
    const n = a.storage.cuda.len;
    const block: u32 = 256;
    const grid: u32 = (n + block - 1) / block;
    try module.launchKernel(ctx, "elementwise", "elw_add",
        .{ a.ptr(), b.ptr(), out_dev.ptr, @as(u32, n) },
        .{ grid, 1, 1 }, .{ block, 1, 1 });

    var out = Tensor{ .device = .cuda, .storage = .{ .cuda = out_dev }, ... };
    // ... tape record (same SavedData; snapshots work for CUDA too) ...
    return out;
}
```

Note the symmetry: both branches produce a `Tensor` whose backing
storage is the correct device variant. The tape record downstream
is identical for both — the `.add` OpKind's backward closure
dispatches through the same elementwise path on whichever device
the gradient happens to live on.

### 12.4 Why this design scales

Every op we've shipped — elementwise, reduce, softmax, matmul,
embedding, AdamW step, CE loss — follows the same three-line shell:

```zig
return switch (a.device) {
    .cpu => cpuOp(...),
    .cuda => cudaOp(...),
};
```

New devices would slot in with one new branch and one new kernel
implementation. The `Tensor` / `Tape` layer above doesn't change.
This is the zig-transformer-lab version of what PyTorch calls its
"dispatcher" — see `docs/10_pytorch_parallels.md` §6.

---

## 13. Why dlopen, Not Link-Time Linking

Stage 7 made a deliberate choice: the library calls into CUDA and
cuBLAS via `dlopen` + `dlsym` at runtime, not via a link-time `-lcuda`
/ `-lcublas`. The reasoning is worth unpacking because it differs
from what a first-time CUDA user might try.

### 13.1 Portability

A `-lcuda` binary only loads on a machine whose `libcuda.so.1` was
installed by the NVIDIA driver. If you distribute a binary and the
user's machine either has no NVIDIA GPU or has an older driver, the
binary fails at *load time* with `error while loading shared
libraries: libcuda.so.1: cannot open`. Before your `main` runs.

With dlopen: the binary loads on any machine. It probes for CUDA at
`CudaContext.init`; if the libraries aren't there, the error surfaces
inside Zig's `error.CudaError` path and your program decides what to
do. That's how this project's Windows tests work (CUDA tests
`SkipZigTest`) and how the CPU-only binary still loads on machines
without the NVIDIA driver.

### 13.2 CI build time

The CI image for Zig does not ship with CUDA. Link-time linking
would require installing the CUDA toolkit on the CI runner — adds
minutes to every build. Runtime loading means `zig build` builds
the binary with no CUDA dependency at link time; CUDA tests are
gated by `-Dcuda=true` which also wires the nvcc step for compiling
`.cu → .ptx`.

### 13.3 Runtime GPU discovery

A program using CUDA should also handle "no GPU present" and "GPU
present but driver too old" gracefully. Link-time linking allows
the second but not the first; dlopen allows both. Our
`CudaContext.init` propagates a meaningful error ("no CUDA device
found") without the binary crashing on load.

### 13.4 What we lose

Dynamic loading means every cuBLAS and cuDriver entry point must
be looked up by name. `src/backend/cuda/bindings.zig` keeps this
in one place — about 30 function pointers, resolved once on first
use of the context. After that, calls are the same speed as
link-time (a single indirect call per invocation).

Symbol-version drift is the one risk: if NVIDIA renames a symbol
in a future driver (which has happened once between CUDA 11 and
12), our `dlsym` fails at load-time and we surface a CudaError.
That beats silent binary incompatibility.

### 13.5 Contrast with the cudart path

Some CUDA codebases link against `libcudart.so` (the "CUDA runtime
library"). cudart bundles a high-level API plus some auto-lifetime
state (default streams, device context management). We **don't**
use it — we use the lower-level Driver API directly via `libcuda`.
Rationale:

- cudart adds one more thing to install on the target machine.
- cudart hides some of the resource management we want the reader
  to see explicitly (context creation, module loading).
- cudart's auto-magic behaviour fights the "no hidden globals"
  policy (P2).

All our CUDA code talks to the Driver API and to cuBLAS — two
libraries, both dlopened.

---

## 14. The Row-Major to Column-Major Worksheet

§3–§4 derived the operand-swap identity symbolically. This section
works two concrete numerical examples by hand so you can verify the
trick on paper before trusting it in code. If the algebra felt
abstract, these examples make it concrete.

### 14.1 Example 1: small `(2, 3) @ (3, 2) = (2, 2)`

Row-major inputs:

```
A_rm = [ 1 2 3 ]        B_rm = [ 7  8  ]
       [ 4 5 6 ]               [ 9  10 ]
                                [ 11 12 ]

A_rm memory: 1 2 3 4 5 6        (row-major: row 0, then row 1)
B_rm memory: 7 8 9 10 11 12     (row-major: row 0, then row 1, then row 2)
```

Expected output `C_rm = A_rm @ B_rm`:

```
C[0][0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
C[0][1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
C[1][0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
C[1][1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154

C_rm = [ 58  64  ]        memory: 58 64 139 154
       [ 139 154 ]
```

Now reinterpret the same bytes as column-major:

```
A_rm memory 1 2 3 4 5 6 reinterpreted as 2-row col-major:
A_cm is (3, 2):
  col 0: [1 2 3]^T        col 1: [4 5 6]^T

Equivalently: A_cm = A_rm^T. Same bytes, different reader.
```

Applying the identity `A_rm ⟷ A^T_cm`:

- `A_rm` with shape (2, 3) is the same bytes as `A^T_cm` with shape
  (3, 2).
- `B_rm` with shape (3, 2) is the same bytes as `B^T_cm` with shape
  (2, 3).

So `A_rm @ B_rm` in row-major corresponds to
`A^T_cm @ B^T_cm = (B_cm @ A_cm)^T` in col-major. cuBLAS computes
`C_cm = B_cm @ A_cm`; reading the result bytes as row-major gives
us the C we wanted.

cuBLAS call:
```
cublasSgemm_v2(handle, N, N,
    n=2, m=2, k=3,        // col-major dims: rows of B_cm, cols of A_cm, contraction
    1.0, B_rm_ptr, ldb=2,  // ldb = col-major rows of B_cm = 2
        A_rm_ptr, lda=3,  // lda = col-major rows of A_cm = 3
    0.0, C_ptr,    ldc=2); // ldc = col-major rows of C_cm = 2
```

The bytes written at `C_ptr` are `58 64 139 154`. Read as row-major
`(2, 2)`, that's exactly the `C_rm` we computed. 

### 14.2 Example 2: asymmetric `(4, 5) @ (5, 3) = (4, 3)`

This is the size used by `tests/integration_cuda.zig`'s "cuda gemm
matmul: asymmetric (4,5) @ (5,3) hand-computed reference" test.
Don't work every cell by hand; the point is the parameter mapping:

```
Row-major problem:
  C_rm = A_rm @ B_rm
  A_rm: (M=4, K=5)
  B_rm: (K=5, N=3)
  C_rm: (M=4, N=3)

Col-major cuBLAS call:
  operand-A = B_rm_ptr (shape treated as col-major (N, K) = (3, 5))
  operand-B = A_rm_ptr (shape treated as col-major (K, M) = (5, 4))
  m_cublas = N = 3    (rows of col-major op-A, rows of col-major C)
  n_cublas = M = 4    (cols of col-major op-B, cols of col-major C)
  k_cublas = K = 5    (contraction)
  lda       = N = 3    (col-major rows of op-A)
  ldb       = K = 5    (col-major rows of op-B)
  ldc       = N = 3    (col-major rows of C)
```

Note the *swap*: `m_cublas = N`, `n_cublas = M`. This is where the
identity bites you if you transcribe the row-major shapes directly.
The test in `src/backend/cuda/gemm.zig` has this exact pattern and
compares against a hand-computed reference.

### 14.3 Leading-dimension reference card

For any row-major `A @ B = C` with shapes `A: (M, K)`, `B: (K, N)`,
`C: (M, N)`:

| cuBLAS arg | Value | Meaning |
|---|---|---|
| `transA` | `N` | do not transpose the first-operand pointer |
| `transB` | `N` | do not transpose the second-operand pointer |
| `m` | `N` | rows of col-major `C` = cols of row-major `C` |
| `n` | `M` | cols of col-major `C` = rows of row-major `C` |
| `k` | `K` | contraction extent (unchanged) |
| A pointer | `B_rm` | the row-major **second** operand |
| `lda` | `N` | row-major col-count of the B operand |
| B pointer | `A_rm` | the row-major **first** operand |
| `ldb` | `K` | row-major col-count of the A operand |
| C pointer | `C_rm` | output buffer (same bytes for both interpretations) |
| `ldc` | `N` | row-major col-count of the C operand |

Print this card and tape it next to your monitor for the week you're
writing CUDA dispatch code. Every other row of the table is the one
that trips people up on their first day.

---

## 15. Common Mistakes

- **Mixing row-major shapes into cuBLAS arguments directly.** If
  `cublasSgemm_v2` gives you wrong bytes, stop and re-derive
  §14.3. Most bugs are here.
- **Forgetting `ctx.synchronize()` before reading host memory.**
  The CUDA default stream is asynchronous; a DtoH read without
  sync returns stale data. Our `toCpu` is blocking, which is why
  it's the safe pattern.
- **Passing a non-contiguous operand to cuBLAS.** A transpose view
  looks fine but its strides are not what cuBLAS expects. Either
  materialise with `broadcastTo` or use a transpose flag on the
  cuBLAS call itself.
- **Freeing a DeviceBuffer after destroying its context.** The
  `cuMemFree_v2` call is a no-op against a destroyed context and
  leaks device memory silently. `Trainer.deinit` frees model
  parameters before `ctx.deinit()`.
- **Loading the wrong PTX module.** `loadPtxFromFile(ctx, io, "elementwise")`
  is keyed by file stem. Typo the stem and `cuModuleLoadData`
  succeeds (the file exists) but the kernel you want isn't there.
  Manifests as `cuda driver error 500: named symbol not found` at
  first launch.
- **Assuming PTX compiled on one driver works on an older one.**
  `-arch=sm_89` (Ada Lovelace) binaries are forward-compatible
  across driver versions but not backward — running an sm_89 PTX
  on an sm_75 GPU fails at `cuModuleLoadData` with `invalid PTX`.

---

## 16. Exercises

**Exercise 1.** Given `A_rm` of shape `(3, 4)` and `B_rm` of shape
`(4, 2)`, what are the values passed to `cublasSgemm_v2` for
`m`, `n`, `k`, `lda`, `ldb`, `ldc`?

<details><summary>Solution</summary>

`m = 2` (= `N`), `n = 3` (= `M`), `k = 4` (= `K`), `lda = 2` (row-major
col-count of B), `ldb = 4` (row-major col-count of A), `ldc = 2`
(row-major col-count of C). Both `transA` and `transB` are `N`.

</details>

**Exercise 2.** You want to compute `C = A @ B^T` where `A: (M, K)`
and `B^T: (K, N)`, i.e. `B: (N, K)`. Can you get cuBLAS to do this
without materialising `B^T`? If so, how do the arguments change?

<details><summary>Solution</summary>

Yes. Set `transA = T` in the cuBLAS call. The effective op becomes
`C_cm = op(A_cuB) @ op(B_cuB)` where the first operand slot still
receives `B_rm_ptr` but is transposed before multiply. The dims and
leading dims change accordingly — see NVIDIA's cublas docs for the
detailed mapping. Our library currently *does* materialise
transposes in `backwardMatmul` because it was simpler for Stage 7;
switching to transpose flags is a clean Stage 9+ optimisation.

</details>

**Exercise 3.** Consider this PTX function signature in
`unary.cu`:

```cuda
extern "C" __global__ void unary_gelu_exact(
    const float* x, float* y, uint32_t n);
```

A caller uses `cuLaunchKernel` with four argument pointers. What's
the bug?

<details><summary>Solution</summary>

The kernel takes three arguments. If the caller passes four arg
pointers to `cuLaunchKernel`, the driver silently reads past the
kernel's expected parameter list. Depending on alignment it may
crash (best case) or proceed with a garbage `n` (worst case, silent
wrong output). Count your kernel arguments exactly; mismatches are
invisible at compile time because the launch API takes opaque
pointers.

</details>

---



## 17. Summary

| Concept | Implementation |
|---|---|
| Row-major → col-major via same bytes | §3 identity: `A_rm ⟷ A^T_cm` |
| Operand swap in sgemm | `A_arg := B_rm_ptr`, `B_arg := A_rm_ptr` |
| Dimensions | `m, n, k = N_out, M_out, K_contraction` |
| Leading dims | `lda = N`, `ldb = K`, `ldc = N` (all row-major column counts) |
| Batched strides | `strideA = K*N`, `strideB = M*K`, `strideC = M*N` |
| Test shape rule | Never square; always asymmetric M, N, K |

`src/backend/cuda/gemm.zig` encodes this exactly once. Every op in
the library that eventually calls a GEMM — matmul, attention QK,
attention AV, linear layers — routes through that one file.
