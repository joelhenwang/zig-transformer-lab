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

## 12. Summary

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
