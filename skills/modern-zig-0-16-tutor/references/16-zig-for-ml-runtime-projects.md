# 16 — Zig for ML / runtime projects (0.16.0)

This is the mentor's guide for the user's long-term goal: a tiny educational
deep-learning runtime in Zig, with optional CUDA interop later. Ref 17
covers the CUDA-specific side.

## Why Zig for this

- **Explicit memory ownership.** Every tensor you allocate you can see;
  every byte has a named owner. No framework magic hiding the cost.
- **Zero hidden allocation.** `std.mem.Allocator` is a parameter, not a
  global. In a training loop that matters enormously.
- **Static shape checking where possible.** `[N]f32`, `*[M]f32` encode
  sizes at the type level when they are comptime-known.
- **Predictable C ABI.** CUDA, BLAS, cuDNN, and custom kernels link in.
- **Compile-time generics.** Broadcasting / reduction rules can be
  encoded in `comptime` without a macro soup.
- **Tests are first-class.** Numerical regressions are silent in Python;
  `zig test` + tolerance comparators make them loud.

## Memory-ownership discipline

Rules every public ML API in this project should follow:

1. **Takes an `allocator`.** No hidden `c_allocator`, no `page_allocator`.
2. **Owns nothing it didn't allocate.** Stores only borrowed slices /
   references.
3. **Either pure or explicitly allocating.** Never both in one function.
4. **Documents shape invariants** in the doc comment.
5. **Fails loud.** Shape mismatches return `error.ShapeMismatch`.

## Tensor2D skeleton (see `templates/tensor2d-skeleton-0-16.zig`)

```zig
pub const Tensor2D = struct {
    data: []f32,    // row-major, length = rows * cols
    rows: usize,
    cols: usize,

    pub fn init(a: std.mem.Allocator, rows: usize, cols: usize) !Tensor2D {
        const data = try a.alloc(f32, rows * cols);
        @memset(data, 0);
        return .{ .data = data, .rows = rows, .cols = cols };
    }

    pub fn deinit(self: *Tensor2D, a: std.mem.Allocator) void {
        a.free(self.data);
        self.* = undefined;
    }

    pub inline fn at(self: Tensor2D, r: usize, c: usize) f32 {
        return self.data[r * self.cols + c];
    }

    pub inline fn atPtr(self: *Tensor2D, r: usize, c: usize) *f32 {
        return &self.data[r * self.cols + c];
    }

    pub fn fill(self: *Tensor2D, v: f32) void {
        @memset(self.data, v);
    }
};
```

**Notes.**
- Row-major by convention. Index formula `r * cols + c` is the contract.
- `@memset` is an intrinsic; no library call.
- Owner must hold the allocator used at `init` to call `deinit`.

## Naive matmul — the starting point

```zig
pub fn matmulNaive(a: Tensor2D, b: Tensor2D, out: *Tensor2D) !void {
    if (a.cols != b.rows) return error.ShapeMismatch;
    if (out.rows != a.rows or out.cols != b.cols) return error.OutShapeMismatch;

    var r: usize = 0;
    while (r < a.rows) : (r += 1) {
        var c: usize = 0;
        while (c < b.cols) : (c += 1) {
            var sum: f32 = 0;
            var k: usize = 0;
            while (k < a.cols) : (k += 1) {
                sum += a.at(r, k) * b.at(k, c);
            }
            out.atPtr(r, c).* = sum;
        }
    }
}
```

Deliberately naive: no tiling, no SIMD, no threads, no autograd. Pedagogical
**baseline**. Subsequent optimizations should be measured against its
output with `expectApproxEqAbs`.

## Finite-difference gradient check

For any scalar loss `L(x)` where `x: []f32`:

```zig
fn fdGrad(
    a: std.mem.Allocator,
    lossFn: anytype,
    x: []f32,
    h: f32,
) ![]f32 {
    const g = try a.alloc(f32, x.len);
    var i: usize = 0;
    while (i < x.len) : (i += 1) {
        const original = x[i];
        x[i] = original + h;
        const loss_plus = lossFn(x);
        x[i] = original - h;
        const loss_minus = lossFn(x);
        x[i] = original;
        g[i] = (loss_plus - loss_minus) / (2.0 * h);
    }
    return g;
}
```

Compare against your analytical gradient with
`expectApproxEqRel(analytical, fd, 1e-3)`. Any autograd-style library you
write must pass this gate.

## Tolerance comparisons

```zig
fn expectSlicesClose(comptime T: type, want: []const T, got: []const T, tol: T) !void {
    try std.testing.expectEqual(want.len, got.len);
    for (want, got, 0..) |w, g, i| {
        if (@abs(w - g) > tol) {
            std.debug.print("index {d}: want {d}, got {d}\n", .{ i, w, g });
            return error.NotClose;
        }
    }
}
```

`@abs` is an intrinsic. No imports.

## CPU performance path (before considering GPU)

1. Start with the naive triple loop above.
2. Use `@Vector(N, f32)` for the inner sum:
   ```zig
   const lanes = 8;
   var acc: @Vector(lanes, f32) = @splat(0);
   // ... SIMD accumulation
   ```
3. Use tiling (`TILE_M`, `TILE_N`, `TILE_K`) to improve cache reuse.
4. Use `std.Io.Group` for embarrassingly parallel batching.
5. Measure every step with `zig test` + `std.time.Timer`.

Only after you exhaust CPU wins should you move to GPU (ref 17).

## Row-major vs column-major

This project is **row-major** (C convention). Document it. A single
mis-transposition silently halves numerical accuracy; catch it with a
`matmulNaive` identity test.

## Data loading

- Prefer **memory-mapped files** via `std.Io.Dir` for large weight files.
- For streaming readers of formats like **safetensors** or **gguf**, use
  `std.Io.Reader` with `.limited(n)` to cap input size.
- Never trust tensor dimensions from untrusted files without a sanity check.

## Testing discipline for numerical code

- Every operator has: a **shape test**, a **zero-input test**, a **small
  hand-computed test**, a **round-trip test** (when applicable), and a
  **finite-difference check** (when gradients exist).
- Use `std.testing.expectApproxEqAbs` with an explicit tolerance.
- Run with `zig build test --test-timeout 2s` in CI.

## Common review findings (ML-specific)

- Hidden global allocator in a "simple" op. Flag.
- `@intCast` hiding a float truncation. Flag.
- `/ 0` possible from unchecked shape mismatch. Flag.
- Reshape via `@ptrCast` without an alignment check. Flag.
- Silent broadcast via `slice[0..n]` against differently-shaped input. Flag.

## Common mentor diagnostic questions

- "What's the CPU baseline for this op? Is it tested?"
- "What's the shape contract at the boundary?"
- "What's the memory budget per batch?"
- "Are we copying the weights or borrowing them?"
- "Can this be unit-tested without a file on disk?"

## Do not

- Depend on PyTorch / NumPy / TensorFlow. You may compare conceptually.
- Introduce `usingnamespace`-style re-exports for convenience.
- Hide allocations inside `at()` / `atPtr()`. Those must be cheap.
- Copy whole tensors on the hot path without profiling.

<!-- ~2.6k tokens -->
