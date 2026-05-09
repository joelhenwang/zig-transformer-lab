//! target: Zig 0.16.0 — illustrative, verify with `zig test` on 0.16.0.
//!
//! `Tensor2D` — minimal row-major 2D tensor skeleton for educational ML work.
//! Deliberately naive: single allocation, no tiling, no SIMD, no threads,
//! no autograd. Every line should be comprehensible to a learner.
//!
//! Included:
//!   - Tensor2D struct: init / deinit / at / atPtr / fill / shapeEq
//!   - matmulNaive: O(M*N*K) scalar triple loop. Pedagogical baseline.
//!   - 3 tests: shape-mismatch, identity multiplication, hand-computed 2x2.
//!
//! TODOs for subsequent optimization stages:
//!   - Tiling (M_TILE, N_TILE, K_TILE) for cache locality
//!   - SIMD inner loop with `@Vector(lanes, f32)`
//!   - Multi-threading via `std.Io.Group`
//!   - CUDA kernel offload (see references/17-zig-cuda-interop-notes.md)

const std = @import("std");
const testing = std.testing;

pub const Tensor2D = struct {
    data: []f32, // length = rows * cols; row-major
    rows: usize,
    cols: usize,

    /// Allocate a zero-initialized tensor. Caller owns; call
    /// `deinit(a)` with the same allocator.
    pub fn init(a: std.mem.Allocator, rows: usize, cols: usize) !Tensor2D {
        const data = try a.alloc(f32, rows * cols);
        errdefer a.free(data);

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

    pub fn shapeEq(self: Tensor2D, other: Tensor2D) bool {
        return self.rows == other.rows and self.cols == other.cols;
    }
};

/// out = a * b.  a: (M,K), b: (K,N), out: (M,N).
///
/// Deliberately naive: triple nested loop, no tiling, no SIMD. This is the
/// correctness baseline every optimized implementation must match (with
/// tolerance) in tests.
///
/// TODO(learner): add K-tiling — pull a K_TILE-length slice of both
/// operands into stack buffers to improve cache reuse.
/// TODO(learner): vectorize the inner reduction with `@Vector(8, f32)`.
/// TODO(learner): parallelize rows via `std.Io.Group`.
/// TODO(learner): add a CUDA kernel path behind a build option. See
/// references/17-zig-cuda-interop-notes.md.
pub fn matmulNaive(a: Tensor2D, b: Tensor2D, out: *Tensor2D) !void {
    if (a.cols != b.rows) return error.ShapeMismatch;
    if (out.rows != a.rows or out.cols != b.cols) return error.OutShapeMismatch;

    var r: usize = 0;
    while (r < a.rows) : (r += 1) {
        var cc: usize = 0;
        while (cc < b.cols) : (cc += 1) {
            var sum: f32 = 0;
            var k: usize = 0;
            while (k < a.cols) : (k += 1) {
                sum += a.at(r, k) * b.at(k, cc);
            }
            out.atPtr(r, cc).* = sum;
        }
    }
}

fn expectSlicesClose(comptime T: type, want: []const T, got: []const T, tol: T) !void {
    try testing.expectEqual(want.len, got.len);
    for (want, got, 0..) |w, g, i| {
        if (@abs(w - g) > tol) {
            std.debug.print("idx {d}: want {d}, got {d}\n", .{ i, w, g });
            return error.NotClose;
        }
    }
}

test "matmulNaive rejects mismatched inner dimension" {
    const a = testing.allocator;
    var A = try Tensor2D.init(a, 2, 3); defer A.deinit(a);
    var B = try Tensor2D.init(a, 4, 2); defer B.deinit(a); // 3 != 4
    var C = try Tensor2D.init(a, 2, 2); defer C.deinit(a);
    try testing.expectError(error.ShapeMismatch, matmulNaive(A, B, &C));
}

test "matmulNaive: A * I == A" {
    const a = testing.allocator;
    var A = try Tensor2D.init(a, 2, 2); defer A.deinit(a);
    var I = try Tensor2D.init(a, 2, 2); defer I.deinit(a);
    var C = try Tensor2D.init(a, 2, 2); defer C.deinit(a);

    A.atPtr(0, 0).* = 3; A.atPtr(0, 1).* = 5;
    A.atPtr(1, 0).* = 7; A.atPtr(1, 1).* = 11;

    I.atPtr(0, 0).* = 1; I.atPtr(1, 1).* = 1;

    try matmulNaive(A, I, &C);
    try expectSlicesClose(f32, A.data, C.data, 0.0);
}

test "matmulNaive: hand-computed 2x2 * 2x2" {
    const a = testing.allocator;
    var A = try Tensor2D.init(a, 2, 2); defer A.deinit(a);
    var B = try Tensor2D.init(a, 2, 2); defer B.deinit(a);
    var C = try Tensor2D.init(a, 2, 2); defer C.deinit(a);

    // A = [[1,2],[3,4]]; B = [[5,6],[7,8]];
    // A*B = [[19,22],[43,50]]
    A.atPtr(0, 0).* = 1; A.atPtr(0, 1).* = 2;
    A.atPtr(1, 0).* = 3; A.atPtr(1, 1).* = 4;
    B.atPtr(0, 0).* = 5; B.atPtr(0, 1).* = 6;
    B.atPtr(1, 0).* = 7; B.atPtr(1, 1).* = 8;

    try matmulNaive(A, B, &C);
    try expectSlicesClose(f32, &.{ 19, 22, 43, 50 }, C.data, 1e-5);
}
