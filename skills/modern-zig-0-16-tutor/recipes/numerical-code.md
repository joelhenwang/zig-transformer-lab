# Recipes: numerical code

## Recipe 14 — Create a small `Tensor2D` skeleton

**Problem.** Build a minimal 2D tensor abstraction that owns its memory and
supports indexing.

**Mentor explanation.** Row-major, single allocation, allocator passed
explicitly, shape invariants documented. This is the foundation of every
operator you'll write.

**Minimal snippet.**
```zig
const std = @import("std");

pub const Tensor2D = struct {
    data: []f32,
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
};
```

**Stale pattern to avoid.** Storing `allocator` inside the struct
(everything else in 0.16 is moving away from it) unless you have a good
reason. Keep it out; the caller who `init`ed is the caller who `deinit`s.

---

## Recipe 15 — Test row-major indexing

**Problem.** Confirm the index formula matches the layout you think it
does.

**Minimal snippet.**
```zig
test "row-major indexing of Tensor2D" {
    const a = std.testing.allocator;
    var t = try Tensor2D.init(a, 2, 3);
    defer t.deinit(a);

    // Write via atPtr, read via data directly
    t.atPtr(0, 0).* = 1;
    t.atPtr(0, 1).* = 2;
    t.atPtr(0, 2).* = 3;
    t.atPtr(1, 0).* = 4;
    t.atPtr(1, 1).* = 5;
    t.atPtr(1, 2).* = 6;

    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5, 6 }, t.data);

    // at() returns the same values
    try std.testing.expectEqual(@as(f32, 5), t.at(1, 1));
}
```

**Exercise.** Change the layout comment to say "column-major" without
changing the formula. Does any test still pass? What does that tell you
about documentation vs. code?

---

## Recipe 16 — Compare two `[]f32` with tolerance

**Problem.** Two floating-point buffers should be "approximately equal" —
but `==` lies. Write a helper and use it in tests.

**Mentor explanation.** Float equality is almost never exact. For unit
tests, use an **absolute** tolerance when values are bounded; a **relative**
tolerance when they span orders of magnitude.

**Minimal snippet.**
```zig
fn expectSlicesClose(comptime T: type, want: []const T, got: []const T, tol: T) !void {
    try std.testing.expectEqual(want.len, got.len);
    for (want, got, 0..) |w, g, i| {
        if (@abs(w - g) > tol) {
            std.debug.print("idx {d}: want {d}, got {d}\n", .{ i, w, g });
            return error.NotClose;
        }
    }
}

test "matmul against precomputed 2x2" {
    const a = std.testing.allocator;
    var A = try Tensor2D.init(a, 2, 2);  defer A.deinit(a);
    var B = try Tensor2D.init(a, 2, 2);  defer B.deinit(a);
    var C = try Tensor2D.init(a, 2, 2);  defer C.deinit(a);

    A.atPtr(0, 0).* = 1; A.atPtr(0, 1).* = 2;
    A.atPtr(1, 0).* = 3; A.atPtr(1, 1).* = 4;

    B.atPtr(0, 0).* = 5; B.atPtr(0, 1).* = 6;
    B.atPtr(1, 0).* = 7; B.atPtr(1, 1).* = 8;

    try matmulNaive(A, B, &C);

    // Hand-computed: [[19,22],[43,50]]
    try expectSlicesClose(f32, &.{ 19, 22, 43, 50 }, C.data, 1e-4);
}

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
```

**Exercise.** Add an identity-matrix test: `A * I == A`. It should pass
with `tol = 0` because identity multiplication is lossless for small
values. Verify.

<!-- ~1.6k tokens -->
