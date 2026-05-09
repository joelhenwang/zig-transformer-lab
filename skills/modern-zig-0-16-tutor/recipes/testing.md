# Recipes: testing

## TDD loop for a numeric helper

**Problem.** Practice writing the test **before** the implementation.

**Mentor explanation.** In Zig, `zig test foo.zig` runs all `test "..."`
blocks in that file. The loop is fast enough to be interactive.

**Step 1 — failing test.**
```zig
const std = @import("std");

test "dot product of two 4-vectors" {
    const a: []const f32 = &.{ 1, 2, 3, 4 };
    const b: []const f32 = &.{ 5, 6, 7, 8 };
    try std.testing.expectApproxEqAbs(@as(f32, 70), dot(a, b), 1e-6);
}
```

Run `zig test` — fails with `dot is not defined`.

**Step 2 — minimum implementation.**
```zig
fn dot(a: []const f32, b: []const f32) f32 {
    var acc: f32 = 0;
    for (a, b) |x, y| acc += x * y;
    return acc;
}
```

`zig test` — passes.

**Step 3 — edge case test.**
```zig
test "dot returns 0 for empty slices" {
    const empty: []const f32 = &.{};
    try std.testing.expectApproxEqAbs(@as(f32, 0), dot(empty, empty), 1e-6);
}
```

**Step 4 — shape-mismatch test. What should happen?**

This test exposes a design choice: should `dot` tolerate mismatched
lengths, or should it error? A numerical runtime should **error**. Change
the signature:

```zig
fn dot(a: []const f32, b: []const f32) !f32 {
    if (a.len != b.len) return error.ShapeMismatch;
    var acc: f32 = 0;
    for (a, b) |x, y| acc += x * y;
    return acc;
}

test "dot errors on mismatched lengths" {
    const a: []const f32 = &.{ 1, 2 };
    const b: []const f32 = &.{ 1, 2, 3 };
    try std.testing.expectError(error.ShapeMismatch, dot(a, b));
}
```

**Stale pattern to avoid.** A test that uses `==` on floats, or a "private"
allocator instead of `std.testing.allocator`.

**Exercise.** Write a `dotInto(out: *f32, a, b)` variant. What's the
trade-off in ergonomics vs allocation? When would you prefer each?

<!-- ~1.0k tokens -->
