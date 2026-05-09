//! target: Zig 0.16.0 — illustrative, verify with `zig test` on 0.16.0.
//!
//! Minimal test file demonstrating:
//!   - leak-detecting `std.testing.allocator`
//!   - unmanaged `std.ArrayList` idiom: `.empty`, `append(a, v)`, `deinit(a)`
//!   - `expectEqual`, `expectEqualSlices`, `expectError`
//!   - `expectApproxEqAbs` for float equality
//!
//! Run with:
//!   zig test basic-test-0-16.zig

const std = @import("std");

fn addOne(n: i32) i32 { return n + 1; }

fn dot(a: []const f32, b: []const f32) !f32 {
    if (a.len != b.len) return error.ShapeMismatch;
    var acc: f32 = 0;
    for (a, b) |x, y| acc += x * y;
    return acc;
}

test "addOne adds one" {
    try std.testing.expectEqual(@as(i32, 42), addOne(41));
}

test "dot of 3-vectors" {
    const a: []const f32 = &.{ 1, 2, 3 };
    const b: []const f32 = &.{ 4, 5, 6 };
    try std.testing.expectApproxEqAbs(@as(f32, 32), try dot(a, b), 1e-6);
}

test "dot shape mismatch" {
    try std.testing.expectError(error.ShapeMismatch, dot(&.{1}, &.{ 1, 2 }));
}

test "no leak in a small ArrayList" {
    const a = std.testing.allocator;

    var list: std.ArrayList(u32) = .empty;
    defer list.deinit(a);

    try list.append(a, 10);
    try list.append(a, 20);
    try list.appendSlice(a, &.{ 30, 40 });

    try std.testing.expectEqualSlices(u32, &.{ 10, 20, 30, 40 }, list.items);
}
