//!
//! 02_autograd_scalar.zig — Micrograd-style hello world for autograd
//!
//! Demonstrates the tape-based reverse-mode autograd on a simple scalar
//! computation graph: loss = (a * b + c)²
//!
//! Expected gradients:
//!   loss = (a*b + c)^2
//!   d/da = 2*(a*b + c) * b
//!   d/db = 2*(a*b + c) * a
//!   d/dc = 2*(a*b + c)
//!
//! With a=2, b=3, c=1:
//!   a*b = 6, loss = (6+1)^2 = 49
//!   d/da = 2*7*3 = 42
//!   d/db = 2*7*2 = 28
//!   d/dc = 2*7    = 14
//;

const std = @import("std");
const ztl = @import("zig_transformer_lab");
const Tensor = ztl.Tensor;
const Shape = ztl.shape.Shape;
const Tape = ztl.Tape;

pub fn main(init: std.process.Init) !void {
    var stderr_buf: [0]u8 = undefined;
    const locked_stderr = init.io.lockStderr(&stderr_buf, null) catch return;
    const w = &locked_stderr.file_writer.interface;
    const allocator = std.heap.page_allocator;

    // Create a fresh tape for this computation
    var tape = Tape.init(allocator);
    defer tape.deinit();

    // --- Create leaf tensors with requires_grad = true ---
    var a = try Tensor.init(allocator, Shape.init1D(1));
    defer a.deinit(allocator);
    a.cpuData()[0] = 2.0;
    a.requires_grad = true;
    _ = try tape.trackLeaf(&a);

    var b = try Tensor.init(allocator, Shape.init1D(1));
    defer b.deinit(allocator);
    b.cpuData()[0] = 3.0;
    b.requires_grad = true;
    _ = try tape.trackLeaf(&b);

    var c = try Tensor.init(allocator, Shape.init1D(1));
    defer c.deinit(allocator);
    c.cpuData()[0] = 1.0;
    c.requires_grad = true;
    _ = try tape.trackLeaf(&c);

    try w.print("=== Autograd Scalar: loss = (a*b + c)^2 ===\n", .{});
    try w.print("a = {d:.1}, b = {d:.1}, c = {d:.1}\n", .{ a.cpuData()[0], b.cpuData()[0], c.cpuData()[0] });

    // --- Forward pass: d = a * b, e = d + c, loss = e^2 ---
    var d = try ztl.ops.elementwise.mul(allocator, a, b, &tape);
    var e = try ztl.ops.elementwise.add(allocator, d, c, &tape);
    var loss = try ztl.ops.elementwise.mul(allocator, e, e, &tape);

    try w.print("d = a*b = {d:.1}\n", .{d.cpuData()[0]});
    try w.print("e = d+c = {d:.1}\n", .{e.cpuData()[0]});
    try w.print("loss = e^2 = {d:.1}\n\n", .{loss.cpuData()[0]});

    // --- Backward pass ---
    try tape.backward(&loss);

    // --- Print gradients ---
    try w.print("=== Gradients ===\n", .{});
    if (a.grad) |g| {
        try w.print("dL/da = {d:.1}  (expected 42.0)\n", .{g.cpuData()[0]});
    }
    if (b.grad) |g| {
        try w.print("dL/db = {d:.1}  (expected 28.0)\n", .{g.cpuData()[0]});
    }
    if (c.grad) |g| {
        try w.print("dL/dc = {d:.1}  (expected 14.0)\n", .{g.cpuData()[0]});
    }

    // --- Verify against hand-computed values ---
    const tol = 1e-4;
    var passed = true;
    if (a.grad) |g| {
        if (@abs(g.cpuData()[0] - 42.0) > tol) passed = false;
    } else passed = false;
    if (b.grad) |g| {
        if (@abs(g.cpuData()[0] - 28.0) > tol) passed = false;
    } else passed = false;
    if (c.grad) |g| {
        if (@abs(g.cpuData()[0] - 14.0) > tol) passed = false;
    } else passed = false;

    if (passed) {
        try w.print("\nAll gradients match expected values!\n", .{});
    } else {
        try w.print("\nSome gradients DO NOT match expected values.\n", .{});
    }

    // Clean up intermediate tensors
    d.deinit(allocator);
    e.deinit(allocator);
    loss.deinit(allocator);
}
