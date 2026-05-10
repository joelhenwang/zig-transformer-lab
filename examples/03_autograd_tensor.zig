//!
//! 03_autograd_tensor.zig — Tensor-level autograd with gradient checking
//!
//! Builds a small computation graph with tensor operations:
//!   X = fixed input (2,3)
//!   W = learnable weight (3,4), requires_grad = true
//!   logits = X @ W                 (2,4)
//!   probs = softmax(logits)       (2,4)
//!   loss = mean(probs^2)          scalar
//!
//! Then runs tape.backward() and verifies gradients against finite differences.
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

    try w.print("=== Autograd Tensor: loss = mean(softmax(X @ W)^2) ===\n\n", .{});

    // --- Create tensors ---
    var X = try Tensor.init(allocator, Shape.init2D(2, 3));
    defer X.deinit(allocator);
    X.data[0] = 0.1;
    X.data[1] = 0.2;
    X.data[2] = 0.3;
    X.data[3] = 0.4;
    X.data[4] = 0.5;
    X.data[5] = 0.6;

    var W = try Tensor.init(allocator, Shape.init2D(3, 4));
    defer W.deinit(allocator);
    W.data[0] = 0.1;
    W.data[1] = 0.2;
    W.data[2] = 0.3;
    W.data[3] = 0.4;
    W.data[4] = 0.5;
    W.data[5] = 0.6;
    W.data[6] = 0.7;
    W.data[7] = 0.8;
    W.data[8] = 0.9;
    W.data[9] = 1.0;
    W.data[10] = 1.1;
    W.data[11] = 1.2;
    W.requires_grad = true;

    // --- Forward + backward ---
    var tape = Tape.init(allocator);
    defer tape.deinit();
    _ = try tape.trackLeaf(&W);

    var logits = try ztl.ops.matmul.matmul(allocator, X, W, &tape);
    var probs = try ztl.ops.softmax.softmax(allocator, logits, &tape);
    var squared = try ztl.ops.elementwise.mul(allocator, probs, probs, &tape);
    var loss = try ztl.ops.reduce.sumAll(allocator, squared, &tape);

    try w.print("Forward pass:\n", .{});
    try w.print("  X shape: (2, 3), W shape: (3, 4)\n", .{});
    try w.print("  logits shape: ({d}, {d})\n", .{ logits.shape.dims[0], logits.shape.dims[1] });
    try w.print("  probs shape:  ({d}, {d})\n", .{ probs.shape.dims[0], probs.shape.dims[1] });
    try w.print("  loss = {d:.6}\n\n", .{loss.data[0]});

    try tape.backward(&loss);

    // --- Print gradient ---
    if (W.grad) |g| {
        try w.print("dL/dW (first row): ", .{});
        for (0..4) |j| {
            try w.print("{d:.4} ", .{g.data[j]});
        }
        try w.print("\n", .{});
    }

    // --- Gradient check ---
    try w.print("\nRunning gradient check (finite differences)...\n", .{});

    var params = [_]*Tensor{&W};
    const result = try ztl.gradcheck.gradCheck(
        allocator,
        lossFn,
        &params,
        1e-3,
        1e-1,
        4,
    );

    try w.print("  Sampled {d} indices, {d} passed\n", .{ result.n_sampled, result.n_passed });
    try w.print("  Max relative error: {d:.6}\n", .{result.max_rel_error});
    if (result.passed) {
        try w.print("  GRADCHECK PASSED\n", .{});
    } else {
        try w.print("  GRADCHECK FAILED (tolerance exceeded)\n", .{});
    }

    // Clean up intermediates
    logits.deinit(allocator);
    probs.deinit(allocator);
    squared.deinit(allocator);
    loss.deinit(allocator);
}

/// The loss function for gradient checking.
/// Recreates X (it's not a parameter), uses the given W.
fn lossFn(allocator: std.mem.Allocator, params: []*Tensor, tape: *Tape) ztl.errors.LabError!*Tensor {
    var X = try Tensor.init(allocator, Shape.init2D(2, 3));
    defer X.deinit(allocator);
    X.data[0] = 0.1;
    X.data[1] = 0.2;
    X.data[2] = 0.3;
    X.data[3] = 0.4;
    X.data[4] = 0.5;
    X.data[5] = 0.6;

    const W_param = params[0];
    var logits = try ztl.ops.matmul.matmul(allocator, X, W_param.*, tape);
    defer logits.deinit(allocator);
    var probs = try ztl.ops.softmax.softmax(allocator, logits, tape);
    defer probs.deinit(allocator);
    var squared = try ztl.ops.elementwise.mul(allocator, probs, probs, tape);
    defer squared.deinit(allocator);
    const loss = try ztl.ops.reduce.sumAll(allocator, squared, tape);

    const loss_ptr = try allocator.create(Tensor);
    loss_ptr.* = loss;
    return loss_ptr;
}
