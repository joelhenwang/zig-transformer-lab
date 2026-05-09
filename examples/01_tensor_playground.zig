//!
//! examples/01_tensor_playground.zig — Interactive tensor API tour
//!
//! Purpose:
//!   A runnable example that walks through every major operation in the
//!   CPU tensor foundation (Stage 2). Run it with:
//!     zig build run-example -Dexample=01_tensor_playground
//!
//!   Each section prints a banner, builds some tensors, applies an operation,
//!   and shows the result.  This is the "hello world" of the tensor layer —
//!   read it top-to-bottom to learn the API, then modify it to experiment.
//!
//! Ownership:
//!   Every tensor created in this example is owned by the arena allocator
//!   obtained from `init.arena`, so we never call `deinit` explicitly — the
//!   arena frees everything at program exit.  In real training code you
//!   would use a GeneralPurposeAllocator and free tensors individually.
//!

const std = @import("std");
const ztl = @import("zig_transformer_lab");

const Shape = ztl.shape.Shape;
const Tensor = ztl.Tensor;
const Rng = ztl.rng.Rng;

// -- helpers for the example -------------------------------------------------

fn banner(writer: *std.Io.Writer, title: []const u8) !void {
    try writer.print("\n== {s} ==\n", .{title});
}

fn shapeStr(shape: Shape, buf: []u8) []const u8 {
    return ztl.shape.toString(shape, buf);
}

// -- main --------------------------------------------------------------------

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    // Use the Io instance provided by the runtime to get a locked stderr.
    // In Zig 0.16.0, there is no global stdout — the Io instance manages I/O.
    var stderr_buf: [0]u8 = undefined;
    const locked_stderr = init.io.lockStderr(&stderr_buf, null) catch return;
    const writer = &locked_stderr.file_writer.interface;

    // We use a fixed seed so the output is deterministic across runs.
    // Change this number and you get a different random sequence.
    var rng = Rng.init(42);

    // ========================================================================
    // 1. Shape basics
    // ========================================================================
    try banner(writer, "1. Shape basics");

    const s1 = Shape.init1D(10);
    const s2 = Shape.init2D(2, 3);
    const s3 = Shape.init3D(2, 3, 4);
    const s4 = Shape.init4D(2, 3, 4, 5);

    var buf: [64]u8 = undefined;
    try writer.print("  s1 = {s}  (ndim={d})\n", .{ shapeStr(s1, &buf), s1.ndim() });
    try writer.print("  s2 = {s}  (ndim={d})\n", .{ shapeStr(s2, &buf), s2.ndim() });
    try writer.print("  s3 = {s}  (ndim={d})\n", .{ shapeStr(s3, &buf), s3.ndim() });
    try writer.print("  s4 = {s}  (ndim={d})\n", .{ shapeStr(s4, &buf), s4.ndim() });

    // Total elements = product of all dimensions
    try writer.print("  totalElements(s2) = {d}\n", .{ztl.shape.totalElements(s2)});
    try writer.print("  totalElements(s3) = {d}\n", .{ztl.shape.totalElements(s3)});

    // Row-major strides: last dim stride=1, earlier dims = product of later dims
    const strides_2d = ztl.shape.computeStrides(s2);
    try writer.print("  strides for (2,3) = [{d},{d}]\n", .{
        strides_2d.values[0],
        strides_2d.values[1],
    });

    // ========================================================================
    // 2. Tensor creation
    // ========================================================================
    try banner(writer, "2. Tensor creation");

    // zeros: all elements initialized to 0.0
    const z = try ztl.ops.create.zeros(allocator, s2);
    try writer.print("  zeros(2,3):  ", .{});
    try ztl.tensor_print.printValues(z, writer, 6);

    // ones: all elements = 1.0
    const o = try ztl.ops.create.ones(allocator, s2);
    try writer.print("  ones(2,3):   ", .{});
    try ztl.tensor_print.printValues(o, writer, 6);

    // full: fill with an arbitrary constant
    const f = try ztl.ops.create.full(allocator, s2, 7.0);
    try writer.print("  full(2,3,7): ", .{});
    try ztl.tensor_print.printValues(f, writer, 6);

    // arange: evenly spaced 1D sequence
    const seq = try ztl.ops.create.arange(allocator, 0.0, 6.0, 1.0);
    try writer.print("  arange(0,6,1): ", .{});
    try ztl.tensor_print.printValues(seq, writer, 6);

    // randn: normal distribution (Box-Muller)
    const noise = try ztl.ops.create.randn(allocator, Shape.init1D(5), &rng, 0.0, 1.0);
    try writer.print("  randn(5,0,1):  ", .{});
    try ztl.tensor_print.printValues(noise, writer, 5);

    // fromSlice: build a tensor from known data
    const from_sl = try ztl.ops.create.fromSlice(allocator, s2, &.{ 1, 2, 3, 4, 5, 6 });
    try writer.print("  fromSlice:     ", .{});
    try ztl.tensor_print.printValues(from_sl, writer, 6);

    // ========================================================================
    // 3. Element access and indexing
    // ========================================================================
    try banner(writer, "3. Element access and indexing");

    // at() reads a value; atPtr() gives a mutable pointer
    try writer.print("  from_sl.at(1,2) = {d:.1}\n", .{from_sl.at(&[_]usize{ 1, 2 })});

    // flatIndex: how multi-dim indices map to the flat buffer
    // For (2,3) with strides [3,1], index (1,2) => 1*3 + 2*1 = 5
    try writer.print("  flatIndex(1,2) = {d}\n", .{from_sl.flatIndex(&[_]usize{ 1, 2 })});

    // ========================================================================
    // 4. Debug printing
    // ========================================================================
    try banner(writer, "4. Debug printing");

    // debugSummary shows shape, strides, dtype, device, ownership + stats
    try writer.print("  ", .{});
    try ztl.tensor_print.debugSummary(from_sl, writer);

    // ========================================================================
    // 5. Broadcasting
    // ========================================================================
    try banner(writer, "5. Broadcasting");

    // NumPy-style broadcasting: shapes are right-aligned, and dims of size 1
    // "stretch" to match the other tensor's dim.
    //
    // (1,3) + (2,3) -> (2,3)
    //   Row 0: [a0, a1, a2] + [b00, b01, b02]
    //   Row 1: [a0, a1, a2] + [b10, b11, b12]  (a is reused)
    const row = try ztl.ops.create.fromSlice(allocator, Shape.init2D(1, 3), &.{ 100, 200, 300 });
    const mat = try ztl.ops.create.fromSlice(allocator, Shape.init2D(2, 3), &.{ 1, 2, 3, 4, 5, 6 });
    const bc_add = try ztl.ops.elementwise.add(allocator, row, mat);
    try writer.print("  (1,3)+(2,3):  ", .{});
    try ztl.tensor_print.printValues(bc_add, writer, 6);

    // (2,1) + (2,3) -> (2,3)
    const col = try ztl.ops.create.fromSlice(allocator, Shape.init2D(2, 1), &.{ 10, 20 });
    const bc_add2 = try ztl.ops.elementwise.add(allocator, col, mat);
    try writer.print("  (2,1)+(2,3):  ", .{});
    try ztl.tensor_print.printValues(bc_add2, writer, 6);

    // broadcastShapes computes the output shape without running the op
    const bc_shape = try ztl.shape.broadcastShapes(Shape.init2D(1, 3), Shape.init2D(2, 3));
    try writer.print("  broadcast((1,3),(2,3)) = {s}\n", .{shapeStr(bc_shape, &buf)});

    // ========================================================================
    // 6. Elementwise binary ops
    // ========================================================================
    try banner(writer, "6. Elementwise binary ops");

    const a = try ztl.ops.create.fromSlice(allocator, Shape.init1D(4), &.{ 10, 20, 30, 40 });
    const b = try ztl.ops.create.fromSlice(allocator, Shape.init1D(4), &.{ 1, 2, 3, 4 });

    const add_res = try ztl.ops.elementwise.add(allocator, a, b);
    try writer.print("  add:  ", .{});
    try ztl.tensor_print.printValues(add_res, writer, 4);

    const sub_res = try ztl.ops.elementwise.sub(allocator, a, b);
    try writer.print("  sub:  ", .{});
    try ztl.tensor_print.printValues(sub_res, writer, 4);

    const mul_res = try ztl.ops.elementwise.mul(allocator, a, b);
    try writer.print("  mul:  ", .{});
    try ztl.tensor_print.printValues(mul_res, writer, 4);

    const div_res = try ztl.ops.elementwise.div(allocator, a, b);
    try writer.print("  div:  ", .{});
    try ztl.tensor_print.printValues(div_res, writer, 4);

    // Scalar ops: add a single value to every element
    const added = try ztl.ops.elementwise.addScalar(allocator, a, 100.0);
    try writer.print("  addScalar(100):  ", .{});
    try ztl.tensor_print.printValues(added, writer, 4);

    const scaled = try ztl.ops.elementwise.mulScalar(allocator, a, 0.1);
    try writer.print("  mulScalar(0.1): ", .{});
    try ztl.tensor_print.printValues(scaled, writer, 4);

    // ========================================================================
    // 7. Unary ops (exp, log, relu, gelu, neg)
    // ========================================================================
    try banner(writer, "7. Unary ops");

    const u = try ztl.ops.create.fromSlice(allocator, Shape.init1D(5), &.{ -2, -1, 0, 1, 2 });

    const relu_out = try ztl.ops.unary.relu(allocator, u);
    try writer.print("  relu:  ", .{});
    try ztl.tensor_print.printValues(relu_out, writer, 5);

    const exp_out = try ztl.ops.unary.exp(allocator, u);
    try writer.print("  exp:   ", .{});
    try ztl.tensor_print.printValues(exp_out, writer, 5);

    const log_out = try ztl.ops.unary.log(allocator, try ztl.ops.create.fromSlice(allocator, Shape.init1D(3), &.{ 1.0, 2.718, 7.389 }));
    try writer.print("  log:   ", .{});
    try ztl.tensor_print.printValues(log_out, writer, 3);

    const gelu_out = try ztl.ops.unary.geluExact(allocator, u);
    try writer.print("  gelu:  ", .{});
    try ztl.tensor_print.printValues(gelu_out, writer, 5);

    // ========================================================================
    // 8. Reductions (sum, mean, max, sumAll)
    // ========================================================================
    try banner(writer, "8. Reductions");

    const r = try ztl.ops.create.fromSlice(allocator, Shape.init2D(2, 3), &.{ 1, 2, 3, 4, 5, 6 });

    const sum0 = try ztl.ops.reduce.sum(allocator, r, 0);
    try writer.print("  sum(axis=0): ", .{});
    try ztl.tensor_print.printValues(sum0, writer, 3);

    const sum1 = try ztl.ops.reduce.sum(allocator, r, 1);
    try writer.print("  sum(axis=1): ", .{});
    try ztl.tensor_print.printValues(sum1, writer, 2);

    const mean0 = try ztl.ops.reduce.mean(allocator, r, 0);
    try writer.print("  mean(axis=0): ", .{});
    try ztl.tensor_print.printValues(mean0, writer, 3);

    const max0 = try ztl.ops.reduce.max(allocator, r, 0);
    try writer.print("  max(axis=0):  ", .{});
    try ztl.tensor_print.printValues(max0, writer, 3);

    const sum_all = try ztl.ops.reduce.sumAll(allocator, r);
    try writer.print("  sumAll:  ", .{});
    try ztl.tensor_print.printValues(sum_all, writer, 1);

    // ========================================================================
    // 9. Matmul
    // ========================================================================
    try banner(writer, "9. Matmul (ikj cache-friendly)");

    // A(2,3) @ B(3,4) = C(2,4)
    const ma = try ztl.ops.create.fromSlice(allocator, Shape.init2D(2, 3), &.{ 1, 2, 3, 4, 5, 6 });
    const mb = try ztl.ops.create.fromSlice(allocator, Shape.init2D(3, 4), &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
    const mc = try ztl.ops.matmul.matmul(allocator, ma, mb);
    try writer.print("  (2,3)@(3,4) = (2,4):  ", .{});
    try ztl.tensor_print.printValues(mc, writer, 8);

    // Batched matmul: (B,M,K) @ (B,K,N) = (B,M,N)
    const ba = try ztl.ops.create.full(allocator, Shape.init3D(2, 3, 4), 1.0);
    const bb = try ztl.ops.create.full(allocator, Shape.init3D(2, 4, 5), 1.0);
    const bc = try ztl.ops.matmul.matmulBatch(allocator, ba, bb);
    try writer.print("  batched (2,3,4)@(2,4,5) shape: {s}\n", .{shapeStr(bc.shape, &buf)});
    try writer.print("  batched result[0]:  ", .{});
    try ztl.tensor_print.printValues(bc, writer, 5);

    // ========================================================================
    // 10. Views: transpose and reshape
    // ========================================================================
    try banner(writer, "10. Views: transpose and reshape");

    const v = try ztl.ops.create.fromSlice(allocator, Shape.init2D(2, 3), &.{ 1, 2, 3, 4, 5, 6 });

    // transpose2d: returns a view (owned=false), no data copied
    const vt = try v.transpose2d();
    try writer.print("  original (2,3):  ", .{});
    try ztl.tensor_print.printValues(v, writer, 6);
    try writer.print("  transpose (3,2): ", .{});
    try ztl.tensor_print.printValues(vt, writer, 6);
    try writer.print("  transposed owned={}\n", .{vt.owned});

    // reshape: zero-copy view if contiguous and element count matches
    const vr = try v.reshape(Shape.init1D(6));
    try writer.print("  reshape (6,):   ", .{});
    try ztl.tensor_print.printValues(vr, writer, 6);
    try writer.print("  reshaped owned={}\n", .{vr.owned});

    // ========================================================================
    // 11. Softmax and log-softmax
    // ========================================================================
    try banner(writer, "11. Softmax (numerically stable)");

    // Softmax converts raw scores (logits) into probabilities that sum to 1.
    // The "max subtraction" trick prevents overflow: exp(x - max(x)) is
    // always <= 1, so we never compute exp(large_number).
    const logits = try ztl.ops.create.fromSlice(allocator, Shape.init2D(1, 3), &.{ 1.0, 2.0, 3.0 });
    const probs = try ztl.ops.softmax.softmax(allocator, logits);
    try writer.print("  logits:  ", .{});
    try ztl.tensor_print.printValues(logits, writer, 3);
    try writer.print("  softmax: ", .{});
    try ztl.tensor_print.printValues(probs, writer, 3);

    // Verify probabilities sum to 1.0
    const prob_sum = try ztl.ops.reduce.sumAll(allocator, probs);
    try writer.print("  sum(probs) = {d:.6} (should be ~1.0)\n", .{prob_sum.data[0]});

    // log_softmax is more numerically stable than log(softmax)
    const lse = try ztl.ops.softmax.logSoftmax(allocator, logits);
    try writer.print("  log_softmax: ", .{});
    try ztl.tensor_print.printValues(lse, writer, 3);

    // Even large logits don't overflow thanks to the max-subtraction trick
    const big = try ztl.ops.create.fromSlice(allocator, Shape.init2D(1, 3), &.{ 100.0, 101.0, 102.0 });
    const big_probs = try ztl.ops.softmax.softmax(allocator, big);
    try writer.print("  softmax([100,101,102]) = ", .{});
    try ztl.tensor_print.printValues(big_probs, writer, 3);

    // ========================================================================
    // 12. Cross-entropy loss
    // ========================================================================
    try banner(writer, "12. Cross-entropy loss");

    // Cross-entropy = -log(P(target class)).
    // It measures how well the predicted probabilities match the target.
    // Uses log_softmax internally for numerical stability.
    const ce_logits = try ztl.ops.create.fromSlice(allocator, Shape.init2D(2, 3), &.{
        2.0, 1.0, 0.1,
        0.1, 2.0, 1.0,
    });
    const ce_targets = try ztl.ops.create.fromSlice(allocator, Shape.init1D(2), &.{ 0.0, 1.0 });
    const ce_loss = try ztl.ops.loss.crossEntropy(allocator, ce_logits, ce_targets);
    try writer.print("  logits (2,3):  ", .{});
    try ztl.tensor_print.printValues(ce_logits, writer, 6);
    try writer.print("  targets:       ", .{});
    try ztl.tensor_print.printValues(ce_targets, writer, 2);
    try writer.print("  CE loss: {d:.4}\n", .{ce_loss.data[0]});

    // When logits are uniform (all equal), CE loss = ln(C) = ln(3) ~ 1.099
    const uni = try ztl.ops.create.ones(allocator, Shape.init2D(1, 3));
    const uni_tgt = try ztl.ops.create.zeros(allocator, Shape.init1D(1));
    const uni_loss = try ztl.ops.loss.crossEntropy(allocator, uni, uni_tgt);
    try writer.print("  uniform CE loss = {d:.4} (ln(3) ~ 1.099)\n", .{uni_loss.data[0]});

    // ========================================================================
    // 13. Squeeze (remove size-1 dimensions)
    // ========================================================================
    try banner(writer, "13. Squeeze");

    const sq_in = Shape.init4D(1, 3, 1, 4);
    const sq_out = ztl.shape.squeeze(sq_in, null);
    try writer.print("  squeeze(1,3,1,4) = {s}\n", .{shapeStr(sq_out, &buf)});

    const sq_axis = ztl.shape.squeeze(Shape.init2D(1, 3), 0);
    try writer.print("  squeeze((1,3), axis=0) = {s}\n", .{shapeStr(sq_axis, &buf)});

    // ========================================================================
    // Done!
    // ========================================================================
    try banner(writer, "Done!");
    try writer.print("  All Stage 2 tensor operations demonstrated successfully.\n", .{});
    try writer.print("  Arena allocator frees all tensors automatically.\n", .{});
}
