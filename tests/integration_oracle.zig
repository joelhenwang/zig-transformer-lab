//!
//! tests/integration_oracle.zig — PyTorch parity tests for CPU ops
//!
//! For each fixture case under `tests/fixtures/`, this suite:
//!   1. Loads the PyTorch-generated input tensors.
//!   2. Runs our corresponding forward op.
//!   3. Compares forward output with the oracle within tolerance.
//!   4. Runs backward via the tape.
//!   5. Compares parameter gradients with the oracle within tolerance.
//!
//! The fixtures are produced by `python tools/oracle.py generate`. If
//! they do not exist, the tests fail with `error.IoError` during
//! fixture load — that is deliberate: "forgot to regenerate" should
//! be a loud failure, not a silent skip.
//!
//! This file is wired into `src/root.zig` so `zig build test` runs it
//! alongside the rest of the suite. We use relative imports through
//! the library's own module path so the test doesn't accidentally
//! get its own copy of the Tensor type.
//!

const std = @import("std");
const ztl = @import("zig_transformer_lab");

const Tensor = ztl.Tensor;
const Shape = ztl.shape.Shape;
const Tape = ztl.Tape;
const oracle = ztl.testing_utils.oracle;

const ops_elementwise = ztl.ops.elementwise;
const ops_matmul = ztl.ops.matmul;
const ops_softmax = ztl.ops.softmax;
const ops_loss = ztl.ops.loss;
const ops_unary = ztl.ops.unary;
const ops_reduce = ztl.ops.reduce;
const ops_shape = ztl.ops.shape_ops;

// -- Io plumbing -----------------------------------------------------------

fn testIo() !std.Io {
    const T = struct {
        var instance: ?std.Io.Threaded = null;
    };
    if (T.instance == null) {
        T.instance = std.Io.Threaded.init(std.testing.allocator, .{});
    }
    return T.instance.?.io();
}

const FIXTURE_ROOT = "tests/fixtures";

fn fixturePath(comptime case: []const u8, comptime file: []const u8) []const u8 {
    return FIXTURE_ROOT ++ "/" ++ case ++ "/" ++ file;
}

// -- Helpers to drive sum-of-output-as-loss backward pattern ----------------

/// Run a forward op that produces `output`, then back-propagate as if
/// the loss were `sum(output)`. This matches what `tools/oracle.py`
/// does for every "sumAll" case and lets us reuse one pattern across
/// most ops.
///
/// The oracle treats the all-ones tensor of `output.shape` as grad_out
/// for backward; we achieve the same effect by calling `sumAll` on
/// `output` (which produces a scalar whose gradient seed is 1.0) and
/// then `tape.backward` on that scalar.
fn backwardThroughSum(tape: *Tape, output: *Tensor) !void {
    const alloc = std.testing.allocator;
    var loss = try ops_reduce.sumAll(alloc, output.*, tape);
    defer loss.deinit(alloc);
    try tape.backward(&loss);
}

// -- Case: add_2d ----------------------------------------------------------

test "oracle add_2d: forward and backward parity" {
    const alloc = std.testing.allocator;
    const io = try testIo();

    var a = try oracle.loadTensor(alloc, io, fixturePath("add_2d", "input_0.ztlt"));
    defer a.deinit(alloc);
    var b = try oracle.loadTensor(alloc, io, fixturePath("add_2d", "input_1.ztlt"));
    defer b.deinit(alloc);
    var expect_y = try oracle.loadTensor(alloc, io, fixturePath("add_2d", "output.ztlt"));
    defer expect_y.deinit(alloc);
    var expect_da = try oracle.loadTensor(alloc, io, fixturePath("add_2d", "grad_input_0.ztlt"));
    defer expect_da.deinit(alloc);
    var expect_db = try oracle.loadTensor(alloc, io, fixturePath("add_2d", "grad_input_1.ztlt"));
    defer expect_db.deinit(alloc);

    // Mark inputs as requires_grad and register on tape.
    a.requires_grad = true;
    b.requires_grad = true;

    var tape = Tape.init(alloc);
    defer tape.deinit();
    _ = try tape.trackLeaf(&a);
    _ = try tape.trackLeaf(&b);

    // Forward.
    var y = try ops_elementwise.add(alloc, a, b, &tape);
    defer y.deinit(alloc);

    // Forward parity.
    try oracle.expectClose(y, expect_y, .{ .rel_tol = 1e-4, .abs_tol = 1e-5 });

    // Backward: loss = sum(y).
    try backwardThroughSum(&tape, &y);

    // Gradient parity.
    try oracle.expectClose(a.grad.?.*, expect_da, .{ .rel_tol = 1e-4, .abs_tol = 1e-5 });
    try oracle.expectClose(b.grad.?.*, expect_db, .{ .rel_tol = 1e-4, .abs_tol = 1e-5 });
}

// -- Case: add_broadcast_2d_1d --------------------------------------------

test "oracle add_broadcast_2d_1d: forward and backward parity (shape reduction)" {
    const alloc = std.testing.allocator;
    const io = try testIo();

    var a = try oracle.loadTensor(alloc, io, fixturePath("add_broadcast_2d_1d", "input_0.ztlt"));
    defer a.deinit(alloc);
    var b = try oracle.loadTensor(alloc, io, fixturePath("add_broadcast_2d_1d", "input_1.ztlt"));
    defer b.deinit(alloc);
    var expect_y = try oracle.loadTensor(alloc, io, fixturePath("add_broadcast_2d_1d", "output.ztlt"));
    defer expect_y.deinit(alloc);
    var expect_da = try oracle.loadTensor(alloc, io, fixturePath("add_broadcast_2d_1d", "grad_input_0.ztlt"));
    defer expect_da.deinit(alloc);
    var expect_db = try oracle.loadTensor(alloc, io, fixturePath("add_broadcast_2d_1d", "grad_input_1.ztlt"));
    defer expect_db.deinit(alloc);

    a.requires_grad = true;
    b.requires_grad = true;
    var tape = Tape.init(alloc);
    defer tape.deinit();
    _ = try tape.trackLeaf(&a);
    _ = try tape.trackLeaf(&b);

    var y = try ops_elementwise.add(alloc, a, b, &tape);
    defer y.deinit(alloc);

    try oracle.expectClose(y, expect_y, .{ .rel_tol = 1e-4, .abs_tol = 1e-5 });
    try backwardThroughSum(&tape, &y);

    // The key check: b's grad must be reduced back to 1D shape. An
    // add-backward bug that forgets the reduction produces (2,3)
    // shape here and fails the shape check inside expectClose.
    try oracle.expectClose(a.grad.?.*, expect_da, .{ .rel_tol = 1e-4, .abs_tol = 1e-5 });
    try oracle.expectClose(b.grad.?.*, expect_db, .{ .rel_tol = 1e-4, .abs_tol = 1e-5 });
}

// -- Case: mul_broadcast --------------------------------------------------

test "oracle mul_broadcast: saved-tensor_pair backward parity" {
    const alloc = std.testing.allocator;
    const io = try testIo();

    var a = try oracle.loadTensor(alloc, io, fixturePath("mul_broadcast", "input_0.ztlt"));
    defer a.deinit(alloc);
    var b = try oracle.loadTensor(alloc, io, fixturePath("mul_broadcast", "input_1.ztlt"));
    defer b.deinit(alloc);
    var expect_y = try oracle.loadTensor(alloc, io, fixturePath("mul_broadcast", "output.ztlt"));
    defer expect_y.deinit(alloc);
    var expect_da = try oracle.loadTensor(alloc, io, fixturePath("mul_broadcast", "grad_input_0.ztlt"));
    defer expect_da.deinit(alloc);
    var expect_db = try oracle.loadTensor(alloc, io, fixturePath("mul_broadcast", "grad_input_1.ztlt"));
    defer expect_db.deinit(alloc);

    a.requires_grad = true;
    b.requires_grad = true;
    var tape = Tape.init(alloc);
    defer tape.deinit();
    _ = try tape.trackLeaf(&a);
    _ = try tape.trackLeaf(&b);

    var y = try ops_elementwise.mul(alloc, a, b, &tape);
    defer y.deinit(alloc);

    try oracle.expectClose(y, expect_y, .{ .rel_tol = 1e-4, .abs_tol = 1e-5 });
    try backwardThroughSum(&tape, &y);

    try oracle.expectClose(a.grad.?.*, expect_da, .{ .rel_tol = 1e-4, .abs_tol = 1e-5 });
    try oracle.expectClose(b.grad.?.*, expect_db, .{ .rel_tol = 1e-4, .abs_tol = 1e-5 });
}

// -- Case: matmul_2d ------------------------------------------------------

test "oracle matmul_2d: asymmetric MxK @ KxN parity" {
    const alloc = std.testing.allocator;
    const io = try testIo();

    var a = try oracle.loadTensor(alloc, io, fixturePath("matmul_2d", "input_0.ztlt"));
    defer a.deinit(alloc);
    var b = try oracle.loadTensor(alloc, io, fixturePath("matmul_2d", "input_1.ztlt"));
    defer b.deinit(alloc);
    var expect_y = try oracle.loadTensor(alloc, io, fixturePath("matmul_2d", "output.ztlt"));
    defer expect_y.deinit(alloc);
    var expect_da = try oracle.loadTensor(alloc, io, fixturePath("matmul_2d", "grad_input_0.ztlt"));
    defer expect_da.deinit(alloc);
    var expect_db = try oracle.loadTensor(alloc, io, fixturePath("matmul_2d", "grad_input_1.ztlt"));
    defer expect_db.deinit(alloc);

    a.requires_grad = true;
    b.requires_grad = true;
    var tape = Tape.init(alloc);
    defer tape.deinit();
    _ = try tape.trackLeaf(&a);
    _ = try tape.trackLeaf(&b);

    var y = try ops_matmul.matmul(alloc, a, b, &tape);
    defer y.deinit(alloc);

    try oracle.expectClose(y, expect_y, .{ .rel_tol = 1e-4, .abs_tol = 1e-4 });
    try backwardThroughSum(&tape, &y);

    try oracle.expectClose(a.grad.?.*, expect_da, .{ .rel_tol = 1e-4, .abs_tol = 1e-4 });
    try oracle.expectClose(b.grad.?.*, expect_db, .{ .rel_tol = 1e-4, .abs_tol = 1e-4 });
}

// -- Case: softmax_3d_last_axis -------------------------------------------

test "oracle softmax_3d_last_axis: forward and backward parity" {
    const alloc = std.testing.allocator;
    const io = try testIo();

    var a = try oracle.loadTensor(alloc, io, fixturePath("softmax_3d_last_axis", "input_0.ztlt"));
    defer a.deinit(alloc);
    var expect_y = try oracle.loadTensor(alloc, io, fixturePath("softmax_3d_last_axis", "output.ztlt"));
    defer expect_y.deinit(alloc);
    var expect_da = try oracle.loadTensor(alloc, io, fixturePath("softmax_3d_last_axis", "grad_input_0.ztlt"));
    defer expect_da.deinit(alloc);

    a.requires_grad = true;
    var tape = Tape.init(alloc);
    defer tape.deinit();
    _ = try tape.trackLeaf(&a);

    var y = try ops_softmax.softmax(alloc, a, &tape);
    defer y.deinit(alloc);

    try oracle.expectClose(y, expect_y, .{ .rel_tol = 1e-4, .abs_tol = 5e-5 });
    try backwardThroughSum(&tape, &y);

    try oracle.expectClose(a.grad.?.*, expect_da, .{ .rel_tol = 1e-4, .abs_tol = 1e-4 });
}

// -- Case: gelu_2d --------------------------------------------------------

test "oracle gelu_2d: exact (erf-based) GELU parity" {
    const alloc = std.testing.allocator;
    const io = try testIo();

    var a = try oracle.loadTensor(alloc, io, fixturePath("gelu_2d", "input_0.ztlt"));
    defer a.deinit(alloc);
    var expect_y = try oracle.loadTensor(alloc, io, fixturePath("gelu_2d", "output.ztlt"));
    defer expect_y.deinit(alloc);
    var expect_da = try oracle.loadTensor(alloc, io, fixturePath("gelu_2d", "grad_input_0.ztlt"));
    defer expect_da.deinit(alloc);

    a.requires_grad = true;
    var tape = Tape.init(alloc);
    defer tape.deinit();
    _ = try tape.trackLeaf(&a);

    var y = try ops_unary.geluExact(alloc, a, &tape);
    defer y.deinit(alloc);

    try oracle.expectClose(y, expect_y, .{ .rel_tol = 1e-4, .abs_tol = 5e-5 });
    try backwardThroughSum(&tape, &y);

    try oracle.expectClose(a.grad.?.*, expect_da, .{ .rel_tol = 1e-4, .abs_tol = 1e-4 });
}

// -- Case: cross_entropy_3d ------------------------------------------------

test "oracle cross_entropy_3d: forward loss and logits-grad parity" {
    const alloc = std.testing.allocator;
    const io = try testIo();

    var logits = try oracle.loadTensor(alloc, io, fixturePath("cross_entropy_3d", "input_0.ztlt"));
    defer logits.deinit(alloc);
    var targets = try oracle.loadTensor(alloc, io, fixturePath("cross_entropy_3d", "input_1.ztlt"));
    defer targets.deinit(alloc);
    var expect_loss = try oracle.loadTensor(alloc, io, fixturePath("cross_entropy_3d", "output.ztlt"));
    defer expect_loss.deinit(alloc);
    var expect_dlogits = try oracle.loadTensor(alloc, io, fixturePath("cross_entropy_3d", "grad_input_0.ztlt"));
    defer expect_dlogits.deinit(alloc);

    logits.requires_grad = true;
    var tape = Tape.init(alloc);
    defer tape.deinit();
    _ = try tape.trackLeaf(&logits);

    // Our crossEntropy expects logits (B*T, V) and targets (B*T,).
    // The fixture stored logits as (B, T, V) and targets as (B, T) —
    // reshape via tape-tracked reshape so backward can restore the
    // original (B, T, V) shape onto logits.grad.
    const B = logits.shape.dims[0];
    const T = logits.shape.dims[1];
    const V = logits.shape.dims[2];
    var logits_2d = try ztl.ops.shape_ops.reshapeTracked(
        alloc, logits, Shape.init2D(B * T, V), &tape,
    );
    defer logits_2d.deinit(alloc);
    // logits_2d is owned by us. The tape holds its own copy via
    // PR-ε's record() → cloneTensorData, so freeing our copy here
    // does not disturb backward.

    const targets_1d = try targets.reshape(Shape.init1D(B * T));
    // targets_1d is an untracked view; targets does not require grad,
    // so no tape bookkeeping is needed. No deinit — it's a view.

    var loss = try ops_loss.crossEntropy(alloc, logits_2d, targets_1d, &tape);
    defer loss.deinit(alloc);

    // Compare scalar loss.
    try oracle.expectClose(loss, expect_loss, .{ .rel_tol = 1e-4, .abs_tol = 1e-4 });

    // Backward.
    try tape.backward(&loss);

    // Gradient of logits should match the oracle. It ends up with the
    // original (B, T, V) shape because `reshapeTracked`'s backward
    // restores it.
    try oracle.expectClose(logits.grad.?.*, expect_dlogits, .{ .rel_tol = 1e-4, .abs_tol = 1e-4 });
}

// -- Case: layernorm_3d ---------------------------------------------------
//
// Our LayerNorm is composed from primitive ops (mean, sub, mul, sqrt,
// div, scalar add, reshapeTracked). The oracle walks the parity check
// end-to-end: forward output and gradients for input, gamma, and beta
// must all match PyTorch's F.layer_norm within tolerance. The backward
// path through seven composed ops is where subtle bugs hide.

test "oracle layernorm_3d: composed LayerNorm forward and backward parity" {
    const alloc = std.testing.allocator;
    const io = try testIo();

    var x = try oracle.loadTensor(alloc, io, fixturePath("layernorm_3d", "input_0.ztlt"));
    defer x.deinit(alloc);
    var gamma = try oracle.loadTensor(alloc, io, fixturePath("layernorm_3d", "input_1.ztlt"));
    defer gamma.deinit(alloc);
    var beta = try oracle.loadTensor(alloc, io, fixturePath("layernorm_3d", "input_2.ztlt"));
    defer beta.deinit(alloc);
    var expect_y = try oracle.loadTensor(alloc, io, fixturePath("layernorm_3d", "output.ztlt"));
    defer expect_y.deinit(alloc);
    var expect_dx = try oracle.loadTensor(alloc, io, fixturePath("layernorm_3d", "grad_input_0.ztlt"));
    defer expect_dx.deinit(alloc);
    var expect_dgamma = try oracle.loadTensor(alloc, io, fixturePath("layernorm_3d", "grad_input_1.ztlt"));
    defer expect_dgamma.deinit(alloc);
    var expect_dbeta = try oracle.loadTensor(alloc, io, fixturePath("layernorm_3d", "grad_input_2.ztlt"));
    defer expect_dbeta.deinit(alloc);

    x.requires_grad = true;
    gamma.requires_grad = true;
    beta.requires_grad = true;

    // Build a LayerNorm whose gamma/beta we can overwrite with the
    // oracle values. Easiest: reach into the LayerNorm directly by
    // constructing it, then copying the fixture values into its
    // internal tensors.
    const LayerNorm = ztl.nn.layernorm.LayerNorm;
    const Rng = ztl.rng.Rng;
    var rng = Rng.init(0);
    var ln = try LayerNorm.init(alloc, gamma.shape.dims[0], 1e-5, &rng);
    defer ln.deinit();

    // Copy oracle gamma/beta into ln's tensors so the op sees those
    // values. `ln.gamma.param_id` (assigned in LayerNorm.init) stays
    // the same, so the optimizer-state invariant from PR-ζ is
    // preserved.
    @memcpy(ln.gamma.data, gamma.data);
    @memcpy(ln.beta.data, beta.data);

    // Track the oracle's x on the tape. We do NOT track the loaded
    // `gamma`/`beta` because `ln.gamma`/`ln.beta` are what the op
    // reads. The op will track them itself when their requires_grad
    // is set; mirror the oracle's requires_grad state.
    ln.gamma.requires_grad = true;
    ln.beta.requires_grad = true;

    var tape = Tape.init(alloc);
    defer tape.deinit();
    _ = try tape.trackLeaf(&x);
    _ = try tape.trackLeaf(&ln.gamma);
    _ = try tape.trackLeaf(&ln.beta);

    var y = try ln.forward(x, &tape);
    defer y.deinit(alloc);

    try oracle.expectClose(y, expect_y, .{ .rel_tol = 1e-4, .abs_tol = 5e-5 });

    try backwardThroughSum(&tape, &y);

    try oracle.expectClose(x.grad.?.*, expect_dx, .{ .rel_tol = 1e-3, .abs_tol = 2e-4 });
    try oracle.expectClose(ln.gamma.grad.?.*, expect_dgamma, .{ .rel_tol = 1e-3, .abs_tol = 2e-4 });
    try oracle.expectClose(ln.beta.grad.?.*, expect_dbeta, .{ .rel_tol = 1e-3, .abs_tol = 2e-4 });
}
