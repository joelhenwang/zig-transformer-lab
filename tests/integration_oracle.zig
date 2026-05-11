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

// -- Case: embedding_3d ---------------------------------------------------
//
// Forward is a row gather; backward is scatter-add (the gradient for
// weight row `id` gets incremented by the output gradient at every
// position with that id). The fixture uses ids with DELIBERATE
// repeats — without scatter-add, each repeat's gradient contribution
// would overwrite the others, and the test would fail on a row with
// multiple references.

test "oracle embedding_3d: forward gather + scatter-add backward parity" {
    const alloc = std.testing.allocator;
    const io = try testIo();

    var weight = try oracle.loadTensor(alloc, io, fixturePath("embedding_3d", "input_0.ztlt"));
    defer weight.deinit(alloc);
    var ids = try oracle.loadTensor(alloc, io, fixturePath("embedding_3d", "input_1.ztlt"));
    defer ids.deinit(alloc);
    var expect_y = try oracle.loadTensor(alloc, io, fixturePath("embedding_3d", "output.ztlt"));
    defer expect_y.deinit(alloc);
    var expect_dweight = try oracle.loadTensor(alloc, io, fixturePath("embedding_3d", "grad_input_0.ztlt"));
    defer expect_dweight.deinit(alloc);

    // Build an Embedding whose `weight` is the oracle's exact bytes.
    // The Embedding.init allocator does randn initialisation, but we
    // then overwrite with oracle bytes so our forward sees the same
    // input as PyTorch.
    const Embedding = ztl.nn.embedding.Embedding;
    const Rng = ztl.rng.Rng;
    var rng = Rng.init(0);
    const V = weight.shape.dims[0];
    const D = weight.shape.dims[1];
    var emb = try Embedding.init(alloc, V, D, &rng);
    defer emb.deinit();
    @memcpy(emb.weight.data, weight.data);
    emb.weight.requires_grad = true;

    var tape = Tape.init(alloc);
    defer tape.deinit();
    _ = try tape.trackLeaf(&emb.weight);

    var y = try emb.forward(ids, &tape);
    defer y.deinit(alloc);

    try oracle.expectClose(y, expect_y, .{ .rel_tol = 1e-4, .abs_tol = 1e-5 });

    try backwardThroughSum(&tape, &y);

    try oracle.expectClose(emb.weight.grad.?.*, expect_dweight, .{ .rel_tol = 1e-4, .abs_tol = 1e-5 });
}

// -- Case: matmul_batch_3d ------------------------------------------------
//
// Batched matmul (B, M, K) @ (B, K, N). Asymmetric shape means a
// row/column-major bug in the stride arithmetic produces wrong
// outputs, not accidentally-correct-by-symmetry ones.

test "oracle matmul_batch_3d: batched (B,M,K) @ (B,K,N) parity" {
    const alloc = std.testing.allocator;
    const io = try testIo();

    var a = try oracle.loadTensor(alloc, io, fixturePath("matmul_batch_3d", "input_0.ztlt"));
    defer a.deinit(alloc);
    var b = try oracle.loadTensor(alloc, io, fixturePath("matmul_batch_3d", "input_1.ztlt"));
    defer b.deinit(alloc);
    var expect_y = try oracle.loadTensor(alloc, io, fixturePath("matmul_batch_3d", "output.ztlt"));
    defer expect_y.deinit(alloc);
    var expect_da = try oracle.loadTensor(alloc, io, fixturePath("matmul_batch_3d", "grad_input_0.ztlt"));
    defer expect_da.deinit(alloc);
    var expect_db = try oracle.loadTensor(alloc, io, fixturePath("matmul_batch_3d", "grad_input_1.ztlt"));
    defer expect_db.deinit(alloc);

    a.requires_grad = true;
    b.requires_grad = true;
    var tape = Tape.init(alloc);
    defer tape.deinit();
    _ = try tape.trackLeaf(&a);
    _ = try tape.trackLeaf(&b);

    var y = try ops_matmul.matmulBatch(alloc, a, b, &tape);
    defer y.deinit(alloc);

    try oracle.expectClose(y, expect_y, .{ .rel_tol = 1e-4, .abs_tol = 1e-4 });
    try backwardThroughSum(&tape, &y);

    try oracle.expectClose(a.grad.?.*, expect_da, .{ .rel_tol = 1e-4, .abs_tol = 1e-4 });
    try oracle.expectClose(b.grad.?.*, expect_db, .{ .rel_tol = 1e-4, .abs_tol = 1e-4 });
}

// -- Case: log_softmax_3d -------------------------------------------------

test "oracle log_softmax_3d: forward and backward parity" {
    const alloc = std.testing.allocator;
    const io = try testIo();

    var a = try oracle.loadTensor(alloc, io, fixturePath("log_softmax_3d", "input_0.ztlt"));
    defer a.deinit(alloc);
    var expect_y = try oracle.loadTensor(alloc, io, fixturePath("log_softmax_3d", "output.ztlt"));
    defer expect_y.deinit(alloc);
    var expect_da = try oracle.loadTensor(alloc, io, fixturePath("log_softmax_3d", "grad_input_0.ztlt"));
    defer expect_da.deinit(alloc);

    a.requires_grad = true;
    var tape = Tape.init(alloc);
    defer tape.deinit();
    _ = try tape.trackLeaf(&a);

    var y = try ops_softmax.logSoftmax(alloc, a, &tape);
    defer y.deinit(alloc);

    try oracle.expectClose(y, expect_y, .{ .rel_tol = 1e-4, .abs_tol = 5e-5 });
    try backwardThroughSum(&tape, &y);

    try oracle.expectClose(a.grad.?.*, expect_da, .{ .rel_tol = 1e-4, .abs_tol = 1e-4 });
}

// -- Case: sum_axis_3d ----------------------------------------------------

test "oracle sum_axis_3d: sum over axis=1 with keepdim, forward and backward parity" {
    const alloc = std.testing.allocator;
    const io = try testIo();

    var a = try oracle.loadTensor(alloc, io, fixturePath("sum_axis_3d", "input_0.ztlt"));
    defer a.deinit(alloc);
    var expect_y = try oracle.loadTensor(alloc, io, fixturePath("sum_axis_3d", "output.ztlt"));
    defer expect_y.deinit(alloc);
    var expect_da = try oracle.loadTensor(alloc, io, fixturePath("sum_axis_3d", "grad_input_0.ztlt"));
    defer expect_da.deinit(alloc);

    a.requires_grad = true;
    var tape = Tape.init(alloc);
    defer tape.deinit();
    _ = try tape.trackLeaf(&a);

    var y = try ops_reduce.sum(alloc, a, 1, &tape);
    defer y.deinit(alloc);

    try oracle.expectClose(y, expect_y, .{ .rel_tol = 1e-4, .abs_tol = 1e-5 });
    try backwardThroughSum(&tape, &y);

    try oracle.expectClose(a.grad.?.*, expect_da, .{ .rel_tol = 1e-4, .abs_tol = 1e-5 });
}

// -- Case: mean_axis_3d ---------------------------------------------------

test "oracle mean_axis_3d: mean over last axis with keepdim, forward and backward parity" {
    const alloc = std.testing.allocator;
    const io = try testIo();

    var a = try oracle.loadTensor(alloc, io, fixturePath("mean_axis_3d", "input_0.ztlt"));
    defer a.deinit(alloc);
    var expect_y = try oracle.loadTensor(alloc, io, fixturePath("mean_axis_3d", "output.ztlt"));
    defer expect_y.deinit(alloc);
    var expect_da = try oracle.loadTensor(alloc, io, fixturePath("mean_axis_3d", "grad_input_0.ztlt"));
    defer expect_da.deinit(alloc);

    a.requires_grad = true;
    var tape = Tape.init(alloc);
    defer tape.deinit();
    _ = try tape.trackLeaf(&a);

    // Axis = ndim - 1 = 2 for our (2,3,4) tensor.
    var y = try ops_reduce.mean(alloc, a, 2, &tape);
    defer y.deinit(alloc);

    try oracle.expectClose(y, expect_y, .{ .rel_tol = 1e-4, .abs_tol = 1e-5 });
    try backwardThroughSum(&tape, &y);

    try oracle.expectClose(a.grad.?.*, expect_da, .{ .rel_tol = 1e-4, .abs_tol = 1e-5 });
}

// -- Case: full_model_forward --------------------------------------------
//
// The end-to-end parity check. Builds a TinyWordTransformer at the
// fixture's config, loads every parameter from the oracle's
// per-param .ztlt files, runs forward, and compares logits.
//
// This catches composition bugs that individual op tests miss: a
// wrong sign in the residual add, a transpose order swapped between
// Q and K, an off-by-one in the causal mask, or any drift between
// our pre-norm block layout and the PyTorch reference.

test "oracle full_model_forward: end-to-end TinyWordTransformer logits parity" {
    const alloc = std.testing.allocator;
    const io = try testIo();

    // Config must match what the Python case used. Keeping it in
    // sync is enforced by the fact that the weight .ztlt files
    // encode the shapes — loading will fail with ShapeMismatch if
    // the config drifts.
    const TransformerConfig = ztl.nn.module.TransformerConfig;
    const TinyWordTransformer = ztl.nn.model.TinyWordTransformer;
    const Rng = ztl.rng.Rng;

    const cfg = TransformerConfig{
        .vocab_size = 8,
        .d_model = 4,
        .max_seq_len = 4,
        .d_ff = 8,
        .ln_eps = 1e-5,
        .bias = true,
    };

    var rng = Rng.init(0);
    var model = try TinyWordTransformer.init(alloc, cfg, &rng);
    defer model.deinit();

    // Load the oracle's weights into the model. `collectNamedParams`
    // provides (name, *Tensor) pairs whose order matches the Python
    // case's param_pairs list.
    const NamedParam = ztl.nn.model.NamedParam;
    var params = std.ArrayList(NamedParam).empty;
    defer params.deinit(alloc);
    try model.collectNamedParams(&params);
    defer model.freeBlockNames(&params);

    // For every named param, load its fixture and memcpy into the
    // model tensor's data. @memcpy on same-size slices is safe; the
    // load itself checks the file's shape against our tensor's
    // shape via the ZTLT header size.
    //
    // Stage 8 M3 renamed `block.*` -> `blocks[0].*`. The existing
    // `full_model_forward` fixture was generated under the old
    // names, so we rewrite our new names back to the fixture's
    // on-disk naming at load time.
    for (params.items) |entry| {
        // Rewrite `blocks[0].foo` back to `block.foo` so the fixture
        // filenames match. For n_layer=1 models this is safe; for
        // n_layer > 1 (not this test) the rewrite would collapse
        // distinct blocks into one — but this case is single-block.
        var fixture_name_buf: [300]u8 = undefined;
        const fixture_name: []const u8 = if (std.mem.startsWith(u8, entry.name, "blocks[0]."))
            try std.fmt.bufPrint(&fixture_name_buf, "block.{s}", .{entry.name["blocks[0].".len..]})
        else
            entry.name;

        const path = try std.fmt.allocPrint(
            alloc,
            "tests/fixtures/full_model_forward/param__{s}.ztlt",
            .{fixture_name},
        );
        defer alloc.free(path);

        var loaded = try oracle.loadTensor(alloc, io, path);
        defer loaded.deinit(alloc);

        if (loaded.data.len != entry.tensor.data.len) {
            std.debug.print("  param {s} size mismatch: loaded={d}, model={d}\n", .{
                entry.name, loaded.data.len, entry.tensor.data.len,
            });
            return error.ShapeMismatch;
        }
        @memcpy(entry.tensor.data, loaded.data);
    }

    // Load the token ids and run forward.
    var ids = try oracle.loadTensor(alloc, io, fixturePath("full_model_forward", "input_0.ztlt"));
    defer ids.deinit(alloc);

    var expect_logits = try oracle.loadTensor(alloc, io, fixturePath("full_model_forward", "output.ztlt"));
    defer expect_logits.deinit(alloc);

    // Forward only; no tape, no gradients.
    var logits = try model.forward(ids, null);
    defer logits.deinit(alloc);

    // Slightly looser tolerance because the forward passes through
    // many ops (embedding, LN, 4 matmuls, softmax, matmulBatch x2,
    // residuals, LN, MLP, residual, final LN, LM head) — f32
    // rounding compounds. 5e-4 absolute is still well under the
    // signal level for any reasonable logit.
    try oracle.expectClose(logits, expect_logits, .{ .rel_tol = 1e-3, .abs_tol = 5e-4 });
}


// -- Case: multihead_attention_3d -------------------------------------------
//
// Stage 8 Milestone 5. Parity target for the multi-head
// `CausalSelfAttention` introduced in Milestone 4. The fixture uses
// B=2, T=3, D=8, H=2 (d_head=4) with bias=true on all four Linears.
//
// Test flow: load the 9 input tensors (x + 4 * (W, b)), stuff them
// into a CausalSelfAttention built with matching config, run forward,
// run backward via sumAll loss, compare forward output and all 9
// gradients.

test "oracle multihead_attention_3d: forward and backward parity" {
    const alloc = std.testing.allocator;
    const io = try testIo();

    const CausalSelfAttention = ztl.nn.attention.CausalSelfAttention;
    const Rng = ztl.rng.Rng;

    var x = try oracle.loadTensor(alloc, io, fixturePath("multihead_attention_3d", "input_0.ztlt"));
    defer x.deinit(alloc);
    var w_q = try oracle.loadTensor(alloc, io, fixturePath("multihead_attention_3d", "input_1.ztlt"));
    defer w_q.deinit(alloc);
    var b_q = try oracle.loadTensor(alloc, io, fixturePath("multihead_attention_3d", "input_2.ztlt"));
    defer b_q.deinit(alloc);
    var w_k = try oracle.loadTensor(alloc, io, fixturePath("multihead_attention_3d", "input_3.ztlt"));
    defer w_k.deinit(alloc);
    var b_k = try oracle.loadTensor(alloc, io, fixturePath("multihead_attention_3d", "input_4.ztlt"));
    defer b_k.deinit(alloc);
    var w_v = try oracle.loadTensor(alloc, io, fixturePath("multihead_attention_3d", "input_5.ztlt"));
    defer w_v.deinit(alloc);
    var b_v = try oracle.loadTensor(alloc, io, fixturePath("multihead_attention_3d", "input_6.ztlt"));
    defer b_v.deinit(alloc);
    var w_o = try oracle.loadTensor(alloc, io, fixturePath("multihead_attention_3d", "input_7.ztlt"));
    defer w_o.deinit(alloc);
    var b_o = try oracle.loadTensor(alloc, io, fixturePath("multihead_attention_3d", "input_8.ztlt"));
    defer b_o.deinit(alloc);

    var expect_y = try oracle.loadTensor(alloc, io, fixturePath("multihead_attention_3d", "output.ztlt"));
    defer expect_y.deinit(alloc);
    var expect_dx = try oracle.loadTensor(alloc, io, fixturePath("multihead_attention_3d", "grad_input_0.ztlt"));
    defer expect_dx.deinit(alloc);
    var expect_dwq = try oracle.loadTensor(alloc, io, fixturePath("multihead_attention_3d", "grad_input_1.ztlt"));
    defer expect_dwq.deinit(alloc);
    var expect_dbq = try oracle.loadTensor(alloc, io, fixturePath("multihead_attention_3d", "grad_input_2.ztlt"));
    defer expect_dbq.deinit(alloc);
    var expect_dwk = try oracle.loadTensor(alloc, io, fixturePath("multihead_attention_3d", "grad_input_3.ztlt"));
    defer expect_dwk.deinit(alloc);
    var expect_dbk = try oracle.loadTensor(alloc, io, fixturePath("multihead_attention_3d", "grad_input_4.ztlt"));
    defer expect_dbk.deinit(alloc);
    var expect_dwv = try oracle.loadTensor(alloc, io, fixturePath("multihead_attention_3d", "grad_input_5.ztlt"));
    defer expect_dwv.deinit(alloc);
    var expect_dbv = try oracle.loadTensor(alloc, io, fixturePath("multihead_attention_3d", "grad_input_6.ztlt"));
    defer expect_dbv.deinit(alloc);
    var expect_dwo = try oracle.loadTensor(alloc, io, fixturePath("multihead_attention_3d", "grad_input_7.ztlt"));
    defer expect_dwo.deinit(alloc);
    var expect_dbo = try oracle.loadTensor(alloc, io, fixturePath("multihead_attention_3d", "grad_input_8.ztlt"));
    defer expect_dbo.deinit(alloc);

    const D = x.shape.dims[2];
    const T = x.shape.dims[1];
    const H: usize = 2;
    var rng = Rng.init(0);
    var attn = try CausalSelfAttention.init(alloc, D, H, T, true, &rng);
    defer attn.deinit();

    @memcpy(attn.w_q.weight.data, w_q.data);
    @memcpy(attn.w_q.bias.?.data, b_q.data);
    @memcpy(attn.w_k.weight.data, w_k.data);
    @memcpy(attn.w_k.bias.?.data, b_k.data);
    @memcpy(attn.w_v.weight.data, w_v.data);
    @memcpy(attn.w_v.bias.?.data, b_v.data);
    @memcpy(attn.w_o.weight.data, w_o.data);
    @memcpy(attn.w_o.bias.?.data, b_o.data);

    x.requires_grad = true;
    attn.w_q.weight.requires_grad = true;
    attn.w_q.bias.?.requires_grad = true;
    attn.w_k.weight.requires_grad = true;
    attn.w_k.bias.?.requires_grad = true;
    attn.w_v.weight.requires_grad = true;
    attn.w_v.bias.?.requires_grad = true;
    attn.w_o.weight.requires_grad = true;
    attn.w_o.bias.?.requires_grad = true;

    var tape = Tape.init(alloc);
    defer tape.deinit();
    _ = try tape.trackLeaf(&x);
    _ = try tape.trackLeaf(&attn.w_q.weight);
    _ = try tape.trackLeaf(&attn.w_q.bias.?);
    _ = try tape.trackLeaf(&attn.w_k.weight);
    _ = try tape.trackLeaf(&attn.w_k.bias.?);
    _ = try tape.trackLeaf(&attn.w_v.weight);
    _ = try tape.trackLeaf(&attn.w_v.bias.?);
    _ = try tape.trackLeaf(&attn.w_o.weight);
    _ = try tape.trackLeaf(&attn.w_o.bias.?);

    var y = try attn.forward(x, &tape);
    defer y.deinit(alloc);

    try oracle.expectClose(y, expect_y, .{ .rel_tol = 1e-4, .abs_tol = 1e-4 });

    try backwardThroughSum(&tape, &y);

    const bwd_tol = oracle.CloseOptions{ .rel_tol = 1e-3, .abs_tol = 5e-4 };
    try oracle.expectClose(x.grad.?.*, expect_dx, bwd_tol);
    try oracle.expectClose(attn.w_q.weight.grad.?.*, expect_dwq, bwd_tol);
    try oracle.expectClose(attn.w_q.bias.?.grad.?.*, expect_dbq, bwd_tol);
    try oracle.expectClose(attn.w_k.weight.grad.?.*, expect_dwk, bwd_tol);
    try oracle.expectClose(attn.w_k.bias.?.grad.?.*, expect_dbk, bwd_tol);
    try oracle.expectClose(attn.w_v.weight.grad.?.*, expect_dwv, bwd_tol);
    try oracle.expectClose(attn.w_v.bias.?.grad.?.*, expect_dbv, bwd_tol);
    try oracle.expectClose(attn.w_o.weight.grad.?.*, expect_dwo, bwd_tol);
    try oracle.expectClose(attn.w_o.bias.?.grad.?.*, expect_dbo, bwd_tol);
}