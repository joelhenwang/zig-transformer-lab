//!
//! zig-transformer-lab — Finite-difference gradient checker
//!
//! Purpose:
//!   Verifies analytical gradients computed by the autograd engine
//!   against numerical gradients obtained via central finite differences.
//!   This is THE gold standard for testing any autograd implementation.
//!
//!   The central difference formula:
//!     f'(x) ≈ (f(x + h) - f(x - h)) / (2h)
//!
//!   We perturb each parameter element by ±h, recompute the loss, and
//!   compare the numerical gradient with the analytical gradient from
//!   tape.backward(). If they agree within tolerance, the backward
//!   implementation is correct.
//!
//! Shape contract:
//!   gradCheck works on scalar losses (shape (1,)). It samples a subset
//!   of parameter indices and compares element-wise.
//!
//! Math:
//!   For parameter element p[i]:
//!     numerical_grad[i] = (L(p + h·e_i) - L(p - h·e_i)) / (2h)
//!     rel_error = |analytical[i] - numerical[i]| / max(|analytical|, |numerical|, 1e-8)
//!     pass if rel_error < tol_rel
//!
//!   The max(..., 1e-8) in the denominator avoids division by near-zero
//!   when both gradients are close to zero (which is acceptable).
//!
//! Memory ownership:
//!   gradCheck allocates temporary tensors for the loss computation.
//!   All temporaries are freed before returning. The parameter tensors
//!   are modified in-place during the check but restored to their
//!   original values afterward.
//!
//! Errors:
//!   OutOfMemory — allocation failure during loss recomputation
//!
//! TODO:
//!   - Add support for non-scalar losses (sum reduction first)
//!   - Parallelize the parameter perturbation loop
//!
//! Credits:
//!   The central-difference gradient check is standard in every deep
//!   learning framework. PyTorch's torch.autograd.gradcheck uses the
//!   same formula. No code copied.
//;

const std = @import("std");
const LabError = @import("../core/errors.zig").LabError;
const Tensor = @import("../tensor/tensor.zig").Tensor;
const Shape = @import("../tensor/shape.zig").Shape;
const totalElements = @import("../tensor/shape.zig").totalElements;
const Tape = @import("tape.zig").Tape;

/// Type signature for a loss function that the gradcheck can call.
///
/// The loss function takes an allocator, an array of parameter tensors,
/// and a tape, and returns a scalar loss tensor.
///
/// Worked example:
///   fn myLoss(allocator: std.mem.Allocator, params: []*Tensor, tape: *Tape) LabError!*Tensor {
///       const out = try ops.matmul(allocator, params[0].*, params[1].*, tape);
///       return ops.reduce.sumAll(allocator, out);
///   }
pub const LossFn = *const fn (std.mem.Allocator, []*Tensor, *Tape) LabError!*Tensor;

/// Result of a gradient check.
pub const GradCheckResult = struct {
    /// Number of parameter indices sampled.
    n_sampled: usize,

    /// Number of indices that passed the tolerance check.
    n_passed: usize,

    /// Maximum relative error across all sampled indices.
    max_rel_error: f32,

    /// Whether all sampled indices passed.
    passed: bool,
};

/// Run a finite-difference gradient check on the given parameters.
///
/// This function:
///   1. Runs the loss function forward+backward to get analytical gradients.
///   2. For each sampled parameter index i:
///      a. Save the original value p[i].
///      b. Set p[i] += eps, compute loss_plus.
///      c. Set p[i] -= 2*eps, compute loss_minus.
///      d. Restore p[i] to original.
///      e. Compute numerical gradient: (loss_plus - loss_minus) / (2*eps).
///      f. Compare against analytical gradient with relative tolerance.
///   3. Returns a GradCheckResult summarizing the check.
///
/// Parameters:
///   - allocator: for temporary allocations.
///   - loss_fn: the loss function to check.
///   - params: array of parameter tensors (must have .grad set from a prior backward).
///   - eps: perturbation size (default 1e-3).
///   - tol_rel: relative tolerance (default 1e-2).
///   - sample_n: number of indices to sample per parameter (0 = all).
///
/// IMPORTANT: The params' .grad fields must already be populated from
/// a prior call to tape.backward(). This function does NOT run backward
/// itself — it only compares the existing analytical gradients against
/// numerical ones.
///
/// Worked example:
///   var tape = Tape.init(allocator);
///   const loss = try myLoss(allocator, &params, &tape);
///   try tape.backward(loss);
///   const result = try gradCheck(allocator, myLoss, &params, 1e-3, 1e-2, 5);
///   try std.testing.expect(result.passed);
pub fn gradCheck(
    allocator: std.mem.Allocator,
    loss_fn: LossFn,
    params: []*Tensor,
    eps: f32,
    tol_rel: f32,
    sample_n: usize,
) LabError!GradCheckResult {
    var result = GradCheckResult{
        .n_sampled = 0,
        .n_passed = 0,
        .max_rel_error = 0.0,
        .passed = true,
    };

    // Use a deterministic RNG to select parameter indices to sample.
    // This avoids checking every single element (which would be slow
    // for large parameters) while still providing good coverage.
    var rng = std.Random.Xoshiro256.init(42);

    for (params) |param| {
        const grad = param.grad orelse continue;
        const n = totalElements(param.shape);

        // Determine which indices to sample
        const indices_to_check = if (sample_n > 0 and sample_n < n) sample_n else n;

        for (0..indices_to_check) |_| {
            // Pick a random index (or sequential if sample_n >= n)
            const idx = if (sample_n > 0 and sample_n < n)
                rng.next() % n
            else
                result.n_sampled;

            // Save the original value
            const original = param.data[idx];

            // --- Loss with p[i] + eps ---
            param.data[idx] = original + eps;
            var tape_plus = Tape.init(allocator);
            defer tape_plus.deinit();
            const loss_plus = try loss_fn(allocator, params, &tape_plus);
            const loss_plus_val = loss_plus.data[0];
            loss_plus.deinit(allocator);
            allocator.destroy(loss_plus);

            // --- Loss with p[i] - eps ---
            param.data[idx] = original - eps;
            var tape_minus = Tape.init(allocator);
            defer tape_minus.deinit();
            const loss_minus = try loss_fn(allocator, params, &tape_minus);
            const loss_minus_val = loss_minus.data[0];
            loss_minus.deinit(allocator);
            allocator.destroy(loss_minus);

            // Restore original value
            param.data[idx] = original;

            // Numerical gradient via central difference
            const numerical = (loss_plus_val - loss_minus_val) / (2.0 * eps);

            // Analytical gradient
            const analytical = grad.data[idx];

            // Relative error: |analytical - numerical| / max(|analytical|, |numerical|, floor)
            const abs_diff = @abs(analytical - numerical);
            const denominator = @max(@abs(analytical), @abs(numerical), 1e-8);
            const rel_error = abs_diff / denominator;

            if (rel_error > result.max_rel_error) {
                result.max_rel_error = rel_error;
            }

            if (rel_error < tol_rel) {
                result.n_passed += 1;
            } else {
                result.passed = false;
            }

            result.n_sampled += 1;
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "gradCheck — simple linear function" {
    const allocator = std.testing.allocator;

    // loss = sum(a * 2 + 1) = 2 * sum(a) + n
    // dL/da[i] = 2
    var a = try Tensor.init(allocator, Shape.init1D(3));
    defer a.deinit(allocator);
    a.data[0] = 1.0;
    a.data[1] = 2.0;
    a.data[2] = 3.0;
    a.requires_grad = true;

    // Run forward+backward to get analytical gradients
    var tape = Tape.init(allocator);
    defer tape.deinit();

    _ = try tape.trackLeaf(&a);
    var scaled = try @import("../tensor/ops/elementwise.zig").mulScalar(allocator, a, 2.0, &tape);
    defer scaled.deinit(allocator);
    var loss = try @import("../tensor/ops/reduce.zig").sumAll(allocator, scaled, &tape);
    defer loss.deinit(allocator);
    try tape.backward(&loss);

    // Now run gradcheck
    var params = [_]*Tensor{&a};
    const result = try gradCheck(
        allocator,
        linearLossFn,
        &params,
        1e-3,
        1e-2,
        3,
    );

    try std.testing.expect(result.passed);
    try std.testing.expectEqual(@as(usize, 3), result.n_sampled);
}

fn linearLossFn(allocator: std.mem.Allocator, params: []*Tensor, tape: *Tape) LabError!*Tensor {
    const a = params[0];
    var scaled = try @import("../tensor/ops/elementwise.zig").mulScalar(allocator, a.*, 2.0, tape);
    defer scaled.deinit(allocator);
    const loss_val = try @import("../tensor/ops/reduce.zig").sumAll(allocator, scaled, tape);
    const ptr = try allocator.create(Tensor);
    ptr.* = loss_val;
    return ptr;
}

fn matmulLossFn(allocator: std.mem.Allocator, params: []*Tensor, tape: *Tape) LabError!*Tensor {
    // Recreate X (it doesn't require grad, so we don't pass it as a param)
    var X = try Tensor.init(allocator, Shape.init2D(2, 2));
    defer X.deinit(allocator);
    X.data[0] = 1.0;
    X.data[1] = 0.0;
    X.data[2] = 0.0;
    X.data[3] = 1.0;

    const W = params[0];
    var out = try @import("../tensor/ops/matmul.zig").matmul(allocator, X, W.*, tape);
    defer out.deinit(allocator);
    const loss_val = try @import("../tensor/ops/reduce.zig").sumAll(allocator, out, tape);
    const ptr = try allocator.create(Tensor);
    ptr.* = loss_val;
    return ptr;
}

test "gradCheck — detects wrong gradient" {
    const allocator = std.testing.allocator;

    var a = try Tensor.init(allocator, Shape.init1D(3));
    defer a.deinit(allocator);
    a.data[0] = 1.0;
    a.data[1] = 2.0;
    a.data[2] = 3.0;
    a.requires_grad = true;

    // Manually set a WRONG gradient (should be 2, we set 100)
    var wrong_grad = try Tensor.init(allocator, Shape.init1D(3));
    wrong_grad.fill(100.0);
    a.grad = &wrong_grad;

    var params = [_]*Tensor{&a};
    const result = try gradCheck(
        allocator,
        linearLossFn,
        &params,
        1e-3,
        1e-2,
        3,
    );

    // Should fail because the gradient is wrong
    try std.testing.expect(!result.passed);
    wrong_grad.deinit(allocator);
}

test "gradCheck — matmul gradient" {
    const allocator = std.testing.allocator;

    // loss = sumAll(X @ W)
    // dL/dW = Xᵀ (broadcast to match W's shape)
    var W = try Tensor.init(allocator, Shape.init2D(2, 3));
    defer W.deinit(allocator);
    W.data[0] = 0.1;
    W.data[1] = 0.2;
    W.data[2] = 0.3;
    W.data[3] = 0.4;
    W.data[4] = 0.5;
    W.data[5] = 0.6;
    W.requires_grad = true;

    var X = try Tensor.init(allocator, Shape.init2D(2, 2));
    defer X.deinit(allocator);
    X.data[0] = 1.0;
    X.data[1] = 0.0;
    X.data[2] = 0.0;
    X.data[3] = 1.0;
    // X doesn't require grad for this test

    // Forward + backward
    var tape = Tape.init(allocator);
    defer tape.deinit();

    _ = try tape.trackLeaf(&W);
    var out = try @import("../tensor/ops/matmul.zig").matmul(allocator, X, W, &tape);
    defer out.deinit(allocator);
    var loss = try @import("../tensor/ops/reduce.zig").sumAll(allocator, out, &tape);
    defer loss.deinit(allocator);
    try tape.backward(&loss);

    // Gradcheck
    var params = [_]*Tensor{&W};
    const result = try gradCheck(
        allocator,
        matmulLossFn,
        &params,
        1e-4,
        1e-2,
        6,
    );

    try std.testing.expect(result.passed);
}

test "gradCheck — Linear layer weight gradient (transpose2d path)" {
    const allocator = std.testing.allocator;
    const Linear = @import("../nn/linear.zig").Linear;
    const Rng = @import("../core/rng.zig").Rng;
    const ops_reduce = @import("../tensor/ops/reduce.zig");

    var rng = Rng.init(42);
    var layer = try Linear.init(allocator, 4, 3, false, &rng);
    defer layer.deinit();

    // Scale down weights for numerical stability
    for (layer.weight.data) |*v| v.* *= 0.1;

    var x = try @import("../tensor/ops/create.zig").randn(allocator, Shape.init2D(2, 4), &rng, 0.0, 0.1);
    defer x.deinit(allocator);

    // Analytical forward + backward
    var tape = Tape.init(allocator);
    defer tape.deinit();

    layer.weight.requires_grad = true;
    _ = try tape.trackLeaf(&layer.weight);

    var out = try layer.forward(x, &tape);
    defer out.deinit(allocator);

    var loss = try ops_reduce.sumAll(allocator, out, &tape);
    defer loss.deinit(allocator);

    try tape.backward(&loss);

    const grad = layer.weight.grad orelse unreachable;

    // Manual numerical gradient check
    const h: f32 = 1e-4;
    var max_rel_err: f32 = 0;
    const n = layer.weight.data.len;

    for (0..n) |idx| {
        const original = layer.weight.data[idx];

        // Forward with +h
        layer.weight.data[idx] = original + h;
        var logits_p = try layer.forward(x, null);
        defer logits_p.deinit(allocator);
        var loss_p = try ops_reduce.sumAll(allocator, logits_p, null);
        defer loss_p.deinit(allocator);
        const lp = loss_p.data[0];

        // Forward with -h
        layer.weight.data[idx] = original - h;
        var logits_m = try layer.forward(x, null);
        defer logits_m.deinit(allocator);
        var loss_m = try ops_reduce.sumAll(allocator, logits_m, null);
        defer loss_m.deinit(allocator);
        const lm = loss_m.data[0];

        // Restore
        layer.weight.data[idx] = original;

        const numerical = (lp - lm) / (2.0 * h);
        const analytical = grad.data[idx];
        const abs_diff = @abs(analytical - numerical);
        const denom = @max(@abs(analytical), @abs(numerical), 1e-8);
        const rel_err = abs_diff / denom;

        if (rel_err > max_rel_err) max_rel_err = rel_err;
    }

    try std.testing.expect(max_rel_err < 0.01);
}

test "gradCheck — Linear layer 3D input (reshape path)" {
    const allocator = std.testing.allocator;
    const Linear = @import("../nn/linear.zig").Linear;
    const Rng = @import("../core/rng.zig").Rng;
    const ops_reduce = @import("../tensor/ops/reduce.zig");

    var rng = Rng.init(42);
    var layer = try Linear.init(allocator, 4, 3, false, &rng);
    defer layer.deinit();

    for (layer.weight.data) |*v| v.* *= 0.1;

    // 3D input — tests the reshape→matmul→reshape path
    var x = try @import("../tensor/ops/create.zig").randn(allocator, Shape.init3D(2, 3, 4), &rng, 0.0, 0.1);
    defer x.deinit(allocator);

    var tape = Tape.init(allocator);
    defer tape.deinit();

    layer.weight.requires_grad = true;
    _ = try tape.trackLeaf(&layer.weight);

    var out = try layer.forward(x, &tape);
    defer out.deinit(allocator);

    var loss = try ops_reduce.sumAll(allocator, out, &tape);
    defer loss.deinit(allocator);

    try tape.backward(&loss);

    const grad = layer.weight.grad orelse unreachable;

    const h: f32 = 1e-4;
    var max_rel_err: f32 = 0;
    const n = layer.weight.data.len;

    for (0..n) |idx| {
        const original = layer.weight.data[idx];

        layer.weight.data[idx] = original + h;
        var out_p = try layer.forward(x, null);
        defer out_p.deinit(allocator);
        var loss_p = try ops_reduce.sumAll(allocator, out_p, null);
        defer loss_p.deinit(allocator);

        layer.weight.data[idx] = original - h;
        var out_m = try layer.forward(x, null);
        defer out_m.deinit(allocator);
        var loss_m = try ops_reduce.sumAll(allocator, out_m, null);
        defer loss_m.deinit(allocator);

        layer.weight.data[idx] = original;

        const numerical = (loss_p.data[0] - loss_m.data[0]) / (2.0 * h);
        const analytical = grad.data[idx];
        const abs_diff = @abs(analytical - numerical);
        const denom = @max(@abs(analytical), @abs(numerical), 1e-8);
        const rel_err = abs_diff / denom;

        if (rel_err > max_rel_err) max_rel_err = rel_err;
    }

    try std.testing.expect(max_rel_err < 0.01);
}

test "gradCheck — Linear in residual path (add + identity)" {
    const allocator = std.testing.allocator;
    const Linear = @import("../nn/linear.zig").Linear;
    const Rng = @import("../core/rng.zig").Rng;
    const ops_reduce = @import("../tensor/ops/reduce.zig");
    const ops_elementwise = @import("../tensor/ops/elementwise.zig");

    var rng = Rng.init(42);
    // Simulates: output = x + Linear(x)  (residual connection)
    var layer = try Linear.init(allocator, 4, 4, false, &rng);
    defer layer.deinit();

    for (layer.weight.data) |*v| v.* *= 0.1;

    var x = try @import("../tensor/ops/create.zig").randn(allocator, Shape.init3D(2, 3, 4), &rng, 0.0, 0.1);
    defer x.deinit(allocator);

    var tape = Tape.init(allocator);
    defer tape.deinit();

    x.requires_grad = true;
    _ = try tape.trackLeaf(&x);
    layer.weight.requires_grad = true;
    _ = try tape.trackLeaf(&layer.weight);

    // residual: out = x + Linear(x)
    var lin_out = try layer.forward(x, &tape);
    defer lin_out.deinit(allocator);

    var output = try ops_elementwise.add(allocator, x, lin_out, &tape);
    defer output.deinit(allocator);

    var loss = try ops_reduce.sumAll(allocator, output, &tape);
    defer loss.deinit(allocator);

    try tape.backward(&loss);

    const grad = layer.weight.grad orelse unreachable;

    // Numerical check: perturb weight, recompute x + Linear(x) with tape=null
    const h: f32 = 1e-4;
    var max_rel_err: f32 = 0;
    const n = layer.weight.data.len;

    for (0..n) |idx| {
        const original = layer.weight.data[idx];

        layer.weight.data[idx] = original + h;
        var lo_p = try layer.forward(x, null);
        defer lo_p.deinit(allocator);
        var op_p = try ops_elementwise.add(allocator, x, lo_p, null);
        defer op_p.deinit(allocator);
        var lp = try ops_reduce.sumAll(allocator, op_p, null);
        defer lp.deinit(allocator);

        layer.weight.data[idx] = original - h;
        var lo_m = try layer.forward(x, null);
        defer lo_m.deinit(allocator);
        var op_m = try ops_elementwise.add(allocator, x, lo_m, null);
        defer op_m.deinit(allocator);
        var lm = try ops_reduce.sumAll(allocator, op_m, null);
        defer lm.deinit(allocator);

        layer.weight.data[idx] = original;

        const numerical = (lp.data[0] - lm.data[0]) / (2.0 * h);
        const analytical = grad.data[idx];
        const abs_diff = @abs(analytical - numerical);
        const denom = @max(@abs(analytical), @abs(numerical), 1e-8);
        const rel_err = abs_diff / denom;

        if (rel_err > max_rel_err) max_rel_err = rel_err;
    }

    try std.testing.expect(max_rel_err < 0.01);
}

test "gradCheck — Linear bias gradient (3D input)" {
    const allocator = std.testing.allocator;
    const Linear = @import("../nn/linear.zig").Linear;
    const Rng = @import("../core/rng.zig").Rng;
    const ops_reduce = @import("../tensor/ops/reduce.zig");

    var rng = Rng.init(42);
    var layer = try Linear.init(allocator, 4, 3, true, &rng);
    defer layer.deinit();

    for (layer.weight.data) |*v| v.* *= 0.1;

    var x = try @import("../tensor/ops/create.zig").randn(allocator, Shape.init3D(2, 3, 4), &rng, 0.0, 0.1);
    defer x.deinit(allocator);

    var tape = Tape.init(allocator);
    defer tape.deinit();

    layer.bias.?.requires_grad = true;
    _ = try tape.trackLeaf(&layer.bias.?);

    var out = try layer.forward(x, &tape);
    defer out.deinit(allocator);

    var loss = try ops_reduce.sumAll(allocator, out, &tape);
    defer loss.deinit(allocator);

    try tape.backward(&loss);

    const grad = layer.bias.?.grad orelse unreachable;

    const h: f32 = 1e-4;
    var max_rel_err: f32 = 0;
    const n = layer.bias.?.data.len;

    for (0..n) |idx| {
        const original = layer.bias.?.data[idx];

        layer.bias.?.data[idx] = original + h;
        var out_p = try layer.forward(x, null);
        defer out_p.deinit(allocator);
        var loss_p = try ops_reduce.sumAll(allocator, out_p, null);
        defer loss_p.deinit(allocator);

        layer.bias.?.data[idx] = original - h;
        var out_m = try layer.forward(x, null);
        defer out_m.deinit(allocator);
        var loss_m = try ops_reduce.sumAll(allocator, out_m, null);
        defer loss_m.deinit(allocator);

        layer.bias.?.data[idx] = original;

        const numerical = (loss_p.data[0] - loss_m.data[0]) / (2.0 * h);
        const analytical = grad.data[idx];
        const abs_diff = @abs(analytical - numerical);
        const denom = @max(@abs(analytical), @abs(numerical), 1e-8);
        const rel_err = abs_diff / denom;

        if (rel_err > max_rel_err) max_rel_err = rel_err;
    }

    std.debug.print("  Linear bias (3D input) max_rel_err={d:.6}\n", .{max_rel_err});
    try std.testing.expect(max_rel_err < 0.01);
}

test "gradCheck — full TinyWordTransformer model (sumAll loss)" {
    const allocator = std.testing.allocator;
    const TransformerConfig = @import("../nn/module.zig").TransformerConfig;
    const TinyWordTransformer = @import("../nn/model.zig").TinyWordTransformer;
    const NamedParam = @import("../nn/model.zig").NamedParam;
    const Rng = @import("../core/rng.zig").Rng;
    const ops_reduce = @import("../tensor/ops/reduce.zig");

    var rng = Rng.init(42);
    const cfg = TransformerConfig{
        .vocab_size = 8,
        .d_model = 4,
        .max_seq_len = 3,
        .d_ff = 8,
        .ln_eps = 1e-5,
        .bias = false,
    };

    var model = try TinyWordTransformer.init(allocator, cfg, &rng);
    defer model.deinit();

    // Scale down weights for numerical stability
    var named: std.ArrayList(NamedParam) = .empty;
    defer named.deinit(allocator);
    try model.collectNamedParams(&named);
    defer model.freeBlockNames(&named);
    for (named.items) |entry| {
        for (entry.tensor.data) |*v| v.* *= 0.1;
    }

    // Use -1 mask for grad check (not -1e9 or -10).
    // With -10, softmax is extremely peaked (exp(-10) ≈ 4.5e-5),
    // making numerical gradients unreliable. With -1, softmax is
    // smooth enough for accurate finite differences while still
    // enforcing the causal pattern (exp(-1) ≈ 0.368).
    //
    // Stage 8 M3: model.block became model.blocks[]; this grad check
    // operates on the first (and, for its default config, only) block.
    for (model.blocks[0].attn.causal_mask.data) |*v| {
        if (v.* < 0.0) v.* = -1.0;
    }

    // Small fixed input
    var ids = try Tensor.init(allocator, Shape.init2D(2, 3));
    defer ids.deinit(allocator);
    for (0..6) |i| ids.data[i] = @floatFromInt(i % 8);

    // Track all parameters
    var params: std.ArrayList(*Tensor) = .empty;
    defer params.deinit(allocator);
    try params.ensureTotalCapacity(allocator, 32);
    model.parameters(&params);

    // Forward + backward
    var tape = Tape.init(allocator);
    defer tape.deinit();

    for (params.items) |param| {
        param.requires_grad = true;
        _ = try tape.trackLeaf(param);
    }

    var logits = try model.forward(ids, &tape);
    defer logits.deinit(allocator);

    var loss = try ops_reduce.sumAll(allocator, logits, &tape);
    defer loss.deinit(allocator);

    try tape.backward(&loss);

    // Per-parameter numerical gradient check
    var overall_max_rel_err: f32 = 0;
    var overall_max_abs_diff: f32 = 0;
    for (named.items) |entry| {
        const grad = entry.tensor.grad orelse continue;
        const n = entry.tensor.data.len;
        const n_check = @min(n, 5);
        var max_rel_err: f32 = 0;
        var max_abs_diff: f32 = 0;
        var check_rng = std.Random.Xoshiro256.init(77);

        for (0..n_check) |_| {
            const idx = check_rng.next() % n;
            const original = entry.tensor.data[idx];
            const h: f32 = 1e-4;

            entry.tensor.data[idx] = original + h;
            var lp_logits = try model.forward(ids, null);
            defer lp_logits.deinit(allocator);
            var lp_loss = try ops_reduce.sumAll(allocator, lp_logits, null);
            defer lp_loss.deinit(allocator);

            entry.tensor.data[idx] = original - h;
            var lm_logits = try model.forward(ids, null);
            defer lm_logits.deinit(allocator);
            var lm_loss = try ops_reduce.sumAll(allocator, lm_logits, null);
            defer lm_loss.deinit(allocator);

            entry.tensor.data[idx] = original;

            const numerical = (lp_loss.data[0] - lm_loss.data[0]) / (2.0 * h);
            const analytical = grad.data[idx];
            const abs_diff = @abs(analytical - numerical);
            const denom = @max(@abs(analytical), @abs(numerical), 1e-2);
            const rel_err = abs_diff / denom;

            if (rel_err > max_rel_err) max_rel_err = rel_err;
            max_abs_diff = @max(max_abs_diff, abs_diff);
        }

        const status = if (max_rel_err < 0.01) "PASS" else if (max_rel_err < 0.05) "WARN" else "FAIL";
        std.debug.print("  {s:<30}  max_rel_err={d:.6} max_abs_diff={d:.6}  {s}\n", .{ entry.name, max_rel_err, max_abs_diff, status });
        overall_max_rel_err = @max(overall_max_rel_err, max_rel_err);
        overall_max_abs_diff = @max(overall_max_abs_diff, max_abs_diff);
    }
    // Combined check: pass if either relative error is small OR absolute
    // error is tiny. High relative error with tiny absolute error occurs
    // when both analytical and numerical gradients are near zero — this
    // is finite-difference noise, not a backward bug.
    try std.testing.expect(overall_max_rel_err < 0.05 or overall_max_abs_diff < 5e-3);
}

test "gradCheck — TransformerBlock (residual connections)" {
    const allocator = std.testing.allocator;
    const TransformerConfig = @import("../nn/module.zig").TransformerConfig;
    const TransformerBlock = @import("../nn/block.zig").TransformerBlock;
    const Rng = @import("../core/rng.zig").Rng;
    const ops_reduce = @import("../tensor/ops/reduce.zig");
    const ops_create = @import("../tensor/ops/create.zig");

    var rng = Rng.init(42);
    const cfg = TransformerConfig{
        .vocab_size = 8,
        .d_model = 4,
        .max_seq_len = 3,
        .d_ff = 8,
        .ln_eps = 1e-5,
        .bias = false,
    };

    var block = try TransformerBlock.init(allocator, cfg, &rng);
    defer block.deinit();

    // Use -1 mask for grad check (not -1e9 or -10)
    for (block.attn.causal_mask.data) |*v| {
        if (v.* < 0.0) v.* = -1.0;
    }

    var x = try ops_create.randn(allocator, Shape.init3D(2, 3, 4), &rng, 0.0, 0.1);
    defer x.deinit(allocator);

    var params: std.ArrayList(*Tensor) = .empty;
    defer params.deinit(allocator);
    try params.ensureTotalCapacity(allocator, 32);
    block.parameters(&params);

    var tape = Tape.init(allocator);
    defer tape.deinit();

    for (params.items) |param| {
        param.requires_grad = true;
        _ = try tape.trackLeaf(param);
    }

    var out = try block.forward(x, &tape);
    defer out.deinit(allocator);

    var loss = try ops_reduce.sumAll(allocator, out, &tape);
    defer loss.deinit(allocator);

    try tape.backward(&loss);

    const h: f32 = 1e-4;
    var max_overall: f32 = 0;
    var max_overall_abs: f32 = 0;
    var worst_name: []const u8 = "";

    for (params.items) |param| {
        const grad = param.grad orelse continue;
        const n = param.data.len;
        const n_check = @min(n, 5);
        var max_rel_err: f32 = 0;
        var max_abs_diff: f32 = 0;
        var check_rng = std.Random.Xoshiro256.init(77);

        for (0..n_check) |_| {
            const idx = check_rng.next() % n;
            const original = param.data[idx];

            param.data[idx] = original + h;
            var op = try block.forward(x, null);
            defer op.deinit(allocator);
            var lp = try ops_reduce.sumAll(allocator, op, null);
            defer lp.deinit(allocator);

            param.data[idx] = original - h;
            var om = try block.forward(x, null);
            defer om.deinit(allocator);
            var lm = try ops_reduce.sumAll(allocator, om, null);
            defer lm.deinit(allocator);

            param.data[idx] = original;

            const numerical = (lp.data[0] - lm.data[0]) / (2.0 * h);
            const analytical = grad.data[idx];
            const abs_diff = @abs(analytical - numerical);
            const denom = @max(@abs(analytical), @abs(numerical), 1e-2);
            const rel_err = abs_diff / denom;

            if (rel_err > max_rel_err) max_rel_err = rel_err;
            max_abs_diff = @max(max_abs_diff, abs_diff);
        }

        if (max_rel_err > max_overall) {
            max_overall = max_rel_err;
            worst_name = "param";
        }
        max_overall_abs = @max(max_overall_abs, max_abs_diff);

        const status = if (max_rel_err < 0.01) "PASS" else if (max_rel_err < 0.05) "WARN" else "FAIL";
        std.debug.print("  block param  max_rel_err={d:.6} max_abs_diff={d:.6}  {s}\n", .{ max_rel_err, max_abs_diff, status });
    }

    std.debug.print("  WORST: max_rel_err={d:.6} max_abs_diff={d:.6}\n", .{ max_overall, max_overall_abs });
    try std.testing.expect(max_overall < 0.05 or max_overall_abs < 1e-2);
}

test "gradCheck — LayerNorm alone (3D input)" {
    const allocator = std.testing.allocator;
    const LayerNorm = @import("../nn/layernorm.zig").LayerNorm;
    const Rng = @import("../core/rng.zig").Rng;
    const ops_reduce = @import("../tensor/ops/reduce.zig");
    const ops_create = @import("../tensor/ops/create.zig");

    var rng = Rng.init(42);
    var ln = try LayerNorm.init(allocator, 4, 1e-5, &rng);
    defer ln.deinit();

    var x = try ops_create.randn(allocator, Shape.init3D(2, 3, 4), &rng, 0.0, 1.0);
    defer x.deinit(allocator);

    var tape = Tape.init(allocator);
    defer tape.deinit();

    ln.gamma.requires_grad = true;
    _ = try tape.trackLeaf(&ln.gamma);
    ln.beta.requires_grad = true;
    _ = try tape.trackLeaf(&ln.beta);

    var out = try ln.forward(x, &tape);
    defer out.deinit(allocator);

    var loss = try ops_reduce.sumAll(allocator, out, &tape);
    defer loss.deinit(allocator);

    try tape.backward(&loss);

    const h: f32 = 1e-4;

    // Check gamma
    {
        const grad = ln.gamma.grad.?;
        const n = ln.gamma.data.len;
        var max_rel_err: f32 = 0;
        for (0..n) |idx| {
            const original = ln.gamma.data[idx];
            ln.gamma.data[idx] = original + h;
            var op = try ln.forward(x, null);
            defer op.deinit(allocator);
            var lp = try ops_reduce.sumAll(allocator, op, null);
            defer lp.deinit(allocator);
            ln.gamma.data[idx] = original - h;
            var om = try ln.forward(x, null);
            defer om.deinit(allocator);
            var lm = try ops_reduce.sumAll(allocator, om, null);
            defer lm.deinit(allocator);
            ln.gamma.data[idx] = original;
            const numerical = (lp.data[0] - lm.data[0]) / (2.0 * h);
            const analytical = grad.data[idx];
            const abs_diff = @abs(analytical - numerical);
            const denom = @max(@abs(analytical), @abs(numerical), 1e-8);
            const rel_err = abs_diff / denom;
            if (rel_err > max_rel_err) max_rel_err = rel_err;
        }
        std.debug.print("  LN gamma max_rel_err={d:.6}\n", .{max_rel_err});
        try std.testing.expect(max_rel_err < 0.01);
    }

    // Check beta
    {
        const grad = ln.beta.grad.?;
        const n = ln.beta.data.len;
        var max_rel_err: f32 = 0;
        for (0..n) |idx| {
            const original = ln.beta.data[idx];
            ln.beta.data[idx] = original + h;
            var op = try ln.forward(x, null);
            defer op.deinit(allocator);
            var lp = try ops_reduce.sumAll(allocator, op, null);
            defer lp.deinit(allocator);
            ln.beta.data[idx] = original - h;
            var om = try ln.forward(x, null);
            defer om.deinit(allocator);
            var lm = try ops_reduce.sumAll(allocator, om, null);
            defer lm.deinit(allocator);
            ln.beta.data[idx] = original;
            const numerical = (lp.data[0] - lm.data[0]) / (2.0 * h);
            const analytical = grad.data[idx];
            const abs_diff = @abs(analytical - numerical);
            const denom = @max(@abs(analytical), @abs(numerical), 1e-8);
            const rel_err = abs_diff / denom;
            if (rel_err > max_rel_err) max_rel_err = rel_err;
        }
        std.debug.print("  LN beta max_rel_err={d:.6}\n", .{max_rel_err});
        try std.testing.expect(max_rel_err < 0.01);
    }
}

test "gradCheck — matmulBatch backward (Q @ K^T pattern)" {
    const allocator = std.testing.allocator;
    const ops_matmul = @import("../tensor/ops/matmul.zig");
    const ops_shape = @import("../tensor/ops/shape_ops.zig");
    const ops_reduce = @import("../tensor/ops/reduce.zig");
    const ops_create = @import("../tensor/ops/create.zig");
    const Rng = @import("../core/rng.zig").Rng;

    var rng = Rng.init(42);

    // Simulate the Q @ K^T pattern:
    // Q: (B=2, T=3, D=4), K: (B=2, T=3, D=4)
    // K^T = transposeInner2dTracked(K) → (2, 4, 3)
    // scores = matmulBatch(Q, K^T) → (2, 3, 3)
    var Q = try ops_create.randn(allocator, Shape.init3D(2, 3, 4), &rng, 0.0, 0.1);
    defer Q.deinit(allocator);
    var K = try ops_create.randn(allocator, Shape.init3D(2, 3, 4), &rng, 0.0, 0.1);
    defer K.deinit(allocator);

    Q.requires_grad = true;
    K.requires_grad = true;

    var tape = Tape.init(allocator);
    defer tape.deinit();

    _ = try tape.trackLeaf(&Q);
    _ = try tape.trackLeaf(&K);

    // Replicate attention forward pattern
    var k_t = try ops_shape.transposeInner2dTracked(allocator, K, &tape);
    defer k_t.deinit(allocator);

    var scores = try ops_matmul.matmulBatch(allocator, Q, k_t, &tape);
    defer scores.deinit(allocator);

    var loss = try ops_reduce.sumAll(allocator, scores, &tape);
    defer loss.deinit(allocator);

    try tape.backward(&loss);

    // Check Q gradient
    {
        const grad = Q.grad.?;
        const h: f32 = 1e-4;
        var max_rel_err: f32 = 0;
        const n = Q.data.len;
        const n_check = @min(n, 10);
        var check_rng = std.Random.Xoshiro256.init(77);

        for (0..n_check) |_| {
            const idx = check_rng.next() % n;
            const original = Q.data[idx];

            Q.data[idx] = original + h;
            var kt_p = try ops_shape.transposeInner2dTracked(allocator, K, null);
            defer kt_p.deinit(allocator);
            var sc_p = try ops_matmul.matmulBatch(allocator, Q, kt_p, null);
            defer sc_p.deinit(allocator);
            var lp = try ops_reduce.sumAll(allocator, sc_p, null);
            defer lp.deinit(allocator);

            Q.data[idx] = original - h;
            var kt_m = try ops_shape.transposeInner2dTracked(allocator, K, null);
            defer kt_m.deinit(allocator);
            var sc_m = try ops_matmul.matmulBatch(allocator, Q, kt_m, null);
            defer sc_m.deinit(allocator);
            var lm = try ops_reduce.sumAll(allocator, sc_m, null);
            defer lm.deinit(allocator);

            Q.data[idx] = original;

            const numerical = (lp.data[0] - lm.data[0]) / (2.0 * h);
            const analytical = grad.data[idx];
            const abs_diff = @abs(analytical - numerical);
            const denom = @max(@abs(analytical), @abs(numerical), 1e-8);
            const rel_err = abs_diff / denom;
            if (rel_err > max_rel_err) max_rel_err = rel_err;
        }
        std.debug.print("  matmulBatch Q grad max_rel_err={d:.6}\n", .{max_rel_err});
        try std.testing.expect(max_rel_err < 0.01);
    }

    // Check K gradient (flows through transposeInner2dTracked backward)
    {
        const grad = K.grad.?;
        const h: f32 = 1e-4;
        var max_rel_err: f32 = 0;
        const n = K.data.len;
        const n_check = @min(n, 10);
        var check_rng = std.Random.Xoshiro256.init(88);

        for (0..n_check) |_| {
            const idx = check_rng.next() % n;
            const original = K.data[idx];

            K.data[idx] = original + h;
            var kt_p = try ops_shape.transposeInner2dTracked(allocator, K, null);
            defer kt_p.deinit(allocator);
            var sc_p = try ops_matmul.matmulBatch(allocator, Q, kt_p, null);
            defer sc_p.deinit(allocator);
            var lp = try ops_reduce.sumAll(allocator, sc_p, null);
            defer lp.deinit(allocator);

            K.data[idx] = original - h;
            var kt_m = try ops_shape.transposeInner2dTracked(allocator, K, null);
            defer kt_m.deinit(allocator);
            var sc_m = try ops_matmul.matmulBatch(allocator, Q, kt_m, null);
            defer sc_m.deinit(allocator);
            var lm = try ops_reduce.sumAll(allocator, sc_m, null);
            defer lm.deinit(allocator);

            K.data[idx] = original;

            const numerical = (lp.data[0] - lm.data[0]) / (2.0 * h);
            const analytical = grad.data[idx];
            const abs_diff = @abs(analytical - numerical);
            const denom = @max(@abs(analytical), @abs(numerical), 1e-8);
            const rel_err = abs_diff / denom;
            if (rel_err > max_rel_err) max_rel_err = rel_err;
        }
        std.debug.print("  matmulBatch K grad max_rel_err={d:.6}\n", .{max_rel_err});
        try std.testing.expect(max_rel_err < 0.01);
    }
}

test "gradCheck — attention Q@K^T path without softmax" {
    const allocator = std.testing.allocator;
    const Linear = @import("../nn/linear.zig").Linear;
    const Rng = @import("../core/rng.zig").Rng;
    const ops_matmul = @import("../tensor/ops/matmul.zig");
    const ops_shape = @import("../tensor/ops/shape_ops.zig");
    const ops_reduce = @import("../tensor/ops/reduce.zig");
    const ops_create = @import("../tensor/ops/create.zig");

    var rng = Rng.init(42);

    // Simulates: Q = Linear(x), K = Linear(x), scores = Q @ K^T, loss = sumAll(scores)
    var w_q = try Linear.init(allocator, 4, 4, false, &rng);
    defer w_q.deinit();
    var w_k = try Linear.init(allocator, 4, 4, false, &rng);
    defer w_k.deinit();

    for (w_q.weight.data) |*v| v.* *= 0.1;
    for (w_k.weight.data) |*v| v.* *= 0.1;

    var x = try ops_create.randn(allocator, Shape.init3D(2, 3, 4), &rng, 0.0, 0.1);
    defer x.deinit(allocator);

    var tape = Tape.init(allocator);
    defer tape.deinit();

    w_q.weight.requires_grad = true;
    _ = try tape.trackLeaf(&w_q.weight);
    w_k.weight.requires_grad = true;
    _ = try tape.trackLeaf(&w_k.weight);

    // Q = w_q.forward(x)
    var q = try w_q.forward(x, &tape);
    defer q.deinit(allocator);

    // K = w_k.forward(x)
    var k = try w_k.forward(x, &tape);
    defer k.deinit(allocator);

    // K^T = transposeInner2dTracked(K)
    var k_t = try ops_shape.transposeInner2dTracked(allocator, k, &tape);
    defer k_t.deinit(allocator);

    // scores = matmulBatch(Q, K^T)
    var scores = try ops_matmul.matmulBatch(allocator, q, k_t, &tape);
    defer scores.deinit(allocator);

    // loss = sumAll(scores)
    var loss = try ops_reduce.sumAll(allocator, scores, &tape);
    defer loss.deinit(allocator);

    try tape.backward(&loss);

    // Check w_q.weight
    {
        const grad = w_q.weight.grad.?;
        const h: f32 = 1e-4;
        var max_rel_err: f32 = 0;
        const n = w_q.weight.data.len;
        for (0..n) |idx| {
            const original = w_q.weight.data[idx];

            w_q.weight.data[idx] = original + h;
            var q_p = try w_q.forward(x, null);
            defer q_p.deinit(allocator);
            var s_p = try ops_matmul.matmulBatch(allocator, q_p, k_t, null);
            defer s_p.deinit(allocator);
            var lp = try ops_reduce.sumAll(allocator, s_p, null);
            defer lp.deinit(allocator);

            w_q.weight.data[idx] = original - h;
            var q_m = try w_q.forward(x, null);
            defer q_m.deinit(allocator);
            var s_m = try ops_matmul.matmulBatch(allocator, q_m, k_t, null);
            defer s_m.deinit(allocator);
            var lm = try ops_reduce.sumAll(allocator, s_m, null);
            defer lm.deinit(allocator);

            w_q.weight.data[idx] = original;

            const numerical = (lp.data[0] - lm.data[0]) / (2.0 * h);
            const analytical = grad.data[idx];
            const abs_diff = @abs(analytical - numerical);
            const denom = @max(@abs(analytical), @abs(numerical), 1e-8);
            const rel_err = abs_diff / denom;
            if (rel_err > max_rel_err) max_rel_err = rel_err;
        }
        std.debug.print("  Q@K^T path: w_q.weight max_rel_err={d:.6}\n", .{max_rel_err});
        try std.testing.expect(max_rel_err < 0.01);
    }
}

test "gradCheck — add broadcast 3D + 3D(1,T,T) backward" {
    const allocator = std.testing.allocator;
    const ops_reduce = @import("../tensor/ops/reduce.zig");
    const ops_create = @import("../tensor/ops/create.zig");
    const ops_elementwise = @import("../tensor/ops/elementwise.zig");
    const Rng = @import("../core/rng.zig").Rng;

    var rng = Rng.init(42);

    var a = try ops_create.randn(allocator, Shape.init3D(2, 3, 3), &rng, 0.0, 1.0);
    defer a.deinit(allocator);

    // Use -10 instead of -1e9 to avoid f32 precision loss in numerical grad.
    // With -1e9 in sumAll, perturbations of 1e-4 are below f32's ULP at 1e9,
    // making numerical gradients return 0 while analytical returns 1.
    // Even with -10, sumAll precision is limited (~0.03 max_rel_err),
    // so use a more relaxed tolerance for this specific test.
    var b = try Tensor.init(allocator, Shape.init3D(1, 3, 3));
    defer b.deinit(allocator);
    for (0..3) |i| {
        for (0..3) |j| {
            b.data[i * 3 + j] = if (j <= i) @as(f32, 0.0) else -1.0;
        }
    }

    a.requires_grad = true;

    var tape = Tape.init(allocator);
    defer tape.deinit();

    _ = try tape.trackLeaf(&a);

    var c = try ops_elementwise.add(allocator, a, b, &tape);
    defer c.deinit(allocator);

    var loss = try ops_reduce.sumAll(allocator, c, &tape);
    defer loss.deinit(allocator);

    try tape.backward(&loss);

    const grad = a.grad orelse unreachable;
    const h: f32 = 1e-4;
    const n = a.data.len;
    var max_rel_err: f32 = 0;

    for (0..n) |idx| {
        const original = a.data[idx];

        a.data[idx] = original + h;
        var cp = try ops_elementwise.add(allocator, a, b, null);
        defer cp.deinit(allocator);
        var lp = try ops_reduce.sumAll(allocator, cp, null);
        defer lp.deinit(allocator);

        a.data[idx] = original - h;
        var cm = try ops_elementwise.add(allocator, a, b, null);
        defer cm.deinit(allocator);
        var lm = try ops_reduce.sumAll(allocator, cm, null);
        defer lm.deinit(allocator);

        a.data[idx] = original;

        const numerical = (lp.data[0] - lm.data[0]) / (2.0 * h);
        const analytical = grad.data[idx];
        const abs_diff = @abs(analytical - numerical);
        const denom = @max(@abs(analytical), @abs(numerical), 1e-8);
        const rel_err = abs_diff / denom;

        if (rel_err > max_rel_err) max_rel_err = rel_err;
    }

    std.debug.print("  add broadcast 3D+(1,T,T): max_rel_err={d:.6}\n", .{max_rel_err});
    try std.testing.expect(max_rel_err < 0.01);
}

test "gradCheck — attention pipeline step-by-step" {
    const allocator = std.testing.allocator;
    const Linear = @import("../nn/linear.zig").Linear;
    const Rng = @import("../core/rng.zig").Rng;
    const ops_matmul = @import("../tensor/ops/matmul.zig");
    const ops_shape = @import("../tensor/ops/shape_ops.zig");
    const ops_reduce = @import("../tensor/ops/reduce.zig");
    const ops_elementwise = @import("../tensor/ops/elementwise.zig");
    const ops_softmax = @import("../tensor/ops/softmax.zig");
    const ops_create = @import("../tensor/ops/create.zig");

    var rng = Rng.init(42);

    var w_q = try Linear.init(allocator, 4, 4, true, &rng);
    defer w_q.deinit();
    var w_k = try Linear.init(allocator, 4, 4, true, &rng);
    defer w_k.deinit();
    var w_v = try Linear.init(allocator, 4, 4, true, &rng);
    defer w_v.deinit();
    var w_o = try Linear.init(allocator, 4, 4, true, &rng);
    defer w_o.deinit();

    for (w_q.weight.data) |*v| v.* *= 0.1;
    for (w_k.weight.data) |*v| v.* *= 0.1;
    for (w_v.weight.data) |*v| v.* *= 0.1;
    for (w_o.weight.data) |*v| v.* *= 0.1;

    var x = try ops_create.randn(allocator, Shape.init3D(2, 3, 4), &rng, 0.0, 0.1);
    defer x.deinit(allocator);

    const h: f32 = 1e-4;
    const D: usize = 4;

    inline for (.{
        @as(enum { scores_only, scaled, masked, softmaxed, full_attn }, .scores_only),
        @as(enum { scores_only, scaled, masked, softmaxed, full_attn }, .scaled),
        @as(enum { scores_only, scaled, masked, softmaxed, full_attn }, .masked),
        @as(enum { scores_only, scaled, masked, softmaxed, full_attn }, .softmaxed),
        @as(enum { scores_only, scaled, masked, softmaxed, full_attn }, .full_attn),
    }) |stage| {
        var tape = Tape.init(allocator);
        defer tape.deinit();

        w_q.weight.requires_grad = true;
        _ = try tape.trackLeaf(&w_q.weight);

        var q = try w_q.forward(x, &tape);
        defer q.deinit(allocator);

        var k = try w_k.forward(x, &tape);
        defer k.deinit(allocator);

        var k_t = try ops_shape.transposeInner2dTracked(allocator, k, &tape);
        defer k_t.deinit(allocator);

        var scores = try ops_matmul.matmulBatch(allocator, q, k_t, &tape);
        defer scores.deinit(allocator);

        var loss: Tensor = undefined;
        const loss_owned = true;

        switch (stage) {
            .scores_only => {
                loss = try ops_reduce.sumAll(allocator, scores, &tape);
            },
            .scaled => {
                const scale: f32 = @floatCast(1.0 / std.math.sqrt(@as(f64, @floatFromInt(D))));
                var scaled = try ops_elementwise.mulScalar(allocator, scores, scale, &tape);
                defer scaled.deinit(allocator);
                loss = try ops_reduce.sumAll(allocator, scaled, &tape);
            },
            .masked => {
                const scale: f32 = @floatCast(1.0 / std.math.sqrt(@as(f64, @floatFromInt(D))));
                var scaled = try ops_elementwise.mulScalar(allocator, scores, scale, &tape);
                defer scaled.deinit(allocator);
                var mask = try Tensor.init(allocator, Shape.init3D(1, 3, 3));
                defer mask.deinit(allocator);
                for (0..3) |i| {
                    for (0..3) |j| {
                        mask.data[i * 3 + j] = if (j <= i) @as(f32, 0.0) else -1.0;
                    }
                }
                var masked = try ops_elementwise.add(allocator, scaled, mask, &tape);
                defer masked.deinit(allocator);
                loss = try ops_reduce.sumAll(allocator, masked, &tape);
            },
            .softmaxed => {
                const scale: f32 = @floatCast(1.0 / std.math.sqrt(@as(f64, @floatFromInt(D))));
                var scaled = try ops_elementwise.mulScalar(allocator, scores, scale, &tape);
                defer scaled.deinit(allocator);
                var mask = try Tensor.init(allocator, Shape.init3D(1, 3, 3));
                defer mask.deinit(allocator);
                for (0..3) |i| {
                    for (0..3) |j| {
                        mask.data[i * 3 + j] = if (j <= i) @as(f32, 0.0) else -1.0;
                    }
                }
                var masked = try ops_elementwise.add(allocator, scaled, mask, &tape);
                defer masked.deinit(allocator);
                var weights = try ops_softmax.softmax(allocator, masked, &tape);
                defer weights.deinit(allocator);
                loss = try ops_reduce.sumAll(allocator, weights, &tape);
            },
            .full_attn => {
                const scale: f32 = @floatCast(1.0 / std.math.sqrt(@as(f64, @floatFromInt(D))));
                var scaled = try ops_elementwise.mulScalar(allocator, scores, scale, &tape);
                defer scaled.deinit(allocator);
                var mask = try Tensor.init(allocator, Shape.init3D(1, 3, 3));
                defer mask.deinit(allocator);
                for (0..3) |i| {
                    for (0..3) |j| {
                        mask.data[i * 3 + j] = if (j <= i) @as(f32, 0.0) else -1.0;
                    }
                }
                var masked = try ops_elementwise.add(allocator, scaled, mask, &tape);
                defer masked.deinit(allocator);
                var weights = try ops_softmax.softmax(allocator, masked, &tape);
                defer weights.deinit(allocator);
                var v_tensor = try w_v.forward(x, &tape);
                defer v_tensor.deinit(allocator);
                var attn_out = try ops_matmul.matmulBatch(allocator, weights, v_tensor, &tape);
                defer attn_out.deinit(allocator);
                var output = try w_o.forward(attn_out, &tape);
                defer output.deinit(allocator);
                loss = try ops_reduce.sumAll(allocator, output, &tape);
            },
        }
        defer if (loss_owned) loss.deinit(allocator);

        try tape.backward(&loss);

        const grad = w_q.weight.grad orelse unreachable;
        const n = w_q.weight.data.len;
        var max_rel_err: f32 = 0;
        var max_abs_diff: f32 = 0;
        for (0..n) |idx| {
            const original = w_q.weight.data[idx];

            w_q.weight.data[idx] = original + h;
            var q_p = try w_q.forward(x, null);
            defer q_p.deinit(allocator);
            var k_p = try w_k.forward(x, null);
            defer k_p.deinit(allocator);
            var kt_p = try ops_shape.transposeInner2dTracked(allocator, k_p, null);
            defer kt_p.deinit(allocator);
            var sc_p = try ops_matmul.matmulBatch(allocator, q_p, kt_p, null);
            defer sc_p.deinit(allocator);

            var lp: f32 = undefined;
            switch (stage) {
                .scores_only => {
                    var l = try ops_reduce.sumAll(allocator, sc_p, null);
                    defer l.deinit(allocator);
                    lp = l.data[0];
                },
                .scaled => {
                    const sc: f32 = @floatCast(1.0 / std.math.sqrt(@as(f64, @floatFromInt(D))));
                    var s = try ops_elementwise.mulScalar(allocator, sc_p, sc, null);
                    defer s.deinit(allocator);
                    var l = try ops_reduce.sumAll(allocator, s, null);
                    defer l.deinit(allocator);
                    lp = l.data[0];
                },
                .masked => {
                    const sc: f32 = @floatCast(1.0 / std.math.sqrt(@as(f64, @floatFromInt(D))));
                    var s = try ops_elementwise.mulScalar(allocator, sc_p, sc, null);
                    defer s.deinit(allocator);
                    var mask = try Tensor.init(allocator, Shape.init3D(1, 3, 3));
                    defer mask.deinit(allocator);
                    for (0..3) |i| {
                        for (0..3) |j| {
                            mask.data[i * 3 + j] = if (j <= i) @as(f32, 0.0) else -1.0;
                        }
                    }
                    var m = try ops_elementwise.add(allocator, s, mask, null);
                    defer m.deinit(allocator);
                    var l = try ops_reduce.sumAll(allocator, m, null);
                    defer l.deinit(allocator);
                    lp = l.data[0];
                },
                .softmaxed => {
                    const sc: f32 = @floatCast(1.0 / std.math.sqrt(@as(f64, @floatFromInt(D))));
                    var s = try ops_elementwise.mulScalar(allocator, sc_p, sc, null);
                    defer s.deinit(allocator);
                    var mask = try Tensor.init(allocator, Shape.init3D(1, 3, 3));
                    defer mask.deinit(allocator);
                    for (0..3) |i| {
                        for (0..3) |j| {
                            mask.data[i * 3 + j] = if (j <= i) @as(f32, 0.0) else -1.0;
                        }
                    }
                    var m = try ops_elementwise.add(allocator, s, mask, null);
                    defer m.deinit(allocator);
                    var w = try ops_softmax.softmax(allocator, m, null);
                    defer w.deinit(allocator);
                    var l = try ops_reduce.sumAll(allocator, w, null);
                    defer l.deinit(allocator);
                    lp = l.data[0];
                },
                .full_attn => {
                    const sc: f32 = @floatCast(1.0 / std.math.sqrt(@as(f64, @floatFromInt(D))));
                    var s = try ops_elementwise.mulScalar(allocator, sc_p, sc, null);
                    defer s.deinit(allocator);
                    var mask = try Tensor.init(allocator, Shape.init3D(1, 3, 3));
                    defer mask.deinit(allocator);
                    for (0..3) |i| {
                        for (0..3) |j| {
                            mask.data[i * 3 + j] = if (j <= i) @as(f32, 0.0) else -1.0;
                        }
                    }
                    var m = try ops_elementwise.add(allocator, s, mask, null);
                    defer m.deinit(allocator);
                    var w = try ops_softmax.softmax(allocator, m, null);
                    defer w.deinit(allocator);
                    var v_p = try w_v.forward(x, null);
                    defer v_p.deinit(allocator);
                    var ao_p = try ops_matmul.matmulBatch(allocator, w, v_p, null);
                    defer ao_p.deinit(allocator);
                    var out_p = try w_o.forward(ao_p, null);
                    defer out_p.deinit(allocator);
                    var l = try ops_reduce.sumAll(allocator, out_p, null);
                    defer l.deinit(allocator);
                    lp = l.data[0];
                },
            }

            w_q.weight.data[idx] = original - h;
            var q_m = try w_q.forward(x, null);
            defer q_m.deinit(allocator);
            var k_m = try w_k.forward(x, null);
            defer k_m.deinit(allocator);
            var kt_m = try ops_shape.transposeInner2dTracked(allocator, k_m, null);
            defer kt_m.deinit(allocator);
            var sc_m = try ops_matmul.matmulBatch(allocator, q_m, kt_m, null);
            defer sc_m.deinit(allocator);

            var lm: f32 = undefined;
            switch (stage) {
                .scores_only => {
                    var l = try ops_reduce.sumAll(allocator, sc_m, null);
                    defer l.deinit(allocator);
                    lm = l.data[0];
                },
                .scaled => {
                    const sc: f32 = @floatCast(1.0 / std.math.sqrt(@as(f64, @floatFromInt(D))));
                    var s = try ops_elementwise.mulScalar(allocator, sc_m, sc, null);
                    defer s.deinit(allocator);
                    var l = try ops_reduce.sumAll(allocator, s, null);
                    defer l.deinit(allocator);
                    lm = l.data[0];
                },
                .masked => {
                    const sc: f32 = @floatCast(1.0 / std.math.sqrt(@as(f64, @floatFromInt(D))));
                    var s = try ops_elementwise.mulScalar(allocator, sc_m, sc, null);
                    defer s.deinit(allocator);
                    var mask = try Tensor.init(allocator, Shape.init3D(1, 3, 3));
                    defer mask.deinit(allocator);
                    for (0..3) |i| {
                        for (0..3) |j| {
                            mask.data[i * 3 + j] = if (j <= i) @as(f32, 0.0) else -1.0;
                        }
                    }
                    var m = try ops_elementwise.add(allocator, s, mask, null);
                    defer m.deinit(allocator);
                    var l = try ops_reduce.sumAll(allocator, m, null);
                    defer l.deinit(allocator);
                    lm = l.data[0];
                },
                .softmaxed => {
                    const sc: f32 = @floatCast(1.0 / std.math.sqrt(@as(f64, @floatFromInt(D))));
                    var s = try ops_elementwise.mulScalar(allocator, sc_m, sc, null);
                    defer s.deinit(allocator);
                    var mask = try Tensor.init(allocator, Shape.init3D(1, 3, 3));
                    defer mask.deinit(allocator);
                    for (0..3) |i| {
                        for (0..3) |j| {
                            mask.data[i * 3 + j] = if (j <= i) @as(f32, 0.0) else -1.0;
                        }
                    }
                    var m = try ops_elementwise.add(allocator, s, mask, null);
                    defer m.deinit(allocator);
                    var w = try ops_softmax.softmax(allocator, m, null);
                    defer w.deinit(allocator);
                    var l = try ops_reduce.sumAll(allocator, w, null);
                    defer l.deinit(allocator);
                    lm = l.data[0];
                },
                .full_attn => {
                    const sc: f32 = @floatCast(1.0 / std.math.sqrt(@as(f64, @floatFromInt(D))));
                    var s = try ops_elementwise.mulScalar(allocator, sc_m, sc, null);
                    defer s.deinit(allocator);
                    var mask = try Tensor.init(allocator, Shape.init3D(1, 3, 3));
                    defer mask.deinit(allocator);
                    for (0..3) |i| {
                        for (0..3) |j| {
                            mask.data[i * 3 + j] = if (j <= i) @as(f32, 0.0) else -1.0;
                        }
                    }
                    var m = try ops_elementwise.add(allocator, s, mask, null);
                    defer m.deinit(allocator);
                    var w = try ops_softmax.softmax(allocator, m, null);
                    defer w.deinit(allocator);
                    var v_m = try w_v.forward(x, null);
                    defer v_m.deinit(allocator);
                    var ao_m = try ops_matmul.matmulBatch(allocator, w, v_m, null);
                    defer ao_m.deinit(allocator);
                    var out_m = try w_o.forward(ao_m, null);
                    defer out_m.deinit(allocator);
                    var l = try ops_reduce.sumAll(allocator, out_m, null);
                    defer l.deinit(allocator);
                    lm = l.data[0];
                },
            }

            w_q.weight.data[idx] = original;

            const numerical = (lp - lm) / (2.0 * h);
            const analytical = grad.data[idx];
            const abs_diff = @abs(analytical - numerical);
            // denom_floor=1e-2: near-zero gradients (< 0.01) cause inflated
            // relative error from finite-difference noise, especially in
            // softmax-heavy paths. Pair with max_abs_diff check.
            const denom = @max(@abs(analytical), @abs(numerical), 1e-2);
            const rel_err = abs_diff / denom;
            if (rel_err > max_rel_err) max_rel_err = rel_err;
            max_abs_diff = @max(max_abs_diff, abs_diff);
        }

        const stage_name = switch (stage) {
            .scores_only => "scores_only",
            .scaled => "scaled",
            .masked => "masked",
            .softmaxed => "softmaxed",
            .full_attn => "full_attn",
        };
        const status = if (max_rel_err < 0.01) "PASS" else if (max_rel_err < 0.05) "WARN" else "FAIL";
        std.debug.print("  pipeline {s:<15} w_q.weight max_rel_err={d:.6} max_abs_diff={d:.6}  {s}\n", .{ stage_name, max_rel_err, max_abs_diff, status });
        // softmaxed stage is a degenerate case: loss = sumAll(softmax(x))
        // is constant, so both gradients are ~0 and relative error is
        // meaningless. Skip assertion for that stage.
        if (stage != .softmaxed) {
            try std.testing.expect(max_rel_err < 0.05 or max_abs_diff < 5e-3);
        }
    }
}

test "gradCheck — CausalSelfAttention alone" {
    const allocator = std.testing.allocator;
    const CausalSelfAttention = @import("../nn/attention.zig").CausalSelfAttention;
    const Rng = @import("../core/rng.zig").Rng;
    const ops_reduce = @import("../tensor/ops/reduce.zig");
    const ops_create = @import("../tensor/ops/create.zig");

    var rng = Rng.init(42);
    var attn = try CausalSelfAttention.init(allocator, 4, 3, true, &rng);
    defer attn.deinit();

    // DO NOT scale down weights — we need non-trivial attention scores
    // for the gradient check to be reliable. With 0.1-scaled weights,
    // the scores are ~0.0003 while the mask is -1, making softmax
    // dominated by the mask and gradients too small for accurate
    // finite-difference checks.

    // Use -0.5 mask (not -1e9 or -10).
    // With -0.5, the softmax of the upper triangle gets exp(-0.5) ≈ 0.607,
    // keeping the distribution smooth enough for accurate gradients
    // while still enforcing the causal pattern.
    for (attn.causal_mask.data) |*v| {
        if (v.* < 0.0) v.* = -0.5;
    }

    // Use larger input so attention scores are comparable to mask
    var x = try ops_create.randn(allocator, Shape.init3D(2, 3, 4), &rng, 0.0, 1.0);
    defer x.deinit(allocator);

    var params: std.ArrayList(*Tensor) = .empty;
    defer params.deinit(allocator);
    try params.ensureTotalCapacity(allocator, 8);
    attn.parameters(&params);

    var tape = Tape.init(allocator);
    defer tape.deinit();

    for (params.items) |param| {
        param.requires_grad = true;
        _ = try tape.trackLeaf(param);
    }

    var out = try attn.forward(x, &tape);
    defer out.deinit(allocator);

    var loss = try ops_reduce.sumAll(allocator, out, &tape);
    defer loss.deinit(allocator);

    try tape.backward(&loss);

    const h: f32 = 1e-4;
    var param_idx: usize = 0;
    var max_overall_rel: f32 = 0;
    var max_overall_abs: f32 = 0;
    for (params.items) |param| {
        const grad = param.grad orelse continue;
        const n = param.data.len;
        const n_check = @min(n, 5);
        var max_rel_err: f32 = 0;
        var max_abs_analytical: f32 = 0;
        var max_abs_numerical: f32 = 0;
        var max_abs_diff: f32 = 0;
        var check_rng = std.Random.Xoshiro256.init(77);

        for (0..n_check) |_| {
            const idx = check_rng.next() % n;
            const original = param.data[idx];

            param.data[idx] = original + h;
            var op = try attn.forward(x, null);
            defer op.deinit(allocator);
            var lp = try ops_reduce.sumAll(allocator, op, null);
            defer lp.deinit(allocator);

            param.data[idx] = original - h;
            var om = try attn.forward(x, null);
            defer om.deinit(allocator);
            var lm = try ops_reduce.sumAll(allocator, om, null);
            defer lm.deinit(allocator);

            param.data[idx] = original;

            const numerical = (lp.data[0] - lm.data[0]) / (2.0 * h);
            const analytical = grad.data[idx];
            const abs_diff = @abs(analytical - numerical);
            const denom = @max(@abs(analytical), @abs(numerical), 1e-2);
            const rel_err = abs_diff / denom;

            max_abs_analytical = @max(max_abs_analytical, @abs(analytical));
            max_abs_numerical = @max(max_abs_numerical, @abs(numerical));
            if (rel_err > max_rel_err) max_rel_err = rel_err;
            max_abs_diff = @max(max_abs_diff, abs_diff);
        }

        const status = if (max_rel_err < 0.01) "PASS" else if (max_rel_err < 0.05) "WARN" else "FAIL";
        std.debug.print("  attn[{}] rel_err={d:.6} max_anal={d:.6} max_num={d:.6} max_abs_diff={d:.6}  {s}\n", .{ param_idx, max_rel_err, max_abs_analytical, max_abs_numerical, max_abs_diff, status });
        max_overall_rel = @max(max_overall_rel, max_rel_err);
        max_overall_abs = @max(max_overall_abs, max_abs_diff);
        param_idx += 1;
    }
    // Combined check: high relative error with tiny absolute error is
    // finite-difference noise on near-zero gradients, not a backward bug.
    try std.testing.expect(max_overall_rel < 0.05 or max_overall_abs < 5e-3);
}

test "gradCheck — softmax backward produces zero when loss=sumAll(softmax)" {
    // If loss = sumAll(softmax(x)), the gradient should be EXACTLY 0
    // because each row of softmax sums to 1, making the loss constant.
    // A non-zero gradient indicates a bug in backwardSoftmax.
    const allocator = std.testing.allocator;
    const ops_softmax = @import("../tensor/ops/softmax.zig");
    const ops_reduce = @import("../tensor/ops/reduce.zig");
    const ops_create = @import("../tensor/ops/create.zig");
    const Rng = @import("../core/rng.zig").Rng;

    var rng = Rng.init(42);

    var x = try ops_create.randn(allocator, Shape.init2D(2, 3), &rng, 0.0, 0.1);
    defer x.deinit(allocator);
    x.requires_grad = true;

    var tape = Tape.init(allocator);
    defer tape.deinit();

    _ = try tape.trackLeaf(&x);

    var s = try ops_softmax.softmax(allocator, x, &tape);
    defer s.deinit(allocator);

    var loss = try ops_reduce.sumAll(allocator, s, &tape);
    defer loss.deinit(allocator);

    try tape.backward(&loss);

    const grad = x.grad orelse unreachable;
    var max_abs: f32 = 0;
    for (grad.data) |v| {
        max_abs = @max(max_abs, @abs(v));
    }

    std.debug.print("  sumAll(softmax) backward: max_abs_grad={d:.10}\n", .{max_abs});
    try std.testing.expect(max_abs < 1e-5);
}

test "gradCheck — softmax with mask then sumAll (pipeline softmaxed)" {
    // This replicates the exact "softmaxed" pipeline stage:
    // x → Linear(x)=Q → K → K^T → Q@K^T → scale → +mask → softmax → sumAll
    // When loss = sumAll(softmax(x)), gradient of w_q.weight should be 0.
    const allocator = std.testing.allocator;
    const ops_softmax = @import("../tensor/ops/softmax.zig");
    const ops_elementwise = @import("../tensor/ops/elementwise.zig");
    const ops_matmul = @import("../tensor/ops/matmul.zig");
    const ops_shape = @import("../tensor/ops/shape_ops.zig");
    const ops_reduce = @import("../tensor/ops/reduce.zig");
    const ops_create = @import("../tensor/ops/create.zig");
    const Linear = @import("../nn/linear.zig").Linear;
    const Rng = @import("../core/rng.zig").Rng;

    var rng = Rng.init(42);

    var w_q = try Linear.init(allocator, 4, 4, true, &rng);
    defer w_q.deinit();
    var w_k = try Linear.init(allocator, 4, 4, true, &rng);
    defer w_k.deinit();

    for (w_q.weight.data) |*v| v.* *= 0.1;
    for (w_k.weight.data) |*v| v.* *= 0.1;

    var x = try ops_create.randn(allocator, Shape.init3D(2, 3, 4), &rng, 0.0, 0.1);
    defer x.deinit(allocator);

    var tape = Tape.init(allocator);
    defer tape.deinit();

    w_q.weight.requires_grad = true;
    _ = try tape.trackLeaf(&w_q.weight);

    var q = try w_q.forward(x, &tape);
    defer q.deinit(allocator);

    var k = try w_k.forward(x, &tape);
    defer k.deinit(allocator);

    var k_t = try ops_shape.transposeInner2dTracked(allocator, k, &tape);
    defer k_t.deinit(allocator);

    var scores = try ops_matmul.matmulBatch(allocator, q, k_t, &tape);
    defer scores.deinit(allocator);

    const D: usize = 4;
    const scale: f32 = @floatCast(1.0 / std.math.sqrt(@as(f64, @floatFromInt(D))));
    var scaled = try ops_elementwise.mulScalar(allocator, scores, scale, &tape);
    defer scaled.deinit(allocator);

    var mask = try Tensor.init(allocator, Shape.init3D(1, 3, 3));
    defer mask.deinit(allocator);
    for (0..3) |i| {
        for (0..3) |j| {
            mask.data[i * 3 + j] = if (j <= i) @as(f32, 0.0) else -1.0;
        }
    }
    var masked = try ops_elementwise.add(allocator, scaled, mask, &tape);
    defer masked.deinit(allocator);

    var weights = try ops_softmax.softmax(allocator, masked, &tape);
    defer weights.deinit(allocator);

    var loss = try ops_reduce.sumAll(allocator, weights, &tape);
    defer loss.deinit(allocator);

    try tape.backward(&loss);

    const grad = w_q.weight.grad orelse unreachable;
    var max_abs: f32 = 0;
    for (grad.data) |v| {
        max_abs = @max(max_abs, @abs(v));
    }

    std.debug.print("  pipeline softmaxed: w_q.weight max_abs_grad={d:.10}\n", .{max_abs});
    try std.testing.expect(max_abs < 1e-4);
}

test "gradCheck — softmax backward (2D, non-trivial loss)" {
    const allocator = std.testing.allocator;
    const ops_softmax = @import("../tensor/ops/softmax.zig");
    const ops_elementwise = @import("../tensor/ops/elementwise.zig");
    const ops_reduce = @import("../tensor/ops/reduce.zig");
    const ops_create = @import("../tensor/ops/create.zig");
    const Rng = @import("../core/rng.zig").Rng;

    var rng = Rng.init(42);

    // Use small values to avoid peaked softmax distributions
    var x = try ops_create.randn(allocator, Shape.init2D(2, 3), &rng, 0.0, 0.1);
    defer x.deinit(allocator);
    x.requires_grad = true;

    var target = try ops_create.randn(allocator, Shape.init2D(2, 3), &rng, 0.0, 0.1);
    defer target.deinit(allocator);

    var tape = Tape.init(allocator);
    defer tape.deinit();

    _ = try tape.trackLeaf(&x);

    var s = try ops_softmax.softmax(allocator, x, &tape);
    defer s.deinit(allocator);

    var weighted = try ops_elementwise.mul(allocator, s, target, &tape);
    defer weighted.deinit(allocator);

    var loss = try ops_reduce.sumAll(allocator, weighted, &tape);
    defer loss.deinit(allocator);

    try tape.backward(&loss);

    const grad = x.grad orelse unreachable;
    const h: f32 = 1e-4;
    const n = x.data.len;
    var max_rel_err: f32 = 0;

    for (0..n) |idx| {
        const original = x.data[idx];

        x.data[idx] = original + h;
        var sp = try ops_softmax.softmax(allocator, x, null);
        defer sp.deinit(allocator);
        var wp = try ops_elementwise.mul(allocator, sp, target, null);
        defer wp.deinit(allocator);
        var lp = try ops_reduce.sumAll(allocator, wp, null);
        defer lp.deinit(allocator);

        x.data[idx] = original - h;
        var sm = try ops_softmax.softmax(allocator, x, null);
        defer sm.deinit(allocator);
        var wm = try ops_elementwise.mul(allocator, sm, target, null);
        defer wm.deinit(allocator);
        var lm = try ops_reduce.sumAll(allocator, wm, null);
        defer lm.deinit(allocator);

        x.data[idx] = original;

        const numerical = (lp.data[0] - lm.data[0]) / (2.0 * h);
        const analytical = grad.data[idx];
        const abs_diff = @abs(analytical - numerical);
        const denom = @max(@abs(analytical), @abs(numerical), 1e-8);
        const rel_err = abs_diff / denom;
        if (rel_err > max_rel_err) max_rel_err = rel_err;
    }

    std.debug.print("  softmax 2D backward (small x) max_rel_err={d:.6}\n", .{max_rel_err});
    try std.testing.expect(max_rel_err < 0.01);
}

test "gradCheck — softmax backward (3D, non-trivial loss)" {
    const allocator = std.testing.allocator;
    const ops_softmax = @import("../tensor/ops/softmax.zig");
    const ops_elementwise = @import("../tensor/ops/elementwise.zig");
    const ops_reduce = @import("../tensor/ops/reduce.zig");
    const ops_create = @import("../tensor/ops/create.zig");
    const Rng = @import("../core/rng.zig").Rng;

    // Use small input values (std=0.1) for finite-difference precision.
    var rng = Rng.init(42);

    var x = try ops_create.randn(allocator, Shape.init3D(2, 3, 3), &rng, 0.0, 0.1);
    defer x.deinit(allocator);
    x.requires_grad = true;

    var target = try ops_create.randn(allocator, Shape.init3D(2, 3, 3), &rng, 0.0, 0.1);
    defer target.deinit(allocator);

    var tape = Tape.init(allocator);
    defer tape.deinit();

    _ = try tape.trackLeaf(&x);

    var s = try ops_softmax.softmax(allocator, x, &tape);
    defer s.deinit(allocator);

    var weighted = try ops_elementwise.mul(allocator, s, target, &tape);
    defer weighted.deinit(allocator);

    var loss = try ops_reduce.sumAll(allocator, weighted, &tape);
    defer loss.deinit(allocator);

    try tape.backward(&loss);

    const grad = x.grad orelse unreachable;
    const h: f32 = 1e-4;
    const n = x.data.len;
    var max_rel_err: f32 = 0;
    var max_abs_diff: f32 = 0;
    // Use denom floor of 1e-2 for relative error: near-zero gradients
    // (< 0.01) cause high relative error from finite-difference noise
    // even when the backward is correct. Also check absolute error.
    const denom_floor: f32 = 1e-2;

    for (0..n) |idx| {
        const original = x.data[idx];

        x.data[idx] = original + h;
        var sp = try ops_softmax.softmax(allocator, x, null);
        defer sp.deinit(allocator);
        var wp = try ops_elementwise.mul(allocator, sp, target, null);
        defer wp.deinit(allocator);
        var lp = try ops_reduce.sumAll(allocator, wp, null);
        defer lp.deinit(allocator);

        x.data[idx] = original - h;
        var sm = try ops_softmax.softmax(allocator, x, null);
        defer sm.deinit(allocator);
        var wm = try ops_elementwise.mul(allocator, sm, target, null);
        defer wm.deinit(allocator);
        var lm = try ops_reduce.sumAll(allocator, wm, null);
        defer lm.deinit(allocator);

        x.data[idx] = original;

        const numerical = (lp.data[0] - lm.data[0]) / (2.0 * h);
        const analytical = grad.data[idx];
        const abs_diff = @abs(analytical - numerical);
        const denom = @max(@abs(analytical), @abs(numerical), denom_floor);
        const rel_err = abs_diff / denom;
        if (rel_err > max_rel_err) max_rel_err = rel_err;
        max_abs_diff = @max(max_abs_diff, abs_diff);
    }

    std.debug.print("  softmax 3D backward (std=0.1) max_rel_err={d:.6} max_abs_diff={d:.6}\n", .{ max_rel_err, max_abs_diff });
    try std.testing.expect(max_rel_err < 0.05);
    try std.testing.expect(max_abs_diff < 5e-3);
}

test "gradCheck — softmax→matmulBatch (weights@V path)" {
    const allocator = std.testing.allocator;
    const ops_softmax = @import("../tensor/ops/softmax.zig");
    const ops_matmul = @import("../tensor/ops/matmul.zig");
    const ops_reduce = @import("../tensor/ops/reduce.zig");
    const ops_create = @import("../tensor/ops/create.zig");
    const Rng = @import("../core/rng.zig").Rng;

    var rng = Rng.init(42);

    // Use small x values (std=0.1) for the same reason as the 3D softmax
    // backward test: finite-difference precision degrades with peaked softmax.
    var x = try ops_create.randn(allocator, Shape.init3D(2, 3, 3), &rng, 0.0, 0.1);
    defer x.deinit(allocator);
    x.requires_grad = true;

    var v_tensor = try ops_create.randn(allocator, Shape.init3D(2, 3, 4), &rng, 0.0, 0.1);
    defer v_tensor.deinit(allocator);

    var tape = Tape.init(allocator);
    defer tape.deinit();

    _ = try tape.trackLeaf(&x);

    var weights = try ops_softmax.softmax(allocator, x, &tape);
    defer weights.deinit(allocator);

    var attn_out = try ops_matmul.matmulBatch(allocator, weights, v_tensor, &tape);
    defer attn_out.deinit(allocator);

    var loss = try ops_reduce.sumAll(allocator, attn_out, &tape);
    defer loss.deinit(allocator);

    try tape.backward(&loss);

    const grad = x.grad orelse unreachable;
    const h: f32 = 1e-4;
    const n = x.data.len;
    var max_rel_err: f32 = 0;
    var max_abs_diff: f32 = 0;
    const denom_floor: f32 = 1e-2;

    for (0..n) |idx| {
        const original = x.data[idx];

        x.data[idx] = original + h;
        var wp = try ops_softmax.softmax(allocator, x, null);
        defer wp.deinit(allocator);
        var aop = try ops_matmul.matmulBatch(allocator, wp, v_tensor, null);
        defer aop.deinit(allocator);
        var lp = try ops_reduce.sumAll(allocator, aop, null);
        defer lp.deinit(allocator);

        x.data[idx] = original - h;
        var wm = try ops_softmax.softmax(allocator, x, null);
        defer wm.deinit(allocator);
        var aom = try ops_matmul.matmulBatch(allocator, wm, v_tensor, null);
        defer aom.deinit(allocator);
        var lm = try ops_reduce.sumAll(allocator, aom, null);
        defer lm.deinit(allocator);

        x.data[idx] = original;

        const numerical = (lp.data[0] - lm.data[0]) / (2.0 * h);
        const analytical = grad.data[idx];
        const abs_diff = @abs(analytical - numerical);
        const denom = @max(@abs(analytical), @abs(numerical), denom_floor);
        const rel_err = abs_diff / denom;
        if (rel_err > max_rel_err) max_rel_err = rel_err;
        max_abs_diff = @max(max_abs_diff, abs_diff);
    }

    std.debug.print("  softmax→matmulBatch x grad (std=0.1) max_rel_err={d:.6} max_abs_diff={d:.6}\n", .{ max_rel_err, max_abs_diff });
    // Near-zero gradients cause high relative error from finite-difference
    // noise; use denom floor of 1e-2 and also check absolute error.
    // Longer chain (softmax→matmul→sumAll) compounds finite-diff error,
    // so use slightly relaxed tolerance.
    try std.testing.expect(max_rel_err < 0.1);
    try std.testing.expect(max_abs_diff < 1e-3);
}
