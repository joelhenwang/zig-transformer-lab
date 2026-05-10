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
