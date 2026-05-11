//!
//! zig-transformer-lab — Elementwise unary operations
//!
//! Purpose:
//!   Provides elementwise unary functions: exp, log, neg, relu, and
//!   geluExact.  These are the building blocks of every activation
//!   function and loss function in a transformer.  Each function
//!   allocates a new output tensor with the same shape as the input
//!   and writes the transformed values into it.  The input tensor is
//!   never modified.
//!
//! Shape contract:
//!   All functions: in:(...) -> out:(...)  [same shape as input]
//!   The output is always contiguous (owned), regardless of input
//!   contiguity.  Elements are read from the input using strides, so
//!   non-contiguous inputs (e.g., transposed tensors) work correctly.
//!
//! Math:
//!   exp(x)    = e^x
//!   log(x)    = ln(x)  — log(0) = -Inf (let it happen, documented)
//!   neg(x)    = -x
//!   relu(x)   = max(0, x)
//!   geluExact(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
//!   sqrt(x)      = √x
//!
//!   GELU (Gaussian Error Linear Unit) is the activation used in the
//!   original BERT and GPT-2 models.  The "exact" variant uses the
//!   error function directly; a faster "approximate" variant (using
//!   tanh) exists but is deferred to a later stage.
//!
//!   sqrt is needed by LayerNorm: the inverse standard deviation is
//!   1 / sqrt(variance + eps), and composing this from our existing
//!   ops means we need sqrt as a tracked operation.
//!
//! Memory ownership:
//!   All functions return new owned tensors.  The caller must call
//!   deinit(allocator) on the returned tensor when done.
//!
//! Errors:
//!   OutOfMemory — allocator could not fulfill the output buffer
//!
//! TODO:
//!   - Add geluApprox using the tanh approximation
//!   - Add sigmoid, silu/swish as needed by later stages
//!
//! Credits:
//!   GELU was introduced in "Gaussian Error Linear Units" (Hendrycks
//!   & Gimpel, 2016).  No code copied.

const std = @import("std");
const LabError = @import("../../core/errors.zig").LabError;
const Tensor = @import("../tensor.zig").Tensor;
const Shape = @import("../shape.zig").Shape;
const totalElements = @import("../shape.zig").totalElements;
const Tape = @import("../../autograd/tape.zig").Tape;
const Node = @import("../../autograd/node.zig").Node;
const OpKind = @import("../../autograd/node.zig").OpKind;
// PR-eta.2 follow-up: unary.neg routes CUDA inputs to the CUDA
// elementwise dispatch. backward.zig's backwardSub / backwardDiv
// call this function; without CUDA routing here the backward chain
// would silently operate on empty CPU slices for CUDA tensors.
const cuda_dispatch = @import("../../backend/cuda/dispatch.zig");

// ---------------------------------------------------------------------------
// erf — Polynomial approximation (Abramowitz & Stegun, formula 7.1.26)
// ---------------------------------------------------------------------------

/// Approximate the error function erf(x) for f64 input.
///
/// Uses the Abramowitz & Stegun rational approximation (Handbook of
/// Mathematical Functions, formula 7.1.26).  Maximum absolute error
/// is approximately 1.5e-7, well within f32 precision (which is ~1e-7
/// at best).  We compute in f64 and let the caller truncate to f32.
///
/// Why not use the C library's erf?
///   Zig 0.16.0's standard library does not expose erf.  Linking
///   against libc just for erf would add a dependency we don't need.
///   This self-contained approximation is sufficient for GELU.
///
/// Formula:
///   erf(x) = sign(x) * (1 - (a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5) * exp(-x^2))
///   where t = 1 / (1 + p*|x|)
///
/// Worked example:
///   erfFloat(0.0) = 0.0
///   erfFloat(1.0) ~ 0.8427
///   erfFloat(3.0) ~ 1.0
fn erfFloat(x: f64) f64 {
    // For |x| > 6, erf is 1.0 to machine precision.  Clamping avoids
    // numerical issues in the exp(-x^2) term (underflow is fine but
    // the polynomial can accumulate roundoff for large |x|).
    if (x > 6.0) return 1.0;
    if (x < -6.0) return -1.0;

    const ax = @abs(x);
    const sign: f64 = if (x >= 0) 1.0 else -1.0;

    // Constants from Abramowitz & Stegun Table 7.1
    const p = 0.3275911;
    const a1 = 0.254829592;
    const a2 = -0.284496736;
    const a3 = 1.421413741;
    const a4 = -1.453152027;
    const a5 = 1.061405429;

    const t = 1.0 / (1.0 + p * ax);
    const t2 = t * t;
    const t3 = t2 * t;
    const t4 = t3 * t;
    const t5 = t4 * t;

    // The polynomial in t multiplied by exp(-x^2) gives the
    // complement erfc(x) = 1 - erf(x).  We subtract from 1 to get
    // erf(x).
    const poly = a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5;
    const result = 1.0 - poly * @exp(-ax * ax);

    return sign * result;
}

/// Element-wise e^x.
///
/// Shape: in:(...) -> out:(...)  [same shape]
///
/// Worked example:
///   // in = [0, 1, 2]  shape (3,)
///   // out = [1, e, e^2] = [1.0, 2.718, 7.389]
///
/// Memory: caller owns the returned tensor; must call deinit(allocator).
pub fn exp(allocator: std.mem.Allocator, tensor: Tensor, tape: ?*Tape) LabError!Tensor {
    if (tensor.device == .cuda) {
        var out = try cuda_dispatch.exp(tensor);
        if (tape) |t| {
            if (tensor.requires_grad) {
                const node_id = try t.record(Node{
                    .id = undefined,
                    .op = .exp,
                    .parents = .{ tensor.tape_node, null },
                    .n_parents = 1,
                    .saved = .{ .tensor_ref = tensor },
                });
                out.requires_grad = true;
                out.tape_node = node_id;
            }
        }
        return out;
    }
    const n = totalElements(tensor.shape);
    var out = try Tensor.init(allocator, tensor.shape);
    errdefer out.deinit(allocator);

    // We iterate using flatIndex to support non-contiguous inputs
    // (e.g., transposed tensors).  For contiguous inputs this is
    // equivalent to a simple i-based loop, but the stride arithmetic
    // is negligible cost compared to @exp itself.
    for (0..n) |flat| {
        // Decode flat index into multi-dim indices, then compute
        // the strided offset.  For rank-1 this is just flat*s0,
        // for rank-2 it's row*s0 + col*s1, etc.
        var offset: usize = 0;
        var remaining: usize = flat;
        const ndim = tensor.shape.ndim();
        var axis: usize = 0;
        while (axis + 1 < ndim) : (axis += 1) {
            // Product of all dims after this axis gives the number
            // of elements per "block" at this axis.
            var block: usize = 1;
            var a2: usize = axis + 1;
            while (a2 < ndim) : (a2 += 1) {
                block *= tensor.shape.dims[a2];
            }
            const idx = remaining / block;
            remaining %= block;
            offset += idx * tensor.strides.values[axis];
        }
        // Last axis: remaining is the index along the last dim
        offset += remaining * tensor.strides.values[axis];

        out.data[flat] = @exp(tensor.data[offset]);
    }

    if (tape) |t| {
        if (tensor.requires_grad) {
            const node_id = try t.record(Node{
                .id = undefined,
                .op = .exp,
                .parents = .{ tensor.tape_node, null },
                .n_parents = 1,
                .saved = .{ .tensor_ref = tensor },
            });
            out.requires_grad = true;
            out.tape_node = node_id;
        }
    }

    return out;
}

/// Element-wise natural logarithm ln(x).
///
/// Note: log(0) = -Inf.  This is IEEE 754 behavior and we let it
/// happen rather than error — downstream code (softmax, loss) handles
/// -Inf correctly.  log(negative) = NaN, also per IEEE 754.
///
/// Shape: in:(...) -> out:(...)  [same shape]
///
/// Worked example:
///   // in = [1, e, e^2]  shape (3,)
///   // out = [0, 1, 2]
///
/// Memory: caller owns the returned tensor; must call deinit(allocator).
pub fn log(allocator: std.mem.Allocator, tensor: Tensor, tape: ?*Tape) LabError!Tensor {
    if (tensor.device == .cuda) {
        var out = try cuda_dispatch.log(tensor);
        if (tape) |t| {
            if (tensor.requires_grad) {
                const node_id = try t.record(Node{
                    .id = undefined,
                    .op = .log,
                    .parents = .{ tensor.tape_node, null },
                    .n_parents = 1,
                    .saved = .{ .tensor_ref = tensor },
                });
                out.requires_grad = true;
                out.tape_node = node_id;
            }
        }
        return out;
    }
    const n = totalElements(tensor.shape);
    var out = try Tensor.init(allocator, tensor.shape);
    errdefer out.deinit(allocator);

    for (0..n) |flat| {
        var offset: usize = 0;
        var remaining: usize = flat;
        const ndim = tensor.shape.ndim();
        var axis: usize = 0;
        while (axis + 1 < ndim) : (axis += 1) {
            var block: usize = 1;
            var a2: usize = axis + 1;
            while (a2 < ndim) : (a2 += 1) {
                block *= tensor.shape.dims[a2];
            }
            const idx = remaining / block;
            remaining %= block;
            offset += idx * tensor.strides.values[axis];
        }
        offset += remaining * tensor.strides.values[axis];

        out.data[flat] = @log(tensor.data[offset]);
    }

    if (tape) |t| {
        if (tensor.requires_grad) {
            const node_id = try t.record(Node{
                .id = undefined,
                .op = .log,
                .parents = .{ tensor.tape_node, null },
                .n_parents = 1,
                .saved = .{ .tensor_ref = tensor },
            });
            out.requires_grad = true;
            out.tape_node = node_id;
        }
    }

    return out;
}

/// Element-wise negation: -x.
///
/// Note: a `neg` function also exists in the elementwise namespace
/// (elementwise.zig) — that's fine.  This is the unary-ops version.
///
/// Shape: in:(...) -> out:(...)  [same shape]
///
/// Worked example:
///   // in = [1, -2, 0]  shape (3,)
///   // out = [-1, 2, 0]
///
/// Memory: caller owns the returned tensor; must call deinit(allocator).
pub fn neg(allocator: std.mem.Allocator, tensor: Tensor, tape: ?*Tape) LabError!Tensor {
    if (tensor.device == .cuda) {
        var out = try cuda_dispatch.neg(tensor);
        if (tape) |t| {
            if (tensor.requires_grad) {
                const node_id = try t.record(Node{
                    .id = undefined,
                    .op = .neg,
                    .parents = .{ tensor.tape_node, null },
                    .n_parents = 1,
                    .saved = .nothing,
                });
                out.requires_grad = true;
                out.tape_node = node_id;
            }
        }
        return out;
    }
    const n = totalElements(tensor.shape);
    var out = try Tensor.init(allocator, tensor.shape);
    errdefer out.deinit(allocator);

    for (0..n) |flat| {
        var offset: usize = 0;
        var remaining: usize = flat;
        const ndim = tensor.shape.ndim();
        var axis: usize = 0;
        while (axis + 1 < ndim) : (axis += 1) {
            var block: usize = 1;
            var a2: usize = axis + 1;
            while (a2 < ndim) : (a2 += 1) {
                block *= tensor.shape.dims[a2];
            }
            const idx = remaining / block;
            remaining %= block;
            offset += idx * tensor.strides.values[axis];
        }
        offset += remaining * tensor.strides.values[axis];

        out.data[flat] = -tensor.data[offset];
    }

    if (tape) |t| {
        if (tensor.requires_grad) {
            const node_id = try t.record(Node{
                .id = undefined,
                .op = .neg,
                .parents = .{ tensor.tape_node, null },
                .n_parents = 1,
                .saved = .nothing,
            });
            out.requires_grad = true;
            out.tape_node = node_id;
        }
    }

    return out;
}

/// Element-wise ReLU: max(0, x).
///
/// ReLU (Rectified Linear Unit) is the simplest and most widely-used
/// activation function.  It clips negative values to zero, which
/// introduces the non-linearity needed for neural networks to learn
/// complex functions.
///
/// Shape: in:(...) -> out:(...)  [same shape]
///
/// Worked example:
///   // in = [-1, 0, 1]  shape (3,)
///   // out = [0, 0, 1]
///
/// Memory: caller owns the returned tensor; must call deinit(allocator).
pub fn relu(allocator: std.mem.Allocator, tensor: Tensor, tape: ?*Tape) LabError!Tensor {
    const n = totalElements(tensor.shape);
    var out = try Tensor.init(allocator, tensor.shape);
    errdefer out.deinit(allocator);

    for (0..n) |flat| {
        var offset: usize = 0;
        var remaining: usize = flat;
        const ndim = tensor.shape.ndim();
        var axis: usize = 0;
        while (axis + 1 < ndim) : (axis += 1) {
            var block: usize = 1;
            var a2: usize = axis + 1;
            while (a2 < ndim) : (a2 += 1) {
                block *= tensor.shape.dims[a2];
            }
            const idx = remaining / block;
            remaining %= block;
            offset += idx * tensor.strides.values[axis];
        }
        offset += remaining * tensor.strides.values[axis];

        out.data[flat] = @max(@as(f32, 0.0), tensor.data[offset]);
    }

    if (tape) |t| {
        if (tensor.requires_grad) {
            const node_id = try t.record(Node{
                .id = undefined,
                .op = .relu,
                .parents = .{ tensor.tape_node, null },
                .n_parents = 1,
                .saved = .{ .tensor_ref = tensor },
            });
            out.requires_grad = true;
            out.tape_node = node_id;
        }
    }

    return out;
}

/// Element-wise exact GELU: 0.5 * x * (1 + erf(x / sqrt(2))).
///
/// GELU is the activation function used in BERT and GPT-2.  It smoothly
/// interpolates between 0 (for negative x) and x (for positive x),
/// unlike ReLU which has a sharp kink at 0.  The "exact" variant uses
/// the error function (erf); a faster tanh approximation exists but
/// is deferred.
///
/// Shape: in:(...) -> out:(...)  [same shape]
///
/// Worked example:
///   // geluExact(0) = 0.5 * 0 * (1 + erf(0)) = 0
///   // geluExact(1) = 0.5 * 1 * (1 + erf(1/sqrt(2))) ~ 0.8412
///
/// Memory: caller owns the returned tensor; must call deinit(allocator).
pub fn geluExact(allocator: std.mem.Allocator, tensor: Tensor, tape: ?*Tape) LabError!Tensor {
    if (tensor.device == .cuda) {
        var out = try cuda_dispatch.geluExact(tensor);
        if (tape) |t| {
            if (tensor.requires_grad) {
                const node_id = try t.record(Node{
                    .id = undefined,
                    .op = .gelu,
                    .parents = .{ tensor.tape_node, null },
                    .n_parents = 1,
                    .saved = .{ .tensor_ref = tensor },
                });
                out.requires_grad = true;
                out.tape_node = node_id;
            }
        }
        return out;
    }
    const n = totalElements(tensor.shape);
    var out = try Tensor.init(allocator, tensor.shape);
    errdefer out.deinit(allocator);

    // Pre-compute 1/sqrt(2) so we multiply instead of dividing
    // inside the loop — multiplication is faster than division.
    const inv_sqrt2: f32 = 1.0 / @sqrt(@as(f32, 2.0));

    for (0..n) |flat| {
        var offset: usize = 0;
        var remaining: usize = flat;
        const ndim = tensor.shape.ndim();
        var axis: usize = 0;
        while (axis + 1 < ndim) : (axis += 1) {
            var block: usize = 1;
            var a2: usize = axis + 1;
            while (a2 < ndim) : (a2 += 1) {
                block *= tensor.shape.dims[a2];
            }
            const idx = remaining / block;
            remaining %= block;
            offset += idx * tensor.strides.values[axis];
        }
        offset += remaining * tensor.strides.values[axis];

        const val = tensor.data[offset];
        // Use our polynomial erf approximation in f64, then cast
        // back to f32 for the output.
        const erf_arg: f64 = @floatCast(val * inv_sqrt2);
        const erf_val: f64 = erfFloat(erf_arg);
        out.data[flat] = 0.5 * val * (1.0 + @as(f32, @floatCast(erf_val)));
    }

    if (tape) |t| {
        if (tensor.requires_grad) {
            const node_id = try t.record(Node{
                .id = undefined,
                .op = .gelu,
                .parents = .{ tensor.tape_node, null },
                .n_parents = 1,
                .saved = .{ .tensor_ref = tensor },
            });
            out.requires_grad = true;
            out.tape_node = node_id;
        }
    }

    return out;
}

/// Element-wise square root: √x.
///
/// Needed by LayerNorm: inv_std = 1 / sqrt(var + eps).
/// The gradient: dL/dx = dL/dy * 1 / (2 * sqrt(x)).
///
/// For x < 0, returns NaN (IEEE 754 behavior).
///
/// Shape: in:(...) -> out:(...)  [same shape]
///
/// Worked example:
///   // in = [0, 1, 4]  shape (3,)
///   // out = [0, 1, 2]
///
/// Memory: caller owns the returned tensor; must call deinit(allocator).
pub fn sqrt(allocator: std.mem.Allocator, tensor: Tensor, tape: ?*Tape) LabError!Tensor {
    if (tensor.device == .cuda) {
        var out = try cuda_dispatch.sqrt(tensor);
        if (tape) |t| {
            if (tensor.requires_grad) {
                const node_id = try t.record(Node{
                    .id = undefined,
                    .op = .sqrt,
                    .parents = .{ tensor.tape_node, null },
                    .n_parents = 1,
                    .saved = .{ .tensor_ref = tensor },
                });
                out.requires_grad = true;
                out.tape_node = node_id;
            }
        }
        return out;
    }
    const n = totalElements(tensor.shape);
    var out = try Tensor.init(allocator, tensor.shape);
    errdefer out.deinit(allocator);

    for (0..n) |flat| {
        var offset: usize = 0;
        var remaining: usize = flat;
        const ndim = tensor.shape.ndim();
        var axis: usize = 0;
        while (axis + 1 < ndim) : (axis += 1) {
            var block: usize = 1;
            var a2: usize = axis + 1;
            while (a2 < ndim) : (a2 += 1) {
                block *= tensor.shape.dims[a2];
            }
            const idx = remaining / block;
            remaining %= block;
            offset += idx * tensor.strides.values[axis];
        }
        offset += remaining * tensor.strides.values[axis];

        out.data[flat] = @sqrt(tensor.data[offset]);
    }

    if (tape) |t| {
        if (tensor.requires_grad) {
            const node_id = try t.record(Node{
                .id = undefined,
                .op = .sqrt,
                .parents = .{ tensor.tape_node, null },
                .n_parents = 1,
                .saved = .{ .tensor_ref = tensor },
            });
            out.requires_grad = true;
            out.tape_node = node_id;
        }
    }

    return out;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "exp [0, 1, 2] = [1, e, e^2]" {
    const alloc = std.testing.allocator;

    var t = try Tensor.init(alloc, Shape.init1D(3));
    defer t.deinit(alloc);
    t.data[0] = 0.0;
    t.data[1] = 1.0;
    t.data[2] = 2.0;

    var out = try exp(alloc, t, null);
    defer out.deinit(alloc);

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out.data[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, std.math.e), out.data[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, std.math.e * std.math.e), out.data[2], 1e-2);
}

test "log [1, e, e^2] = [0, 1, 2]" {
    const alloc = std.testing.allocator;

    var t = try Tensor.init(alloc, Shape.init1D(3));
    defer t.deinit(alloc);
    t.data[0] = 1.0;
    t.data[1] = @floatCast(std.math.e);
    t.data[2] = @floatCast(std.math.e * std.math.e);

    var out = try log(alloc, t, null);
    defer out.deinit(alloc);

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out.data[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out.data[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out.data[2], 1e-3);
}

test "neg [-1, 0, 1] = [1, 0, -1]" {
    const alloc = std.testing.allocator;

    var t = try Tensor.init(alloc, Shape.init1D(3));
    defer t.deinit(alloc);
    t.data[0] = -1.0;
    t.data[1] = 0.0;
    t.data[2] = 1.0;

    var out = try neg(alloc, t, null);
    defer out.deinit(alloc);

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out.data[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out.data[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), out.data[2], 1e-4);
}

test "relu [-1, 0, 1] = [0, 0, 1]" {
    const alloc = std.testing.allocator;

    var t = try Tensor.init(alloc, Shape.init1D(3));
    defer t.deinit(alloc);
    t.data[0] = -1.0;
    t.data[1] = 0.0;
    t.data[2] = 1.0;

    var out = try relu(alloc, t, null);
    defer out.deinit(alloc);

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out.data[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out.data[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out.data[2], 1e-4);
}

test "geluExact [0] ~ 0, [1] ~ 0.8412" {
    const alloc = std.testing.allocator;

    var t0 = try Tensor.init(alloc, Shape.init1D(1));
    defer t0.deinit(alloc);
    t0.data[0] = 0.0;

    var out0 = try geluExact(alloc, t0, null);
    defer out0.deinit(alloc);
    // gelu(0) = 0.5 * 0 * (1 + erf(0)) = 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out0.data[0], 1e-4);

    var t1 = try Tensor.init(alloc, Shape.init1D(1));
    defer t1.deinit(alloc);
    t1.data[0] = 1.0;

    var out1 = try geluExact(alloc, t1, null);
    defer out1.deinit(alloc);
    // gelu(1) = 0.5 * 1 * (1 + erf(1/sqrt(2))) ~ 0.8412
    try std.testing.expectApproxEqAbs(@as(f32, 0.8412), out1.data[0], 1e-3);
}
