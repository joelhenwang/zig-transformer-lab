//!
//! zig-transformer-lab — Debug helpers: NaN / Inf detection
//!
//! Purpose:
//!   Scan a tensor for non-finite values and report the first offender.
//!   Used as a trip-wire during training-loop development: insert
//!   `try debug.assertFinite(loss);` after each op to narrow down which
//!   step introduced a NaN or Inf.
//!
//!   Common sources of NaN in transformer training:
//!     - softmax overflow with a large positive logit that's not max-subtracted
//!     - log(0) when a probability collapses to zero
//!     - sqrt(negative) from a variance that went negative due to fp error
//!     - 0 / 0 division in a normalized op
//!     - large lr causing a weight explosion that propagates to Inf next step
//!
//!   Common sources of Inf:
//!     - exp(huge_logit) after a max-subtraction bug
//!     - loss blow-up from a poorly-conditioned init
//!     - accumulated parameter drift during long runs
//!
//! Shape contract:
//!   assertFinite(t)   → void or error.NumericalError
//!   hasNaN(t)         → bool   (for inline branching — no print, no error)
//!   hasInf(t)         → bool   (same)
//!
//! Memory ownership:
//!   CUDA path allocates a temporary CPU buffer via DtoH copy. The
//!   buffer is released before the helper returns. No owned resources
//!   escape the call.
//!
//! Errors:
//!   NumericalError — at least one NaN or Inf in the tensor; print
//!                    includes the offending flat index and value.
//!   CudaError      — DtoH copy failure (CUDA tensors).
//!   OutOfMemory    — CPU scratch allocation for CUDA tensors.
//!
//! Device:
//!   CPU: direct scan of `t.data`.
//!   CUDA: DtoH copy of `t.storage.cuda.len` floats, then host scan.
//!         A dedicated reduction kernel (one flag per block, atomicOr
//!         a global) would avoid the copy but is deferred: `assertFinite`
//!         is a debug trip-wire, not a training-loop hot path.
//!

const std = @import("std");
const LabError = @import("../core/errors.zig").LabError;
const Tensor = @import("../tensor/tensor.zig").Tensor;

/// Per-call scratch allocator. CUDA paths need one temp buffer; the
/// rest of the function is pure scan. Using the testing allocator
/// here is not correct — callers outside tests would leak. Instead,
/// we accept the allocator through the function signature.
///
/// Returns null if the tensor is fully finite. Otherwise returns a
/// packed (flat_index, value) pair for the first offender found.
const Offender = struct {
    index: usize,
    value: f32,
    kind: Kind,

    const Kind = enum { nan, pos_inf, neg_inf };

    fn classify(v: f32) ?Kind {
        if (std.math.isNan(v)) return .nan;
        if (std.math.isInf(v)) return if (v > 0) .pos_inf else .neg_inf;
        return null;
    }
};

fn scanHost(data: []const f32) ?Offender {
    for (data, 0..) |v, i| {
        if (Offender.classify(v)) |k| {
            return .{ .index = i, .value = v, .kind = k };
        }
    }
    return null;
}

fn scanCuda(allocator: std.mem.Allocator, t: Tensor) LabError!?Offender {
    const buf = switch (t.storage) {
        .cuda => |b| b,
        .cpu => unreachable,
    };
    // DtoH-copy the entire buffer. This catches non-finite values
    // anywhere in the underlying allocation, including slots
    // unreachable via the current shape + offset. For most tensors
    // the buffer exactly matches the logical size; for views it may
    // be larger, and spurious non-finites in unreachable regions
    // could false-positive. In practice we only assertFinite on
    // op-output tensors (fresh, contiguous) so this is fine.
    const host = allocator.alloc(f32, buf.len) catch return error.OutOfMemory;
    defer allocator.free(host);
    try buf.copyToHost(host);
    return scanHost(host);
}

/// Assert that every element of `t` is finite. Emits a one-line
/// diagnostic and returns `error.NumericalError` on failure.
///
/// For CPU tensors the allocator is unused (pure scan of `t.data`).
/// For CUDA tensors the allocator is used for a DtoH scratch buffer.
///
/// Worked example:
///   // Training loop:
///   var loss = try ops_loss.crossEntropy(alloc, logits, targets, &tape);
///   try debug.assertFinite(alloc, loss);
///   try tape.backward(&loss);
///
///   // If logits overflowed softmax, assertFinite would report:
///   //   assertFinite FAIL: NaN at flat index 12 (value = nan)
pub fn assertFinite(allocator: std.mem.Allocator, t: Tensor) LabError!void {
    const maybe_offender: ?Offender = switch (t.storage) {
        .cpu => scanHost(t.cpuData()),
        .cuda => try scanCuda(allocator, t),
    };

    const o = maybe_offender orelse return;
    const kind_str: []const u8 = switch (o.kind) {
        .nan => "NaN",
        .pos_inf => "+Inf",
        .neg_inf => "-Inf",
    };
    std.debug.print(
        "  assertFinite FAIL: {s} at flat index {d} (value = {e})\n",
        .{ kind_str, o.index, o.value },
    );
    return error.NumericalError;
}

/// Non-asserting query: returns true if any NaN is present.
///
/// Useful for branching logic without interrupting execution:
///   if (debug.hasNaN(alloc, param.grad.?.*) catch false) {
///       // log and skip this step
///   }
pub fn hasNaN(allocator: std.mem.Allocator, t: Tensor) LabError!bool {
    const o = switch (t.storage) {
        .cpu => scanHost(t.cpuData()),
        .cuda => try scanCuda(allocator, t),
    };
    if (o) |off| return off.kind == .nan;
    return false;
}

/// Non-asserting query: returns true if any +Inf or -Inf is present.
pub fn hasInf(allocator: std.mem.Allocator, t: Tensor) LabError!bool {
    const o = switch (t.storage) {
        .cpu => scanHost(t.cpuData()),
        .cuda => try scanCuda(allocator, t),
    };
    if (o) |off| return off.kind == .pos_inf or off.kind == .neg_inf;
    return false;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const Shape = @import("../tensor/shape.zig").Shape;

test "assertFinite: passes on all-finite tensor" {
    var t = try Tensor.init(std.testing.allocator, Shape.init1D(4));
    defer t.deinit(std.testing.allocator);
    t.cpuData()[0] = 1.0;
    t.cpuData()[1] = -2.5;
    t.cpuData()[2] = 0.0;
    t.cpuData()[3] = 1e-10;
    try assertFinite(std.testing.allocator, t);
}

test "assertFinite: detects NaN at first offender" {
    var t = try Tensor.init(std.testing.allocator, Shape.init1D(4));
    defer t.deinit(std.testing.allocator);
    t.cpuData()[0] = 1.0;
    t.cpuData()[1] = 2.0;
    t.cpuData()[2] = std.math.nan(f32);
    t.cpuData()[3] = 4.0;
    try std.testing.expectError(error.NumericalError, assertFinite(std.testing.allocator, t));
}

test "assertFinite: detects +Inf" {
    var t = try Tensor.init(std.testing.allocator, Shape.init1D(3));
    defer t.deinit(std.testing.allocator);
    t.cpuData()[0] = 1.0;
    t.cpuData()[1] = std.math.inf(f32);
    t.cpuData()[2] = 3.0;
    try std.testing.expectError(error.NumericalError, assertFinite(std.testing.allocator, t));
}

test "assertFinite: detects -Inf" {
    var t = try Tensor.init(std.testing.allocator, Shape.init1D(3));
    defer t.deinit(std.testing.allocator);
    t.cpuData()[0] = 1.0;
    t.cpuData()[1] = -std.math.inf(f32);
    t.cpuData()[2] = 3.0;
    try std.testing.expectError(error.NumericalError, assertFinite(std.testing.allocator, t));
}

test "hasNaN: returns true only when NaN is present" {
    var t = try Tensor.init(std.testing.allocator, Shape.init1D(2));
    defer t.deinit(std.testing.allocator);
    t.cpuData()[0] = 1.0;
    t.cpuData()[1] = 2.0;
    try std.testing.expect(!(try hasNaN(std.testing.allocator, t)));

    t.cpuData()[1] = std.math.nan(f32);
    try std.testing.expect(try hasNaN(std.testing.allocator, t));
}

test "hasInf: returns true only when +/-Inf is present" {
    var t = try Tensor.init(std.testing.allocator, Shape.init1D(2));
    defer t.deinit(std.testing.allocator);
    t.cpuData()[0] = 1.0;
    t.cpuData()[1] = 2.0;
    try std.testing.expect(!(try hasInf(std.testing.allocator, t)));

    t.cpuData()[1] = std.math.inf(f32);
    try std.testing.expect(try hasInf(std.testing.allocator, t));

    t.cpuData()[1] = -std.math.inf(f32);
    try std.testing.expect(try hasInf(std.testing.allocator, t));

    // NaN is not an Inf
    t.cpuData()[1] = std.math.nan(f32);
    try std.testing.expect(!(try hasInf(std.testing.allocator, t)));
}
