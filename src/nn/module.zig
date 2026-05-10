//!
//! zig-transformer-lab — Neural network module protocol
//!
//! Purpose:
//!   Defines the convention that all nn layers follow. Each layer is a
//!   concrete struct with four methods: init, forward, parameters, deinit.
//!   There is no vtable — the model manually collects parameters from
//!   each sub-layer into a flat list for the optimizer.
//!
//!   This is deliberately simpler than PyTorch's nn.Module, which uses
//!   runtime reflection to discover parameters. In Zig, we prefer
//!   compile-time composition: the model knows its sub-layers by name
//!   and calls parameters() on each one.
//!
//! Convention:
//!   Every nn layer struct must expose:
//!     pub fn init(allocator, config, rng) !Self
//!     pub fn forward(self, input, tape) !Tensor
//!     pub fn parameters(self, list: *std.ArrayList(*Tensor)) void
//!     pub fn deinit(self) void
//!
//!   - init: allocates weights/biases, initializes them (Kaiming, etc.)
//!   - forward: computes the layer output, recording ops on the tape
//!   - parameters: appends pointers to learnable tensors (weights, biases)
//!   - deinit: frees all owned tensors
//!
//! Shape contract:
//!   forward() takes (B, T, D) or (B*T, D) and returns the same rank.
//!   Each layer documents its specific shapes.
//!
//! Math:
//!   No math in this file — it's purely a convention document.
//!
//! Memory ownership:
//!   Each layer owns its weight and bias tensors. The model owns the
//!   layers. The optimizer borrows pointers to the parameters.
//!
//! Errors:
//!   OutOfMemory — from weight allocation
//!
//! Credits:
//!   Inspired by PyTorch's nn.Module. No code copied.

const std = @import("std");
const Tensor = @import("../tensor/tensor.zig").Tensor;

// ---------------------------------------------------------------------------
// ParamId — stable identity for learnable parameters (PR-ζ)
// ---------------------------------------------------------------------------
//
// Optimizers track per-parameter state (e.g. AdamW's m and v moments).
// Pre-PR-ζ that state was keyed by `@intFromPtr(param.data.ptr)` — the
// raw memory address of the parameter's backing buffer. That key is
// unstable: any operation that replaces the buffer (loading a
// checkpoint into a fresh tensor; PR-ι moving parameters to CUDA and
// back; a future tensor resize) would silently invalidate the key,
// and the optimizer would start the next step with zero moments.
// The correct model — what PyTorch uses internally — is to give each
// parameter a stable identity that travels with the tensor across
// buffer replacements. `ParamId` is that identity.
//
// Design notes:
//   - 32-bit IDs. Even a pathological training run creates far fewer
//     than 4 billion parameters.
//   - Globally monotonic counter. Tests may create parameters across
//     multiple models in the same process; each one gets a fresh ID.
//   - Atomic so that future multi-threaded parameter creation remains
//     safe. Single-threaded today.
//   - `?ParamId` on Tensor so intermediate tensors (which are NOT
//     parameters) carry a null and never appear in optimizer state.

/// A stable, 32-bit identity for a learnable parameter tensor.
/// Assigned by `assignParamId` during layer initialisation and keyed
/// by the optimizer in place of `@intFromPtr`.
pub const ParamId = u32;

var next_param_id: std.atomic.Value(ParamId) = .init(1);

/// Assign a fresh, globally-unique `ParamId` to this tensor if it does
/// not already have one. Idempotent: re-calling on an already-assigned
/// tensor is a no-op, which lets helpers (e.g. checkpoint load) safely
/// preserve IDs when replacing the backing buffer.
pub fn assignParamId(t: *Tensor) void {
    if (t.param_id == null) {
        t.param_id = next_param_id.fetchAdd(1, .monotonic);
    }
}

/// For tests: reset the global counter so IDs are deterministic across
/// test runs. Not called in production.
pub fn resetParamIdCounterForTests() void {
    next_param_id.store(1, .monotonic);
}

/// Configuration for a single transformer model.
///
/// All fields have pedagogically small defaults so the model can
/// run on CPU in reasonable time. The vocab_size must match the
/// tokenizer (Stage 5).
pub const TransformerConfig = struct {
    /// Vocabulary size (number of unique tokens).
    vocab_size: usize = 64,
    /// Model embedding dimension (d_model).
    d_model: usize = 32,
    /// Maximum sequence length (context window).
    max_seq_len: usize = 16,
    /// Feed-forward hidden dimension (usually 4 * d_model).
    d_ff: usize = 128,
    /// LayerNorm epsilon for numerical stability.
    ln_eps: f32 = 1e-5,
    /// Whether to use bias in Linear layers.
    bias: bool = true,
};

/// Helper: collect parameters from a slice of layers that all have
/// the same type. Appends each layer's parameters to the list.
///
/// This is a convenience function for models that have homogeneous
/// sub-layers (e.g., multiple transformer blocks). For heterogeneous
/// layers, the model manually calls each one's parameters().
///
/// Worked example:
///   // Given: var blocks: [2]TransformerBlock = ...
///   // collectParamsSlice(&blocks, &param_list);
///   // Now param_list contains all weights from both blocks
pub fn collectParamsSlice(comptime Layer: type, layers: []const Layer, list: *std.ArrayList(*Tensor)) void {
    for (layers) |*layer| {
        layer.parameters(list);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "TransformerConfig defaults are pedagogically small" {
    const cfg = TransformerConfig{};
    try std.testing.expectEqual(@as(usize, 64), cfg.vocab_size);
    try std.testing.expectEqual(@as(usize, 32), cfg.d_model);
    try std.testing.expectEqual(@as(usize, 16), cfg.max_seq_len);
    try std.testing.expectEqual(@as(usize, 128), cfg.d_ff);
    try std.testing.expectEqual(@as(f32, 1e-5), cfg.ln_eps);
    try std.testing.expect(cfg.bias);
}
