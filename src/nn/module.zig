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
// Pre-PR-ζ that state was keyed by `@intFromPtr(param.cpuData().ptr)` — the
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
///
/// Stage 8 adds `n_layer`, `n_head`, `dropout`. All three default
/// to values that preserve Stage 1–7 behavior bit-for-bit
/// (`n_layer = 1`, `n_head = 1`, `dropout = 0.0`) so every existing
/// test, checkpoint, and example continues to behave identically
/// unless the caller explicitly opts in.
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

    /// Number of stacked TransformerBlock layers. Default 1 matches
    /// the Stage 2–7 model shape; Milestone 3 wires this through to
    /// `TinyWordTransformer.blocks: []TransformerBlock`. Capped at
    /// 255 via the u8 width — enough headroom for any pedagogical
    /// experiment; a `u32` widening is additive if we ever need
    /// a bigger model.
    n_layer: u8 = 1,

    /// Number of attention heads per block. Default 1 matches the
    /// single-head attention of Stages 2–7. Milestone 4 generalises
    /// `CausalSelfAttention` to `n_head ≥ 1`. Requires
    /// `d_model % n_head == 0` (checked in `CausalSelfAttention.init`).
    n_head: u8 = 1,

    /// Dropout probability applied to attention weights and MLP
    /// outputs. Default 0.0 — i.e. dropout is disabled.
    ///
    /// **NOT implemented in Stage 8.** The field is reserved so
    /// future work can thread it through without another config
    /// migration. Milestone 3 reads the field but never acts on it;
    /// a non-zero value is silently ignored until a future stage
    /// implements masked dropout + dropout-aware backward.
    dropout: f32 = 0.0,

    /// Use RMSNorm instead of LayerNorm (Session 6 addition).
    /// Default false preserves existing behavior. When true, the
    /// TransformerBlock uses RMSNorm (no mean-centering, no beta).
    use_rms_norm: bool = false,

    /// Use SwiGLU MLP instead of GELU MLP (Session 6 addition).
    /// Default false preserves existing behavior. When true, the
    /// TransformerBlock uses SwiGLU (3 Linears + SiLU gating).
    use_swiglu: bool = false,
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

test "TransformerConfig Stage 8 defaults preserve Stage 2-7 semantics" {
    // Defaults must be n_layer=1, n_head=1, dropout=0.0 so every
    // existing test, checkpoint, and example stays bit-identical.
    const cfg = TransformerConfig{};
    try std.testing.expectEqual(@as(u8, 1), cfg.n_layer);
    try std.testing.expectEqual(@as(u8, 1), cfg.n_head);
    try std.testing.expectEqual(@as(f32, 0.0), cfg.dropout);
}

test "TransformerConfig accepts non-default n_layer / n_head / dropout" {
    const cfg = TransformerConfig{ .n_layer = 6, .n_head = 4, .dropout = 0.1 };
    try std.testing.expectEqual(@as(u8, 6), cfg.n_layer);
    try std.testing.expectEqual(@as(u8, 4), cfg.n_head);
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), cfg.dropout, 1e-6);
}
