//!
//! zig-transformer-lab — Data type tag
//!
//! Purpose:
//!   Defines the `DType` enum that tags every tensor with its element type.
//!   For Stage 2 the library is f32-only (locked decision D9), but wrapping
//!   the type in an enum now means that adding f16/bf16 later is a one-line
//!   change to this file plus a dispatch branch in each op — no scattered
//!   `f32` literals to hunt down.
//!
//! Shape contract:
//!   DType carries no shape information. It is purely a per-element type tag.
//!
//! Math:
//!   sizeInBytes(.f32) = 4. Future: sizeInBytes(.f16) = 2, sizeInBytes(.bf16) = 2.
//!
//! Memory ownership:
//!   DType is a plain enum — no allocation, no ownership.
//!
//! Error conditions:
//!   None. All variants return a known byte count.
//!
//! TODO:
//!   - future: add .f16, .bf16 when mixed-precision support is
//!     designed (post-Stage-9; gated by decision D7 today).
//!
//! Credits:
//!   Pattern inspired by PyTorch's ScalarType enum (aten/src/ATen/core/scalar_type.h);
//!   no code copied.

const std = @import("std");

/// The set of element types a tensor can store.
///
/// For Stage 2 only `.f32` exists. The enum exists so that every downstream
/// function already accepts a DType parameter; adding a new variant later
/// requires only (a) a new `.sizeInBytes` case and (b) dispatch in each op.
pub const DType = enum {
    f32,

    /// Human-readable name for debug printing.
    ///
    /// Worked example:
    ///   DType.f32.label() returns "f32"
    pub fn label(self: DType) []const u8 {
        return switch (self) {
            .f32 => "f32",
        };
    }

    /// Returns the number of bytes per element for this dtype.
    ///
    /// Worked example:
    ///   DType.f32.sizeInBytes()  ==  4
    ///   // (future) DType.f16.sizeInBytes()  ==  2
    ///
    /// Memory: no allocation.
    pub fn sizeInBytes(self: DType) usize {
        // Each case is explicit so the compiler will warn us if we add a new
        // variant without updating this switch — a common source of bugs in
        // dtype-agnostic code.
        return switch (self) {
            .f32 => @sizeOf(f32),
        };
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "DType.sizeInBytes f32" {
    // f32 is 4 bytes — the only dtype we support in Stage 2.
    try std.testing.expectEqual(@as(usize, 4), DType.f32.sizeInBytes());
}

test "DType is an enum with exactly one variant" {
    // Sanity check: we haven't accidentally added a second variant yet.
    const fields = std.meta.fields(DType);
    try std.testing.expectEqual(@as(usize, 1), fields.len);
}
