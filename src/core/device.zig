//!
//! zig-transformer-lab — Device tag
//!
//! Purpose:
//!   Defines the `Device` enum that records where a tensor's data buffer
//!   lives: on the CPU heap or in CUDA device memory. This is the first step
//!   toward backend dispatch (Stage 7). For Stage 2 every tensor is CPU-only,
//!   but the field is already on the Tensor struct so that adding CUDA later
//!   does not require a layout change.
//!
//! Shape contract:
//!   Device carries no shape information. It is purely a placement tag.
//!
//! Memory ownership:
//!   Device is a plain enum — no allocation, no ownership.
//!
//! Error conditions:
//!   None.
//!
//! TODO:
//!   - Stage 7 will add a `device_id: u8` field alongside Device so that
//!     multi-GPU selection is possible, but for now one GPU is enough.
//!
//! Credits:
//!   Pattern mirrors PyTorch's torch.device('cpu') / torch.device('cuda');
//!   no code copied.

const std = @import("std");

/// Where a tensor's data buffer resides.
///
/// Stage 2 only uses `.cpu`. The `.cuda` variant exists so that Tensor can
/// carry the field from day one; ops that receive a `.cuda` tensor will
/// return `error.NotImplemented` until Stage 7's dispatch is wired up.
pub const Device = enum {
    cpu,
    cuda,

    /// Human-readable name for debug printing.
    ///
    /// Worked example:
    ///   Device.cpu.label() returns "cpu"
    ///   Device.cuda.label() returns "cuda"
    pub fn label(self: Device) []const u8 {
        return switch (self) {
            .cpu => "cpu",
            .cuda => "cuda",
        };
    }

    /// Returns true when the device is CUDA.
    ///
    /// Worked example:
    ///   Device.cpu.isCuda()   ==  false
    ///   Device.cuda.isCuda()  ==  true
    ///
    /// Memory: no allocation.
    ///
    /// Why a helper instead of `self == .cuda`?
    ///   Centralising the check means that if we later add `.cuda_host`
    ///   (pinned host memory accessible by the GPU), we only need to update
    ///   this one function rather than audit every call site.
    pub fn isCuda(self: Device) bool {
        return self == .cuda;
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "Device.isCuda returns false for cpu" {
    try std.testing.expect(!Device.cpu.isCuda());
}

test "Device.isCuda returns true for cuda" {
    try std.testing.expect(Device.cuda.isCuda());
}

test "Device has exactly two variants" {
    const fields = std.meta.fields(Device);
    try std.testing.expectEqual(@as(usize, 2), fields.len);
}
