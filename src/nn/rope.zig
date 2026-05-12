//!
//! zig-transformer-lab — Rotary Position Embedding (RoPE)
//!
//! Purpose:
//!   Encodes position information by rotating Q and K vectors in 2D
//!   subspaces. The rotation angle is a function of position and
//!   frequency, so Q @ K^T naturally encodes relative position.
//!
//!   Used by Llama, Mistral, Gemma, and all modern LLMs since 2023.
//!   Replaces learned positional embeddings.
//!
//! Math:
//!   For each dimension pair (2i, 2i+1) in a head:
//!     theta_i = 10000^(-2i/d_head)
//!     cos_val = cos(pos * theta_i)
//!     sin_val = sin(pos * theta_i)
//!
//!     q[2i]'   = q[2i] * cos_val - q[2i+1] * sin_val
//!     q[2i+1]' = q[2i] * sin_val + q[2i+1] * cos_val
//!
//!   Apply to Q and K only (not V).
//!   The dot product Q_rotated @ K_rotated^T then encodes pos_q - pos_k.
//!
//! Shape contract:
//!   applyRope(x: (B, T, D), pos_offset: usize) -> (B, T, D)
//!   where D = d_head (applied per-head after the Q/K projection)
//!
//! Credits:
//!   Su et al. (2021), "RoFormer: Enhanced Transformer with Rotary
//!   Position Embedding." No code copied.

const std = @import("std");
const LabError = @import("../core/errors.zig").LabError;
const Tensor = @import("../tensor/tensor.zig").Tensor;
const Shape = @import("../tensor/shape.zig").Shape;
const totalElements = @import("../tensor/shape.zig").totalElements;

/// Precomputed cos/sin tables for RoPE.
///
/// These are computed once at model init and reused every forward pass.
/// Shape of each table: (max_seq_len, d_head/2).
pub const RoPECache = struct {
    cos_cache: []f32, // (max_T * d_head/2) flattened
    sin_cache: []f32, // (max_T * d_head/2) flattened
    max_seq_len: usize,
    d_head: usize,
    allocator: std.mem.Allocator,

    /// Precompute cos/sin tables for all positions up to max_seq_len.
    ///
    /// theta_i = 10000^(-2i/d_head) for i in 0..d_head/2
    /// cos_cache[pos][i] = cos(pos * theta_i)
    /// sin_cache[pos][i] = sin(pos * theta_i)
    pub fn init(allocator: std.mem.Allocator, max_seq_len: usize, d_head: usize) LabError!RoPECache {
        const half_d = d_head / 2;
        const n = max_seq_len * half_d;

        const cos_buf = allocator.alloc(f32, n) catch return error.OutOfMemory;
        errdefer allocator.free(cos_buf);
        const sin_buf = allocator.alloc(f32, n) catch return error.OutOfMemory;
        errdefer allocator.free(sin_buf);

        // Precompute frequencies: theta_i = 1.0 / (10000^(2i/d_head))
        for (0..max_seq_len) |pos| {
            for (0..half_d) |i| {
                const freq_exp: f32 = @floatCast(@as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(d_head)));
                const theta: f32 = 1.0 / std.math.pow(f32, 10000.0, freq_exp);
                const angle: f32 = @as(f32, @floatFromInt(pos)) * theta;
                cos_buf[pos * half_d + i] = @cos(angle);
                sin_buf[pos * half_d + i] = @sin(angle);
            }
        }

        return RoPECache{
            .cos_cache = cos_buf,
            .sin_cache = sin_buf,
            .max_seq_len = max_seq_len,
            .d_head = d_head,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *RoPECache) void {
        self.allocator.free(self.cos_cache);
        self.allocator.free(self.sin_cache);
    }

    /// Apply RoPE rotation to a tensor in-place.
    ///
    /// Input x has shape (B*n_heads, T, d_head) — this is the shape
    /// AFTER the Q/K projection and head-split in attention.
    ///
    /// For each position t and dimension pair (2i, 2i+1):
    ///   x'[2i]   = x[2i] * cos(t*theta_i) - x[2i+1] * sin(t*theta_i)
    ///   x'[2i+1] = x[2i] * sin(t*theta_i) + x[2i+1] * cos(t*theta_i)
    ///
    /// Modifies x in-place (no allocation, no tape recording — this is
    /// done before the matmul that IS tape-recorded).
    pub fn applyInPlace(self: RoPECache, x: *Tensor) void {
        const data = x.cpuData();
        const ndim = x.shape.ndim();

        // Determine T and d_head from shape
        // Expected shape: 3D (BH, T, d_head) or 2D (T, d_head)
        const T: usize = if (ndim == 3) x.shape.dims[1] else x.shape.dims[0];
        const d_head = self.d_head;
        const half_d = d_head / 2;
        const batch_heads: usize = if (ndim == 3) x.shape.dims[0] else 1;

        for (0..batch_heads) |bh| {
            for (0..T) |t| {
                for (0..half_d) |i| {
                    const cos_val = self.cos_cache[t * half_d + i];
                    const sin_val = self.sin_cache[t * half_d + i];

                    const base_idx = bh * T * d_head + t * d_head + 2 * i;
                    const x0 = data[base_idx];
                    const x1 = data[base_idx + 1];

                    data[base_idx] = x0 * cos_val - x1 * sin_val;
                    data[base_idx + 1] = x0 * sin_val + x1 * cos_val;
                }
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "RoPE at position 0 is identity (cos=1, sin=0)" {
    const alloc = std.testing.allocator;
    var cache = try RoPECache.init(alloc, 16, 4);
    defer cache.deinit();

    // Create a (1, 1, 4) tensor — one position, d_head=4
    var x = try Tensor.init(alloc, Shape.init3D(1, 1, 4));
    defer x.deinit(alloc);
    x.cpuData()[0] = 1.0;
    x.cpuData()[1] = 2.0;
    x.cpuData()[2] = 3.0;
    x.cpuData()[3] = 4.0;

    // At position 0: cos(0)=1, sin(0)=0. Rotation is identity.
    cache.applyInPlace(&x);

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), x.cpuData()[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), x.cpuData()[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), x.cpuData()[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), x.cpuData()[3], 1e-6);
}

test "RoPE at position > 0 modifies values" {
    const alloc = std.testing.allocator;
    var cache = try RoPECache.init(alloc, 16, 4);
    defer cache.deinit();

    // (1, 2, 4) — two positions
    var x = try Tensor.init(alloc, Shape.init3D(1, 2, 4));
    defer x.deinit(alloc);
    x.cpuData()[0] = 1.0;
    x.cpuData()[1] = 0.0;
    x.cpuData()[2] = 1.0;
    x.cpuData()[3] = 0.0;
    // Position 1 values
    x.cpuData()[4] = 1.0;
    x.cpuData()[5] = 0.0;
    x.cpuData()[6] = 1.0;
    x.cpuData()[7] = 0.0;

    cache.applyInPlace(&x);

    // Position 0 should be unchanged (cos=1, sin=0)
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), x.cpuData()[0], 1e-6);
    // Position 1 should be rotated
    // theta_0 = 1/10000^0 = 1.0, angle = 1*1 = 1.0 radian
    // x'[0] = 1.0*cos(1) - 0.0*sin(1) = cos(1) ≈ 0.5403
    try std.testing.expectApproxEqAbs(@as(f32, 0.5403), x.cpuData()[4], 0.001);
    // x'[1] = 1.0*sin(1) + 0.0*cos(1) = sin(1) ≈ 0.8415
    try std.testing.expectApproxEqAbs(@as(f32, 0.8415), x.cpuData()[5], 0.001);
}
