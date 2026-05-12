//!
//! backend/cuda/edu_kernels.zig — Host launchers for pedagogical CUDA kernels
//!
//! Purpose:
//!   Provides Zig functions to load and launch the tiled GEMM and WMMA
//!   demo kernels from src/backend/cuda/kernels/. These are NOT
//!   production ops — they exist to run timing benchmarks and
//!   correctness checks as part of the docs/cuda_depth.md chapter.
//!
//! These functions are called by examples/11_cuda_tiled_gemm_demo.zig
//! and examples/12_cuda_wmma_demo.zig.

const std = @import("std");
const LabError = @import("../../core/errors.zig").LabError;
const CudaContext = @import("context.zig").CudaContext;
const DeviceBuffer = @import("mem.zig").DeviceBuffer;
const cuda_module = @import("module.zig");
const bindings = @import("bindings.zig");

/// Launch the naive GEMM kernel (one thread per output element).
/// A, B, C are device buffers of size M*K, K*N, M*N respectively.
pub fn launchNaiveGemm(
    ctx: *const CudaContext,
    A: DeviceBuffer,
    B: DeviceBuffer,
    C: DeviceBuffer,
    M: u32,
    N: u32,
    K: u32,
) LabError!void {
    const mut_ctx: *CudaContext = @constCast(ctx);
    const kfn = try cuda_module.getKernel(mut_ctx, "tiled_gemm", "naive_gemm");

    var m_arg: c_int = @intCast(M);
    var n_arg: c_int = @intCast(N);
    var k_arg: c_int = @intCast(K);
    var a_ptr = A.ptr;
    var b_ptr = B.ptr;
    var c_ptr = C.ptr;

    const args = [_]?*anyopaque{
        @ptrCast(&a_ptr),
        @ptrCast(&b_ptr),
        @ptrCast(&c_ptr),
        @ptrCast(&m_arg),
        @ptrCast(&n_arg),
        @ptrCast(&k_arg),
    };

    const block_dim: c_uint = 16;
    const grid_x: c_uint = (N + block_dim - 1) / block_dim;
    const grid_y: c_uint = (M + block_dim - 1) / block_dim;

    try cuda_module.launch(mut_ctx, kfn, .{ grid_x, grid_y, 1 }, .{ block_dim, block_dim, 1 }, 0, &args);
}

/// Launch the tiled shared-memory GEMM kernel (TILE=16).
pub fn launchTiledGemm(
    ctx: *const CudaContext,
    A: DeviceBuffer,
    B: DeviceBuffer,
    C: DeviceBuffer,
    M: u32,
    N: u32,
    K: u32,
) LabError!void {
    const mut_ctx: *CudaContext = @constCast(ctx);
    const kfn = try cuda_module.getKernel(mut_ctx, "tiled_gemm", "tiled_gemm");

    var m_arg: c_int = @intCast(M);
    var n_arg: c_int = @intCast(N);
    var k_arg: c_int = @intCast(K);
    var a_ptr = A.ptr;
    var b_ptr = B.ptr;
    var c_ptr = C.ptr;

    const args = [_]?*anyopaque{
        @ptrCast(&a_ptr),
        @ptrCast(&b_ptr),
        @ptrCast(&c_ptr),
        @ptrCast(&m_arg),
        @ptrCast(&n_arg),
        @ptrCast(&k_arg),
    };

    const tile: c_uint = 16;
    const grid_x: c_uint = (N + tile - 1) / tile;
    const grid_y: c_uint = (M + tile - 1) / tile;

    try cuda_module.launch(mut_ctx, kfn, .{ grid_x, grid_y, 1 }, .{ tile, tile, 1 }, 0, &args);
}

/// Launch the WMMA 16x16 demo kernel (one warp, one tile).
/// A_fp16 and B_fp16 are device buffers of 256 half values (16*16).
/// C_fp32 is a device buffer of 256 f32 values (16*16 output).
pub fn launchWmmaDemo(
    ctx: *const CudaContext,
    A_fp16: DeviceBuffer,
    B_fp16: DeviceBuffer,
    C_fp32: DeviceBuffer,
) LabError!void {
    const mut_ctx: *CudaContext = @constCast(ctx);
    const kfn = try cuda_module.getKernel(mut_ctx, "wmma_demo", "wmma_matmul_16x16");

    var a_ptr = A_fp16.ptr;
    var b_ptr = B_fp16.ptr;
    var c_ptr = C_fp32.ptr;

    const args = [_]?*anyopaque{
        @ptrCast(&a_ptr),
        @ptrCast(&b_ptr),
        @ptrCast(&c_ptr),
    };

    // One warp (32 threads), one block, one grid cell
    try cuda_module.launch(mut_ctx, kfn, .{ 1, 1, 1 }, .{ 32, 1, 1 }, 0, &args);
}
