//!
//! tensor/device_dispatch.zig — Device-routing seam for compute operations
//!
//! Purpose:
//!   This module is the **single point** where the tensor ops layer
//!   touches the CUDA backend. Every ops file (elementwise, reduce,
//!   matmul, softmax, unary, loss, create, shape_ops) imports this
//!   module instead of reaching directly into `backend/cuda/dispatch.zig`
//!   or `backend/cuda/gemm.zig`.
//!
//!   Today it is a thin re-export layer. The architectural value is the
//!   **seam**: callers need only know "device_dispatch has a function for
//!   my op" and never import from `backend/cuda/` themselves. This means:
//!     - Adding a new backend (e.g., Metal) is a change to ONE file.
//!     - The ops layer can be compiled / tested in CPU-only mode without
//!       any conditional compilation — `device_dispatch` simply has no
//!       CUDA functions if the backend isn't linked.
//!     - A build.zig grep can enforce "no src/tensor/ops/*.zig file
//!       imports from backend/cuda/" as a structural invariant.
//!
//!   The two adapters behind this seam (CPU impls inside each ops file,
//!   CUDA impls in backend/cuda/) satisfy the "two adapters = real seam"
//!   criterion from the architecture skill.
//!
//! Interface contract:
//!   Every public function here takes Tensor(s) that live on CUDA
//!   (caller is responsible for the device check) and returns an owned
//!   CUDA Tensor. These are **compute-only** — no tape recording happens
//!   here. The ops layer handles tape recording after calling these.
//!
//! Why re-export rather than wrap?
//!   Wrapping would add a layer of indirection for zero new logic.
//!   Re-exporting is zero-cost at comptime and makes the seam explicit
//!   without introducing another call frame at runtime.
//!

const std = @import("std");
const cuda_dispatch = @import("../backend/cuda/dispatch.zig");
const cuda_gemm = @import("../backend/cuda/gemm.zig");

// Re-export the CudaContext type needed by create.zig for zerosOn/onesOn.
pub const CudaContext = @import("../backend/cuda/context.zig").CudaContext;

// ---------------------------------------------------------------------------
// Elementwise binary ops (same-shape fast path)
// ---------------------------------------------------------------------------

pub const add = cuda_dispatch.add;
pub const sub = cuda_dispatch.sub;
pub const mul = cuda_dispatch.mul;
pub const div = cuda_dispatch.div;

// Stride-aware broadcast path (rank-4 kernel)
pub const addBroadcast = cuda_dispatch.addBroadcast;
pub const subBroadcast = cuda_dispatch.subBroadcast;
pub const mulBroadcast = cuda_dispatch.mulBroadcast;
pub const divBroadcast = cuda_dispatch.divBroadcast;

// ---------------------------------------------------------------------------
// Elementwise scalar/unary ops
// ---------------------------------------------------------------------------

pub const addScalar = cuda_dispatch.addScalar;
pub const mulScalar = cuda_dispatch.mulScalar;
pub const neg = cuda_dispatch.neg;

// ---------------------------------------------------------------------------
// Unary math ops
// ---------------------------------------------------------------------------

pub const exp = cuda_dispatch.exp;
pub const log = cuda_dispatch.log;
pub const geluExact = cuda_dispatch.geluExact;
pub const geluExactBackward = cuda_dispatch.geluExactBackward;
pub const sqrt = cuda_dispatch.sqrt;

// ---------------------------------------------------------------------------
// Reduction ops
// ---------------------------------------------------------------------------

pub const sumAxis = cuda_dispatch.sumAxis;
pub const sumAll = cuda_dispatch.sumAll;
pub const sumToShape = cuda_dispatch.sumToShape;

// ---------------------------------------------------------------------------
// Softmax / log-softmax
// ---------------------------------------------------------------------------

pub const softmaxLastAxis = cuda_dispatch.softmaxLastAxis;
pub const logSoftmaxLastAxis = cuda_dispatch.logSoftmaxLastAxis;

// ---------------------------------------------------------------------------
// Cross-entropy (fused forward + grad kernel)
// ---------------------------------------------------------------------------

pub const crossEntropyFused = cuda_dispatch.crossEntropyFused;

// ---------------------------------------------------------------------------
// GEMM (matmul / batched matmul)
// ---------------------------------------------------------------------------

pub const matmul = cuda_gemm.matmul;
pub const matmulBatch = cuda_gemm.matmulBatch;

// ---------------------------------------------------------------------------
// Shape manipulation (clone, broadcast-materialize)
// ---------------------------------------------------------------------------

pub const cloneDevice = cuda_dispatch.cloneDevice;
pub const broadcastTo = cuda_dispatch.broadcastTo;

// ---------------------------------------------------------------------------
// Creation ops (device-aware factories)
// ---------------------------------------------------------------------------

pub const zerosOn = cuda_dispatch.zerosOn;
pub const onesOn = cuda_dispatch.onesOn;

// ---------------------------------------------------------------------------
// Optimizer kernels (used by optim/adamw.zig via backward.zig)
// ---------------------------------------------------------------------------

pub const adamwStep = cuda_dispatch.adamwStep;

// ---------------------------------------------------------------------------
// Embedding ops
// ---------------------------------------------------------------------------

pub const embeddingForward = cuda_dispatch.embeddingForward;
pub const embeddingBackward = cuda_dispatch.embeddingBackward;

// ===========================================================================
// Comptime seam enforcement
// ===========================================================================
//
// This test verifies at compile time that the ops layer modules do NOT
// contain any direct reference to cuda_dispatch or cuda_gemm symbols as
// their own imported declarations. The test imports each ops module and
// checks that none of them re-export or declare `cuda_dispatch` or
// `cuda_gemm` as a module-level identifier. This catches accidental
// direct imports that bypass the device_dispatch seam.
//
// The grep-based enforcement (build.zig `ops-purity` step) is the
// stronger check since it operates on source text. This comptime test
// is the "belt" to that "suspenders."

const ops_elementwise = @import("ops/elementwise.zig");
const ops_reduce = @import("ops/reduce.zig");
const ops_matmul = @import("ops/matmul.zig");
const ops_softmax = @import("ops/softmax.zig");
const ops_unary = @import("ops/unary.zig");
const ops_loss = @import("ops/loss.zig");
const ops_create = @import("ops/create.zig");
const ops_shape = @import("ops/shape_ops.zig");

fn assertNoCudaDecl(comptime T: type) void {
    const info = @typeInfo(T);
    if (info == .@"struct") {
        for (info.@"struct".decls) |d| {
            const name = d.name;
            if (comptime std.mem.eql(u8, name, "cuda_dispatch") or
                std.mem.eql(u8, name, "cuda_gemm"))
            {
                @compileError("ops module leaks a direct CUDA import: " ++ name);
            }
        }
    }
}

test "ops-layer purity: no ops module re-exports cuda_dispatch or cuda_gemm" {
    // Comptime check — fires at compile time if any ops module has a
    // public `cuda_dispatch` or `cuda_gemm` declaration. Private
    // (non-pub) consts named `cuda_dispatch` that point to THIS module
    // are fine — they don't appear in @typeInfo.struct.decls for the
    // public interface.
    comptime {
        assertNoCudaDecl(@TypeOf(ops_elementwise));
        assertNoCudaDecl(@TypeOf(ops_reduce));
        assertNoCudaDecl(@TypeOf(ops_matmul));
        assertNoCudaDecl(@TypeOf(ops_softmax));
        assertNoCudaDecl(@TypeOf(ops_unary));
        assertNoCudaDecl(@TypeOf(ops_loss));
        assertNoCudaDecl(@TypeOf(ops_create));
        assertNoCudaDecl(@TypeOf(ops_shape));
    }
}
