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
