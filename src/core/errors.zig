//!
//! zig-transformer-lab — Error set
//!
//! Purpose:
//!   Single library-wide error set used by all fallible functions.
//!   Every public function returns `!T` using this set.
//!
//! Error conditions:
//!   ShapeMismatch  — tensor shapes are incompatible for the requested operation
//!   OutOfMemory    — allocator could not fulfill the request
//!   InvalidArgument — caller passed an out-of-range or nonsensical value
//!   InvalidLayout  — tensor layout (contiguity / aliasing) is not supported
//!                    by this op. Example: calling `addInPlace` on a
//!                    transposed view where the flat-index assumption
//!                    would silently corrupt data.
//!   InvalidIndex   — a token/class/axis index is outside its valid range
//!                    (e.g. embedding lookup with id >= vocab_size).
//!   DeviceMismatch — a binary op received tensors on different devices
//!                    (CPU + CUDA). Implicit transfer is forbidden.
//!   CudaError      — a CUDA Driver API or cuBLAS call returned a non-success code
//!   IoError        — file I/O failure (checkpoint, data, vocab)
//!   NotImplemented — code path exists but is not yet implemented
//!   NumericalError — NaN, Inf, or other numerical anomaly detected
//!

pub const LabError = error{
    ShapeMismatch,
    OutOfMemory,
    InvalidArgument,
    InvalidLayout,
    InvalidIndex,
    DeviceMismatch,
    CudaError,
    IoError,
    NotImplemented,
    NumericalError,
};


