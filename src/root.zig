//!
//! zig-transformer-lab — Package entry point
//!
//! Purpose:
//!   Root module that re-exports every public namespace in the library.
//!   Consumers `@import("zig_transformer_lab")` and get access to everything.
//!
//! Ownership:
//!   This file owns nothing. It is a pure namespace re-export.
//!
//! Convention:
//!   Sub-modules are imported as private consts and re-exported as public.
//!   As new source files are added in later stages, add them here
//!   and in the test block at the bottom so their `test` blocks are discovered.
//!

pub const errors = @import("core/errors.zig");

// Stage 2 additions:
pub const dtype = @import("core/dtype.zig");
pub const device = @import("core/device.zig");
pub const rng = @import("core/rng.zig");
pub const shape = @import("tensor/shape.zig");
const tensor_mod = @import("tensor/tensor.zig");
pub const Tensor = tensor_mod.Tensor;
pub const Storage = tensor_mod.Storage;
pub const CpuStorage = tensor_mod.CpuStorage;
pub const nonOwningStorage = tensor_mod.nonOwningStorage;
pub const requireSameDevice = tensor_mod.requireSameDevice;
pub const debugCheckInvariants = tensor_mod.debugCheckInvariants;
pub const tensor_print = @import("tensor/print.zig");
pub const ops = struct {
    pub const create = @import("tensor/ops/create.zig");
    pub const elementwise = @import("tensor/ops/elementwise.zig");
    pub const reduce = @import("tensor/ops/reduce.zig");
    pub const matmul = @import("tensor/ops/matmul.zig");
    pub const unary = @import("tensor/ops/unary.zig");
    pub const softmax = @import("tensor/ops/softmax.zig");
    pub const loss = @import("tensor/ops/loss.zig");
    pub const shape_ops = @import("tensor/ops/shape_ops.zig");
};
pub const device_dispatch = @import("tensor/device_dispatch.zig");

// Stage 3 additions:
pub const autograd = @import("autograd/node.zig");
const tape_mod = @import("autograd/tape.zig");
pub const Tape = tape_mod.Tape;
const backward_mod = @import("autograd/backward.zig");
pub const backward = backward_mod;
const gradcheck_mod = @import("autograd/gradcheck.zig");
pub const gradcheck = gradcheck_mod;

// Stage 4 additions:
pub const nn = struct {
    pub const module = @import("nn/module.zig");
    pub const linear = @import("nn/linear.zig");
    pub const embedding = @import("nn/embedding.zig");
    pub const layernorm = @import("nn/layernorm.zig");
    pub const activations = @import("nn/activations.zig");
    pub const attention = @import("nn/attention.zig");
    pub const mlp = @import("nn/mlp.zig");
    pub const block = @import("nn/block.zig");
    pub const model = @import("nn/model.zig");
};
pub const optim = struct {
    pub const optimizer = @import("optim/optimizer.zig");
    pub const sgd = @import("optim/sgd.zig");
    pub const adamw = @import("optim/adamw.zig");
};

// Stage 5 additions:
pub const tokenizer = struct {
    pub const vocab = @import("tokenizer/vocab.zig");
    pub const word = @import("tokenizer/word.zig");
};
pub const data = struct {
    pub const dataset = @import("data/dataset.zig");
    pub const windowing = @import("data/windowing.zig");
    pub const batcher = @import("data/batcher.zig");
};

// Stage 6 additions:
pub const lab = @import("lab/train.zig");

// Stage 8 additions: opt-in debug utilities for shape assertions,
// NaN/Inf detection, device-aware comparisons, and tensor dumps.
// Consumers do `debug.assertShape(t, expected)` etc.; the helpers
// are intentionally NOT used inside `src/` production code — they
// are for example scripts, ad-hoc debugging, and new-op bring-up.
pub const debug = struct {
    pub const shape = @import("debug/shape.zig");
    pub const finite = @import("debug/finite.zig");
    pub const compare = @import("debug/compare.zig");
    pub const dump = @import("debug/dump.zig");
};

// Test-only utilities (PyTorch oracle parity, etc.):
pub const testing_utils = struct {
    pub const oracle = @import("testing/oracle.zig");
};

// Stage 7 additions:
// The `backend.cuda` namespace exposes the CUDA runtime layer in
// incremental PRs. PR-alpha ships just the dynamic-loader bindings;
// subsequent PRs will add `context`, `mem`, `module`, `dispatch`,
// and the kernel wrappers. The sub-namespace is always present so
// that tests/integration_cuda.zig can reach bindings regardless of
// the `-Dcuda` build option (the file itself is platform-portable
// because it uses std.DynLib rather than linking libcuda directly).
pub const backend = struct {
    pub const cuda = struct {
        pub const bindings = @import("backend/cuda/bindings.zig");
        pub const context = @import("backend/cuda/context.zig");
        pub const mem = @import("backend/cuda/mem.zig");
        pub const module = @import("backend/cuda/module.zig");
        pub const dispatch = @import("backend/cuda/dispatch.zig");
        pub const gemm = @import("backend/cuda/gemm.zig");
    };
};

test {
    _ = errors;
    _ = dtype;
    _ = device;
    _ = rng;
    _ = shape;
    _ = tensor_mod;
    _ = tensor_print;
    _ = ops.create;
    _ = ops.elementwise;
    _ = ops.reduce;
    _ = ops.matmul;
    _ = ops.unary;
    _ = ops.softmax;
    _ = ops.loss;
    _ = ops.shape_ops;
    _ = device_dispatch;
    _ = autograd;
    _ = tape_mod;
    _ = backward_mod;
    _ = gradcheck_mod;

    // Stage 4:
    _ = nn.module;
    _ = nn.linear;
    _ = nn.embedding;
    _ = nn.layernorm;
    _ = nn.activations;
    _ = nn.attention;
    _ = nn.mlp;
    _ = nn.block;
    _ = nn.model;
    _ = optim.optimizer;
    _ = optim.sgd;
    _ = optim.adamw;

    // Stage 5:
    _ = tokenizer.vocab;
    _ = tokenizer.word;
    _ = data.dataset;
    _ = data.windowing;
    _ = data.batcher;

    // Stage 6:
    _ = lab;

    // Stage 8: debug utilities.
    _ = debug.shape;
    _ = debug.finite;
    _ = debug.compare;
    _ = debug.dump;

    // Oracle + testing utilities (PR post-6.5):
    _ = testing_utils.oracle;
}
