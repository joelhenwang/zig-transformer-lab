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
//!   As new source files are added in later stages, add them here AND in
//!   tests/unit_all.zig so their `test` blocks are discovered.
//!

pub const errors = @import("core/errors.zig");

// Stage 2 additions (uncomment as files are created):
// pub const dtype = @import("core/dtype.zig");
// pub const device = @import("core/device.zig");
// pub const rng = @import("core/rng.zig");
// pub const shape = @import("tensor/shape.zig");
// pub const Tensor = @import("tensor/tensor.zig").Tensor;
// pub const tensor_print = @import("tensor/print.zig");
// pub const ops = struct {
//     pub const create = @import("tensor/ops/create.zig");
//     pub const elementwise = @import("tensor/ops/elementwise.zig");
//     pub const reduce = @import("tensor/ops/reduce.zig");
//     pub const matmul = @import("tensor/ops/matmul.zig");
//     pub const unary = @import("tensor/ops/unary.zig");
//     pub const softmax = @import("tensor/ops/softmax.zig");
//     pub const loss = @import("tensor/ops/loss.zig");
// };

// Stage 3 additions:
// pub const autograd = @import("autograd/node.zig");
// pub const tape = @import("autograd/tape.zig");
// pub const backward = @import("autograd/backward.zig");
// pub const gradcheck = @import("autograd/gradcheck.zig");

// Stage 4 additions:
// pub const nn = struct {
//     pub const module = @import("nn/module.zig");
//     pub const linear = @import("nn/linear.zig");
//     pub const embedding = @import("nn/embedding.zig");
//     pub const layernorm = @import("nn/layernorm.zig");
//     pub const activations = @import("nn/activations.zig");
//     pub const attention = @import("nn/attention.zig");
//     pub const mlp = @import("nn/mlp.zig");
//     pub const block = @import("nn/block.zig");
//     pub const model = @import("nn/model.zig");
// };
// pub const optim = struct {
//     pub const optimizer = @import("optim/optimizer.zig");
//     pub const sgd = @import("optim/sgd.zig");
//     pub const adamw = @import("optim/adamw.zig");
// };

// Stage 5 additions:
// pub const tokenizer = struct {
//     pub const vocab = @import("tokenizer/vocab.zig");
//     pub const word = @import("tokenizer/word.zig");
// };
// pub const data = struct {
//     pub const dataset = @import("data/dataset.zig");
//     pub const windowing = @import("data/windowing.zig");
//     pub const batcher = @import("data/batcher.zig");
// };

// Stage 7 additions:
// pub const backend = @import("backend/backend.zig");

test {
    _ = errors;
}
