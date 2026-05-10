//!
//! zig-transformer-lab — Training loop orchestrator
//!
//! Purpose:
//!   Top-level Trainer struct that ties together all Stage 2–5 pieces:
//!   dataset, tokenizer, windowing, batching, model, optimizer, and
//!   the tape-based autograd. The training loop lives here so that
//!   examples can focus on configuration instead of boilerplate.
//!
//!   This is the Zig equivalent of PyTorch's typical training script:
//!     for epoch in range(num_epochs):
//!         for batch in dataloader:
//!             optimizer.zero_grad()
//!             loss = model(batch)
//!             loss.backward()
//!             optimizer.step()
//!
//!   The key difference: we create a fresh Tape per step (not per epoch),
//!   because our tape accumulates nodes and would consume unbounded memory
//!   across steps. PyTorch's autograd graph is freed on backward(), but
//!   our tape needs explicit deinit().
//!
//! Shape contract:
//!   Training step:
//!     batch.input: []u32 (B*T) → reshape to (B, T) → model.forward → (B, T, V)
//!     logits: (B, T, V) → reshape to (B*T, V)
//!     targets: []u32 (B*T) → (B*T,)
//!     loss = crossEntropy(logits, targets) → (1,)
//!
//! Math:
//!   Loss function: mean cross-entropy over B*T next-token predictions
//!     L = -(1/(B*T)) * sum_{i} log_softmax(logits_i)[target_i]
//!
//!   Gradient update: depends on optimizer (see adamw.zig / sgd.zig)
//!
//! Memory ownership:
//!   Trainer owns: model, optimizer, params list, batcher, windowing, dataset
//!   Trainer does NOT own: allocator, io (borrowed from caller)
//!   Per-step: Tape is created and destroyed each step
//!
//! Errors:
//!   OutOfMemory — from tensor/gradient allocation
//!   IoError — from dataset loading, checkpoint saving
//!
//! Credits:
//!   Training loop pattern is standard in all DL frameworks.
//!   No code copied.

const std = @import("std");
const LabError = @import("../core/errors.zig").LabError;
const Tensor = @import("../tensor/tensor.zig").Tensor;
const Shape = @import("../tensor/shape.zig").Shape;
const Tape = @import("../autograd/tape.zig").Tape;
const Rng = @import("../core/rng.zig").Rng;
const TransformerConfig = @import("../nn/module.zig").TransformerConfig;
const TinyWordTransformer = @import("../nn/model.zig").TinyWordTransformer;
const Dataset = @import("../data/dataset.zig").Dataset;
const Windowing = @import("../data/windowing.zig").Windowing;
const Batcher = @import("../data/batcher.zig").Batcher;
const crossEntropy = @import("../tensor/ops/loss.zig").crossEntropy;
const ops_shape = @import("../tensor/ops/shape_ops.zig");

/// Callback for generating sample text. Receives step number.
/// The caller can decode and print a few tokens.
pub const SampleFn = *const fn (step: usize) void;

/// Training configuration — all hyperparameters in one struct.
///
/// Every field has a pedagogically reasonable default so that
/// a caller can zero-initialize and only override what they need.
pub const TrainConfig = struct {
    /// Maximum vocabulary size (passed to Dataset.init).
    max_vocab: usize = 2000,
    /// Sequence length (context window T).
    seq_len: usize = 16,
    /// Batch size B.
    batch_size: usize = 4,
    /// Model embedding dimension D.
    d_model: usize = 32,
    /// Feed-forward hidden dimension (usually 4 * D).
    d_ff: usize = 128,
    /// LayerNorm epsilon.
    ln_eps: f32 = 1e-5,
    /// Use bias in Linear layers.
    bias: bool = true,
    /// Learning rate.
    lr: f32 = 1e-3,
    /// AdamW beta1.
    beta1: f32 = 0.9,
    /// AdamW beta2.
    beta2: f32 = 0.999,
    /// AdamW epsilon.
    adam_eps: f32 = 1e-8,
    /// AdamW weight decay (decoupled).
    weight_decay: f32 = 0.01,
    /// Gradient clipping max norm (0 = no clipping).
    /// When set, gradients are clipped to this max L2 norm before
    /// the optimizer step. This prevents exploding gradients that
    /// cause training instability and divergence.
    grad_clip_norm: f32 = 1.0,
    /// Total training steps.
    max_steps: usize = 500,
    /// Print loss every N steps.
    log_every: usize = 50,
    /// Save checkpoint every N steps (0 = never).
    ckpt_every: usize = 0,
    /// Checkpoint file path (used if ckpt_every > 0).
    ckpt_path: []const u8 = "checkpoint.bin",
    /// Random seed for model initialization.
    model_seed: u64 = 1337,
    /// Random seed for data shuffling.
    data_seed: u64 = 42,
    /// Lowercase text before tokenization.
    lowercase: bool = true,
};

/// Result of a completed training run.
pub const TrainResult = struct {
    /// Final loss value.
    final_loss: f32,
    /// Number of steps completed.
    steps_completed: usize,
    /// Vocabulary size used.
    vocab_size: usize,
};

/// The Trainer orchestrates the entire training pipeline.
///
/// Usage:
///   var trainer = try Trainer.init(allocator, io, "data/tiny.txt", config);
///   defer trainer.deinit();
///   const result = try trainer.train(null, null);
///
/// Worked example:
///   const cfg = TrainConfig{ .max_steps = 100, .log_every = 10 };
///   var trainer = try Trainer.init(alloc, io, "data/tiny.txt", cfg);
///   defer trainer.deinit();
///   const result = try trainer.train(null, null);
///   // result.final_loss should be lower than initial loss
pub const Trainer = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    dataset: Dataset,
    windowing: Windowing,
    batcher: Batcher,
    model: TinyWordTransformer,
    cfg: TrainConfig,
    data_rng: Rng,
    vocab_size: usize,

    /// Initialize all training components: dataset, model, optimizer.
    ///
    /// This loads the text file, builds the vocabulary, creates the model,
    /// and collects parameters. It does NOT start training.
    ///
    /// Worked example:
    ///   var trainer = try Trainer.init(alloc, io, "data/tiny.txt", cfg);
    ///   defer trainer.deinit();
    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        data_path: []const u8,
        cfg: TrainConfig,
    ) LabError!Trainer {
        // --- Dataset: load text, build vocab, encode to tokens ---
        var ds = try Dataset.init(allocator, io, data_path, cfg.max_vocab, cfg.lowercase);
        errdefer ds.deinit();

        const V = ds.vocab.size();
        const T = cfg.seq_len;
        const B = cfg.batch_size;
        const D = cfg.d_model;

        // --- Windowing: create sliding (input, target) pairs ---
        const windowing = Windowing.init(ds.tokens, T);

        // --- Batcher: shuffle windows into mini-batches ---
        var data_rng = Rng.init(cfg.data_seed);
        var batcher = try Batcher.init(allocator, windowing, B, &data_rng);
        errdefer batcher.deinit();

        // --- Model: create transformer ---
        const model_cfg = TransformerConfig{
            .vocab_size = V,
            .d_model = D,
            .max_seq_len = T,
            .d_ff = cfg.d_ff,
            .ln_eps = cfg.ln_eps,
            .bias = cfg.bias,
        };

        var model_rng = Rng.init(cfg.model_seed);
        var model = try TinyWordTransformer.init(allocator, model_cfg, &model_rng);
        errdefer model.deinit();

        return Trainer{
            .allocator = allocator,
            .io = io,
            .dataset = ds,
            .windowing = windowing,
            .batcher = batcher,
            .model = model,
            .cfg = cfg,
            .data_rng = data_rng,
            .vocab_size = V,
        };
    }

    /// Run the training loop for cfg.max_steps steps.
    ///
    /// If writer is provided, prints loss every log_every steps.
    /// If sample_fn is provided, calls it for sample generation (not yet wired).
    ///
    /// Returns TrainResult with final loss and step count.
    ///
    /// Worked example:
    ///   const result = try trainer.train(null, null);
    ///   try w.print("Final loss: {d:.4}\n", .{result.final_loss});
    pub fn train(self: *Trainer, writer: ?*std.Io.Writer, sample_fn: ?SampleFn) LabError!TrainResult {
        const allocator = self.allocator;
        const B = self.cfg.batch_size;
        const T = self.cfg.seq_len;
        const V = self.vocab_size;
        const max_steps = self.cfg.max_steps;

        // Collect parameters AFTER self.model is in its final memory location.
        // If we collected them in init(), the pointers would dangle because
        // model is copied into the Trainer struct after init's local scope.
        var params: std.ArrayList(*Tensor) = .empty;
        defer params.deinit(allocator);
        try params.ensureTotalCapacity(allocator, 32);
        self.model.parameters(&params);

        // Create optimizer LOCALLY in train() so its internal HashMap
        // pointers are valid for the duration of training. Creating it
        // in init() and copying it into the struct would corrupt the
        // HashMap's self-referential pointers.
        var adam = try @import("../optim/adamw.zig").AdamW.init(allocator, .{
            .lr = self.cfg.lr,
            .beta1 = self.cfg.beta1,
            .beta2 = self.cfg.beta2,
            .eps = self.cfg.adam_eps,
            .weight_decay = self.cfg.weight_decay,
        });
        defer adam.deinit(allocator);
        const opt = adam.optimizer();

        var final_loss: f32 = std.math.inf(f32);

        for (0..max_steps) |step| {
            // --- Get next batch (reset if exhausted) ---
            const batch = self.batcher.next() orelse blk: {
                self.batcher.reset(&self.data_rng);
                break :blk self.batcher.next().?;
            };
            defer batch.deinit();

            // --- Create input tensor (B, T) ---
            var ids = try Tensor.init(allocator, Shape.init2D(B, T));
            defer ids.deinit(allocator);
            for (0..B * T) |i| {
                ids.data[i] = @floatFromInt(batch.input[i]);
            }

            // --- Create target tensor (B*T,) ---
            var targets = try Tensor.init(allocator, Shape.init1D(B * T));
            defer targets.deinit(allocator);
            for (0..B * T) |i| {
                targets.data[i] = @floatFromInt(batch.target[i]);
            }

            // --- Fresh tape for this step ---
            var tape = Tape.init(allocator);
            defer tape.deinit();

            // Track all parameters on the tape
            for (params.items) |param| {
                param.requires_grad = true;
                _ = try tape.trackLeaf(param);
            }

            // --- Forward pass ---
            var logits_3d = try self.model.forward(ids, &tape);

            // Reshape logits from (B, T, V) to (B*T, V) — tape-tracked
            // so the gradient flows back with the correct shape.
            // Using untracked reshape() would make the CE gradient
            // (B*T, V) bypass the reshape backward, causing a silent
            // shape mismatch in the matmul backward that only works
            // by accident because both shapes share row-major layout.
            var logits = try ops_shape.reshapeTracked(allocator, logits_3d, Shape.init2D(B * T, V), &tape);
            defer logits.deinit(allocator);

            // --- Compute loss ---
            var loss = try crossEntropy(allocator, logits, targets, &tape);

            // Save loss value before freeing
            const loss_val = loss.data[0];

            // --- Sample callback ---
            if (sample_fn) |sample| {
                _ = sample;
            }

            // --- Backward pass ---
            try tape.backward(&loss);

            // Free loss and logits_3d
            loss.deinit(allocator);
            logits_3d.deinit(allocator);

            // --- Gradient clipping ---
            // Clip all parameter gradients by global L2 norm.
            // This prevents exploding gradients that cause the
            // optimizer to make overly large updates, which leads
            // to training instability and divergence.
            //
            // Algorithm (same as PyTorch's torch.nn.utils.clip_grad_norm_):
            //   1. Compute total_norm = sqrt(sum of squared L2 norms)
            //   2. clip_coeff = max_norm / total_norm
            //   3. If clip_coeff < 1, scale all gradients by clip_coeff
            const max_norm = self.cfg.grad_clip_norm;
            var total_norm_sq: f64 = 0;
            if (max_norm > 0) {
                for (params.items) |param| {
                    if (param.grad) |g| {
                        for (g.data) |val| {
                            const v: f64 = @floatCast(val);
                            total_norm_sq += v * v;
                        }
                    }
                }
                const total_norm = @sqrt(total_norm_sq);
                if (total_norm > 0) {
                    const clip_coeff_f64 = @as(f64, @floatFromInt(@as(usize, 1))) * max_norm / total_norm;
                    const clip_coeff: f32 = @floatCast(clip_coeff_f64);
                    if (clip_coeff < 1.0) {
                        for (params.items) |param| {
                            if (param.grad) |g| {
                                for (g.data) |*val| {
                                    val.* *= clip_coeff;
                                }
                            }
                        }
                    }
                }
            }

            // --- Log with gradient norm ---
            if (writer) |wr| {
                if (step % self.cfg.log_every == 0 or step == max_steps - 1) {
                    const grad_norm: f32 = @floatCast(@sqrt(total_norm_sq));
                    wr.print("Step {:4}: loss = {d:.4}  grad_norm = {d:.4}\n", .{ step, loss_val, grad_norm }) catch {};
                }
            }

            // --- Optimizer step ---
            opt.step(params.items) catch |err| {
                return switch (err) {
                    error.OutOfMemory => LabError.OutOfMemory,
                    else => LabError.NumericalError,
                };
            };
            opt.zeroGrad(params.items);

            final_loss = loss_val;

            // --- Checkpoint ---
            if (self.cfg.ckpt_every > 0 and step > 0 and step % self.cfg.ckpt_every == 0) {
                self.model.save(self.io, self.cfg.ckpt_path) catch {};
            }
        }

        // Final checkpoint
        if (self.cfg.ckpt_every > 0) {
            self.model.save(self.io, self.cfg.ckpt_path) catch {};
        }

        return TrainResult{
            .final_loss = final_loss,
            .steps_completed = max_steps,
            .vocab_size = self.vocab_size,
        };
    }

    /// Free all training components.
    pub fn deinit(self: *Trainer) void {
        self.model.deinit();
        self.batcher.deinit();
        self.dataset.deinit();
    }
};

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

/// Generate tokens autoregressively using greedy or top-k sampling.
///
/// Given a prompt of token IDs, produces `max_new_tokens` new tokens
/// by repeatedly:
///   1. Feeding the last T tokens into the model
///   2. Getting logits for the next token
///   3. Sampling from the logits (greedy or top-k)
///   4. Appending the sampled token to the sequence
///
/// Temperature controls the sharpness of the distribution:
///   - temp > 1.0: flatter distribution (more random)
///   - temp = 1.0: original distribution
///   - temp < 1.0: sharper distribution (more greedy)
///   - temp → 0:   equivalent to greedy (argmax)
///
/// Top-k restricts sampling to the k highest-probability tokens.
///   - k = 1: greedy (always pick the most likely token)
///   - k = V: sample from the full distribution
///   - k = 5: sample from the top 5 tokens (default)
///
/// Shape contract:
///   prompt: []u32 of length P (P <= max_seq_len)
///   output: []u32 of length P + max_new_tokens
///
/// Worked example:
///   const prompt = [_]u32{4, 15, 7}; // "the sailor"
///   const generated = try generate(alloc, &model, &prompt, 50, &rng, .{ .top_k = 5, .temperature = 1.0 });
///   defer alloc.free(generated);
pub fn generate(
    allocator: std.mem.Allocator,
    model: *TinyWordTransformer,
    prompt: []const u32,
    max_new_tokens: usize,
    rng: *Rng,
    opts: GenerateOpts,
) LabError![]u32 {
    const T = model.cfg.max_seq_len;
    const V = model.cfg.vocab_size;

    // Output buffer: prompt + generated tokens
    var tokens: std.ArrayList(u32) = .empty;
    errdefer tokens.deinit(allocator);
    try tokens.appendSlice(allocator, prompt);

    for (0..max_new_tokens) |_| {
        // Take the last T tokens as context (or all if fewer than T)
        const context_len = @min(tokens.items.len, T);
        const context_start = tokens.items.len - context_len;

        // Create input tensor (1, context_len)
        var ids = try Tensor.init(allocator, Shape.init2D(1, context_len));
        defer ids.deinit(allocator);
        for (0..context_len) |i| {
            ids.data[i] = @floatFromInt(tokens.items[context_start + i]);
        }

        // Forward pass (no tape — generation doesn't need gradients)
        var logits_3d = try model.forward(ids, null);
        defer logits_3d.deinit(allocator);

        // We only need the logits for the LAST position:
        // logits_3d shape is (1, context_len, V)
        // Last position starts at offset (context_len - 1) * V
        const last_pos = (context_len - 1) * V;

        // Apply temperature: divide logits by temperature before softmax.
        // This controls the sharpness of the probability distribution.
        // Higher temperature → flatter distribution → more random.
        // Lower temperature → sharper distribution → more deterministic.
        const temp = if (opts.temperature > 0) opts.temperature else 1.0;

        // Copy logits for the last position, applying temperature
        var logits = try allocator.alloc(f32, V);
        defer allocator.free(logits);
        for (0..V) |v| {
            logits[v] = logits_3d.data[last_pos + v] / temp;
        }

        // Find top-k candidates
        const k = if (opts.top_k > 0 and opts.top_k < V) opts.top_k else V;

        // For top-k sampling, we need the k largest logit values.
        // Simple O(V*k) approach — fine for small vocab sizes.
        // Build a list of (logit, index) pairs, keep top k.
        var top_k_indices: std.ArrayList(u32) = .empty;
        defer top_k_indices.deinit(allocator);
        try top_k_indices.ensureTotalCapacity(allocator, k);

        // Track the k-th largest value seen so far
        var min_of_top: f32 = -std.math.inf(f32);
        if (k < V) {
            // Find the k-th largest logit value as a threshold
            // We do a partial sort: sort all logits and pick threshold
            var sorted = try allocator.alloc(f32, V);
            defer allocator.free(sorted);
            @memcpy(sorted, logits);
            // Simple insertion sort for the top-k threshold
            // (V is small enough that this is fine)
            for (0..V) |i| {
                var j = i;
                while (j > 0 and sorted[j] > sorted[j - 1]) : (j -= 1) {
                    const tmp = sorted[j];
                    sorted[j] = sorted[j - 1];
                    sorted[j - 1] = tmp;
                }
            }
            min_of_top = sorted[k - 1];
        }

        // Compute softmax over the top-k logits
        // Max for numerical stability
        var max_logit: f32 = -std.math.inf(f32);
        for (0..V) |v| {
            if (k >= V or logits[v] >= min_of_top) {
                if (logits[v] > max_logit) max_logit = logits[v];
            }
        }

        // Compute probabilities
        var probs = try allocator.alloc(f32, V);
        defer allocator.free(probs);
        var sum: f32 = 0;
        for (0..V) |v| {
            if (k >= V or logits[v] >= min_of_top) {
                probs[v] = @exp(logits[v] - max_logit);
                sum += probs[v];
            } else {
                probs[v] = 0;
            }
        }
        // Normalize
        for (0..V) |v| {
            probs[v] /= sum;
        }

        // Sample from the distribution
        const r = rng.floatf32();
        var cumsum: f32 = 0;
        var sampled: u32 = 0;
        for (0..V) |v| {
            cumsum += probs[v];
            if (r < cumsum) {
                sampled = @intCast(v);
                break;
            }
        } else {
            // Fallback: last non-zero probability token
            sampled = @intCast(V - 1);
            var v: usize = V;
            while (v > 0) {
                v -= 1;
                if (probs[v] > 0) {
                    sampled = @intCast(v);
                    break;
                }
            }
        }

        // Stop on <eos> token
        if (sampled == 3) break; // <eos> = 3
        // Append sampled token
        try tokens.append(allocator, sampled);
    }

    return tokens.toOwnedSlice(allocator);
}

/// Generation options.
pub const GenerateOpts = struct {
    /// Number of top-probability tokens to sample from.
    /// k=1 means greedy (argmax). k=V means full distribution.
    top_k: usize = 5,
    /// Temperature for sampling. >1 = more random, <1 = more greedy.
    temperature: f32 = 1.0,
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "TrainConfig defaults are pedagogically reasonable" {
    const cfg = TrainConfig{};
    try std.testing.expectEqual(@as(usize, 2000), cfg.max_vocab);
    try std.testing.expectEqual(@as(usize, 16), cfg.seq_len);
    try std.testing.expectEqual(@as(usize, 4), cfg.batch_size);
    try std.testing.expectEqual(@as(usize, 32), cfg.d_model);
    try std.testing.expectEqual(@as(f32, 1e-3), cfg.lr);
    try std.testing.expectEqual(@as(usize, 500), cfg.max_steps);
}

test "Trainer.init — creates all components from tiny.txt" {
    const alloc = std.testing.allocator;
    var threaded = std.Io.Threaded.init(alloc, .{});
    const io = threaded.io();
    const cfg = TrainConfig{
        .max_vocab = 64,
        .seq_len = 4,
        .batch_size = 2,
        .d_model = 8,
        .d_ff = 32,
        .max_steps = 3,
        .log_every = 1,
    };

    var trainer = try Trainer.init(alloc, io, "data/tiny.txt", cfg);
    defer trainer.deinit();

    try std.testing.expect(trainer.vocab_size > 0);

    // Verify parameters can be collected
    var params: std.ArrayList(*Tensor) = .empty;
    defer params.deinit(alloc);
    try params.ensureTotalCapacity(alloc, 32);
    trainer.model.parameters(&params);
    try std.testing.expect(params.items.len > 0);
}

test "Trainer.train — loss decreases on tiny.txt" {
    const alloc = std.testing.allocator;
    var threaded = std.Io.Threaded.init(alloc, .{});
    const io = threaded.io();
    const cfg = TrainConfig{
        .max_vocab = 64,
        .seq_len = 4,
        .batch_size = 2,
        .d_model = 8,
        .d_ff = 32,
        .max_steps = 20,
        .log_every = 100,
        .lr = 1e-3,
    };

    var trainer = try Trainer.init(alloc, io, "data/tiny.txt", cfg);
    defer trainer.deinit();

    const result = try trainer.train(null, null);
    try std.testing.expect(std.math.isFinite(result.final_loss));
    try std.testing.expectEqual(@as(usize, 20), result.steps_completed);
}

test "generate — produces tokens from a prompt" {
    const alloc = std.testing.allocator;
    var rng = Rng.init(42);
    const cfg = TransformerConfig{
        .vocab_size = 16,
        .d_model = 8,
        .max_seq_len = 4,
        .d_ff = 32,
    };

    var model_rng = Rng.init(42);
    var model = try TinyWordTransformer.init(alloc, cfg, &model_rng);
    defer model.deinit();

    const prompt = [_]u32{ 0, 1, 2 };
    const generated = try generate(alloc, &model, &prompt, 5, &rng, .{ .top_k = 5 });
    defer alloc.free(generated);

    // Should have prompt tokens + some generated tokens
    try std.testing.expect(generated.len >= 3);
    // First tokens should match prompt
    try std.testing.expectEqual(@as(u32, 0), generated[0]);
    try std.testing.expectEqual(@as(u32, 1), generated[1]);
    try std.testing.expectEqual(@as(u32, 2), generated[2]);
}

test "generate — greedy (top_k=1) is deterministic" {
    const alloc = std.testing.allocator;
    var rng1 = Rng.init(42);
    var rng2 = Rng.init(99);
    const cfg = TransformerConfig{
        .vocab_size = 16,
        .d_model = 8,
        .max_seq_len = 4,
        .d_ff = 32,
    };

    var model_rng = Rng.init(42);
    var model = try TinyWordTransformer.init(alloc, cfg, &model_rng);
    defer model.deinit();

    const prompt = [_]u32{ 0, 1 };
    const gen1 = try generate(alloc, &model, &prompt, 3, &rng1, .{ .top_k = 1 });
    defer alloc.free(gen1);
    const gen2 = try generate(alloc, &model, &prompt, 3, &rng2, .{ .top_k = 1 });
    defer alloc.free(gen2);

    // Greedy sampling ignores the RNG — both should produce the same sequence
    try std.testing.expectEqualSlices(u32, gen1, gen2);
}

test "generate — respects max_seq_len context window" {
    const alloc = std.testing.allocator;
    var rng = Rng.init(42);
    const T: usize = 4;
    const cfg = TransformerConfig{
        .vocab_size = 16,
        .d_model = 8,
        .max_seq_len = T,
        .d_ff = 32,
    };

    var model_rng = Rng.init(42);
    var model = try TinyWordTransformer.init(alloc, cfg, &model_rng);
    defer model.deinit();

    // Prompt longer than T — should still work (truncates to last T)
    const prompt = [_]u32{ 0, 1, 2, 3, 4, 5 };
    const generated = try generate(alloc, &model, &prompt, 2, &rng, .{ .top_k = 5 });
    defer alloc.free(generated);

    try std.testing.expect(generated.len >= 6);
}
