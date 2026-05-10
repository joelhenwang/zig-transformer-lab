//!
//! 07_generate.zig — Load checkpoint and generate text
//!
//! Demonstrates how to load a saved model checkpoint and generate
//! text autoregressively using top-k sampling with temperature.
//!
//! This example should be run AFTER 06_train_shakespeare, which
//! creates the checkpoint file "shakespeare_ckpt.bin".
//!
//! Usage:
//!   zig build run-example -Dexample=07_generate
//!
//! Generation algorithm (autoregressive):
//!   1. Start with a prompt (e.g., "the")
//!   2. Feed the last T tokens into the model
//!   3. Get logits for the next token from the last position
//!   4. Apply temperature, then top-k filtering
//!   5. Sample from the resulting probability distribution
//!   6. Append the sampled token and repeat from step 2
//!
//! The model generates ONE token at a time, using its own previous
//! outputs as context for the next prediction. This is why it's called
//! "autoregressive" — the model regresses on (feeds on) its own outputs.
//;

const std = @import("std");
const ztl = @import("zig_transformer_lab");
const TransformerConfig = ztl.nn.module.TransformerConfig;
const TinyWordTransformer = ztl.nn.model.TinyWordTransformer;
const Rng = ztl.rng.Rng;
const Dataset = ztl.data.dataset.Dataset;
const generate = ztl.lab.generate;
const GenerateOpts = ztl.lab.GenerateOpts;
const decode = ztl.tokenizer.word.decode;

pub fn main(init: std.process.Init) !void {
    var stderr_buf: [0]u8 = undefined;
    const locked_stderr = init.io.lockStderr(&stderr_buf, null) catch return;
    const w = &locked_stderr.file_writer.interface;
    const allocator = std.heap.page_allocator;
    const io = init.io;

    const ckpt_path = "shakespeare_ckpt.bin";
    const max_vocab: usize = 2000;
    const T: usize = 16;
    const D: usize = 32;
    const max_new_tokens: usize = 50;

    try w.print("=== 07: Generate Text from Checkpoint ===\n\n", .{});

    // We need the vocabulary to encode the prompt and decode the output.
    // The simplest way: load the same dataset (which builds the same vocab).
    var ds = try Dataset.init(allocator, io, "data/tinyshakespeare.txt", max_vocab, true);
    defer ds.deinit();

    const V = ds.vocab.size();

    // Create a model with the same architecture as the saved checkpoint
    const cfg = TransformerConfig{
        .vocab_size = V,
        .d_model = D,
        .max_seq_len = T,
        .d_ff = D * 4,
        .ln_eps = 1e-5,
        .bias = true,
    };

    var model_rng = Rng.init(0);
    var model = try TinyWordTransformer.init(allocator, cfg, &model_rng);
    defer model.deinit();

    // Load the checkpoint (overwrites the random initialization)
    try w.print("Loading checkpoint: {s}\n", .{ckpt_path});
    model.load(io, ckpt_path) catch {
        try w.print("WARNING: Could not load checkpoint. Using random weights.\n", .{});
        try w.print("Run example 06 first to create the checkpoint.\n\n", .{});
    };

    // Encode a prompt
    const prompt_text = "the";
    const prompt_tokens = try ztl.tokenizer.word.encode(allocator, prompt_text, &ds.vocab);
    defer allocator.free(prompt_tokens);

    try w.print("Prompt: \"{s}\" → tokens {any}\n\n", .{ prompt_text, prompt_tokens });

    // Generate with different settings
    const gen_configs = [_]struct {
        label: []const u8,
        opts: GenerateOpts,
        seed: u64,
    }{
        .{
            .label = "Greedy (top_k=1, temp=1.0)",
            .opts = .{ .top_k = 1, .temperature = 1.0 },
            .seed = 42,
        },
        .{
            .label = "Top-5 sampling (temp=1.0)",
            .opts = .{ .top_k = 5, .temperature = 1.0 },
            .seed = 42,
        },
        .{
            .label = "Top-5 sampling (temp=0.7, more focused)",
            .opts = .{ .top_k = 5, .temperature = 0.7 },
            .seed = 42,
        },
        .{
            .label = "Top-10 sampling (temp=1.2, more creative)",
            .opts = .{ .top_k = 10, .temperature = 1.2 },
            .seed = 42,
        },
    };

    for (gen_configs) |gc| {
        var gen_rng = Rng.init(gc.seed);
        const generated_ids = try generate(
            allocator,
            &model,
            prompt_tokens,
            max_new_tokens,
            &gen_rng,
            gc.opts,
        );
        defer allocator.free(generated_ids);

        // Decode the generated tokens back to text
        const text = try decode(allocator, generated_ids, &ds.vocab);
        defer allocator.free(text);

        try w.print("--- {s} ---\n", .{gc.label});
        try w.print("  {s}\n\n", .{text});
    }

    try w.print("=== Generation Complete ===\n", .{});
}
