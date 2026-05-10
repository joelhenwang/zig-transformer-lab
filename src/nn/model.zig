//!
//! zig-transformer-lab — TinyWordTransformer model
//!
//! Purpose:
//!   The complete transformer language model: token embedding +
//!   positional embedding → transformer block → final LayerNorm →
//!   output projection (lm_head). This model predicts the next token
//!   given a sequence of token IDs.
//!
//!   Architecture (pre-norm, 1 block, 1 head):
//!     tok_embed(ids) + pos_embed(positions)  → (B, T, D)
//!     block(x)                                → (B, T, D)
//!     ln_f(x)                                → (B, T, D)
//!     lm_head(x)                             → (B, T, V)  logits
//!
//! Shape contract:
//!   forward(ids: (B, T), tape) → logits: (B, T, V)
//!
//! Math:
//!   logits = lm_head(LN_f(Block(tok_embed(ids) + pos_embed)))
//!
//!   Position IDs are [0, 1, ..., T-1] for each position in the
//!   sequence. Weight tying (sharing tok_embed and lm_head weights)
//!   is off by default — see plan.md Appendix for the exercise.
//!
//! Memory ownership:
//!   Owns all sub-layers and embeddings. Freed in deinit().
//!
//! Errors:
//!   OutOfMemory — from sub-layer allocation
//!   IoError — from save/load
//!   InvalidArgument — from invalid checkpoint data
//!
//! Credits:
//!   Architecture follows GPT-2 (Radford et al., 2019).
//!   No code copied.

const std = @import("std");
const LabError = @import("../core/errors.zig").LabError;
const Tensor = @import("../tensor/tensor.zig").Tensor;
const Shape = @import("../tensor/shape.zig").Shape;
const Tape = @import("../autograd/tape.zig").Tape;
const Rng = @import("../core/rng.zig").Rng;
const TransformerConfig = @import("module.zig").TransformerConfig;
const Embedding = @import("embedding.zig").Embedding;
const LayerNorm = @import("layernorm.zig").LayerNorm;
const Linear = @import("linear.zig").Linear;
const TransformerBlock = @import("block.zig").TransformerBlock;
const ops_elementwise = @import("../tensor/ops/elementwise.zig");
const ops_create = @import("../tensor/ops/create.zig");

pub const TinyWordTransformer = struct {
    tok_embed: Embedding,
    pos_embed: Embedding,
    block: TransformerBlock,
    ln_f: LayerNorm,
    lm_head: Linear,
    allocator: std.mem.Allocator,
    cfg: TransformerConfig,

    /// Create a TinyWordTransformer from a config.
    ///
    /// All parameters are randomly initialized. The model is
    /// pedagogically small: V=64, D=32, T=16 by default.
    ///
    /// Worked example:
    ///   const cfg = TransformerConfig{};
    ///   var model = try TinyWordTransformer.init(alloc, cfg, &rng);
    ///   defer model.deinit();
    pub fn init(
        allocator: std.mem.Allocator,
        cfg: TransformerConfig,
        rng: *Rng,
    ) LabError!TinyWordTransformer {
        var tok_embed = try Embedding.init(allocator, cfg.vocab_size, cfg.d_model, rng);
        errdefer tok_embed.deinit();
        var pos_embed = try Embedding.init(allocator, cfg.max_seq_len, cfg.d_model, rng);
        errdefer pos_embed.deinit();
        var block = try TransformerBlock.init(allocator, cfg, rng);
        errdefer block.deinit();
        var ln_f = try LayerNorm.init(allocator, cfg.d_model, cfg.ln_eps, rng);
        errdefer ln_f.deinit();
        var lm_head = try Linear.init(allocator, cfg.d_model, cfg.vocab_size, false, rng);
        errdefer lm_head.deinit();

        return TinyWordTransformer{
            .tok_embed = tok_embed,
            .pos_embed = pos_embed,
            .block = block,
            .ln_f = ln_f,
            .lm_head = lm_head,
            .allocator = allocator,
            .cfg = cfg,
        };
    }

    /// Forward pass: ids → logits.
    ///
    /// Token embeddings + positional embeddings → block → LN → lm_head.
    ///
    /// Worked example:
    ///   // ids shape (2, 4) with values in [0, 64)
    ///   var logits = try model.forward(ids, &tape);
    ///   // logits.shape == (2, 4, 64)
    pub fn forward(self: TinyWordTransformer, ids: Tensor, tape: ?*Tape) LabError!Tensor {
        _ = ids.shape.dims[0]; // B (unused directly, T used below)
        const T = ids.shape.dims[1];

        // Token embeddings: (B, T) → (B, T, D)
        var tok = try self.tok_embed.forward(ids, tape);
        if (tape) |t| try t.keepAlive(&tok);
        defer tok.deinit(self.allocator);

        // Position IDs: [0, 1, ..., T-1], shape (1, T) or (T,)
        var pos_ids = try Tensor.init(self.allocator, Shape.init2D(1, T));
        if (tape) |t| try t.keepAlive(&pos_ids);
        defer pos_ids.deinit(self.allocator);
        for (0..T) |t| {
            pos_ids.data[t] = @floatFromInt(t);
        }

        // Position embeddings: (1, T) → (1, T, D)
        var pos = try self.pos_embed.forward(pos_ids, tape);
        if (tape) |t| try t.keepAlive(&pos);
        defer pos.deinit(self.allocator);

        // x = tok_embed + pos_embed  (pos broadcasts over B)
        var x = try ops_elementwise.add(self.allocator, tok, pos, tape);
        if (tape) |t| try t.keepAlive(&x);
        defer x.deinit(self.allocator);

        // Transformer block
        var block_out = try self.block.forward(x, tape);
        if (tape) |t| try t.keepAlive(&block_out);
        defer block_out.deinit(self.allocator);

        // Final LayerNorm
        var ln_out = try self.ln_f.forward(block_out, tape);
        if (tape) |t| try t.keepAlive(&ln_out);
        defer ln_out.deinit(self.allocator);

        // Output projection: (B, T, D) → (B, T, V)
        const logits = try self.lm_head.forward(ln_out, tape);

        return logits;
    }

    /// Collect all learnable parameters into a flat list.
    ///
    /// The optimizer uses this list to iterate over all parameters.
    /// Order doesn't matter for correctness — every parameter gets
    /// updated exactly once per step.
    pub fn parameters(self: *TinyWordTransformer, list: *std.ArrayList(*Tensor)) void {
        self.tok_embed.parameters(list);
        self.pos_embed.parameters(list);
        self.block.parameters(list);
        self.ln_f.parameters(list);
        self.lm_head.parameters(list);
    }

    /// Count total parameters (for display / debugging).
    pub fn paramCount(self: TinyWordTransformer) usize {
        const te = self.tok_embed.vocab_size * self.tok_embed.d_model;
        const pe = self.pos_embed.vocab_size * self.pos_embed.d_model;
        // LN: 2 * D, Block has LN*2 + Attn(4*2D) + MLP(2*2D)... approximate
        const ln_f = 2 * self.cfg.d_model;
        // Rough estimate — exact count would iterate parameters
        return te + pe + ln_f;
    }

    /// Save model checkpoint to a binary file.
    ///
    /// Format: see plan.md Appendix H:
    ///   magic "TWTL" (4 bytes)
    ///   version: u32 = 1
    ///   num_params: u32
    ///   For each param: name_len(u32), name([]u8), rank(u8), dims([4]u32), data_len(u32), data([]f32)
    ///
    /// Worked example:
    ///   try model.save(io, "checkpoint.bin");
    pub fn save(self: TinyWordTransformer, io: std.Io, path: []const u8) !void {
        const cwd = std.Io.Dir.cwd();
        const file = try cwd.createFile(io, path, .{});
        defer file.close(io);
        var buf: [4096]u8 = undefined;
        var writer = file.writer(io, &buf);

        // Magic
        try writer.interface.writeAll("TWTL");
        // Version
        try writer.interface.writeInt(u32, 1, .little);

        // Collect parameters with names
        var param_list: std.ArrayList(struct { name: []const u8, tensor: *Tensor }) = .empty;
        defer param_list.deinit(self.allocator);
        try self.collectNamedParams(&param_list);

        try writer.interface.writeInt(u32, @intCast(param_list.items.len), .little);

        for (param_list.items) |entry| {
            const t = entry.tensor;
            try writer.interface.writeInt(u32, @intCast(entry.name.len), .little);
            try writer.interface.writeAll(entry.name);
            try writer.interface.writeInt(u8, @intCast(t.shape.ndim()), .little);
            for (0..4) |i| {
                const dim: u32 = if (i < t.shape.ndim()) @intCast(t.shape.dims[i]) else 0;
                try writer.interface.writeInt(u32, dim, .little);
            }
            const data_len: u32 = @intCast(t.data.len * 4);
            try writer.interface.writeInt(u32, data_len, .little);
            // Write f32 data as little-endian bytes
            const bytes = std.mem.sliceAsBytes(t.data);
            try writer.interface.writeAll(bytes);
        }
        try writer.flush();
    }

    /// Load model checkpoint from a binary file.
    ///
    /// Verifies magic and version. Overwrites existing parameter data.
    ///
    /// Worked example:
    ///   try model.load(io, "checkpoint.bin");
    pub fn load(self: TinyWordTransformer, io: std.Io, path: []const u8) !void {
        const cwd = std.Io.Dir.cwd();
        const file = try cwd.openFile(io, path, .{});
        defer file.close(io);
        var read_buf: [4096]u8 = undefined;
        var reader = file.reader(io, &read_buf);

        // Verify magic
        var magic: [4]u8 = undefined;
        try reader.interface.readNoEof(&magic);
        if (!std.mem.eql(u8, &magic, "TWTL")) return error.IoError;

        // Verify version
        const version = try reader.interface.readInt(u32, .little);
        if (version != 1) return error.IoError;

        const num_params = try reader.interface.readInt(u32, .little);

        // Collect parameters with names for matching
        var param_list: std.ArrayList(struct { name: []const u8, tensor: *Tensor }) = .empty;
        defer param_list.deinit(self.allocator);
        try self.collectNamedParams(&param_list);

        for (0..num_params) |_| {
            const name_len = try reader.interface.readInt(u32, .little);
            var name_buf: [256]u8 = undefined;
            try reader.interface.readNoEof(name_buf[0..name_len]);
            const name = name_buf[0..name_len];

            _ = try reader.interface.readInt(u8, .little); // rank (unused for validation)

            // Read dims
            var dims: [4]u32 = undefined;
            for (0..4) |i| {
                dims[i] = try reader.interface.readInt(u32, .little);
            }

            const data_len: usize = try reader.interface.readInt(u32, .little);

            // Find matching parameter by name
            for (param_list.items) |entry| {
                if (std.mem.eql(u8, name, entry.name)) {
                    const bytes = std.mem.sliceAsBytes(entry.tensor.data);
                    try reader.interface.readNoEof(bytes);
                    break;
                }
            } else {
                // Unknown parameter — skip its data
                var skip_buf: [4096]u8 = undefined;
                var remaining: usize = data_len;
                while (remaining > 0) {
                    const to_read = @min(remaining, skip_buf.len);
                    try reader.interface.readNoEof(skip_buf[0..to_read]);
                    remaining -= to_read;
                }
            }
        }
    }

    /// Internal: collect parameters with their names.
    fn collectNamedParams(
        self: TinyWordTransformer,
        list: *std.ArrayList(struct { name: []const u8, tensor: *Tensor }),
    ) !void {
        try list.append(self.allocator, .{ .name = "tok_embed.weight", .tensor = &self.tok_embed.weight });
        try list.append(self.allocator, .{ .name = "pos_embed.weight", .tensor = &self.pos_embed.weight });
        try list.append(self.allocator, .{ .name = "block.ln1.gamma", .tensor = &self.block.ln1.gamma });
        try list.append(self.allocator, .{ .name = "block.ln1.beta", .tensor = &self.block.ln1.beta });
        try list.append(self.allocator, .{ .name = "block.attn.w_q.weight", .tensor = &self.block.attn.w_q.weight });
        try list.append(self.allocator, .{ .name = "block.attn.w_k.weight", .tensor = &self.block.attn.w_k.weight });
        try list.append(self.allocator, .{ .name = "block.attn.w_v.weight", .tensor = &self.block.attn.w_v.weight });
        try list.append(self.allocator, .{ .name = "block.attn.w_o.weight", .tensor = &self.block.attn.w_o.weight });
        try list.append(self.allocator, .{ .name = "block.ln2.gamma", .tensor = &self.block.ln2.gamma });
        try list.append(self.allocator, .{ .name = "block.ln2.beta", .tensor = &self.block.ln2.beta });
        try list.append(self.allocator, .{ .name = "block.mlp.fc1.weight", .tensor = &self.block.mlp.fc1.weight });
        try list.append(self.allocator, .{ .name = "block.mlp.fc2.weight", .tensor = &self.block.mlp.fc2.weight });
        try list.append(self.allocator, .{ .name = "ln_f.gamma", .tensor = &self.ln_f.gamma });
        try list.append(self.allocator, .{ .name = "ln_f.beta", .tensor = &self.ln_f.beta });
        try list.append(self.allocator, .{ .name = "lm_head.weight", .tensor = &self.lm_head.weight });
    }

    /// Free all sub-layers and embeddings.
    pub fn deinit(self: *TinyWordTransformer) void {
        self.tok_embed.deinit();
        self.pos_embed.deinit();
        self.block.deinit();
        self.ln_f.deinit();
        self.lm_head.deinit();
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "TinyWordTransformer init — creates all sub-layers" {
    const alloc = std.testing.allocator;
    var rng = Rng.init(42);
    const cfg = TransformerConfig{
        .vocab_size = 16,
        .d_model = 8,
        .max_seq_len = 4,
        .d_ff = 32,
    };

    var model = try TinyWordTransformer.init(alloc, cfg, &rng);
    defer model.deinit();

    // Check embedding shapes
    try std.testing.expectEqual(@as(usize, 16), model.tok_embed.weight.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 4), model.pos_embed.weight.shape.dims[0]);
}

test "TinyWordTransformer forward — produces (B, T, V) logits" {
    const alloc = std.testing.allocator;
    var rng = Rng.init(42);
    const cfg = TransformerConfig{
        .vocab_size = 16,
        .d_model = 8,
        .max_seq_len = 4,
        .d_ff = 32,
    };

    var model = try TinyWordTransformer.init(alloc, cfg, &rng);
    defer model.deinit();

    // Input: batch of 2 sequences of length 3
    var ids = try Tensor.init(alloc, Shape.init2D(2, 3));
    defer ids.deinit(alloc);
    for (0..6) |i| ids.data[i] = @floatFromInt(i % 16);

    var logits = try model.forward(ids, null);
    defer logits.deinit(alloc);

    try std.testing.expectEqual(@as(usize, 2), logits.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 3), logits.shape.dims[1]);
    try std.testing.expectEqual(@as(usize, 16), logits.shape.dims[2]);

    // All logits should be finite
    for (0..logits.data.len) |i| {
        try std.testing.expect(std.math.isFinite(logits.data[i]));
    }
}

test "TinyWordTransformer parameters — non-empty list" {
    const alloc = std.testing.allocator;
    var rng = Rng.init(42);
    const cfg = TransformerConfig{
        .vocab_size = 16,
        .d_model = 8,
        .max_seq_len = 4,
        .d_ff = 32,
    };

    var model = try TinyWordTransformer.init(alloc, cfg, &rng);
    defer model.deinit();

    var param_list: std.ArrayList(*Tensor) = .empty;
    defer param_list.deinit(alloc);
    try param_list.ensureTotalCapacity(alloc, 32);
    model.parameters(&param_list);

    try std.testing.expect(param_list.items.len > 0);
}
