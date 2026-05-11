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
// Session 3: moveToCuda lives on the model and needs a CudaContext.
// Threading the import through here keeps the transfer signature
// symmetric with Tensor.toCuda.
const CudaContext = @import("../backend/cuda/context.zig").CudaContext;

/// Named struct type for parameter entries in save/load.
/// Required because anonymous structs create different types per scope
/// in Zig, which prevents passing ArrayLists across function boundaries.
pub const NamedParam = struct { name: []const u8, tensor: *Tensor };

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
        defer tok.deinit(self.allocator);

        // Position IDs: [0, 1, ..., T-1], shape (1, T). Built on CPU.
        var pos_ids_cpu = try Tensor.init(self.allocator, Shape.init2D(1, T));
        defer pos_ids_cpu.deinit(self.allocator);
        for (0..T) |t| {
            pos_ids_cpu.data[t] = @floatFromInt(t);
        }

        // Session 3: when the model lives on CUDA, upload pos_ids
        // to the same context so pos_embed.forward finds it on the
        // correct device. This is a trivial one-off (~T floats).
        var pos_ids_cuda: ?Tensor = null;
        defer {
            if (pos_ids_cuda) |*pc| pc.storage.deinit(self.allocator);
        }
        const pos_ids: Tensor = if (self.pos_embed.weight.device == .cuda) blk: {
            const ctx = switch (self.pos_embed.weight.storage) {
                .cuda => |b| b.ctx,
                .cpu => unreachable,
            };
            pos_ids_cuda = try pos_ids_cpu.toCuda(ctx);
            break :blk pos_ids_cuda.?;
        } else pos_ids_cpu;

        // Position embeddings: (1, T) → (1, T, D)
        var pos = try self.pos_embed.forward(pos_ids, tape);
        defer pos.deinit(self.allocator);

        // x = tok_embed + pos_embed  (pos broadcasts over B)
        var x = try ops_elementwise.add(self.allocator, tok, pos, tape);
        defer x.deinit(self.allocator);

        // Transformer block
        var block_out = try self.block.forward(x, tape);
        defer block_out.deinit(self.allocator);

        // Final LayerNorm
        var ln_out = try self.ln_f.forward(block_out, tape);
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

    /// Move every learnable parameter to the given CUDA context.
    ///
    /// In-place transfer: each parameter tensor's storage is replaced
    /// with a fresh `DeviceBuffer` via `Tensor.toCuda`; the old CPU
    /// buffer is freed. Shape, strides, `param_id`, and
    /// `requires_grad` are preserved. `grad` and `tape_node` are
    /// reset to `null` (consistent with `Tensor.toCuda`) — callers
    /// must re-track the moved parameters on a fresh tape.
    ///
    /// Idempotent-ish: calling `moveToCuda` on a model whose params
    /// already live on CUDA returns `error.DeviceMismatch` from
    /// `Tensor.toCuda`. Use `moveToCpu` first if you need to switch
    /// back.
    ///
    /// Worked example:
    ///   var model = try TinyWordTransformer.init(alloc, cfg, &rng);
    ///   var ctx = try CudaContext.init(alloc);
    ///   defer ctx.deinit();
    ///   try model.moveToCuda(&ctx);
    ///   // every weight now lives on GPU; forward(ids_gpu, ...) works.
    pub fn moveToCuda(self: *TinyWordTransformer, ctx: *const CudaContext) !void {
        const alloc = self.allocator;
        var list: std.ArrayList(*Tensor) = .empty;
        defer list.deinit(alloc);
        // Room for every parameter the model holds. 16 is the tight
        // bound for our 1-block config (2 LNs * 2 + 4 attn linears *
        // 2 + 2 mlp linears * 2 + 2 embeddings + 1 lm_head weight).
        try list.ensureTotalCapacity(alloc, 32);
        self.parameters(&list);

        for (list.items) |p| {
            if (p.device == .cuda) continue; // already moved — idempotent
            const new_tensor = try p.toCuda(ctx);
            // Free the old CPU buffer before overwriting the Tensor
            // struct fields. `p.*.deinit` takes an allocator and
            // handles both owned and non-owning cases.
            p.*.deinit(alloc);
            p.* = new_tensor;
        }
    }

    /// Move every learnable parameter back to host memory. Inverse
    /// of `moveToCuda`. The allocator is the one the model was
    /// created with; it owns the fresh `[]f32` buffers.
    pub fn moveToCpu(self: *TinyWordTransformer) !void {
        const alloc = self.allocator;
        var list: std.ArrayList(*Tensor) = .empty;
        defer list.deinit(alloc);
        try list.ensureTotalCapacity(alloc, 32);
        self.parameters(&list);

        for (list.items) |p| {
            if (p.device == .cpu) continue;
            const new_tensor = try p.toCpu(alloc);
            // CUDA deinit releases the DeviceBuffer via cuMemFree_v2.
            // The top-level `owned` alias is false on CUDA; the
            // storage union's `.cuda.owned` flag is what matters.
            p.*.deinit(alloc);
            p.* = new_tensor;
        }
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

    /// Save model checkpoint to a binary file (format v2, PR-η).
    ///
    /// Format — little-endian throughout, f32 payloads written as raw
    /// IEEE 754 bytes. All padding fields are zeros.
    ///
    /// Header:
    ///     magic        [4]u8  = "ZTLC"
    ///     version      u32    = 2
    ///     model_kind   u32    = 1  (TinyWordTransformer)
    ///     vocab_size   u32
    ///     max_seq_len  u32
    ///     d_model      u32
    ///     d_ff         u32
    ///     bias         u8     (0 or 1)
    ///     _pad         [3]u8  (zeros, aligns next field to 4)
    ///     param_count  u32
    ///
    /// Per-parameter record (repeated param_count times):
    ///     name_len     u32   (1..=255)
    ///     name         [name_len]u8
    ///     rank         u8
    ///     _pad         [3]u8
    ///     dims         [4]u32  (entries beyond rank are zero)
    ///     data_len     u32     (payload bytes; total_elements * 4)
    ///     data         [data_len]u8  (f32 little-endian)
    ///
    /// Trailer:
    ///     end_magic    [4]u8  = "END."
    ///
    /// The trailer lets the loader detect truncation: if we read
    /// param_count successfully but the trailer is missing, the file
    /// was cut short and we refuse to use it.
    ///
    /// Worked example:
    ///   try model.save(io, "checkpoint.bin");
    pub fn save(self: *TinyWordTransformer, io: std.Io, path: []const u8) !void {
        const cwd = std.Io.Dir.cwd();
        const file = try cwd.createFile(io, path, .{});
        defer file.close(io);
        var buf: [4096]u8 = undefined;
        var writer = file.writer(io, &buf);
        const w = &writer.interface;

        // Header
        try w.writeAll("ZTLC");
        try w.writeInt(u32, 2, .little); // version
        try w.writeInt(u32, 1, .little); // model_kind = TinyWordTransformer
        try w.writeInt(u32, @intCast(self.cfg.vocab_size), .little);
        try w.writeInt(u32, @intCast(self.cfg.max_seq_len), .little);
        try w.writeInt(u32, @intCast(self.cfg.d_model), .little);
        try w.writeInt(u32, @intCast(self.cfg.d_ff), .little);
        try w.writeInt(u8, @intFromBool(self.cfg.bias), .little);
        try w.writeAll(&.{ 0, 0, 0 }); // 3-byte pad

        // Collect parameters with names
        var param_list: std.ArrayList(NamedParam) = .empty;
        defer param_list.deinit(self.allocator);
        try self.collectNamedParams(&param_list);

        try w.writeInt(u32, @intCast(param_list.items.len), .little);

        for (param_list.items) |entry| {
            const t = entry.tensor;
            if (entry.name.len == 0 or entry.name.len > 255) return error.IoError;
            try w.writeInt(u32, @intCast(entry.name.len), .little);
            try w.writeAll(entry.name);
            try w.writeInt(u8, @intCast(t.shape.ndim()), .little);
            try w.writeAll(&.{ 0, 0, 0 }); // 3-byte pad
            for (0..4) |i| {
                const dim: u32 = if (i < t.shape.ndim()) @intCast(t.shape.dims[i]) else 0;
                try w.writeInt(u32, dim, .little);
            }
            const data_len: u32 = @intCast(t.data.len * 4);
            try w.writeInt(u32, data_len, .little);
            // Raw f32 bytes. Target platforms are little-endian; if we
            // ever port to a big-endian system we would byte-swap here.
            const bytes = std.mem.sliceAsBytes(t.data);
            try w.writeAll(bytes);
        }

        // Trailer.
        try w.writeAll("END.");
        try writer.flush();
    }

    /// Load model checkpoint from a binary file (format v2, PR-η).
    ///
    /// Validates every header field, rejects:
    ///   - wrong magic               → error.IoError
    ///   - unsupported version       → error.IoError
    ///   - wrong model_kind          → error.IoError
    ///   - config mismatch           → error.ShapeMismatch
    ///   - wrong param_count         → error.ShapeMismatch
    ///   - duplicate param name      → error.IoError
    ///   - unknown param name        → error.IoError
    ///   - missing expected param    → error.ShapeMismatch
    ///   - rank/dim/data_len mismatch for an expected param
    ///                              → error.ShapeMismatch
    ///   - missing or wrong trailer → error.IoError
    ///
    /// This strictness is deliberately noisy: a silently-mismatched
    /// checkpoint is the single most common way to waste a day of
    /// training on the wrong weights.
    ///
    /// Worked example:
    ///   try model.load(io, "checkpoint.bin");
    pub fn load(self: *TinyWordTransformer, io: std.Io, path: []const u8) !void {
        const cwd = std.Io.Dir.cwd();
        const file = try cwd.openFile(io, path, .{});
        defer file.close(io);
        var read_buf: [4096]u8 = undefined;
        var reader = file.reader(io, &read_buf);
        const r = &reader.interface;

        // --- Header ---
        var magic: [4]u8 = undefined;
        try r.readSliceAll(&magic);
        if (!std.mem.eql(u8, &magic, "ZTLC")) return error.IoError;

        const version = try r.takeInt(u32, .little);
        if (version != 2) return error.IoError;

        const model_kind = try r.takeInt(u32, .little);
        if (model_kind != 1) return error.IoError;

        const vocab_size = try r.takeInt(u32, .little);
        const max_seq_len = try r.takeInt(u32, .little);
        const d_model = try r.takeInt(u32, .little);
        const d_ff = try r.takeInt(u32, .little);
        const bias_byte = try r.takeInt(u8, .little);
        var pad3: [3]u8 = undefined;
        try r.readSliceAll(&pad3);

        if (vocab_size != self.cfg.vocab_size or
            max_seq_len != self.cfg.max_seq_len or
            d_model != self.cfg.d_model or
            d_ff != self.cfg.d_ff or
            (bias_byte != 0) != self.cfg.bias)
        {
            return error.ShapeMismatch;
        }

        const param_count = try r.takeInt(u32, .little);

        // Collect expected params.
        var param_list: std.ArrayList(NamedParam) = .empty;
        defer param_list.deinit(self.allocator);
        try self.collectNamedParams(&param_list);

        if (param_count != param_list.items.len) return error.ShapeMismatch;

        // Track which expected params we've seen; reject duplicates and
        // unknown names.
        const seen = try self.allocator.alloc(bool, param_list.items.len);
        defer self.allocator.free(seen);
        @memset(seen, false);

        var name_buf: [256]u8 = undefined;

        for (0..param_count) |_| {
            const name_len = try r.takeInt(u32, .little);
            if (name_len == 0 or name_len > 255) return error.IoError;
            try r.readSliceAll(name_buf[0..name_len]);
            const name = name_buf[0..name_len];

            const rank = try r.takeInt(u8, .little);
            var rec_pad3: [3]u8 = undefined;
            try r.readSliceAll(&rec_pad3);

            var dims: [4]u32 = undefined;
            for (0..4) |i| {
                dims[i] = try r.takeInt(u32, .little);
            }

            const data_len = try r.takeInt(u32, .little);

            // Locate the expected parameter with this name.
            var matched_index: ?usize = null;
            for (param_list.items, 0..) |entry, idx| {
                if (std.mem.eql(u8, name, entry.name)) {
                    matched_index = idx;
                    break;
                }
            }
            const idx = matched_index orelse return error.IoError;
            if (seen[idx]) return error.IoError; // duplicate
            seen[idx] = true;

            const t = param_list.items[idx].tensor;
            if (rank != t.shape.ndim()) return error.ShapeMismatch;
            for (0..4) |i| {
                const expected_dim: u32 = if (i < t.shape.ndim()) @intCast(t.shape.dims[i]) else 0;
                if (dims[i] != expected_dim) return error.ShapeMismatch;
            }
            const expected_bytes: u32 = @intCast(t.data.len * 4);
            if (data_len != expected_bytes) return error.ShapeMismatch;

            const bytes = std.mem.sliceAsBytes(t.data);
            try r.readSliceAll(bytes);
        }

        // Every expected parameter must have been written.
        for (seen, 0..) |was_seen, idx| {
            if (!was_seen) {
                _ = idx;
                return error.ShapeMismatch;
            }
        }

        // Trailer.
        var trailer: [4]u8 = undefined;
        try r.readSliceAll(&trailer);
        if (!std.mem.eql(u8, &trailer, "END.")) return error.IoError;
    }

    /// Collect parameters with their names (for debugging / grad checking).
    pub fn collectNamedParams(
        self: *TinyWordTransformer,
        list: *std.ArrayList(NamedParam),
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

// -- Checkpoint round-trip + strict validation (PR-η) -----------------------

fn testIo() !std.Io {
    const T = struct {
        var instance: ?std.Io.Threaded = null;
    };
    if (T.instance == null) {
        T.instance = std.Io.Threaded.init(std.testing.allocator, .{});
    }
    return T.instance.?.io();
}

fn tmpCheckpointPath(comptime name: []const u8) []const u8 {
    return "zig-out/tmp-" ++ name ++ ".ckpt";
}

test "Checkpoint save/load round-trip" {
    const alloc = std.testing.allocator;
    const io = try testIo();
    // Ensure zig-out/ exists for test artefacts.
    std.Io.Dir.cwd().createDirPath(io, "zig-out") catch {};

    var rng = Rng.init(7);
    const cfg = TransformerConfig{ .vocab_size = 8, .d_model = 4, .max_seq_len = 4, .d_ff = 8 };
    var model_a = try TinyWordTransformer.init(alloc, cfg, &rng);
    defer model_a.deinit();

    const path = tmpCheckpointPath("roundtrip");
    try model_a.save(io, path);
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    var rng2 = Rng.init(99); // different init — weights should be overwritten.
    var model_b = try TinyWordTransformer.init(alloc, cfg, &rng2);
    defer model_b.deinit();
    try model_b.load(io, path);

    // Compare token-embedding weights — they came from rng(7) on save
    // and rng(99) on init, so if load worked, they are now equal.
    for (model_a.tok_embed.weight.data, model_b.tok_embed.weight.data) |x, y| {
        try std.testing.expectEqual(x, y);
    }
}

test "Checkpoint load rejects wrong config" {
    const alloc = std.testing.allocator;
    const io = try testIo();
    std.Io.Dir.cwd().createDirPath(io, "zig-out") catch {};

    var rng = Rng.init(1);
    const saved_cfg = TransformerConfig{ .vocab_size = 8, .d_model = 4, .max_seq_len = 4, .d_ff = 8 };
    var saver = try TinyWordTransformer.init(alloc, saved_cfg, &rng);
    defer saver.deinit();
    const path = tmpCheckpointPath("cfg-mismatch");
    try saver.save(io, path);
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    // Different vocab_size — load must fail.
    var rng2 = Rng.init(2);
    const load_cfg = TransformerConfig{ .vocab_size = 16, .d_model = 4, .max_seq_len = 4, .d_ff = 8 };
    var loader = try TinyWordTransformer.init(alloc, load_cfg, &rng2);
    defer loader.deinit();
    try std.testing.expectError(error.ShapeMismatch, loader.load(io, path));
}

test "Checkpoint load rejects corrupt magic" {
    const alloc = std.testing.allocator;
    const io = try testIo();
    std.Io.Dir.cwd().createDirPath(io, "zig-out") catch {};

    const cwd = std.Io.Dir.cwd();
    const path = tmpCheckpointPath("bad-magic");

    // Write a file that starts with the wrong magic.
    const bad = try cwd.createFile(io, path, .{});
    defer cwd.deleteFile(io, path) catch {};
    defer bad.close(io);
    var wbuf: [64]u8 = undefined;
    var bw = bad.writer(io, &wbuf);
    try bw.interface.writeAll("WRONG");
    try bw.flush();

    var rng = Rng.init(3);
    const cfg = TransformerConfig{ .vocab_size = 8, .d_model = 4, .max_seq_len = 4, .d_ff = 8 };
    var loader = try TinyWordTransformer.init(alloc, cfg, &rng);
    defer loader.deinit();

    try std.testing.expectError(error.IoError, loader.load(io, path));
}

test "Checkpoint load rejects truncated file (missing trailer)" {
    const alloc = std.testing.allocator;
    const io = try testIo();
    std.Io.Dir.cwd().createDirPath(io, "zig-out") catch {};

    var rng = Rng.init(4);
    const cfg = TransformerConfig{ .vocab_size = 8, .d_model = 4, .max_seq_len = 4, .d_ff = 8 };
    var model = try TinyWordTransformer.init(alloc, cfg, &rng);
    defer model.deinit();

    const cwd = std.Io.Dir.cwd();
    const good_path = tmpCheckpointPath("trunc-source");
    try model.save(io, good_path);
    defer cwd.deleteFile(io, good_path) catch {};

    // Read the full file, write back everything EXCEPT the 4-byte trailer.
    const bytes = try cwd.readFileAlloc(io, good_path, alloc, .limited(1 << 20));
    defer alloc.free(bytes);

    const bad_path = tmpCheckpointPath("trunc-bad");
    const bad = try cwd.createFile(io, bad_path, .{});
    defer cwd.deleteFile(io, bad_path) catch {};
    defer bad.close(io);
    var wbuf: [4096]u8 = undefined;
    var bw = bad.writer(io, &wbuf);
    try bw.interface.writeAll(bytes[0 .. bytes.len - 4]);
    try bw.flush();

    var rng2 = Rng.init(5);
    var loader = try TinyWordTransformer.init(alloc, cfg, &rng2);
    defer loader.deinit();

    // Truncated → we successfully read all params but the trailer read
    // hits EOF, which comes back as end-of-stream.
    const result = loader.load(io, bad_path);
    try std.testing.expect(std.meta.isError(result));
}



