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
    /// Stack of transformer blocks. Length = `cfg.n_layer`. Stages 2-7
    /// always had exactly one block; Stage 8 Milestone 3 generalises
    /// to an owned slice. `cfg.n_layer = 1` preserves the single-block
    /// shape bit-for-bit.
    ///
    /// Owned by the model: `init` allocates via the caller's allocator
    /// and `deinit` frees each block plus the slice itself.
    blocks: []TransformerBlock,
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
        if (cfg.n_layer == 0) return error.InvalidArgument;

        var tok_embed = try Embedding.init(allocator, cfg.vocab_size, cfg.d_model, rng);
        errdefer tok_embed.deinit();
        var pos_embed = try Embedding.init(allocator, cfg.max_seq_len, cfg.d_model, rng);
        errdefer pos_embed.deinit();

        // Allocate the block slice. Construct blocks in order, with
        // errdefer logic that matches the "how many were
        // successfully built" counter. This follows the pattern used
        // elsewhere in the project for partial-init rollback.
        const blocks = allocator.alloc(TransformerBlock, cfg.n_layer) catch return error.OutOfMemory;
        errdefer allocator.free(blocks);
        var blocks_built: usize = 0;
        errdefer {
            var i: usize = 0;
            while (i < blocks_built) : (i += 1) blocks[i].deinit();
        }
        while (blocks_built < cfg.n_layer) : (blocks_built += 1) {
            blocks[blocks_built] = try TransformerBlock.init(allocator, cfg, rng);
        }

        var ln_f = try LayerNorm.init(allocator, cfg.d_model, cfg.ln_eps, rng);
        errdefer ln_f.deinit();
        var lm_head = try Linear.init(allocator, cfg.d_model, cfg.vocab_size, false, rng);
        errdefer lm_head.deinit();

        return TinyWordTransformer{
            .tok_embed = tok_embed,
            .pos_embed = pos_embed,
            .blocks = blocks,
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

        // Stack of transformer blocks. Milestone 3: generalised from
        // a single block to a loop over `self.blocks`. We defer
        // deinit of each intermediate with a single variable that we
        // reassign after each block's forward; the defer runs at
        // scope exit, so we track ownership with a captured pointer.
        //
        // Pattern: `x` is the current "live" activation. Each block
        // produces a fresh tensor; we free the previous live one
        // before overwriting, except for the very first `x` which
        // was produced by `ops_elementwise.add` above and is already
        // scope-defer'd via `defer x.deinit(...)` above.
        var block_out = x;
        var block_out_owned = false;
        defer {
            if (block_out_owned) block_out.deinit(self.allocator);
        }
        for (self.blocks) |blk| {
            const next = try blk.forward(block_out, tape);
            if (block_out_owned) block_out.deinit(self.allocator);
            block_out = next;
            block_out_owned = true;
        }

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
        for (self.blocks) |*blk| blk.parameters(list);
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

    /// Save model checkpoint to a binary file (format v3, Stage 8 M6).
    ///
    /// Device-aware: works for both CPU and CUDA parameters. CUDA
    /// tensors are DtoH-copied into per-parameter scratch buffers
    /// during the write (Stage 8 M8-c). No moveToCpu round-trip is
    /// required, so the CUDA model's device residency is preserved
    /// across the save call.
    ///
    /// Format — little-endian throughout, f32 payloads written as raw
    /// IEEE 754 bytes. All padding fields are zeros.
    ///
    /// Header (v3):
    ///     magic        [4]u8  = "ZTLC"
    ///     version      u32    = 3
    ///     model_kind   u32    = 1  (TinyWordTransformer)
    ///     vocab_size   u32
    ///     max_seq_len  u32
    ///     d_model      u32
    ///     d_ff         u32
    ///     bias         u8     (0 or 1)
    ///     _pad         [3]u8  (zeros, aligns next field to 4)
    ///     n_layer      u32    (NEW in v3)
    ///     n_head       u32    (NEW in v3)
    ///     dropout      f32    (NEW in v3; reserved, currently unused)
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
    /// v2 compatibility: `load()` accepts version==2 checkpoints (the
    /// Stage 7 format with a single `block.*` prefix). A v2 file is
    /// loaded as if `n_layer=1, n_head=1, dropout=0.0` and the
    /// `block.*` names are rewritten to `blocks[0].*` on lookup.
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

        // Header (v3)
        try w.writeAll("ZTLC");
        try w.writeInt(u32, 3, .little); // version = 3 (was 2 in Stage 7)
        try w.writeInt(u32, 1, .little); // model_kind = TinyWordTransformer
        try w.writeInt(u32, @intCast(self.cfg.vocab_size), .little);
        try w.writeInt(u32, @intCast(self.cfg.max_seq_len), .little);
        try w.writeInt(u32, @intCast(self.cfg.d_model), .little);
        try w.writeInt(u32, @intCast(self.cfg.d_ff), .little);
        try w.writeInt(u8, @intFromBool(self.cfg.bias), .little);
        try w.writeAll(&.{ 0, 0, 0 }); // 3-byte pad
        // New in v3: n_layer, n_head, dropout. Written as
        // u32/u32/f32 for future headroom beyond the u8 cap in
        // TransformerConfig.
        try w.writeInt(u32, @intCast(self.cfg.n_layer), .little);
        try w.writeInt(u32, @intCast(self.cfg.n_head), .little);
        const dropout_bits: u32 = @bitCast(self.cfg.dropout);
        try w.writeInt(u32, dropout_bits, .little);

        // Collect parameters with names
        var param_list: std.ArrayList(NamedParam) = .empty;
        defer param_list.deinit(self.allocator);
        try self.collectNamedParams(&param_list);
        defer self.freeBlockNames(&param_list);

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
            // Element count in a device-agnostic way. On CPU this is
            // equivalent to `t.data.len`; on CUDA `t.data` is the stub
            // empty slice so we must consult the storage length.
            const n_elems = t.storage.len();
            const data_len: u32 = @intCast(n_elems * 4);
            try w.writeInt(u32, data_len, .little);
            // Raw f32 bytes. Target platforms are little-endian; if we
            // ever port to a big-endian system we would byte-swap here.
            //
            // CUDA payloads are DtoH-copied into a scratch host buffer
            // per-parameter. This keeps the save path simple (one HtoD
            // burst per param) and side-steps having to temporarily
            // `moveToCpu` the whole model. The extra cost is dominated
            // by the write itself; the copy and free both complete
            // before the writer flushes.
            switch (t.storage) {
                .cpu => |s| try w.writeAll(std.mem.sliceAsBytes(s.data)),
                .cuda => |b| {
                    const scratch = try self.allocator.alloc(f32, n_elems);
                    defer self.allocator.free(scratch);
                    try b.copyToHost(scratch);
                    try w.writeAll(std.mem.sliceAsBytes(scratch));
                },
            }
        }

        // Trailer.
        try w.writeAll("END.");
        try writer.flush();
    }

    /// Load model checkpoint from a binary file (format v2, PR-η).
    ///
    /// Device-aware: works for both CPU and CUDA parameters. CUDA
    /// tensors are filled via scratch buffer + HtoD upload (Stage 8
    /// M8-c). No moveToCuda/moveToCpu round-trip required.
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
        // Accept v2 (Stage 7 format, implies n_layer=1, n_head=1,
        // dropout=0.0) and v3 (Stage 8 format with explicit
        // n_layer/n_head/dropout fields). Reject everything else.
        if (version != 2 and version != 3) return error.IoError;
        const is_v3 = version == 3;

        const model_kind = try r.takeInt(u32, .little);
        if (model_kind != 1) return error.IoError;

        const vocab_size = try r.takeInt(u32, .little);
        const max_seq_len = try r.takeInt(u32, .little);
        const d_model = try r.takeInt(u32, .little);
        const d_ff = try r.takeInt(u32, .little);
        const bias_byte = try r.takeInt(u8, .little);
        var pad3: [3]u8 = undefined;
        try r.readSliceAll(&pad3);

        // v3-only header fields. For v2 files we default to the
        // single-block / single-head / no-dropout shape so the
        // load matches the shape the file implicitly assumed.
        const file_n_layer: u32 = if (is_v3) try r.takeInt(u32, .little) else 1;
        const file_n_head: u32 = if (is_v3) try r.takeInt(u32, .little) else 1;
        const file_dropout: f32 = if (is_v3) blk: {
            const bits = try r.takeInt(u32, .little);
            break :blk @bitCast(bits);
        } else 0.0;

        if (vocab_size != self.cfg.vocab_size or
            max_seq_len != self.cfg.max_seq_len or
            d_model != self.cfg.d_model or
            d_ff != self.cfg.d_ff or
            (bias_byte != 0) != self.cfg.bias)
        {
            return error.ShapeMismatch;
        }

        // Stage 8 M6: if the checkpoint explicitly stored
        // n_layer / n_head, validate they match ours. Dropout is
        // informational (reserved field) so we accept any value.
        // For v2 files the defaults above (1/1/0.0) serve as the
        // implicit check: they must match `self.cfg` or the
        // param_count check below will fail first.
        if (file_n_layer != self.cfg.n_layer) return error.ShapeMismatch;
        if (file_n_head != self.cfg.n_head) return error.ShapeMismatch;
        _ = file_dropout; // reserved; accept any value

        const param_count = try r.takeInt(u32, .little);

        // Collect expected params.
        var param_list: std.ArrayList(NamedParam) = .empty;
        defer param_list.deinit(self.allocator);
        try self.collectNamedParams(&param_list);
        defer self.freeBlockNames(&param_list);

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
            //
            // v2 backward compatibility: Stage 7 checkpoints (version 2,
            // written before Milestone 3) use the single-block name
            // prefix `block.*`. After Milestone 3 every block is
            // indexed, so our `collectNamedParams` emits `blocks[0].*`
            // for a single-block model. Rewrite incoming v2 names on
            // the fly so `shakespeare_ckpt.bin` (created Stage 6)
            // continues to load. v3 files already use the canonical
            // `blocks[<i>].*` names so no rewrite is needed.
            var matched_index: ?usize = null;
            var rewritten_buf: [300]u8 = undefined;
            const lookup_name: []const u8 = if (!is_v3 and self.cfg.n_layer == 1 and std.mem.startsWith(u8, name, "block.")) blk: {
                const suffix = name["block.".len..];
                break :blk std.fmt.bufPrint(&rewritten_buf, "blocks[0].{s}", .{suffix}) catch return error.IoError;
            } else name;
            for (param_list.items, 0..) |entry, idx| {
                if (std.mem.eql(u8, lookup_name, entry.name)) {
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
            // Device-agnostic size check. CUDA tensors have `t.data.len
            // == 0`; the true element count lives in `t.storage.len()`.
            const n_elems = t.storage.len();
            const expected_bytes: u32 = @intCast(n_elems * 4);
            if (data_len != expected_bytes) return error.ShapeMismatch;

            // Read payload. On CPU: straight into the parameter's
            // data slice. On CUDA: read into a scratch host buffer
            // then HtoD-upload, symmetric with the save path.
            switch (t.storage) {
                .cpu => |s| try r.readSliceAll(std.mem.sliceAsBytes(s.data)),
                .cuda => |b| {
                    const scratch = try self.allocator.alloc(f32, n_elems);
                    defer self.allocator.free(scratch);
                    try r.readSliceAll(std.mem.sliceAsBytes(scratch));
                    try b.copyFromHost(scratch);
                },
            }
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
    ///
    /// For multi-block models (`cfg.n_layer > 1`), each block contributes
    /// 10 parameters named `blocks[<i>].*`. The per-block prefix strings
    /// are allocated via `self.allocator` and their lifetime is tied to
    /// the list: callers are responsible for `list.deinit(self.allocator)`
    /// AND for freeing each name slice via `self.allocator.free(entry.name)`
    /// when the name was allocated (allocated names always start with the
    /// "blocks[" prefix; fixed names like "tok_embed.weight" are static
    /// literals and MUST NOT be freed).
    ///
    /// The `save` / `load` helpers below handle this discipline internally.
    pub fn collectNamedParams(
        self: *TinyWordTransformer,
        list: *std.ArrayList(NamedParam),
    ) !void {
        try list.append(self.allocator, .{ .name = "tok_embed.weight", .tensor = &self.tok_embed.weight });
        try list.append(self.allocator, .{ .name = "pos_embed.weight", .tensor = &self.pos_embed.weight });

        // Per-block names: 10 parameters each, prefixed with
        // "blocks[<i>]." so a 2-block model has 20 block params, a
        // 6-block model has 60, etc. Stage 7 (single-block) used
        // `block.*` names; the v2 loader below rewrites those on
        // the fly when `cfg.n_layer == 1` for backward compatibility.
        for (self.blocks, 0..) |*blk, i| {
            try self.appendBlockNames(list, i, blk);
        }

        try list.append(self.allocator, .{ .name = "ln_f.gamma", .tensor = &self.ln_f.gamma });
        try list.append(self.allocator, .{ .name = "ln_f.beta", .tensor = &self.ln_f.beta });
        try list.append(self.allocator, .{ .name = "lm_head.weight", .tensor = &self.lm_head.weight });
    }

    /// Internal: emit the 10 `blocks[<i>].<field>` entries for one
    /// block. Each name is allocated via `self.allocator` and must be
    /// freed by the caller of `collectNamedParams` when done with the
    /// list. `freeBlockNames` below does the cleanup in one pass.
    fn appendBlockNames(
        self: *TinyWordTransformer,
        list: *std.ArrayList(NamedParam),
        i: usize,
        blk: *TransformerBlock,
    ) !void {
        const alloc = self.allocator;
        const fields = [_]struct { suffix: []const u8, tensor: *Tensor }{
            .{ .suffix = "ln1.gamma", .tensor = &blk.ln1.gamma },
            .{ .suffix = "ln1.beta", .tensor = &blk.ln1.beta },
            .{ .suffix = "attn.w_q.weight", .tensor = &blk.attn.w_q.weight },
            .{ .suffix = "attn.w_k.weight", .tensor = &blk.attn.w_k.weight },
            .{ .suffix = "attn.w_v.weight", .tensor = &blk.attn.w_v.weight },
            .{ .suffix = "attn.w_o.weight", .tensor = &blk.attn.w_o.weight },
            .{ .suffix = "ln2.gamma", .tensor = &blk.ln2.gamma },
            .{ .suffix = "ln2.beta", .tensor = &blk.ln2.beta },
            .{ .suffix = "mlp.fc1.weight", .tensor = &blk.mlp.fc1.weight },
            .{ .suffix = "mlp.fc2.weight", .tensor = &blk.mlp.fc2.weight },
        };
        for (fields) |f| {
            const name = try std.fmt.allocPrint(alloc, "blocks[{d}].{s}", .{ i, f.suffix });
            try list.append(alloc, .{ .name = name, .tensor = f.tensor });
        }
    }

    /// Free all dynamically-allocated names in a `NamedParam` list.
    /// Safe to call with lists that mix allocated and literal names
    /// (identified by the "blocks[" prefix).
    pub fn freeBlockNames(self: *TinyWordTransformer, list: *const std.ArrayList(NamedParam)) void {
        const alloc = self.allocator;
        for (list.items) |entry| {
            if (std.mem.startsWith(u8, entry.name, "blocks[")) {
                alloc.free(entry.name);
            }
        }
    }

    /// Free all sub-layers and embeddings.
    pub fn deinit(self: *TinyWordTransformer) void {
        self.tok_embed.deinit();
        self.pos_embed.deinit();
        for (self.blocks) |*blk| blk.deinit();
        self.allocator.free(self.blocks);
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

test "TinyWordTransformer init — default n_layer=1 has single block" {
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
    try std.testing.expectEqual(@as(usize, 1), model.blocks.len);
}

test "TinyWordTransformer init — n_layer=3 allocates 3 blocks" {
    const alloc = std.testing.allocator;
    var rng = Rng.init(42);
    const cfg = TransformerConfig{
        .vocab_size = 16,
        .d_model = 8,
        .max_seq_len = 4,
        .d_ff = 32,
        .n_layer = 3,
    };

    var model = try TinyWordTransformer.init(alloc, cfg, &rng);
    defer model.deinit();
    try std.testing.expectEqual(@as(usize, 3), model.blocks.len);

    // Each block's attention has its own causal mask; verify they
    // are independent allocations.
    const m0 = model.blocks[0].attn.causal_mask.data.ptr;
    const m1 = model.blocks[1].attn.causal_mask.data.ptr;
    const m2 = model.blocks[2].attn.causal_mask.data.ptr;
    try std.testing.expect(m0 != m1);
    try std.testing.expect(m1 != m2);
    try std.testing.expect(m0 != m2);
}

test "TinyWordTransformer init — n_layer=0 returns InvalidArgument" {
    const alloc = std.testing.allocator;
    var rng = Rng.init(42);
    const cfg = TransformerConfig{
        .vocab_size = 16,
        .d_model = 8,
        .max_seq_len = 4,
        .d_ff = 32,
        .n_layer = 0,
    };
    try std.testing.expectError(error.InvalidArgument, TinyWordTransformer.init(alloc, cfg, &rng));
}

test "TinyWordTransformer forward — 2-block model produces finite logits" {
    const alloc = std.testing.allocator;
    var rng = Rng.init(42);
    const cfg = TransformerConfig{
        .vocab_size = 16,
        .d_model = 8,
        .max_seq_len = 4,
        .d_ff = 32,
        .n_layer = 2,
    };

    var model = try TinyWordTransformer.init(alloc, cfg, &rng);
    defer model.deinit();

    var ids = try Tensor.init(alloc, Shape.init2D(2, 3));
    defer ids.deinit(alloc);
    for (0..6) |i| ids.data[i] = @floatFromInt(i % 16);

    var logits = try model.forward(ids, null);
    defer logits.deinit(alloc);

    try std.testing.expectEqual(@as(usize, 2), logits.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 3), logits.shape.dims[1]);
    try std.testing.expectEqual(@as(usize, 16), logits.shape.dims[2]);

    for (0..logits.data.len) |i| {
        try std.testing.expect(std.math.isFinite(logits.data[i]));
    }
}

test "TinyWordTransformer parameters — 2-block list has twice as many block-scoped entries" {
    const alloc = std.testing.allocator;
    var rng = Rng.init(42);
    const cfg = TransformerConfig{
        .vocab_size = 16,
        .d_model = 8,
        .max_seq_len = 4,
        .d_ff = 32,
    };

    var model1 = try TinyWordTransformer.init(alloc, cfg, &rng);
    defer model1.deinit();
    var p1: std.ArrayList(*Tensor) = .empty;
    defer p1.deinit(alloc);
    try p1.ensureTotalCapacity(alloc, 64);
    model1.parameters(&p1);
    const count1 = p1.items.len;

    const cfg2 = TransformerConfig{
        .vocab_size = 16,
        .d_model = 8,
        .max_seq_len = 4,
        .d_ff = 32,
        .n_layer = 2,
    };
    var rng2 = Rng.init(42);
    var model2 = try TinyWordTransformer.init(alloc, cfg2, &rng2);
    defer model2.deinit();
    var p2: std.ArrayList(*Tensor) = .empty;
    defer p2.deinit(alloc);
    try p2.ensureTotalCapacity(alloc, 64);
    model2.parameters(&p2);
    const count2 = p2.items.len;

    // Each extra block adds exactly 10 learnable params:
    //   LN1 (gamma, beta) + 4 attn Linears (w+b = 8) + LN2 (2) +
    //   MLP fc1 (w+b = 2) + MLP fc2 (w+b = 2) = 16 params per block.
    //   Wait — Linear uses bias=true by default here so each Linear
    //   contributes 2 params. 4 attn linears = 8, 2 mlp linears = 4,
    //   2 LNs = 4. Total per block = 16. Test verifies delta matches.
    try std.testing.expectEqual(count1 + 16, count2);
}

test "TinyWordTransformer collectNamedParams — 2-block names include blocks[0] and blocks[1]" {
    const alloc = std.testing.allocator;
    var rng = Rng.init(42);
    const cfg = TransformerConfig{
        .vocab_size = 16,
        .d_model = 8,
        .max_seq_len = 4,
        .d_ff = 32,
        .n_layer = 2,
    };

    var model = try TinyWordTransformer.init(alloc, cfg, &rng);
    defer model.deinit();

    var named: std.ArrayList(NamedParam) = .empty;
    defer named.deinit(alloc);
    try model.collectNamedParams(&named);
    defer model.freeBlockNames(&named);

    var has_blocks0 = false;
    var has_blocks1 = false;
    var has_tok_embed = false;
    for (named.items) |entry| {
        if (std.mem.startsWith(u8, entry.name, "blocks[0].")) has_blocks0 = true;
        if (std.mem.startsWith(u8, entry.name, "blocks[1].")) has_blocks1 = true;
        if (std.mem.eql(u8, entry.name, "tok_embed.weight")) has_tok_embed = true;
    }
    try std.testing.expect(has_blocks0);
    try std.testing.expect(has_blocks1);
    try std.testing.expect(has_tok_embed);
}

test "Checkpoint v3 save/load round-trip preserves n_layer/n_head/dropout" {
    const alloc = std.testing.allocator;
    const io = try testIo();
    std.Io.Dir.cwd().createDirPath(io, "zig-out") catch {};

    var rng = Rng.init(17);
    const cfg = TransformerConfig{
        .vocab_size = 8,
        .d_model = 4,
        .max_seq_len = 4,
        .d_ff = 8,
        .n_layer = 2,
        .n_head = 2,
        .dropout = 0.25, // stored, loads back identical (reserved)
    };
    var model_a = try TinyWordTransformer.init(alloc, cfg, &rng);
    defer model_a.deinit();

    const path = tmpCheckpointPath("v3-roundtrip");
    try model_a.save(io, path);
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    var rng2 = Rng.init(99);
    var model_b = try TinyWordTransformer.init(alloc, cfg, &rng2);
    defer model_b.deinit();
    try model_b.load(io, path);

    // Every parameter in the 2-block model must match after load.
    // Compare embedding + first attention weight as representatives.
    for (model_a.tok_embed.weight.data, model_b.tok_embed.weight.data) |x, y| {
        try std.testing.expectEqual(x, y);
    }
    for (model_a.blocks[0].attn.w_q.weight.data, model_b.blocks[0].attn.w_q.weight.data) |x, y| {
        try std.testing.expectEqual(x, y);
    }
    for (model_a.blocks[1].attn.w_q.weight.data, model_b.blocks[1].attn.w_q.weight.data) |x, y| {
        try std.testing.expectEqual(x, y);
    }
}

test "Checkpoint v3 load rejects n_layer mismatch" {
    const alloc = std.testing.allocator;
    const io = try testIo();
    std.Io.Dir.cwd().createDirPath(io, "zig-out") catch {};

    var rng = Rng.init(1);
    const saved_cfg = TransformerConfig{
        .vocab_size = 8,
        .d_model = 4,
        .max_seq_len = 4,
        .d_ff = 8,
        .n_layer = 2,
    };
    var saver = try TinyWordTransformer.init(alloc, saved_cfg, &rng);
    defer saver.deinit();
    const path = tmpCheckpointPath("v3-n-layer-mismatch");
    try saver.save(io, path);
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    var rng2 = Rng.init(2);
    const load_cfg = TransformerConfig{
        .vocab_size = 8,
        .d_model = 4,
        .max_seq_len = 4,
        .d_ff = 8,
        .n_layer = 1, // different!
    };
    var loader = try TinyWordTransformer.init(alloc, load_cfg, &rng2);
    defer loader.deinit();
    try std.testing.expectError(error.ShapeMismatch, loader.load(io, path));
}

test "Checkpoint v2 backward compat — single-block load via name rewrite" {
    const alloc = std.testing.allocator;
    const io = try testIo();
    std.Io.Dir.cwd().createDirPath(io, "zig-out") catch {};

    // Construct a synthetic v2 checkpoint in-memory: same header
    // layout as Stage 7 save(), same "block.*" param naming. Write
    // to a temp file, then load with the v3 loader and verify the
    // name-rewrite path kicks in.
    //
    // We piggyback on the existing save() by first saving a v3
    // file, then patching bytes [4..8] from 3 to 2 and rewriting
    // every "blocks[0]." name prefix to "block." in the on-disk
    // record name strings. The v3 file has 3 extra u32/u32/f32
    // (12 bytes) right after the bias pad; we also need to delete
    // those from the v2 shape.
    //
    // Simpler path: hand-write the full v2 byte stream for a 1-
    // parameter model. Keep it minimal to avoid re-deriving the
    // full record format: we emit ONE parameter (lm_head.weight)
    // which avoids the per-block naming altogether.

    const cwd = std.Io.Dir.cwd();
    const path = tmpCheckpointPath("v2-synthetic");
    defer cwd.deleteFile(io, path) catch {};

    // Compute shape for lm_head.weight in our target cfg.
    var rng = Rng.init(0);
    const cfg = TransformerConfig{
        .vocab_size = 8,
        .d_model = 4,
        .max_seq_len = 4,
        .d_ff = 8,
        .bias = true,
        // Defaults: n_layer=1, n_head=1, dropout=0
    };

    // Build model to get lm_head.weight shape (D, V)-shape = (4, 8)
    // wait: our Linear weight is (d_out, d_in) = (V, D) for lm_head
    // with d_in=D, d_out=V. So shape = (8, 4).
    var model = try TinyWordTransformer.init(alloc, cfg, &rng);
    defer model.deinit();

    const lm_w = &model.lm_head.weight;
    const dim0: u32 = @intCast(lm_w.shape.dims[0]);
    const dim1: u32 = @intCast(lm_w.shape.dims[1]);
    // Seed the weight with known values so the load can verify
    // round-trip semantics.
    for (lm_w.data, 0..) |*v, i| v.* = @floatFromInt(i);

    // To exercise the v2 compat path we need the checkpoint to
    // include a parameter named `block.*`. But our n_layer=1 model's
    // param list AFTER M3 has "blocks[0].*" -- so emitting a param
    // named "block.attn.w_q.weight" is the right fixture. We write
    // just that one param to a minimal v2 file and check that load
    // rewrites the name to "blocks[0].attn.w_q.weight" and matches
    // the tensor on the loader side.
    //
    // For the loader to pass, the saved param_count must equal the
    // number of params our model expects (16). Hand-writing 16
    // param records is verbose but straightforward -- we'll just
    // re-use the name rewrite: write all 16 names with the `block.*`
    // prefix (substituting `blocks[0]` -> `block`), with the tensor
    // data from a freshly-init'd model.
    //
    // To keep the test focused, we use the existing save() after
    // temporarily patching the "blocks[0]." prefix out of the
    // record names. The simplest way is: save() with a custom
    // path, then post-process the file bytes to (a) set version=2
    // (b) rewrite names (c) strip the n_layer/n_head/dropout 12 bytes.
    try model.save(io, path);

    // Read whole file into memory.
    const bytes_read = cwd.readFileAlloc(io, path, alloc, .limited(1 * 1024 * 1024)) catch {
        return error.IoError;
    };
    defer alloc.free(bytes_read);

    // The v3 header is 32 bytes base + 12 new bytes =
    //   [0..4)  "ZTLC"
    //   [4..8)  u32 version = 3
    //   [8..12) model_kind
    //   [12..28) vocab/seq/d/ff (16 bytes)
    //   [28..29) bias
    //   [29..32) pad
    //   [32..36) n_layer  <-- NEW
    //   [36..40) n_head   <-- NEW
    //   [40..44) dropout  <-- NEW
    //   [44..48) param_count
    //   [48..)   records + "END."
    //
    // v2 layout omits bytes [32..44) (12 bytes) and the magic
    // version byte is 2. Build a new buffer with those edits.
    var v2_buf: std.ArrayList(u8) = .empty;
    defer v2_buf.deinit(alloc);
    // Header up to bias+pad (first 32 bytes).
    try v2_buf.appendSlice(alloc, bytes_read[0..32]);
    // Patch version: bytes [4..8) -> 2 little-endian.
    v2_buf.items[4] = 2;
    v2_buf.items[5] = 0;
    v2_buf.items[6] = 0;
    v2_buf.items[7] = 0;
    // Skip the 12 bytes of v3 extras, then append param_count
    // through end-of-file (with name rewrites).
    const v3_body = bytes_read[44..]; // after the 3 extra fields
    // Now rewrite names in the record stream. Param records use
    // a u32 name_len + [name_len]u8 name layout; each "blocks[0]."
    // prefix in a name string gets rewritten to "block.".
    // We scan: u32 param_count, then for each param read name_len
    // and name, then rank+pad(4)+dims(16)+data_len(4)+data.
    const pc_bytes = v3_body[0..4];
    try v2_buf.appendSlice(alloc, pc_bytes);
    const pc: u32 = std.mem.readInt(u32, v3_body[0..4], .little);
    var cursor: usize = 4;
    var p_i: u32 = 0;
    while (p_i < pc) : (p_i += 1) {
        const nlen = std.mem.readInt(u32, v3_body[cursor..][0..4], .little);
        const name = v3_body[cursor + 4 .. cursor + 4 + nlen];
        // Rewrite blocks[0]. -> block.
        if (std.mem.startsWith(u8, name, "blocks[0].")) {
            const suffix = name["blocks[0].".len..];
            const new_name_len: u32 = @intCast("block.".len + suffix.len);
            // Append new name_len as 4 LE bytes.
            var nlen_bytes: [4]u8 = undefined;
            std.mem.writeInt(u32, &nlen_bytes, new_name_len, .little);
            try v2_buf.appendSlice(alloc, &nlen_bytes);
            try v2_buf.appendSlice(alloc, "block.");
            try v2_buf.appendSlice(alloc, suffix);
        } else {
            try v2_buf.appendSlice(alloc, v3_body[cursor..][0 .. 4 + nlen]);
        }
        cursor += 4 + nlen;
        // rank(1) + pad(3) + dims(16) + data_len(4)
        try v2_buf.appendSlice(alloc, v3_body[cursor..][0..24]);
        const data_len = std.mem.readInt(u32, v3_body[cursor + 20 ..][0..4], .little);
        cursor += 24;
        try v2_buf.appendSlice(alloc, v3_body[cursor..][0..data_len]);
        cursor += data_len;
    }
    // End magic.
    try v2_buf.appendSlice(alloc, v3_body[cursor..]);

    // Write the synthetic v2 buffer back to disk (overwrite).
    {
        const f = try cwd.createFile(io, path, .{});
        defer f.close(io);
        var wbuf: [4096]u8 = undefined;
        var w = f.writer(io, &wbuf);
        try w.interface.writeAll(v2_buf.items);
        try w.flush();
    }

    // Now load the v2 file into a fresh model. If the v2 rewrite
    // path works, this succeeds and the lm_head weight matches
    // what we seeded above.
    var rng2 = Rng.init(99);
    var loader = try TinyWordTransformer.init(alloc, cfg, &rng2);
    defer loader.deinit();
    try loader.load(io, path);

    for (loader.lm_head.weight.data, 0..) |v, i| {
        const expected: f32 = @floatFromInt(i);
        try std.testing.expectEqual(expected, v);
    }
    _ = dim0;
    _ = dim1;
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



