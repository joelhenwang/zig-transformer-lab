//!
//! zig-transformer-lab — Byte-level BPE Tokenizer
//!
//! Purpose:
//!   Implements Byte Pair Encoding for subword tokenization. Starts with
//!   a base vocabulary of 256 byte values, then iteratively merges the
//!   most frequent adjacent pair to build a larger vocabulary.
//!
//!   This is the tokenization algorithm used by GPT-2, GPT-3, GPT-4,
//!   Llama, and most modern LLMs.
//!
//! Algorithm:
//!   Training:
//!     1. Initialize vocab with bytes 0-255
//!     2. Count all adjacent token pairs in corpus
//!     3. Merge most frequent pair -> new token
//!     4. Repeat until vocab_size reached
//!
//!   Encoding:
//!     1. Convert text to bytes
//!     2. Apply merges in priority order (earliest learned = highest priority)
//!
//!   Decoding:
//!     1. Map each token ID to its byte sequence
//!     2. Concatenate and decode as UTF-8
//!
//! Credits:
//!   Sennrich et al. (2016), "Neural Machine Translation of Rare Words
//!   with Subword Units." Karpathy's "minbpe" for reference. No code copied.

const std = @import("std");
const LabError = @import("../core/errors.zig").LabError;

/// A merge rule: pair (a, b) -> merged token c.
pub const Merge = struct {
    a: u32,
    b: u32,
    result: u32,
};

/// Byte-level BPE tokenizer.
pub const BpeTokenizer = struct {
    /// Ordered list of merge rules (index = priority, 0 = highest).
    merges: std.ArrayList(Merge),
    /// Vocab: maps token_id -> byte sequence.
    vocab: std.ArrayList([]const u8),
    /// Total vocabulary size (256 base + number of merges).
    vocab_size: u32,
    allocator: std.mem.Allocator,

    /// Train a BPE tokenizer on a corpus.
    ///
    /// Starting from 256 byte tokens, learns `num_merges` merge rules
    /// by iteratively finding the most frequent adjacent pair.
    ///
    /// Worked example:
    ///   var tok = try BpeTokenizer.train(alloc, "hello hello", 5);
    ///   defer tok.deinit();
    ///   // tok.vocab_size == 261 (256 bytes + 5 merges)
    pub fn train(allocator: std.mem.Allocator, corpus: []const u8, num_merges: u32) LabError!BpeTokenizer {
        var merges: std.ArrayList(Merge) = .empty;
        errdefer merges.deinit(allocator);

        var vocab: std.ArrayList([]const u8) = .empty;
        errdefer {
            for (vocab.items) |v| allocator.free(v);
            vocab.deinit(allocator);
        }

        // Initialize vocab with single bytes
        try vocab.ensureTotalCapacity(allocator, 256 + num_merges);
        for (0..256) |byte_val| {
            const slice = try allocator.alloc(u8, 1);
            slice[0] = @intCast(byte_val);
            try vocab.append(allocator, slice);
        }

        // Convert corpus to token IDs (initially just bytes)
        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(allocator);
        try ids.ensureTotalCapacity(allocator, corpus.len);
        for (corpus) |byte| {
            try ids.append(allocator, @intCast(byte));
        }

        // Iteratively merge the most frequent pair
        try merges.ensureTotalCapacity(allocator, num_merges);
        for (0..num_merges) |_| {
            if (ids.items.len < 2) break;

            // Count pair frequencies
            var pair_counts: std.AutoHashMap(u64, u32) = .init(allocator);
            defer pair_counts.deinit();

            for (0..ids.items.len - 1) |i| {
                const key = packPair(ids.items[i], ids.items[i + 1]);
                const entry = try pair_counts.getOrPut(key);
                if (!entry.found_existing) {
                    entry.value_ptr.* = 0;
                }
                entry.value_ptr.* += 1;
            }

            // Find most frequent pair
            var best_pair: u64 = 0;
            var best_count: u32 = 0;
            var iter = pair_counts.iterator();
            while (iter.next()) |entry| {
                if (entry.value_ptr.* > best_count) {
                    best_count = entry.value_ptr.*;
                    best_pair = entry.key_ptr.*;
                }
            }

            if (best_count < 2) break; // No pair appears more than once

            const a = @as(u32, @intCast(best_pair >> 32));
            const b = @as(u32, @intCast(best_pair & 0xFFFFFFFF));
            const new_id: u32 = @intCast(vocab.items.len);

            // Create the merged token's byte sequence
            const a_bytes = vocab.items[a];
            const b_bytes = vocab.items[b];
            const merged = try allocator.alloc(u8, a_bytes.len + b_bytes.len);
            @memcpy(merged[0..a_bytes.len], a_bytes);
            @memcpy(merged[a_bytes.len..], b_bytes);
            try vocab.append(allocator, merged);

            // Record the merge
            try merges.append(allocator, Merge{ .a = a, .b = b, .result = new_id });

            // Apply merge to the corpus IDs
            var new_ids: std.ArrayList(u32) = .empty;
            defer new_ids.deinit(allocator);
            try new_ids.ensureTotalCapacity(allocator, ids.items.len);

            var i: usize = 0;
            while (i < ids.items.len) {
                if (i + 1 < ids.items.len and ids.items[i] == a and ids.items[i + 1] == b) {
                    try new_ids.append(allocator, new_id);
                    i += 2;
                } else {
                    try new_ids.append(allocator, ids.items[i]);
                    i += 1;
                }
            }

            // Swap
            ids.clearRetainingCapacity();
            try ids.appendSlice(allocator, new_ids.items);
        }

        return BpeTokenizer{
            .merges = merges,
            .vocab = vocab,
            .vocab_size = @intCast(vocab.items.len),
            .allocator = allocator,
        };
    }

    /// Encode text to token IDs using the learned merges.
    pub fn encode(self: *const BpeTokenizer, allocator: std.mem.Allocator, text: []const u8) LabError![]u32 {
        if (text.len == 0) {
            return allocator.alloc(u32, 0) catch return error.OutOfMemory;
        }

        // Start with byte-level IDs
        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(allocator);
        ids.ensureTotalCapacity(allocator, text.len) catch return error.OutOfMemory;
        for (text) |byte| {
            ids.append(allocator, @intCast(byte)) catch return error.OutOfMemory;
        }

        // Apply merges in priority order
        for (self.merges.items) |merge| {
            var i: usize = 0;
            while (i + 1 < ids.items.len) {
                if (ids.items[i] == merge.a and ids.items[i + 1] == merge.b) {
                    ids.items[i] = merge.result;
                    // Remove the next element by shifting
                    var j: usize = i + 1;
                    while (j + 1 < ids.items.len) : (j += 1) {
                        ids.items[j] = ids.items[j + 1];
                    }
                    ids.items.len -= 1;
                    // Don't advance i — check if the new token merges with next
                } else {
                    i += 1;
                }
            }
        }

        // Return owned slice
        const result = allocator.alloc(u32, ids.items.len) catch return error.OutOfMemory;
        @memcpy(result, ids.items);
        return result;
    }

    /// Decode token IDs back to text.
    pub fn decode(self: *const BpeTokenizer, allocator: std.mem.Allocator, ids: []const u32) LabError![]u8 {
        // Calculate total byte length
        var total_len: usize = 0;
        for (ids) |id| {
            if (id >= self.vocab.items.len) return error.InvalidArgument;
            total_len += self.vocab.items[id].len;
        }

        const result = allocator.alloc(u8, total_len) catch return error.OutOfMemory;
        var offset: usize = 0;
        for (ids) |id| {
            const bytes = self.vocab.items[id];
            @memcpy(result[offset..][0..bytes.len], bytes);
            offset += bytes.len;
        }
        return result;
    }

    pub fn deinit(self: *BpeTokenizer) void {
        for (self.vocab.items) |v| self.allocator.free(v);
        self.vocab.deinit(self.allocator);
        self.merges.deinit(self.allocator);
    }

    /// Pack a (u32, u32) pair into a u64 key for hashing.
    fn packPair(a: u32, b: u32) u64 {
        return (@as(u64, a) << 32) | @as(u64, b);
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "BPE train produces correct vocab size" {
    const alloc = std.testing.allocator;
    var tok = try BpeTokenizer.train(alloc, "aaabbb", 3);
    defer tok.deinit();
    // 256 base + up to 3 merges
    try std.testing.expect(tok.vocab_size > 256);
    try std.testing.expect(tok.vocab_size <= 259);
}

test "BPE encode-decode roundtrip" {
    const alloc = std.testing.allocator;
    const corpus = "hello world hello world";
    var tok = try BpeTokenizer.train(alloc, corpus, 10);
    defer tok.deinit();

    const encoded = try tok.encode(alloc, "hello");
    defer alloc.free(encoded);

    const decoded = try tok.decode(alloc, encoded);
    defer alloc.free(decoded);

    try std.testing.expectEqualStrings("hello", decoded);
}

test "BPE encoding produces fewer tokens than bytes" {
    const alloc = std.testing.allocator;
    const corpus = "the cat sat on the mat the cat sat on the mat";
    var tok = try BpeTokenizer.train(alloc, corpus, 20);
    defer tok.deinit();

    const encoded = try tok.encode(alloc, "the cat");
    defer alloc.free(encoded);

    // "the cat" is 7 bytes. With merges, should be fewer tokens.
    try std.testing.expect(encoded.len < 7);
}

test "BPE empty string" {
    const alloc = std.testing.allocator;
    var tok = try BpeTokenizer.train(alloc, "abc", 2);
    defer tok.deinit();

    const encoded = try tok.encode(alloc, "");
    defer alloc.free(encoded);
    try std.testing.expectEqual(@as(usize, 0), encoded.len);
}
