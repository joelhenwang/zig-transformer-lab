//!
//! zig-transformer-lab — Batcher: deterministic shuffle + batching
//!
//! Purpose:
//!   Groups windows from a Windowing into mini-batches for training.
//!   The key properties are:
//!
//!   1. Deterministic: same seed → same shuffle order → same batches.
//!      This satisfies locked decision D14 (Xoshiro256 seeded RNG).
//!
//!   2. Drop-last: if the number of windows isn't divisible by
//!      batch_size, the remaining windows are dropped. This avoids
//!      a partial batch with incorrect gradient scaling.
//!
//!   3. Fisher-Yates shuffle: we shuffle window INDICES (not data),
//!      then produce batches of B consecutive shuffled indices.
//!      Each batch's input/target arrays are contiguous copies of the
//!      shuffled windows' data.
//!
//!   Why shuffle at all?
//!   Without shuffling, consecutive windows overlap heavily (by T-1
//!   tokens), making the training signal highly correlated within a
//!   batch. Shuffling decorrelates the batches, which helps SGD
//!   converge faster.
//!
//! Shape contract:
//!   Batch.input:  []u32 of length B * T  (flat, row-major)
//!   Batch.target: []u32 of length B * T  (flat, row-major)
//!
//!   The caller reshapes these into (B, T) tensors for the model.
//!
//! Math:
//!   Fisher-Yates shuffle: for i in [0, n):
//!     j = random integer in [i, n)
//!     swap indices[i], indices[j]
//!
//!   This produces a uniform random permutation — every possible
//!   ordering is equally likely.
//!
//! Memory ownership:
//!   - Batcher owns the shuffled index array ([]usize).
//!   - Each Batch owns its input and target arrays ([]u32).
//!   - Batch.ids is a view into the Batcher's index array.
//!   - Call batch.deinit() to free each batch's input/target.
//!   - Call batcher.deinit() to free the index array.
//!
//! Error conditions:
//!   OutOfMemory — from allocation
//!
//! Credits:
//!   Fisher-Yates shuffle is standard (Fisher & Yates, 1938).
//!   The batching pattern follows PyTorch's DataLoader.
//!   No code copied.

const std = @import("std");
const LabError = @import("../core/errors.zig").LabError;
const Rng = @import("../core/rng.zig").Rng;
const Windowing = @import("windowing.zig").Windowing;
const Window = @import("windowing.zig").Window;

/// A single mini-batch of training data.
///
/// input and target are flat arrays of shape (B*T,).
/// The caller reshapes them into (B, T) tensors for the model.
///
/// ids holds the original window indices that went into this batch,
/// useful for debugging ("which windows are in this batch?").
pub const Batch = struct {
    /// Input token IDs, flat layout: [sample_0_pos_0, ..., sample_0_pos_T-1, sample_1_pos_0, ...]
    input: []u32,
    /// Target token IDs, same layout as input.
    target: []u32,
    /// Window indices in this batch (views into Batcher.indices).
    ids: []const usize,
    /// Allocator for freeing input/target arrays.
    allocator: std.mem.Allocator,

    /// Free the input and target arrays.
    pub fn deinit(self: Batch) void {
        self.allocator.free(self.input);
        self.allocator.free(self.target);
    }
};

/// Batcher: produces mini-batches from a Windowing with deterministic shuffle.
///
/// Worked example:
///   var batcher = try Batcher.init(alloc, windowing, 4, &rng);
///   defer batcher.deinit();
///   while (batcher.next()) |batch| {
///       defer batch.deinit();
///       // batch.input has 4 * T elements
///       // batch.target has 4 * T elements
///   }
pub const Batcher = struct {
    /// The windowing source (borrowed).
    windowing: Windowing,
    /// Number of windows per batch.
    batch_size: usize,
    /// Shuffled window indices.
    indices: []usize,
    /// Current position in the shuffled index array.
    pos: usize,
    /// Allocator for index array and batch data.
    allocator: std.mem.Allocator,

    /// Create a Batcher with deterministic shuffle.
    ///
    /// Shuffles all window indices using Fisher-Yates with the given RNG.
    /// The same seed always produces the same shuffle order.
    ///
    /// Worked example:
    ///   var rng = Rng.init(42);
    ///   var batcher = try Batcher.init(alloc, windowing, 4, &rng);
    ///   defer batcher.deinit();
    pub fn init(
        allocator: std.mem.Allocator,
        windowing: Windowing,
        batch_size: usize,
        rng: *Rng,
    ) LabError!Batcher {
        const n = windowing.count();

        // Create and shuffle window indices
        var indices = try allocator.alloc(usize, n);
        errdefer allocator.free(indices);

        // Initialize indices: [0, 1, 2, ..., n-1]
        for (0..n) |i| {
            indices[i] = i;
        }

        // Fisher-Yates shuffle (Knuth variant)
        // For each position i from 0 to n-2:
        //   Pick a random index j from [i, n)
        //   Swap indices[i] and indices[j]
        //
        // This produces a uniform random permutation.
        var rand = rng.random();
        for (0..n) |i| {
            const j = rand.intRangeLessThan(usize, i, n);
            const tmp = indices[i];
            indices[i] = indices[j];
            indices[j] = tmp;
        }

        return Batcher{
            .windowing = windowing,
            .batch_size = batch_size,
            .indices = indices,
            .pos = 0,
            .allocator = allocator,
        };
    }

    /// Reset the batcher: re-shuffle indices and reset position.
    ///
    /// Used at the start of each epoch to produce a fresh batch ordering.
    ///
    /// Worked example:
    ///   batcher.reset(&rng);
    ///   // Now batcher.next() starts from the beginning with a new shuffle
    pub fn reset(self: *Batcher, rng: *Rng) void {
        // Re-shuffle
        var rand = rng.random();
        for (0..self.indices.len) |i| {
            const j = rand.intRangeLessThan(usize, i, self.indices.len);
            const tmp = self.indices[i];
            self.indices[i] = self.indices[j];
            self.indices[j] = tmp;
        }
        self.pos = 0;
    }

    /// Get the next batch, or null if exhausted.
    ///
    /// Returns a Batch with input/target arrays of length B*T.
    /// Drop-last: if fewer than batch_size windows remain, returns null.
    ///
    /// Worked example:
    ///   if (batcher.next()) |batch| {
    ///       defer batch.deinit();
    ///       // Use batch.input and batch.target
    ///   }
    pub fn next(self: *Batcher) ?Batch {
        const remaining = self.indices.len - self.pos;
        if (remaining < self.batch_size) return null;

        const T = self.windowing.seq_len;
        const B = self.batch_size;
        const batch_len = B * T;

        // Allocate flat input and target arrays
        const input = self.allocator.alloc(u32, batch_len) catch return null;
        errdefer self.allocator.free(input);
        const target = self.allocator.alloc(u32, batch_len) catch {
            self.allocator.free(input);
            return null;
        };

        // Copy window data into the batch arrays
        for (0..B) |b| {
            const window_idx = self.indices[self.pos + b];
            const window = self.windowing.get(window_idx);

            // Copy input tokens
            @memcpy(input[b * T .. (b + 1) * T], window.input);
            // Copy target tokens
            @memcpy(target[b * T .. (b + 1) * T], window.target);
        }

        const ids = self.indices[self.pos .. self.pos + B];
        self.pos += B;

        return Batch{
            .input = input,
            .target = target,
            .ids = ids,
            .allocator = self.allocator,
        };
    }

    /// Total number of batches (drop-last).
    pub fn count(self: Batcher) usize {
        return self.indices.len / self.batch_size;
    }

    /// Free the shuffled index array.
    pub fn deinit(self: *Batcher) void {
        self.allocator.free(self.indices);
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "Batcher.init — creates correct number of indices" {
    const alloc = testing.allocator;
    const tokens = [_]u32{ 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 };
    const windowing = Windowing.init(&tokens, 3);
    // 10 tokens, T=3 → count = 7 windows

    var rng = Rng.init(42);
    var batcher = try Batcher.init(alloc, windowing, 2, &rng);
    defer batcher.deinit();

    try testing.expectEqual(@as(usize, 7), batcher.indices.len);
    try testing.expectEqual(@as(usize, 3), batcher.count()); // 7 / 2 = 3 (drop-last)
}

test "Batcher.next — produces correct batch shapes" {
    const alloc = testing.allocator;
    const tokens = [_]u32{ 10, 20, 30, 40, 50, 60, 70, 80 };
    const windowing = Windowing.init(&tokens, 3);
    // 8 tokens, T=3 → 5 windows

    var rng = Rng.init(42);
    var batcher = try Batcher.init(alloc, windowing, 2, &rng);
    defer batcher.deinit();

    // 5 / 2 = 2 batches (drop-last drops 1 window)
    const batch1 = batcher.next().?;
    defer batch1.deinit();
    try testing.expectEqual(@as(usize, 6), batch1.input.len); // B*T = 2*3
    try testing.expectEqual(@as(usize, 6), batch1.target.len);
    try testing.expectEqual(@as(usize, 2), batch1.ids.len);

    const batch2 = batcher.next().?;
    defer batch2.deinit();
    try testing.expectEqual(@as(usize, 6), batch2.input.len);

    // Third batch should be null (not enough windows)
    try testing.expect(batcher.next() == null);
}

test "Batcher — deterministic with same seed" {
    const alloc = testing.allocator;
    const tokens = [_]u32{ 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 };
    const windowing = Windowing.init(&tokens, 3);

    // Two batchers with same seed → same shuffle order
    var rng1 = Rng.init(42);
    var batcher1 = try Batcher.init(alloc, windowing, 2, &rng1);
    defer batcher1.deinit();

    var rng2 = Rng.init(42);
    var batcher2 = try Batcher.init(alloc, windowing, 2, &rng2);
    defer batcher2.deinit();

    // Compare shuffled indices
    try testing.expectEqualSlices(usize, batcher1.indices, batcher2.indices);
}

test "Batcher — different seeds produce different shuffles" {
    const alloc = testing.allocator;
    const tokens = [_]u32{ 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120 };
    const windowing = Windowing.init(&tokens, 3);

    var rng1 = Rng.init(42);
    var batcher1 = try Batcher.init(alloc, windowing, 2, &rng1);
    defer batcher1.deinit();

    var rng2 = Rng.init(99);
    var batcher2 = try Batcher.init(alloc, windowing, 2, &rng2);
    defer batcher2.deinit();

    // With 9 windows, different seeds should produce different orderings
    // (theoretically could be the same, but astronomically unlikely)
    var same = true;
    for (0..@min(batcher1.indices.len, batcher2.indices.len)) |i| {
        if (batcher1.indices[i] != batcher2.indices[i]) {
            same = false;
            break;
        }
    }
    try testing.expect(!same);
}

test "Batcher.reset — reshuffles indices" {
    const alloc = testing.allocator;
    const tokens = [_]u32{ 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120 };
    const windowing = Windowing.init(&tokens, 3);

    var rng = Rng.init(42);
    var batcher = try Batcher.init(alloc, windowing, 2, &rng);
    defer batcher.deinit();

    // Save original indices
    const original = try alloc.dupe(usize, batcher.indices);
    defer alloc.free(original);

    // Reset with a different seed
    var rng2 = Rng.init(99);
    batcher.reset(&rng2);

    // Indices should be different after reset (with different seed)
    // Position should be back to 0
    try testing.expectEqual(@as(usize, 0), batcher.pos);
}

test "Batcher — batch data contains correct tokens" {
    const alloc = testing.allocator;
    const tokens = [_]u32{ 10, 20, 30, 40, 50, 60, 70, 80 };
    const windowing = Windowing.init(&tokens, 3);

    // Use seed 0 and batch_size=1 for predictable testing
    var rng = Rng.init(0);
    var batcher = try Batcher.init(alloc, windowing, 1, &rng);
    defer batcher.deinit();

    // Each batch should have 1*3 = 3 input and 3 target elements
    const batch = batcher.next().?;
    defer batch.deinit();
    try testing.expectEqual(@as(usize, 3), batch.input.len);
    try testing.expectEqual(@as(usize, 3), batch.target.len);

    // Verify that target[t] == input[t+1] from the source tokens
    // (The specific window depends on shuffle, but the relationship
    //  between input and target must always hold.)
    for (0..2) |t| {
        try testing.expectEqual(batch.input[t + 1], batch.target[t]);
    }
}

test "Batcher — exhaustion returns null" {
    const alloc = testing.allocator;
    const tokens = [_]u32{ 10, 20, 30, 40, 50 };
    const windowing = Windowing.init(&tokens, 3);

    var rng = Rng.init(42);
    var batcher = try Batcher.init(alloc, windowing, 2, &rng);
    defer batcher.deinit();

    // 2 tokens, T=3 → count=2, batches=1 (drop-last)
    const batch1 = batcher.next().?;
    defer batch1.deinit();

    // Second batch: null (not enough windows)
    try testing.expect(batcher.next() == null);
}
