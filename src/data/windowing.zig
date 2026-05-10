//!
//! zig-transformer-lab — Windowing: (input, target) pairs for next-token prediction
//!
//! Purpose:
//!   Creates training examples from a flat token stream by sliding a
//!   window of length T over the tokens. Each window produces one
//!   (input, target) pair where:
//!     input[t]  = tokens[i + t]       for t = 0..T-1
//!     target[t] = tokens[i + t + 1]   for t = 0..T-1
//!
//!   This is the standard next-token prediction setup for autoregressive
//!   language models: given the first T tokens, predict the next token
//!   at each position.
//!
//!   Why not shift the target by a full sequence?
//!   The target for position t is always the token at t+1. This means
//!   the input and target overlap by T-1 tokens — they share T-1 out
//!   of T positions. This is correct: the model learns to predict the
//!   next token at every position simultaneously.
//!
//! Shape contract:
//!   Window.input  — []u32 of length T (view into the token stream)
//!   Window.target — []u32 of length T (view, offset by 1)
//!
//!   For a token stream of length N with window size T:
//!     Number of windows = N - T (if N > T, else 0)
//!
//!   Example with T=4, tokens = [10, 20, 30, 40, 50, 60]:
//!     Window 0: input=[10,20,30,40], target=[20,30,40,50]
//!     Window 1: input=[20,30,40,50], target=[30,40,50,60]
//!
//! Memory ownership:
//!   Windowing does NOT copy token data — input and target are slices
//!   (views) into the underlying []u32 token stream. The caller must
//!   ensure the token stream outlives the windows.
//!
//! Error conditions:
//!   None — Windowing is infallible (no allocation for views).
//!   allWindows() allocates and can return OutOfMemory.
//!
//! Credits:
//!   Standard sliding-window approach for autoregressive LM training.
//!   nanoGPT uses the same offset-by-1 pattern. No code copied.

const std = @import("std");
const LabError = @import("../core/errors.zig").LabError;

/// A single training window: (input, target) pair.
///
/// Both are views (slices) into the same token stream.
/// input[t] and target[t] differ by one position:
///   target[t] = input[t+1] for all t < T
///
/// Worked example:
///   // Given: tokens = [10, 20, 30, 40, 50], T=3
///   // Window 0: input=[10,20,30], target=[20,30,40]
///   // Window 1: input=[20,30,40], target=[30,40,50]
pub const Window = struct {
    /// Input token IDs — the context the model sees.
    input: []const u32,
    /// Target token IDs — what the model should predict.
    target: []const u32,
};

/// Sliding window iterator over a token stream.
///
/// Creates (input, target) pairs for next-token prediction.
/// No allocation — windows are views into the token stream.
///
/// Worked example:
///   var windowing = Windowing.init(tokens, 8);
///   const n = windowing.count();   // number of valid windows
///   const w = windowing.get(0);    // first (input, target) pair
pub const Windowing = struct {
    /// The full token stream (borrowed — not owned).
    tokens: []const u32,
    /// Window size (sequence length T).
    seq_len: usize,

    /// Create a Windowing over a token stream with the given window size.
    ///
    /// No allocation — just stores the slice and seq_len.
    ///
    /// Worked example:
    ///   var w = Windowing.init(&.{10, 20, 30, 40, 50}, 3);
    ///   // w.count() == 2
    pub fn init(tokens: []const u32, seq_len: usize) Windowing {
        return .{
            .tokens = tokens,
            .seq_len = seq_len,
        };
    }

    /// Number of valid windows in the token stream.
    ///
    /// A window at starting index i exposes:
    ///   input  = tokens[i        .. i + T]     (T tokens)
    ///   target = tokens[i + 1    .. i + 1 + T] (T tokens)
    ///
    /// The last valid index needs tokens[i + T] to exist, i.e.
    /// `i + T < tokens.len`. So i ranges from 0 to tokens.len - T - 1
    /// inclusive, giving `tokens.len - T` windows when `tokens.len > T`,
    /// and 0 otherwise.
    ///
    /// Derivation in one line:
    ///   count = max(tokens.len - seq_len, 0)
    ///
    /// Worked example:
    ///   tokens.len = 5, T = 3
    ///   valid starts: i = 0 (needs indices 0..3), i = 1 (needs indices 1..4)
    ///   count = 5 - 3 = 2  ✓
    pub fn count(self: Windowing) usize {
        if (self.tokens.len <= self.seq_len) return 0;
        return self.tokens.len - self.seq_len;
    }

    /// Get the window at index `idx`.
    ///
    /// Returns a Window with input and target as views into the token stream.
    /// Caller must ensure idx < count().
    ///
    /// Worked example:
    ///   // tokens = [10, 20, 30, 40, 50], T = 3
    ///   const w = windowing.get(0);
    ///   // w.input  = [10, 20, 30]
    ///   // w.target = [20, 30, 40]
    pub fn get(self: Windowing, idx: usize) Window {
        const start = idx;
        return .{
            .input = self.tokens[start .. start + self.seq_len],
            .target = self.tokens[start + 1 .. start + 1 + self.seq_len],
        };
    }

    /// Materialize all windows into a heap-allocated array.
    ///
    /// Useful for the batcher, which needs to shuffle window indices.
    /// Each Window in the array is still a view (no data copied).
    ///
    /// Worked example:
    ///   var windows = try windowing.allWindows(alloc);
    ///   defer alloc.free(windows);
    ///   // windows.len == windowing.count()
    pub fn allWindows(self: Windowing, allocator: std.mem.Allocator) LabError![]Window {
        const n = self.count();
        const windows = try allocator.alloc(Window, n);
        for (0..n) |i| {
            windows[i] = self.get(i);
        }
        return windows;
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "Windowing.count — correct number of windows" {
    const tokens = [_]u32{ 10, 20, 30, 40, 50 };
    const w = Windowing.init(&tokens, 3);

    // 5 tokens, T=3: windows at i=0 and i=1 → count=2
    try testing.expectEqual(@as(usize, 2), w.count());
}

test "Windowing.count — too few tokens" {
    const tokens = [_]u32{ 10, 20 };
    const w = Windowing.init(&tokens, 3);

    // 2 tokens, T=3: need at least 4 tokens for one window
    try testing.expectEqual(@as(usize, 0), w.count());
}

test "Windowing.count — exactly one window" {
    const tokens = [_]u32{ 10, 20, 30, 40 };
    const w = Windowing.init(&tokens, 3);

    // 4 tokens, T=3: one window (needs tokens[0..3] and tokens[1..4])
    try testing.expectEqual(@as(usize, 1), w.count());
}

test "Windowing.get — correct input and target" {
    const tokens = [_]u32{ 10, 20, 30, 40, 50, 60 };
    const w = Windowing.init(&tokens, 3);

    // Window 0: input=[10,20,30], target=[20,30,40]
    const w0 = w.get(0);
    try testing.expectEqualSlices(u32, &.{ 10, 20, 30 }, w0.input);
    try testing.expectEqualSlices(u32, &.{ 20, 30, 40 }, w0.target);

    // Window 1: input=[20,30,40], target=[30,40,50]
    const w1 = w.get(1);
    try testing.expectEqualSlices(u32, &.{ 20, 30, 40 }, w1.input);
    try testing.expectEqualSlices(u32, &.{ 30, 40, 50 }, w1.target);

    // Window 2: input=[30,40,50], target=[40,50,60]
    const w2 = w.get(2);
    try testing.expectEqualSlices(u32, &.{ 30, 40, 50 }, w2.input);
    try testing.expectEqualSlices(u32, &.{ 40, 50, 60 }, w2.target);
}

test "Windowing.allWindows — correct length" {
    const alloc = testing.allocator;
    const tokens = [_]u32{ 10, 20, 30, 40, 50 };
    const w = Windowing.init(&tokens, 3);

    const windows = try w.allWindows(alloc);
    defer alloc.free(windows);

    try testing.expectEqual(@as(usize, 2), windows.len);
}

test "Windowing — views share the same underlying data" {
    const tokens = [_]u32{ 10, 20, 30, 40, 50, 60 };
    const w = Windowing.init(&tokens, 3);

    const w0 = w.get(0);
    const w1 = w.get(1);

    // input[1] of window 0 == input[0] of window 1
    try testing.expectEqual(w0.input[1], w1.input[0]);
    try testing.expectEqual(@as(u32, 20), w0.input[1]);
}
