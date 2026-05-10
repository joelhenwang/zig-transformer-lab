//!
//! zig-transformer-lab — Word-level tokenizer: encode and decode
//!
//! Purpose:
//!   Converts raw text to a sequence of integer token IDs (encode) and
//!   converts a sequence of IDs back to text (decode). This is the bridge
//!   between human-readable text and the numeric representations the
//!   transformer operates on.
//!
//!   Tokenization algorithm (word-level with punctuation peeling):
//!     1. Optionally lowercase the entire text.
//!     2. Collapse runs of whitespace (spaces, tabs, newlines) into single spaces.
//!     3. Split on whitespace to get "raw tokens" (words with attached punctuation).
//!     4. Peel punctuation from each raw token:
//!        - Leading punctuation chars are split off as separate tokens.
//!        - Trailing punctuation chars are split off as separate tokens.
//!        - The middle (if any) is the word token.
//!        - Apostrophes are treated like any other punctuation: "don't" → "don" "'" "t"
//!     5. Look up each token string in the Vocab to get its ID.
//!        Unknown tokens get the <unk> ID (0).
//!
//!   Punctuation characters that get peeled:
//!     . , ! ? ; : " ' ( )
//!
//!   Examples:
//!     "Hello, world!"  →  ["hello", ",", "world", "!"]
//!     "don't stop"     →  ["don", "'", "t", "stop"]
//!     "the cat."       →  ["the", "cat", "."]
//!     "(hello)"        →  ["(", "hello", ")"]
//!
//! Shape contract:
//!   No tensor shapes — the tokenizer produces []u32 (1D integer array).
//!   The downstream code reshapes this into (B, T) for the model.
//!
//! Math:
//!   No arithmetic — this is pure string processing. The "math" is in
//!   the embedding layer that converts IDs to vectors.
//!
//! Memory ownership:
//!   - tokenize(): returns an ArrayList of owned string slices.
//!     Caller must free each slice and the list itself.
//!   - encode(): returns a heap-allocated []u32. Caller must free.
//!   - decode(): returns a heap-allocated []u8. Caller must free.
//!
//! Error conditions:
//!   OutOfMemory — from ArrayList/allocator allocation
//!
//! Credits:
//!   The punctuation-peeling approach is common in simple word tokenizers.
//!   PyTorch's torchtext basic_english tokenizer uses a similar strategy.
//!   No code copied.

const std = @import("std");
const LabError = @import("../core/errors.zig").LabError;
const Vocab = @import("vocab.zig").Vocab;

/// The set of punctuation characters that get peeled off words.
/// Each character in this string becomes its own token when found
/// at the start or end of a raw token.
///
/// Why these characters?
///   - Sentence-ending: . ! ?
///   - Clause-ending: , ; :
///   - Quotation: " '
///   - Grouping: ( )
///
/// Apostrophe (') is included, which means contractions like "don't"
/// get split into ["don", "'", "t"]. This is simpler and more
/// pedagogically transparent than trying to handle English morphology.
const PUNCTUATION = ".,!?;:\"'()";

/// Check if a character is a punctuation character that should be peeled.
fn isPunct(ch: u8) bool {
    return std.mem.indexOfScalar(u8, PUNCTUATION, ch) != null;
}

/// Tokenize raw text into a list of word/punctuation token strings.
///
/// Steps:
///   1. Optionally lowercase
///   2. Collapse whitespace runs
///   3. Split on whitespace
///   4. Peel leading/trailing punctuation from each token
///
/// The returned ArrayList contains owned string slices. Each slice must
/// be freed by the caller, and the list itself must be deinitialized.
///
/// Worked example:
///   var tokens = try tokenize(alloc, "Hello, world!", true);
///   defer { for (tokens.items) |t| alloc.free(t); tokens.deinit(alloc); }
///   // tokens == ["hello", ",", "world", "!"]
pub fn tokenize(allocator: std.mem.Allocator, text: []const u8, lowercase: bool) LabError!std.ArrayList([]const u8) {
    var result: std.ArrayList([]const u8) = .empty;
    errdefer {
        for (result.items) |tok| allocator.free(tok);
        result.deinit(allocator);
    }

    // Step 1: Lowercase if requested
    // We work in a scratch buffer to avoid modifying the input.
    // For large inputs, this means a full copy — acceptable for our
    // ~1 MB corpus size.
    var buf = try allocator.alloc(u8, text.len);
    defer allocator.free(buf);

    if (lowercase) {
        for (text, 0..) |ch, i| {
            buf[i] = std.ascii.toLower(ch);
        }
    } else {
        @memcpy(buf, text);
    }

    // Step 2: Split on whitespace (collapses runs automatically)
    // std.mem.splitAny handles any whitespace character and skips
    // empty segments between consecutive delimiters.
    var it = std.mem.splitAny(u8, buf, " \t\n\r");
    while (it.next()) |raw_tok| {
        if (raw_tok.len == 0) continue;

        // Step 3 & 4: Peel punctuation from this raw token
        try peelPunctuation(allocator, raw_tok, &result);
    }

    return result;
}

/// Peel leading and trailing punctuation from a raw token string,
/// then split the remaining word on internal apostrophes, emitting
/// each piece as a separate token.
///
/// Algorithm:
///   1. Peel leading punctuation characters one by one.
///   2. Peel trailing punctuation characters one by one.
///   3. The remaining middle (if any) is split on apostrophes (').
///      Each apostrophe becomes its own token; the substrings between
///      apostrophes become word tokens.
///
/// Why split on internal apostrophes?
///   Our punctuation set includes ' (apostrophe), but the peel-leading/
///   peel-trailing step doesn't catch apostrophes in the MIDDLE of a
///   word (e.g., "don't", "cat's"). We explicitly split on internal
///   apostrophes so that:
///     "don't"  → ["don", "'", "t"]
///     "cat's"  → ["cat", "'", "s"]
///     "it's"   → ["it", "'", "s"]
///   This keeps the tokenizer simple and transparent — every punctuation
///   character is always its own token, regardless of position.
///
/// Worked example:
///   Input:  "hello,"
///   Output: ["hello", ","]
///
///   Input:  "(world!)"
///   Output: ["(", "world", "!", ")"]
///
///   Input:  "..."
///   Output: [".", ".", "."]
///
///   Input:  "don't"
///   Output: ["don", "'", "t"]
fn peelPunctuation(allocator: std.mem.Allocator, raw: []const u8, result: *std.ArrayList([]const u8)) !void {
    var start: usize = 0;
    const end: usize = raw.len;

    // Peel leading punctuation
    while (start < end and isPunct(raw[start])) {
        const tok = try allocator.dupe(u8, raw[start .. start + 1]);
        try result.append(allocator, tok);
        start += 1;
    }

    // Peel trailing punctuation (from the back)
    var trailing_start = end;
    while (trailing_start > start and isPunct(raw[trailing_start - 1])) {
        trailing_start -= 1;
    }

    // The middle word (between leading and trailing punctuation)
    // Split on internal apostrophes: "don't" → "don" + "'" + "t"
    if (start < trailing_start) {
        const middle = raw[start..trailing_start];
        var seg_start: usize = 0;
        while (seg_start < middle.len) {
            // Find next apostrophe in the remaining middle
            const rel_idx = std.mem.indexOfScalar(u8, middle[seg_start..], '\'');
            if (rel_idx) |ri| {
                const abs_idx = seg_start + ri;
                // Emit the word segment before the apostrophe (if non-empty)
                if (abs_idx > seg_start) {
                    const tok = try allocator.dupe(u8, middle[seg_start..abs_idx]);
                    try result.append(allocator, tok);
                }
                // Emit the apostrophe itself
                const apostrophe = try allocator.dupe(u8, "'");
                try result.append(allocator, apostrophe);
                seg_start = abs_idx + 1;
            } else {
                // No more apostrophes — emit the rest as one token
                const tok = try allocator.dupe(u8, middle[seg_start..]);
                try result.append(allocator, tok);
                break;
            }
        }
    }

    // Emit trailing punctuation characters in left-to-right order
    for (trailing_start..end) |i| {
        const tok = try allocator.dupe(u8, raw[i .. i + 1]);
        try result.append(allocator, tok);
    }
}

/// Encode a text string into a sequence of token IDs using a Vocab.
///
/// Steps:
///   1. Tokenize the text (split + punctuation peeling)
///   2. Look up each token in the Vocab → u32 ID
///   3. Unknown tokens get the <unk> ID (0)
///
/// Worked example:
///   var ids = try encode(alloc, "the cat.", &vocab);
///   defer alloc.free(ids);
///   // ids == [4, 5, 6]  (the=4, cat=5, .=6, assuming vocab order)
pub fn encode(allocator: std.mem.Allocator, text: []const u8, vocab: *const Vocab) LabError![]u32 {
    var tokens = try tokenize(allocator, text, vocab.lowercase);
    defer {
        for (tokens.items) |tok| allocator.free(tok);
        tokens.deinit(allocator);
    }

    var ids = try allocator.alloc(u32, tokens.items.len);
    errdefer allocator.free(ids);

    for (tokens.items, 0..) |tok, i| {
        ids[i] = vocab.encodeWord(tok);
    }

    return ids;
}

/// Decode a sequence of token IDs back into a text string.
///
/// Joins the decoded words with single spaces. Punctuation tokens
/// get spaces around them (e.g., "hello , world !"), which is the
/// accepted difference from the original text per the round-trip
/// criterion: "differs from the original only in whitespace collapse."
///
/// Worked example:
///   var text = try decode(alloc, &.{4, 5, 6}, &vocab);
///   defer alloc.free(text);
///   // text == "the cat ."  (space before period is expected)
pub fn decode(allocator: std.mem.Allocator, ids: []const u32, vocab: *const Vocab) LabError![]u8 {
    if (ids.len == 0) {
        const empty = try allocator.alloc(u8, 0);
        return empty;
    }

    // First pass: compute total length needed
    var total_len: usize = 0;
    for (ids) |id| {
        const word = vocab.decodeId(id);
        total_len += word.len;
    }
    // Add spaces between words: (n-1) spaces
    total_len += ids.len - 1;

    var result = try allocator.alloc(u8, total_len);
    errdefer allocator.free(result);

    var pos: usize = 0;
    for (ids, 0..) |id, i| {
        if (i > 0) {
            result[pos] = ' ';
            pos += 1;
        }
        const word = vocab.decodeId(id);
        @memcpy(result[pos .. pos + word.len], word);
        pos += word.len;
    }

    return result;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "tokenize — simple words" {
    const alloc = testing.allocator;
    var tokens = try tokenize(alloc, "hello world", true);
    defer {
        for (tokens.items) |t| alloc.free(t);
        tokens.deinit(alloc);
    }

    try testing.expectEqual(@as(usize, 2), tokens.items.len);
    try testing.expectEqualStrings("hello", tokens.items[0]);
    try testing.expectEqualStrings("world", tokens.items[1]);
}

test "tokenize — punctuation peeling" {
    const alloc = testing.allocator;
    var tokens = try tokenize(alloc, "hello, world!", true);
    defer {
        for (tokens.items) |t| alloc.free(t);
        tokens.deinit(alloc);
    }

    try testing.expectEqual(@as(usize, 4), tokens.items.len);
    try testing.expectEqualStrings("hello", tokens.items[0]);
    try testing.expectEqualStrings(",", tokens.items[1]);
    try testing.expectEqualStrings("world", tokens.items[2]);
    try testing.expectEqualStrings("!", tokens.items[3]);
}

test "tokenize — apostrophe splits contractions" {
    const alloc = testing.allocator;
    var tokens = try tokenize(alloc, "don't stop", true);
    defer {
        for (tokens.items) |t| alloc.free(t);
        tokens.deinit(alloc);
    }

    // "don't" → "don" + "'" + "t"
    try testing.expectEqual(@as(usize, 4), tokens.items.len);
    try testing.expectEqualStrings("don", tokens.items[0]);
    try testing.expectEqualStrings("'", tokens.items[1]);
    try testing.expectEqualStrings("t", tokens.items[2]);
    try testing.expectEqualStrings("stop", tokens.items[3]);
}

test "tokenize — possessive apostrophe" {
    const alloc = testing.allocator;
    var tokens = try tokenize(alloc, "cat's tail", true);
    defer {
        for (tokens.items) |t| alloc.free(t);
        tokens.deinit(alloc);
    }

    // "cat's" → "cat" + "'" + "s"
    try testing.expectEqual(@as(usize, 4), tokens.items.len);
    try testing.expectEqualStrings("cat", tokens.items[0]);
    try testing.expectEqualStrings("'", tokens.items[1]);
    try testing.expectEqualStrings("s", tokens.items[2]);
    try testing.expectEqualStrings("tail", tokens.items[3]);
}

test "tokenize — parentheses peeling" {
    const alloc = testing.allocator;
    var tokens = try tokenize(alloc, "(hello)", true);
    defer {
        for (tokens.items) |t| alloc.free(t);
        tokens.deinit(alloc);
    }

    try testing.expectEqual(@as(usize, 3), tokens.items.len);
    try testing.expectEqualStrings("(", tokens.items[0]);
    try testing.expectEqualStrings("hello", tokens.items[1]);
    try testing.expectEqualStrings(")", tokens.items[2]);
}

test "tokenize — only punctuation" {
    const alloc = testing.allocator;
    var tokens = try tokenize(alloc, "...", true);
    defer {
        for (tokens.items) |t| alloc.free(t);
        tokens.deinit(alloc);
    }

    try testing.expectEqual(@as(usize, 3), tokens.items.len);
    try testing.expectEqualStrings(".", tokens.items[0]);
    try testing.expectEqualStrings(".", tokens.items[1]);
    try testing.expectEqualStrings(".", tokens.items[2]);
}

test "tokenize — whitespace collapse" {
    const alloc = testing.allocator;
    var tokens = try tokenize(alloc, "hello   world", true);
    defer {
        for (tokens.items) |t| alloc.free(t);
        tokens.deinit(alloc);
    }

    try testing.expectEqual(@as(usize, 2), tokens.items.len);
    try testing.expectEqualStrings("hello", tokens.items[0]);
    try testing.expectEqualStrings("world", tokens.items[1]);
}

test "tokenize — lowercase flag" {
    const alloc = testing.allocator;

    var tokens_upper = try tokenize(alloc, "HELLO", false);
    defer {
        for (tokens_upper.items) |t| alloc.free(t);
        tokens_upper.deinit(alloc);
    }
    try testing.expectEqualStrings("HELLO", tokens_upper.items[0]);

    var tokens_lower = try tokenize(alloc, "HELLO", true);
    defer {
        for (tokens_lower.items) |t| alloc.free(t);
        tokens_lower.deinit(alloc);
    }
    try testing.expectEqualStrings("hello", tokens_lower.items[0]);
}

test "tokenize — mixed punctuation" {
    const alloc = testing.allocator;
    var tokens = try tokenize(alloc, "the cat, the dog.", true);
    defer {
        for (tokens.items) |t| alloc.free(t);
        tokens.deinit(alloc);
    }

    // "the cat, the dog." → ["the", "cat", ",", "the", "dog", "."]
    try testing.expectEqual(@as(usize, 6), tokens.items.len);
    try testing.expectEqualStrings("the", tokens.items[0]);
    try testing.expectEqualStrings("cat", tokens.items[1]);
    try testing.expectEqualStrings(",", tokens.items[2]);
    try testing.expectEqualStrings("the", tokens.items[3]);
    try testing.expectEqualStrings("dog", tokens.items[4]);
    try testing.expectEqualStrings(".", tokens.items[5]);
}

test "encode — produces correct IDs" {
    const alloc = testing.allocator;

    // Build a vocab first
    var vocab = try Vocab.buildFromText(alloc, "hello world hello", 100, true, tokenize);
    defer vocab.deinit();

    const ids = try encode(alloc, "hello world", &vocab);
    defer alloc.free(ids);

    try testing.expectEqual(@as(usize, 2), ids.len);
    // Both should be >= NUM_SPECIALS (valid vocab entries)
    try testing.expect(ids[0] >= 4);
    try testing.expect(ids[1] >= 4);
}

test "encode — unknown words get UNK ID" {
    const alloc = testing.allocator;

    var vocab = try Vocab.buildFromText(alloc, "hello world", 100, true, tokenize);
    defer vocab.deinit();

    const ids = try encode(alloc, "hello unknown", &vocab);
    defer alloc.free(ids);

    try testing.expectEqual(@as(u32, 0), ids[1]); // "unknown" → <unk>
}

test "decode — joins with spaces" {
    const alloc = testing.allocator;

    var vocab = try Vocab.buildFromText(alloc, "hello world", 100, true, tokenize);
    defer vocab.deinit();

    const ids = try encode(alloc, "hello world", &vocab);
    defer alloc.free(ids);

    const text = try decode(alloc, ids, &vocab);
    defer alloc.free(text);

    try testing.expectEqualStrings("hello world", text);
}

test "encode/decode round-trip" {
    const alloc = testing.allocator;

    const original = "hello world";
    var vocab = try Vocab.buildFromText(alloc, original, 100, true, tokenize);
    defer vocab.deinit();

    const ids = try encode(alloc, original, &vocab);
    defer alloc.free(ids);

    const decoded = try decode(alloc, ids, &vocab);
    defer alloc.free(decoded);

    try testing.expectEqualStrings("hello world", decoded);
}

test "encode/decode round-trip with punctuation" {
    const alloc = testing.allocator;

    const original = "hello, world!";
    var vocab = try Vocab.buildFromText(alloc, original, 100, true, tokenize);
    defer vocab.deinit();

    const ids = try encode(alloc, original, &vocab);
    defer alloc.free(ids);

    const decoded = try decode(alloc, ids, &vocab);
    defer alloc.free(decoded);

    // Decoded text has spaces around punctuation: "hello , world !"
    // This is the accepted whitespace-collapse difference.
    try testing.expectEqualStrings("hello , world !", decoded);
}

test "encode — empty text" {
    const alloc = testing.allocator;

    var vocab = try Vocab.buildFromText(alloc, "hello", 100, true, tokenize);
    defer vocab.deinit();

    const ids = try encode(alloc, "", &vocab);
    defer alloc.free(ids);

    try testing.expectEqual(@as(usize, 0), ids.len);
}
