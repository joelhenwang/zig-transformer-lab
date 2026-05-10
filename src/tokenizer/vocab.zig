//!
//! zig-transformer-lab — Vocabulary struct for word-level tokenization
//!
//! Purpose:
//!   Maps words to integer IDs and back. This is the core data structure
//!   for the tokenizer: every unique word gets an ID, and the model learns
//!   an embedding for each ID.
//!
//!   Why word-level (not character-level or BPE)?
//!   - Simplicity: no merge rules, no byte-level encoding, just split on
//!     whitespace and peel punctuation.
//!   - Pedagogical clarity: the mapping is transparent — "the" → ID 4,
//!     "cat" → ID 5, etc.
//!   - Tradeoff: large vocabularies for real corpora; we handle this with
//!     a frequency cutoff (top-K by count, rest → <unk>).
//!
//! Shape contract:
//!   No tensor shapes — Vocab is a lookup table, not a numeric array.
//!   The vocab_size (number of entries including specials) feeds into
//!   TransformerConfig.vocab_size for the embedding layer.
//!
//! Special token layout (reserved IDs 0-3):
//!   ID 0: <unk> — unknown / out-of-vocabulary words
//!   ID 1: <pad> — padding (for batching variable-length sequences)
//!   ID 2: <bos> — beginning of sequence
//!   ID 3: <eos> — end of sequence
//!
//!   These four IDs are always present, even for an empty corpus.
//!   Real words start at ID 4.
//!
//! Serialization format:
//!   One line per entry: "ID<TAB>WORD\n" (tab-separated, sorted by ID).
//!   UTF-8 throughout. Stable across runs (sorted by id).
//!
//!   Example:
//!     0  <unk>
//!     1  <pad>
//!     2  <bos>
//!     3  <eos>
//!     4  the
//!     5  cat
//!     6  sat
//!
//! Memory ownership:
//!   Vocab owns all heap-allocated data:
//!   - word_to_id: keys are copies of word strings
//!   - id_to_word: items are copies of word strings
//!   - counts: frequency counts for each word (including specials, set to 0)
//!   Call deinit() to free everything.
//!
//! Error conditions:
//!   OutOfMemory — from HashMap/ArrayList allocation
//!   IoError — from save/load file I/O
//!
//! TODO:
//!   - Stage 7: consider CUDA-side vocab for GPU text generation
//!
//! Credits:
//!   Word-level tokenization is standard in NLP. The special-token layout
//!   follows the convention used by GPT-2 and many modern LMs.
//!   No code copied.

const std = @import("std");
const LabError = @import("../core/errors.zig").LabError;

/// Special token IDs — reserved at the start of the vocabulary.
/// These are always present and cannot be overwritten.
pub const UNK_ID: u32 = 0;
pub const PAD_ID: u32 = 1;
pub const BOS_ID: u32 = 2;
pub const EOS_ID: u32 = 3;
pub const NUM_SPECIALS: u32 = 4;

pub const UNK_TOKEN = "<unk>";
pub const PAD_TOKEN = "<pad>";
pub const BOS_TOKEN = "<bos>";
pub const EOS_TOKEN = "<eos>";

/// Type signature for a tokenize function.
/// Returns an ArrayList of owned string slices (caller frees each + the list).
pub const TokenizeFn = fn (std.mem.Allocator, []const u8, bool) LabError!std.ArrayList([]const u8);

pub const Vocab = struct {
    /// Map from word string → token ID.
    /// Keys are owned copies allocated with self.allocator.
    word_to_id: std.StringHashMap(u32),

    /// Map from token ID → word string.
    /// Items are owned copies allocated with self.allocator.
    id_to_word: std.ArrayList([]const u8),

    /// Frequency count for each token (index == ID).
    /// Specials have count 0.
    counts: std.ArrayList(u32),

    /// Maximum vocabulary size (including specials).
    /// Words beyond the top-(max_vocab - 4) by frequency map to <unk>.
    max_vocab: usize,

    /// Whether to lowercase all text before processing.
    lowercase: bool,

    /// Allocator used for all string copies and data structures.
    allocator: std.mem.Allocator,

    /// Create an empty Vocab with reserved special tokens.
    ///
    /// Worked example:
    ///   var vocab = Vocab.init(allocator, 256, true);
    ///   defer vocab.deinit();
    pub fn init(allocator: std.mem.Allocator, max_vocab: usize, lowercase: bool) Vocab {
        return .{
            .word_to_id = std.StringHashMap(u32).init(allocator),
            .id_to_word = .empty,
            .counts = .empty,
            .max_vocab = max_vocab,
            .lowercase = lowercase,
            .allocator = allocator,
        };
    }

    /// Build a Vocab from raw text by counting word frequencies,
    /// then keeping the top (max_vocab - 4) words.
    ///
    /// Steps:
    ///   1. Reserve special token slots (IDs 0-3)
    ///   2. Tokenize the text (whitespace split + punctuation peeling)
    ///   3. Count word frequencies
    ///   4. Sort words by count (descending), break ties alphabetically
    ///   5. Keep top (max_vocab - 4) words; assign IDs starting at 4
    ///   6. All other words map to <unk> (ID 0)
    ///
    /// Worked example:
    ///   const text = "the cat sat. the dog ran.";
    ///   var vocab = try Vocab.buildFromText(allocator, text, 100, true, tokenize);
    ///   defer vocab.deinit();
    pub fn buildFromText(
        allocator: std.mem.Allocator,
        text: []const u8,
        max_vocab: usize,
        lowercase: bool,
        comptime tokenizeFn: *const TokenizeFn,
    ) LabError!Vocab {
        var self = init(allocator, max_vocab, lowercase);
        errdefer self.deinit();

        // Reserve the 4 special token slots
        try self.addSpecial(UNK_ID, UNK_TOKEN);
        try self.addSpecial(PAD_ID, PAD_TOKEN);
        try self.addSpecial(BOS_ID, BOS_TOKEN);
        try self.addSpecial(EOS_ID, EOS_TOKEN);

        // Tokenize the input text
        var tokens = try tokenizeFn(allocator, text, lowercase);
        defer {
            for (tokens.items) |tok| allocator.free(tok);
            tokens.deinit(allocator);
        }

        // Count word frequencies using a temporary HashMap
        var freq_map = std.StringHashMap(u32).init(allocator);
        defer {
            var it = freq_map.iterator();
            while (it.next()) |entry| allocator.free(entry.key_ptr.*);
            freq_map.deinit();
        }

        for (tokens.items) |token| {
            // Skip special tokens if they appear in text
            if (isSpecial(token)) continue;

            if (freq_map.get(token)) |count| {
                try freq_map.put(token, count + 1);
            } else {
                const owned = try allocator.dupe(u8, token);
                try freq_map.put(owned, 1);
            }
        }

        // Collect all (word, count) pairs for sorting
        const WordCount = struct { word: []const u8, count: u32 };
        var pairs: std.ArrayList(WordCount) = .empty;
        defer pairs.deinit(allocator);

        var it = freq_map.iterator();
        while (it.next()) |entry| {
            try pairs.append(allocator, .{ .word = entry.key_ptr.*, .count = entry.value_ptr.* });
        }

        // Sort by count descending, then alphabetically for tie-breaking
        std.sort.pdq(
            WordCount,
            pairs.items,
            {},
            struct {
                fn lessThan(_: void, a: WordCount, b: WordCount) bool {
                    if (a.count != b.count) return a.count > b.count;
                    return std.mem.order(u8, a.word, b.word) == .lt;
                }
            }.lessThan,
        );

        // Keep top (max_vocab - NUM_SPECIALS) words
        const max_real: usize = max_vocab - NUM_SPECIALS;
        const to_add = @min(pairs.items.len, max_real);

        for (pairs.items[0..to_add]) |pair| {
            // Dupe into vocab — freq_map will be freed separately
            const owned_word = try allocator.dupe(u8, pair.word);
            const id: u32 = @intCast(self.id_to_word.items.len);
            try self.id_to_word.append(allocator, owned_word);
            try self.counts.append(allocator, pair.count);
            try self.word_to_id.put(owned_word, id);
        }

        return self;
    }

    /// Add a special token at a reserved ID.
    fn addSpecial(self: *Vocab, id: u32, token: []const u8) !void {
        while (self.id_to_word.items.len < id) {
            try self.id_to_word.append(self.allocator, "");
            try self.counts.append(self.allocator, 0);
        }
        if (self.id_to_word.items.len == id) {
            const owned = try self.allocator.dupe(u8, token);
            try self.id_to_word.append(self.allocator, owned);
            try self.counts.append(self.allocator, 0);
            try self.word_to_id.put(owned, id);
        }
    }

    /// Look up the ID for a word. Returns <unk> (ID 0) if not found.
    ///
    /// Worked example:
    ///   const id = vocab.encodeWord("cat");  // e.g. 5
    pub fn encodeWord(self: Vocab, word: []const u8) u32 {
        return self.word_to_id.get(word) orelse UNK_ID;
    }

    /// Look up the word for an ID. Returns "<unk>" if out of range.
    ///
    /// Worked example:
    ///   const word = vocab.decodeId(5);  // e.g. "cat"
    pub fn decodeId(self: Vocab, id: u32) []const u8 {
        if (id >= self.id_to_word.items.len) return UNK_TOKEN;
        const word = self.id_to_word.items[id];
        if (word.len == 0) return UNK_TOKEN;
        return word;
    }

    /// Current vocabulary size (including specials).
    pub fn size(self: Vocab) usize {
        return self.id_to_word.items.len;
    }

    /// Check if a token string is one of the four specials.
    pub fn isSpecial(token: []const u8) bool {
        return std.mem.eql(u8, token, UNK_TOKEN) or
            std.mem.eql(u8, token, PAD_TOKEN) or
            std.mem.eql(u8, token, BOS_TOKEN) or
            std.mem.eql(u8, token, EOS_TOKEN);
    }

    /// Save the vocabulary to a text file.
    ///
    /// Format: one "ID<TAB>WORD\n" line per entry, sorted by ID (ascending).
    ///
    /// Worked example:
    ///   try vocab.save(io, "my_vocab.tsv");
    pub fn save(self: Vocab, io: std.Io, path: []const u8) !void {
        const cwd = std.Io.Dir.cwd();
        const file = try cwd.createFile(io, path, .{});
        defer file.close(io);
        var buf: [4096]u8 = undefined;
        var writer = file.writer(io, &buf);

        for (self.id_to_word.items, 0..) |word, id| {
            const id_u32: u32 = @intCast(id);
            try writer.interface.print("{d}\t{s}\n", .{ id_u32, word });
        }
        try writer.flush();
    }

    /// Load a vocabulary from a text file.
    ///
    /// Reads the "ID<TAB>WORD\n" format produced by save().
    ///
    /// Worked example:
    ///   var vocab = try Vocab.load(allocator, io, "my_vocab.tsv", 256, true);
    ///   defer vocab.deinit();
    pub fn load(allocator: std.mem.Allocator, io: std.Io, path: []const u8, max_vocab: usize, lowercase: bool) !Vocab {
        const cwd = std.Io.Dir.cwd();
        const file = try cwd.openFile(io, path, .{});
        defer file.close(io);
        var read_buf: [4096]u8 = undefined;
        var reader = file.reader(io, &read_buf);

        var self = init(allocator, max_vocab, lowercase);
        errdefer self.deinit();

        // Read lines: "ID<TAB>WORD\n"
        while (true) {
            const line = reader.interface.takeDelimiter('\n') catch break orelse break;
            if (line.len == 0) continue;

            const content = if (line.len > 0 and line[line.len - 1] == '\n') line[0 .. line.len - 1] else line;

            const tab_idx = std.mem.indexOfScalar(u8, content, '\t') orelse continue;
            const id_str = content[0..tab_idx];
            const word = content[tab_idx + 1 ..];

            const id: u32 = std.fmt.parseInt(u32, id_str, 10) catch continue;

            // Pad id_to_word with empty slots if IDs have gaps
            while (self.id_to_word.items.len < id) {
                try self.id_to_word.append(allocator, "");
                try self.counts.append(allocator, 0);
            }

            const owned = try allocator.dupe(u8, word);

            if (self.id_to_word.items.len == id) {
                try self.id_to_word.append(allocator, owned);
                try self.counts.append(allocator, 0);
            } else {
                const old = self.id_to_word.items[id];
                if (old.len > 0) allocator.free(self.id_to_word.items[id]);
                self.id_to_word.items[id] = owned;
            }

            try self.word_to_id.put(owned, id);
        }

        return self;
    }

    /// Free all owned memory.
    pub fn deinit(self: *Vocab) void {
        for (self.id_to_word.items) |word| {
            if (word.len > 0) self.allocator.free(word);
        }
        self.id_to_word.deinit(self.allocator);

        // word_to_id keys are the same pointers as id_to_word items,
        // so we must NOT free them separately. Just deinit the HashMap.
        self.word_to_id.deinit();

        self.counts.deinit(self.allocator);
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;
const tokenize = @import("word.zig").tokenize;

/// Helper: minimal tokenize function for testing Vocab in isolation.
/// Just splits on whitespace — no punctuation peeling.
fn simpleTokenize(allocator: std.mem.Allocator, text: []const u8, lowercase: bool) LabError!std.ArrayList([]const u8) {
    var list: std.ArrayList([]const u8) = .empty;
    errdefer list.deinit(allocator);

    var buf: [4096]u8 = undefined;
    const processed = if (lowercase) blk: {
        for (text, 0..) |ch, i| buf[i] = std.ascii.toLower(ch);
        break :blk buf[0..text.len];
    } else text;

    var it = std.mem.splitScalar(u8, processed, ' ');
    while (it.next()) |tok| {
        if (tok.len == 0) continue;
        const owned = try allocator.dupe(u8, tok);
        try list.append(allocator, owned);
    }
    return list;
}

/// Create a std.Io for use in tests.
/// Uses a Threaded Io with the testing allocator.
fn testIo() !std.Io {
    // We store the Threaded in a thread-local so it survives the test.
    const T = struct {
        var instance: ?std.Io.Threaded = null;
    };
    if (T.instance == null) {
        T.instance = std.Io.Threaded.init(std.testing.allocator, .{});
    }
    return T.instance.?.io();
}

test "Vocab.init — creates empty vocab" {
    const alloc = testing.allocator;
    var vocab = Vocab.init(alloc, 100, true);
    defer vocab.deinit();
    try testing.expectEqual(@as(usize, 0), vocab.size());
}

test "Vocab.buildFromText — populates vocab from simple text" {
    const alloc = testing.allocator;
    const text = "the cat sat on the mat";

    var vocab = try Vocab.buildFromText(alloc, text, 100, true, simpleTokenize);
    defer vocab.deinit();

    // 4 specials + 5 unique words: the, cat, sat, on, mat
    try testing.expectEqual(@as(usize, 9), vocab.size());
    try testing.expectEqual(@as(u32, 4), vocab.encodeWord("the"));
    try testing.expectEqual(@as(u32, 0), vocab.encodeWord("unknown"));
}

test "Vocab.buildFromText — frequency cutoff works" {
    const alloc = testing.allocator;
    const text = "the the the cat cat sat on mat";

    var vocab = try Vocab.buildFromText(alloc, text, 7, true, simpleTokenize);
    defer vocab.deinit();

    try testing.expectEqual(@as(usize, 7), vocab.size());
    try testing.expect(vocab.encodeWord("the") >= NUM_SPECIALS);
    try testing.expect(vocab.encodeWord("cat") >= NUM_SPECIALS);
}

test "Vocab.encodeWord — returns UNK for unknown words" {
    const alloc = testing.allocator;
    const text = "hello world";
    var vocab = try Vocab.buildFromText(alloc, text, 100, true, simpleTokenize);
    defer vocab.deinit();

    try testing.expectEqual(@as(u32, UNK_ID), vocab.encodeWord("nonexistent"));
    try testing.expectEqual(@as(u32, UNK_ID), vocab.encodeWord(""));
}

test "Vocab.decodeId — round-trips with encodeWord" {
    const alloc = testing.allocator;
    const text = "the cat sat";
    var vocab = try Vocab.buildFromText(alloc, text, 100, true, simpleTokenize);
    defer vocab.deinit();

    const id = vocab.encodeWord("the");
    const word = vocab.decodeId(id);
    try testing.expectEqualStrings("the", word);
}

test "Vocab.decodeId — out-of-range returns <unk>" {
    const alloc = testing.allocator;
    const text = "hello";
    var vocab = try Vocab.buildFromText(alloc, text, 100, true, simpleTokenize);
    defer vocab.deinit();

    const word = vocab.decodeId(9999);
    try testing.expectEqualStrings(UNK_TOKEN, word);
}

test "Vocab.isSpecial — identifies all four specials" {
    try testing.expect(Vocab.isSpecial("<unk>"));
    try testing.expect(Vocab.isSpecial("<pad>"));
    try testing.expect(Vocab.isSpecial("<bos>"));
    try testing.expect(Vocab.isSpecial("<eos>"));
    try testing.expect(!Vocab.isSpecial("the"));
    try testing.expect(!Vocab.isSpecial(""));
}

test "Vocab save/load round-trip" {
    const alloc = testing.allocator;
    const io = try testIo();
    const text = "the cat sat on the mat";
    var vocab1 = try Vocab.buildFromText(alloc, text, 100, true, simpleTokenize);
    defer vocab1.deinit();

    try vocab1.save(io, "/tmp/test_vocab.tsv");

    var vocab2 = try Vocab.load(alloc, io, "/tmp/test_vocab.tsv", 100, true);
    defer vocab2.deinit();

    try testing.expectEqual(vocab1.size(), vocab2.size());

    for (0..vocab1.size()) |id_raw| {
        const id: u32 = @intCast(id_raw);
        try testing.expectEqualStrings(vocab1.decodeId(id), vocab2.decodeId(id));
    }
}

test "Vocab.buildFromText — lowercase flag works" {
    const alloc = testing.allocator;
    const text = "Hello HELLO hello";

    var vocab = try Vocab.buildFromText(alloc, text, 100, true, simpleTokenize);
    defer vocab.deinit();

    // 4 specials + 1 word ("hello" after lowercasing all three)
    try testing.expectEqual(@as(usize, 5), vocab.size());
    try testing.expect(vocab.encodeWord("hello") >= NUM_SPECIALS);
}

test "Vocab.buildFromText — with real tokenizer (punctuation peeling)" {
    const alloc = testing.allocator;
    const text = "hello, world!";

    var vocab = try Vocab.buildFromText(alloc, text, 100, true, tokenize);
    defer vocab.deinit();

    // 4 specials + "hello" + "," + "world" + "!" = 8
    try testing.expectEqual(@as(usize, 8), vocab.size());
}
