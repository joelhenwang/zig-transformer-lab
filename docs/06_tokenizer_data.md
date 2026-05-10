# Chapter 06: Tokenizer & Data Pipeline

> **Stage 5 source files:** `src/tokenizer/vocab.zig`, `src/tokenizer/word.zig`,
> `src/data/dataset.zig`, `src/data/windowing.zig`, `src/data/batcher.zig`
>
> **What you'll learn:** Why we break text into tokens, how a word-level
> tokenizer works, how to build a vocabulary from raw text, how sliding-window
> pairs feed next-token prediction, and how deterministic batching ensures
> reproducible training runs.

---

## Table of Contents

1. [Why Tokenize at All?](#1-why-tokenize-at-all)
2. [Why Word-Level?](#2-why-word-level)
3. [Special Tokens](#3-special-tokens)
4. [The Vocab Struct](#4-the-vocab-struct)
5. [Frequency Cutoff](#5-frequency-cutoff)
6. [Tokenization Algorithm](#6-tokenization-algorithm)
7. [Punctuation Peeling — Worked Examples](#7-punctuation-peeling--worked-examples)
8. [Apostrophe Splitting](#8-apostrophe-splitting)
9. [Encoding: Text → IDs](#9-encoding-text--ids)
10. [Decoding: IDs → Text](#10-decoding-ids--text)
11. [The Round-Trip Property](#11-the-round-trip-property)
12. [Vocab Serialization Format](#12-vocab-serialization-format)
13. [Dataset: File → Token Stream](#13-dataset-file--token-stream)
14. [Windowing for Next-Token Prediction](#14-windowing-for-next-token-prediction)
15. [Batching: Shuffle → Group → Drop Last](#15-batching-shuffle--group--drop-last)
16. [Fisher-Yates Shuffle](#16-fisher-yates-shuffle)
17. [Shape Trace: Text → Model](#17-shape-trace-text--model)
18. [PyTorch Equivalents](#18-pytorch-equivalents)
19. [Common Mistakes](#19-common-mistakes)

---

## 1. Why Tokenize at All?

Neural networks don't read text. They read **numbers** — specifically, vectors of
floating-point values. A transformer expects a sequence of integer IDs, where each
ID indexes into an embedding table:

```
Text:    "the cat sat"
          ↓ tokenize
IDs:     [5, 12, 89]
          ↓ embed
Vectors: [0.1, -0.3, ...], [0.7, 0.2, ...], [0.0, 0.5, ...]
```

The **tokenizer** is the bridge between human-readable text and machine-readable
integers. Without it, there's no way to feed a string into a neural network.

But *how* you break text into units — the **tokenization strategy** — profoundly
affects:

| Aspect | Character-level | Word-level | BPE/Subword |
|--------|----------------|------------|-------------|
| Vocab size | ~100 | ~10K–100K | ~32K |
| Sequence length | Long | Short | Medium |
| Unknown words | Never | Common | Rare |
| Semantic units | Poor | Good | Good |
| Implementation | Trivial | Simple | Complex |

This project uses **word-level** tokenization. The next section explains why.

---

## 2. Why Word-Level?

Our project is **pedagogical** — its goal is to teach how transformers work, not to
achieve state-of-the-art perplexity. Word-level tokenization is the simplest
strategy that still produces semantically meaningful units.

### Advantages for learning

1. **Intuitive.** "cat" maps to one token. You can inspect the embedding table
   and say "row 5 is the word *cat*." With BPE, "cat" might be `c` + `at`, and
   row 5 corresponds to the subword `at` — much harder to reason about.

2. **Shorter sequences.** A 100-word sentence produces 100 tokens (word-level)
   vs. ~300 tokens (character-level). Shorter sequences mean faster training
   on CPU, which matters for a teaching library.

3. **Transparent vocabulary.** You can print the vocab, grep for a word, and
   count frequencies. No merge tables, no byte-level tricks.

### The trade-off: unknown words

Word-level tokenization produces a large vocabulary. If you cap it at `V` words
(e.g., V=256 for a tiny model), any word not in the top `V-4` by frequency
becomes `<unk>`. This is fine for small, controlled corpora like `tiny.txt`
but would be disastrous for open-domain text.

For a production model, you'd use BPE (GPT-2) or SentencePiece (LLaMA). But
understanding word-level tokenization first makes those advanced methods
much easier to learn.

---

## 3. Special Tokens

Every vocabulary reserves a few IDs for special purposes:

| ID | Token | Purpose |
|----|-------|---------|
| 0 | `<unk>` | Unknown word — any word not in the vocabulary |
| 1 | `<pad>` | Padding — unused in this project (future: variable-length sequences) |
| 2 | `<bos>` | Beginning of sequence — marks the start (unused in training, reserved) |
| 3 | `<eos>` | End of sequence — marks the end (unused in training, reserved) |

**Why reserve IDs starting from 0?** The embedding table has `V` rows. Row 0 is
the embedding for `<unk>`, row 1 for `<pad>`, etc. Real words start at ID 4.
This convention ensures that common words always have the same ID regardless of
the corpus.

```
Embedding table (V=256):
  Row 0:   <unk> → [0.0, 0.0, ...]    (initialized small)
  Row 1:   <pad> → [0.0, 0.0, ...]    (usually zero)
  Row 2:   <bos> → [0.1, -0.2, ...]
  Row 3:   <eos> → [-0.1, 0.3, ...]
  Row 4:   "the"  → [0.5, -0.1, ...]  ← first real word
  Row 5:   "cat"  → [0.2, 0.4, ...]
  ...
  Row 255: "xylophone" → [...]
```

---

## 4. The Vocab Struct

The `Vocab` struct in `src/tokenizer/vocab.zig` is the heart of the tokenizer.
It maintains two-way mappings:

```
Vocab {
    word_to_id: HashMap([]const u8, u32)   // "cat" → 5
    id_to_word: ArrayList([]const u8)       // [5] → "cat"
    freq:        HashMap([]const u8, u32)  // "cat" → 47 (count in corpus)
}
```

### Construction

The vocab is built in three phases:

**Phase 1 — Count.** Scan the entire corpus, splitting on whitespace. For each
word, increment its frequency counter.

**Phase 2 — Cutoff.** Sort words by frequency (descending). Keep the top
`max_vocab - 4` words (reserving 4 slots for special tokens). All remaining
words map to `<unk>`.

**Phase 3 — Assign IDs.** Special tokens get IDs 0–3. Real words get IDs
starting at 4, in frequency order (most common word = ID 4).

```
Corpus: "the cat sat on the mat the cat ate"

Phase 1 — frequencies:
  "the" → 3, "cat" → 2, "sat" → 1, "on" → 1, "mat" → 1, "ate" → 1

Phase 2 — cutoff (max_vocab = 8, keep top 4):
  Keep: "the"(3), "cat"(2), "sat"(1), "on"(1)
  Drop: "mat"(1), "ate"(1)  → map to <unk>

Phase 3 — assign IDs:
  0: <unk>, 1: <pad>, 2: <bos>, 3: <eos>
  4: "the", 5: "cat", 6: "sat", 7: "on"
```

### Key methods

```zig
// Look up a word's ID. Returns null if not in vocab (caller decides <unk>).
pub fn encodeWord(self: *const Vocab, word: []const u8) ?u32

// Look up a word by ID. Returns null for out-of-range IDs.
pub fn decodeId(self: *const Vocab, id: u32) ?[]const u8

// Total number of entries including special tokens.
pub fn size(self: *const Vocab) u32
```

### Memory ownership

Words stored in the HashMaps and ArrayList are **slices into the original text
buffer**. The `Vocab` does not copy word strings. This means the Vocab is valid
only as long as the underlying text buffer lives. In practice, the `Dataset`
owns the text buffer and the Vocab, and they share the same lifetime.

---

## 5. Frequency Cutoff

Why do we cut off rare words?

1. **Memory.** The embedding table is `V × D` floats. With D=64 and V=100,000,
   that's 25 MB just for one weight matrix. For a tiny teaching model, we want
   V ≈ 256.

2. **Learning signal.** A word that appears once in the entire corpus provides
   almost no gradient signal. The model can't learn a meaningful embedding from
   a single occurrence. Better to fold it into `<unk>`.

3. **Generalization.** If "xylophone" appears once and gets its own embedding,
   the model memorizes that single occurrence rather than learning general
   patterns. `<unk>` forces the model to rely on context.

### How the cutoff works

```zig
// Pseudocode for the cutoff logic:
const num_real = max_vocab - 4;  // reserve 4 special tokens
// Sort words by frequency, descending
// Take the top num_real words
// Everything else → encodeWord returns null → caller maps to <unk>=0
```

### Example with ties

When multiple words share the same frequency, the tie is broken by **lexicographic
order** (alphabetical). This ensures deterministic vocab construction across runs:

```
Frequencies: "mat"→1, "ate"→1, "sat"→1, "on"→1
Sorted:      "ate"(1), "mat"(1), "on"(1), "sat"(1)
               ↑ alphabetical first
```

---

## 6. Tokenization Algorithm

Our word tokenizer processes text in two passes:

```
┌──────────────────────────────────────────────────┐
│  Input: "Hello, world! Don't panic."             │
│                                                  │
│  Step 1: Whitespace split                        │
│    → ["Hello,", "world!", "Don't", "panic."]     │
│                                                  │
│  Step 2: Punctuation peeling (per token)         │
│    → ["Hello", ",", "world", "!", "Don't",       │
│       "panic", "."]                              │
│                                                  │
│  Step 3: Apostrophe splitting (per token)        │
│    → ["Hello", ",", "world", "!", "Don",         │
│       "'", "t", "panic", "."]                    │
│                                                  │
│  Output: 9 tokens                                │
└──────────────────────────────────────────────────┘
```

### Step 1: Whitespace split

Split the input on any whitespace character (space, tab, newline). This is
the coarsest split — it produces one "token" per whitespace-delimited chunk.

```zig
// Zig pseudocode:
var it = std.mem.splitSequence(u8, text, " ");
while (it.next()) |chunk| {
    // process chunk
}
```

Consecutive whitespace is collapsed: `"hello  world"` still produces `["hello", "world"]`.

### Step 2: Punctuation peeling

Each chunk from Step 1 may contain leading/trailing punctuation. We peel it off
so that `.` is its own token, not glued to the preceding word.

### Step 3: Apostrophe splitting

After punctuation peeling, some tokens still contain apostrophes (e.g., `"Don't"`).
We split these into their component parts.

The following sections cover Steps 2 and 3 in detail.

---

## 7. Punctuation Peeling — Worked Examples

We peel **10 punctuation characters** from both sides of each whitespace-delimited
chunk:

```
Peel set: . , ! ? ; : " ' ( )
```

The algorithm is:

```
function peelPunctuation(chunk):
    tokens = []
    // Peel leading punctuation
    while chunk starts with a peel character:
        tokens.append(chunk[0])
        chunk = chunk[1:]
    // Now chunk has no leading punctuation
    if chunk is empty:
        return tokens  // was all punctuation, e.g. "..."
    // Peel trailing punctuation
    while chunk ends with a peel character:
        trailing.append(chunk[last])
        chunk = chunk[0..last]
    tokens.append(chunk)  // the word core
    // Append trailing punctuation in original order
    tokens.append(trailing...)
    return tokens
```

### Example 1: Simple trailing period

```
Input chunk: "panic."
Leading peel: none (starts with 'p')
Trailing peel: "." ← peel 'p', 'a', 'n', 'i', 'c', '.'
  → word core: "panic"
  → trailing: ["."]
Result: ["panic", "."]
```

### Example 2: Multiple trailing punctuation

```
Input chunk: "What?!"
Leading peel: none
Trailing peel: first '!', then '?' (right-to-left peeling)
  → word core: "What"
  → trailing: ["?", "!"]
Result: ["What", "?", "!"]
```

### Example 3: Leading punctuation

```
Input chunk: "(hello"
Leading peel: "("
  → remaining: "hello"
Trailing peel: none
Result: ["(", "hello"]
```

### Example 4: Both sides

```
Input chunk: "'Hello,'"
Leading peel: "'" → remaining: "Hello,'"
              "'" → remaining: "Hello,"
Trailing peel: "," → remaining: "Hello"
Result: ["'", "'", "Hello", ",", "'"]
```

### Example 5: Ellipsis (all punctuation)

```
Input chunk: "..."
Leading peel: "." → ".."
              "." → "."
              "." → ""
Word core: "" (empty)
Trailing: none (nothing left)
Result: [".", ".", "."]
```

### Example 6: Quoted speech

```
Input chunk: '"Hello"'
Leading peel: '"' → remaining: 'Hello"'
Trailing peel: '"' → remaining: "Hello"
Result: ['"', "Hello", '"']
```

### Why peel instead of split?

You might wonder: why not just split on punctuation characters everywhere, not
just at boundaries? Because **internal punctuation** (except apostrophes) is
extremely rare in English text. Peeling at boundaries handles 99.9% of cases and
keeps the algorithm simple. The apostrophe is handled separately because it
occurs *inside* words ("don't", "it's") and has specific splitting rules.

---

## 8. Apostrophe Splitting

After punctuation peeling, tokens like `"Don't"` still contain an apostrophe.
We split on the apostrophe character, producing three sub-tokens:

```
"Don't" → "Don" + "'" + "t"
```

### The algorithm

```
function splitApostrophe(token):
    if token contains no apostrophe:
        return [token]
    
    parts = split token on "'"
    result = []
    for i, part in parts:
        if part is not empty:
            result.append(part)
        if i is not the last part:
            result.append("'")   // the apostrophe itself
    
    return result
```

### Worked examples

| Input | After split | Output tokens |
|-------|-------------|--------------|
| `"don't"` | `["don", "t"]` | `["don", "'", "t"]` |
| `"cat's"` | `["cat", "s"]` | `["cat", "'", "s"]` |
| `"it's"` | `["it", "s"]` | `["it", "'", "s"]` |
| `"I'm"` | `["I", "m"]` | `["I", "'", "m"]` |
| `"we'll"` | `["we", "ll"]` | `["we", "'", "ll"]` |
| `"they're"` | `["they", "re"]` | `["they", "'", "re"]` |
| `"o'clock"` | `["o", "clock"]` | `["o", "'", "clock"]` |
| `"can't"` | `["can", "t"]` | `["can", "'", "t"]` |
| `"won't"` | `["won", "t"]` | `["won", "'", "t"]` |
| `"hello"` | (no apostrophe) | `["hello"]` |
| `"'"` | `["", ""]` | `["'"]` (empty parts skipped) |

### Why split contractions?

You might ask: why not treat `"don't"` as a single token? Two reasons:

1. **Vocabulary size.** If `"don't"`, `"doesn't"`, `"didn't"`, `"can't"`,
   `"couldn't"`, `"won't"`, `"wouldn't"`, `"shouldn't"`, `"isn't"`,
   `"aren't"`, `"wasn't"`, `"weren't"`, `"hasn't"`, `"haven't"`,
   `"hadn't"` are each their own token, that's 15 vocabulary slots for
   negation contractions alone. Splitting lets `"not"` share one embedding
   (well, `"t"` is a morpheme here, but it's still better than 15 unique tokens).

2. **Generalization.** The model learns that `"'"` + `"t"` is a negation pattern.
   When it encounters `"shan't"` (rare), the `"t"` embedding already encodes
   negation. If `"shan't"` were its own token with ID 0 (`<unk>`), the model
   would learn nothing.

### Limitation

The split `"don't" → ["don", "'", "t"]` is not a morphological analysis. `"don"`
is not a word — it's the stem *"do"* without the *"o"*. A proper morphological
tokenizer would produce `["do", "n't"]`. But morphological analysis requires a
rule engine or lookup table, which is far beyond the scope of a teaching project.
Our simple apostrophe split is a reasonable approximation.

---

## 9. Encoding: Text → IDs

The `encode` function combines all three steps (whitespace split → punctuation
peeling → apostrophe splitting) and maps each resulting token to its ID:

```
┌──────────────────────────────────────────────────────────────┐
│  encode("Hello, world! Don't panic.", vocab)                 │
│                                                              │
│  Step 1: Whitespace split                                    │
│    ["Hello,", "world!", "Don't", "panic."]                    │
│                                                              │
│  Step 2: Punctuation peeling                                  │
│    ["Hello", ",", "world", "!", "Don't", "panic", "."]       │
│                                                              │
│  Step 3: Apostrophe splitting                                 │
│    ["Hello", ",", "world", "!", "Don", "'", "t",             │
│     "panic", "."]                                            │
│                                                              │
│  Step 4: Map to IDs                                          │
│    "Hello" → vocab.encodeWord("Hello") → 42                  │
│    ","     → vocab.encodeWord(",")     → 7                   │
│    "world" → vocab.encodeWord("world") → 89                  │
│    "!"     → vocab.encodeWord("!")     → 12                  │
│    "Don"   → vocab.encodeWord("Don")   → 156                 │
│    "'"     → vocab.encodeWord("'")     → 8                   │
│    "t"     → vocab.encodeWord("t")     → 34                  │
│    "panic" → vocab.encodeWord("panic") → 201                 │
│    "."     → vocab.encodeWord(".")     → 6                   │
│                                                              │
│  Result: [42, 7, 89, 12, 156, 8, 34, 201, 6]                 │
└──────────────────────────────────────────────────────────────┘
```

### Unknown words

If `vocab.encodeWord(word)` returns `null`, we use ID 0 (`<unk>`):

```zig
const id = vocab.encodeWord(word) orelse 0;  // 0 = <unk>
```

In the example above, if "Don" is not in the vocabulary (because the training
corpus only had "don" lowercase), it becomes `<unk>=0`:

```
Result with unknown "Don": [42, 7, 89, 12, 0, 8, 34, 201, 6]
                                    ^^^
                                    <unk>
```

This is a real issue with case sensitivity. Our tokenizer is **case-sensitive**:
"the" and "The" are different tokens. For a tiny model, you often want to
lowercase the entire corpus before building the vocab. The `Dataset` handles
this by providing a `lowercase` flag.

---

## 10. Decoding: IDs → Text

The `decode` function reverses the encoding: given a slice of IDs, produce a
human-readable string.

```zig
pub fn decode(vocab: *const Vocab, ids: []const u32, allocator: Allocator) ![]u8
```

The algorithm:

1. For each ID, look up the word via `id_to_word`.
2. Join words with a single space between them.
3. Special tokens are decoded by their string representation (`"<unk>"`,
   `"<pad>"`, etc.).

```
IDs:    [42, 7, 89, 12, 156, 8, 34, 201, 6]
Words:  ["Hello", ",", "world", "!", "Don", "'", "t", "panic", "."]
Joined: "Hello , world ! Don ' t panic ."
```

Notice the spaces around punctuation. This is different from the original text
(`"Hello, world! Don't panic."`). The next section explains why.

---

## 11. The Round-Trip Property

Encoding followed by decoding does **not** reproduce the original text exactly:

```
Original:  "Hello, world! Don't panic."
Encode:    [42, 7, 89, 12, 156, 8, 34, 201, 6]
Decode:    "Hello , world ! Don ' t panic ."
                       ^   ^      ^ ^       ^
              spaces inserted around punctuation
```

The differences are always in **whitespace**:

| Original | Decoded | Difference |
|----------|---------|------------|
| `Hello,` | `Hello ,` | Space before comma |
| `world!` | `world !` | Space before exclamation |
| `Don't` | `Don ' t` | Spaces around apostrophe |
| `panic.` | `panic .` | Space before period |

This is acceptable because:

1. **The model never sees whitespace.** It only sees token IDs. Whether the
   comma was adjacent to "Hello" or separated by a space is invisible to the
   model.

2. **Information is preserved.** All semantic content — the words and their
   punctuation — is retained. Only formatting is lost.

3. **The alternative is much more complex.** To reconstruct exact whitespace,
   you'd need to store whitespace tokens or use a detokenizer with heuristics
   (e.g., "don't insert space before a comma"). That's a production concern,
   not a pedagogical one.

### Formal statement

For any text `T`:

```
decode(encode(T)) ≠ T           (whitespace differs)
tokenize(decode(encode(T))) = tokenize(T)  (token sequence is identical)
```

The second property ensures that re-encoding the decoded text produces the
same token IDs. This is the **round-trip property**: tokenization is stable.

---

## 12. Vocab Serialization Format

The vocab is serialized as a simple text file, one line per entry:

```
0\t<unk>
1\t<pad>
2\t<bos>
3\t<eos>
4\tthe
5\tcat
6\tsat
7\ton
...
```

Each line is `ID<TAB>WORD`. The file is sorted by ID (ascending).

### Why this format?

1. **Human-readable.** You can `cat vocab.txt` and immediately see the
   vocabulary.

2. **No binary dependencies.** No protobuf, no MessagePack, no JSON parser.
   Just line-by-line text parsing.

3. **Deterministic.** Sorted by ID, the file is identical regardless of
   HashMap iteration order.

### Deserialization

```zig
pub fn loadFromFile(path: []const u8, allocator: Allocator) !Vocab
```

Reads the file line by line. For each line:

1. Split on `\t`.
2. Parse the first field as a `u32` (the ID).
3. The second field is the word string (allocated/copied).
4. Insert into `word_to_id` and `id_to_word`.

### Error handling

- If a line has no tab character → `error.InvalidFormat`.
- If the ID field doesn't parse as a number → `error.InvalidFormat`.
- If IDs are not sequential starting from 0 → `error.InvalidFormat`.
- If the file can't be opened → propagated from `std.Io.Dir.cwd().openFile`.

---

## 13. Dataset: File → Token Stream

The `Dataset` struct in `src/data/dataset.zig` owns the end-to-end pipeline
from raw text to token IDs.

```
┌─────────────────────────────────────────────────────┐
│  Dataset                                            │
│                                                     │
│  ┌───────────┐    ┌──────────┐    ┌───────────────┐ │
│  │  raw text  │ → │ tokenize  │ → │ encode w/     │ │
│  │  (file)    │    │          │    │ vocab         │ │
│  └───────────┘    └──────────┘    └───────────────┘ │
│       ↓                               ↓             │
│  text: []const u8              ids: []u32           │
│       ↓                                              │
│  vocab: Vocab (built from text)                     │
└─────────────────────────────────────────────────────┘
```

### Construction

```zig
pub fn init(allocator: Allocator, path: []const u8, max_vocab: u32, lowercase: bool) !Dataset
```

Steps:

1. Read the entire file into a buffer.
2. Optionally lowercase the buffer (for case-insensitive vocabularies).
3. Tokenize the text (whitespace → punctuation → apostrophe).
4. Count word frequencies.
5. Build the Vocab (frequency cutoff → ID assignment).
6. Encode all tokens into `[]u32`.

### Key fields

```zig
pub const Dataset = struct {
    text: []const u8,          // owned buffer
    tokens: [][]const u8,      // slices into text
    ids: []u32,                // encoded token IDs
    vocab: Vocab,              // word↔ID mapping
    allocator: Allocator,
};
```

### Example: tiny.txt

Suppose `data/tiny.txt` contains:

```
the cat sat on the mat
the cat ate the mat
a dog sat on the mat
```

After `Dataset.init`:

```
text:    "the cat sat on the mat\nthe cat ate the mat\na dog sat on the mat"
tokens:  ["the", "cat", "sat", "on", "the", "mat", "the", "cat", "ate",
          "the", "mat", "a", "dog", "sat", "on", "the", "mat"]
ids:     [4, 5, 6, 7, 4, 8, 4, 5, 9, 4, 8, 10, 11, 6, 7, 4, 8]
vocab:   {0:<unk>, 1:<pad>, 2:<bos>, 3:<eos>, 4:the, 5:cat,
          6:sat, 7:on, 8:mat, 9:ate, 10:a, 11:dog}
```

Note: with `max_vocab=16`, we keep 12 real words (16 - 4 special). All words
in this tiny corpus fit, so no `<unk>` tokens appear.

---

## 14. Windowing for Next-Token Prediction

A transformer trained with cross-entropy loss learns to predict the **next
token** given the preceding context. The `windowing` module creates
(input, target) pairs from a flat token stream.

### The core idea

Given a sequence of token IDs:

```
IDs: [4, 5, 6, 7, 4, 8, 4, 5, 9, 4, 8, 10, 11, 6, 7, 4, 8]
      t0 t1 t2 t3 t4 t5 t6 t7 t8 t9 ...
```

A window of length `T` (context length) produces:

```
Input:  [t0, t1, t2, ..., t_{T-1}]
Target: [t1, t2, t3, ..., t_T]
```

The target is the input **shifted by one position**. At each position `i`,
the model learns: "given tokens up to position `i`, predict the token at
position `i+1`."

### Diagram

```
Token stream:   4  5  6  7  4  8  4  5  9  4  8  10 11 6  7  4  8
                │  │  │  │  │  │  │  │
Window 0:       ┌──┬──┬──┬──┬──┬──┐
  Input:        │ 4│ 5│ 6│ 7│ 4│ 8│   (positions 0-5)
  Target:       │ 5│ 6│ 7│ 4│ 8│ 4│   (positions 1-6)
                └──┴──┴──┴──┴──┴──┘
                                    │  │  │  │  │  │  │
Window 1:                            ┌──┬──┬──┬──┬──┬──┐
  Input:                             │ 5│ 9│ 4│ 8│10│11│   (positions 6-11)
  Target:                            │ 9│ 4│ 8│10│11│ 6│   (positions 7-12)
                                     └──┴──┴──┴──┴──┴──┘
```

Wait — that's wrong. Let me redo this with **non-overlapping windows** for
clarity. Our windowing uses a **stride** of `T`, so windows don't overlap:

```
Token stream:  [4, 5, 6, 7, 4, 8, 4, 5, 9, 4, 8, 10, 11, 6, 7, 4, 8]
                 ├─ window 0 ─┤                ├─ window 1 ─┤

With T=6 (context length):

Window 0:
  Input:  [4, 5, 6, 7, 4, 8]    positions 0..5
  Target: [5, 6, 7, 4, 8, 4]    positions 1..6

Window 1:
  Input:  [5, 9, 4, 8, 10, 11]  positions 6..11
  Target: [9, 4, 8, 10, 11, 6]  positions 7..12

Window 2:
  Input:  [6, 7, 4, 8]          positions 12..16 (length 5 < T=6)
  → DROPPED (too short)
```

Wait, that's still not right. Let me be precise.

The windowing module creates windows by sliding with stride `T`:

```
For window index w:
  start = w * T
  input  = ids[start .. start + T]
  target = ids[start + 1 .. start + T + 1]
```

This means we need `T + 1` consecutive tokens for each window. The last token
of `target` is at position `start + T`.

```
Total tokens: 17 (IDs [0..16])
T = 6

Window 0: needs positions 0..6 (7 tokens) ✓
  Input:  [4, 5, 6, 7, 4, 8]     ids[0..6]
  Target: [5, 6, 7, 4, 8, 4]     ids[1..7]

Window 1: needs positions 6..12 (7 tokens) ✓
  Input:  [5, 9, 4, 8, 10, 11]   ids[6..12]
  Target: [9, 4, 8, 10, 11, 6]   ids[7..13]

Window 2: needs positions 12..18 (7 tokens) ✗ (only 5 available)
  → Dropped
```

### The Window struct

```zig
pub const Window = struct {
    input: []const u32,   // length T
    target: []const u32,  // length T
};
```

Both slices point into the dataset's `ids` array — no copying.

### Why non-overlapping?

Overlapping windows (stride 1) would produce `N - T` windows from `N` tokens,
giving the model many more training examples. But:

1. **Overfitting.** Adjacent windows share `T - 1` tokens. The model sees
   almost the same input twice. For a tiny corpus, this amplifies overfitting.

2. **Simplicity.** Non-overlapping windows are trivial to understand and
   implement. Overlapping windows require a stride parameter and complicate
   epoch counting.

3. **Epoch semantics.** One epoch = one pass through all non-overlapping
   windows. Clean and simple.

For a production model, you'd use overlapping windows with random starting
positions. But this is a teaching project — simplicity wins.

---

## 15. Batching: Shuffle → Group → Drop Last

The `Batcher` takes a list of windows and produces mini-batches for training.

### Three steps

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: Shuffle                                                │
│    Fisher-Yates shuffle on window indices                       │
│    → randomized order, but deterministic with same seed         │
│                                                                 │
│  Step 2: Group                                                  │
│    Reshape shuffled windows into batches of size B              │
│    batch[i] = windows[i*B .. i*B + B]                           │
│                                                                 │
│  Step 3: Drop last                                              │
│    If num_windows is not divisible by B, drop the final partial  │
│    batch. This avoids irregular batch sizes that complicate     │
│    GPU kernels (and our tensor code assumes fixed shapes).      │
└─────────────────────────────────────────────────────────────────┘
```

### The Batch struct

```zig
pub const Batch = struct {
    inputs: []const u32,   // shape: (B * T), row-major
    targets: []const u32,  // shape: (B * T), row-major
};
```

Inputs and targets are stored as flat arrays. The caller reshapes them into
`(B, T)` tensors before feeding to the model.

### Example

```
8 windows, B=3:

Step 1: Shuffle (example permutation):
  [3, 7, 1, 5, 0, 2, 6, 4]

Step 2: Group into batches of 3:
  Batch 0: windows [3, 7, 1]
  Batch 1: windows [5, 0, 2]
  Batch 2: windows [6, 4]     ← only 2 windows, incomplete

Step 3: Drop last:
  Batch 2 is dropped.
  Final: 2 batches of 3 windows each.

  8 windows → 2 batches × 3 = 6 windows used, 2 dropped.
```

### Why drop last?

1. **Fixed tensor shapes.** Our `Tensor` struct requires known dimensions at
   creation time. A partial batch (2 windows instead of 3) would need a
   different-shaped tensor. Supporting variable batch sizes adds complexity
   for negligible benefit.

2. **Batch normalization and statistics.** Some layers (not in our model, but
   in general) compute statistics across the batch dimension. A partial batch
   has different statistics, which can destabilize training.

3. **Deterministic shapes.** With drop-last, every batch has exactly `B`
   windows. The model always processes inputs of shape `(B, T)`. This makes
   the training loop simple and predictable.

### How many windows are wasted?

```
dropped = num_windows % B
```

With `B=2` and `T=8` on `tiny.txt` (~1000 tokens), we get ~125 windows and
drop at most 1. The waste is negligible for any reasonable corpus size.

---

## 16. Fisher-Yates Shuffle

The Fisher-Yates algorithm produces a **uniform random permutation** — every
possible ordering of `N` items is equally likely. It's the gold standard for
shuffling.

### Algorithm

```
for i from N-1 down to 1:
    j = random integer in [0, i]
    swap array[i] and array[j]
```

### Why not `rand.shuffle()`?

Zig's standard library may provide a shuffle function, but implementing
Fisher-Yates ourselves serves two purposes:

1. **Pedagogical.** The student can see exactly how the shuffle works, step
   by step, with comments explaining each line.

2. **Deterministic.** We use `std.Random.DefaultPrng` (Xoshiro256) with a
   fixed seed. Two runs with the same seed produce the same permutation.
   This makes training runs reproducible, which is critical for debugging
   and for the project's D14 (deterministic runs).

### Worked example

```
Array: [0, 1, 2, 3, 4]    (window indices)
Seed:  42

i=4: j in [0,4] → j=3    swap [4]↔[3]  → [0, 1, 2, 4, 3]
i=3: j in [0,3] → j=1    swap [3]↔[1]  → [0, 4, 2, 1, 3]
i=2: j in [0,2] → j=0    swap [2]↔[0]  → [2, 4, 0, 1, 3]
i=1: j in [0,1] → j=0    swap [1]↔[0]  → [4, 2, 0, 1, 3]

Result: [4, 2, 0, 1, 3]
```

### Properties

- **Uniform.** Every permutation of `N` elements has probability `1/N!`.
- **In-place.** Only O(1) extra memory.
- **O(N).** Exactly `N-1` swaps.
- **Deterministic.** Same seed → same shuffle.

---

## 17. Shape Trace: Text → Model

Let's trace the complete shape of data as it flows from a text file to the
model's forward pass.

```
┌─────────────────────────────────────────────────────────────────┐
│  1. Raw text                                                    │
│     "the cat sat on the mat\nthe dog ate the mat\n"             │
│     Shape: scalar string (no tensor shape)                      │
│                                                                 │
│  2. Tokens (after tokenization)                                 │
│     ["the", "cat", "sat", "on", "the", "mat", "the", "dog",    │
│      "ate", "the", "mat"]                                       │
│     Count: 11 tokens                                            │
│                                                                 │
│  3. IDs (after encoding)                                        │
│     [4, 5, 6, 7, 4, 8, 4, 11, 9, 4, 8]                        │
│     Shape: (11,) — 1D array                                     │
│                                                                 │
│  4. Windows (T=4)                                               │
│     Window 0: input=[4,5,6,7], target=[5,6,7,4]                │
│     Window 1: input=[4,8,4,11], target=[8,4,11,9]              │
│     Window 2: input=[4,8], target=[8]                           │
│       → Dropped (length < T)                                    │
│     Result: 2 windows                                           │
│     Shape per window: input (4,), target (4,)                   │
│                                                                 │
│  5. Batch (B=2)                                                 │
│     After shuffle + group:                                      │
│     inputs:  [[4,5,6,7], [4,8,4,11]]  → flat: [4,5,6,7,4,8,4,11]│
│     targets: [[5,6,7,4], [8,4,11,9]]  → flat: [5,6,7,4,8,4,11,9]│
│     Shape: (B*T,) = (8,) each, reshaped to (B, T) = (2, 4)    │
│                                                                 │
│  6. Model input tensors                                         │
│     input_ids:  Tensor shape (2, 4) dtype u32                   │
│     target_ids: Tensor shape (2, 4) dtype u32                   │
│                                                                 │
│  7. Embedding lookup (V=16, D=64)                               │
│     tok_embed(input_ids) → shape (2, 4, 64)                    │
│     pos_embed[0..T]         → shape (1, 4, 64) → broadcast     │
│     x = tok_embed + pos_embed → shape (2, 4, 64)              │
│                                                                 │
│  8. Transformer block                                            │
│     LayerNorm → Attn → +residual → LayerNorm → MLP → +residual │
│     Input:  (2, 4, 64)                                         │
│     Output: (2, 4, 64)                                          │
│                                                                 │
│  9. LM head                                                     │
│     ln_f(x) → shape (2, 4, 64)                                 │
│     lm_head(x) → shape (2, 4, V) = (2, 4, 16)                │
│     logits → shape (2, 4, 16)                                  │
│                                                                 │
│  10. Loss                                                       │
│     crossEntropy(logits, target_ids) → scalar loss              │
│     Shape: () — 0D scalar                                       │
└─────────────────────────────────────────────────────────────────┘
```

### Key insight: the data pipeline's job

The data pipeline's entire purpose is to produce two tensors of shape `(B, T)`:
`input_ids` (the context) and `target_ids` (the next-token labels). Everything
before step 6 is just preparation to fill these two arrays with the right
integers.

```
  ┌────────────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
  │  raw text file  │ ──→ │ tokenize │ ──→ │  encode  │ ──→ │  window  │
  │  (bytes on disk)│     │ (words)  │     │ (u32 IDs)│     │ (pairs)  │
  └────────────────┘     └──────────┘     └──────────┘     └──────────┘
                                                                │
                                                                ▼
                                                          ┌──────────┐
                                       input_ids (B, T) ← │  batch   │
                                       target_ids(B, T) ← │  shuffle │
                                                          └──────────┘
```

---

## 18. PyTorch Equivalents

If you're familiar with PyTorch, here's how our components map:

| Our code | PyTorch equivalent | Notes |
|---------|-------------------|-------|
| `Vocab` | `torchtext.vocab.Vocab` | Our version is simpler — no BPE, no byte fallback |
| `word.tokenize()` | `torchtext.data.utils.get_tokenizer("basic_english")` | PyTorch's basic_english tokenizer also lowercases and splits punctuation |
| `Vocab.encodeWord()` | `vocab['word']` or `vocab.lookup_indices(["word"])` | Same O(1) hash lookup |
| `Vocab.decodeId()` | `vocab.get_itos()[id]` or `vocab.lookup_tokens([id])` | Same array index |
| `Dataset` | `torchtext.datasets.TextClassificationDataset` | Ours is simpler — just token IDs, no label field |
| `windowing.createWindows()` | Custom `Dataset.__getitem__` | PyTorch doesn't have a built-in windowing utility |
| `Batcher` | `torch.utils.data.DataLoader` | DataLoader handles shuffling, batching, and drop_last |
| `Fisher-Yates shuffle` | `torch.randperm()` or `DataLoader(shuffle=True)` | Both use the same algorithm internally |
| `drop-last` | `DataLoader(drop_last=True)` | Identical behavior |
| `Xoshiro256 RNG` | `torch.Generator().manual_seed(seed)` | Both produce deterministic shuffles |

### Key difference: no `collate_fn`

PyTorch's DataLoader uses a `collate_fn` to merge individual samples into a
batch. Our Batcher directly creates flat arrays, which is simpler but less
flexible. If we needed variable-length sequences (which we don't — all windows
have length `T`), we'd need a padding-aware collate function.

### DataLoader pseudocode equivalent

```python
# PyTorch equivalent of our Batcher:
dataloader = DataLoader(
    dataset=window_dataset,   # our windowing output
    batch_size=B,             # our B
    shuffle=True,             # our Fisher-Yates shuffle
    drop_last=True,           # our drop-last
    generator=torch.Generator().manual_seed(42),  # our Xoshiro256 seed
)
```

---

## 19. Common Mistakes

### Mistake 1: Off-by-one in windowing

```
WRONG:
  input  = ids[start .. start + T]
  target = ids[start .. start + T]    ← same as input!

CORRECT:
  input  = ids[start .. start + T]
  target = ids[start + 1 .. start + T + 1]  ← shifted by 1
```

The target must be shifted by one position. If input and target are the same,
the model learns to copy the input instead of predicting the next token. This
is one of the most common bugs in transformer implementations — and it
doesn't crash, so it's hard to detect. The only symptom is that loss doesn't
decrease as expected.

**Debugging tip:** Print the first window's input and target. The target
should be the input shifted left by one, with a new token at the end.

### Mistake 2: Including special tokens in the training loss

```
WRONG:
  loss = crossEntropy(logits, target_ids)  // all positions contribute

CORRECT:
  // (Not an issue in our simple model, but in general:)
  // Ignore <pad> positions in the loss:
  loss = crossEntropy(logits, target_ids, ignore_index=1)
```

If `<pad>` tokens appear in the target (they shouldn't in our drop-last
scheme, but might with variable-length sequences), including them in the loss
teaches the model to predict padding — a waste of gradient signal.

### Mistake 3: Not lowercasing the corpus

```
WRONG:
  "The cat sat" → tokens: ["The", "cat", "sat"]
  "the dog ran" → tokens: ["the", "dog", "ran"]
  // "The" and "the" are different tokens!
  // With a tiny vocab, "The" might map to <unk>

CORRECT:
  Lowercase the entire corpus before tokenization.
  "the cat sat" → tokens: ["the", "cat", "sat"]
  "the dog ran" → tokens: ["the", "dog", "ran"]
  // Same "the" token, better embedding quality
```

For a tiny model (V=256), case sensitivity doubles the effective vocabulary
and wastes slots on capitalized variants. Always lowercase unless you have a
specific reason not to.

### Mistake 4: Forgetting that token IDs are u32, not f32

```
WRONG:
  var input_tensor = Tensor.init(allocator, Shape.init2D(B, T));
  // Fill with token IDs (u32 values reinterpreted as f32) — garbage!

CORRECT:
  // Token IDs are indices, not values.
  // They go through an embedding lookup, not direct tensor arithmetic.
  const embeddings = tok_embed.forward(input_ids, tape);  // input_ids is []const u32
  // The embedding layer gathers rows from the weight table using u32 indices.
```

Token IDs are **indices** into the embedding table. They are never used as
floating-point values. The embedding layer (which uses `@atomicLoad` or
gather operations) converts them to continuous vectors.

### Mistake 5: Shuffling every epoch without resetting the seed

```
WRONG:
  // Epoch 1: shuffle with seed 42
  // Epoch 2: shuffle with seed 42 again → same order!
  // The model sees windows in the same order every epoch.

CORRECT:
  // Option A: Advance the RNG state across epochs (natural with Xoshiro256)
  // Option B: Use epoch number as part of the seed: seed + epoch
  // Our Batcher creates a new RNG per call, so pass seed + epoch_number.
```

If the RNG is re-seeded to the same value each epoch, the model always sees
the same ordering, which reduces the diversity of gradient updates.

### Mistake 6: Building the vocab from the test set

```
WRONG:
  // Build vocab from ALL data, then split into train/test
  // → Test set words leak into vocab, inflating test performance

CORRECT:
  // Build vocab from training data only.
  // Test set words not in train vocab → <unk>
  // This measures true generalization.
```

This is called **data leakage**. The vocab is part of the model — it must be
built from training data only. For our pedagogical project, we train on the
full `tiny.txt`, so this isn't an issue. But it's a critical mistake in
production settings.

### Mistake 7: Using token IDs as array indices without bounds checking

```
WRONG:
  const embedding = weight_data[token_id * D .. (token_id + 1) * D];
  // If token_id >= V, this reads out of bounds — undefined behavior!

CORRECT:
  std.debug.assert(token_id < vocab.size());
  const embedding = weight_data[token_id * D .. (token_id + 1) * D];
```

In debug mode, Zig's slice bounds are checked. In release-safe, they're
checked too. In release-fast, they're not — and an out-of-bounds read
silently returns garbage. Always validate token IDs against the vocab size.

### Mistake 8: Dropping the wrong windows

```
WRONG:
  // Drop windows that contain <unk> tokens
  // → Reduces training data, introduces bias (rare words never learned)

WRONG:
  // Drop windows at the end that are "too short" — but miscount
  //   num_windows = ids.len / T;   ← off by one for the target shift!
  //   Need ids.len >= (T + 1) for the last window

CORRECT:
  // Drop windows that don't have enough tokens for the target shift:
  const max_start = ids.len - T;  // last valid start position
  // A window starting at position `start` needs ids[start + T] to exist
  // for the target. So start ranges from 0 to max_start (inclusive).
  const num_windows = ids.len / (T + 1);  // for non-overlapping windows
  // Wait, let me think again...
  //
  // For non-overlapping windows with stride T:
  //   Window w uses positions [w*T, w*T + T] — that's T+1 positions.
  //   Last valid w: (w+1)*T <= ids.len - 1
  //   → w <= (ids.len - 1) / T - 1
  //
  // Actually, the cleanest way:
  //   We need T + 1 consecutive tokens per window.
  //   With stride T, window w starts at w*T and needs position w*T + T.
  //   So we need w*T + T < ids.len, i.e., w < (ids.len - T) / T.
  //   num_windows = (ids.len - T) / T    (integer division, assuming ids.len > T)
  //
  // But this is for overlapping windows with stride T.
  // Our non-overlapping windows with stride T:
  //   num_windows = ids.len / T   (but each window needs T+1 tokens)
  //   Wait, that doesn't work either because the last window's target
  //   would go out of bounds.
  //
  // The correct formula for non-overlapping windows:
  //   Each window occupies T tokens of input and 1 extra token for the
  //   last target position. But windows share the boundary — window 0's
  //   last target is window 1's first input.
  //
  // With our implementation: input = ids[w*T..w*T+T], target = ids[w*T+1..w*T+T+1]
  //   We need w*T + T + 1 <= ids.len
  //   → w <= (ids.len - T - 1) / T
  //   → num_windows = (ids.len - 1) / T    (integer division)
  //   Hmm, that's still not right. Let me just be precise:
  //
  //   Window w needs ids[w*T + T] to exist.
  //   So w*T + T < ids.len → w < (ids.len - T) / T.
  //   For ids.len = 17, T = 6: w < (17-6)/6 = 11/6 = 1 → w = 0, 1 → 2 windows. ✓
```

The formula `num_windows = (ids.len - 1) / T` (integer division) gives the
number of complete non-overlapping windows where each window's target has all
T positions filled. Be careful — the off-by-one here is extremely easy to make.

---

## Appendix A: Complete Tokenization Walkthrough

Input text:

```
"I can't believe it," she said. "Really?"
```

**Step 1: Whitespace split**

```
['"I', "can't", 'believe', 'it,"', 'she', 'said.', '"Really?"']
```

**Step 2: Punctuation peeling** (per chunk)

```
'"I'       → ['"', "I"]              (leading '"' peeled)
"can't"    → ["can't"]               (no leading/trailing punctuation... yet)
             Wait — apostrophe IS in the peel set. But "can't" starts with 'c',
             not an apostrophe. And it ends with 't', not punctuation.
             So no leading or trailing peel. The apostrophe is internal.
"believe"  → ["believe"]
'it,"'     → ["it", ",", '"']        (trailing ',', then '"' peeled)
"she"      → ["she"]
'said.'    → ["said", "."]            (trailing '.' peeled)
'"Really?"' → ['"', "Really", "?", '"']  (leading '"', trailing '?', then '"' peeled)
```

**Step 3: Apostrophe splitting** (per token from Step 2)

```
'"'       → ['"']                    (no apostrophe inside)
"I"       → ["I"]                    (no apostrophe inside)
"can't"   → ["can", "'", "t"]        (apostrophe split!)
"believe" → ["believe"]
"it"      → ["it"]
","       → [","]
'"'       → ['"']
"she"     → ["she"]
"said"    → ["said"]
"."       → ["."]
'"'       → ['"']
"Really"  → ["Really"]
"?"       → ["?"]
'"'       → ['"']
```

**Final tokens (17 tokens):**

```
["\"", "I", "can", "'", "t", "believe", "it", ",", "\"", "she",
 "said", ".", "\"", "Really", "?", "\""]
```

**With a hypothetical vocab:**

```
[8, 23, 44, 8, 55, 17, 31, 7, 8, 78, 92, 6, 8, 63, 12, 8]
```

(Where `"` = ID 8, `I` = ID 23, `can` = ID 44, `'` = ID 8, etc.)

---

## Appendix B: Vocab File Example

For a model trained on `tiny.txt` with `max_vocab=32`:

```
0	<unk>
1	<pad>
2	<bos>
3	<eos>
4	the
5	and
6	.
7	,
8	to
9	of
10	a
11	in
12	"
13	is
14	:
15	that
16	it
17	for
18	-
19	was
20	on
21	'
22	;
23	)
24	(
25	?
26	!
27	with
28	as
29	his
30	are
31	this
```

Notice that the most common tokens are function words ("the", "and", "to") and
punctuation (".", ",", '"'). This is typical for English text — Zipf's law in
action.

---

## Appendix C: Memory Layout

How the data pipeline's memory is organized:

```
┌────────────────────────────────────────────────────────────────┐
│  Dataset (owns all memory)                                     │
│                                                                │
│  text: [t h e   c a t   s a t   o n   t h e   m a t \n ...]  │
│        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^              │
│        tokens[i] are slices into this buffer                    │
│                                                                │
│  tokens: ["the", "cat", "sat", "on", "the", "mat", ...]        │
│           ^                                                    │
│           each is a []const u8 pointing into text               │
│                                                                │
│  ids: [4, 5, 6, 7, 4, 8, ...]                                 │
│        owned u32 array                                         │
│                                                                │
│  vocab:                                                        │
│    word_to_id: HashMap → u32 values                            │
│    id_to_word: ArrayList of []const u8 → slices into text      │
│    freq: HashMap → u32 counts                                 │
│                                                                │
│  windows: []Window  (owned array)                              │
│    .input  and .target are []const u32 slices into ids         │
│                                                                │
│  batches: []Batch  (owned array)                               │
│    .inputs  and .targets are []const u32 slices into ids       │
│    (post-shuffle, the indices are indirect through the          │
│     shuffle permutation)                                       │
└────────────────────────────────────────────────────────────────┘
```

Key insight: **almost no data is copied.** Tokens are slices into the text
buffer. Windows are slices into the ids array. The only owned allocation
is the ids array itself and the shuffle permutation. This minimizes memory
usage and allocation overhead.

---

## Appendix D: From Pipeline to Training Loop

Here's how the data pipeline fits into the training loop from `examples/05_train_tiny.zig`:

```
┌─────────────────────────────────────────────────────────────────────┐
│  // 1. Build dataset                                                │
│  var dataset = Dataset.init(allocator, "data/tiny.txt",             │
│                             .max_vocab = 256, .lowercase = true);  │
│                                                                     │
│  // 2. Create windows                                               │
│  var windows = windowing.createWindows(dataset.ids, T=8);          │
│                                                                     │
│  // 3. Training loop                                                 │
│  for (0..num_epochs) |epoch| {                                      │
│                                                                     │
│      // 4. Shuffle and batch                                        │
│      var batcher = Batcher.init(allocator, windows, B=4, seed=42); │
│      batcher.shuffle(seed + epoch);                                 │
│                                                                     │
│      // 5. Iterate batches                                          │
│      while (batcher.next()) |batch| {                               │
│                                                                     │
│          // 6. Convert to tensors                                   │
│          var input = Tensor.fromU32(batch.inputs, Shape.init2D(B, T));│
│          var target = Tensor.fromU32(batch.targets, Shape.init2D(B, T));│
│                                                                     │
│          // 7. Forward pass                                          │
│          var logits = model.forward(input, tape);                   │
│                                                                     │
│          // 8. Loss                                                  │
│          var loss = ops.loss.crossEntropy(logits, target, tape);    │
│                                                                     │
│          // 9. Backward                                              │
│          tape.backward(loss);                                        │
│                                                                     │
│          // 10. Optimizer step                                       │
│          optimizer.step();                                           │
│          optimizer.zeroGrad();                                       │
│                                                                     │
│          // 11. Cleanup                                              │
│          tape.reset();                                               │
│          input.deinit();                                             │
│          target.deinit();                                            │
│      }                                                              │
│  }                                                                  │
└─────────────────────────────────────────────────────────────────────┘
```

Steps 1–2 are **once** per program. Steps 4–5 are **once per epoch**.
Steps 6–11 are **once per batch**.

The data pipeline (steps 1–4) produces the `input` and `target` tensors.
Everything after step 6 is the model, autograd, and optimizer — the components
we built in Stages 2–4.

---

## Summary

| Component | File | Input | Output | Key idea |
|-----------|------|-------|--------|----------|
| Vocab | `vocab.zig` | word frequencies | word↔ID map | Top-K cutoff + special tokens |
| tokenize | `word.zig` | raw text | token list | Whitespace → peel → split |
| encode | `word.zig` | tokens + vocab | `[]u32` | HashMap lookup, `<unk>` for OOV |
| decode | `word.zig` | `[]u32` + vocab | text string | Join with spaces |
| Dataset | `dataset.zig` | file path | `[]u32` + Vocab | Read → tokenize → build vocab → encode |
| windowing | `windowing.zig` | `[]u32` + T | `[]Window` | Stride-T non-overlapping, target = input + 1 |
| Batcher | `batcher.zig` | `[]Window` + B | `[]Batch` | Fisher-Yates shuffle, drop-last |

The data pipeline transforms human text into the integer arrays that a
transformer consumes. Understanding this pipeline is essential — every
training bug either originates here (wrong tokenization, off-by-one in
windowing) or manifests here (loss not decreasing because target is wrong).

Next: [Chapter 07 — Training Loop & Loss Functions](07_training_loop.md)
