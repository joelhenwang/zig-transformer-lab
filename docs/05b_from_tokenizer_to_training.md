# 05b — From Tokenizer and Data Pipeline to Training

> Bridges `docs/05_transformer_math.md` and `docs/06_tokenizer_data.md` (the
> mechanics) to the ML/DL concepts they serve. Assumes you've read both and
> now want the conceptual picture: what does this data pipeline *model*, and
> how does one batch become one forward pass?
>
> Companion code: `src/tokenizer/*.zig`, `src/data/*.zig`. Worked examples
> all use `data/tiny.txt` or `data/tinyshakespeare.txt` — both committed to
> the repo so you can run every snippet.

Training a transformer is 80% plumbing. The model itself — embeddings,
attention, MLPs — is a few hundred lines. The rest is moving text from disk
into token IDs into batches into the model without introducing bugs. This
chapter is about that plumbing: why it's shaped the way it is, what each
piece corresponds to in PyTorch, and what subtle design decisions you would
have to make all over again if you rewrote it from scratch.

---

## 1. The big picture

In PyTorch you write:

```python
dataset = ShakespeareDataset("data.txt", seq_len=16)
loader = DataLoader(dataset, batch_size=4, shuffle=True)
for batch_idx, (input_ids, targets) in enumerate(loader):
    logits = model(input_ids)
    loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))
    ...
```

We have the same pipeline with different names and explicit memory
management:

```zig
var dataset = try Dataset.init(alloc, io, "data.txt", max_vocab, lowercase);
const windowing = Windowing.init(dataset.tokens, T);
var batcher = try Batcher.init(alloc, windowing, B, &rng);
while (batcher.next()) |batch| {
    defer batch.deinit();
    // forward + loss + backward + step
}
```

The three primitives — `Dataset`, `Windowing`, `Batcher` — decompose what
PyTorch compresses into `Dataset` + `DataLoader`. The decomposition makes
each step visible, which is the whole point of the library.

### Why decompose this far?

A typical DataLoader hides:

1. Where do tokens come from? (`Dataset`)
2. What's a "sample" of training data? (`Windowing`)
3. How are samples grouped into batches? (`Batcher`)
4. How do we randomise order across epochs? (RNG state inside the batcher)

Every one of these is a real design decision. By naming them
independently we force ourselves — and you, the reader — to answer each
question explicitly.

---

## 2. From text to token IDs

The first mile: bytes in a file → `u32` token IDs.

### 2.1 Why word-level, not BPE or char

This project uses **word-level tokenization**: split on whitespace +
punctuation, lowercase optional, assign a stable integer to each unique
word. See `src/tokenizer/word.zig` for the splitter and
`src/tokenizer/vocab.zig:137` (`buildFromText`) for the vocabulary builder.

Why word-level? Three reasons:

1. **Pedagogy.** Word-level is trivially understandable — `"the"` → `4`,
   always. BPE requires a separate chapter on subword algorithms.
2. **Small vocab.** A 1 MB Shakespeare corpus has ~2 000 unique words.
   That fits in `u16` if needed; our `max_vocab` default is 2 000.
3. **Matches the model size.** A 1-block D=32 transformer can't learn
   rare-word statistics anyway. Word-level lets us see meaningful
   next-token predictions in ~500 training steps.

Trade-off: OOV (out-of-vocabulary) words map to `<unk>`. On the held-out
portion of Shakespeare this is ~15% of tokens. Production pipelines would
use BPE (GPT-2 style) or SentencePiece.

### 2.2 Special tokens and their IDs

`src/tokenizer/vocab.zig:68-72`:

```zig
pub const UNK_ID: u32 = 0;
pub const PAD_ID: u32 = 1;
pub const BOS_ID: u32 = 2;
pub const EOS_ID: u32 = 3;
pub const NUM_SPECIALS: u32 = 4;
```

The first four IDs are reserved for special tokens. Real words start at
`4`. This matters for two reasons:

- A fresh vocab with only specials has `size = 4`; the embedding matrix
  has 4 rows. Adding a word grows it.
- If you see a token ID `< 4` flowing through training data that *isn't*
  a special token (e.g. your tokeniser emits `1` somewhere), your model
  will confuse real data with the `<pad>` marker. Easy bug.

### 2.3 Dataset: file → token IDs

`Dataset.init` in `src/data/dataset.zig:65` does a lot in one call:

1. Read the file via `std.Io.Dir.cwd().readFileAlloc`.
2. Tokenise with `src/tokenizer/word.zig:92`.
3. Build the vocabulary with `Vocab.buildFromText`.
4. Encode the tokenised stream into a flat `[]u32` of token IDs.
5. Store `tokens` as the owned buffer (freed in `deinit`).

After `Dataset.init`, `self.tokens` is a long `[]u32` (for Shakespeare:
~260 000 entries). This is the training corpus in the form the model
understands.

PyTorch parallel: roughly equivalent to `tokenizer.encode(text,
return_tensors="pt").squeeze()` followed by a cached tensor.

---

## 3. The dataloader mental model

### 3.1 `torch.utils.data.Dataset` vs. our `Dataset`

`torch.utils.data.Dataset` is an abstract base with `__len__` and
`__getitem__`. `DataLoader` iterates it, handles batching, shuffling,
prefetch, multi-worker, etc.

Our `Dataset` is concrete: it just owns a flat `tokens: []u32` buffer.
Indexing into "windows" of training examples happens separately in
`Windowing`. This separation lets us swap out the sampling strategy
(next-token, BERT-style masking, next-sentence) without touching the
data loader.

### 3.2 What our `Batcher` does not do

A production DataLoader does:

- Multi-process data loading with prefetch.
- Pinned memory for fast CUDA transfer.
- Custom `collate_fn` for irregular-shape batches.
- Distributed sampling across nodes.

Our Batcher does none of these. It is a shuffled index generator and
batch packer. That's it.

Why? Because every one of those features adds weight without teaching
anything new to the reader. Production DataLoaders are
engineering-heavy; their educational value is zero.

---

## 4. Window plus shift equals next-token supervision

This is the conceptual centrepiece of language-model training and the
thing most beginners get wrong on their first try.

Given a token stream `t_0 t_1 t_2 ... t_n`, a **window** of size `T` at
position `i` is the pair:

```
input  = [t_i    , t_{i+1}, ..., t_{i+T-1}]    (T tokens)
target = [t_{i+1}, t_{i+2}, ..., t_{i+T  }]    (T tokens)
```

The `target` is the same as the `input` shifted left by one. For each
position `j` in the input, the model's job is to predict the token at
position `j+1`.

### 4.1 Why shift by one, not by T

The naive thing is to shift by `T`:

```
input  = [t_0 ... t_{T-1}]
target = [t_T ... t_{2T-1}]
```

This gives you one next-token prediction per window. With our
Shakespeare corpus of 260 000 tokens and `T=16`, that's ~16 000 windows.

The shift-by-one approach gives you `T` next-token predictions per
window — one for each position. With 260 000 tokens and `T=16`, that's
close to 260 000 predictions per epoch (instead of 16 000). A 16× gain
in training signal per pass through the data.

This is why transformer training uses causal masking: each position in
the output predicts the token *at that position*, and the causal mask
ensures no position "cheats" by looking at future tokens.

### 4.2 The windowing walk

`src/data/windowing.zig:87` (`Windowing.init`) stores the original
token buffer and the sequence length `T`. It does NOT pre-materialise
windows. Instead `Windowing.get(idx)` (`src/data/windowing.zig:127`)
returns a `Window` struct containing:

```zig
pub const Window = struct {
    input: []const u32,   // points into tokens[idx..idx+T]
    target: []const u32,  // points into tokens[idx+1..idx+T+1]
};
```

These are *views* into the original tokens buffer — zero copies. The
same bytes are read 2× (once as input, once as target) but the memory
footprint doesn't grow.

### 4.3 Pictures

Here's a tiny worked example with `tokens = [5, 2, 7, 9, 3, 1]` and
`T = 3`:

```
Window idx=0:
  input  = tokens[0..3] = [5, 2, 7]
  target = tokens[1..4] = [2, 7, 9]
  Predictions:
    input[0]=5 -> predict target[0]=2
    input[1]=2 -> predict target[1]=7
    input[2]=7 -> predict target[2]=9

Window idx=1:
  input  = tokens[1..4] = [2, 7, 9]
  target = tokens[2..5] = [7, 9, 3]

Window idx=2:
  input  = tokens[2..5] = [7, 9, 3]
  target = tokens[3..6] = [9, 3, 1]
```

Three windows, three next-token predictions each, total nine training
examples from six tokens. Windows overlap heavily — the model sees the
same bigram `(2, 7)` in two consecutive windows. This is by design; it
helps the model learn consistent short-range patterns.

### 4.4 How many windows total?

`Windowing.count()` returns `tokens.len - T`. With `tokens.len = n` and
`T = 16`, you get `n - 16` windows. For Shakespeare's 260 000 tokens,
that's ~259 984 windows. Each batch draws `B` of them (shuffled), so a
batch processes `B * T` tokens. One epoch through all windows takes
`count() / B` batches.

---

## 5. Batch size and sequence length tradeoffs

The second-most-common beginner question: "how do I pick B and T?"
Here's the thinking.

### 5.1 Memory × compute square

Every training step computes a `(B, T, V)` logits tensor. For
Shakespeare with `V=2000`, a `(B=4, T=16, V=2000)` logits tensor is
512 KB. Scaling either axis by 2× scales memory and compute by 2×.

The attention matrix is `(B, T, T)`. Scaling `T` by 2× scales it
quadratically — `4×` memory and compute for the attention step.

Practical implications:

- `T` dominates attention cost at long sequences. Doubling T is 4×
  more expensive than doubling B.
- `B` dominates matmul cost at short sequences. Doubling B is 2×
  more expensive than doubling T when T is small.
- Larger B improves gradient stability (averaged over more samples).
  Larger T improves the model's ability to use long-range context.

### 5.2 Our defaults

`TrainConfig` (`src/lab/train.zig:72`) uses `B=4, T=16, D=32`. These
are deliberately tiny so the 1-block model trains in < 30s on a CPU.
`examples/10_train_deep.zig` goes to `D=64, n_layer=2, n_head=2` and
keeps `B=4, T=16` for comparability with the baseline.

Rule of thumb: once you trust the pipeline, scale B upward to fill
your memory budget, then tune T for context length.

---

## 6. From a batch to a forward pass

Final mile. `batcher.next()` returned a `Batch{input, target}` where
both are `[]u32` of length `B * T`. How do these become `(B, T, D)`
embeddings?

### 6.1 From `[]u32` to `Tensor`

The training loop (`src/lab/train.zig:285`) reshapes `B*T` indices into
a `(B, T)` f32 tensor by element-wise cast:

```zig
var ids = try Tensor.init(allocator, Shape.init2D(B, T));
for (0..B * T) |i| ids.data[i] = @floatFromInt(batch.input[i]);
```

The cast is lossless for token IDs up to 2²⁴ (well beyond our 2 000
vocab). The model sees a `(B, T)` tensor of floats, each float an
integer-valued ID in [0, V).

### 6.2 Embedding lookup

`Embedding.forward(ids)` (`src/nn/embedding.zig`) does the gather: for
each `(b, t)` position, it reads row `ids[b, t]` from the weight matrix
`(V, D)` and emits a `(B, T, D)` tensor.

This is the conceptual "bridge" between integers and vectors. Token `42`
is now a 32-dim vector. The rest of the transformer is
vector-arithmetic.

### 6.3 Positional embedding

Our model adds a learned positional embedding (`src/nn/model.zig`):
`pos_embed.weight` has shape `(max_seq_len, D)`. We look up rows
`[0 .. T]` and add element-wise to the token embeddings.

After this, the tensor flowing into the first transformer block has
shape `(B, T, D)` and every entry is a sum of one token embedding and
one position embedding. That's the "input representation" the
transformer stack refines.

### 6.4 Targets stay flat

The `targets` tensor stays 1-D `(B*T,)`. It's the companion to the
logits-reshape in the training loop:

```zig
// logits from model: (B, T, V)
// reshape to (B*T, V) via reshapeTracked
// targets: already (B*T,)
// crossEntropy expects (B*T, V) + (B*T,)
```

The `(B, T)` structure is useful during forward (causal mask, positional
embeddings) but flat is cleaner for the loss.

---

## 7. Common mistakes

- **Off-by-one in windowing.** `Windowing.count()` returns
  `tokens.len - T`, not `tokens.len - T + 1` or `tokens.len - T - 1`.
  The last valid window start index is `n - T - 1` (inclusive), giving
  `n - T` windows total. Getting this wrong cuts 1 window off the end
  or produces an out-of-bounds read.
- **Forgetting `batcher.reset()` between epochs.** After exhausting one
  epoch, `batcher.next()` returns `null`. The training loop calls
  `batcher.reset(&data_rng)` which shuffles the index list with a new
  RNG draw. Forgetting this freezes training at `null` forever.
- **Using the wrong dtype on target indices.** Targets are `u32` in
  the Batch struct and cast to f32 in the trainer. The CE backward
  uses `@round` + `@intFromFloat` to recover the class ID. If you
  skip the cast and try to pass `[]u32` directly to `crossEntropy`,
  the API breaks at the first op.
- **Mutating `Window.input` or `Window.target`.** Both are
  `[]const u32` views into the Dataset's token buffer. If you
  `@constCast` and write into them, you corrupt every other window
  that shares the same underlying memory. Always treat windows as
  read-only.
- **Setting `max_vocab` too small.** If a corpus has more unique words
  than `max_vocab`, the tail of the distribution collapses to `<unk>`.
  For Shakespeare with `max_vocab = 500` you'd see `<unk>` dominate
  after the top 500 words, which ruins training signal.
- **Lowercasing inconsistently.** `Vocab.init` and the tokenizer both
  take a `lowercase` flag. If they disagree, the tokenizer emits
  `"The"` but the vocab only has `"the"`. Every capitalised word in
  the corpus maps to `<unk>`.

---

## 8. Exercises

**Exercise 1.** You have a corpus of 1 000 tokens and `T = 4`. How many
windows does `Windowing.count()` return? How many distinct pairs of
(input_position, target_token) does the model see in one epoch?

<details><summary>Solution</summary>

`count() = 1000 - 4 = 996` windows. Each window produces `T = 4`
predictions (one per input position, predicting the next token).
Total predictions per epoch: `996 * 4 = 3984` distinct
(input_position, target_token) pairs. Note that many of these pairs
repeat — the same bigram `(t_i, t_{i+1})` appears in up to `T = 4`
consecutive windows.

</details>

**Exercise 2.** Suppose you want to train on pairs of Shakespeare
plays, separated by a sentence boundary marker. The current pipeline
would split windows that cross the boundary, producing windows whose
input is the end of play A and target is the beginning of play B. How
would you modify `Windowing.init` to prevent this?

<details><summary>Solution</summary>

Accept an additional `segment_boundaries: []const usize` argument: a
sorted list of token indices where no window may cross. Modify
`count()` to subtract `T - 1` for each boundary (since a boundary at
index `k` invalidates windows starting at indices `k - T + 1 .. k`).
Modify `get(idx)` to skip invalidated indices by maintaining a
mapping from logical window-index to actual start-position.

Alternative: pre-filter the token buffer to replace the boundary with
a sequence of `<eos><bos>` tokens and train the model to handle
these as regular tokens. This is what GPT-style pretraining does —
no windowing logic changes required.

</details>

**Exercise 3.** Why does `Batcher.reset(rng: *Rng)` take an RNG rather
than reading from a stored RNG state?

<details><summary>Solution</summary>

The training loop owns the RNG. Passing it by pointer means the
batcher's shuffle draws advance the same stream that the rest of the
training loop uses. That matters for reproducibility: with a single
seed, the exact sequence of shuffles + model init + random sampling
is deterministic. If the batcher owned its own RNG, you'd need two
seeds and two call patterns to reproduce a run.

This is a general Zig pattern: components that need randomness
borrow an RNG from the caller. No hidden sources of entropy,
consistent with policy P2 (no hidden globals).

</details>

**Exercise 4.** On the Shakespeare config (`V = 2000, T = 16, B = 4`),
how many tokens does the model see per second if each training step
takes 140 ms on CPU? Compare against a rough budget for one epoch of
the Shakespeare corpus.

<details><summary>Solution</summary>

Tokens per step: `B * T = 64`. Steps per second: `1 / 0.140 ≈ 7.1`.
Tokens per second: `64 * 7.1 ≈ 455`.

One epoch: `count() ≈ 260 000 - 16 ≈ 259 984` windows. At `B = 4`
per step: `259 984 / 4 ≈ 64 996` steps per epoch. At 140 ms/step:
`64 996 * 0.140 ≈ 9 100 s ≈ 2.5 hours` per epoch on CPU.

This is why we ship a CUDA backend. Stage 7's 30× speedup brings
one epoch down to ~5 minutes. Stage 8's 2/2/64 acceptance config at
12.3 ms/step brings it to ~13 minutes even with 2× the parameters.

</details>

---

## 9. PyTorch equivalents

| Our code | PyTorch equivalent |
|---|---|
| `Dataset.init(alloc, io, path, max_vocab, lowercase)` | `torch.load(path)` + tokenizer call |
| `Vocab.buildFromText(...)` | `Counter(tokens).most_common(max_vocab)` |
| `Windowing.init(tokens, T)` | `torch.utils.data.Dataset` subclass with `__getitem__` returning shifted pairs |
| `Batcher.init(alloc, windowing, B, &rng)` | `DataLoader(ds, batch_size=B, shuffle=True)` |
| `Batcher.next()` | `next(iter(loader))` |
| `Batcher.reset(&rng)` | Re-instantiate iter at epoch boundary |
| `Window{input, target}` | `input_ids, targets` tuple from loader |

---

## 10. End-to-end integration checklist

When you wire up a new training run — especially after changing the
tokenizer or data file — run through this list before expecting
meaningful loss curves. Each entry maps to a real bug we've seen.

### 10.1 Data integrity

- [ ] **Does `Dataset.len()` match the expected token count?** For
  `data/tinyshakespeare.txt` at 2 000-vocab lowercase, this is
  about 260 000. Major deviations mean the tokeniser is behaving
  unexpectedly (wrong split rule, wrong lowercase flag).
- [ ] **Is `vocab.size()` less than or equal to `max_vocab`?** A vocab
  builder that ignores the cap would produce embeddings of the wrong
  shape — catch this early, not after the first OOM.
- [ ] **Does the tokenizer emit any ID outside `[0, vocab.size())`?**
  A `<unk>` fallback produces `UNK_ID = 0`, which is in range, but a
  bug could emit `u32` max (sentinel unassigned IDs). Grep the
  encoded token stream for outliers.

### 10.2 Windowing correctness

- [ ] **Does `Windowing.count()` match `tokens.len - T` exactly?** Not
  `-T+1`, not `-T-1`. Off-by-one here cuts training signal by 1
  window per run; with overlapping windows the impact is tiny but the
  bug is a tell.
- [ ] **For a hand-picked window, does `window.target[i] == tokens[start
  + i + 1]` for all `i`?** The shift-by-one is the defining
  property of next-token supervision. A corrupted target offset
  means the model is trained to predict the wrong thing.

### 10.3 Batching semantics

- [ ] **Do all batches of one epoch cover every window exactly once?**
  With `B = 4` and `count() = 12`, an epoch should emit 3 batches,
  total 12 windows — no duplicates, no misses. If you see 11 or 13,
  there's an off-by-one in the batcher's index list.
- [ ] **After `reset(&rng)`, is the shuffle order different from the
  previous epoch?** Deterministic seed across resets is a bug; the
  whole point of reset is to refresh the order. Print the first few
  window indices each reset to verify.
- [ ] **Does `batch.input.len == B * T`?** Each batch packs `B` windows
  of length `T`. If this doesn't hold you have a batcher bug; every
  downstream tensor shape will be wrong.

### 10.4 Round-trip encoding

A useful smoke test: encode a small string, decode it, compare to the
original after lowercase normalisation.

```zig
const src = "the quick brown fox";
const ids = try word.encode(alloc, src, &vocab);
defer alloc.free(ids);
const back = try word.decode(alloc, ids, &vocab);
defer alloc.free(back);
// back should equal "the quick brown fox" if all words are in vocab
```

Any discrepancy at this stage is a vocab or tokeniser bug, not a
model bug. Catching it here saves hours of wondering why training
looks weird.

### 10.5 Timing and memory

- [ ] **How long does `Dataset.init` take?** For the 1 MB Shakespeare
  corpus: < 200 ms is healthy. > 2 s means the tokeniser is
  pathological (quadratic on word count, for example).
- [ ] **How much memory does `tokens: []u32` occupy?** `tokens.len * 4`
  bytes. For 260 000 tokens that's 1 MB. Anything larger means
  encoding is storing something besides `u32` IDs.
- [ ] **Do the deferred deinits actually fire?** Run under the
  `DebugAllocator`; any leak is either a forgotten `defer` or a
  consumed-but-not-cleaned borrow somewhere.

Running through this list end to end takes under a minute once you're
used to it and has caught every real bug we've shipped in Stages 5
through 8.

---

*End of Chapter 05b — From Tokenizer and Data Pipeline to Training.*
