# `src/tokenizer/`

Word-level tokeniser. Text → `[][]const u8` words → `[]u32` token IDs.

## Files

- `word.zig` — tokeniser + encoder + decoder. `tokenize(text, lowercase)`
  splits on whitespace/punctuation, applies optional lowercasing, and
  emits an owned `std.ArrayList([]const u8)`. `encode(text, vocab)`
  is the one-shot path to `[]u32` IDs. `decode(ids, vocab)` goes
  back to text.
- `vocab.zig` — `Vocab`: a bidirectional map between words and
  `u32` IDs. `buildFromText(alloc, io, text, max_vocab, lowercase)`
  tokenises a whole corpus, counts frequencies, reserves the top
  `max_vocab - NUM_SPECIALS` words by frequency, and emits the
  finished `Vocab`. Also defines the reserved special-token IDs
  (`UNK_ID = 0`, `PAD_ID = 1`, `BOS_ID = 2`, `EOS_ID = 3`).

## If you're new here

Read `docs/06_tokenizer_data.md` (mechanics) then
`docs/05b_from_tokenizer_to_training.md` §2 (the word-level vs BPE
trade-off).

## Cross-references

- Why word-level (pedagogy + small vocab):
  `docs/05b_from_tokenizer_to_training.md` §2.1
- Special-token numbering: `src/tokenizer/vocab.zig:68-72`
- OOV handling (15% of held-out Shakespeare falls to `<unk>`):
  `docs/05b_from_tokenizer_to_training.md` §2.1
- The encode/decode round-trip smoke test:
  `docs/05b_from_tokenizer_to_training.md` §10.4
