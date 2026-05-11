# `src/data/`

The data pipeline: from an on-disk text corpus to `(input_ids, target_ids)`
batches ready for a model's forward pass. PyTorch parallel:
`torch.utils.data.Dataset` + `DataLoader`.

## Files

- `dataset.zig` — `Dataset`. Reads a text file, tokenises it, builds
  a `Vocab`, stores the flat `tokens: []u32` stream. One per training
  run. Owns the token buffer; `deinit` frees it.
- `windowing.zig` — `Windowing`. Given a token stream and a sequence
  length `T`, exposes the `Window{ input: []const u32, target:
  []const u32 }` pairs where `target` is `input` shifted left by 1.
  Zero-copy views into the Dataset's buffer.
- `batcher.zig` — `Batcher`. Shuffles window indices with a borrowed
  `Rng`, packs `B` windows per batch, emits `Batch{ input: []u32,
  target: []u32 }` with `input.len == target.len == B * T`. Call
  `reset(&rng)` between epochs to reshuffle.

## If you're new here

Read `docs/06_tokenizer_data.md` (mechanics) then
`docs/05b_from_tokenizer_to_training.md` (concepts). The learning
guide covers the shift-by-1 supervision pattern, the batch-size / seq-len
tradeoff, and the end-to-end integration checklist.

## Cross-references

- Full training-step integration: `docs/07_cpu_training.md` §7.1
- PyTorch parallel (DataLoader vs our Batcher):
  `docs/05b_from_tokenizer_to_training.md` §3
- Why we decompose Dataset / Windowing / Batcher instead of one
  DataLoader: `docs/05b_from_tokenizer_to_training.md` §1
