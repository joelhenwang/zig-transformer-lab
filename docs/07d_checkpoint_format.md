# 07d — Checkpoint Format ZTLC v2

This chapter documents the binary checkpoint format introduced in
PR-η and walks through every validation the loader performs. A
pedagogical tensor library has two choices for checkpoint handling:
*permissive*, where load silently reshapes or zero-fills anything
mismatched, or *strict*, where any deviation from the saved format is
an immediate error. We chose strict, because the alternative is the
single largest class of "I wasted three days on a silently-wrong
training run" bugs in ML practice.

Read `docs/04_nn.md` and `docs/07_cpu_training.md` for the model
layout and training loop this format serialises.

---

## 1. Why format strictness matters

A checkpoint is a snapshot of a trained model's parameters on disk.
The load path takes a file and a fresh model object, and copies the
file's values into the model's parameter tensors.

Everything between "a file" and "values into tensors" is where silent
bugs live. Consider:

- The checkpoint was saved with `vocab_size = 2000` but loaded into a
  model with `vocab_size = 1500`. Do you truncate the embedding
  matrix? Pad it? Fail?
- The checkpoint was saved with `d_model = 64`; the loader's model has
  `d_model = 32`. Every matmul shape is wrong. Do you re-sample?
  Pick the first half of each row? Error?
- The checkpoint's `block.attn.w_q.weight` is missing from the file
  (perhaps an earlier save version didn't include it). Do you
  initialise it randomly? Zero it? Error?
- The file has an *extra* parameter `block.attn.w_hidden_state` from a
  different model architecture. Do you silently discard it? Error?
- The file is corrupted at byte 1024 — truncated mid-parameter. Do
  you read whatever you can? Error?

For a production system with a diverse user base, PyTorch argues for
permissive behaviour plus clear warnings. For a pedagogical system
teaching correctness, we argue for error-on-every-deviation. The
savings in debugging time dwarf the cost in occasional "please
regenerate the checkpoint" friction.

---

## 2. Pre-PR-η format (ZTLC v1, deprecated)

The v1 format saved in Stage 6 was minimal:

```
magic      "TWTL"          (4 bytes)
version    u32 = 1         (4 bytes)
num_params u32             (4 bytes)

Per parameter:
  name_len   u32
  name       [name_len]u8
  rank       u8
  dims       [4]u32        (dims beyond rank set to 0)
  data_len   u32           (byte count)
  data       [data_len]u8  (f32 little-endian)
```

Every field typechecked, but the loader did not validate:

- Model architecture (vocab_size, d_model, etc.) matches the file
- Every expected parameter is present
- No unexpected parameters are present
- No parameter is listed twice
- Each parameter's shape matches what the current model expects
- The file is not truncated

A v1 checkpoint from a `V=2000 D=32` run loaded into a `V=1500 D=32`
model would write the first 1500 rows of the saved embedding table
into the new (smaller) embedding table and proceed. The model would
appear to load successfully. Training would produce garbage because
some of those 1500 rows belong to vocabulary positions that don't
exist in the new model.

ZTLC v2 fixes this.

---

## 3. The ZTLC v2 format

All multi-byte integers and floats are little-endian. Structures
pack tightly with explicit pad bytes marked `_pad` to keep field
offsets at natural 4-byte alignment. Every byte of the format is
accounted for below.

### 3.1 Header (32 bytes)

```
offset  size  field          value / notes
------  ----  -------------  -------------------------------------
  0     4     magic          "ZTLC"
  4     4     version        u32 = 2
  8     4     model_kind     u32 = 1  (1 = TinyWordTransformer)
 12     4     vocab_size     u32
 16     4     max_seq_len    u32
 20     4     d_model        u32
 24     4     d_ff           u32
 28     1     bias           u8 (0 or 1)
 29     3     _pad           3 × 0x00 (align to 4)
 32     4     param_count    u32
```

Everything the model needs to be structurally compatible is encoded
here. The loader compares every field of the header (except the
padding) to the current model's config and refuses to proceed on any
mismatch.

### 3.2 Per-parameter record (variable size)

```
offset  size         field       value / notes
------  -----------  ----------  -------------------------------
  0     4            name_len    u32 (1..=255)
  4     name_len     name        ASCII bytes, no terminator
  *     1            rank        u8
  *     3            _pad        3 × 0x00
  *     16           dims        4 × u32 (unused dims = 0)
  *     4            data_len    u32 (= total_elements * 4)
  *     data_len     data        f32 little-endian bytes
```

The `*` entries mean "wherever the previous field ended". Fields
have natural alignment within the record.

### 3.3 Trailer (4 bytes)

```
offset  size  field         value
------  ----  ------------  --------
  0     4     end_magic     "END."
```

The trailer exists so the loader can detect truncation. A valid v2
file is exactly:

```
[header (32 bytes)]
[param 0]
[param 1]
...
[param N-1]
[trailer (4 bytes)]
```

with no extraneous bytes anywhere.

---

## 4. Hexdump of a tiny checkpoint

Here is a complete ZTLC v2 file for a toy model with
`V=2, D=2, max_seq_len=2, d_ff=2, bias=true`. It has 15 parameters
but for brevity we show only the first two.

```
Offset   Bytes (hex)                                    ASCII         Field
-------  ---------------------------------------------  ------------  ------------------
0x0000:  5A 54 4C 43                                    ZTLC          magic
0x0004:  02 00 00 00                                    ....          version = 2
0x0008:  01 00 00 00                                    ....          model_kind = 1
0x000C:  02 00 00 00                                    ....          vocab_size = 2
0x0010:  02 00 00 00                                    ....          max_seq_len = 2
0x0014:  02 00 00 00                                    ....          d_model = 2
0x0018:  02 00 00 00                                    ....          d_ff = 2
0x001C:  01 00 00 00                                    ....          bias = true, 3×pad
0x0020:  0F 00 00 00                                    ....          param_count = 15

// --- param 0: tok_embed.weight ---
0x0024:  10 00 00 00                                    ....          name_len = 16
0x0028:  74 6F 6B 5F 65 6D 62 65 64 2E 77 65 69 67 68 74 tok_embed.weight  name
0x0038:  02                                             .             rank = 2
0x0039:  00 00 00                                       ...           _pad
0x003C:  02 00 00 00                                    ....          dims[0] = 2 (vocab)
0x0040:  02 00 00 00                                    ....          dims[1] = 2 (d_model)
0x0044:  00 00 00 00                                    ....          dims[2] = 0
0x0048:  00 00 00 00                                    ....          dims[3] = 0
0x004C:  10 00 00 00                                    ....          data_len = 16 bytes = 4 f32
0x0050:  00 00 80 3F 00 00 00 40 00 00 40 40 00 00 80 40  ....@@@@      data: 1.0, 2.0, 3.0, 4.0

// --- param 1: pos_embed.weight ---
0x0060:  10 00 00 00                                    ....          name_len = 16
0x0064:  70 6F 73 5F 65 6D 62 65 64 2E 77 65 69 67 68 74 pos_embed.weight  name
0x0074:  02                                             .             rank = 2
0x0075:  00 00 00                                       ...           _pad
0x0078:  02 00 00 00                                    ....          dims[0] = 2 (max_seq_len)
0x007C:  02 00 00 00                                    ....          dims[1] = 2 (d_model)
0x0080:  00 00 00 00  00 00 00 00                        ........      dims[2..4] = 0
0x0088:  10 00 00 00                                    ....          data_len
0x008C:  ...data...

// ... 13 more params ...

// --- trailer ---
0xNNNN:  45 4E 44 2E                                    END.          end_magic
EOF
```

### 4.1 Decoding the f32 payload

`00 00 80 3F` is IEEE 754 `1.0` little-endian. To verify:

```
bytes:           00 00 80 3F
as u32 little:   0x3F800000
IEEE 754:        sign=0, exp=01111111 (127), mantissa=0
                 value = 1.0 × 2^(127-127) = 1.0 ✓
```

Little-endian is non-negotiable because the format specifies it.
Reading this file on a big-endian machine would require byte-swapping
every f32.

### 4.2 Total file size for the toy model

For a tiny `V=2 D=2 T=2 F=2 bias=true` `TinyWordTransformer`, the
parameter count is 15 (see §6 for the full list) and the payload is
128 f32s = 512 bytes. The whole file is about 1.3 KB including
headers and names.

For the Stage 6 production config `V=2000 D=32 T=16 F=128`, the
total is ~280 KB (dominated by `lm_head.weight` at
2000×32 = 64 000 f32 = 256 KB).

---

## 5. What the loader validates (in order)

This is `src/nn/model.zig:TinyWordTransformer.load`, step by step.

### 5.1 Magic

```zig
var magic: [4]u8 = undefined;
try r.readSliceAll(&magic);
if (!std.mem.eql(u8, &magic, "ZTLC")) return error.IoError;
```

Any file whose first four bytes are not `"ZTLC"` is rejected. This
catches:

- v1 files (which start with `"TWTL"`)
- non-checkpoint files the user pointed at by accident
- corrupted files where the first bytes were damaged

### 5.2 Version

```zig
const version = try r.takeInt(u32, .little);
if (version != 2) return error.IoError;
```

Only v2 is currently supported. A v1 file with intact magic would
fail magic first; a future v3 file would fail here. The rigid match
means we can introduce v3 (with, say, a checksum) without any
ambiguity about what an old loader should do.

### 5.3 Model kind

```zig
const model_kind = try r.takeInt(u32, .little);
if (model_kind != 1) return error.IoError;
```

`model_kind = 1` means "TinyWordTransformer". If someone adds
`model_kind = 2` for a multi-block variant later, this check stops a
multi-block checkpoint from loading into a single-block model and
vice versa. The model architecture is itself a versioning axis
independent of the file format.

### 5.4 Config fields

```zig
if (vocab_size != self.cfg.vocab_size or
    max_seq_len != self.cfg.max_seq_len or
    d_model != self.cfg.d_model or
    d_ff != self.cfg.d_ff or
    (bias_byte != 0) != self.cfg.bias)
{
    return error.ShapeMismatch;
}
```

Every config field must exactly match. Why so strict?

- `vocab_size`: changes the shape of `tok_embed.weight` and
  `lm_head.weight`. Silent mismatch produces the wrong logits.
- `max_seq_len`: changes the shape of `pos_embed.weight`. Silent
  mismatch produces position embeddings past the context window.
- `d_model`: changes the shape of *every* matmul. Silent mismatch
  breaks the whole model.
- `d_ff`: changes the shape of MLP layers. Silent mismatch breaks
  the feed-forward path.
- `bias`: changes the parameter count (Linear with bias has 2
  parameters; without has 1). Silent mismatch produces incorrect
  param count.

No field is "safe to relax". We error on any deviation, returning
`error.ShapeMismatch` because the fundamental issue is that shapes
no longer match.

### 5.5 Parameter count

```zig
if (param_count != param_list.items.len) return error.ShapeMismatch;
```

The current model's `collectNamedParams()` yields exactly 15 entries
for a `TinyWordTransformer`. If the file reports a different count,
either we're missing a parameter or we have an extra one — either
way, it's wrong.

### 5.6 Per-parameter name matching

For each parameter record in the file:

```zig
const name_len = try r.takeInt(u32, .little);
if (name_len == 0 or name_len > 255) return error.IoError;
try r.readSliceAll(name_buf[0..name_len]);
const name = name_buf[0..name_len];
```

Then locate the matching expected parameter:

```zig
var matched_index: ?usize = null;
for (param_list.items, 0..) |entry, idx| {
    if (std.mem.eql(u8, name, entry.name)) {
        matched_index = idx;
        break;
    }
}
const idx = matched_index orelse return error.IoError;
if (seen[idx]) return error.IoError; // duplicate
seen[idx] = true;
```

Three failure modes are caught:

- **Unknown name** (`matched_index == null`): the file contains a
  parameter name the current model doesn't recognise. Reject.
- **Duplicate name** (`seen[idx] == true`): the file contains the
  same name twice. Reject.
- **Empty or too-long name** (`name_len == 0 or > 255`): the file is
  malformed. Reject with `error.IoError`.

Valid name strings are chosen from the 15 `collectNamedParams`
entries and are short (max 30 chars). The 255-byte cap is generous;
it's there to bound buffer size without allocating.

### 5.7 Per-parameter shape validation

```zig
const t = param_list.items[idx].tensor;
if (rank != t.shape.ndim()) return error.ShapeMismatch;
for (0..4) |i| {
    const expected_dim: u32 = if (i < t.shape.ndim()) @intCast(t.shape.dims[i]) else 0;
    if (dims[i] != expected_dim) return error.ShapeMismatch;
}
const expected_bytes: u32 = @intCast(t.data.len * 4);
if (data_len != expected_bytes) return error.ShapeMismatch;
```

Once a name is matched, the saved parameter's rank, every dim, and
the payload byte count must match the current model's tensor for
that name. The `dims` array has all four entries checked (not just
the first `rank`) so that if the saved rank was 2 but dims[2] is not
0, we catch the corruption.

### 5.8 Payload read

```zig
const bytes = std.mem.sliceAsBytes(t.data);
try r.readSliceAll(bytes);
```

`readSliceAll` reads exactly `bytes.len` bytes or errors. If the
file is shorter than expected, the underlying reader returns
`error.EndOfStream` and we propagate. This is the "truncated file"
detection.

### 5.9 Completeness check

After the per-parameter loop:

```zig
for (seen, 0..) |was_seen, idx| {
    if (!was_seen) {
        _ = idx;
        return error.ShapeMismatch;
    }
}
```

Every expected parameter must have been seen in the file. If a name
in `collectNamedParams` never appeared, the file is missing a
parameter and we refuse to load.

Why `ShapeMismatch` and not `IoError`? Because the issue is
"the model's parameter set doesn't match the file's" — a shape /
structure problem — rather than "the file couldn't be read" — an I/O
problem. The two errors steer debugging in different directions, so
picking the right one matters.

### 5.10 Trailer

```zig
var trailer: [4]u8 = undefined;
try r.readSliceAll(&trailer);
if (!std.mem.eql(u8, &trailer, "END.")) return error.IoError;
```

The final four bytes must be `"END."`. Missing trailer means the
file was truncated or the writer was killed mid-flush.

We do *not* currently check for bytes past the trailer. An extra-
bytes file would pass the loader today. A future version could track
total bytes read and compare to file size, returning an error on
mismatch — this would require a `file.stat(io)` call and a running
byte counter.

---

## 6. The full parameter list

For reference: the 15 parameters `collectNamedParams` emits in order.

```
tok_embed.weight        (vocab_size, d_model)
pos_embed.weight        (max_seq_len, d_model)
block.ln1.gamma         (d_model,)
block.ln1.beta          (d_model,)
block.attn.w_q.weight   (d_model, d_model)    bias=false for attention
block.attn.w_k.weight   (d_model, d_model)
block.attn.w_v.weight   (d_model, d_model)
block.attn.w_o.weight   (d_model, d_model)
block.ln2.gamma         (d_model,)
block.ln2.beta          (d_model,)
block.mlp.fc1.weight    (d_ff, d_model)       bias=false for MLP
block.mlp.fc2.weight    (d_model, d_ff)
ln_f.gamma              (d_model,)
ln_f.beta               (d_model,)
lm_head.weight          (vocab_size, d_model) bias=false for output head
```

Notice that the attention projections, MLP layers, and LM head all
have `use_bias=false`. This reduces the parameter count from the
"everything has bias" variant by `4*D + 2*(F+D) + V`. For the Stage 6
config `D=32, F=128, V=2000`, that's 2388 fewer parameters, roughly
4% of the total.

If you change any `Linear` in the model to use bias, you'll need to:

1. Add a new `bias` entry to `collectNamedParams`
2. Expect `param_count` to grow
3. Ensure ID assignment in `Linear.init` covers the new bias (it
   already does)
4. Checkpoints made before the change will fail to load with
   `ShapeMismatch` — that's correct behaviour.

---

## 7. Format comparison

How does ZTLC v2 stack up against the formats you'll encounter in the
wild?

### 7.1 vs `torch.save` / pickle

PyTorch's default format is Python pickle plus tensor blobs. It's:

- Python-object-centric; requires the same class hierarchy to load
- Unsafe for untrusted files (can execute arbitrary code)
- Verbose (pickle overhead dominates small checkpoints)
- Flexible (anything picklable can be saved)
- Poor at strict schema enforcement (a dict-shaped state_dict doesn't
  prevent silent key mismatches)

ZTLC v2 is the opposite on every axis: simpler, safer, smaller, more
strict, but also less flexible.

### 7.2 vs safetensors

[safetensors](https://github.com/huggingface/safetensors) is
HuggingFace's modern format:

- JSON header + raw tensor bytes
- Shape, dtype, byte offset per tensor in the header
- No code execution on load
- Cross-language by design (header is JSON)

ZTLC v2 is similar in spirit but:

- Binary header instead of JSON (smaller, not human-readable)
- Embedded model kind and config (safetensors treats the file as
  pure tensor data and leaves model identity to the caller)
- Strict validation in the loader (safetensors exposes tensors and
  lets the caller decide what to check)

For a pedagogical library where we control both save and load,
embedded strict validation is the right choice. For a cross-
ecosystem format, safetensors' philosophy wins.

### 7.3 vs GGUF (llama.cpp)

GGUF is the format used by llama.cpp for quantised models:

- Binary header with arbitrary metadata key-value pairs
- Tensor descriptors in a table
- Alignment padding
- Quantisation info per tensor (f16, q4_0, q8_0, etc.)

ZTLC v2 is simpler because we're f32-only. The structure (header +
per-tensor records + payloads) is similar. GGUF's metadata
flexibility (arbitrary key-value pairs) is attractive — it lets you
stash things like "this model was trained on Shakespeare" alongside
the weights. We chose not to mirror that because:

- It adds complexity to the loader
- We already have a fixed config embedded in the header
- Free-form metadata is exactly what `torch.save` got wrong

If a future PR needs per-checkpoint metadata (training step count,
loss history, random seed), the right place is a new header field
with a defined meaning, not a generic key-value bag.

### 7.4 vs ONNX

ONNX is protobuf-based and serialises the *computation graph* in
addition to the weights. That's a much larger surface than we need.
ZTLC v2 is weights-only; the model architecture is implicit in the
`model_kind` field.

---

## 8. Endianness policy

ZTLC v2 is little-endian throughout. This includes:

- Every `u32` in the header (magic, version, model_kind, config)
- Every `u8`/`u32` in parameter records
- Every `f32` in the data payloads

This matches x86-64 and ARM64 native byte order, so on every machine
where we run training, no byte-swapping happens. On a hypothetical
big-endian machine (PowerPC BE, old MIPS), the loader would need to
swap f32 bytes explicitly — we would detect this via `builtin.cpu`
at compile time and add swap logic only for BE targets.

We do not detect BE targets today because no BE target is supported.
If you port to one, start by adding a `comptime assert
builtin.cpu.arch.endian() == .little` in `model.zig:save`, and
handle big-endian f32 loading explicitly.

---

## 9. What we do NOT save

ZTLC v2 saves *only* model parameter weights. It does not save:

- **Optimizer state** (AdamW's `m`, `v`; SGD's velocity). See
  `07c_optimizer_state.md` — the infrastructure for this exists
  (`ParamId`-keyed state) but serialisation is a future PR.
- **Random seed state**. The RNG used for initialisation is not
  captured. Reproducibility across save/load cycles requires you to
  seed explicitly on each run.
- **Training step count or loss history**. No built-in place for
  these; a wrapper JSON file is the usual approach.
- **Tokenizer**. The vocabulary is saved separately by
  `src/tokenizer/vocab.zig`. ZTLC v2 assumes the loader already has
  the right vocab.

A future ZTLC v3 might add optional blocks for these, preserving
backward compatibility with v2 loaders by positioning new sections
after the trailer.

---

## 10. PyTorch parallels

### 10.1 save

```python
# PyTorch
torch.save(model.state_dict(), 'ckpt.pt')

# Ours
try model.save(io, "ckpt.bin");
```

PyTorch's state_dict is a Python dict of (name, Tensor). Our loop
over `collectNamedParams()` produces the equivalent (name, tensor)
pairs.

### 10.2 load

```python
# PyTorch (permissive by default)
state = torch.load('ckpt.pt')
model.load_state_dict(state, strict=True)  # strict mode available
# strict=False would silently skip mismatches.

# Ours (always strict)
try model.load(io, "ckpt.bin");
# error.ShapeMismatch / error.IoError on any deviation.
```

Our behaviour matches `strict=True`. There is no opt-out. If you
need permissive loading, you would have to hand-roll it — we
deliberately do not offer the knob.

### 10.3 Mismatch diagnostics

PyTorch's `load_state_dict` with `strict=False` returns
`(missing_keys, unexpected_keys)` so the user can see what changed.
Our loader returns one of a few error variants but not the specific
offending name. A future enhancement would be to collect all
mismatches before erroring, so a user can fix them all at once. For
Stage 6.5 we accept "fix one error, re-run, fix next".

---

## 11. Common mistakes

### "My checkpoint loads fine locally but fails in CI"

Most likely cause: different model config. Check that the test
creating the model uses the same `TransformerConfig` as the test
that saved the checkpoint. A single-field drift (say, `d_ff = 128`
vs `d_ff = 64`) produces `error.ShapeMismatch`.

### "I added a new layer but my old checkpoints still load successfully"

They shouldn't, and they don't — unless you forgot to register the
new layer's parameters in `collectNamedParams`. In that case the
loader doesn't see the new parameters at all, so the count still
matches and validation passes; the new parameters retain their
random initialisation.

Fix: add entries for the new layer in `collectNamedParams`. Your
old checkpoints will then fail to load (`error.ShapeMismatch` on
param count), which is correct — they are from a different model
architecture.

### "The 'name_len' check rejects my parameter name"

Names longer than 255 bytes are rejected. The longest name in the
current model is `block.mlp.fc2.weight` at 20 characters, far under
the limit. If you're seeing this, the file is corrupted or was saved
by a different writer.

### "I want to edit the file by hand"

Don't. Every field is interlocked — change `vocab_size` and the
shape of `tok_embed.weight` must follow; change a parameter's shape
and its `data_len` must follow; change anything in the middle of the
file and the trailer is still at the old offset (which is fine) but
any size changes will break.

If you need to surgically modify a checkpoint, write a small Python
or Zig helper that round-trips through `load` + `save` after mutating
the in-memory state. That way validation catches your mistakes.

### "My load succeeded but the model produces garbage"

Something is structurally right but semantically wrong. Most likely
one of:

1. Training continued from a non-deterministic initialisation and
   the model was bad before saving.
2. The tokenizer used at inference doesn't match the one used during
   training (ZTLC v2 does not include the vocab).
3. The input preprocessing (tokenisation, position encoding) differs
   between save and load environments.

ZTLC v2 strictness catches *structural* mismatches. It cannot catch
*semantic* ones.

---

## 12. Exercises

### Exercise 1 — Hex-read a real checkpoint

Run example 06 to produce a `shakespeare_ckpt.bin`, then use a
hex editor (`xxd`, `HxD`, `od -x`) to verify:

1. The first 32 bytes match the header layout in §3.1.
2. You can locate and decode the first parameter name (it should be
   `"tok_embed.weight"`).
3. The last 4 bytes of the file are `"END."`.

If step 3 fails, the trainer was killed before `writer.flush()`.

### Exercise 2 — Introduce and detect corruption

Make a copy of a valid checkpoint. With a hex editor, flip one byte
in:

a) The magic (should fail with `IoError`).
b) The `vocab_size` field in the header (should fail with
   `ShapeMismatch`).
c) A byte in a parameter's f32 data (should load successfully but
   produce slightly different model outputs).

For each, predict which error variant fires before you run the load.

### Exercise 3 — Add a SHA-256 checksum field (ZTLC v3)

Design — do not implement yet — a ZTLC v3 format that adds a
SHA-256 checksum of the entire payload between the header and the
trailer. Sketch the load-time verification logic.

Considerations:

- Where does the checksum go — inside the header (requires
  streaming the file twice) or just before the trailer (requires
  buffering)?
- How do v2 loaders interact with a v3 file? (Answer: they should
  reject at the version check.)
- How do v3 loaders interact with a v2 file? (Answer: they could
  accept v2 files if we reserve a "no checksum present" sentinel.)

Think through the tradeoffs; don't write the code yet.

### Exercise 4 — Cross-machine round-trip

If you have access to both Windows and Linux:

1. Save a checkpoint on Windows (e.g., via `zig build run-example
   -Dexample=06_train_shakespeare`).
2. Copy the file to the Linux machine.
3. Load it there and generate text.

The output should match exactly. If it doesn't, either the random
seed in `Rng.init` isn't being honoured, or the f32 byte order
differs (it shouldn't; both platforms are little-endian).

---

## 13. File reference

| File | What to read |
|---|---|
| `src/nn/model.zig` | `save`, `load`, `collectNamedParams` |
| `src/nn/model.zig` | The four checkpoint tests (round-trip, bad-config, bad-magic, truncated) |
| `src/core/errors.zig` | `IoError`, `ShapeMismatch` — the two error variants the loader returns |

---

## 14. Test commands

```bash
# The four PR-η tests
zig build test -- --test-filter "Checkpoint"

# Full suite
zig build test

# Real-world round-trip
zig build run-example -Dexample=06_train_shakespeare
zig build run-example -Dexample=07_generate
# The second should load the checkpoint written by the first.
```

---

## 15. Closing the Stage 6.5 docs

This is the last of the Stage 6.5 teaching chapters. The series
walks through:

- **02c** `tensor_invariants.md` — the structural guarantees a
  `Tensor` makes to the rest of the system (PR-γ)
- **02d** `storage_and_views.md` — the storage/view split and the
  backend seam (PR-δ)
- **03c** `saved_tensors.md` — how the tape owns its own backward
  data, retiring `keepAlive` (PR-ε)
- **07c** `optimizer_state.md` — `ParamId`-keyed optimizer state
  (PR-ζ)
- **07d** `checkpoint_format.md` — ZTLC v2, every field and every
  validation (PR-η)

Read in sequence, they explain *why Stage 6.5 exists* — a single
coherent refactor that turns a CPU prototype into an educational
library whose internals are consistent, whose bugs are loud, and
whose backend seam is ready for Stage 7's CUDA work on your RTX
4060 Ti box.

Next reading: `docs/08_backends_cuda.md` (to be written during
Stage 7), which explains how the `Storage.cuda = void` placeholder
from chapter 02d becomes the real `CudaStorage`, and how our CPU
parity tests constrain the CUDA implementation to produce
identical results.
