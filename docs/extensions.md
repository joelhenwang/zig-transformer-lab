# Extensions — Learning-by-Implementing Projects

14 extension projects for deepening your understanding of modern LLM
techniques by implementing them in this codebase. Each project lists
prerequisites, files to touch, design decisions, testing strategy,
and acceptance criteria.

**How to use this document:** pick a project at your level, read the
prerequisites, then implement it. The 5 projects marked PLANNED will
be delivered as part of the codebase (Sessions 6-9); the remaining 9
are exercises for you.

**Difficulty scale:**
- **Easy** — single afternoon, < 200 lines of new code
- **Medium** — 1-3 days, 200-600 lines of new code
- **Hard** — 1-2 weeks, 600+ lines, requires deep CUDA or architectural work

---

## Easy Projects (single afternoon each)

---

### E1. Add RMSNorm

**Status:** PLANNED — Session 6

**Goal:** implement Root Mean Square Layer Normalization as an opt-in
replacement for LayerNorm.

**What you'll learn:** why modern LLMs dropped mean-centering and bias
(speed + equivalent quality at scale).

**Prerequisites:**
- Understand LayerNorm in `src/nn/layernorm.zig`
- Read `docs/gap_map.md` section 2.1

**Files you'll touch:**
- Add: `src/nn/rmsnorm.zig` (~120 lines)
- Modify: `src/nn/module.zig` (add `use_rms_norm` to TransformerConfig)
- Modify: `src/nn/block.zig` (dispatch based on config flag)
- Modify: `src/root.zig` (re-export + test block)
- Add: `tools/oracle.py` new case, `tests/fixtures/rmsnorm_3d.ztlt`

**Design decisions:**
- RMSNorm has one learnable parameter `gamma` (no `beta`). This means
  your struct is simpler than LayerNorm.
- The forward is: `x / sqrt(mean(x^2, last_axis) + eps) * gamma`
- You can compose from existing ops: `mul(x, x)` -> `mean(_, last_axis)`
  -> `addScalar(_, eps)` -> `sqrt(_)` -> `div(x, _)` -> `mul(_, gamma)`
- All ops must pass tape for gradient flow.
- Epsilon default: 1e-6 (same as LayerNorm).

**Testing strategy:**
- Unit: `zig build test` — gradcheck on RMSNorm forward
- Oracle: compare against `torch.nn.RMSNorm` output within 5e-5
- Integration: train 50 steps with `use_rms_norm=true`, verify loss decreases

**You know it works when:**
- Gradcheck passes with max_rel_err < 0.01
- Oracle parity within 5e-5 abs
- Training produces decreasing loss

**Stretch goal:** add a CUDA-fused RMSNorm kernel (one kernel for the
entire norm, avoiding 6 separate op launches). This is a Medium project
in disguise.

---

### E2. Add SwiGLU Activation

**Status:** PLANNED — Session 6

**Goal:** implement the SwiGLU MLP as an opt-in replacement for the
GELU-based MLP.

**What you'll learn:** gated activation mechanisms, how three weight
matrices can outperform two at the same parameter budget.

**Prerequisites:**
- Understand the current MLP in `src/nn/mlp.zig`
- Read `docs/gap_map.md` section 2.2

**Files you'll touch:**
- Add: `src/nn/swiglu.zig` (~180 lines)
- Modify: `src/nn/module.zig` (add `use_swiglu` to TransformerConfig)
- Modify: `src/nn/block.zig` (dispatch MLP vs SwiGLU based on config)
- Modify: `src/root.zig` (re-export + test block)
- Add: oracle fixture `swiglu_3d`

**Design decisions:**
- SwiGLU uses 3 Linears: W_gate (d -> d_ff), W_up (d -> d_ff), W_down (d_ff -> d)
- Hidden dim: use `d_ff = (8 * d_model) / 3` rounded to nearest multiple of 8
  (matches Llama's approach to keep parameter count comparable to GELU MLP)
- SiLU activation: `silu(x) = x * sigmoid(x)`. You'll need to add a `silu` op
  to `unary.zig` (trivial: it's just `x * (1 / (1 + exp(-x)))` composed).
- Alternative: compute `gate = mulScalar(x_gate, sigmoid(x_gate))` using
  existing ops without adding silu as a dedicated op.

**Testing strategy:**
- Gradcheck on the full SwiGLU forward
- Oracle parity vs. PyTorch's `nn.Linear` + `F.silu` composition
- Integration: train and compare loss trajectory to GELU MLP baseline

**You know it works when:**
- Both RMSNorm and SwiGLU can be enabled together (`use_rms_norm=true,
  use_swiglu=true`) and training converges.
- Parameter count is within 10% of the GELU MLP (d_ff adjustment).

---

### E3. Add Cosine LR Schedule + Warmup

**Goal:** implement a learning rate scheduler that warms up linearly
then decays with a cosine curve.

**What you'll learn:** why constant LR is suboptimal for transformers,
why warmup prevents early divergence.

**Prerequisites:**
- Understand the Trainer loop in `src/lab/train.zig`
- Read docs/07_cpu_training.md section on gradient clipping (similar concept)

**Files you'll touch:**
- Add: `src/optim/lr_schedule.zig` (~80 lines)
- Modify: `src/lab/train.zig` (call scheduler each step, pass LR to optimizer)
- Modify: `src/nn/module.zig` (add `warmup_steps`, `max_steps` to TrainConfig)

**Design decisions:**
- Formula: `lr(step) = lr_max * 0.5 * (1 + cos(pi * (step - warmup) / (total - warmup)))`
  for step > warmup; linear ramp from 0 to lr_max during warmup.
- Minimum LR: `lr_min = lr_max / 10` (common practice; prevents zero LR).
- The optimizer's `lr` field needs to be settable per-step. Currently AdamW
  takes `lr` at init. You'll modify it to accept an LR parameter per step.

**Testing strategy:**
- Unit: verify LR values at step 0, warmup end, midpoint, final step
- Integration: train and compare loss curve with vs without schedule

**You know it works when:**
- LR starts at 0, reaches lr_max at warmup_steps, decays to lr_min at max_steps
- Training with schedule converges faster/better than constant LR on Shakespeare

---

### E4. Add Gradient Accumulation

**Status:** PLANNED — Session 9

**Goal:** accumulate gradients across multiple micro-batches before
stepping the optimizer, achieving larger effective batch sizes.

**What you'll learn:** the relationship between batch size, learning
rate, and training stability; how to work within memory constraints.

**Prerequisites:**
- Understand the training loop in `src/lab/train.zig`

**Files you'll touch:**
- Modify: `src/lab/train.zig` (~30 lines changed)
- Modify: `src/nn/module.zig` (add `grad_accum_steps` to TrainConfig)

**Design decisions:**
- Scale loss by `1/grad_accum_steps` before backward. This keeps gradient
  magnitudes equivalent to a full large batch.
- Only step optimizer + zero_grad after accumulating all micro-batches.
- Logging: report effective batch size = `B * grad_accum_steps`.
- Gradient clipping: apply once after all micro-batches (on the accumulated gradient).

**Testing strategy:**
- Numerical equivalence: `accum_steps=4, B=1` should give approximately
  the same final params as `B=4, accum_steps=1` (within f32 noise)
- Integration: verify loss trajectory is smooth with large effective batch

**You know it works when:**
- 20 steps with `accum_steps=4, B=2` produces similar loss to
  20 steps with `B=8` (within 5% at each logging point)

---

### E5. Add Top-p (Nucleus) Sampling

**Goal:** implement nucleus sampling as an alternative to top-k for
text generation.

**What you'll learn:** stochastic generation strategies, cumulative
distribution functions, why top-p produces more natural text.

**Prerequisites:**
- Understand `generate()` in `src/lab/train.zig`

**Files you'll touch:**
- Modify: `src/lab/train.zig` (add `top_p` parameter to GenerateOpts,
  add the nucleus sampling logic ~40 lines)

**Design decisions:**
- Sort logits descending. Compute cumulative softmax. Find the smallest
  set whose cumulative probability exceeds `p` (e.g., 0.9). Sample from
  that set only.
- Combine with temperature: apply temperature first, then nucleus cut.
- Top-k and top-p can coexist: apply top-k first, then top-p within the k.

**Testing strategy:**
- Unit: verify that with p=1.0, sampling uses all tokens
- Unit: with p=0.0, sampling picks only the argmax
- Integration: generate text with p=0.9, verify it reads naturally

**You know it works when:**
- Generated text with top_p=0.9, temperature=0.8 reads more coherent
  than top_k=10 alone (subjective but verifiable)

---

## Medium Projects (1-3 days each)

---

### M1. Add RoPE Position Encoding

**Status:** PLANNED — Session 7

**Goal:** implement rotary position embeddings as an opt-in replacement
for learned positional embeddings.

**What you'll learn:** how position information can be encoded via
rotation (the complex-number interpretation), why RoPE enables length
extrapolation.

**Prerequisites:**
- Understand attention in `src/nn/attention.zig`
- Read `docs/gap_map.md` section 1.3
- Read Su et al. (2021), "RoFormer" paper (at least sections 1-3)

**Files you'll touch:**
- Add: `src/nn/rope.zig` (~200 lines: frequency table + applyRope)
- Add: `src/backend/cuda/kernels/rope.cu` (~60 lines: in-place rotation kernel)
- Modify: `src/nn/attention.zig` (apply RoPE to Q and K before attention)
- Modify: `src/nn/model.zig` (skip pos_embed addition when use_rope=true)
- Modify: `src/nn/module.zig` (add `use_rope` to TransformerConfig)
- Modify: `build.zig` (add rope.cu to kernel_names)
- Add: oracle fixture `rope_qk`

**Design decisions:**
- Precompute cos/sin tables at model init: shape `(max_T, d_head/2)`.
  Store as two tensors (cos_cache, sin_cache).
- Frequency formula: `theta_i = 10000^(-2i/d_head)` for dimension pair i.
- Apply to Q and K only (not V). RoPE modifies the dot product to encode
  relative position.
- The rotation is: for each pair (x[2i], x[2i+1]):
  `new[2i] = x[2i]*cos - x[2i+1]*sin`
  `new[2i+1] = x[2i]*sin + x[2i+1]*cos`
- CUDA kernel: one thread per (batch, position, dim_pair). In-place.

**Testing strategy:**
- Unit: RoPE(x, pos=0) should equal x (cos(0)=1, sin(0)=0)
- Unit: RoPE is invertible (apply twice with negative angle = identity)
- Gradcheck: backprop through RoPE
- Oracle: compare against PyTorch's reference implementation
- CUDA parity: CPU vs GPU within f32 noise
- Integration: train 100 steps with `use_rope=true`, verify convergence

**You know it works when:**
- Oracle parity within 1e-5 abs
- Training converges with RoPE (loss decreases)
- Model can generate text at sequence lengths not seen during training
  (basic extrapolation test)

**Stretch goal:** implement YaRN (Yet another RoPE Notation) for better
long-context extrapolation.

---

### M2. Add KV Cache for Inference

**Goal:** implement key-value caching so autoregressive generation
doesn't recompute K/V for all previous tokens each step.

**What you'll learn:** the fundamental distinction between training mode
(all positions processed in parallel) and inference mode (one token at
a time with cache).

**Prerequisites:**
- Understand attention in `src/nn/attention.zig`
- Understand generation in `src/lab/train.zig` (generate function)
- Read `docs/gap_map.md` section 1.2

**Files you'll touch:**
- Add: `src/nn/kv_cache.zig` (~150 lines: cache struct, append, get)
- Modify: `src/nn/attention.zig` (accept optional cache, grow K/V)
- Modify: `src/nn/model.zig` (per-layer cache allocation/management)
- Modify: `src/lab/train.zig` (use cache in generate(), not in train())

**Design decisions:**
- Cache shape per layer: `(B, n_heads, max_seq_len, d_head)` for both K and V
- Pre-allocate for max_seq_len at generation start. Append new K/V each step.
- Forward signature change: attention needs a `step: usize` parameter in
  inference mode to know where in the cache to write.
- Training path: no cache (full parallel forward). Inference path: with cache.
- Config: add `max_seq_len: usize = 512` for cache pre-allocation.

**Testing strategy:**
- Correctness: generate with cache vs. without cache should produce
  identical token sequences (bit-exact with same seed)
- Speed: measure tokens/second with and without cache. Expect ~T/2 x
  speedup on average (where T is sequence length).

**You know it works when:**
- Identical generation output with/without cache (correctness)
- Measurable speedup for sequences > 32 tokens

---

### M3. Add BPE Tokenizer

**Status:** PLANNED — Session 8

**Goal:** implement byte-pair encoding tokenizer that handles arbitrary
text (including code, Unicode, special characters) with a fixed vocab.

**What you'll learn:** how subword tokenization works, why it replaced
word-level tokenization in every modern LLM.

**Prerequisites:**
- Understand the word tokenizer in `src/data/dataset.zig`
- Read `docs/gap_map.md` section 10.1
- Read Karpathy's "minbpe" Python implementation (reference)

**Files you'll touch:**
- Add: `src/data/bpe.zig` (~600 lines: train, encode, decode, save, load)
- Add: `src/data/bpe_train.zig` (~200 lines: pair-frequency algorithm)
- Add: `tools/train_bpe.py` (~50 lines: reference implementation)
- Add: `examples/13_bpe_demo.zig` (~80 lines: train + demo)
- Modify: `src/lab/train.zig` (support BPE dataset)
- Add: `tests/fixtures/bpe_shakespeare.json`

**Design decisions:**
- Start with byte vocabulary (256 base tokens). No UNK token needed.
- Merge algorithm: iteratively merge the most frequent adjacent pair.
- Vocab size target: 8000 tokens for Shakespeare (small but educational).
- Encoding: greedy left-to-right merge application (standard BPE).
- File format: JSON with `vocab`, `merges`, `special_tokens` fields.
- Special tokens: `<|endoftext|>` at minimum.

**Testing strategy:**
- Roundtrip: encode(decode(tokens)) == tokens for all test strings
- Parity: compare tokenization against Python reference (`tools/train_bpe.py`)
- Determinism: same corpus + same vocab_size -> same merges
- Edge cases: empty string, single byte, only spaces, Unicode

**You know it works when:**
- Roundtrip is perfect for arbitrary Unicode strings
- Vocab of 8000 on Shakespeare produces ~3.5 tokens per word (typical BPE ratio)
- Can train the transformer using BPE tokens and loss decreases

---

### M4. Add LoRA Adapters

**Goal:** implement Low-Rank Adaptation so you can fine-tune a frozen
model with a tiny trainable parameter count.

**What you'll learn:** parameter-efficient fine-tuning, low-rank
decomposition, why r=16 is usually sufficient.

**Prerequisites:**
- Understand Linear in `src/nn/linear.zig`
- Understand the optimizer in `src/optim/adamw.zig`
- Read `docs/gap_map.md` section 11.1
- Read Hu et al. (2021), "LoRA" paper (sections 1-4)

**Files you'll touch:**
- Add: `src/nn/lora.zig` (~200 lines: LoraLinear wrapper)
- Modify: `src/nn/model.zig` (method to apply LoRA to selected layers)
- Modify: parameter collection (only collect LoRA params for optimizer)

**Design decisions:**
- LoRA wraps a Linear: `y = x @ W + x @ A @ B` where W is frozen.
- A is (d_in, r) initialized with Kaiming, B is (r, d_out) initialized to 0.
  This means at init LoRA contribution is 0 (model starts unchanged).
- Alpha scaling: `y = x @ W + (alpha/r) * x @ A @ B`. Default alpha = r.
- Only A and B are in the parameter list for the optimizer.
- Apply LoRA to: Q, K, V, output projections (standard configuration).
- Rank `r`: default 16 (good balance of quality and parameter count).

**Testing strategy:**
- At init: LoRA model produces identical output to base model (B=0)
- After training: LoRA params change but base params don't
- Integration: fine-tune on a different text (e.g., tiny.txt) with LoRA

**You know it works when:**
- Base weights unchanged after fine-tuning (frozen correctly)
- Loss decreases during LoRA fine-tuning
- Trainable parameters = ~0.5% of total (for r=16 on a 2/2/64 model)

---

### M5. Add Activation Checkpointing

**Goal:** trade compute for memory by discarding intermediate
activations during forward and recomputing them during backward.

**What you'll learn:** the compute-memory tradeoff, how to modify
an autograd tape for selective recomputation.

**Prerequisites:**
- Understand the tape in `src/autograd/tape.zig`
- Understand keepAlive in tape mechanics
- Read `docs/gap_map.md` section 5.2

**Files you'll touch:**
- Modify: `src/autograd/tape.zig` (add checkpoint boundary marking)
- Modify: `src/nn/block.zig` (mark TransformerBlock as a checkpoint boundary)
- Modify: `src/lab/train.zig` (enable/disable checkpointing via config)

**Design decisions:**
- Checkpoint at block boundaries: save input to each TransformerBlock,
  discard all intermediates within the block. During backward, re-run
  the block's forward to regenerate intermediates.
- Implementation: instead of keepAlive on intermediates, store a
  "recompute function" (closure or function pointer + saved input).
- Memory savings: proportional to number of layers. For N layers,
  memory goes from O(N * activations_per_layer) to O(sqrt(N) * ...).
- At our scale (2 layers), savings are minimal — this is a learning
  exercise, not a practical optimization for this model size.

**Testing strategy:**
- Correctness: gradients with/without checkpointing must be identical
- Memory: measure peak allocation with/without (expect ~50% reduction for N=4 layers)
- Integration: train and compare loss curves (must be identical)

**You know it works when:**
- Bit-exact gradients with/without checkpointing
- Peak memory measurably lower with checkpointing enabled

---

### M6. Mini-Llama (RMSNorm + SwiGLU + RoPE combined)

**Goal:** enable all three modern components together and verify the
combined architecture trains correctly.

**What you'll learn:** how architectural choices interact, ablation
methodology.

**Prerequisites:**
- Complete E1 (RMSNorm), E2 (SwiGLU), and M1 (RoPE) first.

**Files you'll touch:**
- Modify: config in training examples (set all three flags to true)
- Add: `examples/14_train_mini_llama.zig` (~80 lines)
- Add: comparison script or doc section showing loss curves

**Design decisions:**
- Config: `use_rms_norm=true, use_swiglu=true, use_rope=true`
- Also remove bias from Linear layers (Llama-style). Add `use_bias: bool = true`
  to Linear and set false for mini-Llama config.
- Test at 2/2/64 AND at a slightly larger scale (e.g., 4/4/128) to
  see if the architectural benefits emerge with more capacity.

**Testing strategy:**
- All individual component tests still pass
- Combined training converges (loss < 5.0 after 200 steps on Shakespeare)
- Optional: compare final loss with vs. without each component (ablation)

**You know it works when:**
- Mini-Llama config trains stably on Shakespeare
- Can generate coherent text after 500+ steps

---

## Hard Projects (1-2 weeks each)

---

### H1. Add f16 Mixed Precision

**Note:** this conflicts with locked decision D7 (f32-only). Listed as
a learning exercise — if you attempt it, understand you're modifying
the fundamental dtype assumption throughout the codebase.

**Goal:** support FP16 forward/backward with FP32 master weights and
optimizer state, plus dynamic loss scaling.

**What you'll learn:** the numerical challenges of reduced precision,
loss scaling to prevent gradient underflow, where precision matters
most in a training pipeline.

**Prerequisites:**
- Understand Storage union in `src/tensor/tensor.zig`
- Read `docs/gap_map.md` section 3
- Read Micikevicius et al. (2018), "Mixed Precision Training"

**Files you'll touch:**
- Modify: `src/core/dtype.zig` (add f16, bf16 variants)
- Modify: `src/tensor/tensor.zig` (Storage generic over dtype)
- Modify: every op file (dtype dispatch)
- Add: `src/optim/grad_scaler.zig` (dynamic loss scaling)
- Add: CUDA kernels for FP16 variants (or use cuBLAS cublasGemmEx)
- Modify: `src/lab/train.zig` (autocast context + unscaling)

**Estimated scope:** 4-8 weeks, ~2000+ lines of changes across 30+ files.

**You know it works when:**
- Training in mixed precision produces similar loss to FP32 (within 1%)
- No inf/NaN in gradients (loss scaler working)
- Measurable speedup from tensor cores (~2x at large enough N)

---

### H2. Add Flash Attention Kernel

**Goal:** implement the flash attention algorithm as a CUDA kernel with
shared-memory tiling and online softmax.

**What you'll learn:** the deepest CUDA optimization in modern LLMs,
online algorithms, how to achieve near-peak memory bandwidth.

**Prerequisites:**
- Complete `docs/cuda_depth.md` (especially sections 3, 5, 9)
- Read Dao (2022), "FlashAttention" paper (full algorithm)
- Understand shared memory tiling from tiled_gemm.cu

**Files you'll touch:**
- Add: `src/backend/cuda/kernels/flash_attn.cu` (~400 lines)
- Add: `src/backend/cuda/flash_attn.zig` (~150 lines: host launcher)
- Modify: `src/nn/attention.zig` (use flash kernel when CUDA + config flag)
- Add: comprehensive correctness test against naive attention

**Design decisions:**
- Block sizes: B_r = B_c = 64 (good for SM 8.9 shared memory budget)
- Causal mask: handle in the tiling loop (skip blocks above diagonal)
- Online softmax: maintain running max + running sum per query row
- Backward: implement the recomputation-based backward from the paper
  (or start forward-only and add backward later)
- Start with FP32 (since we're D7-locked). Real flash attention uses FP16.

**Testing strategy:**
- Numerical parity: flash output vs. naive attention within 1e-4
- Memory measurement: peak memory with flash vs. without
- Speed measurement: flash vs. naive at T=256, 512, 1024

**You know it works when:**
- Output matches naive attention within 1e-4 for T=256
- Memory usage is O(T) not O(T^2)
- At T=1024 flash is measurably faster than naive

---

### H3. Scale to 50M Parameters on RTX 4060 Ti

**Goal:** train a model large enough to produce genuinely good text on
Shakespeare, pushing the hardware to its limits.

**What you'll learn:** where memory bottlenecks appear at scale, which
optimizations actually matter, how to estimate compute requirements.

**Prerequisites:**
- Complete E4 (gradient accumulation) and ideally M5 (checkpointing)
- Understand memory budget: 16 GB GDDR6, model + optimizer + activations must fit

**Files you'll touch:**
- Modify: training configs (n_layer=12, n_head=12, d_model=768)
- Possibly: add gradient accumulation + checkpointing to fit in memory
- Add: `examples/15_train_50m.zig`

**Design decisions:**
- Architecture: 12 layers, 12 heads, d_model=768, d_ff=3072
  Parameters: ~50M (embeddings + 12 blocks + output projection)
- Memory budget breakdown:
  - Parameters (FP32): 50M * 4 = 200 MB
  - Optimizer state (AdamW m + v): 50M * 2 * 4 = 400 MB
  - Activations (per layer): depends on B, T, d. For B=4, T=256: ~50 MB per layer
  - Total estimate: ~1.5-2 GB. Fits easily in 16 GB!
- Training: Shakespeare (1.1 MB), B=4, T=256, ~5000 steps
- Expected time: ~15-30 minutes on RTX 4060 Ti

**Testing strategy:**
- Model compiles and runs without OOM
- Loss converges below 3.0 (good language modeling for Shakespeare)
- Generated text is recognizably Shakespearean

**You know it works when:**
- Final perplexity < 20 on held-out Shakespeare text
- Can generate full coherent sentences (not just plausible n-grams)

---

## Project status

| # | Project | Difficulty | Status |
|---|---|---|---|
| E1 | RMSNorm | Easy | PLANNED (Session 6) |
| E2 | SwiGLU | Easy | PLANNED (Session 6) |
| E3 | Cosine LR schedule | Easy | Exercise |
| E4 | Gradient accumulation | Easy | PLANNED (Session 9) |
| E5 | Top-p sampling | Easy | Exercise |
| M1 | RoPE | Medium | PLANNED (Session 7) |
| M2 | KV cache | Medium | Exercise |
| M3 | BPE tokenizer | Medium | PLANNED (Session 8) |
| M4 | LoRA | Medium | Exercise |
| M5 | Activation checkpointing | Medium | Exercise |
| M6 | Mini-Llama combined | Medium | Exercise (after E1+E2+M1) |
| H1 | f16 mixed precision | Hard | Exercise (conflicts D7) |
| H2 | Flash attention | Hard | Exercise |
| H3 | Scale to 50M | Hard | Exercise (after E4) |

---

## Suggested order (if you want to do them all)

1. E1 + E2 (RMSNorm + SwiGLU) — delivered in Session 6
2. E3 (cosine LR) — quick win after understanding optimizer
3. E4 (gradient accumulation) — delivered in Session 9
4. E5 (top-p) — quick generation improvement
5. M1 (RoPE) — delivered in Session 7
6. M3 (BPE) — delivered in Session 8
7. M6 (mini-Llama) — combines E1 + E2 + M1
8. M2 (KV cache) — unlocks fast inference
9. M4 (LoRA) — parameter-efficient fine-tuning
10. M5 (activation checkpointing) — memory optimization
11. H3 (50M model) — uses E4 + possibly M5
12. H2 (flash attention) — deep CUDA project
13. H1 (mixed precision) — architectural rework
14. (Bonus: combine H1 + H2 + H3 for the ultimate LLM training setup)
