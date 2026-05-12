# Gap Map — What This Codebase Does Not Teach (Yet)

An honest catalog of modern LLM techniques not implemented here, each
with conceptual explanation, practical importance, reading references,
difficulty assessment, and current status.

**Purpose:** you need to know what's missing so you don't mistake this
codebase for a complete production system. Some gaps are being filled
(see status markers); others are intentionally deferred or out of scope.

**Status markers:**

- PLANNED — will be filled in a future session
- DEFERRED — conflicts with a locked design decision
- DOC-ONLY — too large for this codebase; learn from external sources
- DELIVERED — implemented; see referenced commit

---

## 1. Modern Attention Mechanisms

### 1.1 Flash Attention

**Status:** DOC-ONLY (covered conceptually in `docs/cuda_depth.md` S9)

**What it is.**
Standard attention materializes the full scores matrix `S = Q @ K^T`
of shape `(B, H, T, T)`. For T=4096 and H=32, this is 2 GB of f32.
Flash attention tiles the computation along the sequence dimension,
computing softmax in an online fashion so that at no point is the full
`T x T` matrix resident in GPU memory.

**Why it matters.**
Without flash attention, training with context length > 2048 on 16 GB
GPUs is impossible (OOM). With it, memory scales as O(T) instead of
O(T^2). Speed also improves 2-3x because the tiled access pattern is
more bandwidth-efficient (data stays in SRAM, not DRAM).

**Rough idea.**
1. Divide Q into blocks of size B_r (e.g., 64 rows)
2. For each Q block, iterate over K/V blocks of size B_c
3. Compute local scores, apply online softmax (keep running max + sum)
4. Accumulate output block-by-block
5. Never write the full `T x T` matrix to HBM

The "online softmax" trick: `softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))`.
When computing blockwise, you maintain a running max and re-scale
previous blocks when a new max is found.

**Where to read more.**
- Paper: Dao (2022), "FlashAttention: Fast and Memory-Efficient Exact
  Attention with IO-Awareness." The foundational paper with the tiling
  algorithm and IO-complexity analysis.
- Paper: Dao (2023), "FlashAttention-2: Faster Attention with Better
  Parallelism and Work Partitioning." Improved version with better
  GPU utilization.
- Code: `flash-attn` repository (Triton + CUDA implementations).
  Start with the Triton version — more readable than raw CUDA.
- Book: PMPP (Kirk & Hwu), Chapter 6 "Tiling." Covers the general
  shared-memory tiling pattern that flash attention builds on.

**How hard to add here:** Hard (3-4 weeks). Requires shared-memory
CUDA kernel with online softmax, careful numerical testing against
naive attention, and new op integration. See `docs/extensions.md` H2.

---

### 1.2 KV Cache

**Status:** DOC-ONLY (described in `docs/extensions.md` M2)

**What it is.**
During autoregressive generation, each new token attends to all
previous tokens. Without caching, you recompute K and V projections
for every prior token at every step — O(T^2) redundant work.

KV cache stores the K and V tensors from prior steps and only
computes K/V for the new token, then concatenates with the cache.

**Why it matters.**
Inference speed scales from O(T^2) per token to O(T) per token.
For a 4096-token response, this is a ~2000x inference speedup.
Every production LLM serving system uses KV caching.

**Rough idea.**
```
# At generation step t:
K_new = Linear_k(x_t)             # shape (1, d_head)
V_new = Linear_v(x_t)             # shape (1, d_head)
K_cache = concat(K_cache, K_new)  # shape (t, d_head)
V_cache = concat(V_cache, V_new)  # shape (t, d_head)
Q = Linear_q(x_t)                 # shape (1, d_head)
attn = softmax(Q @ K_cache^T / sqrt(d)) @ V_cache
```

**Where to read more.**
- Karpathy, "nanoGPT" repository, `generate()` function. Shows the
  basic cache pattern in PyTorch.
- Kwon et al. (2023), "PagedAttention." Extends KV caching with
  virtual-memory-style block management for serving.

**How hard to add here:** Medium (1 week). Requires separating
"training forward" from "inference forward" in the model. See
`docs/extensions.md` M2.

---

### 1.3 Rotary Position Embedding (RoPE)

**Status:** PLANNED — Session 7

**What it is.**
RoPE encodes position by rotating Q and K vectors in 2D subspaces.
The rotation angle is a function of position and frequency: higher
dimensions rotate more slowly, encoding position at multiple scales.

Unlike learned positional embeddings (our current approach), RoPE:
- Has no learnable parameters (pure mathematical function)
- Naturally extends to longer sequences than seen during training
- Encodes relative position (Q @ K^T depends on position *difference*)

**Why it matters.**
Every modern LLM (Llama, Mistral, Gemma, Phi) uses RoPE. It's the
standard. Learned positional embeddings (what this codebase uses) are
considered outdated for language models > 100M parameters.

**Rough idea.**
For each pair of dimensions (2i, 2i+1) in the head:
```
theta_i = 10000^(-2i/d_head)
cos_val = cos(pos * theta_i)
sin_val = sin(pos * theta_i)

q_new[2i]   = q[2i] * cos_val - q[2i+1] * sin_val
q_new[2i+1] = q[2i] * sin_val + q[2i+1] * cos_val
```
Apply to Q and K (not V). The dot product `q_rotated @ k_rotated^T`
then depends on `pos_q - pos_k` (relative position).

**Where to read more.**
- Paper: Su et al. (2021), "RoFormer: Enhanced Transformer with
  Rotary Position Embedding." The original paper with the mathematical
  derivation and the complex-number interpretation.
- Code: `llama2.c` by Karpathy, the `rope()` function. ~20 lines of C
  that implement the rotation. Very readable.

**How hard to add here:** Medium (3-4 days). See Session 7 plan.

---

### 1.4 Grouped-Query Attention (GQA)

**Status:** DOC-ONLY (described in `docs/extensions.md` as a stretch goal)

**What it is.**
Standard multi-head attention (MHA) has separate K, V projections per
head. GQA shares K/V across groups of heads (e.g., 8 Q heads share
1 K/V head group). Reduces KV cache memory proportionally.

**Why it matters.**
At inference, KV cache size = `2 * n_layers * n_heads * T * d_head * sizeof(f16)`.
GQA with 8-to-1 sharing reduces this by 8x. Llama 3 uses GQA with
8 KV heads for 32 Q heads.

**Rough idea.**
```
# Instead of: K_h = x @ W_k_h for each head h
# GQA: K_g = x @ W_k_g for each KV-group g
# Then: head h uses K_{h // group_size}
```

**Where to read more.**
- Paper: Ainslie et al. (2023), "GQA: Training Generalized
  Multi-Query Transformer Models from Multi-Head Checkpoints."
- Llama 3 technical report. GQA config details.

**How hard to add here:** Medium (3-5 days). Modify `attention.zig`
to accept `n_kv_heads` parameter. When `n_kv_heads < n_heads`, repeat
K/V across Q heads.

---

### 1.5 ALiBi (Attention with Linear Biases)

**Status:** DOC-ONLY

**What it is.**
Instead of position embeddings, ALiBi adds a pre-defined linear bias
to the attention scores: `score[i,j] -= m * |i-j|` where `m` is a
head-specific slope. Encodes recency bias directly into attention.

**Why it matters.**
Simpler than RoPE (no sin/cos computation). Excellent length
extrapolation. Used by some models (BLOOM, MPT).

**Where to read more.**
- Paper: Press et al. (2022), "Train Short, Test Long: Attention
  with Linear Biases Enables Input Length Extrapolation."

**How hard to add here:** Easy (2 days). Add a constant bias tensor
to the attention scores before softmax, conditioned on a config flag.

---

## 2. Modern Activations and Norms

### 2.1 RMSNorm

**Status:** PLANNED — Session 6

**What it is.**
Root Mean Square Layer Normalization. Simpler than LayerNorm: it
normalizes by the RMS of the input without subtracting the mean and
without a bias (beta) parameter.

```
RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma
```

LayerNorm does: `(x - mean(x)) / sqrt(var(x) + eps) * gamma + beta`
RMSNorm drops the mean-centering and the beta bias.

**Why it matters.**
- ~15-30% faster than LayerNorm (no mean computation, no beta)
- Used by Llama, Mistral, Gemma, and most modern LLMs
- Empirically equivalent quality to LayerNorm at LLM scale
- Fewer parameters (no beta)

**Rough idea.**
```python
# Python pseudocode:
def rms_norm(x, gamma, eps=1e-6):
    rms = sqrt(mean(x ** 2, dim=-1, keepdim=True) + eps)
    return (x / rms) * gamma
```

One learnable parameter `gamma` of shape `(d_model,)`.
No learnable `beta`.

**Where to read more.**
- Paper: Zhang & Sennrich (2019), "Root Mean Square Layer
  Normalization." The original paper with ablation studies.
- Code: Llama source code, `class RMSNorm`. ~10 lines of PyTorch.

**How hard to add here:** Easy (1 day). ~120 lines of Zig. Gradcheck
tests. Oracle parity fixture. See Session 6 plan.

---

### 2.2 SwiGLU / GeGLU

**Status:** PLANNED — Session 6

**What it is.**
Gated Linear Unit variants that replace the standard MLP:

Standard MLP:    `FFN(x) = GELU(x @ W1) @ W2`
SwiGLU:          `FFN(x) = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down`
GeGLU:           `FFN(x) = (GELU(x @ W_gate) * (x @ W_up)) @ W_down`

The "gate" mechanism: one projection produces a gate that modulates
another projection. This gives the network a way to selectively
suppress information in the hidden layer.

**Why it matters.**
- ~1% perplexity improvement over standard GELU MLP at same compute
- Used by Llama, Mistral, PaLM, and most modern LLMs
- Standard since 2022 — the field consensus is "always use SwiGLU"

**Rough idea.**
```python
# Standard MLP (what this codebase has):
hidden = gelu(x @ fc1.weight)
out = hidden @ fc2.weight

# SwiGLU MLP:
gate = silu(x @ W_gate)   # SiLU = x * sigmoid(x)
up = x @ W_up
hidden = gate * up         # element-wise gating
out = hidden @ W_down
```

Note: SwiGLU uses 3 weight matrices vs. 2 for standard MLP. To match
parameter count, the hidden dimension is reduced from 4*d to (8/3)*d.

**Where to read more.**
- Paper: Shazeer (2020), "GLU Variants Improve Transformer." Systematic
  comparison of activation variants. Established SwiGLU as the winner.
- Paper: Dauphin et al. (2017), "Language Modeling with Gated
  Convolutional Networks." Original GLU paper (different context).

**How hard to add here:** Easy (2 days). ~180 lines of Zig (SwiGLU MLP
struct with 3 Linear layers). See Session 6 plan.

---

## 3. Mixed Precision Training

**Status:** DEFERRED (locked decision D7: f32 only)

**What it is.**
Train with a mix of FP16/BF16 (for forward/backward compute) and FP32
(for optimizer state + master weights). Cuts memory ~2x, doubles
throughput on hardware with tensor cores.

**Why it matters.**
- Without mixed precision, a 7B model needs ~28 GB just for parameters
  (in f32). With FP16 parameters: ~14 GB.
- Tensor cores (Ampere+) achieve ~2x throughput for FP16 vs FP32 matmul.
- Every production LLM training run uses mixed precision. It's not
  optional at scale.

**Rough idea.**
1. Keep "master weights" in FP32 (full precision for optimizer updates)
2. Cast weights to FP16/BF16 for forward pass
3. Compute loss in FP32 (numerical stability)
4. Scale loss by a large factor before backward (prevent FP16 underflow)
5. Unscale gradients after backward
6. If inf/NaN detected: skip step, reduce scale factor
7. Update master weights in FP32

BF16 is preferred over FP16 because it has the same exponent range as
FP32 (no need for loss scaling). FP16 has limited range and requires
careful scaling.

**Where to read more.**
- Paper: Micikevicius et al. (2018), "Mixed Precision Training." The
  foundational paper from Nvidia on loss scaling technique.
- Nvidia, "Training with Mixed Precision" developer guide. Practical
  recommendations and gotchas.
- PyTorch AMP documentation. Shows the `autocast` + `GradScaler` API.

**How hard to add here:** Very hard (4-8 weeks). Would require making
`Storage` generic over dtype, adding autocast context, implementing loss
scaling in the Trainer, and adding FP16 kernel variants. Conflicts with
D7 (f32-only design decision).

**Why we're not doing it:** the pedagogical value is in understanding
the CONCEPT (this section), not in the plumbing (which is mostly dtype
generics and special-casing). Read the Nvidia guide + PyTorch AMP docs
for the implementation perspective.

---

## 4. Tensor Cores

**Status:** DOC-ONLY (pedagogical WMMA demo in `docs/cuda_depth.md` S5)

**What it is.**
Dedicated hardware units on Nvidia GPUs (Volta+) that compute small
matrix multiplications in a single clock cycle. A single tensor core
computes a 4x4x4 matrix multiply-accumulate (D = A*B + C) per cycle.

**Why it matters.**
- RTX 4060 Ti: 165 TFLOPS FP16 tensor core vs. 22 TFLOPS FP32 CUDA core
- That's a 7.5x throughput difference for operations that can use them
- cuBLAS automatically uses tensor cores when inputs are FP16/BF16/TF32

**Rough idea.**
```
// CUDA WMMA (Warp Matrix Multiply-Accumulate):
wmma::fragment<wmma::matrix_a, 16, 16, 16, half> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
wmma::load_matrix_sync(a_frag, a_ptr, 16);
wmma::load_matrix_sync(b_frag, b_ptr, 16);
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);  // ONE instruction!
wmma::store_matrix_sync(c_ptr, c_frag, 16, wmma::mem_row_major);
```

The hardware multiplies 16x16 FP16 matrices and accumulates in FP32.
A whole warp (32 threads) cooperates on one 16x16x16 tile.

**Where to read more.**
- Nvidia, "CUDA C++ Programming Guide," Appendix B.27 "Warp Matrix
  Functions." Reference for WMMA API.
- `docs/cuda_depth.md` section 5 in this repo. Worked example with
  a pedagogical WMMA kernel you can run.
- CUTLASS repository. Production GEMM templates using tensor cores.

**How hard to add here:** already partially done (Session 3 adds a demo).
Full integration requires FP16 dtypes (see gap #3).

---

## 5. Memory Optimization

### 5.1 Gradient Accumulation

**Status:** PLANNED — Session 9

**What it is.**
Instead of updating weights after every batch, accumulate gradients
across N micro-batches and then step the optimizer once. Effective
batch size = N * micro_batch_size without N * memory.

**Why it matters.**
- Larger effective batch size stabilizes training (smoother gradients)
- Works within fixed GPU memory (same memory as small batch)
- Essential when batch size is limited by memory (large models)

**Rough idea.**
```python
for micro_step in range(accum_steps):
    loss = model(batch[micro_step]) / accum_steps
    loss.backward()  # gradients accumulate (+=)
# After accum_steps:
optimizer.step()
optimizer.zero_grad()
```

The `/accum_steps` scaling ensures gradient magnitude matches what
you'd get with the full batch.

**Where to read more.**
- PyTorch documentation, "Gradient Accumulation" tutorial. Clear
  explanation with code.
- Deep Learning (Goodfellow et al.), Chapter 8.1.3 on batch size
  effects.

**How hard to add here:** Easy (2 days). Add `grad_accum_steps: u32`
to `TrainConfig`, modify the training loop to delay optimizer steps.
See Session 9 plan.

---

### 5.2 Activation Checkpointing

**Status:** DOC-ONLY (described in `docs/extensions.md` M5)

**What it is.**
During forward pass, discard intermediate activations instead of
keeping them for backward. During backward, recompute each layer's
activations on-the-fly. Trades compute (2x forward) for memory
(O(sqrt(N)) instead of O(N) activations stored).

**Why it matters.**
- A 7B model's activations at batch size 1, seq_len 4096 can exceed
  10 GB. Checkpointing reduces this to ~1-2 GB.
- Enables training larger models on smaller GPUs.
- Standard practice for any model > 1B parameters.

**Where to read more.**
- Paper: Chen et al. (2016), "Training Deep Nets with Sublinear
  Memory Cost." Original gradient checkpointing paper.
- PyTorch `torch.utils.checkpoint` documentation and source code.

**How hard to add here:** Medium (1 week). Requires tape modifications
to mark "checkpoint boundaries" and trigger recomputation. See
`docs/extensions.md` M5.

---

### 5.3 CPU Offloading

**Status:** DOC-ONLY

**What it is.**
Keep optimizer states (or even parameters) on CPU RAM and transfer to
GPU only when needed. ZeRO-Offload moves optimizer state to CPU;
ZeRO-Infinity also offloads parameters and gradients.

**Why it matters.**
- A 7B model with AdamW needs ~56 GB of optimizer state in FP32.
  CPU RAM is cheaper and larger than GPU memory.
- Enables training models larger than GPU memory (at the cost of
  PCIe transfer bandwidth).

**Where to read more.**
- Paper: Ren et al. (2021), "ZeRO-Offload: Democratizing
  Billion-Scale Model Training." Offloading optimizer to CPU.
- DeepSpeed documentation, ZeRO Stage 3 with offloading.

**How hard to add here:** Medium (1 week), but limited pedagogical
value at our scale (16 GB GPU handles everything we train).

---

## 6. Distributed Training

**Status:** DEFERRED (locked decision D14: single GPU only)

**What it is.**
Splitting training across multiple GPUs. Three main strategies:

1. **Data Parallel (DDP):** Each GPU has full model copy; different data
   per GPU; sync gradients after backward. Simple, scales well.
2. **Fully Sharded (FSDP / ZeRO):** Shard parameters + optimizer state
   across GPUs. Each GPU holds only 1/N of the model. Gather params
   for forward, reduce-scatter grads for backward.
3. **Tensor Parallel:** Split individual weight matrices across GPUs.
   Each GPU computes part of each layer. Requires custom all-reduce
   communication patterns per layer type.
4. **Pipeline Parallel:** Different layers on different GPUs. Micro-
   batching to keep all GPUs busy (reduces "pipeline bubble").

**Why it matters.**
- A 70B model in FP16 = 140 GB. No single GPU can hold it.
- Training a 7B model in reasonable time (days, not months) requires
  8+ GPUs.
- Every frontier LLM uses a combination of all four strategies.

**Rough idea (DDP):**
```python
# Each GPU rank:
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    all_reduce(model.gradients)  # average grads across ranks
    optimizer.step()
```

**Where to read more.**
- Paper: Rajbhandari et al. (2020), "ZeRO: Memory Optimizations
  Toward Training Trillion Parameter Models." The foundation of FSDP.
- Paper: Shoeybi et al. (2019), "Megatron-LM." Tensor + pipeline
  parallelism for large models.
- PyTorch `DistributedDataParallel` documentation. Clearest DDP API.

**How hard to add here:** Not feasible without multi-GPU hardware.
Our remote has one RTX 4060 Ti. D14 keeps this out of scope.

---

## 7. Advanced CUDA

**Status:** DOC-ONLY (covered in `docs/cuda_depth.md`)

This entire category is addressed by the companion document
`docs/cuda_depth.md`, which covers:

- Shared memory tiling (with runnable tiled GEMM kernel)
- Warp-level primitives (__shfl_down, __ballot)
- Tensor cores via WMMA (with runnable demo kernel)
- Kernel fusion principles (walks through existing CE kernel)
- CUDA streams and graphs (conceptual)
- Profiling with Nsight Compute (conceptual, ERR_NVGPUCTRPERM blocks)
- Flash attention conceptual walkthrough

See `docs/cuda_depth.md` for the full treatment.

---

## 8. Inference Optimization

### 8.1 KV Cache

See section 1.2 above.

### 8.2 Continuous Batching

**Status:** DOC-ONLY

**What it is.**
In static batching, all requests in a batch must finish before any new
request starts. In continuous batching, completed requests are replaced
immediately with new ones. GPU utilization goes from 30-50% to 90%+.

**Where to read more.**
- Paper: Yu et al. (2022), "Orca: A Distributed Serving System for
  Transformer-Based Generative Models." Introduced iteration-level
  scheduling.
- Kwon et al. (2023), "vLLM" and PagedAttention. Production system.

**How hard to add here:** Hard (2-3 weeks). Requires request scheduler,
variable-length batch management, paged KV cache.

### 8.3 Speculative Decoding

**Status:** DOC-ONLY

**What it is.**
Use a small "draft" model to generate N candidate tokens cheaply, then
verify them all at once with the large model (single forward pass).
If most candidates are accepted, you get N tokens for the cost of ~1
large-model forward.

**Where to read more.**
- Paper: Leviathan et al. (2023), "Fast Inference from Transformers
  via Speculative Decoding."
- Paper: Chen et al. (2023), "Accelerating Large Language Model
  Decoding with Speculative Sampling."

**How hard to add here:** Hard (2-3 weeks). Needs two models, acceptance
logic, and careful handling of the probability distributions.

---

## 9. Quantization

**Status:** DOC-ONLY

**What it is.**
Represent model weights (and sometimes activations) with fewer bits:
INT8 (8-bit integer), INT4 (4-bit integer), or even lower. Reduces
model size and memory bandwidth requirements for inference.

### Types:

- **Post-Training Quantization (PTQ):** quantize after training.
  Simple but may lose quality.
- **Quantization-Aware Training (QAT):** simulate quantization during
  training so the model learns to be robust to it.
- **Weight-only quantization:** only weights are quantized; activations
  stay in FP16. Simpler, works well for inference.

**Why it matters.**
- A 7B model in FP16: 14 GB. In INT4: 3.5 GB. Fits on consumer GPUs.
- Inference is memory-bandwidth-bound; fewer bits = faster loads = faster inference.
- Quality loss is typically < 1% perplexity with good calibration.

**Where to read more.**
- Paper: Frantar et al. (2023), "GPTQ: Accurate Post-Training
  Quantization for Generative Pre-trained Transformers." The standard
  INT4 weight quantization method.
- Paper: Lin et al. (2024), "AWQ: Activation-aware Weight Quantization."
  Better quality than GPTQ via importance-weighted quantization.
- Paper: Dettmers et al. (2023), "QLoRA: Efficient Finetuning of
  Quantized Language Models." INT4 base + FP16 LoRA adapters.

**How hard to add here:** Hard (3-4 weeks for GPTQ-level). Simple
min-max PTQ is easier (~1 week) but lower quality.

---

## 10. Tokenization

### 10.1 BPE (Byte Pair Encoding)

**Status:** PLANNED — Session 8

**What it is.**
An algorithm that builds a vocabulary by iteratively merging the most
frequent adjacent token pairs. Starts with individual bytes (or
characters) and grows the vocabulary to a target size (e.g., 32K-128K
tokens).

**Why it matters.**
- Word-level tokenization (what this codebase has) can't handle unseen
  words, requires huge vocabularies, and wastes tokens on common words.
- BPE handles any text (including code, non-English, special characters)
  with a fixed vocabulary size.
- GPT-2, GPT-3, GPT-4, Llama, and most LLMs use BPE variants.

**Rough idea.**
```
# Training BPE:
vocab = list of all byte values (0-255)
merges = []
for i in range(num_merges):
    pair = most_frequent_adjacent_pair(corpus)
    merges.append(pair)
    vocab.append(pair[0] + pair[1])
    corpus = corpus.replace(pair, merged_token)

# Encoding text:
tokens = list(text.encode('utf-8'))  # start with bytes
for merge in merges:
    tokens = apply_merge(tokens, merge)
return tokens
```

**Where to read more.**
- Sennrich et al. (2016), "Neural Machine Translation of Rare Words
  with Subword Units." Original BPE-for-NLP paper.
- Karpathy, "minbpe" repository. Minimal BPE implementation in Python.
  Extremely clear. Read this first.
- HuggingFace tokenizers library documentation. Production implementation.

**How hard to add here:** Medium (1 week). See Session 8 plan.

---

### 10.2 SentencePiece / Unigram

**Status:** DOC-ONLY

**What it is.**
An alternative to BPE that starts with a large vocabulary and prunes
it down based on a unigram language model. Often produces slightly
better compression than BPE.

**Where to read more.**
- Paper: Kudo (2018), "Subword Regularization: Improving Neural
  Network Translation Models with Multiple Subword Candidates."
- Kudo & Richardson (2018), "SentencePiece: A simple and language
  independent subword tokenizer and detokenizer for Neural Text
  Processing."

**How hard to add here:** Hard (2-3 weeks). More complex algorithm
than BPE. Usually you'd use the SentencePiece C++ library.

---

## 11. Fine-tuning Techniques

### 11.1 LoRA (Low-Rank Adaptation)

**Status:** DOC-ONLY (described in `docs/extensions.md` M4)

**What it is.**
Instead of updating all parameters during fine-tuning, freeze the base
model and add small trainable low-rank matrices to each Linear layer:

```
# Original: y = x @ W
# LoRA:     y = x @ W + x @ A @ B
# Where A is (d_in, r) and B is (r, d_out), r << d_in, d_out
```

Only A and B are trained. The base model stays frozen.

**Why it matters.**
- A 7B model has 7 billion parameters. Fine-tuning all of them requires
  full optimizer state (~56 GB).
- LoRA with rank 16 adds ~0.1% trainable parameters. Optimizer state
  fits easily in GPU memory.
- Quality is within 1-2% of full fine-tuning for most tasks.

**Where to read more.**
- Paper: Hu et al. (2021), "LoRA: Low-Rank Adaptation of Large
  Language Models." The original paper.
- HuggingFace PEFT library documentation. Production LoRA implementation.

**How hard to add here:** Medium (1-2 weeks). Add a `LoraLinear` wrapper
that holds frozen base + trainable A/B. See `docs/extensions.md` M4.

---

### 11.2 QLoRA

**Status:** DEFERRED (requires D7 revisit for INT4 base weights)

**What it is.**
LoRA applied to a 4-bit quantized base model. Combines quantization
(section 9) with LoRA (section 11.1) to fine-tune 65B models on a
single 48 GB GPU.

**Where to read more.**
- Paper: Dettmers et al. (2023), "QLoRA: Efficient Finetuning of
  Quantized Language Models."

---

### 11.3 Prompt / Prefix Tuning

**Status:** DOC-ONLY

**What it is.**
Add learnable "virtual tokens" to the input (prompt tuning) or to
every layer's hidden states (prefix tuning). Train only these tokens;
freeze everything else.

**Where to read more.**
- Paper: Lester et al. (2021), "The Power of Scale for Parameter-
  Efficient Prompt Tuning." Prompt tuning.
- Paper: Li & Liang (2021), "Prefix-Tuning: Optimizing Continuous
  Prompts for Generation." Prefix tuning.

**How hard to add here:** Easy-Medium (3 days - 1 week).

---

## 12. Alignment

### 12.1 Supervised Fine-Tuning (SFT)

**Status:** DOC-ONLY

**What it is.**
Train the model on (prompt, response) pairs with standard cross-entropy
loss. The simplest alignment step — teaches the model the format.

**Why it matters.**
A base model generates text continuations. SFT teaches it to generate
*responses to instructions*. This is the foundation of ChatGPT-style
behavior.

**Rough idea.**
```
# Training data format:
{"prompt": "Explain gravity", "response": "Gravity is..."}

# Training: standard LM loss on the response tokens only
# (mask out the prompt tokens from the loss)
```

**Where to read more.**
- Ouyang et al. (2022), "Training language models to follow
  instructions with human feedback." Section 3.1 (SFT).
- HuggingFace TRL library, `SFTTrainer` class.

**How hard to add here:** Easy (2 days). It's just our existing Trainer
with a loss mask that ignores prompt tokens.

---

### 12.2 DPO (Direct Preference Optimization)

**Status:** DOC-ONLY (described in `docs/extensions.md`)

**What it is.**
Given pairs of (chosen, rejected) responses, train the model to prefer
the chosen response without needing a separate reward model. The loss
is derived from the RLHF objective but computed in closed form.

```
# DPO loss (simplified):
loss = -log(sigmoid(
    beta * (log_prob(chosen) - log_prob_ref(chosen))
    - beta * (log_prob(rejected) - log_prob_ref(rejected))
))
```

**Why it matters.**
- Simpler than full RLHF (no reward model, no PPO)
- Similar quality to RLHF in most benchmarks
- Used by Llama 3, Mistral, and many recent models

**Where to read more.**
- Paper: Rafailov et al. (2023), "Direct Preference Optimization:
  Your Language Model is Secretly a Reward Model."
- HuggingFace TRL library, `DPOTrainer` class.

**How hard to add here:** Medium (1-2 weeks). Needs a preference dataset
loader and the DPO loss function.

---

### 12.3 RLHF (PPO)

**Status:** DOC-ONLY

**What it is.**
Full reinforcement learning from human feedback:
1. Train a reward model on preference data
2. Use PPO to optimize the policy (LLM) against the reward model
3. Add a KL penalty to prevent reward hacking

**Why it matters.**
The technique that made ChatGPT possible. More flexible than DPO
(can optimize arbitrary reward functions) but harder to implement
and tune.

**Where to read more.**
- Paper: Ouyang et al. (2022), "InstructGPT." The RLHF paper.
- Paper: Schulman et al. (2017), "Proximal Policy Optimization."
  The PPO algorithm itself.
- HuggingFace TRL library, `PPOTrainer` class.

**How hard to add here:** Hard (3-5 weeks). Full PPO implementation
with reward model, value head, advantage estimation, and policy update.

---

## 13. Evaluation

**Status:** DOC-ONLY

**What it is.**
Measuring how good a language model actually is. Standard metrics:

- **Perplexity:** exp(average cross-entropy loss). Lower = better.
  The most basic measure.
- **Task-specific benchmarks:** MMLU (knowledge), HumanEval (code),
  GSM8K (math), HellaSwag (commonsense), etc.
- **LM Evaluation Harness:** a standardized framework that runs models
  against 100+ benchmarks in a reproducible way.

**Why it matters.**
Without evaluation, you don't know if your model is learning
anything useful. "Loss went down" is necessary but not sufficient.

**Where to read more.**
- EleutherAI, "lm-evaluation-harness" repository. The standard
  evaluation framework. Run your model against MMLU, HellaSwag, etc.
- Hendrycks et al. (2021), "Measuring Massive Multitask Language
  Understanding." The MMLU benchmark paper.

**How hard to add here:** basic perplexity is trivial (2 days); full
harness integration is months. We already have all pieces for
perplexity — just evaluate cross-entropy on a held-out split.

---

## Summary table

| # | Gap | Status | Session |
|---|---|---|---|
| 1.1 | Flash attention | DOC-ONLY | — |
| 1.2 | KV cache | DOC-ONLY | — |
| 1.3 | RoPE | PLANNED | 7 |
| 1.4 | GQA | DOC-ONLY | — |
| 1.5 | ALiBi | DOC-ONLY | — |
| 2.1 | RMSNorm | PLANNED | 6 |
| 2.2 | SwiGLU | PLANNED | 6 |
| 3 | Mixed precision | DEFERRED (D7) | — |
| 4 | Tensor cores | DOC-ONLY | — |
| 5.1 | Gradient accumulation | PLANNED | 9 |
| 5.2 | Activation checkpointing | DOC-ONLY | — |
| 5.3 | CPU offloading | DOC-ONLY | — |
| 6 | Distributed training | DEFERRED (D14) | — |
| 7 | Advanced CUDA | DOC-ONLY | `cuda_depth.md` |
| 8.1 | KV cache (inference) | DOC-ONLY | — |
| 8.2 | Continuous batching | DOC-ONLY | — |
| 8.3 | Speculative decoding | DOC-ONLY | — |
| 9 | Quantization | DOC-ONLY | — |
| 10.1 | BPE tokenizer | PLANNED | 8 |
| 10.2 | SentencePiece | DOC-ONLY | — |
| 11.1 | LoRA | DOC-ONLY | — |
| 11.2 | QLoRA | DEFERRED (D7) | — |
| 11.3 | Prompt/prefix tuning | DOC-ONLY | — |
| 12.1 | SFT | DOC-ONLY | — |
| 12.2 | DPO | DOC-ONLY | — |
| 12.3 | RLHF/PPO | DOC-ONLY | — |
| 13 | Evaluation | DOC-ONLY | — |

**5 gaps PLANNED for implementation. 3 DEFERRED by locked decisions.
19 DOC-ONLY (learn from external sources).**
