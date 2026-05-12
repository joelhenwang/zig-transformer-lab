# From This Codebase to Llama — The Scale Reality Check

A technical walkthrough of what changes between this project's 2/2/64
toy transformer and a modern production LLM like Llama 3 8B. Covers
parameter counts, architecture differences, training data, compute
requirements, efficiency techniques, and realistic goals for your
hardware.

**Purpose:** ground your expectations. "I want to train my own LLM"
is achievable — but what that looks like depends entirely on your
hardware budget and what you mean by "LLM."

---

## 1. The Gap in One Picture

| Dimension | This codebase (2/2/64) | Llama 3 8B |
|-----------|----------------------|------------|
| Parameters | ~130K | 8,030,000,000 |
| Layers | 2 | 32 |
| Attention heads | 2 | 32 (+ 8 KV heads via GQA) |
| d_model | 64 | 4096 |
| d_ff (MLP hidden) | 256 | 14336 (SwiGLU) |
| Vocabulary | 2000 (word-level) | 128,256 (BPE) |
| Context length | 16 tokens | 8192 tokens |
| Position encoding | Learned embedding | RoPE |
| Normalization | LayerNorm | RMSNorm |
| MLP activation | GELU | SwiGLU |
| Attention variant | Full MHA | Grouped-Query (GQA) |
| Training data | 1.1 MB (Shakespeare) | 15 trillion tokens (~60 TB) |
| Training time | ~20 seconds (RTX 4060 Ti) | ~1.5M GPU-hours (H100) |
| Training cost | ~$0.001 (electricity) | ~$5-10 million |
| Inference memory | ~1 MB | ~16 GB (FP16) |
| Generation quality | Babbling Shakespeare | Fluent general-purpose |

The parameter ratio is approximately **60,000:1**. This is the gap
between a bicycle and a commercial airplane — both "get you places"
but the engineering is categorically different.

---

## 2. Parameter Count Walkthrough

### Our model (2 layers, 2 heads, d_model=64, V=2000)

```
Token embedding:     V * d_model            = 2000 * 64      = 128,000
Position embedding:  T * d_model            = 16 * 64        = 1,024
Per-layer attention:
  W_q, W_k, W_v, W_o: 4 * d^2              = 4 * 64^2       = 16,384
Per-layer LN (x2):  2 * 2 * d              = 4 * 64         = 256
Per-layer MLP:
  fc1: d * 4d                               = 64 * 256       = 16,384
  fc2: 4d * d                               = 256 * 64       = 16,384
  MLP total per layer:                                       = 32,768
Output projection:   d * V                  = 64 * 2000      = 128,000
(Often tied with embedding; if tied: 0 extra)

Total per layer:     16,384 + 256 + 32,768  =               = 49,408
Total (2 layers):    128,000 + 1,024 + 2*49,408 + 128,000   = ~356K
(With weight tying: ~228K. Actual count: ~130K due to smaller actual V)
```

### Llama 3 8B (32 layers, 32 Q heads / 8 KV heads, d=4096)

```
Token embedding:     128,256 * 4096                          = 525M
Per-layer attention:
  W_q: d * d                                = 4096^2         = 16.8M
  W_k: d * (d/4)  [GQA: 8 KV heads]        = 4096 * 1024    = 4.2M
  W_v: d * (d/4)                            = 4096 * 1024    = 4.2M
  W_o: d * d                                = 4096^2         = 16.8M
  Attention total per layer:                                 = 42M
Per-layer RMSNorm (x2):  2 * d             = 2 * 4096       = 8K (negligible)
Per-layer SwiGLU MLP:
  W_gate: d * d_ff                          = 4096 * 14336   = 58.7M
  W_up:   d * d_ff                          = 4096 * 14336   = 58.7M
  W_down: d_ff * d                          = 14336 * 4096   = 58.7M
  MLP total per layer:                                       = 176M
Output projection:   d * V                  = 4096 * 128,256 = 525M
(Not tied in Llama 3)

Total per layer:     42M + 176M + 8K        ≈                = 218M
Total (32 layers):   525M + 32*218M + 525M  ≈                = 8.03B
```

### The ratio

8,030,000,000 / 130,000 ≈ **61,769x** more parameters.

To get from our model to Llama-scale, you need to multiply:
- d_model by 64 (64 → 4096)
- n_layers by 16 (2 → 32)
- vocabulary by 64 (2000 → 128K)
- context length by 512 (16 → 8192)

Each multiplication is quadratic or worse in terms of memory and compute.

---

## 3. Architecture Differences (Side-by-Side)

### 3.1 Normalization: LayerNorm vs RMSNorm

**Ours (LayerNorm):**
```
mean = mean(x, dim=-1)
var = var(x, dim=-1)
x_norm = (x - mean) / sqrt(var + eps)
output = x_norm * gamma + beta
```
Learnable: gamma (scale) + beta (shift). Two parameters per dimension.

**Llama (RMSNorm):**
```
rms = sqrt(mean(x^2, dim=-1) + eps)
output = (x / rms) * gamma
```
Learnable: gamma only. One parameter per dimension. No mean subtraction.

**Why the change:**
- ~30% fewer FLOPs (no mean computation, no subtraction)
- Empirically equivalent quality at LLM scale
- Simpler gradient (fewer ops in the backward graph)

### 3.2 MLP: GELU + 2 Linears vs SwiGLU + 3 Linears

**Ours:**
```
hidden = gelu(x @ fc1.weight)     # (B,T,d) -> (B,T,4d)
output = hidden @ fc2.weight      # (B,T,4d) -> (B,T,d)
```

**Llama (SwiGLU):**
```
gate = silu(x @ W_gate)           # (B,T,d) -> (B,T,d_ff)
up = x @ W_up                     # (B,T,d) -> (B,T,d_ff)
hidden = gate * up                 # element-wise gating
output = hidden @ W_down           # (B,T,d_ff) -> (B,T,d)
```

**Why the change:**
- The gate mechanism lets the network learn to suppress dimensions
- ~1% perplexity improvement at same compute budget
- Llama uses d_ff = (8/3)*d instead of 4*d to compensate for the
  third matrix (same total parameter count)

### 3.3 Position encoding: Learned vs RoPE

**Ours:**
```
pos_embed = learned_table[0:T]    # (T, d) lookup table
x = tok_embed(ids) + pos_embed    # added to input
```

**Llama (RoPE):**
```
# No addition to input. Instead, rotate Q and K:
for each dimension pair (2i, 2i+1):
    theta = 10000^(-2i/d_head)
    q[2i], q[2i+1] = rotate(q[2i], q[2i+1], pos * theta)
    k[2i], k[2i+1] = rotate(k[2i], k[2i+1], pos * theta)
# Then: attention scores = q @ k^T (encodes relative position)
```

**Why the change:**
- RoPE encodes *relative* position (Q@K^T depends on pos_q - pos_k)
- No learnable parameters (pure math)
- Extrapolates to longer sequences than seen during training
- Works with KV caching (each token's rotation is independent)

### 3.4 Attention: MHA vs GQA

**Ours (Multi-Head Attention):**
```
Q = x @ W_q  -> split into n_heads heads
K = x @ W_k  -> split into n_heads heads  (same number as Q)
V = x @ W_v  -> split into n_heads heads  (same number as Q)
```

**Llama (Grouped-Query Attention):**
```
Q = x @ W_q  -> split into 32 heads
K = x @ W_k  -> split into 8 heads   (fewer!)
V = x @ W_v  -> split into 8 heads   (fewer!)
# Each group of 4 Q heads shares 1 K/V head
```

**Why the change:**
- KV cache at inference: 32 KV heads → 8 KV heads = 4x less memory
- For T=8192: KV cache goes from 2 GB to 512 MB (per batch element)
- Quality loss is negligible (< 0.5% perplexity increase)

### 3.5 Bias terms: present vs removed

**Ours:** all Linear layers have bias terms (W @ x + b).

**Llama:** no bias anywhere (W @ x only).

**Why:** at scale, bias terms add negligible capacity but complicate
the optimizer state (extra momentum/variance vectors). Removing them
is free quality (marginally) and simplifies the model.

---

## 4. Training Data Scale

### Our dataset

- Source: Shakespeare's complete works
- Size: 1.1 MB of text
- Tokens (word-level, V=2000): ~200K tokens
- Unique words: ~2000 (our vocabulary cap)
- Genre: one author, one style, one era

### Llama 3's dataset

- Sources: Common Crawl, Wikipedia, books, code, scientific papers,
  social media, multilingual text
- Size: ~15 trillion tokens (estimated from training FLOP reports)
- Raw text: approximately 60 TB uncompressed
- Languages: primarily English + multilingual
- Data engineering: extensive deduplication, quality filtering, domain
  balancing, PII removal

### The ratio

15,000,000,000,000 tokens / 200,000 tokens = **75,000,000x** more data.

### What this means

A language model needs to see enough data to learn the statistical
patterns of language. The Chinchilla scaling law suggests:

```
optimal_tokens ≈ 20 * parameters
```

For our 130K model: optimal training = ~2.6M tokens. We train on 200K —
under-trained by ~13x, which is why our model produces plausible but
not coherent Shakespeare.

For Llama 3 8B: optimal = ~160B tokens. They trained on 15T — 
*over-trained* by ~100x. This is intentional: over-training makes the
model better at inference without increasing inference cost.

---

## 5. Compute Scale

### Estimating training FLOPs

The standard approximation for transformer training:

```
FLOPs ≈ 6 * N * D
```

Where N = parameter count, D = training tokens.

**Our model:**
```
6 * 130,000 * 200,000 = 156 billion FLOPs = 156 GFLOP
Time at 22 TFLOPS FP32: 156e9 / 22e12 = 0.007 seconds (theoretical)
Actual time: ~20 seconds (efficiency ~0.04% — launch-overhead dominated)
```

**Llama 3 8B:**
```
6 * 8e9 * 15e12 = 720,000,000 billion FLOPs = 7.2e23 FLOP
Time at 989 TFLOPS FP16 per H100: 7.2e23 / 989e12 = 7.3e8 seconds per GPU
With 2048 GPUs at 40% MFU: 7.3e8 / (2048 * 0.4) = ~890,000 seconds ≈ 10 days
(Actual: reportedly ~24 days on 2048 H100s)
```

### Cost comparison

| | Our training | Llama 3 8B |
|---|---|---|
| Hardware | 1x RTX 4060 Ti | ~2048x H100 (80GB) |
| Time | 20 seconds | ~24 days |
| GPU-hours | 0.006 | ~1,200,000 |
| Cost (at $2/H100-hr) | $0.00 | ~$2.4 million |
| Electricity | ~0.003 kWh | ~700,000 kWh |

---

## 6. Efficiency Techniques Required at Scale

At our scale (130K params, 20s training), no optimization is needed —
everything fits in memory, every approach works.

At Llama scale, you MUST use these techniques or training is impossible:

### 6.1 Mixed precision (Gap 3)

Without: 8B params * 4 bytes = 32 GB just for parameters in FP32.
With FP16: 16 GB for params. Tensor cores give 2x throughput.
AdamW state (m + v): 2 * 8B * 4 = 64 GB even with FP16 params.
**Total without mixed precision: >100 GB per GPU.**

### 6.2 Flash attention (Gap 1.1)

Without: attention scores matrix for T=8192, H=32 = 
32 * 8192^2 * 2 bytes = 4 GB per batch element.
With flash attention: O(T) memory ≈ negligible.
**Without flash attention, batch size 1 at T=8192 is already 4 GB just for attention.**

### 6.3 Gradient checkpointing (Gap 5.2)

Without: store activations for all 32 layers.
Per layer at B=1, T=8192, d=4096: ~256 MB (rough estimate).
32 layers: ~8 GB of activations.
With checkpointing: store only layer boundaries: ~256 MB total.
**Saves ~7.7 GB per batch element.**

### 6.4 Distributed training (Gap 6)

8B * 4 bytes (params) + 8B * 8 bytes (optimizer) = 96 GB.
No single GPU has 96 GB. Even H100 (80 GB) can't fit it alone.
**You MUST shard across GPUs (FSDP/ZeRO).**

### 6.5 Fused kernels (Section 6 of cuda_depth.md)

Without fusion: ~500 kernel launches per layer per step.
32 layers: ~16,000 launches per step. At 10 us each = 160 ms overhead.
With fusion: ~50 launches per layer = ~1,600 per step = 16 ms overhead.
**10x reduction in launch overhead.**

### Summary: what happens without each

| Technique | Without it at 8B scale |
|---|---|
| Mixed precision | OOM (params + optimizer > 100 GB) |
| Flash attention | OOM at T > 2048 (attention matrix) |
| Grad checkpointing | OOM at B > 1 (activation memory) |
| Distributed (FSDP) | OOM (no single GPU fits full state) |
| Kernel fusion | ~10% wall-clock wasted on launches |

At our 130K scale: none of these matter. At 8B: all are mandatory.

---

## 7. Realistic Hardware Tiers

What you can actually train from scratch at each hardware level:

### Tier 1: Single RTX 4060 Ti (16 GB) — your hardware

- **Max model size:** ~50-100M parameters (FP32)
- **With gradient accumulation:** effective batch size up to ~32
- **Practical training:** domain-specific models on < 10M tokens
- **Training time:** hours to days for 50M params
- **Quality ceiling:** good enough for narrow tasks (style transfer,
  small-domain Q&A, code completion for one library)
- **Examples:**
  - 50M param Shakespeare model (genuinely good text, ~15 min)
  - 20M param code model trained on one repo (~1 hour)
  - Fine-tune a quantized 7B model with LoRA (INT4 base fits in 4 GB)

### Tier 2: Single A100/H100 80 GB (cloud rental, $1-5/hr)

- **Max model size:** ~1-3B parameters (mixed precision)
- **Practical training:** medium-domain models on 10-100B tokens
- **Training time:** days to weeks
- **Quality ceiling:** competitive on specific tasks, not general-purpose
- **Examples:**
  - 1B param model on a large code corpus (competitive with small Codex)
  - 400M param multilingual model on specific language pairs

### Tier 3: 8x H100 node ($20-40/hr)

- **Max model size:** ~7-13B parameters (FSDP + mixed precision)
- **Practical training:** full pretraining on 100B-1T tokens
- **Training time:** weeks to months
- **Quality ceiling:** GPT-3.5 level for specific domains
- **Examples:**
  - 7B param general-purpose model (competitive with early Llama)
  - 13B param domain-specific model (medical, legal, scientific)

### Tier 4: Research cluster (64-512 GPUs, $100K+)

- **Max model size:** 30-70B parameters
- **Practical training:** frontier-quality pretraining
- **Training time:** weeks (with experienced engineering team)
- **This is where Llama, Mistral, and Gemma are trained.**

---

## 8. Training Cost Estimation

### The Chinchilla rule of thumb

```
optimal_tokens = 20 * parameters
FLOPs = 6 * parameters * tokens
cost = FLOPs / (GPUs * TFLOPS_per_GPU * MFU * 3600) * cost_per_GPU_hour
```

### Worked examples

**50M model on RTX 4060 Ti (your realistic target):**
```
Tokens needed: 20 * 50M = 1B (but Shakespeare is only 200K tokens;
  you'd need ~5000 epochs or a larger corpus)
FLOPs: 6 * 50e6 * 1e9 = 3e17
Time: 3e17 / (22e12 * 0.3) = 45,000 seconds ≈ 12.5 hours
Cost: ~$0.50 (electricity)
```

**1B model on A100 (cloud rental):**
```
Tokens: 20B
FLOPs: 6 * 1e9 * 20e9 = 1.2e20
Time: 1.2e20 / (312e12 * 0.4) = 960,000 seconds ≈ 11 days
Cost: 11 days * 24 hrs * $2/hr = ~$530
```

**7B model on 8x H100:**
```
Tokens: 140B
FLOPs: 6 * 7e9 * 140e9 = 5.9e21
Time per GPU: 5.9e21 / (989e12 * 0.4 * 8) = 1.87e6 seconds ≈ 21 days
Cost: 21 days * 24 hrs * $30/hr (8-GPU node) = ~$15,000
```

### The honest math for "training my own LLM"

If "my own LLM" means:
- A 50M model on a specific domain → **$0.50 and 12 hours.** Very doable.
- A 1B general model → **$530 and 11 days.** Affordable cloud experiment.
- A 7B competitive model → **$15,000 and 3 weeks.** Serious project budget.
- A 70B frontier model → **$500K+ and months.** Research lab territory.

---

## 9. The Honest Ceiling of "Train from Scratch"

### What's genuinely achievable on one RTX 4060 Ti

1. **Educational replications** — train a 50M transformer and actually
   understand every line of code, every gradient, every kernel launch.
   This is what this codebase is designed for.

2. **Domain-specific small models** — train a 20-100M model on a
   specific corpus (code for one library, medical abstracts, legal
   text). These can be genuinely useful for narrow applications.

3. **Architecture experiments** — test whether RMSNorm beats LayerNorm,
   whether SwiGLU helps at small scale, whether RoPE extrapolates.
   Run ablations. Write papers.

4. **Fine-tuning large models** — use LoRA/QLoRA to fine-tune a
   quantized 7B model (INT4 = ~4 GB, fits easily). This is often more
   useful than training from scratch.

### What's NOT achievable without significant resources

1. **General-purpose chatbot quality** — requires 7B+ params and
   trillions of tokens. Not a single-GPU project.

2. **Competitive benchmarks (MMLU, HumanEval)** — requires both scale
   and diverse high-quality data. Even a perfect small model on
   Shakespeare won't score on general knowledge.

3. **Multilingual capability** — requires massive multilingual corpus
   + large model capacity.

### The right framing

This codebase teaches you to BUILD the tool. Actually training a
frontier LLM is a different problem (data engineering + compute budget
+ team coordination). Both are valuable skills; this repo gives you
the first one completely and points toward the second.

---

## 10. Realistic Learning Goals (Given One RTX 4060 Ti)

### Goal 1: Complete the curriculum (Phases 1-4 of learning_roadmap.md)

**Time:** 4-6 weeks.
**Outcome:** you understand tensors, autograd, optimizers, transformers,
and basic CUDA at the implementation level.

### Goal 2: Implement 5 extensions from extensions.md

**Time:** 3-4 weeks.
**Outcome:** you've built RMSNorm, SwiGLU, RoPE, BPE, and gradient
accumulation. You now have a mini-Llama architecture.

### Goal 3: Train a 50M model on a real corpus

**Time:** 1 week (setup) + 12 hours (training).
**Outcome:** a model that generates genuinely coherent text in a
specific domain. This is a portfolio piece.

### Goal 4: Read and understand Llama source code

**Time:** 1-2 weeks.
**Outcome:** you can trace a forward pass through the full Llama 3
implementation, identify every component, and explain why each
architectural choice was made.

### Goal 5: Fine-tune a real LLM with LoRA

**Time:** 1-2 weeks (after completing M4 in extensions.md or using
HuggingFace PEFT).
**Outcome:** you've adapted a 7B model to your domain. This is the
most practically valuable single skill for applied ML work.

**At this point you are employable in the ML/LLM field.** You
understand both theory and implementation at a level most practitioners
don't reach.

---

## 11. What's Next

After completing Goals 1-5, you're ready to engage with the research
frontier:

- **Read papers** — you have the vocabulary and implementation intuition
  to follow new architecture papers, training papers, and kernel papers.
- **Contribute to open-source** — projects like vLLM, llama.cpp, TRL,
  and Triton all need contributors who understand the full stack.
- **Run experiments** — test ideas at small scale (50M on your 4060 Ti)
  before scaling up. Most ideas can be validated or invalidated at 50M.
- **Specialize** — pick one area (kernels? alignment? data engineering?
  architecture?) and go deep. The breadth from this curriculum gives
  you enough context to choose wisely.

The journey from "I know Python" to "I can train my own LLM" is real
and achievable. This codebase is the hardest part — understanding the
internals. Everything after is scale and engineering.
