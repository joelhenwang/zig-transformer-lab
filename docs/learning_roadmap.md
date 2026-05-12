# The Learning Roadmap

A structured curriculum for learning GPU programming, kernel development,
machine learning, and deep learning fundamentals — ending with the skills
to train your own LLM from scratch.

**Who this is for:** a Python programmer who wants to deeply understand
how tensor libraries, autograd engines, GPU kernels, and transformer
training actually work at the implementation level — not just call
`model.fit()` in a framework.

**What you'll end up with:** after completing all 8 phases, you'll be
able to read and understand modern LLM source code (Llama, Mistral),
design training runs, write custom CUDA kernels, and make informed
architecture decisions for models up to your hardware budget.

**How long:** 12-16 weeks of focused study (2-3 hours/day). Some phases
can be compressed if you already have background in the area.

---

## How to use this document

Each phase has:
- **What to study** — ordered reading list (this repo's docs + external)
- **Hands-on work** — exercises, projects, or code to read
- **Self-check** — questions you should be able to answer before moving on
- **External pairings** — books, papers, or codebases that complement
- **You're ready for the next phase when...** — concrete milestone

Do the phases in order. Each builds on the previous. Don't skip Phase 0
even if you think you know the math — the Zig primer section is essential.

---

## Phase 0: Prerequisites (the part outside this repo)

**Time estimate:** 1-2 weeks, depending on how rusty your math is.

This phase gets your mental toolkit ready. Everything in later phases
assumes you can do basic matrix operations by hand and read Zig code
without panicking.

### 0.1 Linear algebra for ML

You need to be fluent in:

- **Vectors** — addition, scaling, dot product, norm (L2)
- **Matrices** — multiplication (row-by-column), transpose, shape rules
- **Broadcasting** — how NumPy/PyTorch stretch dimensions to make shapes compatible
- **Outer product vs inner product** — what each produces
- **Element-wise operations** — same shape, apply operation per-element

You do NOT need eigenvalues, SVD, or determinants for this curriculum.
Those matter later (PCA, LoRA theory) but not for implementation work.

**Self-check:**
- What shape is the result of multiplying a (3, 4) matrix by a (4, 2) matrix?
- If A has shape (2, 3) and B has shape (1, 3), what does `A + B` produce?
- What is the L2 norm of the vector [3, 4]?
- What does "contiguous in memory" mean for a row-major matrix?

**Reading:**
- 3Blue1Brown, "Essence of Linear Algebra" video series (10 episodes,
  ~3 hours total). Why: builds geometric intuition you'll use daily.
- Deep Learning (Goodfellow, Bengio, Courville), Chapter 2 "Linear
  Algebra." Why: the standard ML-focused linear algebra reference.

### 0.2 Calculus for ML

You need to be fluent in:

- **Derivatives** — the rate-of-change intuition, basic rules (power, product, chain)
- **Chain rule** — THE central rule in backpropagation: d/dx f(g(x)) = f'(g(x)) * g'(x)
- **Partial derivatives** — derivative with respect to one variable, holding others fixed
- **Gradient** — the vector of all partial derivatives of a function
- **Gradient descent** — move parameters in the direction that reduces loss

You do NOT need integration, differential equations, or measure theory.

**Self-check:**
- What is d/dx of x^2 + 3x? What about d/dx of e^(2x)?
- If loss = (prediction - target)^2, what is d(loss)/d(prediction)?
- In the chain rule, if y = f(g(x)), what are the two pieces you multiply?
- Why does gradient descent move in the negative gradient direction?

**Reading:**
- 3Blue1Brown, "Essence of Calculus" video series. Why: visual intuition
  for derivatives that makes backprop feel natural.
- Deep Learning (Goodfellow, Bengio, Courville), Chapter 4.3 "Gradient-
  Based Optimization." Why: connects calculus to the ML training loop.

### 0.3 Probability for ML

You need to be fluent in:

- **Probability distribution** — assigns probabilities to outcomes (sum to 1)
- **Softmax** — turns arbitrary numbers into a probability distribution:
  softmax(x_i) = exp(x_i) / sum(exp(x_j))
- **Cross-entropy loss** — measures how "wrong" a predicted distribution is:
  CE(target, pred) = -sum(target_i * log(pred_i))
- **Expectation** — the weighted average of outcomes

You do NOT need Bayesian inference, Markov chains, or information theory
beyond the definition of cross-entropy.

**Self-check:**
- If logits = [2.0, 1.0, 0.1], what does softmax produce (approximately)?
- Why do we subtract the max before computing exp in softmax?
- If the true class is index 2 and softmax gives [0.7, 0.2, 0.1],
  what is the cross-entropy loss?
- Why is cross-entropy the standard loss for classification?

**Reading:**
- Deep Learning (Goodfellow, Bengio, Courville), Chapter 3 "Probability
  and Information Theory." Why: the mathematical foundation for every
  loss function in the transformer.

### 0.4 Zig primer for Python programmers (thorough)

This codebase is written in Zig 0.16.0. You need to read Zig fluently.
Here's how Zig maps to Python concepts you already know:

#### Types and variables

```
Python:                          Zig:
x = 42                           const x: i32 = 42;
x = 42  (later: x = 99)         var x: i32 = 42; x = 99;
name = "hello"                   const name = "hello";  // []const u8
numbers = [1, 2, 3]             const numbers = [_]i32{ 1, 2, 3 };
```

Key difference: Zig has `const` (immutable, default) and `var` (mutable,
must be explicitly requested). The compiler errors if you use `var` but
never mutate — this is NOT a warning, it's a hard error.

#### Functions

```
Python:                          Zig:
def add(a, b):                   fn add(a: i32, b: i32) i32 {
    return a + b                     return a + b;
                                 }

def risky():                     fn risky() !i32 {
    if bad:                          if (bad) return error.BadThing;
        raise ValueError             return 42;
    return 42                    }
```

Key difference: Zig errors are in the return type (`!i32` means "returns
i32 or an error"). `try` in Zig propagates errors (like Python's bare
`raise` inside an except). `catch` in Zig handles errors (like except).

#### Memory (the big conceptual gap)

Python: garbage-collected. You never think about memory.
Zig: manual. You MUST free what you allocate.

```
Python:                          Zig:
data = [0.0] * 1000             const data = try alloc.alloc(f32, 1000);
# GC handles cleanup             defer alloc.free(data);  // freed at scope exit
```

`defer` is Zig's RAII pattern — it schedules cleanup to run when the
current scope exits. This is how every tensor, buffer, and context in
this codebase manages memory.

The `Allocator` is passed explicitly to every function that allocates.
No hidden global heap. This makes memory usage transparent and testable
(the test allocator detects leaks).

#### Structs (closest to Python classes)

```
Python:                          Zig:
class Tensor:                    pub const Tensor = struct {
    def __init__(self, data):        shape: Shape,
        self.data = data             storage: Storage,

    def forward(self, x):            pub fn forward(self: Tensor, x: Tensor) !Tensor {
        ...                              ...
                                     }
                                 };
```

Key differences:
- No inheritance. Composition only.
- Methods take `self` as an explicit typed parameter.
- Fields have explicit types. No `None` — optionals are `?T`.
- No constructors — just `init()` functions that return the struct.

#### Error handling

```
Python:                          Zig:
try:                             const result = doThing() catch |err| {
    result = do_thing()              // handle error
except ValueError as e:              return err;
    ...                          };
```

`try` in Zig is shorthand for `catch |e| return e` — it propagates the
error to the caller. Nearly every function in this codebase returns `!T`
(can-fail return type) and uses `try` to propagate.

#### Slices and arrays

```
Python:                          Zig:
data = [1.0, 2.0, 3.0]         var data = [_]f32{ 1.0, 2.0, 3.0 };
x = data[1]                     const x = data[1];
chunk = data[1:3]               const chunk = data[1..3];
for item in data:               for (data) |item| {
    print(item)                     std.debug.print("{}\n", .{item});
                                }
```

Key difference: `[]f32` in Zig is a slice (pointer + length). `[3]f32`
is a fixed-size array. Slices do bounds-checking in debug builds.

#### Imports

```
Python:                          Zig:
from tensor import Tensor        const Tensor = @import("tensor.zig").Tensor;
import numpy as np               const np = @import("numpy.zig");
```

Every file is a struct in Zig. `@import` returns that struct. You access
its public declarations with dot syntax.

#### Optionals and null

```
Python:                          Zig:
x = None                         var x: ?i32 = null;
if x is not None:                if (x) |val| {
    use(x)                           use(val);
                                 }
```

`?T` means "T or null." You must unwrap optionals explicitly — no
NullPointerException surprises.

#### Key patterns you'll see in this codebase

```zig
// Allocate a tensor, defer cleanup
var t = try Tensor.init(allocator, shape);
defer t.deinit(allocator);

// Create a tape for autograd
var tape = Tape.init(allocator);
defer tape.deinit();

// Forward pass with tape recording
var out = try ops.add(allocator, a, b, &tape);

// Backward pass
try tape.backward(allocator, &loss);
```

This pattern repeats hundreds of times. Once you recognize it,
you can read the entire codebase.

**Reading:**
- The Zig Language Reference (ziglearn.org or ziglang.org/documentation).
  Why: authoritative reference when you hit an unfamiliar construct.
- `docs/01_zig_primer.md` in this repo. Why: covers 0.16.0-specific
  patterns and gotchas we've hit during development.

### 0.5 You're ready for Phase 1 when...

- You can multiply two matrices by hand and state the output shape.
- You can apply the chain rule to d/dx sin(x^2).
- You can explain why softmax(x_i - max(x)) is numerically safer than
  softmax(x_i) directly.
- You can read a 20-line Zig function without looking up every keyword.
- You know what `defer`, `try`, `const`, `var`, `?T`, and `!T` mean.

---

## Phase 1: Tensor foundations (this repo, ~2 weeks)

**Goal:** understand how multi-dimensional arrays are stored in memory,
how views/transposes work without copying data, and how the Storage
abstraction enables multi-device support.

### What to read (in order)

1. `docs/00_overview.md` — high-level architecture map
2. `docs/01_zig_primer.md` — Zig patterns specific to this project
3. `docs/02_tensors.md` — the Tensor struct, shapes, strides
4. `docs/02b_from_tensors_to_training.md` — bridge chapter
5. `docs/02c_tensor_invariants.md` — what makes a tensor "valid"
6. `docs/02d_storage_and_views.md` — Storage union, cpuData(), ownership

### Source files to trace through

- `src/tensor/tensor.zig` — the struct definition, init, deinit, view,
  reshape, transpose2d, cpuData
- `src/tensor/shape.zig` — Shape, Strides, computeStrides, totalElements,
  broadcastShapes, isContiguous

### Exercises

1. Open `src/tensor/tensor.zig` and trace what happens when you call
   `Tensor.init(alloc, Shape.init2D(3, 4))`. Write down every field
   value of the returned struct.

2. After calling `t.transpose2d()`, what are the new shape and strides?
   Why doesn't this copy any data?

3. Why does `cpuData()` assert `device == .cpu`? What would happen if
   a CUDA tensor didn't have this check?

### Self-check

- What is the flat index of element [1, 2] in a (3, 4) tensor with
  row-major strides [4, 1]?
- If a tensor has strides [1, 3] instead of [3, 1], what happened?
- What does `storage.cpu.owned = false` mean?
- Why can't you reshape a non-contiguous tensor without copying?

### External pairings

- NumPy documentation on "ndarray internal memory layout." Why: same
  concepts (strides, contiguous, C-order vs F-order), different language.

### You're ready for Phase 2 when...

- You can explain strides, contiguity, and views in your own words.
- You can predict the strides of a 3D tensor after transposing axes 1 and 2.
- You understand why owned vs. view tensors have different deinit behavior.

---

## Phase 2: Autograd and optimizers (this repo, ~2 weeks)

**Goal:** understand tape-based automatic differentiation, how gradients
flow backward through a computation graph, and how optimizers use
gradients to update parameters.

### What to read (in order)

1. `docs/03_autograd.md` — tape, nodes, OpKind, SavedData
2. `docs/03b_from_autograd_to_training.md` — bridge chapter
3. `docs/03c_tape_mechanics.md` — node lifecycle, keepAlive, cloning
4. `docs/04_nn.md` — neural network layers built on autograd
5. `docs/04b_from_nn_to_training.md` — why each layer exists

### Source files to trace through

- `src/autograd/tape.zig` — Tape.record, Tape.backward, keepAlive
- `src/autograd/node.zig` — Node struct, OpKind enum, SavedData union
- `src/autograd/grad_helpers.zig` — heapAlloc, broadcastTo, accumulateGrad
- `src/tensor/ops/elementwise.zig` — how `add()` records on the tape,
  then `backwardAdd()` at the bottom computes the gradient
- `src/optim/adamw.zig` — AdamW with bias correction

### Exercises

1. Trace a complete forward + backward for `c = a + b` where a=(2,3)
   and b=(1,3). What shape is each gradient? Why does `b`'s gradient
   get sumToShape'd?

2. Open `src/autograd/gradcheck.zig` and understand how finite-difference
   gradient checking works. Why is `denom_floor=1e-2` important?

3. In `backwardMul`, why is `dL/da = dL/dc * b`? Derive it from the
   chain rule applied to c = a * b (element-wise).

### Self-check

- What does the tape store for each operation?
- Why does `SavedData.tensor_pair` save both inputs a and b by value?
- What would happen if you ran backward twice without resetting the tape?
- Why does AdamW use bias correction in early steps?
- What is the difference between coupled and decoupled weight decay?

### External pairings

- Karpathy, "micrograd" repository. Why: a 100-line autograd engine in
  Python — read it BEFORE this repo's version to build intuition, then
  see how the same ideas scale to tensors.
- Deep Learning (Goodfellow, Bengio, Courville), Chapter 6.5 "Back-
  Propagation." Why: the theoretical derivation behind everything here.

### You're ready for Phase 3 when...

- You can derive dL/dW for a linear layer y = Wx + b from the chain rule.
- You can explain why `tape.keepAlive(&tensor)` is necessary.
- You know what happens to gradients at a broadcast boundary.
- You can explain AdamW's three hyperparameters (lr, beta1, beta2).

---

## Phase 3: Small transformer end-to-end (this repo, ~2 weeks)

**Goal:** understand the full transformer architecture (embedding,
attention, MLP, LayerNorm, output projection) and the training loop
(data pipeline, loss, backward, optimizer step, gradient clipping).

### What to read (in order)

1. `docs/05_transformer_math.md` — attention mechanism math
2. `docs/05b_from_tokenizer_to_training.md` — bridge chapter
3. `docs/06_tokenizer_data.md` — word-level tokenizer, Dataset, Batcher
4. `docs/07_cpu_training.md` — Trainer, gradient clipping, generation
5. `docs/09_debugging.md` — debugging toolkit for when things go wrong
6. `docs/10_pytorch_parallels.md` — map every concept to PyTorch

### Source files to trace through

- `src/nn/attention.zig` — CausalSelfAttention (multi-head)
- `src/nn/block.zig` — TransformerBlock (pre-norm residual pattern)
- `src/nn/model.zig` — TinyWordTransformer (full model assembly)
- `src/lab/train.zig` — Trainer (training loop orchestration)
- `examples/10_train_deep.zig` — the 2/2/64 CUDA training example

### Exercises

1. Trace the forward pass for a single token through the full model.
   What is the shape at each stage? (Hint: start with ids shape (B, T)
   and follow it through embedding, attention, MLP, output projection.)

2. The causal mask prevents attending to future tokens. Where in
   `attention.zig` is this implemented? What value is used for masking?

3. Why does the Trainer use `sumOfSquaresAll` for gradient clipping
   instead of iterating over individual gradient values?

### Self-check

- What is the shape of the attention scores matrix for B=2, H=2, T=8?
- Why is pre-norm (LN before attention) preferred over post-norm?
- What does "teacher forcing" mean in the context of language model training?
- Why do we need positional encodings? What happens without them?
- What is the vocabulary size's relationship to the output projection shape?

### External pairings

- Vaswani et al. (2017), "Attention Is All You Need." Why: the original
  transformer paper — read after you understand the code, not before.
- Karpathy, "nanoGPT" repository. Why: a clean 300-line GPT-2
  implementation in PyTorch. Compare the architecture decisions to ours.
- Karpathy, "Let's build GPT" video (~2 hours). Why: walks through
  nanoGPT construction live, connecting code to concepts.

### You're ready for Phase 4 when...

- You can describe the full forward pass shape-by-shape.
- You understand why multi-head attention splits d_model into heads.
- You can explain what gradient clipping prevents and how it works.
- You know the difference between training (teacher forcing) and
  inference (autoregressive generation).

---

## Phase 4: CUDA basics (this repo, ~1 week)

**Goal:** understand how the CPU↔GPU boundary works, how kernels are
launched, what device memory management looks like, and how cuBLAS
wraps matrix multiplication.

### What to read (in order)

1. `docs/08_backends_cuda.md` — CUDA backend architecture, dispatch
2. `docs/08b_from_cuda_to_training.md` — device-aware training bridge
3. `src/backend/cuda/` README

### Source files to trace through

- `src/backend/cuda/bindings.zig` — dlopen shim for libcuda + libcublas
- `src/backend/cuda/context.zig` — CudaContext (device init)
- `src/backend/cuda/mem.zig` — DeviceBuffer (RAII for device allocation)
- `src/backend/cuda/module.zig` — PTX loader (cuModuleLoadData)
- `src/backend/cuda/dispatch.zig` — 28 kernel dispatch entry points
- `src/backend/cuda/gemm.zig` — cuBLAS row-major wrapping trick
- `src/tensor/device_dispatch.zig` — the seam (Phase 2 of arch work)

### Exercises

1. Trace what happens when you call `ops.add(allocator, a, b, tape)`
   where `a.device == .cuda`. Follow the call through device_dispatch →
   dispatch.zig → kernel launch.

2. Read `src/backend/cuda/kernels/elementwise.cu`. The `elw_add` kernel
   is ~5 lines. Understand: thread index calculation, grid stride loop,
   bounds check.

3. Why does `gemm.zig` swap A and B when calling cuBLAS? Draw the
   row-major vs. column-major layout and show why the swap is correct.

### Self-check

- What is the difference between a CUDA thread, a block, and a grid?
- Why do we use dlopen instead of linking libcuda at compile time?
- What is PTX and why do we compile .cu to .ptx offline?
- What does `DeviceBuffer.fromHost` do step by step?
- Why is there no `@memcpy` for CUDA tensors — what do we use instead?

### External pairings

- Programming Massively Parallel Processors (Kirk & Hwu), Chapters 1-5.
  Why: the standard CUDA textbook. Chapters 1-5 cover the programming
  model, memory hierarchy, and basic kernel writing.
- CUDA C Programming Guide (Nvidia), Chapter 2 "Programming Model."
  Why: authoritative reference for thread/block/grid concepts.

### You're ready for Phase 5 when...

- You can explain how a CUDA kernel launch maps threads to data elements.
- You understand the HtoD/DtoH transfer cost model.
- You can read a simple .cu kernel and predict what it computes.
- You know why cuBLAS exists (hand-tuned GEMM >> naive kernel).

---

## Phase 5: CUDA depth (external + docs/cuda_depth.md, ~2 weeks)

**Goal:** understand shared memory tiling, warp-level primitives, tensor
cores, kernel fusion, and GPU profiling — the techniques that make real
LLM kernels 10-100x faster than naive launches.

### What to read

1. `docs/cuda_depth.md` (companion doc #3) — all 10 sections
2. Run `examples/11_cuda_tiled_gemm_demo.zig` and observe the speedup
3. Run `examples/12_cuda_wmma_demo.zig` and verify correctness

### External pairings (heavy reading phase)

- Programming Massively Parallel Processors (Kirk & Hwu), Chapters 6-12.
  Why: tiled algorithms (Ch 6), memory coalescing (Ch 7), floating point
  (Ch 8), parallel patterns (Ch 9-12). The core of GPU performance work.
- CUTLASS repository (Nvidia), `examples/` directory. Why: production
  GEMM templates. Read after understanding tiling conceptually.
- Tri Dao et al. (2022), "FlashAttention: Fast and Memory-Efficient
  Exact Attention." Why: the most important kernel optimization in modern
  LLMs. Read the paper after understanding tiling from cuda_depth.md.
- Nvidia Developer Blog posts on WMMA and tensor cores. Why: practical
  examples beyond what the docs cover.

### Self-check

- Why does shared memory tiling reduce global memory accesses?
- What is a bank conflict and how do you avoid it?
- What is occupancy and why isn't 100% always optimal?
- How does `__shfl_down_sync` help implement a fast reduction?
- What does WMMA stand for and what hardware instruction does it use?
- Why does flash attention need the "online softmax" trick?

### You're ready for Phase 6 when...

- You can sketch a tiled GEMM algorithm from memory.
- You understand why flash attention is memory-efficient (O(N) vs O(N^2)).
- You can read an Nsight Compute output and identify the bottleneck.
- You know when to use cuBLAS vs. a custom kernel.

---

## Phase 6: Modern LLM architecture (external, ~2 weeks)

**Goal:** understand the architectural decisions that distinguish modern
LLMs (Llama, Mistral, Gemma) from the basic transformer in this repo.

### What to read

1. `docs/gap_map.md` sections 1, 2 (companion doc #2)
2. `docs/from_this_to_llama.md` section 3 (companion doc #5)
3. `docs/extensions.md` entries M1, M6 (companion doc #4)

### External pairings

- Touvron et al. (2023), "LLaMA: Open and Efficient Foundation Language
  Models." Why: the architecture paper that defined the modern LLM stack.
- Touvron et al. (2023), "Llama 2." Why: adds RLHF, longer context.
- Meta AI (2024), "Llama 3." Why: latest architecture iteration.
- Su et al. (2021), "RoFormer: Enhanced Transformer with Rotary
  Position Embedding." Why: the RoPE paper — position encoding used
  by every modern LLM.
- Karpathy, "llama2.c" repository. Why: minimal C implementation of
  Llama 2 inference. Extremely readable. ~700 lines.
- Shazeer (2020), "GLU Variants Improve Transformer." Why: SwiGLU/GeGLU
  paper explaining why gated activations outperform plain GELU.

### Self-check

- What are the 5 main architecture changes from GPT-2 to Llama 3?
- Why is RMSNorm preferred over LayerNorm? (Hint: computational cost.)
- How does RoPE encode position without an embedding table?
- What is GQA and why does it save memory during inference?
- Why did Llama remove all bias terms?

### You're ready for Phase 7 when...

- You can read Llama source code and identify every layer.
- You understand why each architecture choice was made.
- You could implement a mini-Llama in this codebase (see extensions.md M6).

---

## Phase 7: Training at scale (external, ~2 weeks)

**Goal:** understand how models are trained across multiple GPUs, how
memory is managed at scale, and what infrastructure is needed.

### What to read

1. `docs/gap_map.md` sections 5, 6 (companion doc #2)
2. `docs/from_this_to_llama.md` sections 5-8 (companion doc #5)

### External pairings

- Rajbhandari et al. (2020), "ZeRO: Memory Optimizations Toward
  Training Trillion Parameter Models." Why: the memory-sharding
  technique behind DeepSpeed and FSDP.
- Zhao et al. (2023), "PyTorch FSDP: Experiences on Scaling Fully
  Sharded Data Parallel." Why: how PyTorch implements ZeRO.
- Shoeybi et al. (2019), "Megatron-LM: Training Multi-Billion
  Parameter Language Models Using Model Parallelism." Why: tensor
  parallelism — splitting individual layers across GPUs.
- Narayanan et al. (2021), "Efficient Large-Scale Language Model
  Training on GPU Clusters Using Megatron-LM." Why: pipeline
  parallelism + micro-batching.
- Deep Learning (Goodfellow et al.), Chapter 12 "Applications."
  Why: discusses computational costs and practical training advice.

### Self-check

- What is the difference between data parallelism and model parallelism?
- How does ZeRO Stage 3 shard optimizer state, gradients, AND parameters?
- What is pipeline parallelism and why does it cause "bubble" idle time?
- How much GPU memory does a 7B parameter model need in FP16?
  (Answer: ~14 GB for parameters alone, ~56 GB with optimizer state.)
- What is the "Chinchilla scaling law" and what does it predict?

### You're ready for Phase 8 when...

- You can estimate the memory and compute budget for training a model
  of a given size on given hardware.
- You understand when to use DDP vs. FSDP vs. tensor parallel.
- You can read DeepSpeed configs and understand what they do.

---

## Phase 8: Alignment and inference (external, ~2 weeks)

**Goal:** understand how trained base models are fine-tuned for
instruction-following and safety, and how they're served at inference.

### What to read

1. `docs/gap_map.md` sections 8, 9, 11, 12 (companion doc #2)

### External pairings — Alignment

- Ouyang et al. (2022), "Training language models to follow instructions
  with human feedback." Why: the InstructGPT paper that defined RLHF.
- Rafailov et al. (2023), "Direct Preference Optimization: Your
  Language Model is Secretly a Reward Model." Why: DPO — simpler
  alignment without a separate reward model.
- HuggingFace TRL library source code. Why: production RLHF/DPO
  implementation you can read and modify.

### External pairings — Inference

- Kwon et al. (2023), "Efficient Memory Management for Large Language
  Model Serving with PagedAttention." Why: the vLLM paper. Continuous
  batching + paged KV cache.
- Leviathan et al. (2023), "Fast Inference from Transformers via
  Speculative Decoding." Why: use a small model to draft, large model
  to verify — 2-3x speedup.
- Frantar et al. (2023), "GPTQ: Accurate Post-Training Quantization
  for Generative Pre-trained Transformers." Why: INT4 quantization
  that makes 7B models fit on consumer GPUs.
- Lin et al. (2024), "AWQ: Activation-aware Weight Quantization for
  LLM Compression and Acceleration." Why: better quantization quality
  than GPTQ.

### Self-check

- What is the reward model in RLHF and why is it needed?
- How does DPO avoid training a reward model? What's the trade-off?
- What is KV caching and why is it essential for autoregressive generation?
- What does INT4 quantization do to model quality? (Typical answer:
  < 1% perplexity increase for well-calibrated methods.)
- What is continuous batching and why does it increase GPU utilization?

### You're ready to build your own LLM when...

- You can design a training pipeline: data → tokenization → pretraining
  → SFT → alignment → serving.
- You can estimate whether your hardware can handle a given model size.
- You know which techniques to apply at each stage.
- You can read a modern LLM codebase (Llama, vLLM) and understand
  what every major component does.

---

## Where you end up

After completing all 8 phases:

- **You can read** any modern ML paper's architecture section and
  understand the implementation implications.
- **You can write** custom CUDA kernels for operations not available
  in cuBLAS/cuDNN.
- **You can design** a training run for a model within your hardware
  budget (realistic: 50M-200M params on a single RTX 4060 Ti).
- **You can fine-tune** existing models using LoRA/QLoRA for your domain.
- **You can evaluate** trade-offs (speed vs. memory vs. quality) at
  every level of the stack.
- **You can contribute** meaningfully to open-source LLM projects.

---

## What's beyond this curriculum

If you want to go further after Phase 8:

- **Mixture of Experts** (Fedus et al. 2022, Switch Transformer) —
  sparse computation for scaling
- **State-space models** (Gu & Dao 2023, Mamba) — alternative to
  attention for long sequences
- **Multimodal models** (Liu et al. 2023, LLaVA) — vision + language
- **Retrieval-augmented generation** (Lewis et al. 2020, RAG) —
  grounding in external knowledge
- **Research-level optimization** — custom hardware kernels, novel
  attention patterns, new training objectives

These are research frontiers. This curriculum gives you the foundation
to engage with them productively.

---

## Quick reference: where this repo's docs map to the phases

| Phase | This repo's docs | Companion docs |
|-------|-----------------|----------------|
| 0 | 01 (Zig primer) | learning_roadmap.md (this file) |
| 1 | 00, 02, 02b, 02c, 02d | — |
| 2 | 03, 03b, 03c, 04, 04b | — |
| 3 | 05, 05b, 06, 07, 09, 10 | — |
| 4 | 08, 08b | — |
| 5 | — | cuda_depth.md |
| 6 | — | gap_map.md, from_this_to_llama.md, extensions.md |
| 7 | — | gap_map.md, from_this_to_llama.md |
| 8 | — | gap_map.md, extensions.md |
