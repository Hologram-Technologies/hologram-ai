The two versions of `roadmap.md` are identical. There are no differences between the ARCH VERSION and the SUBPROJECT VERSION - they contain exactly the same content, structure, sections, and text.

Since they are identical, the merged output is simply the document itself:

---

# hologram-ai: Roadmap

---

## MVP (Weeks 1–4)

**Goal:** Compile a GGUF decoder-only LLM to a `.holo` archive with named
`lm.prefill` and `lm.decode` entrypoints. Execute via `KvExecutor` directly.

### Scope

- GGUF importer for LLaMA-family models (Q4_0 quantization)
- `AiGraph` IR with core ops and quant descriptors
- Optimization: constant folding, shape propagation, attention fusion
- Memory planner: KV-cache layout computation → `KvCacheLayout`
- Multi-graph lowering: prefill graph + decode graph (separate, same weights)
- `hologram::compile()` produces `ExecutionSchedule` for each graph
- Archive writer populates `LayerHeader` (0x0002) and `SECTION_LLM_META` (0x0011)
- Validation: single-pass logits golden test via `KvExecutor` against committed fixture
- CLI: `hologram-ai compile tinyllama.gguf -o tinyllama.holo`
- CI: unit tests + integration smoke test

### Exit criteria

- `hologram-ai compile tinyllama.gguf` produces a valid `.holo` archive
- Archive `LayerHeader` declares `lm.prefill` and `lm.decode` with correct tensor ports
- `SECTION_LLM_META` reports correct `KvCacheLayout` for TinyLlama 1.1B
- Calling `KvExecutor::execute_layer("lm.prefill", ...)` yields logits of correct shape
- Top-1 logit matches llama.cpp reference (greedy) on golden prompt
- All unit tests pass on `aarch64-apple-darwin` and `x86_64-unknown-linux-gnu`

### Explicitly deferred from MVP

- ONNX importer
- GGML importer
- Metal backend
- Tokenizer integration
- Bucketed (multi-variant) compilation

---

## Phase 2 (Weeks 5–10)

**Goal:** Full compiler coverage — ONNX + GGML importers, extended arch recognizers,
symbolic shapes, bucketed multi-entrypoint archives, tokenizer embedding, and validation
harness. (ADR-0016: hologram-ai is a compiler; session management is caller-side.)

### Scope

- ONNX importer (opset 13–21, covering BERT + GPT-2 + encoder-decoder models)
- GGML importer (full topology, migration path for legacy weights)
- Extended GGUF arch recognizers: Mistral, Phi, Phi-3, Qwen, Qwen2, Gemma, Gemma2,
  Mixtral, DeepSeek
- Symbolic shape system: `DimExpr` algebra, `DimVarTable` with bounds, `ShapePropagation`
  pass, constraint validation, bucketed `LayerHeader` emission (ADR-0015)
- Bucketed archives: `ShapeStrategy::Bucketed` emits N named `lm.decode.*` layers in
  `LayerHeader`; `SECTION_LLM_META` records bucket sizes for caller-side selection
- Tokenizer embedding: `SECTION_TOKENIZER` (0x1001) in output archive (ADR-0012)
- Validation harness: compile model + call `KvExecutor` directly + compare to ORT/llama.cpp
- CLI: `hologram-ai compile`, `hologram-ai inspect`, `hologram-ai validate`
- CLI: `hologram-ai download` — HuggingFace model acquisition + ONNX conversion
- CLI: `hologram run` — generic archive runner; reads `SECTION_LLM_META` +
  `SECTION_TOKENIZER`, drives `KvExecutor` directly (defined in `hologram`, not
  `hologram-ai`; the archive is self-describing)
- `--stats`: compile time, archive size; BF16 dtype support

### Milestones

| Milestone | Deliverable |
|-----------|------------|
| M2.1 | Archive: `lm.prefill` + `lm.decode` layers properly declared in `LayerHeader` |
| M2.2 | Archive: `SECTION_LLM_META` (0x0011) records `KvCacheLayout` + entrypoint IDs |
| M2.3 | ONNX: BERT base classification archive passes numerical validation vs ORT |
| M2.4 | ONNX: GPT-2 small archive produces logits matching ORT reference |
| M2.5 | CLI: `hologram-ai compile model.gguf -o model.holo` produces valid archive |
| M2.6 | CLI: `hologram-ai inspect model.holo` reports layer names, tensor ports, KV layout |
| M2.7 | CLI: `hologram-ai download` acquires GGUF models from HuggingFace |
| M2.8 | CLI: `hologram-ai download --format onnx` triggers Python virtualenv conversion |
| M2.9 | CLI: `hologram run model.holo "prompt"` runs decode loop, prints text (reads `SECTION_LLM_META` + `SECTION_TOKENIZER`) |
| M2.10 | Tokenizer: `NativeTokenizer` BPE encode/decode passes golden tests for LLaMA |
| M2.11 | Tokenizer: GGUF importer extracts vocab/merges into `ConstantStore` |
| M2.12 | Tokenizer: `.holo` archives include embedded tokenizer section (0x1001) |
| M2.13 | Tokenizer: `hologram run` decodes token IDs to text via embedded `SECTION_TOKENIZER` |
| M2.14 | Shapes: `DimExpr` + `DimVarTable` types replace `Dim` enum (Phase 0, no behavior change) |
| M2.15 | Shapes: `ShapePropagation` pass fills output shapes symbolically for TinyLlama graph |
| M2.16 | Shapes: Bucketed archive emits N `lm.decode.*` layers; caller selects by seq_len |

---

## Phase 3 (Weeks 11–18)

**Goal:** Metal backend, quantized kernels, performance validation.

### Scope

- Metal backend integration (`hologram-ai-backend-metal`)
- Quantized GEMM kernels on Metal (Q4_0, Q8_0)
- Flash attention kernel integration (Metal and CPU)
- `hologram-ai-opt`: FFN fusion, layer-norm fusion passes
- Larger model support: 7B/13B models with mmap weight loading
- Performance benchmarking harness
- BF16 support
- Float8 support (experimental)

### Milestones

| Milestone | Deliverable |
|-----------|------------|
| M3.0a | Tokenizer: SentencePiece (unigram) support in `NativeTokenizer` |
| M3.0b | Tokenizer: WordPiece support in `NativeTokenizer` (BERT-class models) |
| M3.1 | Metal backend: TinyLlama runs on Apple Silicon GPU |
| M3.2 | Quantized GEMM: 2x throughput improvement vs eager dequant |
| M3.3 | 7B model: Mistral 7B Q4_K_M generates 10+ tokens/sec on M2 |
| M3.4 | Benchmark suite published (tokens/sec, memory usage) |

---

## Phase 4 (Future)

**Goal:** Multi-backend portability, larger models, and advanced inference features.

### Items

- CUDA backend (`hologram-ai-backend-cuda`)
- WebGPU backend (`hologram-ai-backend-webgpu`)
- Multi-GPU tensor parallelism
- Speculative decoding
- Continuous batching (for server workloads)
- LoRA / adapter layer support in GGUF
- GGUF v4 format support (as it evolves)
- INT4 block quantization on all backends
- GPTQ / AWQ quantization import
- Vision-language model support (multi-modal inputs)
- Autograd / fine-tuning exploration (separate branch, not MVP concern)

---

## Technical Milestones vs Demo Milestones

### Technical milestones (internal quality gates)

| ID | Description | Phase |
|----|-------------|-------|
| T1 | GGUF parser handles all current quant types | MVP |
| T2 | `AiGraph` validation passes for all committed fixtures | MVP |
| T3 | Lowering table covers all ops in LLaMA graph | MVP |
| T4 | Memory planner deterministic across runs | MVP |
| T5 | `LayerHeader` declares correct tensor ports for `lm.prefill` + `lm.decode`; KV-cache offset math in KvSlotWrite/Read nodes correct | Phase 2 |
| T6 | ONNX opset 13–17 coverage >90% of ops in test model set | Phase 2 |
| T7 | f32 numerical error < 1e-5 vs ORT on all ONNX test models | Phase 2 |
| T10 | BPE encode/decode round-trip matches HuggingFace tokenizers for LLaMA vocab | Phase 2 |
| T11 | `.holo` archives with embedded tokenizer load and function correctly | Phase 2 |
| T12 | `ShapePropagation` fills all output shapes for TinyLlama GGUF graph with symbolic dims | Phase 2 |
| T13 | Shape constraints (MatMul inner dim, broadcast compat) collected and validated at concretization | Phase 2 |
| T8 | Metal backend passes same golden tests as CPU | Phase 3 |
| T9 | 7B model generates at ≥10 tokens/sec on M2 Pro | Phase 3 |

### Demo milestones (user-visible)

| ID | Description | Phase |
|----|-------------|-------|
| D1 | `hologram-ai compile tinyllama.gguf -o t.holo && hologram run t.holo "Hello"` produces coherent output | MVP |
| D2 | Multi-turn conversation via `hologram run` CLI (inline generation loop, embedded tokenizer, `SECTION_LLM_META`) | Phase 2 |
| D3 | BERT sentiment classification demo via ONNX | Phase 2 |
| D4 | 7B model chat demo on Apple Silicon | Phase 3 |
| D5 | Side-by-side perf comparison with llama.cpp on same hardware | Phase 3 |

---

## Explicit Sequencing Rationale

**GGUF before ONNX** because:
- GGUF is the active LLM ecosystem format
- Decoder-only LLMs are the primary inference workload
- GGUF exercizes quantization early (critical for design validation)
- ONNX adds import complexity (protobuf, opset, external data) that distracts
  from the core compiler pipeline bring-up

**CPU before Metal** because:
- CPU backend is available on all CI machines
- Numerical correctness is easier to debug on CPU
- Metal backend depends on hologram-metal being ready
- Architecture is designed to be backend-agnostic from day one

**Two graphs (prefill + decode) rather than one** because:
- Prefill processes variable-length prompts; decode processes one token at a time
- They have different graph shapes and different KV-cache write/read patterns
- Emitting them as separate named layers in `LayerHeader` lets the caller
  (or hologram-network executor) dispatch each independently
- Shared weights are stored once in the `ConstantStore` (both graphs reference
  the same `ConstantId`s)

---

## Deferred Items (explicitly not in any phase above)

- Training / autograd
- Distributed inference (beyond single-machine multi-GPU)
- On-device fine-tuning
- Model compression utilities (post-training quantization, pruning)
- Safetensors format import
- PyTorch TorchScript import
- Inference session management library (callers use `HoloLoader` + `KvExecutor` directly
  per ADR-0016; session convenience wrappers are application-scope, not hologram-ai scope)

---

## Risk Register

### R-01: Operator Coverage Gaps

**Impact:** High | **Likelihood:** High | **Phase:** MVP+

ONNX has 200+ ops. Models from unsupported architectures produce `AiOp::Opaque`
nodes that block lowering.

**Mitigation:** `AiOp::Opaque` is explicit — failures are clear errors, not panics.
Track coverage gaps via HuggingFace ONNX model zoo runs. Close gaps by model popularity.

---

### R-02: Quantization Complexity

**Impact:** High | **Likelihood:** High | **Phase:** MVP+

GGUF has 20+ schemes with subtly different block layouts. Incorrect dequant
produces numerically wrong outputs that may look plausible.

**Mitigation:** Unit test every quant scheme with precomputed reference values
from GGML source. Validate against llama.cpp `--debug-dump-quants` mode.
Priority: Q4_0 → Q8_0 → Q4_K_M → Q5_K_M → Q6_K → remainder.

---

### R-03: Backend Kernel Capability Mismatch

**Impact:** Medium | **Likelihood:** Medium | **Phase:** Phase 2–3

`hologram-ai-lower` assumes certain hologram kernels exist. If they don't,
lowering falls back to slower decomposed paths.

**Mitigation:** Every kernel has a software fallback path. CPU backend is
all-software and guaranteed to work. Sync with hologram team before Phase 3.

---

### R-04: Memory Planning Bugs

**Impact:** High | **Likelihood:** Medium | **Phase:** MVP+

Incorrect liveness → double-write or use-after-free of aliased buffers.

**Mitigation:** Use conservative (no-alias) planning for MVP (`conservative: bool`
flag, default true). Introduce aliasing incrementally with explicit tests.

---

### R-05: Dynamic Shape Complexity

**Impact:** Medium | **Likelihood:** Medium | **Phase:** Phase 2

`seq_len` is dynamic; `hologram::Graph` needs concrete shapes.

**Resolution:** Symbolic shape system (ADR-0015) with phased lowering strategies:
- **MVP (FixToMax):** Fix `seq_len = max_seq_len` at lowering time; rebuild graph
  when different concrete seq_len is required.
- **Phase 2 (Bucketed):** Compile N variants for `seq_len` bucket sizes (e.g.,
  128, 512, 1024, 2048). Runtime selects smallest bucket ≥ actual value.
- **Phase 3 (Profiles/PaddedMax):** Multi-variable specialization or fixed-max
  with attention masking for actual length.

See [symbolic-shapes.md](specs/projects/hologram-ai/symbolic-shapes.md) for the
full strategy taxonomy and `DimExpr`/`DimVarTable` type design.

---

### R-06: LLM Runtime Semantics Drift

**Impact:** Medium | **Likelihood:** Medium | **Phase:** Phase 3+

New architectures (Mamba, RWKV, SSM) don't map cleanly to transformer-centric
`AiOp` set.

**Mitigation:** Keep `AiOp` extensible with `Opaque` and custom op escape hatches.
Do not over-specialize in MVP. Review op set every 6 months.

---

### R-07: Native Tokenizer Implementation

**Impact:** Medium | **Likelihood:** Low | **Phase:** Phase 2

Native BPE/SentencePiece/WordPiece tokenizer implementation using hologram
primitives (see ADR-0012). Risks include: BPE correctness (merge ordering,
byte-fallback edge cases), Unicode normalization (NFC/NFKC, combining
characters, zero-width joiners), pre-tokenization regex compatibility
across LLaMA/GPT/Mistral tokenizer variants, and special token handling
differences between model families.

**Mitigation:** Golden test suite validated against HuggingFace `tokenizers`
crate as reference (test-only dependency, not runtime). Start with LLaMA BPE
(best documented, widest coverage). Expand incrementally by model family.

---

### R-08: Portability (Windows / WASM)

**Impact:** Medium | **Likelihood:** Medium | **Phase:** Phase 2–3

SIMD intrinsics differ across targets. WASM has no threading model.

**Mitigation:** `#[cfg(target_arch = ...)]` guards everywhere. Pure-Rust
software fallbacks for all SIMD ops. WASM: importer + IR crates compile fine;
backends are feature-gated.

---

### R-09: Performance Validation

**Impact:** Medium | **Likelihood:** Medium | **Phase:** Phase 2–3

Without benchmarking infrastructure, performance regressions go undetected.

**Mitigation:** Define tokens/sec benchmark harness in Phase 2. Track memory
footprint as a second metric. Do not micro-optimize in MVP.

---

### R-10: hologram API Instability

**Impact:** High | **Likelihood:** Medium | **Phase:** MVP+

`Graph`, `KvExecutor`, `CustomOpRegistry`, `BufferArena`, or `ConstantStore`
API changes break `hologram-ai` compilation.

**Mitigation:** Pin hologram dependency to a specific revision for MVP.
Document the minimal hologram API surface in section 2 of `architecture.md`.

---

### Risk Summary

| ID | Risk | Impact | Likelihood | Phase |
|----|------|--------|------------|-------|
| R-01 | Operator coverage gaps | High | High | MVP+ |
| R-02 | Quantization complexity | High | High | MVP+ |
| R-03 | Backend kernel mismatch | Medium | Medium | Phase 2–3 |
| R-04 | Memory planning bugs | High | Medium | MVP+ |
| R-05 | Dynamic shape complexity (ADR-0015) | Medium | Medium | Phase 2 |
| R-06 | LLM semantics drift | Medium | Medium | Phase 3+ |
| R-07 | Native tokenizer impl | Medium | Low | Phase 2 |
| R-08 | Portability (WASM/Win) | Medium | Medium | Phase 2–3 |
| R-09 | Performance validation | Medium | Medium | Phase 2–3 |
| R-10 | hologram API instability | High | Medium | MVP+ |
