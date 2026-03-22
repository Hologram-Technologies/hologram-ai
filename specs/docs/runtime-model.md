# hologram-ai: Runtime Model

---

## Overview

The runtime model defines how compiled models are loaded, how inference sessions
are created and managed, how weights are loaded lazily, and how autoregressive
token generation is driven.

---

## Model Compilation

`ModelCompiler` drives the full compile pipeline:

```rust
pub struct ModelCompiler {
    opt_pipeline: OptPipeline,
    mem_planner: MemoryPlanner,
    lower_opts: LoweringOptions,
    executor: Arc<hologram::KvExecutor>,
}

impl ModelCompiler {
    pub fn compile(&self, source: ModelSource) -> Result<CompiledModel>
}

pub enum ModelSource {
    OnnxBytes(Bytes),
    OnnxPath(PathBuf),
    GgufPath(PathBuf),
    GgmlPath(PathBuf),
    AiGraph(AiGraph),    // already-imported graph
}
```

**Compile pipeline steps:**

```
ModelSource
  → import_*(...)                           → AiGraph (raw)
  → opt_pipeline.run(graph)                 → AiGraph (AI-level fusions applied)
  → mem_planner.plan_kv_cache(&graph)       → KvCacheLayout  (KV-cache sizing only)
  → lower(&graph, &kv_layout, &opts)        → LoweringOutput { graph, registry }
  → hologram::compile(lower.graph)          → CompilationOutput { archive, schedule, stats }
  → extract tokenizer data from metadata   → NativeTokenizer (packed into ConstantStore + .holo section)
  → CompiledModel { archive, schedule, registry, kv_layout, tokenizer } → ready for sessions
```

The two optimization phases are complementary (see ADR-0008):

- **`opt_pipeline`** — semantic AI passes on `AiGraph` (attention fusion, FFN fusion,
  QuantMatMul fusion). Runs before lowering. `hologram-compiler` cannot perform
  these because it has no concept of `AiOp` variants.

- **`hologram::compile()`** — generic graph passes on `hologram::Graph` (LUT chain
  fusion, CSE, liveness analysis, workspace slot reuse via bin packing). Runs after
  lowering. `hologram-ai` should not re-implement these.

**`MemoryPlanner` scope** is limited to KV-cache layout. Intermediate activation buffer
reuse is handled by `hologram::compile()` internally.

---

## `CompiledModel`

A `CompiledModel` is a reusable, shareable compiled artifact:

```rust
pub struct CompiledModel {
    // hologram-compiler output
    archive: Arc<Vec<u8>>,                       // serialized compiled plan (for caching)
    schedule: Arc<hologram::ExecutionSchedule>,  // produced by hologram::compile()
    // lowering output
    registry: Arc<hologram::CustomOpRegistry>,   // AI-specific op handlers
    // execution
    executor: Arc<hologram::KvExecutor>,
    kv_layout: Option<KvCacheLayout>,
    input_metadata: Vec<TensorMeta>,
    output_metadata: Vec<TensorMeta>,
    metadata: ModelMetadata,    // arch info, context len, vocab size, etc.
    // tokenizer (see ADR-0012)
    tokenizer: Option<Arc<dyn Tokenizer>>,       // embedded native tokenizer from .holo
}

impl CompiledModel {
    pub fn session(&self, opts: SessionOptions) -> Result<InferenceSession>
    pub fn metadata(&self) -> &ModelMetadata
    /// Returns the embedded tokenizer, if available.
    /// Auto-constructed from ConstantStore when loading .holo archives
    /// that contain SECTION_TOKENIZER (0x1001).
    pub fn tokenizer(&self) -> Option<&dyn Tokenizer>
}
```

Multiple `InferenceSession` instances can be created from a single
`CompiledModel` (they share the compiled plan; each has its own KV-cache).

---

## Weight Loading

Weights are stored as `ConstantData` in `hologram::ConstantStore`, which is embedded
in the `hologram::Graph`. Large models use `ConstantData::Deferred` for lazy loading.

### Strategies

**Eager (full load):** All weights materialized as `ConstantData::Bytes` at `CompiledModel`
creation. Preferred for small models or when the full model fits in RAM.

**Lazy (mmap):** Weights backed by `hologram::HoloLoader` (memory-mapped `.holo` archive).
Pages are faulted in on first access. Preferred for large GGUF models (70B+).
`HoloLoader` guarantees mmap validity across multiple execution calls (confirmed API).

`ParamStorage` in `AiParam` controls this:
```rust
pub enum ParamStorage {
    Inline(Bytes),                    // → ConstantData::Bytes in ConstantStore
    Lazy(hologram::ConstantId),       // → ConstantData::Deferred, loaded via HoloLoader
}
```

---

## Inference Session

```rust
pub struct InferenceSession {
    model: Arc<CompiledModel>,
    kv_cache: Option<KvCache>,     // owns BufferArena for KV storage
    opts: SessionOptions,
}

pub struct SessionOptions {
    pub max_seq_len: Option<usize>,   // cap for KV-cache allocation
    pub kv_cache_dtype: DType,        // f16 recommended for memory efficiency
    pub threads: Option<usize>,       // CPU thread count
    pub seed: Option<u64>,            // for sampling reproducibility
}
```

### `session.run()` — Single Forward Pass

```rust
impl InferenceSession {
    pub fn run(
        &mut self,
        inputs: HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>>
}
```

Steps:
1. Validate input tensor shapes against `CompiledModel::input_metadata`
2. If KV-cache is present, inject cache pointers and current offset as `GraphInputs`
3. Call `model.executor.execute_with_registry(&model.schedule, &inputs, &model.registry)`
4. Wait for completion (synchronous — `KvExecutor::execute` returns `GraphOutputs`)
5. Extract output tensors from `GraphOutputs`
6. If KV-cache present, update `kv_cache.present_len`
7. Return output tensors

---

## KV-Cache

### Design

The KV-cache stores the key and value tensors from all attention layers for
all previously processed tokens. This avoids recomputing them on each decode step,
trading memory for compute — O(seq_len) per decode step instead of O(seq_len^2).

```rust
pub struct KvCache {
    pub layers: Vec<KvLayer>,
    pub layout: KvCacheLayout,
    pub present_len: usize,         // tokens currently in cache
    pub max_seq_len: usize,
}

pub struct KvLayer {
    pub k: Vec<u8>,     // [max_seq_len, n_kv_heads, head_dim] — raw bytes in BufferArena
    pub v: Vec<u8>,     // [max_seq_len, n_kv_heads, head_dim]
}

pub struct KvCacheLayout {
    pub layers: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub dtype: DType,
    pub bytes_per_layer: u64,
    pub total_bytes: u64,
}
```

### Contiguous allocation (current)

The current implementation pre-allocates flat contiguous buffers per layer,
sized to `max_seq_len`:

```
Total KV memory = 2 * n_layers * n_kv_heads * max_seq_len * head_dim * sizeof(dtype)
```

For TinyLlama (22 layers, 4 KV heads, 64 head_dim, 2048 max_seq): ~90 MB.
For larger models (70B, 128K context), this can exceed available memory even
when actual generation is much shorter than the maximum.

### Paged attention (planned — Plan 016)

Paged attention replaces monolithic per-layer buffers with small fixed-size
pages allocated on demand, borrowing virtual memory concepts from operating systems.

**Core data structures:**

- **Page**: fixed-size block holding `page_size` token slots (e.g., 16 tokens)
  for one KV head group. Layout: `[page_size, n_kv_heads, head_dim]` as flat f32.
- **PagePool**: arena of physical pages with a free list for recycling.
  Pages are never deallocated — reset returns them to the free list.
- **BlockTable**: per-layer mapping of logical block index to physical page ID.
  `block_table[i]` holds the page for tokens `[i*page_size .. (i+1)*page_size)`.
- **PagedKvCache**: wraps pool + per-layer K/V block tables + write_pos.

**How it works:**

1. On `KvWrite`, if `write_pos` crosses a page boundary (`write_pos % page_size == 0`),
   a new page is allocated from the pool for that layer's K and V.
2. Data is written to `block_table[write_pos / page_size][write_pos % page_size]`.
3. On `KvRead`, pages are gathered (iterated in block table order) and concatenated
   into a contiguous `[seq_cached, n_kv_heads, head_dim]` tensor — paging is invisible
   to the attention kernel.
4. On reset, all pages return to the free list. No deallocation, just recycling.

**Benefits:**

- No upfront `max_seq_len` allocation — pages allocated only as the sequence grows
- A 5-token generation on TinyLlama allocates 1 page per layer (~360 KB) instead of
  the full 90 MB
- Foundation for multi-request batching: per-request block tables sharing a global
  page pool, copy-on-write for shared system prompt prefixes (future phase)

**Configuration:** `kv_page_size` field in `ModelMetaSection`. Value of 0 falls back
to contiguous allocation for backward compatibility. Default: 16 tokens per page.

See `specs/plans/016-paged-attention.md` for the full implementation plan.

### KV-cache in the execution plan

Two special node types in the lowered plan:

`FloatOp::KvWrite { layer, n_kv_heads, head_dim, is_key }` — writes the current
token's K/V projections into the cache. Pass-through at the AiGraph level (returns
input unchanged); lowering converts it to runtime cache writes.

`FloatOp::KvRead { layer, n_kv_heads, head_dim }` — reads all cached K/V up to
`write_pos` for attention computation. Shape is runtime-dependent (grows each step).

These are lowered from their AiGraph equivalents:

- `AiOp::KvSlotWrite { layer, is_key, n_kv_heads, head_dim }` — injected by
  `KvSlotInjection` pass (ONNX) or during graph construction (GGUF)
- `AiOp::KvSlotRead { layer, n_kv_heads, head_dim }` — appears in decode-phase graphs

Metadata flows forward (import → IR → lowered graph → archive → runtime), never backward.
All architecture params (`n_kv_heads`, `head_dim`) are carried on the ops themselves.

### Prefill vs. decode modes

The compiled LLM produces a **pipeline archive** containing two sub-graphs:

**Prefill graph (step 0):**
- Input: full prompt (variable length, padded to compiled seq_len)
- Contains `KvSlotWrite` nodes on all K/V projections
- Returns logits for all positions
- Populates KV cache for all prompt positions at once

**Decode graph (steps 1+):**
- Input: single token (seq=1)
- Contains both `KvSlotRead` (retrieve cached K/V) and `KvSlotWrite` (append new K/V)
- Returns logits for the new token only
- Cache grows by 1 position each step

The runtime routes to the correct sub-graph based on `write_pos`:
`write_pos == 0` → prefill graph, `write_pos > 0` → decode graph.

### Multi-component pipeline archives (Plan 021)

The 2-graph LLM pipeline is a specialization of a general N-component
pipeline archive. `compile_components()` compiles any number of independent
`AiGraph` instances into a single `.holo` file via `PipelineWriter`.

**Archive structure:**

```
Wrapper .holo archive
├─ Weights section: [sub-archive 0 | padding | sub-archive 1 | ... | sub-archive N]
├─ SECTION_PIPELINE: PipelineHeader { models: [name, offset, size, crc32] }
└─ SECTION_META: MetaSection { components: [...], connections: [...] }
```

**MetaSection** describes component roles and data flow:

```rust
pub struct MetaSection {
    pub components: Vec<ComponentDescriptor>,  // name, role, weight_group, weight_source
    pub connections: Vec<ComponentConnection>, // from_component.output → to_component.input
}
```

For the LLM case, `MetaSection` contains 2 descriptors (Prefill + Decode,
both in weight_group "lm") and 1 connection (shared KV-cache).

For a CALM model, it would contain 4 descriptors (Encoder, Backbone,
GenerativeHead, Decoder) and 3 connections describing data flow.

Each component can specify an `OptProfile` to control which optimization
passes run:
- `Llm` — full MVP pipeline (attention fusion, KV injection, SwiGLU fusion)
- `Generic` — shape/data propagation + constant folding only

Components that don't use attention get `MemoryPlan::empty()` (no KV-cache
allocation).

See [Plan 021](../plans/021-multi-component-archives.md) for the full design.

---

## Token Generation

### `session.generate()` — Blocking generation

```rust
impl InferenceSession {
    pub fn generate(
        &mut self,
        token_ids: &[u32],           // prompt tokens
        opts: &GenerateOptions,
    ) -> Result<Vec<u32>>            // generated tokens (not including prompt)
}
```

### `stream_tokens()` — Async streaming

```rust
pub fn stream_tokens(
    session: InferenceSession,
    tokenizer: Option<Box<dyn Tokenizer>>,  // None → use model's embedded tokenizer
    prompt: &str,
    opts: GenerateOptions,
) -> TokenStream

impl Stream for TokenStream {
    type Item = Result<Token>;
}
```

### `GenerateOptions`

```rust
pub struct GenerateOptions {
    pub max_new_tokens: usize,
    pub temperature: f32,        // 1.0 = no scaling
    pub top_p: f32,              // 1.0 = disabled
    pub top_k: usize,            // 0 = disabled
    pub min_p: f32,              // 0.0 = disabled
    pub repetition_penalty: f32, // 1.0 = disabled
    pub stop_sequences: Vec<Vec<u32>>,
    pub seed: Option<u64>,
}
```

### Sampling pipeline

```
logits (f32 vector of vocab_size)
  → apply temperature scaling
  → apply repetition penalty (if enabled)
  → top-k filter (if k > 0)
  → top-p filter (if p < 1.0)
  → min-p filter (if p > 0.0)
  → softmax
  → sample from distribution (or argmax for greedy)
  → check stop sequences
  → emit Token
```

---

## Session Lifecycle

```
CompiledModel::session(opts)          → InferenceSession (fresh, empty cache)
  │
  ├── run(inputs)                     → single forward pass
  │
  ├── generate(prompt_tokens, opts)   → blocking full generation
  │
  ├── stream_tokens(...)              → async Token stream
  │                                     (uses compiled.tokenizer() by default)
  │
  ├── reset_cache()                   → clear KV-cache, reuse session
  │
  └── drop                           → buffers released

CompiledModel::tokenizer()            → Option<&dyn Tokenizer> (embedded, from .holo)
```

Multiple concurrent sessions from the same `CompiledModel`:
- Each has independent KV-cache `BufferArena`
- Shared `Arc<Graph> + Arc<ExecutionSchedule>` — read-only
- Shared `Arc<CustomOpRegistry>` — read-only after registration
- Shared `Arc<KvExecutor>` — stateless (`execute` takes `&self`), concurrent-safe

---

## Execution Dispatch

Execution goes through `hologram::KvExecutor` with a `CustomOpRegistry`:

```rust
// Shared across all sessions from the same CompiledModel
// schedule was produced by hologram::compile() — not by lower()
let outputs = model.executor.execute_with_registry(
    &model.schedule,
    &inputs,
    &model.registry,
)?;
```

`KvExecutor::execute_with_registry` is stateless and takes `&self`. There is no
`ExecutionBackend` trait, no capability query, and no backend selection. AI-specific
capabilities are determined at lowering time by which handlers are registered in
`CustomOpRegistry`. The registry is immutable after `CompiledModel` creation.

---

## Resolved Questions (previously open)

These questions were open when the architecture was first written but are now
resolved based on the actual hologram API (see ADR-0007).

1. **Dynamic shape support** — Resolved. `hologram::Graph` is constructed with
   concrete shapes. For MVP, `seq_len` is fixed at `max_seq_len` at compile time.
   The session rebuilds the `Graph` if a different concrete seq_len is required.
   No runtime variable shape dispatch is needed for MVP.

2. **mmap semantics** — Resolved. `hologram::HoloLoader` (hologram-archive crate)
   provides memory-mapped loading via `HoloLoader::open(path)` + `HoloLoader::load()`.
   The loaded `LoadedPlan` keeps the mmap alive for its lifetime. `ConstantData::Deferred`
   allows lazy page faulting. There is no `hologram-artifacts` crate.

3. **Thread-safety of execution** — Resolved. `KvExecutor::execute_with_registry`
   takes `&self` (immutable reference). `KvExecutor` is stateless — it has no
   per-call mutable state. Multiple sessions can call it concurrently from different
   threads with no synchronization required.

4. **KV-cache buffer ownership** — Resolved. Session owns KV-cache as a
   `BufferArena` (stack of `Vec<u8>`). The arena is passed into the executor per call.
   There is no pool allocation in the MVP. Each session allocates its own arena at
   construction from `KvCacheLayout` sizing.

---

## Testing Strategy

### Guiding Principles

1. **Validate numerically, not structurally.** Tests must check that outputs are
   correct, not just that no panic occurred.

2. **Reference runtimes are the ground truth.** ONNX Runtime and llama.cpp are
   used as oracles for correctness, not as execution backends.

3. **Fixtures first.** Every test has a committed or script-reproducible fixture.
   Tests that require download are marked `#[ignore]` and run separately in CI.

4. **Tolerance policy must be explicit.** Every floating-point comparison
   declares its tolerance and the reason for it.

5. **Layer tests at every boundary.** Unit tests per crate. Integration tests at
   the importer-IR boundary and the IR-session boundary.

---

### Test Taxonomy

#### Unit Tests (per-crate)

| Crate | Primary test targets |
|-------|---------------------|
| `hologram-ai-quant` | dequant numerical correctness (Q4_0, Q8_0, Q4_K, etc.) |
| `hologram-ai-common` | `AiGraph` construction/validation, opt passes, memory planner, lowering dispatch |
| `hologram-ai-onnx` | op_map completeness, shape inference, specific op behaviors |
| `hologram-ai-gguf` | header parsing, metadata extraction, quant type mapping |
| `hologram-ai-ggml` | header parsing, tensor extraction |
| `hologram-ai` | session lifecycle, run() input/output validation, sampler distributions, stop sequences |

#### Integration Tests (`tests/integration/`)

```rust
#[test]
fn gguf_tinyllama_single_forward_pass_cpu() {
    let model = ModelCompiler::default().compile(ModelSource::GgufPath("...")).unwrap();
    let mut session = model.session(Default::default()).unwrap();
    let inputs = hashmap!["input_ids" => Tensor::from(&[1u32, 2, 3, 4][..])];
    let outputs = session.run(inputs).unwrap();
    assert!(outputs["logits"].shape()[1] > 0);
}
```

---

### Importer Fixture Tests

Tiny/synthetic models committed into the repo:

| Fixture | Format | Purpose |
|---------|--------|---------|
| `matmul-f32.onnx` | ONNX | single MatMul node, known inputs/outputs |
| `relu-f32.onnx` | ONNX | single Relu, activation shape test |
| `attention-opset17.onnx` | ONNX | simplified MHA block |
| `tinyllama-tiny.gguf` | GGUF | 2-layer, 32-dim synthetic llama; Q4_0 |
| `phi-tiny.gguf` | GGUF | 2-layer, 32-dim synthetic phi; F16 |

Fixtures are generated via `scripts/gen-fixtures.py` (Python). Committed at small size (<1MB each).

---

### Golden Tensor Tests

```
tests/golden/
  matmul-f32/
    input.npz
    output.npz
  tinyllama-tiny/
    input_ids.npz
    logits.npz
```

---

### Reference Runtime Comparison Tests

These tests compare `hologram-ai` against external runtimes (ONNX Runtime, llama.cpp).
They are `#[ignore]` by default and run in nightly CI only.

```rust
#[test]
#[ignore = "requires ort CLI"]
fn onnx_bert_base_matches_onnxruntime() {
    let input = load_test_input("bert-base");
    let holo_out = run_hologram_ai("bert-base.onnx", &input);
    let ort_out = run_ort_cli("bert-base.onnx", &input);
    assert_tensors_close(&holo_out["logits"], &ort_out["logits"], Tolerance {
        max_abs_err: 1e-5,
        mean_abs_err: 1e-6,
        cosine_sim_min: 0.9999,
    });
}
```

---

### Tolerance Policy

| Model dtype | max_abs_err | mean_abs_err | cosine_sim_min |
|-------------|-------------|--------------|----------------|
| F32 | 1e-5 | 1e-6 | 0.9999 |
| F16 | 1e-3 | 1e-4 | 0.999 |
| Quantized | `quant_noise_floor(scheme) * 2.0` | `quant_noise_floor(scheme)` | 0.99 |

For token generation: **top-1 greedy token must match** the reference for the same model,
prompt, and temperature (greedy = 0). Top-5 match is a warning, not a failure.

---

### CI Test Matrix

```yaml
jobs:
  unit_tests:
    runs-on: [ubuntu-latest, macos-latest]
    steps: [cargo test --workspace]

  integration_tests:
    runs-on: ubuntu-latest
    steps: [cargo test --test integration]

  golden_tests:
    runs-on: ubuntu-latest
    steps: [cargo test --test golden]

  reference_tests:
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule'   # nightly only
    steps:
      - ./scripts/download-test-models.sh
      - cargo test --test reference -- --ignored
```
