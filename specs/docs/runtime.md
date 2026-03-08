The file doesn't exist yet. Let me output the completed documentation directly for you:

# Runtime Model — hologram-ai

## Execution Lifecycle

hologram-ai follows a **compiler-only architecture** (ADR-0016). It is not a
runtime engine but a compiler that produces `.holo` archives consumed by the
hologram executor.

**Compilation Pipeline:**

```
ModelSource (ONNX/GGUF/GGML)
  → Import via format-specific importer
  → AiGraph IR (canonical semantic representation)
  → OptPipeline::mvp() (constant folding, dead code elimination)
  → Validation (DAG check, tensor_info consistency)
  → MemoryPlanner.plan() (KV-cache layout computation)
  → lower() (AiOp → hologram::GraphOp mapping, handler registration)
  → hologram::compile() (LUT fusion, CSE, buffer reuse, schedule)
  → Archive writer (populate LayerHeader, LlmMetaSection)
  → .holo archive (self-describing, ready for execution)
```

**Execution Dispatch:**

The compiled archive declares named execution entrypoints:
- `"lm.prefill"` — variable-length prompt ingestion subgraph
- `"lm.decode"` — single-token autoregressive decode step

Callers use `hologram::HoloLoader` + `hologram::KvExecutor` to:
1. Load the archive
2. Extract `SECTION_LLM_META` for KV-cache layout and layer IDs
3. Allocate KV-cache buffer (mutable, passed through each invocation)
4. Call `KvExecutor::execute_with_weights()` for each graph layer
5. Thread buffer and `present_len` state through the decode loop

**MVP Status:**

Single forward pass via `InferenceSession::run(token_ids) → Vec<f32>`. Multi-turn
context and KV-cache management are deferred to Phase 2.

---

## State Management

**CompiledModel (immutable, shareable):**

```rust
pub struct CompiledModel {
    archive: Vec<u8>,                              // serialized .holo bytes
    schedule: Arc<hologram::ExecutionSchedule>,   // produced by hologram::compile()
    registry: Arc<hologram::CustomOpRegistry>,    // AI-specific op handlers
    kv_layout: Option<KvCacheLayout>,             // KV-cache sizing (None in MVP)
    metadata: ModelMetadata,                      // arch, vocab_size, context_len
}
```

**Ownership Model:**
- `CompiledModel` is thread-safe and reusable; wrap in `Arc` to share
- Multiple `InferenceSession` instances can reference one `CompiledModel`
- Shared: compiled plan (archive), execution schedule, custom op registry
- Per-session: none in MVP; Phase 2 adds per-session KV-cache buffers

**InferenceSession (MVP):**

```rust
pub struct InferenceSession {
    model: Arc<CompiledModel>,
}
```

- Stateless forward pass only in MVP
- Phase 2 extends to hold `Option<KvCache>` for autoregressive generation

**Weight Storage:**

```rust
pub enum AiParam {
    Inline { data: Vec<u8>, info: TensorInfo },  // small weights, embedded
    Mmap { path: PathBuf, offset: u64, ... },    // large weights, memory-mapped
}
```

**Initialization/Teardown:**
- `ModelCompiler::compile()` orchestrates the pipeline; errors propagate via
  `anyhow::Result<CompiledModel>`
- `CompiledModel` dropped automatically; buffers released
- `InferenceSession::new()` wraps compiled model; drop releases reference

---

## Error Handling

**Error Types:**

- `anyhow::Result<T>` — primary Result type for public APIs
- `CommonError` — structured errors from `hologram-ai-common`:

```rust
#[derive(Debug, Error)]
pub enum CommonError {
    #[error("lowering failed: {0}")]
    Lowering(String),
    #[error("graph validation failed: {count} error(s)")]
    Validation { count: usize },
    #[error("unsupported op: {op_type}")]
    UnsupportedOp { op_type: String },
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}
```

**Panic Policy:**

No panics on invalid input. All fallible operations return `Result`. Metadata
extraction uses `.unwrap_or()` for graceful degradation:

```rust
let vocab_size = meta_u32(graph, "vocab_size").unwrap_or(0);
```

**Error Propagation:**

Each pipeline step adds context via `.context()`:

```rust
let ai_graph = self.import(source)
    .with_context(|| format!("importing from {path:?}"))?;
let ai_graph = OptPipeline::mvp().run(ai_graph)
    .context("optimization pass failed")?;
```

**Validation:**

`AiGraph::validate()` returns `Vec<ValidationError>` (collected, not panicked).
Checks include tensor_info consistency, DAG acyclicity, and node I/O
registration. Errors are fatal only if the list is non-empty.

---

## Resource Limits

**Configuration (MVP):**

```rust
pub struct ModelCompiler {
    pub mmap: bool,  // enable memory-mapping for weight loading (default: true)
}
```

**Memory Planning:**

```rust
pub struct MemoryPlan {
    pub kv_cache_layout: KvCacheLayout,
    pub total_weight_bytes: u64,       // exact param byte count
    pub total_activation_bytes: u64,   // conservative estimate
}
```

**Limits:**

| Resource | MVP Behavior | Phase 2 Plan |
|----------|--------------|--------------|
| Model size | No limit (system RAM) | Lazy-loading via mmap |
| Context length | Fixed at compile time | Bucketed variants |
| Batch size | Implicit 1 | Multi-batch prefill/decode |
| Threads | Not configurable | `SessionOptions::threads` |
| Precision | Auto-detected | `LoweringOptions::quant_strategy` |

**Weight Loading Strategy:**

- **Eager (Inline):** Small weights materialized as `ConstantData::Bytes`
- **Lazy (Mmap):** Large weights backed by `hologram::HoloLoader`

Strategy selected at compile time based on `AiParam` variant.

**Timeouts:**

Compilation is synchronous. Execution timeouts are the caller's responsibility
(via OS signals, `tokio::timeout`, etc.).

---

## Thread Safety

**Thread-Safe Types:**

| Type | Safety | Notes |
|------|--------|-------|
| `CompiledModel` | `Send + Sync` via `Arc` internals | Share across threads |
| `Arc<CompiledModel>` | `Send + Sync` | Intended usage pattern |
| `hologram::KvExecutor` | `Send + Sync` | `execute_with_registry(&self)` is immutable |
| `hologram::CustomOpRegistry` | `Send + Sync` | Immutable after registration |
| `hologram::ExecutionSchedule` | `Send + Sync` | Read-only |

**Thread-Unsafe Types (Phase 2):**

| Type | Reason |
|------|--------|
| `InferenceSession` | Will hold per-session KV-cache buffers |
| KV-cache buffers | Per-session, not shared |

**Send/Sync Bounds:**

```rust
// Optimization passes must be Send + Sync for potential parallelization
pub trait Pass: Send + Sync {
    fn name(&self) -> &str;
    fn run(&self, graph: AiGraph) -> anyhow::Result<AiGraph>;
}

// Tokenizer trait requires Send + Sync for embedded access
pub trait Tokenizer: Send + Sync { ... }
```

**Concurrency Patterns:**

- Multiple sessions can call `execute_with_registry()` simultaneously
- Archive loading via `hologram::load_from_bytes()` is read-only and safe to share
- No `Mutex` or `RwLock` required in MVP; stateless execution model

**Usage:**

```rust
let compiled = Arc::new(ModelCompiler::default().compile(source)?);
let mut sess1 = InferenceSession::new(Arc::clone(&compiled));
let mut sess2 = InferenceSession::new(Arc::clone(&compiled));
// sess1 and sess2 share the compiled model, execute independently
```
