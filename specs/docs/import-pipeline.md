# hologram-ai: Import Pipeline

---

## Overview

The import pipeline converts a foreign model artifact into an `AiGraph`.

The pipeline has a hard boundary: format-specific parsing logic must be fully
contained within its importer crate. After `import_*()` returns, no downstream
code has knowledge of which format produced the graph.

```
┌─────────────────────────────────────────────────────────────────┐
│                       FORMAT BOUNDARY                           │
│                                                                 │
│  .onnx ──► hologram-ai-onnx ─┐                                     │
│  .gguf ──► hologram-ai-gguf ─┼──► AiGraph  ──► (rest of pipeline) │
│  .bin  ──► hologram-ai-ggml ─┘                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Supported Formats

| Format | Status | Notes |
|--------|--------|-------|
| GGUF | MVP | LLaMA-family models; Q4_0 quantization first, then Q8_0, Q4_K_M, Q5_K_M, Q6_K |
| ONNX | Phase 2 | Opset 13–21; BERT, GPT-2, encoder-decoder models |
| GGML | Phase 2 | Legacy pre-GGUF format; migration path for older weights |

---

## Pipeline Stages

### 1. Parse

Each importer reads the format-specific binary representation:

- **GGUF:** Binary header + KV metadata + raw tensor data. `hologram-ai-gguf` uses `memmap2` for zero-copy weight access.
- **ONNX:** Protobuf `ModelProto` decoded via `prost`-generated bindings. External data files resolved via `data_resolver`.
- **GGML:** Fixed binary layout with hardcoded tensor offsets.

Format-specific types (GGUF `GgufMetadata`, ONNX `ModelProto`, etc.) are private to their importer crate per ADR-0003.

### 2. Validate

Each importer validates format-specific invariants before proceeding:

- **GGUF:** Version check (v1/v2/v3), architecture string recognized, required KV keys present (`block_count`, `embedding_length`, `attention.head_count`)
- **ONNX:** Opset version in supported range (13–21), no unsupported IR version, graph structure is a DAG
- **GGML:** Magic bytes and tensor count match expected layout

### 3. Normalize

Translate format-specific structures into the canonical `AiGraph` IR:

- **Op mapping:** ONNX `op_type` strings → `AiOp` enum; GGUF architecture recognition → explicit transformer block structure
- **Shape inference:** ONNX `value_info` hints → `TensorInfo.shape`; GGUF metadata → concrete dimensions
- **Quantization descriptors:** GGUF quant type → `QuantDescriptor`; ONNX INT8/INT4 → `QuantDescriptor`
- **Metadata extraction:** Context length, rope config, vocab size → `AiGraph.metadata` KV map
- **Tokenizer data:** Vocab, merges, scores extracted and stored for `SECTION_TOKENIZER` (ADR-0012)

### 4. Emit IR

Return a validated `AiGraph` ready for downstream passes:

```rust
pub fn import_gguf(path: &Path, opts: GgufImportOptions) -> Result<AiGraph>
pub fn import_onnx(bytes: &[u8], opts: OnnxImportOptions) -> Result<AiGraph>
pub fn import_ggml(path: &Path, opts: GgmlImportOptions) -> Result<AiGraph>
```

After import, no downstream code knows which format was used. The `AiGraph` carries no format provenance.

---

## ONNX Import (`hologram-ai-onnx`)

### Stage 1: Binary parsing

- Parse protobuf using `prost`-generated bindings from `onnx.proto3`
- No dependency on the ONNX Runtime C library
- Produces a `ModelProto` Rust struct tree

### Stage 2: External data resolution

- ONNX "large model format" stores weight tensors in external files
- `DataResolver` trait handles path-based, URL-based, or in-memory resolution
- Default: filesystem `DataResolver` relative to the `.onnx` file path

```rust
pub trait DataResolver: Send + Sync {
    fn resolve(&self, location: &str, offset: u64, len: u64) -> Result<Bytes>;
}
```

### Stage 3: Graph extraction

- Walk `GraphProto.node` in topological order
- For each `NodeProto`:
  - Look up `op_type` in `op_map` → `AiOp`
  - Extract `attribute` fields → `NodeAttrs`
  - Register input/output tensor names → `TensorId`
  - Emit `AiNode`

### Stage 4: Initializer extraction

- `GraphProto.initializer` → `AiParam::Inline(Bytes)` for small tensors
- Tensors exceeding a size threshold → `AiParam::Lazy(ConstantId)` (deferred via `HoloLoader`)
- `TensorProto.data_type` → `DType`
- Shape from `TensorProto.dims`

### Stage 5: Shape and dtype annotation

- `GraphProto.value_info` provides type/shape hints for intermediate tensors
- `ShapeInferenceGraph` runs forward propagation for missing shape info
- Outputs `TensorInfo` for all known tensors
- ONNX `dim_param` strings mapped to symbolic `DimExpr::Var(id)` and interned in `DimVarTable`
- Missing shapes emit `Shape::Dynamic` and are filled by `ShapePropagation` pass

### Stage 6: `AiGraph` assembly

- Assemble nodes, params, tensor_info, inputs, outputs into `AiGraph`
- Apply basic structural validation (no dangling tensor ids, etc.)

### Op coverage strategy

- Supported ops: all ops required to express BERT, GPT-2, T5, LLaMA ONNX exports
- Opset 13–21 target
- Unsupported ops → `AiOp::Opaque { op_type, raw_attrs }` with warning
- Importer does not fail on unknown ops; lowering fails if opaque nodes remain

### Key ops to support at MVP

```
MatMul, Gemm, Conv (basic), Add, Sub, Mul, Div
Relu, Gelu, Tanh, Sigmoid, Softmax, LogSoftmax
LayerNormalization, BatchNormalization
Gather, GatherElements, ScatterElements
Reshape, Transpose, Concat, Split, Slice, Unsqueeze, Squeeze
Cast, Expand, Tile
ReduceMean, ReduceSum, ReduceMax
Attention (from onnxruntime ops)
```

---

## GGUF Import (`hologram-ai-gguf`)

GGUF is a binary format storing metadata key-values and raw tensor data.
Unlike ONNX, it does **not** store a graph. The graph topology must be
**reconstructed from architecture metadata**.

### Stage 1: Binary parsing

Header:
```
magic:   4 bytes ("GGUF")
version: uint32
n_tensors: uint64
n_kv:    uint64
```

KV metadata block → parsed into `GgufMetadata`:
```rust
pub struct GgufMetadata {
    pub general_architecture: String,   // "llama", "mistral", etc.
    pub general_name: Option<String>,
    pub context_length: Option<u64>,
    pub embedding_length: Option<u64>,
    pub feed_forward_length: Option<u64>,
    pub block_count: Option<u64>,
    pub attention_head_count: Option<u64>,
    pub attention_head_count_kv: Option<u64>,
    pub rope_freq_base: Option<f32>,
    pub vocab_size: Option<u64>,
    // tokenizer data (see ADR-0012)
    pub tokenizer_model: Option<String>,      // "llama", "gpt2", etc.
    pub tokens: Option<Vec<String>>,          // vocabulary tokens
    pub scores: Option<Vec<f32>>,             // unigram scores
    pub token_type: Option<Vec<u32>>,         // token type flags
    pub merges: Option<Vec<String>>,          // BPE merge pairs
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub padding_token_id: Option<u32>,
    pub unknown_token_id: Option<u32>,
    pub add_bos_token: Option<bool>,
    pub add_eos_token: Option<bool>,
    // ... all arch-specific fields
    pub raw: HashMap<String, GgufValue>,
}
```

Tensor info block → `TensorIndex`:
```rust
pub struct TensorIndex {
    entries: Vec<TensorEntry>,
}
pub struct TensorEntry {
    pub name: String,
    pub n_dims: u32,
    pub dims: SmallVec<[u64; 4]>,
    pub ggml_type: GgmlType,    // the GGUF quant type enum
    pub offset: u64,            // byte offset from data start
}
```

### Stage 2: Quantization mapping

Map `GgmlType` → `QuantDescriptor`:

| GgmlType | QuantScheme |
|----------|------------|
| F32 | `None` (F32) |
| F16 | `None` (F16) |
| Q4_0 | `Q4_0` |
| Q4_1 | `Q4_1` |
| Q5_0 | `Q5_0` |
| Q8_0 | `Q8_0` |
| Q2_K | `Q2_K` |
| Q4_K | `Q4_K_M` |
| Q6_K | `Q6_K` |
| IQ4_XS | `IQ4_XS` |
| ... | ... |

All GGUF quant types → `AiParam::Lazy(ConstantId)` — weight bytes are deferred into `ConstantStore` and memory-mapped via `HoloLoader`, not eagerly copied.

Quantization priority: Q4_0 → Q8_0 → Q4_K_M → Q5_K_M → Q6_K.

### Stage 3: Architecture recognition

The `ArchRecognizer` trait matches on `metadata.general_architecture` and
reconstructs the model graph:

```rust
pub trait ArchRecognizer: Send + Sync {
    fn arch_name(&self) -> &str;
    fn matches(&self, meta: &GgufMetadata) -> bool;
    fn build_graph(&self, meta: &GgufMetadata, tensors: &TensorIndex) -> Result<AiGraph>;
}
```

Example: `LlamaArch::build_graph` constructs:
- Token embedding lookup
- N × transformer blocks (attention + FFN + norms)
- Final layer norm
- LM head

Built-in recognizers and the architectures they support:

| Recognizer | Architectures |
|------------|--------------|
| `LlamaArch` | llama, llama2, llama3, codellama |
| `MistralArch` | mistral, mixtral |
| `PhiArch` | phi, phi2 |
| `Phi3Arch` | phi3 |
| `QwenArch` | qwen, qwen2, qwen2_5 |
| `GemmaArch` | gemma, gemma2 |
| `DeepSeekArch` | deepseek |

### Stage 4: Tokenizer data extraction

GGUF metadata contains tokenizer data under `tokenizer.ggml.*` keys. This data
is extracted and stored in `AiGraph::metadata` under the `tokenizer.*` namespace
for downstream use by `hologram-ai-tokenizer` (see ADR-0012).

| GGUF key | AiGraph metadata key |
|----------|---------------------|
| `tokenizer.ggml.model` | `tokenizer.model` |
| `tokenizer.ggml.tokens` | `tokenizer.tokens` |
| `tokenizer.ggml.scores` | `tokenizer.scores` |
| `tokenizer.ggml.token_type` | `tokenizer.token_type` |
| `tokenizer.ggml.merges` | `tokenizer.merges` |
| `tokenizer.ggml.bos_token_id` | `tokenizer.bos_token_id` |
| `tokenizer.ggml.eos_token_id` | `tokenizer.eos_token_id` |
| `tokenizer.ggml.padding_token_id` | `tokenizer.padding_token_id` |
| `tokenizer.ggml.unknown_token_id` | `tokenizer.unknown_token_id` |
| `tokenizer.ggml.add_bos_token` | `tokenizer.add_bos_token` |
| `tokenizer.ggml.add_eos_token` | `tokenizer.add_eos_token` |

If tokenizer metadata is missing (rare for modern GGUF files), no tokenizer
entries are added. The downstream `ModelCompiler` handles the absence gracefully.

### Stage 5: `AiGraph` assembly

- Params from `TensorIndex` bound to graph nodes via name conventions
- Metadata stored in `AiGraph::metadata` for downstream use (context_length, rope config, tokenizer data, etc.)
- `AiOp::RotaryEmbedding`, `AiOp::RmsNorm`, `AiOp::GroupedQueryAttention` used for LLaMA-family
- Large weight tensors accessed via `memmap2` for lazy loading; `AiParam::Lazy` defers reads until lowering

---

## GGML Import (`hologram-ai-ggml`)

GGML is the original pre-GGUF format. Simpler, less extensible.

### Stage 1: Header parsing

```
magic: uint32 (0x67676d6c or 0x67676d66)
vocab_size, embd_size, mult, n_head, n_layer, rot, ftype: int32
```

### Stage 2: Vocabulary parsing

Token strings and scores from the header.

### Stage 3: Tensor parsing

Tensors follow sequentially: n_dims, dim array, name, data.

### Stage 4: Graph construction

Hardcoded topology for supported model families (llama v1 format).
Produces `AiGraph` with same structure as the equivalent GGUF recognizer.

Supported architectures: LlamaV1Arch, FalconArch, BloomArch.

### Strategy note

GGML import is a **migration utility**. The primary ongoing format is GGUF.
After initial support, GGML support is in maintenance mode. Intended for
converting older checkpoints to GGUF or directly to `.holo` archives.

---

## Format Priority

| Priority | Format | Rationale |
|----------|--------|-----------|
| P0 — MVP | GGUF | Covers the active LLM ecosystem; GGUF is the de facto format |
| P1 — Phase 2 | ONNX | Covers encoder models (BERT, ViT), non-LLM inference |
| P2 — Phase 2 | GGML | Legacy migration only; low effort once GGUF is done |

---

## Canonicalization: What Happens at the Boundary

After `import_*()` returns, the `AiGraph` must be:

1. **Topologically valid** — no cycles, all tensor IDs resolve
2. **Type-annotated** — every tensor has a `TensorInfo` with at minimum a `DType`
3. **Shape-partial OK** — some shapes may be `Dim::Dynamic` if not inferrable
4. **Quant-complete** — every quantized param has a `QuantDescriptor`
5. **Format-clean** — no format-specific type leaks into the graph
6. **Tokenizer-portable** — if the source format contains tokenizer data,
   it must be stored in `AiGraph::metadata` under the `tokenizer.*` namespace

The importer is responsible for these invariants. Downstream passes may
strengthen them (e.g. shape propagation fills in `Dim::Dynamic` where possible)
but must not depend on them being stronger than the above.

---

## Error Handling

### Recoverable Errors

- **Unknown op:** Emitted as `AiOp::Opaque { op_type, raw_attrs }`. Import succeeds with a warning; lowering fails if the op is reachable.
- **Missing shape info:** Tensor marked with `Shape::Dynamic`. `ShapePropagation` pass may resolve it; unresolved shapes trigger warnings.
- **Unsupported quant scheme:** Falls back to eager dequantization at import time with a warning.

### Fatal Errors (import fails immediately)

- **Corrupt file:** Invalid magic bytes, truncated header, protobuf decode failure.
- **Unsupported version:** ONNX IR version too new, unknown GGUF version.
- **Missing required data:** GGUF missing `block_count` or `embedding_length`; ONNX missing `graph` field.
- **Unrecognized architecture:** GGUF `general.architecture` key not in recognizer registry (no fallback topology).
- **Cyclic graph:** ONNX graph contains cycles (not a valid DAG).

### Error Types

```rust
pub enum ImportError {
    Io(std::io::Error),
    ParseError { detail: String },
    CorruptFile { reason: String },
    UnsupportedOpset { version: u32 },
    UnsupportedVersion { found: u32, max_supported: u32 },
    UnknownArchitecture { arch: String },
    MissingTensor { name: String },
    MissingMetadata { key: String },
    CorruptData { detail: String },
    CyclicGraph,
    ProtobufDecode(prost::DecodeError),
    // ...
}
```

Import errors are non-recoverable. An importer either produces a valid
`AiGraph` or returns `Err(ImportError)`.

Unsupported ops within an otherwise valid ONNX graph produce `AiOp::Opaque`
entries and a non-fatal `ImportWarning` list attached to the `AiGraph`.

---

## Adding a New Format

1. **Create importer crate:** `crates/hologram-ai-<format>/` with standard layout:
   ```
   crates/hologram-ai-<format>/
   ├── Cargo.toml
   ├── src/
   │   ├── lib.rs         # pub fn import_<format>(...) -> Result<AiGraph>
   │   ├── parser.rs      # format-specific binary/proto parsing
   │   ├── op_map.rs      # format ops → AiOp translation
   │   └── shape_infer.rs # format-specific shape extraction
   ```

2. **Define public interface:** Single entry point returning `AiGraph`:
   ```rust
   pub fn import_<format>(input: ..., opts: <Format>ImportOptions) -> Result<AiGraph, ImportError>
   ```

3. **Implement op mapping:** Map format-specific operations to `AiOp` variants. Use `AiOp::Opaque` for unrecognized ops.

4. **Extract quantization info:** Populate `TensorInfo.storage_dtype` and `TensorInfo.quant` for quantized weights.

5. **Populate metadata:** Fill `AiGraph.metadata` with format-specific config (context length, rope params, etc.).

6. **Add test fixtures:** Commit small test models to `tests/fixtures/<format>/`. Generate with reference tools (ONNX Python, llama.cpp converter).

7. **Wire into facade:** Add `ModelSource::<Format>Path` variant to `hologram-ai` facade crate.

8. **Document in this file:** Add format row to Supported Formats table and Format-Specific Notes section.

Key constraint: All format-specific types remain private to the importer crate (ADR-0003). No format provenance escapes into `AiGraph`.
