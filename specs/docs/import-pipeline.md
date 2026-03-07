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

### Strategy note

GGML import is a **migration utility**. The primary ongoing format is GGUF.
After initial support, GGML support is in maintenance mode.

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

## Error Model

Importers use a structured error type:

```rust
pub enum ImportError {
    Io(io::Error),
    ParseError { detail: String },
    UnsupportedOpset { version: u32 },
    UnknownArchitecture { arch: String },
    MissingTensor { name: String },
    CorruptData { detail: String },
    // ...
}
```

Import errors are non-recoverable. An importer either produces a valid
`AiGraph` or returns `Err(ImportError)`.

Unsupported ops within an otherwise valid ONNX graph produce `AiOp::Opaque`
entries and a non-fatal `ImportWarning` list attached to the `AiGraph`.