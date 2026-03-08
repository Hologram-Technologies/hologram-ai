The two versions of architecture.md are identical. There are no differences between the ARCH VERSION and the SUBPROJECT VERSION - they contain exactly the same content, structure, sections, and text.

Since there's nothing to merge, the output is simply the document as-is:

# hologram-ai: Full Architecture

---

## 1. System Purpose

`hologram-ai` is a **compiler**. Given a foreign AI model artifact (ONNX, GGUF,
GGML), it produces a `.holo` archive. The archive is the complete, self-describing
output: it contains the compiled execution graphs, quantized weights, tokenizer
data, and LLM metadata. It is consumed directly by `hologram`'s standard loader
and executor — no hologram-ai types are required at inference time.

See [ADR-0016](../../adrs/0016-hologram-ai-compiler-only.md) for the decision record.

The canonical pipeline:

```
Foreign model artifact
        │
   ┌────▼─────────────────────────┐
   │   Format Importer            │  hologram-ai-{onnx,gguf,ggml}
   └────┬─────────────────────────┘
        │ AiGraph  ←── canonical AI IR (hologram-ai-common)
   ┌────▼─────────────────────────┐
   │   Optimization Passes        │  hologram-ai-common
   │   (attention/FFN/quant fuse) │  semantic AI passes —
   └────┬─────────────────────────┘  hologram-compiler cannot perform these
        │ AiGraph (optimized)
   ┌────▼─────────────────────────┐
   │   KV-Cache Planner           │  hologram-ai-common
   └────┬─────────────────────────┘  KV sizing + layout descriptor
        │ AiGraph + KvCacheLayout
   ┌────▼─────────────────────────┐
   │   Multi-Graph Lowering       │  hologram-ai-common
   │   prefill graph              │  AiOp → GraphOp
   │   decode graph               │  (separate graphs, same weights)
   └────┬─────────────────────────┘
        │ hologram::Graph × N
   ┌────▼─────────────────────────┐
   │   hologram-compiler          │  (hologram crate, `compiler` feature)
   │   compile(graph) × N         │  LUT fusion, CSE, buffer reuse, schedule
   └────┬─────────────────────────┘
        │ ExecutionSchedule × N
   ┌────▼─────────────────────────┐
   │   Archive Writer             │  hologram::HoloWriter
   │   LayerHeader (0x0002)       │  named layers: "lm.prefill", "lm.decode"
   │   LlmMeta    (0x0011)        │  KvCacheLayout, model type, entrypoint IDs
   │   Tokenizer  (0x1001)        │  vocab, merges (via ADR-0012)
   └────┬─────────────────────────┘
        │ .holo archive
```

At this point hologram-ai's job is done. The archive is executed by the caller
using standard `hologram` APIs (`HoloLoader`, `KvExecutor`) with the generation
loop implemented in application code (≤ 30 lines; see ADR-0016).

---

## 2. System Boundaries

### hologram-ai owns

- All AI model format parsing and interpretation
- Canonical `AiGraph` IR and all semantic optimization passes
- Quantization descriptors and quant-aware lowering
- KV-cache layout *computation* (`MemoryPlanner` reads AI model metadata → produces
  `hologram::KvCacheLayout`; the struct itself is defined in `hologram`)
- Multi-graph lowering: separate subgraphs for prefill and decode (or bucketed
  variants), each emitted as a named layer in the archive `LayerHeader`
- Archive construction: populating `hologram::LayerHeader`,
  `hologram::LlmMetaSection` (0x0011), and `hologram::TokenizerSectionData` (0x1001)
- Validation harness (compiles model, executes via `KvExecutor`, compares to
  reference runtimes — no session type required)

### hologram owns (hologram-ai consumes these)

- `hologram::compile(graph)` — LUT fusion, CSE, buffer reuse → `ExecutionSchedule`
- `Graph` + `GraphOp` — the byte-domain graph IR that lowering emits
- `KvExecutor` — the stateless execution engine
- `CustomOpRegistry` — extension point for custom ops registered during lowering
- `BufferArena` — scratch memory during execution (caller-owned)
- `ConstantStore` / `ConstantId` — weight storage and lazy loading
- `HoloLoader` / `HoloWriter` — archive format (read and write)
- `LayerHeader`, `LayerDescriptor`, `LayerEntrypoint` — named execution entrypoints
- **Archive section types for all well-known sections:**
  - `LlmMetaSection` + `KvCacheLayout` + `LlmModelType` + `DecodeLayers`
    (`SECTION_LLM_META`, 0x0011)
  - `TokenizerSectionData` (`SECTION_TOKENIZER`, 0x1001)
  - `SECTION_LLM_META` + `SECTION_TOKENIZER` constants
- `BucketSelector` — utility for reading `DecodeLayers::Bucketed` and mapping
  actual seq_len to the correct named layer; defined in `hologram` alongside
  `LlmMetaSection`

### hologram-ai does NOT own

- Inference session lifecycle
- KV-cache buffer management at runtime
- Autoregressive generation loop
- Token sampling (greedy, temperature, top-p, top-k)
- Streaming / async token delivery
- Kernel implementations (GEMM, attention, etc.)
- Process/WASM/microVM sandbox isolation
- Network transport

### Tokenizer boundary (ADR-0012)

Native tokenizer implementations (`NativeTokenizer`) live in `hologram-ai-tokenizer`.
Vocabulary and merge data are stored in `hologram::ConstantStore` and embedded in
the archive at section `0x1001`. The tokenizer is recovered by the caller via
`HoloLoader` and used independently of any hologram-ai types.

---

## 3. Named Layer Entrypoints

For autoregressive LLMs, hologram-ai emits two named layers into the archive's
`LayerHeader` section:

### `"lm.prefill"` — Variable-length prompt ingestion

```
inputs:
  input_ids:  [batch, seq_len]  i64   — prompt token IDs
  kv_cache:   [n_bytes]         u8    — KV-cache buffer (initially zeroed)
outputs:
  logits:     [batch, vocab]    f32   — next-token probability logits
  kv_cache:   [n_bytes]         u8    — updated KV-cache (seq_len tokens written)
```

### `"lm.decode"` — Single-token autoregressive step

```
inputs:
  input_ids:   [batch, 1]  i64  — single token ID
  present_len: []          u32  — number of tokens currently in KV-cache
  kv_cache:    [n_bytes]   u8   — KV-cache buffer
outputs:
  logits:      [batch, vocab] f32 — next-token probability logits
  kv_cache:    [n_bytes]      u8  — updated KV-cache (1 new token written)
```

The KV-cache is an ordinary mutable byte buffer. The caller allocates it once
(size from `SECTION_LLM_META`), passes it to prefill, and threads it through
each decode step. `KvExecutor` is stateless — it reads and writes the buffer
as part of normal graph execution.

### Non-LLM models (`"model.forward"`)

ONNX encoder models and other non-autoregressive models emit a single unnamed
layer with standardized port names matching the ONNX graph inputs/outputs.

---

## 4. Canonical Model Representation

`AiGraph` is the canonical AI-specific IR above the raw Hologram graph IR
(see ADR-0002).

Foreign AI model formats carry semantic structure (multi-head attention,
rope embeddings, norm layers, MLP blocks) that is expensive to reconstruct
from raw arithmetic ops. Preserving this structure through the optimization
phase enables high-value fusions (attention fusion, FFN fusion, norm fusion)
before lowering to Hologram primitives. Fusing at the Hologram graph level
would require pattern-matching over much lower-level ops.

`AiGraph` preserves semantic structure until the lowering boundary,
then maps cleanly to `hologram::GraphOp` nodes.

---

## 5. Semantic Structure in the IR

The following structures survive into `AiGraph` before lowering:

| Structure | IR representation |
|-----------|------------------|
| Multi-head attention | `AiOp::MultiHeadAttention` |
| Grouped query attention | `AiOp::GroupedQueryAttention` |
| Flash attention hint | `AiOp::FlashAttentionHint` |
| RMS normalization | `AiOp::RmsNorm` |
| Layer normalization | `AiOp::LayerNorm` |
| SwiGLU / SiLU gate | `AiOp::FusedSwiGLU` (post-fusion) |
| Rotary embeddings | `AiOp::RotaryEmbedding` |
| Embedding lookup | `AiOp::Embed` |
| Causal attention mask | represented as `AiOp::CausalMask` |

These high-level ops allow the lowering pass to select optimal Hologram kernel
bindings (e.g. fused MHA, flash attention if supported).

---

## 6. Quantization

Quantization is first-class throughout the pipeline.

**Logical dtype** vs **storage dtype** are distinct in `TensorInfo`:

```rust
pub struct TensorInfo {
    pub logical_dtype: DType,   // F32 — what arithmetic sees it as
    pub storage_dtype: DType,   // Q4_0 — how bits are stored
    pub quant: QuantDescriptor, // scale/zp/block metadata
    pub shape: Shape,
}
```

Dequantization is **explicit in the IR** as `AiOp::Dequantize`. The
`hologram-ai-common` opt pass may fuse `Dequantize → MatMul` into `AiOp::QuantizedMatMul`
when a backend supports the fused kernel.

This keeps the IR honest and lets backends declare their quant kernel
capabilities rather than assuming them.

---

## 7. Shape and DType Propagation

See [symbolic-shapes.md](symbolic-shapes.md) for the full specification and
[ADR-0015](../../adrs/0015-hologram-ai-symbolic-shapes.md) for the decision record.

### Symbolic Dimension Expressions

Tensor dimensions are represented as `DimExpr` — a symbolic expression type supporting
arithmetic (`Add`, `Sub`, `Mul`, `Div`, `Mod`), `CeilDiv` (padding/tiling), `Max`
(broadcast), and `Min` (clamp). Variables are interned via `DimVarId` into a per-graph
`DimVarTable`.

```rust
pub enum DimExpr {
    Concrete(u64),
    Var(DimVarId),
    Add(Box<DimExpr>, Box<DimExpr>),
    Sub(Box<DimExpr>, Box<DimExpr>),
    Mul(Box<DimExpr>, Box<DimExpr>),
    Div(Box<DimExpr>, Box<DimExpr>),
    Mod(Box<DimExpr>, Box<DimExpr>),
    CeilDiv(Box<DimExpr>, Box<DimExpr>),
    Max(Box<DimExpr>, Box<DimExpr>),
    Min(Box<DimExpr>, Box<DimExpr>),
    Dynamic,
}

pub type Shape = SmallVec<[DimExpr; 4]>;
```

### Dimension Variable Registry

`DimVarTable` tracks all dimension variables with optional bounds. Importers intern
variables at import time; bounds are tightened via intersection when the same variable
is encountered from multiple sources. Canonical names: `batch`, `seq_len`, `vocab_size`,
`hidden_dim`, `num_heads`, `num_kv_heads`, `head_dim`, `ffn_dim`.

### Shape Propagation

`ShapePropagation` runs as a required optimization pass before lowering. It walks the
graph in topological order, calling per-op inference rules that produce output shapes
and shape constraints. Three-tier resolution:

1. **Immediate error:** Both sides concrete and unequal → error.
2. **Immediate fix:** One side concrete, other is bare variable → fix the variable.
3. **Deferred constraint:** Both sides symbolic → record in `ConstraintStore`, validate
   at concretization time.

### Shape Concretization and Lowering Strategies

Before lowering, all symbolic dimensions must be concretized. Each call to `lower()`
takes a `ShapeStrategy`:

```rust
pub enum ShapeStrategy {
    FixToMax,                    // fix all dims to upper bound → 1 graph emitted
    Bucketed(BucketConfig),      // N concrete variants → N named layers emitted
    Profiles(Vec<ShapeProfile>), // explicit assignments → N named layers emitted
}
```

For `Bucketed`, hologram-ai emits multiple `("lm.decode.128", "lm.decode.512", ...)`
layers in the `LayerHeader`. The `SECTION_LLM_META` records the bucket sizes so the
caller can select the right entrypoint by seq_len at runtime.

### DType Propagation

- `Dequantize` outputs widen to the widest dtype operand needs (usually f32 or f16)
- `Cast` ops are inserted by the lowering pass where dtypes mismatch
- The planner annotates each node's input/output dtypes before lowering

---

## 8. Archive Sections

hologram-ai *populates* the following sections in the emitted `.holo` archive.
All section type definitions and section ID constants live in `hologram`;
hologram-ai constructs the structs and passes them to `HoloWriter`.

| Section ID | Name | Struct (defined in `hologram`) | hologram-ai's role |
|------------|------|--------------------------------|--------------------|
| `0x0002` | `SECTION_LAYER_HEADER` | `LayerHeader` + `LayerDescriptor` | Populate from `LoweringOutput.tensor_ports` |
| `0x0011` | `SECTION_LLM_META` | `LlmMetaSection` | Compute `KvCacheLayout` via `MemoryPlanner`, fill layer IDs |
| `0x1001` | `SECTION_TOKENIZER` | `TokenizerSectionData` | `hologram-ai-tokenizer` packs vocab/merges into the struct |

`LlmMetaSection` is defined in `hologram` (added with ADR-0016):

```rust
// In hologram crate — hologram-ai consumes this type, does not define it
pub struct LlmMetaSection {
    pub model_type: LlmModelType,      // LlamaFamily, Bert, Gpt2, ...
    pub kv_layout: KvCacheLayout,      // total_bytes, n_layers, head_dim, max_seq_len
    pub prefill_layer: LayerId,        // ID of the "lm.prefill" layer
    pub decode_layers: DecodeLayers,   // Single(LayerId) or Bucketed(Vec<(u64, LayerId)>)
}

// Also in hologram — caller-side utility for bucketed layer selection
pub struct BucketSelector { /* ... */ }
impl BucketSelector {
    pub fn from_meta(meta: &LlmMetaSection) -> Option<Self>
    pub fn select(&self, actual_len: u64) -> Option<LayerId>
}
```

---

## 9. Backend Matrix

### MVP backend

**CPU only** — `hologram-exec` (`KvExecutor`) provides the execution engine. All
AI-specific ops are registered as `CustomOpRegistry` handlers during lowering.

### Phase 2 backends

SIMD-accelerated custom handlers and Metal-accelerated LUT computation within the
existing `KvExecutor` model.

### Backend portability

Execution always goes through `KvExecutor`. Backend-specific capability is declared
via `CustomOpRegistry`. No separate per-backend crate.

---

## 10. Portability

| Target | Priority | Notes |
|--------|----------|-------|
| `aarch64-apple-darwin` (M-series) | P0 | primary dev hardware |
| `x86_64-unknown-linux-gnu` | P0 | CI and server targets |
| `x86_64-apple-darwin` | P1 | Intel Mac |
| `x86_64-pc-windows-msvc` | P2 | Windows server |
| `wasm32-wasi` | P3 | no SIMD; import + lower pipeline only |

---

## 11. Dataflow Summary

```
                     ┌──────────────────┐
                     │  Model artifact  │
                     │ (.onnx/.gguf/    │
                     │  .bin)          │
                     └────────┬─────────┘
                              │
                   ┌──────────▼──────────┐
                   │  Format Importer    │  hologram-ai-{onnx,gguf,ggml}
                   └──────────┬──────────┘
                              │ AiGraph (+ tokenizer data in metadata)
                   ┌──────────▼──────────┐
                   │  Optimization       │  hologram-ai-common
                   │  Passes             │  (fusion, folding, shape prop)
                   └──────────┬──────────┘
                              │ AiGraph (optimized)
                   ┌──────────▼──────────┐
                   │  KV-Cache Planner   │  hologram-ai-common
                   └──────────┬──────────┘
                              │ KvCacheLayout
                   ┌──────────▼──────────┐
                   │  Multi-Graph        │  hologram-ai-common
                   │  Lowering           │  ShapeStrategy → 1..N graphs
                   │  (prefill + decode) │
                   └──────────┬──────────┘
                              │ hologram::Graph × N
                   ┌──────────▼──────────┐
                   │  hologram-compiler  │  hologram (compiler feature)
                   │  compile(graph) × N │  LUT fusion, CSE, buf reuse
                   └──────────┬──────────┘
                              │ ExecutionSchedule × N
                   ┌──────────▼──────────┐
                   │  HoloWriter         │  hologram archive writer
                   │  LayerHeader 0x0002 │  named layers + tensor ports
                   │  LlmMeta    0x0011  │  KV layout, bucket config
                   │  Tokenizer  0x1001  │  vocab, merges
                   └──────────┬──────────┘
                              │
                     ┌────────▼────────┐
                     │   .holo archive │  ← hologram-ai's output
                     └────────┬────────┘
                              │
                   (caller uses HoloLoader
                    + KvExecutor directly)
```

---

## 12. Crate Layout

### Workspace Structure

```
hologram-ai/
├── Cargo.toml                     # workspace root
├── CLAUDE.md                      # agent instructions for this repo
├── README.md
├── crates/
│   ├── hologram-ai-quant/         # quantization schemes, block layouts, dequant
│   ├── hologram-ai-common/        # IR types, opt passes, mem planner, lowering
│   ├── hologram-ai-tokenizer/     # tokenizer (BPE, SentencePiece, WordPiece)
│   ├── hologram-ai-onnx/          # ONNX importer
│   ├── hologram-ai-gguf/          # GGUF importer
│   ├── hologram-ai-ggml/          # GGML checkpoint importer
│   └── hologram-ai/               # public facade: compile, inspect, validate, CLI
├── tests/
│   ├── fixtures/
│   │   ├── onnx/                  # committed ONNX test models (tiny/synthetic)
│   │   ├── gguf/                  # committed GGUF test models (tiny/synthetic)
│   │   └── golden/                # golden tensor ouputs for regression
│   └── integration/               # cross-crate integration tests
└── scripts/
    ├── download-test-models.sh    # optional: fetch larger models for full tests
    └── gen-fixtures.py            # generate synthetic test fixtures
```

### Crate Responsibilities

#### `hologram-ai-quant`

Foundational quantization library. No AI IR types — pure quant primitives.

- `QuantScheme` — all supported quantization schemes
- `QuantDescriptor` — per-tensor quantization metadata
- `Q4_0Block`, `Q8_0Block`, etc. — GGML/GGUF-compatible block structs
- `dequant_tensor()` / `quant_tensor()` — software quant for CPU and fixtures

Block layouts must match `ggml-quants.h` exactly.

**Depends on** `half`, `smallvec`.

---

#### `hologram-ai-common`

The compiler core. All importers and the facade depend on it.

**IR types:** `AiGraph`, `AiNode`, `AiOp`, `AiParam`, `TensorInfo`, `DimExpr`,
`DimVarId`, `DimVarTable`, `Shape`, `DType`, `NodeId`, `TensorId`

**Optimization passes:** `OptPipeline`, `ConstantFolding`, `DeadNodeElimination`,
`ShapePropagation`, `AttentionFusion`, `FfnFusion`, `QuantMatMulFusion`

**Shape system:** `DimExpr`, `DimVarId`, `DimVarTable`, `ShapeConstraint`,
`ConstraintStore`, `ShapeError`, `ShapeStrategy`, `BucketConfig`, `canonical_vars`

**Memory planner:** `MemoryPlanner` — reads AI model metadata, produces
`hologram::KvCacheLayout` (struct defined in `hologram`, not here)

**Lowering:**
```rust
pub struct LoweringOutput {
    pub graph: hologram::Graph,
    pub registry: hologram::CustomOpRegistry,
    pub layer_name: String,               // e.g. "lm.prefill", "lm.decode", "lm.decode.128"
    pub layer_descriptor: hologram::LayerDescriptor,  // tensor ports — hologram type
}

pub fn lower(
    graph: &AiGraph,
    kv_layout: &hologram::KvCacheLayout,  // hologram type
    phase: LowerPhase,                    // Prefill | Decode | DecodeBucket(seq_len)
    opts: &LoweringOptions,
) -> Result<LoweringOutput>
```

**Depends on** `hologram-ai-quant`, `hologram` (root crate, no `compiler` feature).

---

#### `hologram-ai-tokenizer`

Native tokenizer implementations. See [tokenizer.md](tokenizer.md) and ADR-0012.

- `Tokenizer` trait, `NativeTokenizer` (BPE, SentencePiece, WordPiece)
- Populates `hologram::TokenizerSectionData` (struct defined in `hologram`)
- `ConstantStore` pack/unpack helpers for vocab/merges/scores

**Depends on** `hologram-ai-common`, `hologram` (root crate).

---

#### `hologram-ai-onnx`

```rust
pub fn import_onnx(bytes: &[u8], opts: OnnxImportOptions) -> Result<AiGraph>
pub fn import_onnx_path(path: &Path, opts: OnnxImportOptions) -> Result<AiGraph>
```

Supports ONNX opset 13–21. **Depends on** `hologram-ai-common`, `bytes`, `prost`.

---

#### `hologram-ai-gguf`

```rust
pub fn import_gguf(path: &Path, opts: GgufImportOptions) -> Result<AiGraph>
pub fn import_gguf_bytes(bytes: &[u8], opts: GgufImportOptions) -> Result<AiGraph>
```

Supports GGUF v1/v2/v3. Built-in arch recognizers: `LlamaArch`, `MistralArch`,
`PhiArch`, `Phi3Arch`, `QwenArch`, `Qwen2Arch`, `GemmaArch`, `Gemma2Arch`,
`MixtralArch`, `DeepSeekArch`.

**Depends on** `hologram-ai-common`, `hologram-ai-quant`, `bytes`, `memmap2`.

---

#### `hologram-ai-ggml`

```rust
pub fn import_ggml(path: &Path, opts: GgmlImportOptions) -> Result<AiGraph>
```

GGML v1 checkpoint importer (legacy pre-GGUF format). Arch recognizers: `LlamaV1Arch`,
`FalconArch`, `BloomArch`.

**Depends on** `hologram-ai-common`, `hologram-ai-quant`, `bytes`.

---

#### `hologram-ai` (public facade)

The single public entry point for compilation. Consumers need only this crate.

```rust
pub struct ModelCompiler {
    pub strategy: ShapeStrategy,
    pub tokenizer_opts: TokenizerOptions,
    pub compile_opts: CompileOptions,
}

impl ModelCompiler {
    pub fn compile(source: ModelSource) -> Result<HoloArchive>
}

pub enum ModelSource {
    OnnxBytes(Bytes), OnnxPath(PathBuf),
    GgufPath(PathBuf), GgmlPath(PathBuf),
    AiGraph(AiGraph),
}

pub struct HoloArchive {
    pub bytes: Vec<u8>,     // ready to write with std::fs::write()
    pub stats: CompileStats,
}
```

**CLI commands:**
```
hologram-ai compile  <model>        -o model.holo [--strategy bucketed --buckets 128,512,1024]
hologram-ai inspect  <archive.holo>
hologram-ai validate <model>        [--ort-path ...] [--llamacpp-path ...]
hologram-ai download <repo/model>   [--format gguf|onnx] [-o model.holo]
```

Execution of compiled archives is handled by `hologram`'s own CLI:
```
hologram run <archive.holo> "<prompt>"
```
`hologram run` reads `SECTION_LLM_META` for layer entrypoints and KV-cache layout,
reads `SECTION_TOKENIZER` for encode/decode, and drives `KvExecutor` directly.
No hologram-ai dependency is required at runtime — the archive is self-describing.

**Cargo.toml** (facade only):
```toml
[dependencies]
hologram = { path = "../../hologram", features = ["compiler"] }
```

**Depends on** all internal crates + `hologram` root with `compiler` feature, `clap`.

---

### Crate Dependency Matrix

```
hologram-ai-quant      → (no internal deps)
hologram-ai-common     → hologram-ai-quant, hologram (root crate)
hologram-ai-tokenizer  → hologram-ai-common, hologram (root crate)
hologram-ai-onnx       → hologram-ai-common
hologram-ai-gguf       → hologram-ai-common, hologram-ai-quant
hologram-ai-ggml       → hologram-ai-common, hologram-ai-quant
hologram-ai            → hologram-ai-common, hologram-ai-quant,
                         hologram-ai-tokenizer,
                         hologram-ai-onnx, hologram-ai-gguf, hologram-ai-ggml,
                         hologram (root crate + compiler feature)
```

No crate in the hologram-ai workspace imports hologram subcrates directly.
All hologram types are accessed via the root `hologram` crate.
