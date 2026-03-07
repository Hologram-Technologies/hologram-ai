# hologram-ai: Full Architecture

---

## 1. System Purpose

`hologram-ai` is a compiler and runtime integration layer. Given a foreign AI
model artifact, it produces an executable `hologram::ExecutionSchedule` and manages
the inference session lifecycle on top of `hologram::KvExecutor`.

The canonical internal flow:

```
Foreign model artifact
        │
   ┌────▼─────────────────────────┐
   │   Format Importer            │  hologram-ai-{onnx,gguf,ggml}
   └────┬─────────────────────────┘
        │ AiGraph  ←── canonical AI IR (hologram-ai-common)
   ┌────▼─────────────────────────┐
   │   Optimization Passes        │  hologram-ai-common
   │   (attention/FFN/quant fuse) │  semantic AI passes — hologram-compiler
   └────┬─────────────────────────┘  cannot perform these
        │ AiGraph (optimized)
   ┌────▼─────────────────────────┐
   │   KV-Cache Planner           │  hologram-ai-common
   └────┬─────────────────────────┘  KV sizing only
        │ KvCacheLayout
   ┌────▼─────────────────────────┐
   │   Lowering                   │  hologram-ai-common
   └────┬─────────────────────────┘  AiOp → GraphOp
        │ hologram::Graph
   ┌────▼─────────────────────────┐
   │   hologram-compiler          │  (hologram crate, `compiler` feature)
   │   compile(graph)             │  LUT fusion, CSE, buffer reuse, schedule
   └────┬─────────────────────────┘
        │ ExecutionSchedule
   ┌────▼─────────────────────────┐
   │   Inference Session          │  hologram-ai
   │   + KV-Cache                 │
   └────┬─────────────────────────┘
        │ Token / Tensor output
   ┌────▼─────────────────────────┐
   │   Streaming Decoder          │  hologram-ai
   └──────────────────────────────┘
```

---

## 2. System Boundaries

### hologram-ai owns

- All AI model format parsing and interpretation
- Canonical `AiGraph` IR and all semantic optimization passes on it
- Quantization descriptors and quant-aware lowering
- KV-cache layout and sizing (`KvCacheLayout`)
- Lowering from `AiGraph` to `hologram::Graph`
- Inference session lifecycle (`InferenceSession`)
- KV-cache buffer allocation and update logic
- Autoregressive token generation loop
- Streaming output interface
- Validation harness against reference runtimes

### hologram owns (hologram-ai consumes these)

- `hologram::compile(graph)` — LUT fusion, CSE, buffer reuse → `ExecutionSchedule`
- `Graph` + `GraphOp` — the byte-domain graph IR that lowering emits
- `KvExecutor` — the stateless execution engine
- `CustomOpRegistry` — extension point for AI-specific operations
- `BufferArena` — per-session buffer management during execution
- `ConstantStore` / `ConstantId` — weight storage and lazy loading
- `HoloLoader` / `HoloWriter` — archive format for serialized models

### hologram-ai owns (continued)

- Tokenizer implementations: native BPE/SentencePiece/WordPiece via
  `hologram-ai-tokenizer` crate, using `ConstantStore` for vocab data and
  `.holo` custom sections for archive embedding (see ADR-0012).
  External implementations still accepted via `Box<dyn Tokenizer>`.

### hologram-ai does NOT own

- Actual kernel implementations (GEMM, attention, etc.)
- Process/WASM/microVM sandbox isolation
- Network transport

---

## 3. Major Layers

### Layer 1: Format Importers

Three importers, one per format. Each is a standalone crate.

| Crate | Input | Output |
|-------|-------|--------|
| `hologram-ai-onnx` | ONNX protobuf bytes | `AiGraph` |
| `hologram-ai-gguf` | GGUF v1/v2/v3 file | `AiGraph` |
| `hologram-ai-ggml` | GGML checkpoint file | `AiGraph` |

**Key constraint:** Format-specific logic must not escape the importer boundary.
After `import_*()` returns an `AiGraph`, no downstream code knows or cares
which format the model came from.

### Layer 2: Canonical AI IR (`hologram-ai-common`)

`AiGraph` is the single representation all importers target and all downstream
passes consume. It is a typed DAG of `AiNode` operations over `AiTensor` values.

Quantization descriptors (`QuantDescriptor` from `hologram-ai-quant`) are
embedded in `TensorInfo`. They are never stripped.

See [lowering.md](lowering.md) for the IR specification.

### Layer 3: Optimization Passes (`hologram-ai-common`)

Pure graph-to-graph transformations. Each pass is a stateless function:
`fn pass(graph: AiGraph) -> Result<AiGraph>`.

Passes operate on the semantic `AiGraph` level, not on `hologram` graph nodes.

### Layer 4: KV-Cache Planner (`hologram-ai-common`)

Takes an optimized `AiGraph` and produces a `KvCacheLayout`:
- KV-cache layer count, head dimensions, max seq len
- Per-session buffer size estimation
- Dtype for KV storage (f16 recommended)

Intermediate activation buffer reuse is **not** planned here —
`hologram::compile()` handles that via liveness analysis and workspace
slot reuse (first-fit-decreasing bin packing).

### Layer 5: Lowering (`hologram-ai-common`)

Maps `AiGraph + KvCacheLayout` to `hologram::Graph`.
This is the boundary between AI-semantic code and Hologram-native code.

`lower()` does NOT produce an `ExecutionSchedule`. After lowering, call
`hologram::compile(graph)` (Layer 5.5) to get the schedule.

### Layer 5.5: hologram-compiler (via `hologram` crate, `compiler` feature)

```rust
let compilation = hologram::compile(lower_output.graph)?;
// compilation.schedule: ExecutionSchedule
// compilation.archive:  Vec<u8>  (serialized, for caching)
// compilation.stats:    CompilationStats
```

Passes applied: constant folding → LUT chain fusion → CSE → liveness analysis
→ workspace slot reuse. `hologram-compiler` has no concept of AiOp and cannot
perform AI-semantic fusions. These two layers are complementary (see ADR-0008).

### Layer 6: Inference Session (`hologram-ai`)

Manages:
- Compiled `ExecutionSchedule` (shared, read-only; from `hologram::compile()`)
- `KvExecutor` reference (shared, stateless)
- `CustomOpRegistry` (shared, registered once at compile time)
- KV-cache `BufferArena` (per-session)
- Single-pass `run()` and multi-step `generate()` APIs

### Layer 6.5: Tokenizer (`hologram-ai-tokenizer`)

Native tokenizer operating at the session boundary (text ↔ token IDs).
Not part of the computation graph — tokenization does not benefit from LUT
dispatch, CSE, or graph scheduling (see ADR-0012).

- `NativeTokenizer` implements BPE (Phase 2), SentencePiece, WordPiece (Phase 3)
- Vocab tables and merge rules stored in `hologram::ConstantStore`
- Tokenizer metadata serialized in `.holo` archives via `SECTION_TOKENIZER` (0x1001)
- `CompiledModel::tokenizer()` returns the embedded tokenizer if available
- External `Box<dyn Tokenizer>` still accepted for custom implementations

### Layer 7: Streaming Decoder (`hologram-ai`)

Wraps `InferenceSession` in an autoregressive loop. Implements `Stream<Item = Token>`.
Uses `CompiledModel::tokenizer()` by default for text ↔ token conversion.

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

Shape propagation runs as a required optimization pass before lowering.

Symbolic dimensions (`batch_size`, `seq_len`) are preserved through the graph.
Concrete dimensions are folded to constants.

Shape inference failures at import time produce `Dim::Dynamic` annotations.
The lowering pass must handle dynamic shapes via runtime dispatch.

DType propagation:
- `Dequantize` outputs widen to the widest dtype operand needs (usually f32 or f16)
- `Cast` ops are inserted by the lowering pass where dtypes mismatch
- The planner annotates each node's input/output dtypes before lowering

---

## 8. Backend Matrix

### MVP backend

**CPU only** — `hologram-exec` (`KvExecutor` + `KvStore`) provides the execution
engine. All AI-specific operations are registered as `CustomOpRegistry` handlers.

Rationale: maximizes portability for validation, avoids GPU toolchain overhead
during the compiler pipeline bring-up phase, covers all test platforms.

### Phase 2 backends

The hologram project's O(1) LUT model runs on any platform. Phase 2 work focuses
on SIMD-accelerated custom handlers and potentially Metal-accelerated LUT computation
within the existing `KvExecutor` model.

### Backend portability

Execution always goes through `KvExecutor`. AI-specific operation support is
controlled by the `CustomOpRegistry` registered at session construction. There is
no separate per-backend crate (`hologram-cpu`, `hologram-metal`, etc.).

---

## 9. Portability

| Target | Priority | Notes |
|--------|----------|-------|
| `aarch64-apple-darwin` (M-series) | P0 | primary dev hardware |
| `x86_64-unknown-linux-gnu` | P0 | CI and server targets |
| `x86_64-apple-darwin` | P1 | Intel Mac |
| `x86_64-pc-windows-msvc` | P2 | Windows server |
| `wasm32-wasi` | P3 | no SIMD-heavy backends; pure IR + lowering only |

---

## 10. Dataflow Summary

```
                     ┌──────────────────┐
                     │  Model artifact  │
                     │  (.onnx / .gguf  │
                     │   / .bin)        │
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
              ┌───────────────┼───────────────┐
              │               │               │
     ┌────────▼─────┐  ┌──────▼─────┐  ┌─────▼──────┐
     │ hologram-ai- │  │ hologram-ai│  │(shape/dtype│
     │ quant        │  │ -common    │  │  validated)│
     │ (quant descs)│  │ (mem plan) │  └────────────┘
     └──────────────┘  └──────┬─────┘
                              │ AiGraph + KvCacheLayout
                   ┌──────────▼──────────┐
                   │  Lowering           │  hologram-ai-common
                   └──────────┬──────────┘
                              │ hologram::Graph
                   ┌──────────▼──────────┐
                   │  hologram-compiler  │  hologram (compiler feature)
                   │  compile(graph)     │  LUT fusion, CSE, buf reuse
                   └──────────┬──────────┘
                              │ ExecutionSchedule
                   ┌──────────▼──────────┐
                   │  .holo Archive      │  model weights + tokenizer
                   │  (SECTION_TOKENIZER │  vocab/merges in ConstantStore
                   │   = 0x1001)         │
                   └──────────┬──────────┘
                              │
                   ┌──────────▼──────────┐
                   │  Inference Session  │  hologram-ai
                   │  + KV-Cache         │
                   └──────────┬──────────┘
                              │
                   ┌──────────▼──────────┐
                   │  NativeTokenizer    │  hologram-ai-tokenizer
                   │  (text ↔ token IDs) │  loaded from ConstantStore
                   └──────────┬──────────┘
                              │
                   ┌──────────▼──────────┐
                   │  Streaming Decoder  │  hologram-ai
                   └──────────┬──────────┘
                              │ Token stream
                   ┌──────────▼──────────┐
                   │  Application        │
                   └─────────────────────┘
```

---

## 11. Crate Layout

### Workspace Structure

```
hologram-ai/
├── Cargo.toml                     # workspace root
├── CLAUDE.md                      # agent instructions for this repo
├── README.md
├── crates/
│   ├── hologram-ai-quant/         # quantization schemes, block layouts, dequant
│   ├── hologram-ai-common/        # IR types, opt passes, mem planner, lowering
│   ├── hologram-ai-tokenizer/      # native tokenizer (BPE, SentencePiece, WordPiece)
│   ├── hologram-ai-onnx/          # ONNX importer
│   ├── hologram-ai-gguf/          # GGUF importer
│   ├── hologram-ai-ggml/          # GGML checkpoint importer
│   └── hologram-ai/               # session, stream, validate, CLI, public facade
├── tests/
│   ├── fixtures/
│   │   ├── onnx/                  # committed ONNX test models (tiny/synthetic)
│   │   ├── gguf/                  # committed GGUF test models (tiny/synthetic)
│   │   └── golden/                # golden tensor outputs for regression
│   └── integration/               # cross-crate integration tests
└── scripts/
    ├── download-test-models.sh    # optional: fetch larger models for full tests
    └── gen-fixtures.py            # generate synthetic test fixtures
```

### Crate Responsibilities

#### `hologram-ai-quant`

Foundational quantization library. No AI IR types — pure quant primitives.

Defines:
- `QuantScheme` — all supported quantization schemes (Q4_0, Q4_1, Q5_0, Q8_0, Q2_K, Q4_K, Q6_K, IQ4_XS, …)
- `QuantDescriptor` — per-tensor quantization metadata (scheme, block size, scale/zp layout)
- `Q4_0Block`, `Q8_0Block`, etc. — raw block structs matching GGML/GGUF memory layout exactly
- `dequant_tensor()` — software dequantization for CPU fallback
- `quant_tensor()` — software quantization for test fixture generation

Block layouts must match `ggml-quants.h` exactly (validated against llama.cpp).

**Depends on** `half`, `smallvec`.

---

#### `hologram-ai-common`

The compiler core. All importers and the main crate depend on it.

**IR types:** `AiGraph`, `AiNode`, `AiOp`, `AiParam`, `TensorInfo`, `Shape`, `DType`, `NodeId`, `TensorId`

**Optimization passes:** `OptPipeline`, `ConstantFolding`, `DeadNodeElimination`, `ShapePropagation`, `AttentionFusion`, `FfnFusion`, `QuantMatMulFusion`

**Memory planner:** `MemoryPlanner` + `MemoryPlan` — KV-cache sizing only

**Lowering:**
```rust
pub struct LoweringOutput {
    pub graph: hologram::Graph,
    pub registry: hologram::CustomOpRegistry,
    // ExecutionSchedule is NOT produced here — call hologram::compile() after lowering
}

pub fn lower(
    graph: &AiGraph,
    kv_layout: &KvCacheLayout,
    opts: &LoweringOptions,
) -> Result<LoweringOutput>
```

**Depends on** `hologram-ai-quant`, `hologram` (root crate, no `compiler` feature).

---

#### `hologram-ai-tokenizer`

Native tokenizer implementations using hologram primitives. See [tokenizer.md](tokenizer.md)
and [ADR-0012](../../adrs/0012-hologram-native-tokenizer.md).

Defines:
- `Tokenizer` trait (expanded: `encode`, `decode`, `eos_token_id`, `bos_token_id`, `vocab_size`, `id_to_token`, `token_to_id`)
- `NativeTokenizer` — BPE, SentencePiece, WordPiece implementations
- `VocabTable`, `MergeRules`, `UnigramModel` — tokenizer data model
- `TokenizerSectionData` — `.holo` section `SECTION_TOKENIZER` (0x1001) serialization
- `ConstantStore` pack/unpack helpers for vocab/merges/scores

Vocab and merge data stored in `hologram::ConstantStore`. Tokenizer metadata
serialized in `.holo` archive custom sections. No external C dependencies.

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

Supports GGUF v1/v2/v3. Built-in architecture recognizers: `LlamaArch`, `MistralArch`, `PhiArch`, `QwenArch`, `GemmaArch`, `Phi3Arch`.

**Depends on** `hologram-ai-common`, `hologram-ai-quant`, `bytes`, `memmap2`.

---

#### `hologram-ai-ggml`

```rust
pub fn import_ggml(path: &Path, opts: GgmlImportOptions) -> Result<AiGraph>
```

GGML v1 checkpoint importer (pre-GGUF legacy format). **Depends on** `hologram-ai-common`, `hologram-ai-quant`, `bytes`.

---

#### `hologram-ai` (public facade)

The single public entry point. Consumers only need this crate.

```rust
pub struct ModelCompiler { ... }
impl ModelCompiler {
    pub fn compile(source: ModelSource, opts: CompileOptions) -> Result<CompiledModel>
}

pub enum ModelSource {
    OnnxBytes(Bytes), OnnxPath(PathBuf), GgufPath(PathBuf), GgmlPath(PathBuf), AiGraph(AiGraph),
}
```

**CLI commands:**
```
hologram-ai inspect <model>
hologram-ai run <model> --input ...
hologram-ai generate <model> <prompt>
hologram-ai validate <model>
hologram-ai lower <model> --emit-graph
```

**Cargo.toml** (facade only):
```toml
[dependencies]
hologram = { path = "../../hologram", features = ["compiler"] }
```

**Depends on** all internal crates + `hologram` root with `compiler` feature, `futures`, `clap`.

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
                         hologram (root crate)
```

No crate in the hologram-ai workspace imports hologram subcrates directly
(`hologram-graph`, `hologram-exec`, `hologram-archive`). All hologram types
are accessed via the root `hologram` crate.

---

### Naming Rationale

Seven crates. `hologram-ai-quant` is the foundational primitive layer (no IR
dependency). `hologram-ai-common` is the compiler core. `hologram-ai-tokenizer`
provides native tokenization using hologram primitives (see ADR-0012). None of
these are published as a stable API — only `hologram-ai` (the facade) is the
stable public surface. Format importers are separate crates so consumers can opt
in to only the formats they need.
