# Cross-Repo Change Request: hologram base crate

**Status:** Requested
**Date:** 2026-03-07
**Target repo:** hologram (base crate)
**Requesting repo:** hologram-ai

---

## Context

hologram-ai is a **compiler only** (ADR-0016). It parses foreign AI model formats,
runs AI-specific optimizations, lowers to `hologram::Graph`, and writes `.holo`
archives. It should contain **zero runtime code** — no custom op handlers, no
kernel implementations.

Currently, hologram's `GraphOp` is purely byte-domain (Z/256Z ring arithmetic via
`PrimOp`, 256-entry LUT tables via `LutOp`). The only float-aware ops are
`MatMulLut4`/`MatMulLut8`. This forces hologram-ai to register ~60 f32 custom
handlers (`CustomHandler` closures) for basic operations like f32 add, matmul,
softmax, attention, norms, etc. These handlers execute at runtime inside
`KvExecutor`, making hologram-ai a kernel library — not just a compiler.

**Goal:** hologram base crate adds native f32/f16 tensor ops so hologram-ai can
lower `AiGraph` → `hologram::Graph` using native ops and ship no runtime code.

---

## 1. Native Float Tensor Ops (Priority: Critical — Blocking)

hologram needs `GraphOp` variants (or a new `FloatOp` sub-enum) for typed
tensor operations. These would carry shape/dtype metadata and execute on
f32/f16 buffers natively.

### Minimum op set for MVP (LLaMA inference)

**Arithmetic (with broadcast):**
- `FloatAdd`, `FloatSub`, `FloatMul`, `FloatDiv`

**Linear algebra:**
- `FloatMatMul` (general f32 matmul, not just LUT-quantized)
- `FloatBatchMatMul`

**Activations:**
- `FloatSilu`, `FloatGelu`, `FloatRelu`, `FloatSoftmax { axis }`

**Normalization:**
- `FloatRmsNorm { epsilon }`, `FloatLayerNorm { axis, epsilon }`

**Reductions:**
- `FloatReduceSum { axes, keepdims }`, `FloatReduceMean { axes, keepdims }`
- `FloatReduceMax { axes, keepdims }`

**Attention:**
- `FloatMultiHeadAttention { num_heads, head_dim, scale, causal }`
- `FloatGroupedQueryAttention { num_heads, num_kv_heads, head_dim, scale, causal }`

**Other:**
- `FloatEmbed` — embedding lookup (indices → weight rows)
- `FloatRotaryEmbedding { base, dim }` — RoPE positional encoding
- `FloatCast { from, to }` — dtype conversion
- `FloatReshape`, `FloatTranspose { perm }`, `FloatConcat { axis }`
- `FloatGather { axis }` — indexed selection
- `FloatFusedSwiGLU` — SiLU gating (common in LLaMA FFN)
- `FloatDequantize` — Q4_0/Q8_0 → f32 expansion

### Design options

**A) Extend `GraphOp` with a `Float(FloatOp)` variant:**
```rust
pub enum GraphOp {
    // existing...
    Float(FloatOp),
}
pub enum FloatOp {
    Add, Sub, Mul, Div,
    MatMul, BatchMatMul,
    Silu, Gelu, Relu, Softmax { axis: i64 },
    RmsNorm { epsilon: f32 }, LayerNorm { axis: i64, epsilon: f32 },
    // ...
}
```

**B) Typed edges with `TensorOp`:**
```rust
pub enum GraphOp {
    // existing byte-domain ops...
    Tensor(TensorOp),   // typed tensor ops with shape/dtype metadata
}
```

**C) Just add more `Custom` op IDs with built-in handlers:**
Register standard f32 ops as built-in custom handlers in `hologram-exec`,
with well-known `CustomOpId` constants. Less architectural change, but ops
are still opaque to the compiler (no fusion, CSE may not apply).

Option A or B preferred — makes float ops first-class and visible to the
hologram compiler for optimization.

---

## 2. Shape Metadata on Graph Edges (Priority: High)

Currently `hologram::Graph` has no concept of tensor shapes or dtypes on edges.
All wires carry `Vec<u8>`. For float ops to work correctly (broadcasting,
axis-dependent reductions, attention head splitting), the graph needs per-edge
shape/dtype metadata.

**Request:** Add shape and dtype fields to graph edges or nodes:
```rust
pub struct TensorMeta {
    pub shape: Vec<u64>,
    pub dtype: DType,  // F32, F16, BF16, I64, U8, etc.
}
```

This is needed for the compiler to:
- Validate op compatibility at compile time
- Select kernels by shape (e.g., batched vs unbatched matmul)
- Compute buffer sizes for memory planning

---

## 3. Archive Section Types (Priority: High)

The architecture spec (§8) states these types live in hologram, not hologram-ai.

#### `LlmMetaSection` + related types

```rust
pub const SECTION_LLM_META: u32 = 0x0011;

pub struct LlmMetaSection {
    pub model_type: LlmModelType,
    pub kv_layout: KvCacheLayout,
    pub prefill_layer: LayerId,
    pub decode_layers: DecodeLayers,
}

pub enum LlmModelType { LlamaFamily, Bert, Gpt2 }
pub enum DecodeLayers {
    Single(LayerId),
    Bucketed(Vec<(u64, LayerId)>),
}
```

#### `TokenizerSectionData`

```rust
pub const SECTION_TOKENIZER: u32 = 0x1001;

pub struct TokenizerSectionData {
    // vocab, merges, scores, special tokens, algorithm type
}
```

#### `BucketSelector` (Phase 2)

```rust
pub struct BucketSelector { /* ... */ }
impl BucketSelector {
    pub fn from_meta(meta: &LlmMetaSection) -> Option<Self>;
    pub fn select(&self, actual_len: u64) -> Option<LayerId>;
}
```

**Workaround:** hologram-ai defines local `EmbeddableSection` implementations
with `SECTION_CUSTOM_BASE + offset` section kinds until hologram adds these.

---

## 4. Flat Re-exports (Priority: Nice to have)

```rust
pub use hologram_archive::{
    TensorPort, WeightDType, PipelineWriter,
};
```

---

## 5. Layer-Based Execution Helper (Priority: Medium)

```rust
impl KvExecutor {
    pub fn execute_layer(
        &mut self,
        pipeline_bytes: &[u8],
        layer_name: &str,
        inputs: GraphInputs,
    ) -> Result<GraphOutputs>;
}
```

---

## Priority Summary

| Item | Priority | Blocking MVP? | Workaround |
|------|----------|---------------|------------|
| Native float tensor ops (1) | **Critical** | **Yes** | Custom handlers in hologram-ai (current, violates compiler-only) |
| Shape metadata on edges (2) | **High** | Partially | Bake shapes into closures (current, fragile) |
| `LlmMetaSection` + types (3a) | High | No | Local `EmbeddableSection` |
| `TokenizerSectionData` (3b) | High | No | Local `EmbeddableSection` |
| Flat re-exports (4) | Nice to have | No | Deep module paths |
| `execute_layer()` (5) | Medium | No | Manual sub-archive extraction |

---

## Impact on hologram-ai

Once hologram adds native float ops, hologram-ai changes:

1. **Delete `crates/hologram-ai-common/src/lower/custom_ops.rs`** — all ~60
   custom handlers removed
2. **Rewrite `lower/dispatch.rs`** — map `AiOp` → `GraphOp::Float(FloatOp)`
   instead of `GraphOp::Custom`
3. **Delete `CustomOpRegistry` from `LoweringOutput`** — no custom ops to register
4. **hologram-ai ships zero runtime code** — pure compiler as intended
