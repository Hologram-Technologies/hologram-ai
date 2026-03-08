# hologram-ai: Lowering Design

---

## Overview

Lowering is the translation from `AiGraph` (AI-semantic IR) to `hologram::Graph`
(Hologram graph IR).

This is the **critical boundary** between the AI-level compiler and the Hologram
runtime. After lowering, `hologram::compile(graph)` applies generic graph-level
optimizations (LUT fusion, CSE, buffer reuse) and produces the `ExecutionSchedule`
used by `hologram::KvExecutor`.

**Two-phase optimization:**

1. **Pre-lowering** (`hologram-ai-common` opt passes): semantic AI fusions on `AiGraph`
   — attention fusion, FFN fusion, QuantMatMul fusion. These require understanding of
   model structure that `hologram-compiler` does not have.

2. **Post-lowering** (`hologram::compile`): generic graph passes on `hologram::Graph`
   — LUT chain fusion, CSE, liveness analysis, intermediate buffer slot reuse.
   `hologram-ai` must NOT re-implement these.

See ADR-0008 for the decision.

---

## Canonical AI IR (`hologram-ai-common`)

### `AiGraph`

```rust
pub struct AiGraph {
    pub name: String,
    pub nodes: Vec<AiNode>,           // topologically sorted
    pub inputs: Vec<TensorId>,        // graph-level inputs (e.g. token ids)
    pub outputs: Vec<TensorId>,       // graph-level outputs (e.g. logits)
    pub params: HashMap<TensorId, AiParam>,
    pub tensor_info: HashMap<TensorId, TensorInfo>,
    pub metadata: HashMap<String, MetaValue>,  // arch config, rope params, etc.
    pub warnings: Vec<ImportWarning>,
    pub dim_vars: DimVarTable,              // dimension variable registry (ADR-0015)
    pub shape_constraints: ConstraintStore,  // collected shape constraints (ADR-0015)
}
```

### `AiNode`

```rust
pub struct AiNode {
    pub id: NodeId,
    pub op: AiOp,
    pub inputs: Vec<TensorId>,
    pub outputs: Vec<TensorId>,
    pub attrs: NodeAttrs,
}
```

### `AiOp` — Complete Enum

```rust
pub enum AiOp {
    // Core linear algebra
    MatMul,
    BatchMatMul,
    Gemm { alpha: f32, beta: f32, trans_a: bool, trans_b: bool },
    Einsum { equation: String },

    // Activations
    Relu, Gelu, GeluApprox, Silu, Tanh, Sigmoid,
    Softmax { axis: i64 },
    LogSoftmax { axis: i64 },

    // Normalization
    LayerNorm { axis: i64, epsilon: f32 },
    RmsNorm { epsilon: f32 },
    GroupNorm { num_groups: u32, epsilon: f32 },
    BatchNorm { epsilon: f32, momentum: f32, training: bool },

    // High-level attention (semantic ops, pre-fusion)
    MultiHeadAttention {
        num_heads: u32,
        head_dim: u32,
        scale: Option<f32>,
        causal: bool,
    },
    GroupedQueryAttention {
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        scale: Option<f32>,
        causal: bool,
    },
    FlashAttentionHint,    // lowering pass decides if flash attn is available

    // Positional encoding
    RotaryEmbedding { base: f32, dim: u32 },
    AlibiSlope,

    // Shape manipulation
    Reshape { allow_zero: bool },
    Transpose { perm: Vec<u32> },
    Concat { axis: i64 },
    Split { axis: i64, sizes: Vec<u64> },
    Slice { axes: Vec<i64>, starts: Vec<i64>, ends: Vec<i64>, steps: Vec<i64> },
    Gather { axis: i64 },
    GatherElements { axis: i64 },
    Scatter { axis: i64, reduce: ScatterReduce },
    Unsqueeze { axes: Vec<i64> },
    Squeeze { axes: Vec<i64> },
    Expand,
    Tile { repeats: Vec<u64> },

    // Elementwise binary
    Add, Sub, Mul, Div, Pow, Mod,
    Min, Max,
    And, Or, Xor, Not,
    Equal, Less, LessOrEqual, Greater, GreaterOrEqual,

    // Elementwise unary
    Abs, Neg, Sqrt, Exp, Log, Sign, Floor, Ceil, Round, Clip,
    Erf, Reciprocal,

    // Reductions
    ReduceSum { axes: Vec<i64>, keepdims: bool },
    ReduceMean { axes: Vec<i64>, keepdims: bool },
    ReduceMax { axes: Vec<i64>, keepdims: bool },
    ReduceMin { axes: Vec<i64>, keepdims: bool },
    ArgMax { axis: i64, keepdims: bool },
    ArgMin { axis: i64, keepdims: bool },

    // Embeddings
    Embed,                 // token_ids → embedding vectors
    CausalMask,            // generate causal attention mask

    // Quantization (explicit in IR)
    Quantize { scheme: QuantScheme },
    Dequantize,
    QuantizedMatMul { lhs_scheme: QuantScheme, rhs_scheme: QuantScheme },

    // Fused ops (produced by optimization passes)
    FusedSwiGLU,           // gate × up → silu(gate) × up
    FusedLayerNormResidual, // x + residual → layernorm

    // Type / control
    Cast { to: DType },
    Constant { value: AiParam },
    Identity,

    // Fallback for unsupported ops
    Opaque { op_type: String, raw_attrs: Vec<u8> },
}
```

### `TensorInfo`

```rust
pub struct TensorInfo {
    pub logical_dtype: DType,    // arithmetic dtype (what math sees it as)
    pub storage_dtype: DType,    // storage dtype (how it's packed on disk/memory)
    pub shape: Shape,
    pub quant: QuantDescriptor,
}
```

### `Shape` and `DimExpr`

See [symbolic-shapes.md](symbolic-shapes.md) for the full specification and
[ADR-0015](../../adrs/0015-hologram-ai-symbolic-shapes.md) for the decision record.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct DimVarId(u32);  // interned index into DimVarTable

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DimExpr {
    Concrete(u64),                         // known constant
    Var(DimVarId),                         // symbolic variable (batch, seq_len, etc.)
    Add(Box<DimExpr>, Box<DimExpr>),       // arithmetic
    Sub(Box<DimExpr>, Box<DimExpr>),
    Mul(Box<DimExpr>, Box<DimExpr>),
    Div(Box<DimExpr>, Box<DimExpr>),
    Mod(Box<DimExpr>, Box<DimExpr>),
    CeilDiv(Box<DimExpr>, Box<DimExpr>),   // ceil(a/b) — padding/tiling
    Max(Box<DimExpr>, Box<DimExpr>),       // broadcast
    Min(Box<DimExpr>, Box<DimExpr>),       // clamp
    Dynamic,                                // truly unknown (data-dependent)
}

pub type Shape = SmallVec<[DimExpr; 4]>;
```

`DimVarTable` on `AiGraph` tracks all dimension variables with optional bounds.
Importers intern variables at import time (ONNX `dim_param` → `Var(id)`, GGUF
metadata → fixed vars with bounds). Canonical names: `batch`, `seq_len`,
`vocab_size`, `hidden_dim`, `num_heads`, `num_kv_heads`, `head_dim`, `ffn_dim`.

---

## Optimization Passes (pre-lowering)

Passes must run before lowering. Minimum required passes:

| Pass | Required for lowering |
|------|-----------------------|
| `ConstantFolding` | Yes — eliminates shape ops that confuse lowering |
| `ShapePropagation` | Yes — lowering needs shapes for buffer sizing |
| `DeadNodeElimination` | Yes — prevents emitting unreachable plan nodes |
| `AttentionFusion` | Recommended — enables fused MHA kernel |
| `QuantMatMulFusion` | Recommended — enables quantized GEMM |

---

## Lowering Pipeline

```
AiGraph (optimized, shape-propagated)
        +
KvCacheLayout
        │
   ┌────▼────────────────────────┐
   │ 1. Topological Sort         │  (respecting KV read/write ordering)
   └────┬────────────────────────┘
        │ ordered node list
   ┌────▼────────────────────────┐
   │ 2. Op Dispatch              │  AiOp → GraphOp (see ADR-0007 table)
   └────┬────────────────────────┘
        │ hologram::Graph nodes
   ┌────▼────────────────────────┐
   │ 3. Param Packing            │  AiParam → ConstantId in ConstantStore
   └────┬────────────────────────┘
        │
   ┌────▼────────────────────────┐
   │ 4. Custom Op Registration   │  attention/norm/rope → CustomOpRegistry
   └────┬────────────────────────┘
        │
   ┌────▼────────────────────────┐
   │ 5. Graph Assembly           │  → hologram::Graph (ready for compiler)
   └─────────────────────────────┘
```

After `lower()` returns, the caller invokes `hologram::compile(graph)` to get
the `ExecutionSchedule` and handle intermediate-buffer assignment (liveness
analysis + workspace slot reuse). `lower()` does not produce a schedule — that
is hologram-compiler's job. See ADR-0008.

---

## Op Dispatch Table

The dispatch table maps `AiOp` to `GraphOp` (for native hologram ops) or
`CustomOpId` (for ops registered in `CustomOpRegistry`). See ADR-0007 for the
canonical mapping.

| `AiOp` | `GraphOp` | Notes |
|--------|-----------|-------|
| `MatMul` (Q4_0 weights) | `MatMulLut4(ConstantId)` | quantized GEMM, native |
| `MatMul` (Q8_0 weights) | `MatMulLut8(ConstantId)` | Phase 2 |
| `Gelu`, `GeluApprox` | `Lut(LutOp::Gelu)` | O(1) 256-byte LUT |
| `Relu` | `Lut(LutOp::Relu)` | O(1) LUT |
| `Silu` | `Lut(LutOp::Silu)` | O(1) LUT |
| `Tanh` | `Lut(LutOp::Tanh)` | O(1) LUT |
| `Sigmoid` | `Lut(LutOp::Sigmoid)` | O(1) LUT |
| `Add`, `Sub`, `Mul`, `Div` | `Prim(PrimOp::Add/Sub/Mul/Div)` | byte-domain |
| `Neg`, `Abs` | `Prim(PrimOp::Neg)` / `Lut(…)` | unary |
| `Constant` | `GraphOp::Constant(ConstantId)` | native |
| `MultiHeadAttention` | `Custom { id: ATTN_OP, arity: 3 }` | CustomOpRegistry |
| `GroupedQueryAttention` | `Custom { id: GQA_OP, arity: 3 }` | CustomOpRegistry |
| `FlashAttentionHint` | `Custom { id: ATTN_OP, arity: 3 }` | same handler |
| `RmsNorm` | `Custom { id: RMS_NORM_OP, arity: 2 }` | CustomOpRegistry |
| `LayerNorm` | `Custom { id: LAYER_NORM_OP, arity: 2 }` | CustomOpRegistry |
| `Softmax` | `Custom { id: SOFTMAX_OP, arity: 1 }` | CustomOpRegistry |
| `RotaryEmbedding` | `Custom { id: ROPE_OP, arity: 3 }` | CustomOpRegistry |
| `Embed` | `Custom { id: EMBED_OP, arity: 2 }` | CustomOpRegistry |
| `Dequantize` | `Custom { id: DEQUANT_OP, arity: 1 }` | explicit per ADR-0004 |
| `FusedSwiGLU` | `Custom { id: SWIGLU_OP, arity: 2 }` | CustomOpRegistry |
| `Reshape`, `Transpose` | `Custom { id: RESHAPE_OP, arity: 1 }` | shape ops |
| `Cast` | `Custom { id: CAST_OP, arity: 1 }` | type cast |
| `Concat` | `Custom { id: CONCAT_OP, arity: N }` | variadic |
| `Opaque` | **lowering error** | — |

Custom op handlers are registered in `CustomOpRegistry` during lowering inside
`lower()` and returned as `LoweringOutput.registry`. The registry is passed to
`KvExecutor::execute_layer()` on each invocation by the caller.

---

## Quantization Handling in Lowering

Quantized weights stored as `AiParam::Lazy` remain in their quantized storage
format. The lowering pass decides the dequantization strategy:

**Strategy A: Eager dequant at plan start**
- Insert `dequant_{scheme}` nodes at the plan boundary
- All compute runs in f32/f16
- Chosen when: backend has no quantized GEMM kernels

**Strategy B: Fused quant kernels**
- `QuantizedMatMul` → `qgemm_{lhs}_{rhs}`
- Dequantization is kernel-internal
- Chosen when: backend declares `HAS_QGEMM` capability

**Strategy C: Mixed**
- Per-op decision based on backend capability query
- Default for CPU backend (has Q4_0 and Q8_0 GEMM but not all quant types)

The `LoweringOptions::quant_strategy` field selects the strategy or leaves it
to auto-detection.

---

## Shape Handling in Lowering

See [symbolic-shapes.md](symbolic-shapes.md) section 7 (Lowering Strategies) and
[ADR-0015](../../adrs/0015-hologram-ai-symbolic-shapes.md).

### Pre-lowering: Shape Concretization

Before `lower()` is called, all shapes must be fully concrete. The `concretize_shapes()`
function substitutes all fixed variables from `DimVarTable` and evaluates expressions.
Lowering fails if any dimension remains symbolic.

### Shape Strategies

```rust
pub enum ShapeStrategy {
    FixToMax,                         // fix all dims to upper bound (MVP, default)
    Bucketed(BucketConfig),           // compile N variants for a key dim
    Profiles(Vec<ShapeProfile>),      // compile for specific shape assignments
    PaddedMax,                        // fix to max + actual_len masking
}
```

Each strategy concretizes symbolic dims *before* calling `lower()`. The `hologram::Graph`
always receives fully concrete dimensions. No changes to the hologram API.

**FixToMax** (MVP): Fix all unfixed variables to their upper bound via
`DimVarTable::concretize_to_upper()`. Rebuild graph when dimensions differ (e.g., between
prefill and decode).

**Bucketed** (Phase 2): Compile separate graphs for a set of bucket sizes (e.g.,
`seq_len ∈ [128, 512, 1024, 2048]`). At runtime, select the smallest bucket ≥ actual value.

**Profiles** (Phase 3): Compile for specific named shape assignments (e.g., "prefill_short",
"decode_single"). Multi-variable specialization.

**PaddedMax** (Phase 3): Same as FixToMax but attention ops receive `actual_seq_len` as a
runtime input for masking. Single graph, no wasted computation on padding.

### Dynamic shapes

`DimExpr::Dynamic` (truly data-dependent dimensions) → lowering emits placeholder nodes
with runtime size-calculation ops. Higher cost; avoided where possible.

---

## KV-Cache Lowering

KV-cache buffers are allocated via `MemoryPlan::kv_cache_layout`.

At lowering time:
- A `Custom { id: KV_WRITE_OP }` node is emitted after each attention layer's K/V projections
- A `Custom { id: KV_READ_OP }` node is emitted at the beginning of each attention computation
- The KV-cache byte buffer is passed as an explicit input/output to each layer invocation

The caller owns the KV-cache buffer and the `present_len` counter. Both are
passed as named tensor inputs to `KvExecutor::execute_layer()` on each call —
`kv_cache: [n_bytes] u8` in/out, and `present_len: [] u32` on the decode layer.
See `hologram::LlmMetaSection.kv_layout.total_bytes` for buffer sizing.
There is no session type in hologram-ai that manages this state (ADR-0016).

---

## Lowering Errors

Lowering is strict. It fails if:
- Any `AiOp::Opaque` node is encountered
- A required hologram kernel is not available on the target backend
- Shape information is missing for buffer allocation (except for dynamic dims)
- Memory plan is inconsistent with graph (tensor IDs don't match)

Lowering warnings (non-fatal):
- A requested fused kernel is unavailable, falling back to decomposed sequence
- A symbolic dim was not resolved (will require runtime shape dispatch)