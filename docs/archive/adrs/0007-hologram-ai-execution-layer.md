# ADR-0007: hologram-ai execution layer maps directly to hologram-graph + hologram-exec

- Status: Accepted
- Date: 2026-03-06
- Owners: Architecture
- Supersedes: fictional type references in ADR-0005

---

## Context

The hologram-ai architecture documents were initially written assuming the hologram runtime
would expose types named `ExecutionPlan`, `ExecutionBackend`, `MemoryRegion`,
`ArtifactReference`, and a crate named `hologram-cpu`. These types do not exist.

The actual hologram project is an O(1) LUT-based engine. Its API (confirmed from
hologram-graph, hologram-exec, and hologram-archive) uses different names and a different
execution model. hologram-ai must target the real API.

Additionally, the "hologram deps" prompt in the hologram-ai repo presented three options
for handling the execution layer. This ADR records the chosen approach.

---

## Decision

**Option 2: Adapt hologram-ai to target the actual hologram API directly.**

hologram-ai-lower produces a `hologram::Graph + hologram::ExecutionSchedule` (not a
fictional `ExecutionPlan`). hologram-ai-session executes it via `hologram::KvExecutor`
with a `hologram::CustomOpRegistry` that registers handlers for AI-specific operations.

### Type mapping

| Assumed (fictional) | Actual hologram type | Crate |
|---------------------|---------------------|-------|
| `ExecutionPlan` | `Graph` + `ExecutionSchedule` | `hologram-graph` |
| `ExecutionBackend` trait | `KvExecutor` (stateless, `&self`) | `hologram-exec` |
| `MemoryRegion` | `BufferArena` | `hologram-exec` |
| `ArtifactReference` | `ConstantId` / `ConstantStore` | `hologram-graph` |
| `hologram-cpu` crate | `hologram-exec` crate | — |
| Backend capability query | `CustomOpRegistry::register` at init | `hologram-exec` |

All types are accessible as `hologram::TypeName` via the root crate's flat re-exports.

### AiOp → GraphOp mapping

| `AiOp` | `GraphOp` | Notes |
|--------|-----------|-------|
| `MatMul` (Q4_0) | `MatMulLut4(ConstantId)` | weights stored in `ConstantStore` |
| `MatMul` (Q8_0) | `MatMulLut8(ConstantId)` | Phase 2 |
| `Gelu`, `Relu`, `Silu`, `Tanh`, `Sigmoid` | `Lut(LutOp::Gelu/Relu/…)` | O(1) 256-byte LUT |
| `Add`, `Mul`, `Sub`, `Div`, etc. | `Prim(PrimOp::Add/Mul/…)` | byte-domain binary |
| `MultiHeadAttention`, `GroupedQueryAttention` | `Custom { id: AttentionOpId, arity: 3 }` | `CustomOpRegistry` |
| `RmsNorm`, `LayerNorm` | `Custom { id: NormOpId, arity: 2 }` | `CustomOpRegistry` |
| `Dequantize` | `Custom { id: DequantOpId, arity: 1 }` | explicit per ADR-0004 |
| `Softmax` | `Custom { id: SoftmaxOpId, arity: 1 }` | `CustomOpRegistry` |
| `Embed` | `Custom { id: EmbedOpId, arity: 2 }` | `CustomOpRegistry` |
| `RotaryEmbedding` | `Custom { id: RopeOpId, arity: 3 }` | `CustomOpRegistry` |
| `Constant` reference | `GraphOp::Constant(ConstantId)` | native `GraphOp` |
| `Reshape`, `Transpose` | `Custom { id: …, arity: 1 }` | shape ops in custom handler |

### Weight storage

Quantized weights (Q4_0 bytes) are stored in `ConstantStore` as `ConstantData::Bytes`.
Large models use `ConstantData::Deferred { … }` for lazy loading via hologram-archive
`HoloLoader` (mmap-backed). This replaces the fictional `ArtifactReference` system.

### KV-cache

`BufferArena` (hologram-exec) replaces the fictional `MemoryRegion`. `InferenceSession`
owns a `BufferArena` per session. The arena is not shared with `KvExecutor` — it is passed
as a mutable reference into the execution call.

### Concurrency

`KvExecutor::execute` takes `&self` (stateless). Multiple concurrent sessions can call
`execute` on a shared `Arc<KvExecutor>` without synchronization.

---

## Consequences

**Positive:**
- No throwaway self-contained layer; hologram integration is correct from day one
- `KvExecutor` is already optimized and tested; hologram-ai does not duplicate it
- `CustomOpRegistry` is the natural extension point for AI-specific ops
- `hologram::Graph` is a richer IR than a custom `ExecutionPlan` would have been

**Negative:**
- hologram-ai-lower must produce a `hologram::Graph`, not its own intermediate;
  this couples the lowering output format to hologram's Graph type
- AI-specific ops (attention, norm, rope) must be registered as custom handlers;
  their implementations live in hologram-ai-lower, not in hologram itself

**Neutral:**
- The `hologram-ai-backend` crate concept is eliminated; no separate CPU/Metal/CUDA
  backend abstraction layer. All execution goes through `KvExecutor`.

---

## Alternatives Considered

**Self-contained MVP:** Define ExecutionPlan/CPU executor directly in hologram-ai-lower and
hologram-ai-session. Wire to hologram later. Rejected: hologram is production-ready today;
a self-contained layer would be throwaway work.

**Ask hologram team first:** Pause MVP until hologram exposes the expected tensor execution
API surface. Rejected: hologram's API is already available; this option was based on a
misunderstanding of the hologram project state.
