# Research Report: hologram-ai — Revised Design (Post-Review Corrections)

- Date: 2026-03-06
- Status: Accepted — supersedes sections 4, 10, 13, and 15 of the initial report
- Supersedes: `2026-03-06-hologram-ai-architecture.md` (initial draft)

---

## 1. Purpose

This document records the corrections made after reviewing the initial
architecture draft against the actual `hologram` codebase. Three classes of
errors were found and fixed:

1. **Fictional hologram types** — the initial draft invented type names
   (`ExecutionPlan`, `ExecutionBackend`, `MemoryRegion`, `ArtifactReference`)
   that do not match what `hologram` actually exposes. These are replaced with
   the real type names.

2. **Over-fragmented crate layout** — the initial 14-crate layout was split
   too finely. Compilation friction, circular-dependency risk, and testing
   overhead outweigh the theoretical benefits of separate crates for each
   compiler phase. The revised layout consolidates to 6 crates.

3. **Import path rule** — the initial draft did not state explicitly that
   `hologram-ai` must never import `hologram` subcrates directly. Only the
   `hologram` root crate's re-exports are stable; subcrate paths are internal
   implementation details.

---

## 2. Correction 1: Real Hologram Types

The following table replaces the fictional type names used throughout the
initial research report and all plan documents derived from it.

| Role | Fictional name (wrong) | Real hologram type (correct) |
|------|------------------------|------------------------------|
| Compiled execution plan | `ExecutionPlan` | `hologram::Graph` + `hologram::ExecutionSchedule` |
| Backend execution interface | `ExecutionBackend` trait | `hologram::KvExecutor` |
| Activation buffer / scratch | `MemoryRegion`, `BufferView` | `hologram::BufferArena` |
| Immutable weight storage | `ArtifactReference` | `hologram::ConstantId` + `hologram::ConstantStore` |
| Custom op extension point | (not defined) | `hologram::CustomOpRegistry` |
| Compiled model persistence | (not defined) | `hologram::HoloLoader` + `hologram::HoloWriter` |

### Import rule

All of the above types are accessed as `hologram::TypeName` from the
`hologram` root crate. No `hologram-ai-*` crate imports subcrates of
`hologram` directly (e.g. `hologram-graph`, `hologram-exec`, `hologram-archive`).
Subcrate paths are internal to `hologram` and subject to change.

```toml
# correct — only the root crate
[dependencies]
hologram = { path = "../hologram" }

# wrong — subcrate path, breaks on internal hologram refactoring
[dependencies]
hologram-graph = { path = "../hologram/crates/hologram-graph" }
```

### How the types fit together

```
hologram::Graph              ← lowering output; the op graph
       │
hologram::compile(graph)     ← post-lowering optimizer (LUT fusion, CSE, buffer reuse)
       │
hologram::ExecutionSchedule  ← compiled, device-ordered execution schedule
       │
hologram::KvExecutor::run(schedule, inputs, arena)
       │                        ↑
       │                hologram::BufferArena  ← per-session scratch + KV-cache
       │
  outputs: Tensor map
```

`hologram::Graph` is the mutable IR that `hologram-ai-common::lower()` writes
to. After lowering, `hologram::compile()` optimizes the graph into an
`ExecutionSchedule`. The schedule is immutable and shared across sessions.
`KvExecutor::run()` is `&self` (concurrent-safe).

### Weight storage

Immutable model weights are stored in `hologram::ConstantStore`:

```
AiParam::Inline(bytes)
  → ConstantData::Bytes(bytes)
  → stored in ConstantStore
  → referenced by ConstantId in Graph nodes

AiParam::Lazy (large GGUF tensors)
  → ConstantData::Deferred { path, offset, len }
  → loaded via HoloLoader (mmap-backed)
  → referenced by ConstantId
```

---

## 3. Correction 2: Slim 6-Crate Layout

The initial 14-crate layout is replaced with a 6-crate workspace.

### New crate layout

```
hologram-ai/
├── Cargo.toml                     # workspace root (6 members)
├── CLAUDE.md
├── crates/
│   ├── hologram-ai-quant/         # quant primitives: QuantScheme, block types, dequant
│   ├── hologram-ai-common/        # IR + opt passes + mem planner + lowering
│   ├── hologram-ai-onnx/          # ONNX importer → AiGraph
│   ├── hologram-ai-gguf/          # GGUF importer → AiGraph
│   ├── hologram-ai-ggml/          # GGML importer → AiGraph
│   └── hologram-ai/               # session + stream + validate + CLI (facade)
└── tests/
    ├── fixtures/
    │   ├── onnx/
    │   ├── gguf/
    │   └── golden/
    └── integration/
```

### Dependency matrix

```
hologram-ai-quant    → (no internal deps; no hologram dep)
hologram-ai-common   → hologram-ai-quant, hologram (root)
hologram-ai-onnx     → hologram-ai-common
hologram-ai-gguf     → hologram-ai-common, hologram-ai-quant
hologram-ai-ggml     → hologram-ai-common, hologram-ai-quant
hologram-ai          → hologram-ai-quant, hologram-ai-common,
                        hologram-ai-onnx, hologram-ai-gguf, hologram-ai-ggml,
                        hologram (root)
```

### What merged vs. what stayed separate

| Initial crates (14) | Destination in revised layout |
|---------------------|-------------------------------|
| `holo-ai-ir` | merged into `hologram-ai-common/src/ir` |
| `holo-ai-quant` | renamed `hologram-ai-quant` (kept separate) |
| `holo-ai-opt` | merged into `hologram-ai-common/src/opt` |
| `holo-ai-mem` | merged into `hologram-ai-common/src/mem` |
| `holo-ai-lower` | merged into `hologram-ai-common/src/lower` |
| `holo-ai-onnx` | renamed `hologram-ai-onnx` |
| `holo-ai-gguf` | renamed `hologram-ai-gguf` |
| `holo-ai-ggml` | renamed `hologram-ai-ggml` |
| `holo-ai-session` | merged into `hologram-ai/src/session` |
| `holo-ai-stream` | merged into `hologram-ai/src/stream` |
| `holo-ai-backend` | eliminated — backend is hologram-native via `KvExecutor` |
| `holo-ai-validate` | merged into `hologram-ai/src/validate` |
| `holo-ai-cli` | merged into `hologram-ai/src/cli` (binary target) |
| `holo-ai` (facade) | `hologram-ai` is now the facade itself |

**Rationale for `hologram-ai-quant` staying separate:**
- Zero transitive dependencies (no IR, no hologram)
- Block layout structs must exactly match `ggml-quants.h` — isolation prevents accidental coupling
- Can be published independently and used by other crates without pulling in the IR

**Rationale for `hologram-ai-common` consolidating IR + opt + mem + lower:**
- These four compiler phases are tightly coupled by type (`AiGraph` flows through all of them)
- Splitting them created awkward one-directional dependencies with no real abstraction value
- Single `cargo test -p hologram-ai-common` covers the entire compiler
- `pub mod ir`, `pub mod opt`, `pub mod mem`, `pub mod lower` preserve the logical separation

**Rationale for `hologram-ai` consolidating session + stream + validate + CLI:**
- All depend on `KvExecutor` and `BufferArena` from hologram — keeping them together simplifies the hologram dependency surface
- `holo-ai-backend` is eliminated entirely: backend selection is `hologram`'s job, not `hologram-ai`'s
- CLI is a binary target in the same crate (`[[bin]] name = "hologram-ai"`)

---

## 4. Correction 3: Import Path Rule (Formalized)

**Rule:** No `hologram-ai-*` crate may add `hologram-graph`, `hologram-exec`,
`hologram-archive`, or any other `hologram-*` subcrate as a direct `Cargo.toml`
dependency. All hologram types are accessed through the `hologram` root crate.

This rule prevents `hologram-ai` from coupling to hologram's internal module
structure, which is not considered stable API surface.

The rule is documented in `CLAUDE.md` at the `hologram-ai` repo root as a
non-negotiable constraint.

---

## 5. Crate Constraints (MUST / MUST NOT)

These tables are the authoritative source for what each crate is and is not
allowed to contain. Violations indicate functionality in the wrong place.

### `hologram-ai-quant`

| MUST | MUST NOT |
|------|----------|
| Define `QuantScheme`, `QuantDescriptor`, raw block structs (`Q4_0Block`, etc.) | Import `AiGraph`, `AiOp`, or any IR types |
| Implement `dequant_tensor()` and `quant_tensor()` | Contain format parsing logic (ONNX/GGUF/GGML) |
| Match `ggml-quants.h` block layouts exactly | Import the `hologram` crate |
| Have zero transitive dependencies beyond `half` and `smallvec` | Contain optimization or lowering logic |

### `hologram-ai-common`

| MUST | MUST NOT |
|------|----------|
| Define `AiGraph`, `AiNode`, `AiOp`, `TensorInfo`, `Shape`, `DType` | Parse any model file format |
| Own all optimization passes (`OptPipeline`, fusion, folding, elimination) | Manage inference sessions or KV-cache state |
| Own `MemoryPlanner` → `MemoryPlan` | Import hologram subcrates directly |
| Own `lower()` → `LoweringOutput { graph: hologram::Graph, registry: hologram::CustomOpRegistry }` | Define streaming or token generation logic |
| Register custom AI op handlers in `hologram::CustomOpRegistry` during `lower()` | Invoke `hologram::compile()` (that is the facade crate's job) |
| Access hologram only via the `hologram` root crate | |

### `hologram-ai-onnx`

| MUST | MUST NOT |
|------|----------|
| Produce `AiGraph` as its sole public output | Perform optimization passes |
| Contain all ONNX-specific parsing and type logic | Produce `hologram::Graph` directly |
| Support opsets 13–21 | Import other importer crates |
| Depend on `hologram-ai-common` only (no direct hologram dep) | Emit `hologram` types |

### `hologram-ai-gguf`

| MUST | MUST NOT |
|------|----------|
| Produce `AiGraph` as its sole public output | Perform optimization passes |
| Contain all GGUF parsing and architecture recognition | Produce `hologram::Graph` directly |
| Support GGUF v1, v2, v3 | Import other importer crates |

### `hologram-ai-ggml`

| MUST | MUST NOT |
|------|----------|
| Produce `AiGraph` as its sole public output | Perform optimization passes |
| Contain all GGML v1 checkpoint logic | Produce `hologram::Graph` directly |
| | Import other importer crates |

### `hologram-ai` (facade)

| MUST | MUST NOT |
|------|----------|
| Be the sole crate downstream consumers depend on | Contain optimization pass implementations |
| Own `InferenceSession` lifecycle and KV-cache management | Contain format-specific parsing logic |
| Own streaming token generation (`stream_tokens`, `TokenStream`) | Expose hologram subcrate types in public API |
| Own `ModelCompiler` (drives full pipeline incl. `hologram::compile()`) | |
| Depend on `hologram` with `compiler` feature for post-lowering compilation | |
| Own `ValidationSuite` and CLI binary | |
| Re-export key types from all internal crates | |

---

## 6. Revised Lowering Output

The initial draft had `lower()` produce a `hologram::ExecutionPlan` directly.
The correct model has two stages:

### Stage A: `lower()` in `hologram-ai-common`

```rust
pub struct LoweringOutput {
    pub graph:    hologram::Graph,        // mutable AI op graph
    pub registry: hologram::CustomOpRegistry, // custom op handlers
}

pub fn lower(
    graph:    &AiGraph,
    mem_plan: &MemoryPlan,
    opts:     &LoweringOptions,
) -> Result<LoweringOutput>
```

`lower()` maps `AiOp` variants to `hologram::GraphOp` entries and registers
any custom op handlers (attention, norm, rope, dequant) in `CustomOpRegistry`.
It does **not** call `hologram::compile()`.

### Stage B: `compile()` in `hologram-ai` (facade)

```rust
// In hologram-ai/src/session.rs
let lowered = lower(&graph, &mem_plan, &opts)?;
let schedule = hologram::compile(lowered.graph, &lowered.registry)?;
// schedule is Arc<ExecutionSchedule>, shared across all sessions
```

`hologram::compile()` is responsible for post-lowering optimizations: LUT
fusion, common subexpression elimination, and buffer reuse. This work belongs
in `hologram`, not in `hologram-ai`.

**Why the split matters:**
- `hologram-ai-common` does not need `hologram`'s compiler feature, only
  `hologram`'s graph construction types
- `hologram-ai` (the facade) requires `hologram` with `features = ["compiler"]`
- Tests for `hologram-ai-common::lower()` can check the `hologram::Graph`
  structure without running the full compiler pipeline

---

## 7. Revised Op Dispatch Table

The initial op dispatch table used fictional kernel names. This table uses
the actual `hologram::GraphOp` variants.

| `AiOp` | `hologram::GraphOp` | Notes |
|---------|---------------------|-------|
| `MatMul` (Q4_0 weights) | `GraphOp::MatMulLut4(ConstantId)` | weights in `ConstantStore` |
| `MatMul` (Q8_0 weights) | `GraphOp::MatMulLut8(ConstantId)` | weights in `ConstantStore` |
| `MatMul` (F32 weights) | `GraphOp::MatMul(ConstantId)` | |
| `Gelu` | `GraphOp::Lut(LutOp::Gelu)` | O(1) LUT, no kernel needed |
| `Relu` | `GraphOp::Lut(LutOp::Relu)` | |
| `Silu` | `GraphOp::Lut(LutOp::Silu)` | |
| `Tanh` | `GraphOp::Lut(LutOp::Tanh)` | |
| `Sigmoid` | `GraphOp::Lut(LutOp::Sigmoid)` | |
| `Add`, `Mul`, `Sub`, `Div` | `GraphOp::Prim(PrimOp::Add)` etc. | byte-domain prim |
| `Neg`, `Abs`, `Sqrt` | `GraphOp::Prim(PrimOp::…)` | |
| `MultiHeadAttention` | `GraphOp::Custom { id, arity: 3 }` | registered in `CustomOpRegistry` |
| `GroupedQueryAttention` | `GraphOp::Custom { id, arity: 3 }` | |
| `RmsNorm` | `GraphOp::Custom { id, arity: 2 }` | |
| `LayerNorm` | `GraphOp::Custom { id, arity: 2 }` | |
| `RotaryEmbedding` | `GraphOp::Custom { id, arity: 2 }` | |
| `Dequantize` | `GraphOp::Custom { id, arity: 1 }` | explicit per ADR-0004 |
| `Embed` | `GraphOp::Constant(ConstantId)` + gather prim | |
| Weight constants | `GraphOp::Constant(ConstantId)` | native `GraphOp` |
| `Reshape` | `GraphOp::Reshape` | |
| `Transpose` | `GraphOp::Prim(PrimOp::Transpose)` | |
| `Cast` | `GraphOp::Cast { to }` | |
| `Softmax` | `GraphOp::Lut(LutOp::Softmax)` | or decomposed exp+sum+div |

Custom op handler registration example:

```rust
// In hologram-ai-common/src/lower/custom_ops.rs
pub fn register_ai_ops(registry: &mut hologram::CustomOpRegistry) {
    registry.register("mha", |inputs: &[&Tensor], _attrs| {
        // Fused multi-head attention kernel
        mha_cpu(inputs[0], inputs[1], inputs[2])
    });
    registry.register("rms_norm", |inputs, attrs| {
        rms_norm_cpu(inputs[0], inputs[1], attrs.epsilon)
    });
    registry.register("rope", |inputs, attrs| {
        rope_cpu(inputs[0], inputs[1], attrs.base, attrs.dim)
    });
    registry.register("dequant_q4_0", |inputs, _attrs| {
        dequant_q4_0_tensor(inputs[0])
    });
    // ...
}
```

---

## 8. Revised KV-Cache Model (using real types)

KV-cache storage uses `hologram::BufferArena`:

```rust
pub struct InferenceSession {
    compiled:   Arc<CompiledModel>,
    kv_arena:   hologram::BufferArena,   // replaces custom KvCache struct
    present_len: usize,
}

pub struct CompiledModel {
    graph:    Arc<hologram::Graph>,
    schedule: Arc<hologram::ExecutionSchedule>,
    registry: Arc<hologram::CustomOpRegistry>,
    executor: Arc<hologram::KvExecutor>,
    kv_layout: Option<KvCacheLayout>,   // AI-layer metadata
    metadata:  ModelMetadata,
}
```

`hologram::BufferArena` is allocated once per session at construction time.
Its size is computed from `KvCacheLayout::total_bytes`.

On each `run()` call:
1. Session builds the input map (token ids, present_len as scalar, etc.)
2. `KvExecutor::run(&schedule, inputs, &mut kv_arena)` executes the graph
3. KV slot nodes in the graph read and write directly into `kv_arena` slices
4. Session increments `present_len`

`KvExecutor::run()` is `&self` — the executor is stateless. All mutable state
is in `kv_arena` (per-session).

---

## 9. Revised ADR Agenda

The following two ADRs were referenced in the ecosystem doc but not yet written:

### ADR-0007: Execution layer maps to real hologram types

Scope: The `hologram-ai-common::lower()` function targets `hologram::Graph` +
`hologram::CustomOpRegistry`, not fictional type abstractions. The facade
invokes `hologram::compile()` to produce an `ExecutionSchedule`.

### ADR-0008: `hologram::compile()` is invoked by the facade, not by lowering

Scope: `hologram-ai-common::lower()` stops at `hologram::Graph`. The facade
crate (`hologram-ai`) invokes `hologram::compile()` after lowering. This
separates graph construction (AI concern) from schedule compilation (hologram
concern), and allows `hologram-ai-common` tests to validate graph structure
without the full compiler pipeline.

Both ADRs are written in `specs/adrs/0007-*.md` and `specs/adrs/0008-*.md`.

---

## 10. Updated Workspace `Cargo.toml`

```toml
[workspace]
resolver = "2"
members = [
    "crates/hologram-ai-quant",
    "crates/hologram-ai-common",
    "crates/hologram-ai-onnx",
    "crates/hologram-ai-gguf",
    "crates/hologram-ai-ggml",
    "crates/hologram-ai",
]

[workspace.package]
version    = "0.1.0"
edition    = "2021"
license    = "MIT OR Apache-2.0"
repository = "https://github.com/uor-framework/hologram-ai"

[workspace.dependencies]
# hologram root crate only — never subcrates
hologram = { path = "../hologram" }

# workspace-internal
hologram-ai-quant  = { path = "crates/hologram-ai-quant" }
hologram-ai-common = { path = "crates/hologram-ai-common" }
hologram-ai-onnx   = { path = "crates/hologram-ai-onnx" }
hologram-ai-gguf   = { path = "crates/hologram-ai-gguf" }
hologram-ai-ggml   = { path = "crates/hologram-ai-ggml" }

# third-party
bytes        = "1"
half         = { version = "2", features = ["std"] }
smallvec     = { version = "1", features = ["union"] }
thiserror    = "2"
anyhow       = "1"
tracing      = "0.1"
serde        = { version = "1", features = ["derive"] }
serde_json   = "1"
futures      = "0.3"
async-stream = "0.3"
approx       = "0.5"
clap         = { version = "4", features = ["derive"] }
memmap2      = "0.9"
prost        = "0.13"
```

---

## 11. Updated Crate Cargo.toml Files

### `crates/hologram-ai-quant/Cargo.toml`

```toml
[package]
name    = "hologram-ai-quant"
version.workspace = true
edition.workspace = true

[dependencies]
half     = { workspace = true }
smallvec = { workspace = true }
```

### `crates/hologram-ai-common/Cargo.toml`

```toml
[package]
name    = "hologram-ai-common"
version.workspace = true
edition.workspace = true

[dependencies]
hologram-ai-quant = { workspace = true }
hologram          = { workspace = true }
bytes             = { workspace = true }
thiserror         = { workspace = true }
tracing           = { workspace = true }
serde             = { workspace = true }
```

### `crates/hologram-ai-onnx/Cargo.toml`

```toml
[package]
name    = "hologram-ai-onnx"
version.workspace = true
edition.workspace = true

[dependencies]
hologram-ai-common = { workspace = true }
bytes  = { workspace = true }
prost  = { workspace = true }

[build-dependencies]
prost-build = "0.13"
```

### `crates/hologram-ai-gguf/Cargo.toml`

```toml
[package]
name    = "hologram-ai-gguf"
version.workspace = true
edition.workspace = true

[dependencies]
hologram-ai-common = { workspace = true }
hologram-ai-quant  = { workspace = true }
bytes   = { workspace = true }
memmap2 = { workspace = true }
```

### `crates/hologram-ai-ggml/Cargo.toml`

```toml
[package]
name    = "hologram-ai-ggml"
version.workspace = true
edition.workspace = true

[dependencies]
hologram-ai-common = { workspace = true }
hologram-ai-quant  = { workspace = true }
bytes = { workspace = true }
```

### `crates/hologram-ai/Cargo.toml`

```toml
[package]
name    = "hologram-ai"
version.workspace = true
edition.workspace = true

[[bin]]
name = "hologram-ai"
path = "src/main.rs"

[lib]
name = "hologram_ai"
path = "src/lib.rs"

[dependencies]
hologram            = { workspace = true, features = ["compiler"] }
hologram-ai-quant   = { workspace = true }
hologram-ai-common  = { workspace = true }
hologram-ai-onnx    = { workspace = true }
hologram-ai-gguf    = { workspace = true }
hologram-ai-ggml    = { workspace = true }
futures      = { workspace = true }
async-stream = { workspace = true }
clap         = { workspace = true }
anyhow       = { workspace = true }
tracing      = { workspace = true }
serde        = { workspace = true }
serde_json   = { workspace = true }
```

---

## 12. Updated `hologram-ai-common` Module Map

```
crates/hologram-ai-common/src/
├── lib.rs               pub mod ir, opt, mem, lower; re-exports
├── ir/
│   ├── mod.rs
│   ├── dtype.rs         DType enum
│   ├── shape.rs         Shape, Dim
│   ├── op.rs            AiOp enum (complete), NodeAttrs, ScatterReduce
│   ├── param.rs         AiParam, ParamStorage
│   ├── node.rs          AiNode, NodeId
│   └── graph.rs         AiGraph, TensorInfo, TensorId, MetaValue, ImportWarning
├── opt/
│   ├── mod.rs           OptPipeline, Pass trait
│   ├── constant_fold.rs ConstantFolding pass
│   ├── dce.rs           DeadNodeElimination pass
│   ├── shape_prop.rs    ShapePropagation pass
│   ├── attn_fusion.rs   AttentionFusion pass
│   ├── ffn_fusion.rs    FfnFusion pass
│   └── quant_matmul.rs  QuantMatMulFusion pass
├── mem/
│   ├── mod.rs           MemoryPlanner, MemoryPlan, KvCacheLayout
│   ├── liveness.rs      tensor liveness analysis
│   └── alias.rs         buffer alias detection
└── lower/
    ├── mod.rs           lower(), LoweringOutput, LoweringOptions
    ├── op_dispatch.rs   AiOp → GraphOp mapping table
    ├── custom_ops.rs    register_ai_ops() → CustomOpRegistry
    ├── buf_bind.rs      MemoryPlan → BufferArena layout
    └── const_pack.rs    AiParam → ConstantStore
```

---

## 13. Updated `hologram-ai` (facade) Module Map

```
crates/hologram-ai/src/
├── lib.rs       re-exports; public API surface
├── main.rs      [[bin]] entry point → cli::run()
├── session.rs   ModelCompiler, CompiledModel, InferenceSession
├── stream.rs    TokenStream, Tokenizer trait, GenerateOptions, samplers
├── validate.rs  ValidationSuite, ValidationReport, compare_tensors()
└── cli/
    ├── mod.rs
    ├── generate.rs    `hologram-ai generate` subcommand
    ├── validate.rs    `hologram-ai validate` subcommand
    ├── inspect.rs     `hologram-ai inspect` subcommand
    └── run.rs         `hologram-ai run` subcommand
```

---

## 14. Summary of Changes from Initial Draft

| Topic | Initial draft | Revised design |
|-------|---------------|----------------|
| Crate count | 14 | 6 |
| Crate prefix | `holo-ai-*` | `hologram-ai-*` |
| Execution type | `hologram::ExecutionPlan` (fictional) | `hologram::Graph` + `hologram::ExecutionSchedule` |
| Backend type | `hologram::ExecutionBackend` trait (fictional) | `hologram::KvExecutor` |
| Buffer type | `hologram::MemoryRegion` (fictional) | `hologram::BufferArena` |
| Weight ref type | `hologram::ArtifactReference` (fictional) | `hologram::ConstantId` + `hologram::ConstantStore` |
| Custom op ext. | not addressed | `hologram::CustomOpRegistry` |
| Persistence | not addressed | `hologram::HoloLoader` + `hologram::HoloWriter` |
| Post-lowering compile | done inside `lower()` | done by `hologram-ai` facade via `hologram::compile()` |
| Backend portability crate | `holo-ai-backend` | eliminated — `KvExecutor` handles this |
| Import rule | not stated | never import hologram subcrates directly |
| ADRs | 0002–0006 | 0002–0008 (add 0007, 0008) |

---

## 15. Files to Update

The following files from the initial planning package contain the fictional
type names and 14-crate layout and should be updated to match this document:

- `specs/projects/hologram-ai/crate-layout.md` — replace with 6-crate layout
- `specs/projects/hologram-ai/architecture.md` — update type names, crate names
- `specs/projects/hologram-ai/lowering.md` — update `LoweringOutput` type, op dispatch table
- `specs/projects/hologram-ai/runtime-model.md` — update `KvExecutor`, `BufferArena` usage
- `specs/prompts/hologram-ai/02-repo-bootstrap.md` — already updated by reviewer
- `specs/sprints/sprint-001` through `sprint-004` — already updated by reviewer
- `specs/adrs/0002` through `0006` — type name references need updating where they appear
- `specs/adrs/0007-hologram-ai-execution-layer.md` — new, needs writing
- `specs/adrs/0008-hologram-compiler-invoked-after-lowering.md` — new, needs writing
