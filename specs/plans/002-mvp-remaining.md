# Plan 002: MVP Remaining Work

**Status:** Planning
**Date:** 2026-03-07
**ADRs:** ADR-0002, ADR-0005, ADR-0006, ADR-0007, ADR-0012, ADR-0015, ADR-0016
**Roadmap ref:** MVP (Weeks 1–4) exit criteria

---

## Gap Analysis

The spec-alignment work (Plan 001) laid the foundations. The following gaps
remain before the MVP exit criteria in `specs/docs/roadmap.md` are met:

| Gap | Current State | Required (per specs/docs) |
|-----|---------------|---------------------------|
| Multi-graph lowering | Single graph per compile; `_kv_layout` unused | `LowerPhase` enum (Prefill/Decode); separate graphs with KV I/O |
| KV-cache ops | No `KvSlotWrite`/`KvSlotRead` in `AiOp` | KV-cache read/write ops in IR + lowering dispatch |
| KV-cache layout | `KvCacheLayout::none()` always | `MemoryPlanner` computes from arch params (n_layers, n_kv_heads, head_dim, max_seq_len) |
| Pipeline archive | Single flat archive via `HoloWriter` | `PipelineWriter` bundles prefill + decode sub-archives |
| LayerHeader tensor ports | Not emitted | Named `lm.prefill` / `lm.decode` with `TensorPort` entries per spec §3 |
| LLM meta section | Not emitted | `SECTION_LLM_META` (0x0011) — `LlmMetaSection` from hologram base crate |
| Tokenizer section | Not emitted | `SECTION_TOKENIZER` (0x1001) — `TokenizerSectionData` from hologram base crate |
| Tokenizer archive packing | No `archive.rs` in tokenizer crate | `ConstantStore` pack/unpack for vocab/merges/scores |
| ConstantFolding | No-op stub | Remove identity chains on constants |
| Lowering output | Returns `LoweringOutput` without `layer_descriptor` | Must return `LayerDescriptor` with tensor ports |
| CLI validate | Not implemented | `hologram-ai validate <model>` — compare against reference runtimes |
| Logits golden test | No integration test | Compile fixture → `KvExecutor` → compare logits |

---

## Work Items

### 1. KV-Cache Ops in IR

**Goal:** Add KV-cache read/write operations to the AI IR so that lowering
can emit graphs that interact with the KV-cache buffer.

**Changes:**
- Add `AiOp::KvSlotWrite { layer: usize }` — writes key/value to cache at
  current sequence position
- Add `AiOp::KvSlotRead { layer: usize }` — reads cached key/value up to
  `present_len` tokens
- GGUF arch builders (`llama.rs`) emit these ops within each attention block
- Shape propagation rules for KvSlotWrite/KvSlotRead

**Files:**
- `crates/hologram-ai-common/src/ir/op.rs`
- `crates/hologram-ai-common/src/opt/shape_prop.rs`
- `crates/hologram-ai-gguf/src/arch/llama.rs`

### 2. KV-Cache Layout Computation

**Goal:** `MemoryPlanner` computes a real `KvCacheLayout` from the model's
architecture parameters.

**Changes:**
- `MemoryPlanner::plan()` reads `arch_params` from `AiGraph` metadata
- Computes: `total_bytes = n_layers × 2 × n_kv_heads × head_dim × max_seq_len × dtype_size`
- Returns populated `KvCacheLayout` instead of `KvCacheLayout::none()`

**Files:**
- `crates/hologram-ai-common/src/mem/planner.rs`

### 3. Multi-Graph Lowering with LowerPhase

**Goal:** Lower `AiGraph` twice (prefill phase, decode phase) with proper KV
cache tensor ports on each graph.

**Changes:**
- Add `LowerPhase` enum: `Prefill`, `Decode`, `DecodeBucket(u64)`
- Prefill graph: `input_ids: [batch, seq_len]`, `kv_cache: [n_bytes]` in + out,
  `logits: [batch, vocab]` out
- Decode graph: `input_ids: [batch, 1]`, `present_len: [] u32`,
  `kv_cache: [n_bytes]` in + out, `logits: [batch, vocab]` out
- `KvSlotWrite` lowered to cache buffer writes at correct offset
- `KvSlotRead` lowered to cache buffer reads up to present_len
- `lower()` returns `LoweringOutput` with populated `layer_descriptor`
  containing correct `TensorPort` entries

**Signature per architecture spec §12:**
```rust
pub fn lower(
    graph: &AiGraph,
    kv_layout: &hologram::KvCacheLayout,
    phase: LowerPhase,
    opts: &LoweringOptions,
) -> Result<LoweringOutput>
```

**Files:**
- `crates/hologram-ai-common/src/lower/builder.rs`
- `crates/hologram-ai-common/src/lower/dispatch.rs`
- `crates/hologram-ai-common/src/lower/custom_ops.rs`

### 4. Pipeline Archive Construction

**Goal:** Compile prefill + decode into separate sub-archives, bundle via
`PipelineWriter`.

**Changes:**
- `ModelCompiler::compile()` detects LLM models (has `arch` in metadata)
- For LLMs: lower twice → `hologram::compile()` twice → build each sub-archive
  with its own `LayerHeader` → `PipelineWriter::new()
  .add_model("lm.prefill", prefill_holo)
  .add_model("lm.decode", decode_holo)
  .build()`
- Each sub-archive gets a `LayerHeader` with `LayerDescriptor` containing
  correct `TensorPort` entries (names, shapes, dtypes per spec §3)
- Non-LLM models: single archive with `"model.forward"` layer (current behavior)

**Files:**
- `crates/hologram-ai/src/compiler.rs`

**Cross-repo dependency:** `TensorPort`, `WeightDType`, `PipelineWriter` are
not flat re-exported from `hologram::`. Access via deep module paths as
workaround. See `specs/plans/hologram-types-needed.md`.

### 5. LLM Meta Section Embedding

**Goal:** Embed KV-cache layout and model metadata as `SECTION_LLM_META` (0x0011).

**Changes:**
- hologram base crate defines `LlmMetaSection` (per spec §8):
  ```rust
  pub struct LlmMetaSection {
      pub model_type: LlmModelType,
      pub kv_layout: KvCacheLayout,
      pub prefill_layer: LayerId,
      pub decode_layers: DecodeLayers,
  }
  ```
- If `LlmMetaSection` is not yet in hologram base crate, define a local
  `EmbeddableSection` implementation in hologram-ai-common using
  `SECTION_CUSTOM_BASE + 0x11 = 0x1011` as interim section kind
- `ModelCompiler` populates and embeds in each sub-archive

**Files:**
- `crates/hologram-ai-common/src/sections/llm_meta.rs` (new)
- `crates/hologram-ai-common/src/sections/mod.rs` (new)
- `crates/hologram-ai/src/compiler.rs`

**Cross-repo dependency:** `LlmMetaSection`, `LlmModelType`, `DecodeLayers`
should be defined in hologram base crate (per spec §2, §8). See
`specs/plans/hologram-types-needed.md`.

### 6. Tokenizer Section Embedding

**Goal:** Embed tokenizer data from GGUF metadata into `SECTION_TOKENIZER` (0x1001).

**Changes:**
- Add `archive.rs` to tokenizer crate with `ConstantStore` pack/unpack:
  - `pack_tokenizer(tokenizer: &NativeTokenizer) -> TokenizerSectionData`
  - `unpack_tokenizer(data: &TokenizerSectionData) -> NativeTokenizer`
- Serialize: vocab tokens, scores, special token IDs, algorithm type, merges
- `ModelCompiler` extracts tokenizer from GGUF → packs into section → embeds
  in prefill sub-archive

**Files:**
- `crates/hologram-ai-tokenizer/src/archive.rs` (new)
- `crates/hologram-ai/src/compiler.rs`

**Cross-repo dependency:** `TokenizerSectionData` should be defined in hologram
base crate (per spec §2, §8). See `specs/plans/hologram-types-needed.md`.

### 7. ConstantFolding Implementation

**Goal:** Remove identity chains where a `Constant` feeds into `Identity`.

**Changes:**
- Scan for `Identity` nodes whose sole input is a `Constant` node
- Replace with direct output aliasing
- Remove dead constant nodes
- Additional: fold `Reshape` of constant tensors, `Cast` of constant scalars

**Files:**
- `crates/hologram-ai-common/src/opt/constant_fold.rs`

### 8. Lowering Output with LayerDescriptor

**Goal:** `LoweringOutput` includes a fully populated `LayerDescriptor` with
tensor ports, matching spec §12.

**Changes:**
- `LoweringOutput` gets `layer_descriptor: hologram::LayerDescriptor` field
  with `name`, `entrypoint`, `inputs: Vec<TensorPort>`, `outputs: Vec<TensorPort>`
- `layer_name: String` field for the archive layer name
  (e.g., `"lm.prefill"`, `"lm.decode"`, `"lm.decode.128"`)

**Files:**
- `crates/hologram-ai-common/src/lower/builder.rs`

### 9. CLI: validate subcommand

**Goal:** `hologram-ai validate <model>` compiles model, executes via
`KvExecutor`, compares to reference runtime output.

**Changes:**
- Add `Command::Validate` variant to CLI
- Compile model → load archive → `KvExecutor::execute_with_weights()` →
  compare output shape and values against golden fixture
- Stub initially: just compile + verify archive loads

**Files:**
- `crates/hologram-ai/src/cli.rs`
- `crates/hologram-ai/src/validate.rs` (if exists, update; else create)

### 10. Integration Test: Logits Golden Test

**Goal:** Compile a model fixture, run via `KvExecutor`, compare logits.

**Changes:**
- Use a small ONNX fixture (already exists in test data)
- Compile → load via `load_from_bytes` → `KvExecutor::execute_with_weights`
- Assert output shape and value ranges
- For pipeline archives: verify `PipelineHeader` entries, load sub-archive by
  name, verify `LayerHeader` tensor ports

**Files:**
- `tests/integration/golden_logits.rs` (new)

---

## Execution Order

```
1. KV-cache ops in IR (AiOp::KvSlotWrite/Read)
     ↓
2. KV-cache layout computation (MemoryPlanner)
     ↓
3. Multi-graph lowering (LowerPhase) + LoweringOutput with LayerDescriptor
     ↓
4. Pipeline archive construction (PipelineWriter + LayerHeader)
     ↓
5. LLM meta section ──────┐
                           │ (parallel)
6. Tokenizer section ─────┘
     ↓
7. ConstantFolding
     ↓
8. CLI validate + integration test
```

---

## Cross-Repo Dependencies

The following types are specified as living in `hologram` base crate (per
architecture spec §2 and §8). Their availability affects implementation:

| Type | Needed by | Status | Workaround |
|------|-----------|--------|------------|
| `LlmMetaSection` | §5 LLM meta section | Check if exists | Define local `EmbeddableSection` impl |
| `LlmModelType` | §5 LLM meta section | Check if exists | Local enum |
| `DecodeLayers` | §5 LLM meta section | Check if exists | Local enum |
| `TokenizerSectionData` | §6 Tokenizer section | Check if exists | Local `EmbeddableSection` impl |
| `BucketSelector` | Phase 2 bucketed compilation | Not needed for MVP | — |
| `TensorPort` (flat re-export) | §4 Pipeline archive | Not re-exported | Use deep path |
| `WeightDType` (flat re-export) | §4 Pipeline archive | Not re-exported | Use deep path |
| `PipelineWriter` (flat re-export) | §4 Pipeline archive | Not re-exported | Use deep path |
| `KvExecutor::execute_layer()` | §10 Integration test | Does not exist | Manual sub-archive extraction |
| `SECTION_LLM_META` constant | §5 | May not be exported | Define locally |
| `SECTION_TOKENIZER` constant | §6 | May not be exported | Define locally |

See `specs/plans/hologram-types-needed.md` for the full change request.

---

## MVP Exit Criteria (from roadmap.md)

- [ ] `hologram-ai compile tinyllama.gguf` produces a valid `.holo` archive
- [ ] Archive `LayerHeader` declares `lm.prefill` and `lm.decode` with correct tensor ports
- [ ] `SECTION_LLM_META` reports correct `KvCacheLayout` for TinyLlama 1.1B
- [ ] Calling `KvExecutor::execute_layer("lm.prefill", ...)` yields logits of correct shape
- [ ] Top-1 logit matches llama.cpp reference (greedy) on golden prompt
- [ ] All unit tests pass on `aarch64-apple-darwin` and `x86_64-unknown-linux-gnu`
