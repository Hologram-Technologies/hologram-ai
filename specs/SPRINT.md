# Current Sprint — hologram-ai

## Sprint Goal

**Execution Conformance Testing:** Validate the full compile → lower → execute
pipeline, not just individual kernels. Catch shape propagation errors, dynamic
dim sentinel failures, and dtype confusion at compile time and via ORT comparison.
See `specs/plans/006-execution-conformance.md`.

**Previous sprint (complete):** Kernel Conformance Testing (Plan 005) — 187+ tests.
See `specs/plans/005-conformance-testing.md`.

**Design principle:** hologram-ai is a compiler only (ADR-0016). It ships
zero runtime code. All kernels belong in hologram base crate.
CLI: `compile`, `info`, `download` — nothing else.

---

## In Progress

### Execution Conformance Testing (Plan 006)

#### Phase 1: Layer E — Compile-Time Shape Consistency (hologram-ai only)
- [x] Create `shape_consistency.rs` validation pass in hologram-ai-common (8 unit tests)
- [x] Integrate into `compiler.rs` after concretization, before lowering
- [x] Add tests: param/shape mismatch, MatMul k-dim mismatch, dynamic dim, zero-product, Gemm trans_b
- [x] Run on TinyLlama ONNX: found 52 issues (1 MatMul k-dim mismatch, ~50 zero-product shapes from unresolved seq_len=0 sentinels)

#### Phase 2: Layer D — Executor Intermediate Capture (hologram base repo)
- [x] Add `BufferArena::snapshot()` in `arena.rs` (non-destructive, `#[cfg(feature = "profile")]`)
- [x] Add `ShapeMap::snapshot()` (feature-gated)
- [x] Add `KvExecutor::execute_with_intermediates()` + `IntermediateCapture` struct (feature-gated)

#### Phase 3: Layer D — Execution Conformance Harness (hologram-ai repo)
- [x] Add `compile_with_debug_info()` to `ModelCompiler` (ONNX name → NodeId map)
- [ ] Add ORT intermediate tensor capture in `ort_runner.rs`
- [ ] Add `exec_comparator.rs` — node-by-node comparison with tolerances
- [ ] Add `tests/exec_conformance.rs` — multi-node ONNX integration tests

---

## Previous Sprint (Complete): Kernel Conformance Testing (Plan 005)

### Shape Propagation (33 tests)
- [x] Shape conformance for all major op categories (onnx_conformance.rs)
- [x] Conv2d, MatMul, Gemm, MaxPool, GlobalAvgPool, reductions, TopK, etc.
- [x] Subgraph shape propagation (If, Loop)

### Layer A — hologram-exec inline tests (87 tests, pure Rust)
- [x] Known-answer, property, numerical stability tests for all major FloatOp variants
- [x] Exhaustive match ensuring new FloatOp variants require tests
- [x] `tests/float_conformance.rs` integration test

### Layer B — hologram-ai-conformance (reference cross-validation)
- [x] Tolerance, comparator, reference modules
- [x] 31 cross-validation tests (op_conformance.rs)
- [x] ORT runner: 17 ONNX tests (8 unary, 4 binary, 2 softmax, 1 matmul, 2 gemm)
- [x] ORT composite models (RmsNorm, LayerNorm)

### Layer C — Validate CLI + quantization
- [x] `hologram-ai validate --model <path>` (compilation-level validation)
- [x] Quant Tier 1 + Tier 2 (golden vectors)
- [x] CI Tier 1/2/3 (nightly workflow, GitHub Actions)

---

## Previous Sprint (Complete): ONNX Last Mile

### Phase 1: Vision-Critical Ops
- [x] Add 9 AiOp variants: Conv, ConvTranspose, MaxPool, AveragePool, GlobalAveragePool, Resize, Pad, InstanceNorm, LRN
- [x] Add ONNX op mappings + `attr_s()` accessor to OpContext
- [x] Add shape propagation rules (Conv/Pool formula, Resize, Pad, etc.)
- [x] Add data propagation match arms
- [x] Add dynamic param resolution for Pad/Resize (opset 11+ inputs)
- [x] Add lowering dispatch entries → FloatNeedsShape (FloatOp variants added to hologram base)
- [x] Add resolve_op strategy arms for Conv2d, ConvTranspose, MaxPool2d, AvgPool2d, GlobalAvgPool, Resize, Pad, InstanceNorm, LRN

### Phase 2: Utility Ops
- [x] Add 12 AiOp variants: ReduceProd, ReduceL1, ReduceL2, TopK, ScatterND, CumSum, NonZero, OneHot, DepthToSpace, SpaceToDepth, Compress, ReverseSequence
- [x] Add ONNX mappings + quantization integration (QuantizeLinear, DequantizeLinear)
- [x] Add shape propagation rules for utility ops (reductions, TopK, ScatterND, NonZero, OneHot, DepthToSpace, SpaceToDepth, Compress)
- [x] Add lowering dispatch entries → FloatNeedsShape (FloatOp variants added to hologram base)
- [x] Add resolve_op strategy arms for ReduceProd, TopK, ScatterND, CumSum, NonZero, Compress, ReverseSequence
- [x] Add OpDecomposition pass: ReduceL1→Abs+ReduceSum, ReduceL2→Mul+ReduceSum+Sqrt, DepthToSpace/SpaceToDepth→Reshape+Transpose+Reshape

### Phase 3: Proto/Type Gaps
- [x] Add F64 dtype + ONNX type 11 mapping
- [x] Add INT16 dtype
- [x] Add widening casts for UINT16→INT32, UINT32→INT64, UINT64→INT64
- [x] Add opset version validation (parse opset_import, enforce max_opset, store in metadata)
- [x] F64→F32 and INT16→I32 lowering at weight serialization and FloatDType mapping
- [x] Document and handle optional input semantics

### Phase 4: Subgraph Support (If/Loop/Scan)
- [x] Add `subgraphs: HashMap<String, AiGraph>` to AiGraph
- [x] Add AiOp::If, Loop, Scan variants
- [x] Add `attr_g()` graph attribute accessor + ONNX If/Loop/Scan op mappings
- [x] Add recursive ONNX import with subgraph key rewriting
- [x] Add optimization pass recursion into subgraphs
- [x] Add lowering to hologram's native SubgraphDef + CallSubgraph (compile-time flattening)

### Phase 5: Long-Tail + Conformance
- [x] Map remaining niche ops to Opaque with warnings (RNG, ML, linear algebra, sequence, optional)
- [x] Validate recursion into subgraphs
- [x] Verify multi-output ops (TopK, Split, BatchNorm training)
- [x] ONNX conformance test runner (node test suite)

---

## Done

- [x] Remove `InferenceSession` + structural cleanup (ADR-0016)
- [x] Symbolic shapes: `DimExpr` algebra, `DimVarTable`, `ConstraintStore` (ADR-0015)
- [x] Tokenizer expansion: Unigram (Viterbi), WordPiece, multi-algorithm dispatch (ADR-0012)
- [x] GGUF v2/v3 binary parser + metadata extraction (ADR-0006)
- [x] LlamaArch graph construction from GGUF tensors
- [x] Compiler rework: `HoloArchive` + `CompileStats` replacing `CompiledModel`
- [x] CLI: `inspect_gguf`, compile stats output
- [x] Shape propagation optimization pass (`ShapePropagation`)
- [x] Delete `Run` CLI command — users call `hologram run` directly
- [x] 60 tests passing, zero clippy warnings
- [x] Native `FloatOp` in hologram base crate (55 variants, kernels, dispatch, CLI inspect)
- [x] Lowering emits `GraphOp::Float(FloatOp::...)` for ALL ops (zero custom ops remaining)
- [x] Deleted `custom_ops.rs` — all 446 lines removed, no `CustomHandler` closures
- [x] Removed `CustomOpRegistry` from `LoweringOutput` — lowering is pure native ops
- [x] Op extensibility plan documented (`specs/plans/003-op-extensibility.md`)
- [x] KV-cache ops: `AiOp::KvSlotWrite`/`KvSlotRead` in IR, dispatch, shape propagation
- [x] KV-cache layout: `MemoryPlanner` computes real `KvCacheLayout` from arch metadata
- [x] Multi-graph lowering: `LowerPhase` enum (Prefill/Decode/Forward), phase-aware `lower()`
- [x] Pipeline archive: `PipelineWriter` bundles prefill + decode sub-archives for LLMs
- [x] LLM meta section: `LlmMetaSection` with rkyv zero-copy serialization (`SECTION_LLM_META` 0x1011)
- [x] Tokenizer section: `TokenizerSectionData` with rkyv zero-copy serialization (`SECTION_TOKENIZER` 0x1001)
- [x] ConstantFolding: identity elimination, reshape-of-constant folding, dead constant removal
- [x] 67 tests passing, zero clippy warnings
- [x] Shape-tracked execution: `ShapeMap`, `FloatOp::Transpose` with physical permutation,
  actual Reshape (reads shape tensor), N-D broadcasting (Expand), i64/i32 shape auto-detection
- [x] TinyLlama 1.1B end-to-end: ONNX → .holo → execute all 1612 nodes (~215s debug build)
- [x] Tokenizer embedding: `--tokenizer` CLI flag, `TokenizerSectionData::from_tokenizer_json()`
- [x] Output decoding: `hologram run` applies argmax + tokenizer decode when section present
- [x] `--prompt` flag: autoregressive text generation with `MiniBpeEncoder`
- [x] `ModelMetaSection` (0x1002): `ModelKind` enum, arch, capabilities
- [x] `--input-file`: load raw binary inputs from files
- [x] Typed output formatting: f32/f64/i32/i64 dtype-aware display
- [x] Compiler auto-embeds `ModelMetaSection` in compiled archives
- [x] ONNX shape oracle: seed shapes from ValueInfoProto, settled-shape protection
- [x] RmsNorm fusion pass: Pow→ReduceMean→Add→Sqrt→Reciprocal→Mul → AiOp::RmsNorm
- [x] Multi-level DataProp: re-materialization for transitive shape dependencies
- [x] Seq_len sentinel: dynamic dims use 0-sentinel, resolved at runtime
- [x] Inf/NaN diagnostic: scan compiled params for broken scale factors
- [x] Dynamic param resolution: Pad/Resize (opset 11+ inputs), Clip (opset 11+ min/max)
- [x] Optional input semantics: documented pattern, Clip min/max resolved from constant inputs
- [x] Multi-output ops: TopK (values+indices dtype), Split (N outputs), BatchNorm (training 5 outputs)
- [x] ONNX conformance test suite: 29 shape-propagation tests covering all op categories
- [x] 147 tests passing, zero clippy warnings
- [x] Subgraph lowering: If (compile-time flatten + Where), Loop (compile-time unroll), Scan (CallSubgraph fallback)
- [x] GraphBuilder.flatten_registered_subgraph() for compile-time subgraph inlining
- [x] DispatchTarget::Subgraph variant routes If/Loop/Scan through subgraph lowering path
- [x] 4 subgraph lowering tests: If with both branches, If then-only, Loop known trip count, Loop zero trip

See `specs/plans/001-spec-alignment-completed.md`, `specs/plans/002-mvp-remaining.md`,
and `specs/plans/004-onnx-last-mile.md` for full details.

---

## Still Blocked on hologram base crate

- **Shape metadata on graph edges** — hologram graphs have no per-edge
  shape/dtype, forcing shapes to be baked into closure captures
- **`KvExecutor::execute_layer()`** — does not exist; manual sub-archive
  extraction required
- ~~**Vision FloatOp variants**~~ — DONE: Conv2d, ConvTranspose, MaxPool2d, AvgPool2d, GlobalAvgPool, Resize, PadOp, InstanceNorm, LRN added
- ~~**Utility FloatOp variants**~~ — DONE: ReduceProd, TopK, ScatterND, CumSum, NonZero, Compress, ReverseSequence added
- **Vision/utility runtime kernels** — FloatOp variants exist but dispatch returns `UnsupportedOp` (stub); kernels not yet implemented
- ~~**Subgraph lowering**~~ — DONE: compile-time flattening covers If/Loop; dynamic Loop/Scan falls back to `CallSubgraph` (needs runtime dispatch)
- **`LayerEntrypoint::Subgraph(u32)` runtime** — declared but not implemented;
  needed for dynamic Loop/Scan control flow at runtime

---

## Notes

- CLI: exactly 3 commands — `compile`, `info`, `download`
- ONNX importer path still works (single-archive, non-pipeline)
- GGUF importer supports `llama`, `mistral`, `codellama`, `tinyllama` arch names
- No backwards compatibility concerns — can break APIs freely
- Future extensibility: op decomposition (now), serializable op descriptors (Phase 3), WASM kernels (Phase 4+). See `specs/plans/003-op-extensibility.md`.
- Archive sections use rkyv for zero-copy access from memory-mapped files.
- `rkyv = "0.8"` added to workspace dependencies.
