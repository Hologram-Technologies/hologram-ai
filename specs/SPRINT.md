# Current Sprint — hologram-ai

## Sprint Goal

**ONNX Last Mile:** Full op coverage + subgraph support so any ONNX model
imports and compiles. See `specs/plans/004-onnx-last-mile.md`.

**Design principle:** hologram-ai is a compiler only (ADR-0016). It ships
zero runtime code. All kernels belong in hologram base crate.
CLI: `compile`, `info`, `download` — nothing else.

---

## In Progress

### Phase 1: Vision-Critical Ops
- [ ] Add 9 AiOp variants: Conv, ConvTranspose, MaxPool, AveragePool, GlobalAveragePool, Resize, Pad, InstanceNorm, LRN
- [ ] Add ONNX op mappings + `attr_s()` accessor to OpContext
- [ ] Add dynamic param resolution for Pad/Resize (opset 11+ inputs)
- [ ] Add shape propagation rules (Conv/Pool formula, Resize, Pad, etc.)
- [ ] Add data propagation match arms
- [ ] Add lowering dispatch entries (Unsupported until hologram base adds FloatOp)

### Phase 2: Utility Ops
- [ ] Add 12 AiOp variants: ReduceProd, ReduceL1, ReduceL2, TopK, ScatterND, CumSum, NonZero, OneHot, DepthToSpace, SpaceToDepth, Compress, ReverseSequence
- [ ] Add ONNX mappings + quantization integration (QuantizeLinear, DequantizeLinear, MatMulInteger)
- [ ] Add shape propagation + lowering (decompositions where possible)

### Phase 3: Proto/Type Gaps
- [ ] Add F64 dtype + ONNX type 11 mapping
- [ ] Add widening casts for UINT16/INT16/UINT32/UINT64
- [ ] Add opset version validation + version-aware op semantics
- [ ] Document and handle optional input semantics

### Phase 4: Subgraph Support (If/Loop/Scan)
- [ ] Add `subgraphs: HashMap<String, AiGraph>` to AiGraph
- [ ] Add AiOp::If, Loop, Scan variants
- [ ] Add recursive ONNX import with lexical scope capture
- [ ] Add subgraph shape propagation + optimization pass recursion
- [ ] Add lowering to hologram's native SubgraphDef + CallSubgraph

### Phase 5: Long-Tail + Conformance
- [ ] Map remaining niche ops to Opaque with warnings
- [ ] Verify multi-output ops (TopK, Split, BatchNorm training)
- [ ] ONNX conformance test runner (node test suite)

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

See `specs/plans/001-spec-alignment-completed.md`, `specs/plans/002-mvp-remaining.md`,
and `specs/plans/004-onnx-last-mile.md` for full details.

---

## Still Blocked on hologram base crate

- **Shape metadata on graph edges** — hologram graphs have no per-edge
  shape/dtype, forcing shapes to be baked into closure captures
- **`KvExecutor::execute_layer()`** — does not exist; manual sub-archive
  extraction required
- **Vision FloatOp variants** — Conv2d, MaxPool2d, AvgPool2d, GlobalAvgPool,
  Resize, Pad needed for Phase 1 lowering
- **Utility FloatOp variants** — TopK, CumSum, NonZero, ScatterND, ReduceProd
  needed for Phase 2 lowering
- **`LayerEntrypoint::Subgraph(u32)` runtime** — declared but not implemented;
  needed for Phase 4 dynamic control flow

---

## Notes

- CLI: exactly 3 commands — `compile`, `info`, `download`
- ONNX importer path still works (single-archive, non-pipeline)
- GGUF importer supports `llama`, `mistral`, `codellama`, `tinyllama` arch names
- No backwards compatibility concerns — can break APIs freely
- Future extensibility: op decomposition (now), serializable op descriptors (Phase 3), WASM kernels (Phase 4+). See `specs/plans/003-op-extensibility.md`.
- Archive sections use rkyv for zero-copy access from memory-mapped files.
- `rkyv = "0.8"` added to workspace dependencies.
