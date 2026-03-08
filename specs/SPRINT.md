# Current Sprint — hologram-ai

## Sprint Goal

Enforce compiler-only boundary: remove runtime code, add KV-cache and
multi-graph lowering, produce pipeline archives with named entrypoints.

**Design principle:** hologram-ai is a compiler only (ADR-0016). It ships
zero runtime code. All kernels (GEMM, attention, norms, etc.) belong in
hologram base crate. CLI: `compile`, `info`, `download` — nothing else.

---

## In Progress

- [ ] Production-ready multi-modal `hologram run` (testing with real models)
  - `ModelMetaSection` (0x1002): `ModelKind` enum (TextLlm, Vision, Audio, etc.)
  - `MiniBpeEncoder`: lightweight BPE encoder in hologram-archive
  - `--prompt` / `--max-tokens`: autoregressive text generation loop
  - `--input-file SLOT:PATH`: load inputs from binary files
  - Typed output formatting: f32/f64/i32/i64 summary vs raw hex
  - dtype-aware token serialization (I32 vs I64 for `input_ids`)
  - Compiler embeds `ModelMetaSection` + `TokenizerSection` automatically
  - 728+ tests passing, zero clippy warnings across both repos

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
- [x] Tokenizer embedding: `--tokenizer` CLI flag (auto-detects `tokenizer.json` in model dir),
  `TokenizerSectionData::from_tokenizer_json()`, mirror `TokenizerSection` in hologram-archive
- [x] Output decoding: `hologram run` applies argmax + tokenizer decode when section present
- [x] `--prompt` flag: autoregressive text generation with `MiniBpeEncoder`
- [x] `ModelMetaSection` (0x1002): `ModelKind` enum, arch, capabilities
- [x] `--input-file`: load raw binary inputs from files
- [x] Typed output formatting: f32/f64/i32/i64 dtype-aware display
- [x] Compiler auto-embeds `ModelMetaSection` in compiled archives

See `specs/plans/001-spec-alignment-completed.md` and `specs/plans/002-mvp-remaining.md` for full details.

---

## Recently Unblocked

- **All ops are native FloatOp** — `FloatOp` expanded to 55 variants.
  `custom_ops.rs` deleted. Archives are fully self-describing.
- **LLM meta + tokenizer sections** — implemented locally with rkyv
  zero-copy serialization and `EmbeddableSection` trait. No longer blocked
  on hologram base crate.
- **Pipeline archives** — `PipelineWriter` used to bundle prefill + decode
  sub-archives. LLM detection heuristic in compiler selects pipeline vs
  single-archive path automatically.
- **TinyLlama end-to-end execution** — ONNX → .holo compilation and full
  execution of all 1612 nodes (1.1B params, 3.9 GiB weights) in ~215s
  (debug build). Required: `FloatOp::Transpose` with physical data
  permutation, `ShapeMap` for tensor shape tracking, N-D broadcasting
  (Expand), actual Reshape with shape tensor reading, i64/i32 auto-detection
  for ONNX shape constants.

## Still Blocked on hologram base crate

- **Shape metadata on graph edges** — hologram graphs have no per-edge
  shape/dtype, forcing shapes to be baked into closure captures
- **`KvExecutor::execute_layer()`** — does not exist; manual sub-archive
  extraction required

---

## Notes

- CLI: exactly 3 commands — `compile`, `info`, `download`
- ONNX importer path still works (single-archive, non-pipeline)
- GGUF importer supports `llama`, `mistral`, `codellama`, `tinyllama` arch names
- No backwards compatibility concerns — can break APIs freely
- Future extensibility: op decomposition (now), serializable op descriptors (Phase 3), WASM kernels (Phase 4+). See `specs/plans/003-op-extensibility.md`.
- Archive sections use rkyv for zero-copy access from memory-mapped files — no serde_json in archive path.
- `rkyv = "0.8"` added to workspace dependencies.
