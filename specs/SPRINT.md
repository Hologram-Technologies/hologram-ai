# Current Sprint — hologram-ai

## Sprint Goal

Complete MVP exit criteria: compile GGUF LLM to `.holo` pipeline archive with
named `lm.prefill`/`lm.decode` entrypoints, KV-cache layout, and embedded
metadata sections. Validate via `KvExecutor` golden test.

---

## In Progress

- [ ] KV-cache ops: add `AiOp::KvSlotWrite`/`KvSlotRead` to IR + shape prop rules
- [ ] KV-cache layout: `MemoryPlanner` computes real `KvCacheLayout` from arch params
- [ ] Multi-graph lowering: `LowerPhase` enum, prefill/decode graph emission
- [ ] `LoweringOutput` with `LayerDescriptor` and `TensorPort` entries
- [ ] Pipeline archive: `PipelineWriter` bundles prefill + decode sub-archives
- [ ] `LayerHeader` with named `lm.prefill`/`lm.decode` layers + tensor ports
- [ ] LLM meta section: `SECTION_LLM_META` (0x0011) embedding
- [ ] Tokenizer section: `SECTION_TOKENIZER` (0x1001) + `archive.rs` ConstantStore packing
- [ ] ConstantFolding: implement actual constant expression folding (currently no-op)
- [ ] CLI `validate` subcommand (stub: compile + verify archive loads)
- [ ] Integration test: logits golden test via `KvExecutor`

See `specs/plans/002-mvp-remaining.md` for full details and execution order.

---

## Done

- [x] Remove `InferenceSession` + structural cleanup (ADR-0016)
- [x] Symbolic shapes: `DimExpr` algebra, `DimVarTable`, `ConstraintStore` (ADR-0015)
- [x] Tokenizer expansion: Unigram (Viterbi), WordPiece, multi-algorithm dispatch (ADR-0012)
- [x] GGUF v2/v3 binary parser + metadata extraction (ADR-0006)
- [x] LlamaArch graph construction from GGUF tensors
- [x] Compiler rework: `HoloArchive` + `CompileStats` replacing `CompiledModel`
- [x] CLI: `inspect_gguf`, compile stats output, facade to `hologram run`
- [x] Shape propagation optimization pass (`ShapePropagation`)
- [x] 60 tests passing, zero clippy warnings

See `specs/plans/001-spec-alignment-completed.md` for full details.

---

## Blocked

- `LlmMetaSection`, `TokenizerSectionData`, `LlmModelType`, `DecodeLayers`
  are specified as living in hologram base crate (spec §2, §8) — may need to
  define local implementations until hologram adds them
  (see `specs/plans/hologram-types-needed.md`)
- `KvExecutor::execute_layer()` does not exist in hologram base crate — need
  manual pipeline sub-archive extraction for golden test
- `TensorPort`, `WeightDType`, `PipelineWriter` not in `hologram::` flat
  re-exports — use deep module paths as workaround

---

## Notes

- ONNX importer path still works (single-archive, non-pipeline)
- `CompiledModel` is kept as a type alias to `HoloArchive` for backward compat
- GGUF importer supports `llama`, `mistral`, `codellama`, `tinyllama` arch names
- Shape propagation handles: MatMul, elementwise broadcast, norms, concat,
  attention (MHA/GQA), reduce, embed, cast, FusedSwiGLU, RotaryEmbedding
- MVP exit criteria (from roadmap.md):
  - `hologram-ai compile tinyllama.gguf` → valid `.holo` pipeline archive
  - `LayerHeader` declares `lm.prefill` + `lm.decode` with correct tensor ports
  - `SECTION_LLM_META` reports correct `KvCacheLayout` for TinyLlama 1.1B
  - Top-1 logit matches llama.cpp reference on golden prompt
