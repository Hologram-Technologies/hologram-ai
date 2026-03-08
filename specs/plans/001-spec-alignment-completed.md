# Plan 001: Spec Alignment — Completed

**Status:** Done
**Date:** 2026-03-07
**ADRs:** ADR-0002, ADR-0006, ADR-0012, ADR-0015, ADR-0016

---

## Summary

Recrafted hologram-ai to match updated spec docs. Implemented the four
foundational phases required before MVP exit criteria can be met.

## What Changed

### Phase 0 — Remove InferenceSession (ADR-0016)

hologram-ai is a compiler, not a runtime.

- Deleted `InferenceSession` and `stream.rs`
- Renamed `session.rs` → `compiler.rs`
- CLI `run` is now a facade to `hologram run`
- For non-holo files, compile to temp `.holo` first, then delegate

**Files:** `compiler.rs`, `lib.rs`, `cli.rs` (hologram-ai crate)

### Phase 1 — Symbolic Shapes (ADR-0015)

- `DimExpr` enum: `Concrete`, `Var`, `Add`, `Sub`, `Mul`, `Div`, `Mod`,
  `CeilDiv`, `Max`, `Min`, `Dynamic`
- `DimVarTable`: variable registry with bounds, interning, concretization
- `ConstraintStore`: shape constraints (`DimEqual`, `BroadcastCompatible`,
  `Divisible`, `ProductEqual`)
- `Shape` changed from `SmallVec<[Dim; 6]>` to `SmallVec<[DimExpr; 4]>`
- `AiGraph` now has `dim_vars: DimVarTable`, `shape_constraints: ConstraintStore`
- Canonical var names: `batch`, `seq_len`, `vocab_size`, `hidden_dim`, etc.

**Files:** `shape/dim_expr.rs`, `shape/dim_var.rs`, `shape/constraint.rs`,
`shape/mod.rs`, `graph.rs` (hologram-ai-common crate)

### Phase 2 — Tokenizer Expansion (ADR-0012)

- `UnigramEncoder` — Viterbi dynamic programming segmentation
- `WordPieceEncoder` — greedy longest-prefix match with `##` prefix
- `TokenizerAlgorithm` enum: `Bpe`, `Unigram`, `WordPiece`
- `NativeTokenizer` with `EncoderBackend` dispatch
- HuggingFace JSON parsing for all 3 algorithms

**Files:** `unigram.rs`, `wordpiece.rs`, `config.rs`, `native.rs`, `lib.rs`
(hologram-ai-tokenizer crate)

### Phase 3 — GGUF Importer + Compiler Rework (ADR-0006)

- **GGUF parser** (`parser.rs`): v2/v3 binary format, metadata KV, tensor
  descriptors, alignment handling
- **Metadata extraction** (`metadata.rs`): `ArchParams` (arch, context_length,
  embedding_length, head counts, etc.) + `TokenizerMeta` from GGUF keys
- **LlamaArch** (`arch/llama.rs`): builds full transformer AiGraph
  (embed → N × [attn_norm → GQA → residual → ffn_norm → SwiGLU → residual] →
  final_norm → lm_head)
- **Compiler rework**: `CompiledModel` → `HoloArchive` with `CompileStats`
- **CLI**: added `inspect_gguf`, compile shows stats, info supports .gguf

**Files:** `parser.rs`, `metadata.rs`, `arch/mod.rs`, `arch/llama.rs`, `lib.rs`
(hologram-ai-gguf crate); `compiler.rs`, `cli.rs`, `lib.rs` (hologram-ai crate)

### Phase 4 — Shape Propagation Pass

- `ShapePropagation` pass: forward shape inference in topological order
- Handles: MatMul, elementwise with broadcasting, norms, concat, attention,
  reduce, embed, cast, FusedSwiGLU, RotaryEmbedding
- Added to `OptPipeline::mvp()` (runs first, before ConstantFolding and
  DeadNodeElimination)

**Files:** `opt/shape_prop.rs`, `opt/mod.rs`, `opt/pipeline.rs`
(hologram-ai-common crate)

## Test Count

60 tests passing across all workspace crates:
- hologram-ai-common: 23 (15 DimExpr/DimVar, 2 shape_prop, 3 graph, 2 dead_node, 1 const_fold)
- hologram-ai: 11 (download tests)
- hologram-ai-gguf: 2 (parser tests)
- hologram-ai-onnx: 2 (import tests)
- hologram-ai-quant: 8 (Q4_0/Q8_0 tests)
- hologram-ai-tokenizer: 14 (BPE, Unigram, WordPiece, NativeTokenizer tests)
