I don't have write permission for this file. Here is the filled-in template:

# Repository Layout — hologram-ai

## Top-Level Structure

```
hologram-ai/
AGENTS.md         # agent coding rules (holoarch-managed section + project rules)
CLAUDE.md         # Claude Code instructions
Cargo.toml        # workspace root (or single-crate manifest)
specs/            # all project documentation
src/ or crates/   # implementation code
```

---

## specs/ Layout

```
specs/
docs/             # project documentation (managed by holoarch)
plans/          # planning documents
adrs/           # Architecture Decision Records
SPRINT.md         # current sprint tracking
```

Do NOT create a top-level `docs/` directory. All docs go under `specs/docs/`.

---

## Source Layout

This repository uses a Cargo workspace with multiple crates under `crates/`.

```
crates/
├── hologram-ai-core/         # app-domain foundations: manifests, AI events, reducer, runner trait
│   └── src/
│       ├── lib.rs            # flat re-exports for the app-domain surface
│       ├── domain.rs         # KappaRef adapter, manifests, requests, outputs, AiView
│       ├── reducer.rs        # deterministic reduce(events) -> AiView
│       └── runner.rs         # ModelRunner trait + test-only deterministic echo runner
│
├── hologram-ai/              # public facade: compile, validate, download, CLI
│   └── src/
│       ├── lib.rs            # re-exports public API from all crates
│       ├── cli.rs            # CLI binary entrypoint (hologram-ai compile/inspect/validate/download)
│       ├── session.rs        # ModelCompiler, CompiledModel, ModelSource
│       ├── stream.rs         # streaming token output utilities
│       ├── validate.rs       # validation harness (compare to ONNX Runtime / llama.cpp)
│       └── download/         # HuggingFace download + ONNX conversion
│           ├── mod.rs
│           ├── hf_api.rs     # HuggingFace Hub API client
│           ├── convert.rs    # ONNX conversion via optimum-cli
│           └── progress.rs   # download progress reporting
│
├── hologram-ai-common/       # IR types, optimization passes, memory planner, lowering
│   └── src/
│       ├── lib.rs            # crate root, flat re-exports
│       ├── error.rs          # CommonError type
│       ├── ir/               # canonical AiGraph IR
│       │   ├── mod.rs
│       │   ├── graph.rs      # AiGraph, AiNode
│       │   ├── op.rs         # AiOp enum (all AI operations)
│       │   ├── node.rs       # node metadata
│       │   ├── param.rs      # AiParam (operator parameters)
│       │   ├── dtype.rs      # DType enum
│       │   └── shape.rs      # Shape, TensorInfo, TensorId, NodeId
│       ├── opt/              # optimization passes
│       │   ├── mod.rs
│       │   ├── pipeline.rs   # OptPipeline orchestration
│       │   ├── dead_node.rs  # dead node elimination
│       │   └── constant_fold.rs  # constant folding
│       ├── mem/              # memory planning
│       │   ├── mod.rs
│       │   └── planner.rs    # MemoryPlanner → KvCacheLayout
│       └── lower/            # AiGraph → hologram::Graph lowering
│           ├── mod.rs        # lower() entrypoint, LoweringOutput
│           ├── builder.rs    # graph construction helpers
│           ├── dispatch.rs   # op dispatch to GraphOp
│           └── custom_ops.rs # CustomOpRegistry population
│
├── hologram-ai-quant/        # quantization primitives (no IR dependency)
│   └── src/
│       ├── lib.rs            # crate root
│       ├── scheme.rs         # QuantScheme, QuantDescriptor, ScaleDtype
│       ├── q4_0.rs           # Q4_0 block layout and dequant
│       └── q8_0.rs           # Q8_0 block layout and dequant
│
├── hologram-ai-tokenizer/    # native tokenizer implementations
│   └── src/
│       ├── lib.rs            # Tokenizer trait
│       ├── config.rs         # TokenizerConfig, NormalizationConfig
│       ├── bpe.rs            # BPE tokenization core
│       ├── vocab.rs          # VocabTable, MergeRules
│       └── native.rs         # NativeTokenizer implementation
│
├── hologram-ai-onnx/         # ONNX importer (priority importer)
│   ├── build.rs              # prost-build for onnx.proto
│   ├── proto/
│   │   └── onnx.proto        # ONNX protobuf schema
│   └── src/
│       ├── lib.rs            # import_onnx(), import_onnx_path()
│       ├── error.rs          # OnnxError type
│       ├── dtype_map.rs      # ONNX dtype → DType mapping
│       ├── op_map.rs         # ONNX op → AiOp mapping
│       ├── tensor_map.rs     # tensor resolution
│       └── graph_builder.rs  # GraphProto → AiGraph construction
│
├── hologram-ai-gguf/         # GGUF importer (Phase 2)
│   └── src/
│       └── lib.rs            # import_gguf() — stub, not yet implemented
│
└── hologram-ai-ggml/         # GGML importer (Phase 3)
    └── src/
        └── lib.rs            # import_ggml() — stub, not yet implemented
```

### Crate Dependency Graph

```
hologram-ai-quant      → (no internal deps)
hologram-ai-common     → hologram-ai-quant, hologram
hologram-ai-tokenizer  → hologram-ai-common, hologram
hologram-ai-onnx       → hologram-ai-common
hologram-ai-gguf       → hologram-ai-common, hologram-ai-quant
hologram-ai-ggml       → hologram-ai-common, hologram-ai-quant
hologram-ai            → all internal crates + hologram (with compiler feature)
```

### Key Files by Function

| Function | Location |
|----------|----------|
| CLI binary | `crates/hologram-ai/src/cli.rs` |
| Public API facade | `crates/hologram-ai/src/lib.rs` |
| Canonical IR types | `crates/hologram-ai-common/src/ir/` |
| Optimization passes | `crates/hologram-ai-common/src/opt/` |
| Lowering to hologram | `crates/hologram-ai-common/src/lower/` |
| Quantization schemes | `crates/hologram-ai-quant/src/scheme.rs` |
| ONNX import | `crates/hologram-ai-onnx/src/lib.rs` |
| Tokenizer trait | `crates/hologram-ai-tokenizer/src/lib.rs` |
