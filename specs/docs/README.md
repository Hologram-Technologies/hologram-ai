The two versions are identical - there are no differences between the ARCH VERSION and the SUBPROJECT VERSION. The merged result is the same as both inputs:

# hologram-ai — Project Overview

`hologram-ai` is a Rust-first, inference-first compiler and runtime integration
layer that imports AI model artifacts (ONNX, GGUF, GGML) and makes them
executable on the Hologram architecture.

---

## What it is

A compiler pipeline and runtime session library. Its job is to:

1. **Ingest** foreign model artifacts (ONNX protobuf, GGUF binary, GGML checkpoint)
2. **Normalize** them into a canonical `AiGraph` intermediate representation
3. **Optimize** the graph via compile-time passes (fusion, folding, shape propagation)
4. **Plan memory** — resolve tensor lifetimes, buffer aliasing, KV-cache sizing
5. **Lower** the graph into a `hologram::Graph + ExecutionSchedule`
6. **Execute** inference via the Hologram execution backend
7. **Stream** autoregressive token output for LLM workloads

---

## What it is not

- Not a wrapper around ONNX Runtime, llama.cpp, or any other inference engine
- Not an AI application framework
- Not a training system (inference-first; training future scope)
- Not a fork of any existing system

Reference runtimes (ONNX Runtime, llama.cpp) are used only for **validation
and testing**, never as the execution substrate.

---

## Relationship to hologram

```
hologram-ai
  └── depends on → hologram  (graph execution, memory, runtime, artifacts)
```

`hologram` remains AI-agnostic and sandbox-agnostic. All AI-specific
concerns — model formats, quantization, attention semantics, KV-cache,
token generation — live in `hologram-ai`.

See [ADR-0001](../../adrs/0001-repo-boundary.md) for the general repo boundary
policy and [ADR-0002](../../adrs/0002-hologram-ai-canonical-ir.md) for the
hologram-ai specific boundary.

---

## Phases

| Phase | Scope |
|-------|-------|
| **MVP** | GGUF TinyLlama on CPU, single forward pass, core IR + lowering |
| **Phase 2** | ONNX encoder/decoder, streaming token generation, KV-cache |
| **Phase 3** | Metal backend, quantized kernels, GGML migration path |
| **Phase 4** | Multi-backend, CUDA, WebGPU, multi-GPU sharding |

---

## Where to read next

| Topic | File |
|-------|------|
| Full architecture | [architecture.md](architecture.md) |
| CLI specification | [cli.md](cli.md) |
| Crate layout | [crate-layout.md](crate-layout.md) |
| Import pipeline | [import-pipeline.md](import-pipeline.md) |
| Lowering design | [lowering.md](lowering.md) |
| Tokenizer architecture | [tokenizer.md](tokenizer.md) |
| Runtime model | [runtime-model.md](runtime-model.md) |
| Testing strategy | [testing-strategy.md](testing-strategy.md) |
| Roadmap | [roadmap.md](roadmap.md) |
| Risks | [risks.md](risks.md) |
| Research baseline | [../../research/2026-03-06-hologram-ai-architecture.md](../../research/2026-03-06-hologram-ai-architecture.md) |

---

## ADRs

| Number | Decision |
|--------|---------|
| [0002](../../adrs/0002-hologram-ai-canonical-ir.md) | Canonical AI IR above raw Hologram graph |
| [0003](../../adrs/0003-hologram-ai-import-boundary.md) | Format-specific logic terminates at importer boundary |
| [0004](../../adrs/0004-hologram-ai-quantization-model.md) | Quantization is first-class in AiGraph |
| [0005](../../adrs/0005-hologram-ai-runtime-boundary.md) | Session owns plan + KV-cache; hologram owns execution |
| [0006](../../adrs/0006-hologram-ai-mvp-scope.md) | MVP = GGUF + CPU + single-pass inference |
| [0007](../../adrs/0007-hologram-ai-execution-layer.md) | Execution layer maps to real hologram types |
| [0008](../../adrs/0008-hologram-compiler-invoked-after-lowering.md) | hologram-compiler invoked after lowering |
| [0009](../../adrs/0009-cli-compile-delegates-to-hologram.md) | CLI compile delegates to hologram binary |
| [0010](../../adrs/0010-huggingface-download-onnx-conversion.md) | HuggingFace download and ONNX conversion |
| [0012](../../adrs/0012-hologram-ai-native-tokenizer.md) | Hologram-native tokenizer via ConstantStore and .holo archives |
