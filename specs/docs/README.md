# hologram-ai — Project Overview

`hologram-ai` is a Rust-first, inference-first **compiler** that imports AI
model artifacts (ONNX, GGUF, GGML) and compiles them into `.holo` archives
executable on the Hologram architecture. It is a compiler only — it ships
zero runtime code (see ADR-0016).

---

## What it is

A compiler pipeline. Its job is to:

1. **Ingest** foreign model artifacts (ONNX protobuf, GGUF binary, GGML checkpoint)
2. **Normalize** them into a canonical `AiGraph` intermediate representation
3. **Optimize** the graph via compile-time passes (fusion, folding, shape propagation)
4. **Plan memory** — resolve tensor lifetimes, buffer aliasing, KV-cache sizing
5. **Lower** the graph into a `hologram::Graph` with `FloatOp` tensor operations
6. **Compile** via `hologram::compile()` into an executable plan
7. **Emit** `.holo` archives — single-graph or multi-component pipelines

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

## Multi-Component Pipeline Archives

hologram-ai compiles multi-component models into a single `.holo` pipeline
archive. Each component is independently compiled into a sub-archive, then
bundled via `PipelineWriter` with a `MetaSection` describing component roles,
weight groups, and data flow connections.

**Supported model topologies:**

| Topology | Components | Example |
|----------|-----------|---------|
| LLM (current) | prefill + decode | TinyLlama, Llama 3 |
| Encoder-decoder | encoder + decoder | Whisper, T5 |
| CALM | autoencoder + backbone + head + decoder | CALM (next-vector prediction) |
| Stable Diffusion | VAE encoder + UNet + VAE decoder | SD 1.5, SDXL |
| MoE | router + N experts | Mixtral |

**Key types:**

- `ComponentSpec` — specification for a single component (name, role, weight group,
  graph, memory plan, lowering phase)
- `MetaSection` — pipeline metadata section describing N components and their
  connections (embedded in the `.holo` wrapper via `PipelineWriter::add_section`)
- `ComponentRole` — Prefill, Decode, Encoder, Decoder, Backbone, GenerativeHead,
  Forward, Custom
- `OptProfile` — Llm (full MVP passes) or Generic (shape propagation + constant folding only)

**Compilation flow:**

```
N × ModelSource → import → optimize (per OptProfile) → concretize → lower
  → compile_one_component() → N sub-archives
  → compile_components() → PipelineWriter + MetaSection → single .holo
```

See [Plan 021](../plans/021-multi-component-archives.md) for the full design.

---

## Phases

| Phase | Scope |
|-------|-------|
| **MVP** | GGUF TinyLlama on CPU, single forward pass, core IR + lowering |
| **Phase 2** | ONNX encoder/decoder, streaming token generation, KV-cache |
| **Phase 3** | Metal backend, quantized kernels, GGML migration path |
| **Phase 4** | Multi-backend, CUDA, WebGPU, multi-GPU sharding |
| **Phase 5** | Multi-component pipelines, weight deduplication, generic multi-ONNX |

---

## Where to read next

| Topic | File |
|-------|------|
| Full architecture | [architecture.md](architecture.md) |
| CLI specification | [cli.md](cli.md) |
| Import pipeline | [import-pipeline.md](import-pipeline.md) |
| Lowering design | [lowering.md](lowering.md) |
| Tokenizer architecture | [tokenizer.md](tokenizer.md) |
| Runtime model | [runtime-model.md](runtime-model.md) |
| KV-cache & paged attention | [runtime-model.md — KV-Cache](runtime-model.md#kv-cache) |
| Multi-component pipelines | [Plan 021](../plans/021-multi-component-archives.md) |
| Repository layout | [repository-layout.md](repository-layout.md) |
| Testing strategy | [testing.md](testing.md) |
| Roadmap | [roadmap.md](roadmap.md) |

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
