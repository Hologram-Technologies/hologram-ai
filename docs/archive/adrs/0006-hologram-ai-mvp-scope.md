# ADR-0006: MVP scope is GGUF import + CPU backend + single forward pass

- Status: Accepted
- Date: 2026-03-06
- Owners: Architecture

---

## Context

`hologram-ai` is a new project with many surface areas: three format importers,
a canonical IR, optimization passes, memory planning, lowering, session
management, streaming, and multiple backend targets.

Attempting to implement all of this simultaneously is high-risk. An MVP must
establish the core pipeline before adding breadth.

The goal of the MVP is to validate the complete pipeline from model file to
correct inference output — not to support all formats or all features.

---

## Decision

The MVP is:

1. **GGUF importer only** — supports LLaMA-family models (llama, llama2, llama3)
2. **Q4_0 quantization only** — the most common GGUF quant format in the wild
3. **CPU backend only** — via `hologram-exec` (`KvExecutor`); no Metal, CUDA, or WebGPU
4. **Single forward pass** — `InferenceSession::run()` only; no KV-cache
5. **No streaming** — `generate()` and `TokenStream` are Phase 2
6. **No ONNX importer** — Phase 2
7. **No GGML importer** — Phase 2

Exit criteria:
- Import TinyLlama 1.1B Q4_0 from GGUF without error
- Single forward pass produces logits of correct shape `[1, vocab_size]`
- Top-1 greedy token matches llama.cpp reference for a fixed prompt
- All unit tests pass on `aarch64-apple-darwin` and `x86_64-unknown-linux-gnu`

---

## Consequences

**Positive:**
- MVP is achievable in 4 weeks by a single engineer
- Validates the entire pipeline (import → IR → opt → mem → lower → execute)
  end-to-end before adding breadth
- GGUF + Q4_0 + LLaMA covers the highest-value use case first
- CPU backend is always available; no GPU toolchain dependencies in MVP

**Negative:**
- ONNX models are not supported at MVP; this delays encoder-model use cases
- No streaming at MVP; the demo is less impressive than llama.cpp out of the box
- Single forward pass means no conversational demo in MVP

**Neutral:**
- The architecture is designed for the full scope; the MVP is a subset of the
  code, not a simplified version of the architecture

---

## Alternatives Considered

**Include ONNX in MVP**
Rejected. ONNX adds significant complexity (protobuf, 200+ ops, shape inference,
external data). It distracts from the core pipeline bring-up. GGUF is a better
MVP target because it exercises quantization early.

**Include KV-cache in MVP**
Rejected. KV-cache adds stateful session complexity on top of a pipeline
that hasn't been validated yet. Single-pass correctness is a prerequisite
for multi-pass correctness.

**Use a smaller synthetic model for MVP validation**
Considered but rejected. A synthetic model doesn't exercise the real GGUF
parser or real quantization types. TinyLlama 1.1B is small enough to be
practical and real enough to provide meaningful validation.

**Metal backend in MVP**
Rejected. Metal toolchain availability varies across CI environments.
CPU is universally available. Metal is Phase 3.
