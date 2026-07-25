# ADR-0002: R4G1-only inference

Status: Accepted (rewrite baseline)
Date: 2026-07-24

## Context

The pre-rewrite hologram-ai shipped a tensor executor (with `MatMul`), ORT
conformance lanes, speculative decoding, and a transformer-style generation
stack. uor-r4's R4G1 runtime provides deterministic, integer-only,
status-aware inference with abstention and witnesses.

## Decision

R4G1 is the only inference path exposed by hologram-ai. Specifically:

- No TLA/TLS fallback; `tless_store.bin` is never packaged in model bundles.
- No Hologram `TensorPlan` is compiled or executed as a fallback, and R4G1 is
  never disguised as a tensor plan, extension, or wasm codemodule.
- No transformer runtime, no `MatMul` in the deployed model or inference path.
- No GPU dependency or execution path (CUDA, Metal, WGPU, ROCm) anywhere in
  the tree; inference is CPU-native, including any future browser/WASM build.
- No external inference provider at runtime; no network access during model
  loading or inference.
- No probabilistic sampling machinery (tensor logits, floating-point
  probabilities, temperature, top-p). Generation follows uor-r4's
  deterministic, status-aware decisions; abstention is a typed completion
  (`FinishReason::Abstained`), never an exception and never a fabricated
  token.

## Consequences

- CI enforces the boundary with dependency inspection (`cargo tree` audit:
  no ort/onnxruntime/candle/tch/GPU crates) and artifact inspection (no
  `tless_store.bin` in bundles), not only string scans.
- Multimodal schemas exist before multimodal compilation does; we never
  claim image/audio compilation works until a real uor-r4 path proves it.
