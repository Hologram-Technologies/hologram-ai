# ADR-0004: Multimodal invocation ABI

Status: Accepted (rewrite baseline)
Date: 2026-07-24

## Context

The public API must not be hard-coded around text prompts, and future
modalities must not require new Hologram layer kinds or `.holo` format
changes.

## Decision

A modality-neutral, versioned ABI in `hologram-ai-core` (`no_std + alloc`):

- `InferenceModelManifest` is authoritative: `operations:
  Vec<OperationDescriptor>` with named `inputs`/`outputs`
  (`ValueDescriptor { name, kind, media_type, required }`) and a `streaming`
  flag. Callers discover operations; no model is required to implement any
  particular one. Initial standardized names: `generate`, `predict`,
  `embed`, `classify`, `transcribe`.
- `ValueKind`: `Text`, `Tokens`, `Image`, `Audio`, `Bytes` (inline, media
  typed), `ContentRef` (content-addressed κ reference), `Structured`
  (canonical-encoded), `Buffer` (caller-owned output). Large media travels
  by κ reference so application layers do not copy bulk data.
- Invocation is generic: `session.invoke(operation, InferenceRequest) ->
  InferenceCompletion`. Typed convenience methods (`generate`, `predict`)
  are thin wrappers that check the manifest.
- Low-level allocation-free entrypoints (`predict_next_into`,
  `generate_into`) operate on caller-owned buffers and are the same engine
  path the convenience APIs call.
- Abstention, widening, witnesses, and provenance are first-class:
  `InferenceCompletion { finish_reason, status, widened, output, witness }`.

## Consequences

- New operations and modalities are manifest changes, not format changes.
- The same request/response types serialize across the FFI boundary for
  Python and TypeScript without per-language schema drift.
- Token-only, image-only, and audio-only models express their interface
  honestly in descriptors; bundle validation rejects a capability advertised
  without its required processor assets (ADR-0005).
