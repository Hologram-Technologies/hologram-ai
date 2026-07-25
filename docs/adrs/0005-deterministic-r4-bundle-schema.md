# ADR-0005: Deterministic R4 inference bundle schema

Status: Accepted (rewrite baseline)
Date: 2026-07-24

## Context

Each `InferenceModel` layer needs one deterministic, opaque model blob so
the generic Hologram store can treat a model as content without
understanding R4 internals, and so byte-identical inputs reproduce
byte-identical `.holo` archives.

## Decision

**R4 inference bundle schema v1** (`docs/bundle-format.md` is normative):

- Components are addressed by closed, versioned **semantic roles**
  (`graph`, `signature-artifact`, `tokenizer`, `score-report`,
  `image-processor`, `audio-processor`, `generation-config`,
  `vocabulary-metadata`, `label-map`, `normalization-metadata`,
  `witnesses`) — never by historical filenames. The artifact historically
  named `tless_artifacts.bin` is public only as `signature-artifact`.
- Mandatory: canonical manifest, validated scored R4G1 graph,
  signature-artifact, complete score report, normalized status policy, ABI
  versions, deterministic provenance, per-component lengths and BLAKE3
  digests, operation and value descriptors.
- Conditional (validated at load): tokenizer iff text in/out (unless
  explicitly token-only), image processor iff image input, audio processor
  iff audio input; a multimodal model carries every processor its declared
  operations require.
- Excluded: `tless_store.bin`, corpus/observation/cover data, checkpoints,
  temp dirs, source weights (unless the deployed runtime requires them),
  source/cache paths, wall-clock timestamps.
- Canonical encoding (`hologram-ai-core::canon`): little-endian,
  length-prefixed, declaration-ordered, map-free — determinism by
  construction. Same source revision + options + tool revisions ⇒
  byte-identical bundle and `.holo`.
- Untrusted-input handling: checked arithmetic, explicit size caps, bounded
  allocation, duplicate-role and unknown-mandatory-role rejection, digest
  verification before engine initialization, capability⇔processor
  consistency validation, no path traversal, no decompression, fuzzable
  parsers, typed errors.

## Consequences

- Bundle digest = model identity; the layer's content κ addresses the bundle
  bytes directly.
- The bundle is the only R4 artifact users ever handle; `.r4g1` filenames
  remain private work-dir implementation details.
