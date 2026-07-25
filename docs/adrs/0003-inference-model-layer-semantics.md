# ADR-0003: `InferenceModel` layer semantics

Status: Accepted (rewrite baseline; upstream patch `feature/inference-model-layer`)
Date: 2026-07-24

## Context

`.holo` v3 has a closed `LayerKind` vocabulary (`WasmCodemodule`,
`TensorPlan`, `RootfsImage`, `View`). Packaging a compiled R4G1 model needs a
semantically correct home; reusing `TensorPlan` or `Extension` would lie
about what the content is.

## Decision

`.holo` v4 appends `LayerKind::InferenceModel = 4` (existing discriminants
unchanged, `MIN_READ_VERSION` stays 2). An `InferenceModel` layer is a
**callable AI service**:

- `content` = κ of the deterministic R4 inference bundle (one opaque blob).
- `entry` = unique, non-empty callable service name within the application
  (default `ai.default`; e.g. `ai.language`, `ai.vision`, `ai.fusion`).
- `aux` = the engine identifier (e.g. `uor-r4`) — the existing kind-specific
  typed tag field, mandatory for this kind, so no schema field is added and
  no unrelated field is overloaded.
- Non-exit-bearing: a model-only archive has `primary = None`; a larger
  application keeps a WASM or rootfs layer as `primary`.
- Loading the layer means: verify the bundle, initialize the engine and all
  fixed-capacity runtime state, then register the service for other layers
  and SDK callers.

Layer rules: zero, one, or many `InferenceModel` layers per application;
manifest order is initialization order, never inference dataflow; callers
select a model explicitly by entry name — the API never silently picks the
first layer (an unambiguous default exists only when exactly one service is
declared, and is documented as such).

There are no modality-specific layer kinds (`VisionModel`, `AudioModel`):
modality lives in operation and value descriptors (ADR-0004). One layer may
be a genuinely multimodal model; several layers may cooperate as encoders,
language models, and fusion models.

## Consequences

- hologram-ai performs model discovery by scanning the application manifest
  for `InferenceModel` layers; it never hand-writes container bytes — the
  official `HoloWriter`/`HoloLoader` APIs are used.
- v3 readers reject v4 archives cleanly through the existing version gate.
