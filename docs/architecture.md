# Architecture

Normative sources: `docs/rewrite-plan.md`, `docs/adrs/0001–0009`. This is
the orientation map.

## System view

```text
                 ┌──────────────────────── hologram-ai ───────────────────────┐
                 │                                                            │
 HF repo@sha ───▶│ hologram-ai-huggingface                                    │
 local dir  ────▶│   verified immutable source (content-addressed cache)      │
                 │            │                                               │
                 │            ▼                                               │
                 │ hologram-ai-r4  ──(typed API only)──▶  uor-r4              │
                 │   compile orchestration                compile / R4G1      │
                 │   engine load-from-bytes               runtime             │
                 │            │                                               │
                 │            ▼                                               │
                 │ hologram-ai-bundle   R4 inference bundle schema v1         │
                 │   deterministic, role-addressed, digest-verified           │
                 │            │                                               │
                 │            ▼                                               │
                 │ hologram-ai (facade) ──▶ hologram archive/space (.holo v4) │
                 │   Compiler builder     InferenceModel layer                │
                 │   Application registry                                     │
                 │   Session / invoke / predict / generate                    │
                 └────────────┬───────────────────────────────────────────────┘
                              │
              hologram CLI (`hologram ai …`) · FFI · Python · TypeScript/Node
```

## Crates and boundaries

| Crate | Std? | Depends on | Owns |
|---|---|---|---|
| `hologram-ai-core` | no_std+alloc | nothing | schemas, canonical encoding, errors |
| `hologram-ai-bundle` | no_std+alloc | core, blake3 | bundle codec + validation |
| `hologram-ai-r4` | std | core, bundle, uor-r4-api | private uor-r4 adapter |
| `hologram-ai-huggingface` | std | core, blake3 | acquisition, cache, locking |
| `hologram-ai` | std | all above + hologram archive/space | facade, packaging, registry, sessions |

Rules: only `hologram-ai-r4` sees `uor-r4` types; only
`hologram-ai-huggingface` does network/process I/O; only the facade sees
hologram container types; the public API exposes none of those internals.

## Runtime model

- `Application::open_*` parses a `.holo` (official `HoloLoader`), verifies
  the fingerprint, and discovers `InferenceModel` layers from the app
  manifest. Metadata inspection never initializes an engine.
- `Application::model(entry)` selects explicitly; with several services
  there is no implicit default beyond the documented single-service case.
- `Model::session()` verifies bundle digests, then initializes the R4 engine
  and all fixed-capacity state. After that, `predict_next_into` /
  `generate_into` allocate nothing (ADR-0009).
- Invocation is generic and manifest-driven (ADR-0004); modality lives in
  descriptors, never in layer kinds (ADR-0003).

## Determinism and identity

Bundle digest = model identity; layer content κ = bundle bytes. Compile
provenance records source revision, source digest, compiler revision,
options digest, and quality-report digest — and nothing machine-specific
(ADR-0005).

## Platform support

Native (macOS/Linux) is the supported milestone. The portable core
(`core`, `bundle`) is gated on `wasm32-unknown-unknown`; full browser
inference is deferred per ADR-0008.
