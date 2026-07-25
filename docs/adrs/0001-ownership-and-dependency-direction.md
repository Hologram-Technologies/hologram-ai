# ADR-0001: Project ownership and dependency direction

Status: Accepted (rewrite baseline)
Date: 2026-07-24

## Context

hologram-ai grew up as a tensor/ONNX compiler and runtime on the hologram
substrate. The R⁴ program in `uor-r4` now owns transformerless graph
compilation and deterministic integer inference, and `hologram` owns the
`.holo` application container. Three projects with blurred boundaries cannot
evolve independently.

## Decision

Ownership is exclusive:

- **uor-r4** owns observation, graph compilation, the R4G1 format, validation,
  signature derivation, scoring, inference, generation, resolution status and
  abstention, witnesses, and the multiplication-free / allocation-free
  inference contract.
- **hologram** owns the `.holo` container, manifests, layer declarations,
  archives, the `hologram` CLI, the generic layer-service seam, and the FFI /
  SDK infrastructure.
- **hologram-ai** owns the production AI API: source acquisition (Hugging
  Face and local), immutable revision/cache policy, compile orchestration,
  R4 inference bundles, `.holo` `InferenceModel` packaging, model registry
  and sessions, and the multimodal request/response ABI.

Dependency direction is acyclic:

```text
hologram-ai ─→ hologram (archive, space only)
hologram-ai ─→ uor-r4 (typed facade only, via one private adapter crate)
hologram CLI/FFI ─→ hologram-ai (optional)
```

`hologram` never depends on `uor-r4`. No hologram-ai crate depends on
`hologram-cli`/`hologram-ffi`. There is no public `hologram-ai` binary.

## Consequences

- Compiler, scorer, tokenizer, status-policy, and generation algorithms are
  never copied into this repository; missing upstream APIs are patched
  upstream (see `docs/upstream/`).
- hologram-ai changes to `uor-r4`/`hologram` are minimal, generic, and
  delivered as PR-ready branches with pinned base revisions recorded in
  `docs/rewrite-plan.md`.
