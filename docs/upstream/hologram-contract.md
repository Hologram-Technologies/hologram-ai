# Upstream contract: hologram

Base revision: `94ecb886811115491a77c8229e494965bea03fc2` (Hologram-Technologies/hologram main, v0.12.1+46)
Patch branch: `feature/inference-model-layer` (worktree `../hologram-im`)
Status: PR-ready branch in progress; hologram-ai consumes it by path until merged.

## What hologram-ai needs (and why)

hologram stays engine-agnostic: no `uor-r4` dependency anywhere, no AI
algorithms, no tensor-plan changes. The patch is additive and format-appending.

### 1. `.holo` format v4 + `LayerKind::InferenceModel = 4`

- `hologram-archive`: `FORMAT_VERSION = 4`, `MIN_READ_VERSION` stays 2
  (v2/v3 still readable; v3 readers reject v4 via the existing version gate).
- `hologram-space`: append `InferenceModel = 4` to the closed `LayerKind`
  enum (existing discriminants untouched). Semantics: non-exit-bearing;
  `entry` non-empty (callable service name); `aux` mandatory = engine
  identifier (e.g. `uor-r4`); entry names unique per application;
  `primary = None` valid for model-only archives; `primary` pointing at an
  `InferenceModel` rejected.
- Constructor `Layer::inference_model(content, entry, engine)` + validation
  errors in the existing style.

### 2. `hologram ai` CLI group

`hologram ai download|compile|inspect|infer` delegating to the `hologram-ai`
crate behind an optional feature/dependency (off by default so the default
build is untouched):

- `download <repo> --revision <full-sha> [--offline] [--json]`
- `compile [repo] [--revision <sha> | --source <dir>] --output <file.holo> [--entry ai.default] [--json]`
- `inspect <file.holo> [--json]` — metadata only, never initializes inference
- `infer <file.holo> [--model <entry>] [--operation generate] [--prompt …] [--max-output-tokens N] [--json]` — never downloads or recompiles

### 3. FFI surface

New `hologram_ai_*` exports behind an optional feature: download, compile
(HF + local source), app load (path/bytes), model count/entry listing,
session open/close, `invoke` as JSON envelope; appended stable error-code
range (≥100) mapped from `hologram-ai-core::ErrorCategory`; `FEATURES`
capability-probe strings; SDK metadata regenerated via the generator.

## Explicitly not requested

- No layer-kind-per-modality (modality lives in hologram-ai manifests).
- No synchronous cross-layer call dispatch implemented in hologram yet —
  hologram-ai's registry serves SDK callers; the generic layer-service
  invocation seam for wasm containers is a separate future hologram change.
- No `model-formats` κ-addressing changes.
