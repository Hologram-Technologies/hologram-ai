# ADR-0008: Native first, browser/WASM later

Status: Accepted (rewrite baseline)
Date: 2026-07-24

## Context

The pre-rewrite repo shipped a browser app tied to the tensor/ONNX pipeline
(including multi-threaded wasm decode). That application was removed with
the old architecture. Browser R4 inference remains a goal, but must not be
claimed before it is proven.

## Decision

The first complete runtime milestone is **native**. Portability is
structural, not aspirational:

- `hologram-ai-core` and `hologram-ai-bundle` are `no_std + alloc` and are
  CI-gated on `wasm32-unknown-unknown`.
- Native I/O and process execution are isolated in
  `hologram-ai-huggingface` and the facade; the R4 engine integration is
  path-independent once bytes are loaded.
- No global process state in the public API; no assumption of mmap or
  threads; no Python/Node concepts in the Rust core.

The browser phase, in order:

1. Load and inspect an already-compiled `.holo`.
2. Initialize R4G1 from archive bytes.
3. Invoke inference through the browser-safe TypeScript binding protocol.
4. Browser-side acquisition/compilation only after uor-r4's compiler has a
   credible WASM path.

## Consequences

- No browser application ships in this phase; architecture documentation
  and portable-core tests replace it.
- uor-r4's deployed scorer is std-only today; the wasm engine phase also
  requires upstream work, tracked in `docs/upstream/uor-r4-contract.md`.
- We do not claim complete browser R4 inference anywhere in user-facing
  docs until it is demonstrated end to end.
