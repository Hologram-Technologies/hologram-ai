# ADR-0006: Hugging Face acquisition and immutable revisions

Status: Accepted (rewrite baseline)
Date: 2026-07-24

## Context

Compilation must start from a verified, immutable local source. Downloading
is a `hologram-ai` responsibility — `uor-r4` always receives a verified
local directory and never does networking.

## Decision

A source-provider abstraction (`ModelSourceProvider`) with
`HuggingFaceProvider` and `LocalDirectoryProvider`:

- Revisions are **pinned full commit SHAs by default**; branch/tag
  resolution only when explicitly requested, and the resolved immutable
  commit is recorded and used thereafter.
- Content-addressed cache keyed by blake3(repository + resolved revision);
  downloads are resumable, staged in a sibling `.staging-*` directory, and
  completed by atomic rename with a validated manifest marker. Per-entry
  locking prevents concurrent-process corruption. An interrupted staging
  directory is never a valid cache hit.
- Private repositories authenticate through a credential provider; tokens
  are passed to the child process environment, never logged, never
  persisted, and redacted from errors and progress events.
- The official `hf` executable is invoked directly (opaque argv, no shell,
  no string interpolation). Explicit offline/cache-only mode is supported.
- Structured progress events and cooperative cancellation are part of the
  acquire contract.
- Unsupported models are a typed `UnsupportedModel`/`UnsupportedCapability`
  outcome, decided from the model configuration and the uor-r4
  compatibility surface — not a promise that every HF model compiles.

## Consequences

- Compile provenance can always name the exact source revision and content
  digests (ADR-0005 determinism).
- Normal CI needs no network; live HF tests are opt-in.
