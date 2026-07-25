# ADR-0007: SDK ownership through Hologram

Status: Accepted (rewrite baseline)
Date: 2026-07-24

## Context

The AI capability must reach Rust, Python, and TypeScript/Node users. The
pre-rewrite repo had its own CLI and browser app; the rewrite forbids a
competing public surface.

## Decision

Public SDK branding is Hologram; hologram-ai provides the Rust
implementation and a stable binding-oriented surface, and Hologram exposes
it through its existing FFI and generated SDK infrastructure:

- Rust: the `hologram_ai` crate facade (compiler builder, application
  registry, sessions, invocation).
- Python: `hologram.ai` (ctypes over `hologram-ffi`'s `hologram_ai_*`
  exports).
- TypeScript: `@tryhologram/sdk` AI namespace; Node through
  `@tryhologram/native`; a future `@tryhologram/wasm` adapter implements the
  same binding protocol. The TypeScript SDK stays browser-safe: no direct
  Node imports, native/WASM adapters behind the `NativeBinding` protocol.
- No public `hologram-ai` binary and no competing packages named
  `hologram-ai-python` or `@tryhologram/ai`.
- Errors cross the boundary as stable numeric categories
  (`hologram-ai-core::ErrorCategory`, codes 1–20) mapped into Python and
  TypeScript error classes; request/response payloads use the canonical
  schemas (JSON envelopes at the FFI edge for binding simplicity).

## Consequences

- SDK semantic parity (discovery, selection, invocation, status, abstention,
  witnesses, errors) is tested once against the Rust facade and smoke-tested
  per language.
- FFI additions are minimal Hologram patches (new exports + error-code range
  + capability-probe strings), not a parallel binding system.
