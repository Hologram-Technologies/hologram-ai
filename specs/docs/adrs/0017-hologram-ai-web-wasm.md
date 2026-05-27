# ADR-0017: hologram-ai as a Browser App (WebAssembly + GitHub Pages)

**Status:** Accepted
**Date:** 2026-05-27
**Relates to:** ADR-0005 (runtime boundary), ADR-0016 (compiler, not runtime); CONFORMANCE class **NS**

---

## Context

hologram-ai exposes a CLI and a Tauri desktop GUI. The desktop GUI requires a
native windowing stack (GTK / WebKitGTK) we cannot provision in CI or the
devcontainer, so it cannot be built or exercised here. We want a GUI that runs
in any browser, is reproducibly buildable, and is hostable on **GitHub Pages**.

GitHub Pages is **static hosting with no server-side execution**. So a Pages app
cannot shell out to the `hologram-ai` binary the way the Tauri app does — the
compute must run **client-side, in the browser, as WebAssembly**.

This is not a new direction. The V&V framework already commits hologram-ai to a
multi-target runtime: class **NS** requires the runtime-core crates to build on
`wasm32-unknown-unknown`, and the architecture states the inference compute runs
"on wasm / embedded / platform targets." The browser app is the *realization* of
a property the platform already declares — not a port, and not a workaround.

### What the wasm spike established (2026-05-27)

Building for `wasm32-unknown-unknown`, **the entire compile + execute stack
compiles**: `hologram-types`, `-ops`, `-backend`, `-graph`, `-archive`, `-exec`,
`hologram-compiler`, `hologram-ai-onnx` (protobuf import), `hologram-ai-common`
(IR + lowering), `hologram-ai-quant`, `hologram-ai-tokenizer`. And it **runs**:
a precompiled `.holo` loaded via `InferenceSession::load` executed a MatMul in
wasm under node and returned the correct result (verified, not assumed).

Two facts shaped the decisions below:

1. **`parallel` (rayon) must be off on wasm.** rayon cannot spawn threads on
   `wasm32-unknown-unknown`. The pin `features = ["parallel"]` lived only in
   hologram-ai's `[workspace.dependencies]` (the hologram repo already uses
   `default-features = false`). The spike ran parallel-free by depending on
   `exec`/`backend` without it.
2. **In-browser compile is feasible** — the import + lowering + compiler crates
   all build on wasm32, so ONNX→`.holo` can happen in the browser, not only the
   toolchain.

## Decision

### 1. A `hologram-ai-wasm` crate is the single browser entry point

A `wasm-bindgen` `cdylib` (`crates/hologram-ai-wasm`) wraps the **real**
pipeline — no JS reimplementation, no mocks (the "no workarounds" rule). It
exposes the platform's verbs over byte buffers:

- `compile(onnx: &[u8]) -> Vec<u8>` — ONNX → `.holo` (host-shell logic, proven
  wasm-buildable).
- `describe(holo: &[u8]) -> JsValue` — input/output ports (dtype × element_count).
- `run(holo, inputs, fill) -> JsValue` — the arbitrary-model forward path
  (mirrors `run --fill`).
- `generate(holo, tokenizer_json, prompt, cfg) -> stream` — autoregressive
  generation (shares the CLI's loop via the extracted run-core; see §3).

The browser thus runs the same code paths as the CLI; the CLI and the wasm crate
are two host shells over one runtime core, exactly as ADR-0016 framed the CLI.

### 2. `parallel` becomes an opt-in build choice, not a pin

hologram-ai's `[workspace.dependencies]` no longer pins `features = ["parallel"]`
on `hologram-exec`/`hologram-backend`. Instead:

- the native consumers (`hologram-ai` lib+CLI, benches) enable a `parallel`
  feature **by default**, forwarding to `exec`/`backend`/`common` parallel — so
  native performance (class PV) is unchanged;
- `hologram-ai-wasm` depends on the runtime crates **without** `parallel`, so the
  browser build is single-threaded by construction.

This matches class NS (a no_std / non-parallel runtime-core build is a
first-class target) and the hologram repo's own `default-features = false` stance.

### 3. The run + generate logic is shared, not duplicated

The generation loop (LM port contract, sampling, stop handling, the decode
loop) is factored to operate over a minimal `Forward` capability
(`input_port_info` / `output_port_info` / `execute`) so both the CLI's
`HoloRunner` and the wasm runner drive **one** implementation. "Reuse as much as
possible" applies to the Rust core, not just the UI.

### 4. The web frontend reuses the React app; only the command layer changes

The browser GUI is the existing `apps/desktop/src` React app, served as a static
Vite bundle (`apps/web`). The Tauri command layer (`invoke("run"|"generate"|
"compile"|"list_*")`, which shells to the binary) is replaced by a **command
adapter** that calls `hologram-ai-wasm` and stores archives in the browser's
native **OPFS / IndexedDB** (no filesystem, no process). OPFS is a real browser
capability, not a stub. Filesystem/process-shaped commands (`workspace_paths`,
`recent_logs`) are reimagined for the browser, not faked.

### 5. The GUI surfaces the V&V conceptual model

Features map to the conformance classes — a fully-featured AI platform whose
differentiator is its *verifiable* UOR-native properties, shown live:
content-addressed reuse (CE), realized-memory / quantization (QZ + the
realized-information story — "what fits in a tab"), κ-label model identity (MA),
zero-movement / zero-alloc + per-stage budgets (ZM/ZA/PV), and an in-browser V&V
panel that runs conformance checks so the app demonstrates its own guarantees.

### 6. Deploy: GitHub Actions → Pages

A workflow builds the wasm (`wasm-pack`/`wasm-bindgen`) + the Vite bundle and
publishes to Pages. No headers required (single-threaded, no `SharedArrayBuffer`).

## Consequences

**Positive.** A browser GUI with no native toolchain; the real pipeline runs
client-side; one runtime core behind CLI + desktop + web; `parallel` becomes a
proper target choice; the V&V NS class extends from "the runtime core builds" to
"the engine runs a model in the browser."

**Constraints (honest, tracked — not worked around).**

- **32-bit address space** (~2–4 GB/tab): small / quantized models only. This is
  where the realized-information-content + quantization story is the right frame.
- **Single-threaded** on Pages (no COOP/COEP → no `SharedArrayBuffer`): correct,
  just not multi-core. Aligns with `parallel`-off.
- **Real-LM chat** stays gated on the upstream int64-embedding fix
  (`specs/notes/upstream-request-int-embedding.md`) — the web V&V panel shows it
  as pending, exactly as on native; it is not faked.
- Model acquisition is upload / bundled-demo / CORS `fetch` (no `download`
  command in-browser).

## Alternatives considered

- **Server-rendered / WASM-on-server.** Rejected: Pages is static; a server
  defeats the "hostable on Pages" goal and the UOR-native in-browser thesis.
- **Compile in the toolchain only, browser runs `.holo`.** A valid subset
  (honors the runtime-core/host-shell seam), kept as a fallback, but the spike
  showed in-browser compile is feasible, so we do not restrict to it.
- **Rewrite the UI for the web.** Rejected: reuse the React app; swap only the
  command backend.
