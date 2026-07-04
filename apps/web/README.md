# hologram-ai · web app

The hologram-ai application as a **client-side WebAssembly app**, hosted on
GitHub Pages (static, no server). It runs the **real** pipeline in the browser
via [`hologram-ai-wasm`](../../crates/hologram-ai-wasm) — not a
reimplementation. See [ADR-0017](../../docs/adrs/0017-hologram-ai-web-wasm.md)
and the journey definition in
[`02-user-journey.md`](../../docs/conceptual-model/02-user-journey.md).

## Architecture

- `src/holo.ts` — the command adapter over the wasm binding: compile
  (streamed, weightless), describe, run, generate, κ-hashing,
  `kappa_requirements` + `materialize`.
- `src/ipc.ts` — the app backend: the data-driven catalogue
  (`public/catalogue.json` + user additions), HuggingFace resolution
  (endpoint overridable for hermetic testing), the parametric memory guard,
  OPFS listing, generation dispatch.
- `src/download.worker.ts` — the persistent download worker: streams
  safetensors shards tensor-by-tensor, κ-hashes each tensor incrementally,
  persists `tensors/{κ}.bin` to OPFS, then runs the streamed weightless
  compile at the guard-chosen context length.
- `src/generate.worker.ts` — materializes the k-form archive against the OPFS
  κ-store (sync access handles; every buffer re-hashes to its κ) and runs the
  real generation loop, streaming tokens.
- `src/resources.ts` — the parametric resource model (estimates, budget,
  context choice) as pure, unit-tested functions.
- `bdd/` — the browser BDD runner (cucumber-js + Playwright Chromium): the
  hermetic fixture server + step definitions for the `@executor:browser`
  dictionary rows, including the three-message chat handshake against the
  committed reference transcript.

## Develop

```bash
pnpm install
pnpm wasm        # build crates/hologram-ai-wasm → src/wasm (wasm-pack + wasm32 target)
pnpm dev         # vite dev server
pnpm build       # tsc + vite → dist/ (static, Pages-ready)
pnpm test        # vitest unit tests
pnpm bdd         # the hermetic browser journey (headless Chromium)
pnpm bdd:live    # the live journey against pinned HuggingFace models
```

`VITE_BASE` sets the Pages base path (default `/hologram-ai/`). Deployment is
automated by [`pages.yml`](../../.github/workflows/pages.yml) and **requires
the journey gate green** (dictionary row `deployment-gate`).

## Constraints (ADR-0017)

Single-threaded (no `SharedArrayBuffer` on Pages); 32-bit address space —
which is why the pipeline is k-representation end-to-end: κ-deduplicated
storage, a weightless structural archive, materialization at load, and
elision-based decode reuse. The memory guard rejects models whose
config-derived estimate exceeds the environment budget, before any transfer.
