# hologram-ai · web GUI

The hologram-ai GUI as a **client-side WebAssembly app**, hostable on GitHub
Pages (static, no server). It runs the **real** runtime core in the browser via
[`hologram-ai-wasm`](../../crates/hologram-ai-wasm) — not a reimplementation.
See [ADR-0017](../../specs/docs/adrs/0017-hologram-ai-web-wasm.md).

## Architecture

- `src/holo.ts` — the command adapter (the seam that replaces Tauri `invoke()`):
  it calls the wasm pipeline. `describe` + `run` are live; `compile`/`generate`
  throw a clear "pending" error (honest, not faked — they land as the shared
  compile/run core is factored out of the native facade, and chat additionally
  awaits the int64-embedding upstream fix).
- `src/App.tsx` — a minimal playground (load a `.holo`, inspect ports, run with
  zero/ones/N fill). The reused desktop React components plug into the same
  `holo.ts` seam as the surface grows.

## Develop

```bash
pnpm install
pnpm wasm        # build crates/hologram-ai-wasm → src/wasm (needs wasm-pack + the wasm32 target)
pnpm dev         # vite dev server
pnpm build       # tsc + vite → dist/ (static, Pages-ready)
```

`VITE_BASE` sets the Pages base path (default `/hologram-ai/`; use `/` for a
user/custom-domain site). Deployment is automated by
[`.github/workflows/pages.yml`](../../.github/workflows/pages.yml).

## Constraints (ADR-0017)

Single-threaded (no `SharedArrayBuffer` on Pages); 32-bit address space
(~2–4 GB/tab → small / quantized models); models arrive by upload or CORS
`fetch` (no in-browser download). These are honored, not worked around.
