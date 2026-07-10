import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// GitHub Pages serves a project site under `/<repo>/`. Override with
// VITE_BASE (e.g. "/" for a user/custom-domain site).
const base = process.env.VITE_BASE ?? "/hologram-ai/";

// Cross-origin isolation (ADR-0018): required for SharedArrayBuffer / wasm
// threads (the multi-threaded decode pool). `credentialless` — NOT
// `require-corp` — so the only external origin (HuggingFace + its LFS CDN)
// loads WITHOUT a `Cross-Origin-Resource-Policy` header it does not send.
// These headers cover dev + `vite preview` (the BDD journey); production is
// static GitHub Pages, so `public/coi-serviceworker.js` injects the same pair.
const coiHeaders = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "credentialless",
};

export default defineConfig({
  base,
  plugins: [react()],
  // Nested ES workers: the generate/execute worker spawns the pool workers
  // (`pool.worker.ts`) which `import` the threaded wasm glue as ES modules.
  worker: { format: "es" },
  server: { headers: coiHeaders },
  preview: { headers: coiHeaders },
  // The wasm-bindgen glue references its `.wasm` via `new URL(..., import.meta.url)`;
  // keep it an external asset so Vite emits and fingerprints it.
  build: { target: "es2022" },
});
