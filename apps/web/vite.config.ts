import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// GitHub Pages serves a project site under `/<repo>/`. Override with
// VITE_BASE (e.g. "/" for a user/custom-domain site).
const base = process.env.VITE_BASE ?? "/hologram-ai/";

export default defineConfig({
  base,
  plugins: [react()],
  // The wasm-bindgen glue references its `.wasm` via `new URL(..., import.meta.url)`;
  // keep it an external asset so Vite emits and fingerprints it.
  build: { target: "es2022" },
});
