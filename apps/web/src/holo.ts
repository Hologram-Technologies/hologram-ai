// Command adapter — the architectural seam (ADR-0017 §4) that replaces the
// Tauri `invoke()` backend. The browser GUI calls these functions; they drive
// the REAL hologram-ai pipeline compiled to WebAssembly (`hologram-ai-wasm`),
// not a reimplementation. Build the wasm package first: `pnpm wasm`.
import init, { describe as wasmDescribe, run as wasmRun } from "./wasm/hologram_ai_wasm.js";

export interface Port {
  name: string;
  dtype: number;
  dtype_name: string;
  element_count: number;
  shape: number[];
  bytes: number;
}
export interface ModelInfo {
  inputs: Port[];
  outputs: Port[];
}
export interface Output {
  dtype: number;
  dtype_name: string;
  element_count: number;
  values: number[];
}

let ready: Promise<unknown> | null = null;
/** Instantiate the wasm module once (runs the panic-hook `start`). */
export function ensureReady(): Promise<unknown> {
  if (!ready) ready = init();
  return ready;
}

/** Inspect a compiled `.holo` — its input/output ports (positional, no names). */
export async function describe(holo: Uint8Array): Promise<ModelInfo> {
  await ensureReady();
  return wasmDescribe(holo) as ModelInfo;
}

/**
 * Forward pass over an arbitrary compiled model (mirrors `run --fill`). Pass
 * explicit input byte arrays by index; omit/empty entries are synthesized from
 * `fill` (a number, or undefined ⇒ zeros).
 */
export async function run(
  holo: Uint8Array,
  inputs: Uint8Array[] = [],
  fill?: number,
): Promise<Output[]> {
  await ensureReady();
  return wasmRun(holo, inputs, fill ?? undefined) as Output[];
}

// ── Pending platform verbs ────────────────────────────────────────────────────
// Surfaced honestly (they throw), not faked — consistent with the V&V
// no-silent-fallback ethos. Wired once the shared compile/run core is factored
// out of the native facade, and (for chat) the int64-embedding upstream fix
// lands. See ADR-0017 §3 and specs/notes/upstream-request-int-embedding.md.

export async function compile(_onnx: Uint8Array): Promise<Uint8Array> {
  throw new Error(
    "compile() in-browser is pending the shared compile-core extraction (ADR-0017 §3). " +
      "Until then, compile with the CLI and load the .holo here.",
  );
}

export async function generate(): Promise<never> {
  throw new Error(
    "generate() is pending the run-core extraction and the int64 token-embedding " +
      "upstream fix (specs/notes/upstream-request-int-embedding.md).",
  );
}
