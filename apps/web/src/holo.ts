// Command adapter — the architectural seam (ADR-0017 §4) that replaces the
// Tauri `invoke()` backend. The browser GUI calls these functions; they drive
// the REAL hologram-ai pipeline compiled to WebAssembly (`hologram-ai-wasm`),
// not a reimplementation. Build the wasm package first: `pnpm wasm`.
import init, {
  describe as wasmDescribe,
  run as wasmRun,
  compile as wasmCompile,
  generate as wasmGenerate,
  compute_kappa as wasmComputeKappa,
  compile_safetensors as wasmCompileSafetensors,
} from "./wasm/hologram_ai_wasm.js";

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

/** Compile an ONNX model (bytes) → a `.holo` archive (bytes), in the browser. */
export async function compile(onnx: Uint8Array): Promise<Uint8Array> {
  await ensureReady();
  return wasmCompile(onnx);
}

export async function compileSafetensors(configJson: string, safetensorsShards: Uint8Array[]): Promise<Uint8Array> {
  await ensureReady();
  return wasmCompileSafetensors(configJson, safetensorsShards);
}

/** Compute the holospaces Kappa label for a byte array. */
export async function computeKappa(bytes: Uint8Array): Promise<string> {
  await ensureReady();
  return wasmComputeKappa(bytes);
}

/** Generation options (all optional). */
export interface GenOpts {
  prompt_template?: string;
  max_tokens?: number;
  temperature?: number;
  top_k?: number;
  stop?: string[];
  eos?: number;
  seed?: number;
}

/**
 * Autoregressive text generation over a compiled causal LM. The tokenizer is
 * read from the archive's baked-in extension unless `tokenizer` (a
 * `tokenizer.json`'s bytes) is given. Returns the generated text.
 */
export async function generate(
  holo: Uint8Array,
  prompt: string,
  opts: GenOpts = {},
  tokenizer?: Uint8Array,
): Promise<string> {
  await ensureReady();
  return wasmGenerate(holo, tokenizer ?? undefined, prompt, opts);
}
