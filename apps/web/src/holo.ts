// Command adapter — the architectural seam (ADR-0017 §4) that replaces the
// Tauri `invoke()` backend. The browser GUI calls these functions; they drive
// the REAL hologram-ai pipeline compiled to WebAssembly (`hologram-ai-wasm`),
// not a reimplementation. Build the wasm package first: `pnpm wasm`.
// The wasm binding is chosen at runtime (ADR-0018). When the page is
// cross-origin-isolated AND the caller opted in via `preferThreadedPool()` (the
// generate/execute worker), we load the MULTI-THREADED build (`./wasm-threads`)
// — the substrate embedder worker pool over a shared linear memory — else the
// single-threaded `+simd128` fallback (`./wasm`, byte-identical to before
// ADR-0018). Both are the SAME crate, so they export the same surface; `G` holds
// whichever was initialised, and every verb dispatches through it. Build them
// with `pnpm wasm` and `scripts/build-wasm-threads.sh`.
type GlueModule = typeof import("./wasm/hologram_ai_wasm.js");
let G: GlueModule | null = null;

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
let preferThreaded = false;
let threadedActive = false;
let poolWorkers: Worker[] = [];

/**
 * Opt this context into the multi-threaded decode pool (ADR-0018). Call it
 * BEFORE the first `ensureReady()` — only the generate/execute worker does, so
 * the main thread and download worker stay single-threaded (the pool would need
 * a blocking `Atomics.wait`, disallowed on the main thread; and only the `m == 1`
 * decode GEMV parallelises). No-op unless the page is cross-origin-isolated.
 */
export function preferThreadedPool(v = true): void {
  preferThreaded = v;
}

/** Whether the threaded pool is active and how many workers registered. */
export function poolInfo(): { threaded: boolean; workers: number } {
  return { threaded: threadedActive, workers: poolWorkers.length };
}

/** Instantiate the wasm module once (runs the panic-hook `start`). */
export function ensureReady(): Promise<unknown> {
  if (!ready) ready = initGlue();
  return ready;
}

async function initGlue(): Promise<unknown> {
  const isolated = typeof crossOriginIsolated !== "undefined" && crossOriginIsolated === true;
  if (preferThreaded && isolated) {
    try {
      return await initThreaded();
    } catch (err) {
      // Fail SOFT to the single-threaded build — a coherent slow decode beats a
      // dead tab. (The V&V asserts the pool DOES engage on the isolated deploy,
      // so a silent permanent fallback can't hide — see ADR-0018.)
      console.warn("[holo] threaded pool init failed; using single-threaded build", err);
    }
  }
  const glue = await import("./wasm/hologram_ai_wasm.js");
  await glue.default();
  G = glue;
  threadedActive = false;
  return G;
}

async function initThreaded(): Promise<unknown> {
  const glue = await import("./wasm-threads/hologram_ai_wasm.js");
  const wasmUrl = new URL("./wasm-threads/hologram_ai_wasm_bg.wasm", import.meta.url);
  // Compile ONCE, then instantiate the same module on the execute instance and
  // every pool worker so they all share one linear memory.
  let module: WebAssembly.Module;
  try {
    module = await WebAssembly.compileStreaming(fetch(wasmUrl));
  } catch {
    module = await WebAssembly.compile(await (await fetch(wasmUrl)).arrayBuffer());
  }
  // Execute/"main" instance: no `thread_stack_size` → it creates the shared
  // memory and runs the module's one-time data init.
  const exports = (await glue.default({ module_or_path: module })) as unknown as {
    memory: WebAssembly.Memory;
    hologram_pool_workers: () => number;
  };
  const memory = exports.memory;
  // Leave one core for the execute worker (it is the pool's `+1` participant).
  const n = Math.max(1, Math.min((navigator.hardwareConcurrency || 2) - 1, 32));
  const stackSize = 2 * 1024 * 1024; // 2 MiB per worker (64 KiB-aligned)
  poolWorkers = [];
  for (let id = 0; id < n; id++) {
    const w = new Worker(new URL("./pool.worker.ts", import.meta.url), { type: "module" });
    w.postMessage({ module, memory, id, stackSize });
    poolWorkers.push(w);
  }
  // Gate the first decode on FULL registration — the substrate traps on a
  // worker that registers after the first job.
  const t0 = Date.now();
  while (exports.hologram_pool_workers() < n) {
    if (Date.now() - t0 > 10_000) {
      throw new Error(`pool registration timeout: ${exports.hologram_pool_workers()}/${n}`);
    }
    await new Promise((r) => setTimeout(r, 4));
  }
  G = glue as unknown as GlueModule;
  threadedActive = true;
  // Telemetry the V&V asserts on (a silent fallback to single-threaded must not
  // be able to masquerade as success — ADR-0018, [[dark-gates]]).
  console.log(`[holo] multi-threaded decode pool active: ${n} workers over shared memory`);
  return G;
}

/** Inspect a compiled `.holo` — its input/output ports (positional, no names). */
export async function describe(holo: Uint8Array): Promise<ModelInfo> {
  await ensureReady();
  return G!.describe(holo) as ModelInfo;
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
  return G!.run(holo, inputs, fill ?? undefined) as Output[];
}

/** Compile an ONNX model (bytes) → a `.holo` archive (bytes), in the browser. */
export async function compile(onnx: Uint8Array): Promise<Uint8Array> {
  await ensureReady();
  return G!.compile(onnx);
}

export async function compileOnnxWithData(onnxBytes: Uint8Array, externalData: Uint8Array): Promise<Uint8Array> {
  await ensureReady();
  return G!.compile_onnx_with_data(onnxBytes, externalData);
}

export async function compileSafetensorsStreamed(
  configJson: string,
  keys: string[],
  kappas: string[],
  shapes: string[],
  dtypes: string[],
  contextLength?: number,
): Promise<Uint8Array> {
  await ensureReady();
  return G!.compile_safetensors_streamed(configJson, keys, kappas, shapes, dtypes, contextLength);
}

/** The architecture families the parametric registry supports — the single
 * source the search filter reads (row `supported-search`). */
export async function supportedFamilies(): Promise<string[]> {
  await ensureReady();
  return G!.supported_families() as string[];
}

/** Config-only preflight (S1 step a): registered family + required keys —
 * checked before even the shard headers are fetched. */
export async function validateModelConfig(configJson: string): Promise<void> {
  await ensureReady();
  G!.validate_model_config(configJson);
}

/**
 * Preflight (S1): validate that the parametric graph builds from config.json
 * plus the header-only tensor manifest — before any shard byte moves. Throws
 * naming the family/key/manifest defect.
 */
export async function validateStreamedManifest(
  configJson: string,
  keys: string[],
  shapes: string[],
  dtypes: string[],
  contextLength?: number,
  layersPerStage?: number,
): Promise<void> {
  await ensureReady();
  G!.validate_streamed_manifest(configJson, keys, shapes, dtypes, contextLength, layersPerStage);
}

/** Staged compile (windowed execution over k): one k-form archive per stage
 * (embedding, layer blocks, head). */
export async function compileSafetensorsStaged(
  configJson: string,
  keys: string[],
  kappas: string[],
  shapes: string[],
  dtypes: string[],
  contextLength: number | undefined,
  layersPerStage: number,
): Promise<Uint8Array[]> {
  await ensureReady();
  return Array.from(
    G!.compile_safetensors_staged(configJson, keys, kappas, shapes, dtypes, contextLength, layersPerStage),
  ) as Uint8Array[];
}

/** A persistent staged chat session (rows `staged-window-growth`,
 * `stage-residency-cache`, `warm-turn`): the compiled window, resident stage
 * sessions, verified-κ set, and derived-artifact cache survive across turns,
 * so a warm turn pays decode — never recompile, never rematerialization.
 * Construct once per model; call `generate` per turn. */
/** One quantized derived-artifact record (row `quantized-transit`): the wide
 * tensor's κ, its matmul-ready int8 artifact's κ, and the projection dims. A
 * whole projection carries just those; a **head chunk** additionally carries
 * `offset`/`len` — its byte range within the wide LM-head/embedding tensor — so
 * the several chunks that share one κ (a tied head shares the embedding table's)
 * each map to their own per-chunk artifact under a κ+range key. */
export interface QuantEntry {
  wide: string;
  artifact: string;
  out: number;
  in: number;
  offset?: number;
  len?: number;
}

/** One head-chunk quantization target (row `quantized-transit`, chunked head):
 * a vocab-row byte range of the LM-head weight to crystallize into its own int8
 * artifact. */
export interface HeadChunkTarget {
  kappa: string;
  offset: number;
  len: number;
  out: number;
  in: number;
}

/** The head-chunk quantization targets of the staged plan: the vocab-row ranges
 * of a large LM head the int8 tier derives into per-chunk artifacts (so a
 * chunked head joins the int8 tier instead of remaining a bf16 matmul whose F32
 * panel thrashes residency). Empty where the head is a single chunk. */
export async function headQuantChunks(
  configJson: string,
  keys: string[],
  kappas: string[],
  shapes: string[],
  dtypes: string[],
  contextLength: number | undefined,
  layersPerStage: number,
): Promise<HeadChunkTarget[]> {
  await ensureReady();
  return JSON.parse(
    G!.head_quant_chunks(configJson, keys, kappas, shapes, dtypes, contextLength, layersPerStage),
  ) as HeadChunkTarget[];
}

/** The wide κs the staged plan can rewrite onto quantized artifacts and
 * fully retire — the download derives artifacts for exactly these and their
 * wide blobs go gas-phase. */
export async function quantizableWeights(
  configJson: string,
  keys: string[],
  kappas: string[],
  shapes: string[],
  dtypes: string[],
  contextLength: number | undefined,
  layersPerStage: number,
): Promise<string[]> {
  await ensureReady();
  return Array.from(
    G!.quantizable_weights(configJson, keys, kappas, shapes, dtypes, contextLength, layersPerStage),
  ) as string[];
}

/** Derive the matmul-ready int8 artifact of a wide [out, in] weight —
 * deterministic; mint the artifact's κ from the returned bytes. */
export async function deriveQuantizedArtifact(
  wide: Uint8Array,
  dtype: string,
  outFeatures: number,
  inFeatures: number,
): Promise<Uint8Array> {
  await ensureReady();
  return G!.derive_quantized_artifact(wide, dtype, outFeatures, inFeatures);
}

/** `compileSafetensorsStaged` on the quantized tier: stage graphs bind
 * projection weights to their quantized derived artifacts. */
export async function compileSafetensorsStagedQuantized(
  configJson: string,
  keys: string[],
  kappas: string[],
  shapes: string[],
  dtypes: string[],
  contextLength: number | undefined,
  layersPerStage: number,
  quant: QuantEntry[],
): Promise<Uint8Array[]> {
  await ensureReady();
  return Array.from(
    G!.compile_safetensors_staged_quantized(
      configJson,
      keys,
      kappas,
      shapes,
      dtypes,
      contextLength,
      layersPerStage,
      JSON.stringify(quant),
    ),
  ) as Uint8Array[];
}

export async function createStagedSession(
  configJson: string,
  manifest: { keys: string[]; kappas: string[]; shapes: string[]; dtypes: string[] },
  contextLength: number | undefined,
  layersPerStage: number,
  resolveKappa: (kappa: string) => Uint8Array | undefined,
  invalidateKappa: ((kappa: string) => void) | undefined,
  resolveKappaRange: ((kappa: string, offset: number, len: number) => Uint8Array | undefined) | undefined,
  quant: QuantEntry[] | undefined,
  derived:
    | {
        load: (key: string) => { stages: Uint8Array[]; kappas: string[] } | undefined;
        store: (key: string, stages: Uint8Array[], kappas: string[]) => void;
        evaporate: (key: string) => void;
      }
    | undefined,
  weightBudget: number | undefined,
  sizeKappa: ((kappa: string) => number | undefined) | undefined,
  tokenizer: Uint8Array,
  onProgress?: (line: string) => void,
): Promise<StagedSession> {
  await ensureReady();
  return new G!.StagedChatSession(
    configJson,
    manifest.keys,
    manifest.kappas,
    manifest.shapes,
    manifest.dtypes,
    contextLength,
    layersPerStage,
    resolveKappa,
    invalidateKappa,
    resolveKappaRange,
    quant && quant.length ? JSON.stringify(quant) : undefined,
    derived?.load,
    derived?.store,
    derived?.evaporate,
    weightBudget,
    sizeKappa,
    tokenizer,
    onProgress,
  );
}

/** The decode-plan twin of {@link createStagedSession} (row `decode-plan`):
 * same manifest, κ-store, quant tier, and derived-store wiring; every token
 * is one single-position pass instead of a window-sized forward. */
export async function createDecodeSession(
  configJson: string,
  manifest: { keys: string[]; kappas: string[]; shapes: string[]; dtypes: string[] },
  contextLength: number | undefined,
  layersPerStage: number,
  resolveKappa: (kappa: string) => Uint8Array | undefined,
  invalidateKappa: ((kappa: string) => void) | undefined,
  resolveKappaRange: ((kappa: string, offset: number, len: number) => Uint8Array | undefined) | undefined,
  quant: QuantEntry[] | undefined,
  derived:
    | {
        load: (key: string) => { stages: Uint8Array[]; kappas: string[] } | undefined;
        store: (key: string, stages: Uint8Array[], kappas: string[]) => void;
        evaporate: (key: string) => void;
      }
    | undefined,
  weightBudget: number | undefined,
  sizeKappa: ((kappa: string) => number | undefined) | undefined,
  tokenizer: Uint8Array,
  onProgress?: (line: string) => void,
): Promise<StagedSession> {
  await ensureReady();
  return new G!.DecodeChatSession(
    configJson,
    manifest.keys,
    manifest.kappas,
    manifest.shapes,
    manifest.dtypes,
    contextLength,
    layersPerStage,
    resolveKappa,
    invalidateKappa,
    resolveKappaRange,
    quant && quant.length ? JSON.stringify(quant) : undefined,
    derived?.load,
    derived?.store,
    derived?.evaporate,
    weightBudget,
    sizeKappa,
    tokenizer,
    onProgress,
  );
}

export interface StagedSession {
  generate(prompt: string, opts: GenOpts, callback?: (text: string) => void): string;
  materialization_count(): bigint;
  derived_hits(): bigint;
  prederive_next_window(): number | undefined;
  /** Pair a speculative DRAFT model (row `speculative-draft-pairing`): `draft`
   * is a second decode session built from the paired model's dir, whose growable
   * this session absorbs so speculative decode drafts from the paired model
   * (`ModelDrafter`) instead of by prompt-lookup. Consumes `draft`. Throws if the
   * draft's vocabulary does not cover this target's — the caller then falls back
   * to prompt-lookup. Only the decode session exposes it (the only plan that
   * speculates); optional so a window session need not. */
  attach_draft?(draft: StagedSession): void;
  free(): void;
}

/** Token count of `text` under the model's own tokenizer (session trimming). */
export async function countTokens(tokenizer: Uint8Array, text: string): Promise<number> {
  await ensureReady();
  return G!.count_tokens(tokenizer, text);
}

/** The κ-labels a k-form archive requires (empty for a material archive). */
export async function kappaRequirements(holo: Uint8Array): Promise<string[]> {
  await ensureReady();
  return G!.kappa_requirements(holo) as string[];
}

/**
 * Materialize a k-form archive against a κ-store: `resolve` returns the bytes
 * for a κ (or undefined when absent — the pipeline aborts naming the label).
 * Every buffer is re-hashed and must reproduce its κ (S3, content-verified).
 */
export async function materialize(
  holo: Uint8Array,
  resolve: (kappa: string) => Uint8Array | undefined,
  invalidate?: (kappa: string) => void,
): Promise<Uint8Array> {
  await ensureReady();
  return G!.materialize(holo, resolve, invalidate);
}


/** Compute the holospaces Kappa label for a byte array. */
export async function computeKappa(bytes: Uint8Array): Promise<string> {
  await ensureReady();
  return G!.compute_kappa(bytes);
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
  /** Speculative decode (row `speculative-decode`): draft width K (also the
   * verify chunk). `>= 2` drafts the next tokens from the realized sequence's
   * recurrence and verifies them in one M=K pass. Works at ANY temperature —
   * the accept rule samples per absolute position, so the output is
   * byte-identical to plain decode at that temperature. */
  speculative_draft?: number;
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
  callback?: (text: string) => void,
): Promise<string> {
  await ensureReady();
  return G!.generate(holo, tokenizer ?? undefined, prompt, opts, callback);
}
