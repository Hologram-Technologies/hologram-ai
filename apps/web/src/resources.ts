// The parametric resource model (journey stage S1, dictionary row
// `memory-guard`): every estimate is a function of the model's own
// config.json and manifest sizes — never a per-model constant. Pure functions,
// unit-tested in isolation.

export interface ResourceEstimate {
  /** Bytes persisted to the OPFS κ-store (the shard bytes themselves). */
  storageBytes: number;
  /** The execution WINDOW: peak resident weight bytes (largest stage, F32)
   * plus activations — a function of the stage plan, never the model. */
  windowBytes: number;
  /** The context length the archives will be compiled at. */
  contextLength: number;
  /** Decoder layers per stage (windowed execution over k). */
  layersPerStage: number;
  /** Total stage archives (embedding + layer blocks + head), 1 = monolithic. */
  stageCount: number;
}

/** Bytes per element of a safetensors/torch dtype tag. An unrecognized dtype
 * fails loud: silently assuming 1 byte would under-count the parameter storage
 * and skew every downstream stage/context estimate for a model whose weights we
 * cannot actually size. */
export function dtypeBytes(dtype: string): number {
  switch (dtype.toUpperCase().replace("FLOAT", "F").replace("TORCH.", "")) {
    case "F64":
    case "I64":
    case "U64":
      return 8;
    case "F32":
    case "I32":
    case "U32":
      return 4;
    case "F16":
    case "BF16":
    case "I16":
    case "U16":
      return 2;
    case "F8_E4M3":
    case "F8_E5M2":
    case "I8":
    case "U8":
    case "BOOL":
      return 1;
    default:
      throw new Error(
        `unrecognized torch/safetensors dtype \`${dtype}\` — cannot size the model's weights (add its byte width rather than assuming one)`,
      );
  }
}

/** The model's own maximum context (positions) from config.json, resolved
 * across the architecture-specific aliases HF uses: `max_position_embeddings`
 * (Llama/Qwen/Mistral/Phi), `n_positions`/`n_ctx` (GPT-2 family),
 * `max_sequence_length`/`seq_length` (others). No silent default — a config
 * declaring none of these cannot yield a context window, so we fail loud rather
 * than bake an arbitrary 64-token window into the stage plan. */
export function resolveMaxPositions(config: Record<string, unknown>): number {
  const aliases = [
    "max_position_embeddings",
    "n_positions",
    "n_ctx",
    "max_sequence_length",
    "seq_length",
  ] as const;
  for (const key of aliases) {
    const v = config[key];
    if (typeof v === "number" && Number.isInteger(v) && v > 0) return v;
  }
  throw new Error(
    `config.json declares no context length (looked for ${aliases.join(", ")}) — ` +
      `cannot derive a context window without inventing an arbitrary one`,
  );
}

/**
 * The environment's memory budget in bytes. wasm32 caps a single memory at
 * 4 GiB; browsers commonly grant less per tab, so the budget is half of the
 * device memory (when the browser reports it) clamped to the wasm ceiling.
 */
export function environmentBudgetBytes(deviceMemoryGb?: number): number {
  const wasmCeiling = 4 * 1024 ** 3;
  const device = deviceMemoryGb && deviceMemoryGb > 0 ? deviceMemoryGb * 1024 ** 3 : wasmCeiling;
  return Math.min(wasmCeiling, device / 2);
}

/**
 * Largest context length (power of two, ≥ 64) not exceeding the model's own
 * `max_position_embeddings` whose activation estimate fits `budgetShare` of
 * the budget. Activations ≈ context × hidden × layers × 8 lanes × 4 bytes
 * (Q/K/V/attention/MLP intermediates at F32).
 */
export function chooseContextLength(
  config: { max_position_embeddings?: number; hidden_size: number; num_hidden_layers: number },
  budgetBytes: number,
  budgetShare = 0.25,
): number {
  requireNumeric(config, ["hidden_size", "num_hidden_layers"]);
  const maxPos = resolveMaxPositions(config);
  const lanes = 8;
  let context = 64;
  for (let candidate = 128; candidate <= maxPos; candidate *= 2) {
    const activations = candidate * config.hidden_size * config.num_hidden_layers * lanes * 4;
    if (activations > budgetBytes * budgetShare) break;
    context = candidate;
  }
  return Math.min(context, maxPos);
}

/**
 * The journey estimate for a model: shard storage, F32-inflated runtime
 * weights (the compiled pipeline computes in F32), and activations at the
 * chosen context.
 */
/** A config missing a required numeric key can never produce an estimate —
 * fail loud naming the key (a NaN estimate would silently pass the guard). */
function requireNumeric(config: Record<string, unknown>, keys: string[]): void {
  for (const key of keys) {
    const value = config[key];
    if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
      throw new Error(
        `config.json is missing required numeric key \`${key}\` — cannot derive a resource estimate (is this really the model config?)`,
      );
    }
  }
}

/**
 * The stage plan: the largest number of decoder layers per stage whose
 * F32-inflated weights fit half the window budget (materialization holds a
 * transient archive copy). The window bounds the STAGE, never the model —
 * the k-representation carries the rest in the OPFS κ-store.
 */
/** The F32 weight-byte decomposition of a model: per decoder layer, and the
 * embedding/head structural floors (a single tensor cannot be subdivided). */
export function weightDecomposition(
  config: {
    hidden_size: number;
    num_hidden_layers: number;
    vocab_size?: number;
    tie_word_embeddings?: boolean;
    torch_dtype?: string;
  },
  shardBytes: number,
): { layerBytes: number; embedBytes: number; headBytes: number; totalBytes: number } {
  requireNumeric(config as Record<string, unknown>, ["hidden_size", "num_hidden_layers"]);
  const paramsTotal = shardBytes / dtypeBytes(config.torch_dtype ?? "F32");
  const embedParams = (config.vocab_size ?? 0) * config.hidden_size;
  const headParams = config.tie_word_embeddings ? 0 : embedParams;
  const layerParams = Math.max(
    1,
    (paramsTotal - embedParams - headParams) / config.num_hidden_layers,
  );
  return {
    layerBytes: layerParams * 4,
    embedBytes: embedParams * 4,
    headBytes: headParams * 4,
    totalBytes: paramsTotal * 4,
  };
}

export function planStages(
  config: {
    hidden_size: number;
    num_hidden_layers: number;
    vocab_size?: number;
    tie_word_embeddings?: boolean;
    torch_dtype?: string;
  },
  shardBytes: number,
  windowBudgetBytes: number,
): { layersPerStage: number; stageCount: number; stageWeightBytes: number } {
  const w = weightDecomposition(config, shardBytes);
  const half = windowBudgetBytes / 2;
  let layersPerStage = Math.max(1, Math.floor(half / w.layerBytes));
  layersPerStage = Math.min(layersPerStage, config.num_hidden_layers);
  const layerStages = Math.ceil(config.num_hidden_layers / layersPerStage);
  const monolithic =
    layersPerStage >= config.num_hidden_layers && w.totalBytes <= half;
  const stageCount = monolithic ? 1 : layerStages + 2; // embedding + blocks + head
  const stageWeightBytes = Math.round(
    Math.max(layersPerStage * w.layerBytes, w.embedBytes, w.headBytes),
  );
  return { layersPerStage, stageCount, stageWeightBytes };
}

export function estimateResources(
  config: {
    max_position_embeddings?: number;
    hidden_size: number;
    num_hidden_layers: number;
    vocab_size?: number;
    tie_word_embeddings?: boolean;
    torch_dtype?: string;
  },
  shardBytes: number,
  windowBudgetBytes: number,
  stagePlanBudgetBytes: number = windowBudgetBytes,
): ResourceEstimate {
  // CONTEXT FIRST: the model's own max_position_embeddings is the invariant —
  // an artificially shortened context is an unexpected session limit. Staging
  // absorbs the memory scaling: layers-per-stage shrinks until a stage
  // (weights + activations at the model's own context) fits half the plan
  // budget (materialization holds a transient archive copy). The context
  // halves only when even a SINGLE-layer stage cannot carry its activations —
  // the structural floor, never a preference.
  const w = weightDecomposition(config, shardBytes);
  const half = stagePlanBudgetBytes / 2;
  const perLayerActivation = (ctx: number) => ctx * config.hidden_size * 8 * 4;

  let contextLength = resolveMaxPositions(config);
  while (contextLength > 64 && w.layerBytes + perLayerActivation(contextLength) > half) {
    contextLength = Math.floor(contextLength / 2);
  }
  const perLayer = w.layerBytes + perLayerActivation(contextLength);
  let layersPerStage = Math.max(1, Math.floor(half / perLayer));
  layersPerStage = Math.min(layersPerStage, config.num_hidden_layers);

  const monolithic =
    layersPerStage >= config.num_hidden_layers &&
    w.totalBytes + perLayerActivation(contextLength) * config.num_hidden_layers <= half;
  const stageCount = monolithic ? 1 : Math.ceil(config.num_hidden_layers / layersPerStage) + 2;
  const stageWeightBytes = Math.max(layersPerStage * w.layerBytes, w.embedBytes, w.headBytes);
  const activations = perLayerActivation(contextLength) * (monolithic ? config.num_hidden_layers : layersPerStage);
  return {
    storageBytes: shardBytes,
    windowBytes: Math.round(stageWeightBytes * 2 + activations),
    contextLength,
    layersPerStage,
    stageCount,
  };
}

/** The measured OPFS headroom. When the environment cannot measure
 * (no storage API), the headroom is unknown — reported as Infinity so the
 * cache stays best-effort rather than inventing a limit. */
export async function measuredStorageHeadroomBytes(): Promise<number> {
  if (!("storage" in navigator) || typeof navigator.storage.estimate !== "function") {
    return Number.POSITIVE_INFINITY;
  }
  const estimate = await navigator.storage.estimate();
  const quota = estimate.quota ?? 0;
  const usage = estimate.usage ?? 0;
  return Math.max(0, quota - usage);
}

/** The local cache budget: headroom minus a PROPORTIONAL safety margin for
 * archives/companions (never a fixed constant that could swallow a small
 * measured headroom). */
export function cacheBudgetFromHeadroom(headroomBytes: number): number {
  if (!Number.isFinite(headroomBytes)) return Number.POSITIVE_INFINITY;
  const margin = Math.min(64 * 1024 ** 2, Math.floor(headroomBytes / 10));
  return Math.max(0, headroomBytes - margin);
}

/** Human-readable byte figure for guard messages. */
export function formatBytes(bytes: number): string {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GiB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
  return `${Math.round(bytes / 1024)} KiB`;
}
