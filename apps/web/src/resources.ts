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

/** Bytes per element of a safetensors/torch dtype tag. */
export function dtypeBytes(dtype: string): number {
  switch (dtype.toUpperCase().replace("FLOAT", "F").replace("TORCH.", "")) {
    case "F64":
    case "I64":
      return 8;
    case "F32":
    case "I32":
      return 4;
    case "F16":
    case "BF16":
      return 2;
    default:
      return 1;
  }
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
  const maxPos = config.max_position_embeddings ?? 64;
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
  requireNumeric(config as Record<string, unknown>, ["hidden_size", "num_hidden_layers"]);
  const paramsTotal = shardBytes / dtypeBytes(config.torch_dtype ?? "F32");
  const embedParams = (config.vocab_size ?? 0) * config.hidden_size;
  const headParams = config.tie_word_embeddings ? 0 : embedParams;
  const layerParams = Math.max(
    1,
    (paramsTotal - embedParams - headParams) / config.num_hidden_layers,
  );
  const layerBytesF32 = layerParams * 4;
  const half = windowBudgetBytes / 2;
  let layersPerStage = Math.max(1, Math.floor(half / layerBytesF32));
  layersPerStage = Math.min(layersPerStage, config.num_hidden_layers);
  const layerStages = Math.ceil(config.num_hidden_layers / layersPerStage);
  const monolithic = layersPerStage >= config.num_hidden_layers && layerBytesF32 * config.num_hidden_layers + (embedParams + headParams) * 4 <= half;
  const stageCount = monolithic ? 1 : layerStages + 2; // embedding + blocks + head
  const stageWeightBytes = Math.round(
    Math.max(layersPerStage * layerBytesF32, embedParams * 4, headParams * 4),
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
  requireNumeric(config as Record<string, unknown>, ["hidden_size", "num_hidden_layers"]);
  const plan = planStages(config, shardBytes, stagePlanBudgetBytes);
  // Windowed over k: only a stage is ever resident, so the activation term
  // scales with the RESIDENT depth (layers per stage), never the model depth.
  const residentDepth = { ...config, num_hidden_layers: plan.layersPerStage };
  const contextLength = chooseContextLength(residentDepth, windowBudgetBytes);
  const activations = contextLength * config.hidden_size * plan.layersPerStage * 8 * 4;
  return {
    storageBytes: shardBytes,
    windowBytes: Math.round(plan.stageWeightBytes * 2 + activations),
    contextLength,
    layersPerStage: plan.layersPerStage,
    stageCount: plan.stageCount,
  };
}

/** The measured OPFS headroom — the only genuine storage bound. */
export async function measuredStorageHeadroomBytes(): Promise<number> {
  const estimate = await navigator.storage.estimate();
  const quota = estimate.quota ?? 0;
  const usage = estimate.usage ?? 0;
  return Math.max(0, quota - usage);
}

/** Human-readable byte figure for guard messages. */
export function formatBytes(bytes: number): string {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GiB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
  return `${Math.round(bytes / 1024)} KiB`;
}
