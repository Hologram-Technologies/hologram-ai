// The parametric resource model (journey stage S1, dictionary row
// `memory-guard`): every estimate is a function of the model's own
// config.json and manifest sizes — never a per-model constant. Pure functions,
// unit-tested in isolation.

export interface ResourceEstimate {
  /** Bytes persisted to the OPFS κ-store (the shard bytes themselves). */
  storageBytes: number;
  /** Peak runtime working set: F32-inflated weights + activations at the
   * chosen context length. */
  runtimeBytes: number;
  /** The context length the archive will be compiled at. */
  contextLength: number;
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

export function estimateResources(
  config: {
    max_position_embeddings?: number;
    hidden_size: number;
    num_hidden_layers: number;
    torch_dtype?: string;
  },
  shardBytes: number,
  budgetBytes: number,
): ResourceEstimate {
  requireNumeric(config as Record<string, unknown>, ["hidden_size", "num_hidden_layers"]);
  const contextLength = chooseContextLength(config, budgetBytes);
  const inflation = 4 / dtypeBytes(config.torch_dtype ?? "F32");
  const weightsF32 = shardBytes * inflation;
  const activations = contextLength * config.hidden_size * config.num_hidden_layers * 8 * 4;
  return {
    storageBytes: shardBytes,
    // Weights resident in the session pool + one transient archive copy
    // during materialization, plus activations.
    runtimeBytes: Math.round(weightsF32 * 2 + activations),
    contextLength,
  };
}

/** Human-readable byte figure for guard messages. */
export function formatBytes(bytes: number): string {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GiB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
  return `${Math.round(bytes / 1024)} KiB`;
}
