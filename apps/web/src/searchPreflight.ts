// Search-time derivability preflight (row `supported-search`): discovery
// surfaces EVERY model the backend can actually derive, and honestly refuses
// the rest. The ONLY filter authority is the same config-only preflight the
// download runs (`validate_model_config` → hologram-ai-safetensors::
// parametric::validate_config, via the wasm binding) — NO architecture names
// live in this path (the no-heuristics/parametric law). A refused model is
// SHOWN, greyed out, carrying the preflight's own message verbatim: an honest
// refusal is information, never silently hidden.
//
// Pure/injectable module: the component passes the real `validateModelConfig`
// and the browser `fetch`; tests pass mocks — no wasm needed to witness the
// decision logic.

/** One repo file from the HF listing (`siblings` — present when the search
 * request asks for the full listing). */
export interface SearchSibling {
  rfilename: string;
}

/** The cheap first pass (no extra requests): a repo whose file listing shows
 * NO safetensors export can never start the download journey (the downloader
 * refuses ONNX-only repos), so it is dropped before spending a config fetch.
 * A listing WITHOUT file info cannot prove absence — the candidate stays in
 * and the config probe (the real authority) decides. */
export function lacksSafetensorsExport(siblings?: SearchSibling[] | null): boolean {
  if (!Array.isArray(siblings)) return false;
  return !siblings.some(
    (s) => typeof s?.rfilename === "string" && s.rfilename.endsWith(".safetensors"),
  );
}

/** The preflight verdict for one candidate. A refusal carries the preflight's
 * own message verbatim; `transient` marks a network failure (nothing was
 * learned about the MODEL, so the verdict is not cached). */
export type ProbeOutcome =
  | { status: "derivable" }
  | { status: "refused"; reason: string; transient?: boolean };

/** A search row's probe lifecycle: rows render immediately as `probing` and
 * upgrade in place when their probe resolves. */
export type ProbeState = ProbeOutcome | { status: "probing" };

/** In-flight probe bound — a HOST courtesy constant (concurrent fetches
 * against the hub), never a model/input parameter. */
export const PROBE_CONCURRENCY = 4;

/** The wasm preflight throws a plain string (`JsValue::from_str`); a JS
 * mock may throw an `Error`. Either way the message is surfaced VERBATIM. */
function reasonOf(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/**
 * Probe one candidate: fetch its `config.json` (the same resolve URL base the
 * downloader uses) and run the injected preflight. Never throws — every
 * failure mode is an honest refusal naming what failed:
 *  - missing/unfetchable config.json → refused naming the file and HTTP status;
 *  - network failure → refused (transient — not cached);
 *  - preflight rejection → refused with the preflight's message verbatim.
 */
export async function probeCandidate(
  configUrl: string,
  validate: (configText: string) => void | Promise<void>,
  fetchImpl: typeof fetch = fetch,
): Promise<ProbeOutcome> {
  let res: Response;
  try {
    res = await fetchImpl(configUrl);
  } catch (e) {
    return {
      status: "refused",
      reason: `config.json unreachable (${reasonOf(e)}) — the preflight cannot run`,
      transient: true,
    };
  }
  if (!res.ok) {
    return {
      status: "refused",
      reason: `config.json missing (HTTP ${res.status}) — the preflight cannot run without the model config`,
    };
  }
  const configText = await res.text();
  try {
    await validate(configText);
    return { status: "derivable" };
  } catch (e) {
    return { status: "refused", reason: reasonOf(e) };
  }
}

// ── probe cache ──────────────────────────────────────────────────────────────

const STORAGE_KEY = "hologram_search_preflight_v1";

type StorageLike = Pick<Storage, "getItem" | "setItem">;

function defaultStorage(): StorageLike | null {
  try {
    return typeof sessionStorage !== "undefined" ? sessionStorage : null;
  } catch {
    return null; // storage denied (e.g. sandboxed frame) — Map-only cache
  }
}

/** Probe-verdict cache keyed by repo id + revision (the listing's `sha` when
 * it carries one, else `main`): a Map for the page's lifetime, mirrored to
 * sessionStorage so a re-search in the same session re-fetches nothing. */
export class ProbeCache {
  private map = new Map<string, ProbeOutcome>();
  private storage: StorageLike | null;

  constructor(storage?: StorageLike | null) {
    this.storage = storage === undefined ? defaultStorage() : storage;
    try {
      const raw = this.storage?.getItem(STORAGE_KEY);
      if (raw) {
        for (const [k, v] of Object.entries(JSON.parse(raw) as Record<string, ProbeOutcome>)) {
          this.map.set(k, v);
        }
      }
    } catch {
      // A corrupt mirror never blocks probing — start empty.
    }
  }

  static key(id: string, revision?: string): string {
    return `${id}@${revision ?? "main"}`;
  }

  get(id: string, revision?: string): ProbeOutcome | undefined {
    return this.map.get(ProbeCache.key(id, revision));
  }

  set(id: string, revision: string | undefined, outcome: ProbeOutcome): void {
    this.map.set(ProbeCache.key(id, revision), outcome);
    try {
      this.storage?.setItem(STORAGE_KEY, JSON.stringify(Object.fromEntries(this.map)));
    } catch {
      // Mirror write failure (quota) — the in-memory cache still serves.
    }
  }
}

/** {@link probeCandidate} behind the cache: a hit costs no fetch; a fresh
 * verdict is cached unless it was transient (a network failure says nothing
 * about the model, so the next search retries it). */
export async function probeWithCache(
  id: string,
  revision: string | undefined,
  configUrl: string,
  cache: ProbeCache,
  validate: (configText: string) => void | Promise<void>,
  fetchImpl: typeof fetch = fetch,
): Promise<ProbeOutcome> {
  const hit = cache.get(id, revision);
  if (hit) return hit;
  const outcome = await probeCandidate(configUrl, validate, fetchImpl);
  if (!(outcome.status === "refused" && outcome.transient)) {
    cache.set(id, revision, outcome);
  }
  return outcome;
}

// ── bounded concurrency ──────────────────────────────────────────────────────

/** Run `fn` over `items` with at most `limit` in flight — probes must neither
 * stampede the hub nor serialize the whole result list. Resolves when every
 * item has been processed; `fn` is expected not to throw (probe outcomes are
 * values, never exceptions). */
export async function mapBounded<T>(
  items: readonly T[],
  limit: number,
  fn: (item: T, index: number) => Promise<void>,
): Promise<void> {
  let next = 0;
  const lanes = Array.from({ length: Math.max(1, Math.min(limit, items.length)) }, async () => {
    for (;;) {
      const i = next++;
      if (i >= items.length) return;
      await fn(items[i], i);
    }
  });
  await Promise.all(lanes);
}
