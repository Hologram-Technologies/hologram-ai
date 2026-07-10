// The generation worker (journey stages S3 + S4): materialize the k-form
// archive against the OPFS κ-store, then run the real generation loop,
// streaming tokens back to the page.
//
// Materialization is resolution, not loading: each κ-map entry is read from
// `tensors/{κ}.bin` through a sync access handle (worker-only API), re-hashed
// inside the pipeline, and patched into the archive's constants. A missing or
// corrupt κ aborts the turn with the label.
import {
  generate as wasmGenerate,
  computeKappa,
  createDecodeSession,
  createStagedSession,
  deriveQuantizedArtifact,
  kappaRequirements,
  materialize,
  preferThreadedPool,
  ensureReady,
  poolInfo,
  type QuantEntry,
  type StagedSession,
} from "./holo";

// This is the execute worker: opt into the substrate's multi-threaded decode
// pool (ADR-0018). A no-op unless the page is cross-origin-isolated AND the
// threaded build loads; `holo` then spins up the worker pool over a shared
// linear memory on first `ensureReady()`. The main thread never does this (a
// blocking `Atomics.wait` is disallowed there); decode already runs here.
/** Emit the decode-pool status exactly once (on the first turn). */
let poolReported = false;
/** Set from a main-thread `pool-failed` message; the readiness poll checks it. */
let poolFailed: string | null = null;

async function materializeFromOpfs(holoBytes: Uint8Array, modelDir?: string): Promise<Uint8Array> {
  const required = await kappaRequirements(holoBytes);
  if (required.length === 0) return holoBytes;

  const sources: KappaSources = modelDir ? await loadKappaSources(modelDir) : {};
  const root = await navigator.storage.getDirectory();
  const tensorsDir = await root.getDirectoryHandle("tensors");

  // OPFS sync access handles are worker-only and missing from the DOM lib.
  interface SyncAccessHandle {
    getSize(): number;
    read(buffer: Uint8Array, options?: { at: number }): number;
    close(): void;
  }

  // Pre-open sync handles (opening is async; the wasm resolver must be sync).
  const handles = new Map<string, SyncAccessHandle>();
  try {
    for (const kappa of required) {
      try {
        const fh = await tensorsDir.getFileHandle(`${kappa}.bin`);
        handles.set(
          kappa,
          await (fh as unknown as { createSyncAccessHandle(): Promise<SyncAccessHandle> }).createSyncAccessHandle(),
        );
      } catch {
        // Not in the local cache: the resolver falls back to the recorded
        // provenance; a κ that resolves nowhere aborts naming the label.
      }
    }
    return await materialize(
      holoBytes,
      kappaResolver(handles, sources),
      kappaInvalidator(handles, tensorsDir),
    );
  } finally {
    for (const handle of handles.values()) handle.close();
  }
}

interface OpenHandle {
  getSize(): number;
  read(buffer: Uint8Array, options?: { at: number }): number;
  close(): void;
}

type KappaSources = Record<string, { url: string; start: number; end: number }>;

/** Load the model's recorded κ-provenance map (the resolver's network tier). */
async function loadKappaSources(modelDir: string): Promise<KappaSources> {
  const root = await navigator.storage.getDirectory();
  const models = await root.getDirectoryHandle("models");
  const dir = await models.getDirectoryHandle(modelDir);
  try {
    const handle = await dir.getFileHandle("kappa-sources.json");
    return JSON.parse(await (await handle.getFile()).text());
  } catch {
    return {};
  }
}

/** Synchronous ranged fetch (dedicated workers may block): the κ resolver is
 * synchronous inside wasm, so the network tier rides a sync XHR with the
 * binary-via-latin1 encoding. Content addressing makes this exactly as
 * trustworthy as the local cache — the pipeline re-hashes every buffer. */
function syncFetchRange(url: string, start: number, end: number): Uint8Array {
  const xhr = new XMLHttpRequest();
  xhr.open("GET", url, false);
  xhr.setRequestHeader("Range", `bytes=${start}-${end - 1}`);
  xhr.overrideMimeType("text/plain; charset=x-user-defined");
  xhr.send();
  if (xhr.status !== 206 && xhr.status !== 200) {
    throw new Error(`provenance fetch failed (HTTP ${xhr.status}) for ${url}`);
  }
  const text = xhr.responseText;
  const offset = xhr.status === 200 ? start : 0;
  const out = new Uint8Array(end - start);
  for (let i = 0; i < out.length; i++) out[i] = text.charCodeAt(offset + i) & 0xff;
  return out;
}

/** Build the synchronous κ resolver: local OPFS cache first, then the
 * recorded provenance. A κ that resolves nowhere returns undefined and the
 * pipeline aborts naming the label. */
function kappaResolver(
  handles: Map<string, OpenHandle>,
  sources: KappaSources,
): (kappa: string) => Uint8Array | undefined {
  return (kappa: string) => {
    const handle = handles.get(kappa);
    if (handle) return readAll(handle);
    const source = sources[kappa];
    if (!source) return undefined;
    return syncFetchRange(source.url, source.start, source.end);
  };
}

/** The seekable tier of sub-tensor κ-resolution (row `chunked-head`): a
 * session-verified ranged binding moves ONLY its slice — an OPFS `read({at})`
 * from the cache tier, or a ranged GET inside the recorded provenance span.
 * Returning undefined falls back to whole-resolve + slice in wasm (first
 * touch always resolves whole and verifies; this tier is read-only I/O). */
function kappaRangeResolver(
  handles: Map<string, OpenHandle>,
  sources: KappaSources,
): (kappa: string, offset: number, len: number) => Uint8Array | undefined {
  return (kappa: string, offset: number, len: number) => {
    const handle = handles.get(kappa);
    if (handle) {
      const buf = new Uint8Array(len);
      if (handle.read(buf, { at: offset }) === len) return buf;
      return undefined;
    }
    const source = sources[kappa];
    if (!source) return undefined;
    return syncFetchRange(source.url, source.start + offset, source.start + offset + len);
  };
}

/** Overlay an in-memory content tier (session-held artifacts a saturated
 * quota refused to persist) over the base resolver. */
function overlayResolver(
  overlay: Map<string, Uint8Array>,
  base: (kappa: string) => Uint8Array | undefined,
): (kappa: string) => Uint8Array | undefined {
  if (overlay.size === 0) return base;
  return (kappa: string) => overlay.get(kappa) ?? base(kappa);
}

/** Ranged reads over the same in-memory overlay. */
function overlayRangeResolver(
  overlay: Map<string, Uint8Array>,
  base: (kappa: string, offset: number, len: number) => Uint8Array | undefined,
): (kappa: string, offset: number, len: number) => Uint8Array | undefined {
  if (overlay.size === 0) return base;
  return (kappa: string, offset: number, len: number) => {
    const bytes = overlay.get(kappa);
    if (bytes) return bytes.subarray(offset, offset + len);
    return base(kappa, offset, len);
  };
}

/** The UNPIN hook (row `saturation-residency`): a κ whose cached content
 * failed verification is evicted — the open handle drops (so the resolver's
 * next call falls through to the recorded provenance) and the OPFS entry
 * evaporates (fire-and-forget: sync-context callers cannot await, and the
 * handle removal alone already redirects resolution). Corrupted content
 * leaves the cache by the same law that admitted it. */
function kappaInvalidator(
  handles: Map<string, OpenHandle>,
  tensorsDir: FileSystemDirectoryHandle,
): (kappa: string) => void {
  return (kappa: string) => {
    const handle = handles.get(kappa);
    if (handle) {
      try {
        handle.close();
      } catch {
        // already closed — eviction proceeds regardless
      }
      handles.delete(kappa);
    }
    void tensorsDir.removeEntry(`${kappa}.bin`).catch(() => {
      // nothing to evaporate — the redirect to provenance already happened
    });
  };
}

async function openSync(dir: FileSystemDirectoryHandle, name: string): Promise<OpenHandle> {
  const fh = await dir.getFileHandle(name);
  return (fh as unknown as { createSyncAccessHandle(): Promise<OpenHandle> }).createSyncAccessHandle();
}

function readAll(handle: OpenHandle): Uint8Array {
  const buf = new Uint8Array(handle.getSize());
  handle.read(buf, { at: 0 });
  return buf;
}

/** Read a small JSON/text file from the model dir (async read is fine here —
 * these run before the synchronous generation loop starts). */
async function readModelText(dir: FileSystemDirectoryHandle, name: string): Promise<string> {
  const handle = await dir.getFileHandle(name);
  return (await handle.getFile()).text();
}

type DerivedEntry = { stages: Uint8Array[]; kappas: string[] };

/** Preload the model's derived-artifact store (row `derived-artifact-kappa`):
 * `derived/{key}/{i}.holo` + `kappas.json`, written by earlier sessions. The
 * wasm derived-store callbacks are synchronous, so entries load up front
 * (weightless k-forms — tens of KB per window). */
async function loadDerivedEntries(
  modelDir: FileSystemDirectoryHandle,
): Promise<Map<string, DerivedEntry>> {
  const entries = new Map<string, DerivedEntry>();
  let derivedDir: FileSystemDirectoryHandle;
  try {
    derivedDir = await modelDir.getDirectoryHandle("derived");
  } catch {
    return entries;
  }
  // The async-iterable OPFS surface is missing from the webworker lib.
  const iter = (
    derivedDir as unknown as {
      entries(): AsyncIterableIterator<[string, FileSystemHandle]>;
    }
  ).entries();
  for await (const [key, handle] of iter) {
    if (handle.kind !== "directory") continue;
    try {
      const dir = handle as FileSystemDirectoryHandle;
      const kappas = JSON.parse(
        await (await (await dir.getFileHandle("kappas.json")).getFile()).text(),
      ) as string[];
      const stages: Uint8Array[] = [];
      for (let i = 0; i < kappas.length; i++) {
        const file = await (await dir.getFileHandle(`${i}.holo`)).getFile();
        stages.push(new Uint8Array(await file.arrayBuffer()));
      }
      entries.set(key, { stages, kappas });
    } catch {
      // A torn entry loads as absent; the session re-derives and rewrites.
    }
  }
  return entries;
}

/** The derived-store callbacks over the preloaded map: `store` persists a
 * fresh derivation asynchronously (a lost write only costs re-derivation);
 * `evaporate` unpins a corrupted entry from map and disk. */
function derivedStore(entries: Map<string, DerivedEntry>, modelDir: FileSystemDirectoryHandle) {
  const persist = async (key: string, entry: DerivedEntry) => {
    const derivedDir = await modelDir.getDirectoryHandle("derived", { create: true });
    const dir = await derivedDir.getDirectoryHandle(key, { create: true });
    for (let i = 0; i < entry.stages.length; i++) {
      const handle = await dir.getFileHandle(`${i}.holo`, { create: true });
      const writable = await handle.createWritable();
      await writable.write(entry.stages[i] as unknown as ArrayBufferView<ArrayBuffer>);
      await writable.close();
    }
    const meta = await dir.getFileHandle("kappas.json", { create: true });
    const writable = await meta.createWritable();
    await writable.write(JSON.stringify(entry.kappas));
    await writable.close();
  };
  return {
    load: (key: string) => entries.get(key),
    store: (key: string, stages: Uint8Array[], kappas: string[]) => {
      const entry = { stages, kappas };
      entries.set(key, entry);
      void persist(key, entry).catch(() => {});
    },
    evaporate: (key: string) => {
      entries.delete(key);
      void modelDir
        .getDirectoryHandle("derived")
        .then((d) => d.removeEntry(key, { recursive: true }))
        .catch(() => {});
    },
  };
}

// The persistent staged session (row `warm-turn`): the worker outlives a
// single send, and the session — compiled window, resident stage sessions,
// verified-κ set, derived-artifact cache — carries across turns. A warm turn
// pays decode: no window recompile, no stage rematerialization, no
// re-verification. The session rebuilds on model switch (its inputs are the
// model's), and dies with the worker on cancel/error — the next send is a
// cold turn, same semantics.
let warm: {
  modelDir: string;
  plan: "decode" | "window";
  session: StagedSession;
  handles: Map<string, OpenHandle>;
  /** The paired speculative DRAFT model this session was warmed with (row
   * `speculative-draft-pairing`), if any. Its session is OWNED by `session`
   * (absorbed by `attach_draft`), so it is not freed separately — but its OPFS
   * κ handles were opened by this worker and must be closed on disposal. */
  draftModelDir?: string;
  draftHandles?: Map<string, OpenHandle>;
} | null = null;

function closeHandles(handles: Map<string, OpenHandle> | undefined) {
  if (!handles) return;
  for (const handle of handles.values()) {
    try {
      handle.close();
    } catch {
      // already closed by an invalidation — disposal proceeds
    }
  }
}

function disposeWarm() {
  if (!warm) return;
  closeHandles(warm.handles);
  closeHandles(warm.draftHandles);
  try {
    // Freeing the target frees the draft growable it owns (Rc drop).
    warm.session.free();
  } catch {
    // wasm object already freed
  }
  warm = null;
}

/** Re-derive any missing quantized artifact from its wide κ (OPFS or the
 * recorded provenance) and persist it — derive-as-recovery: the artifact's
 * recorded κ pins the result, a mismatch fails loud naming the label. */
async function ensureQuantArtifacts(
  tensorsDir: FileSystemDirectoryHandle,
  quant: QuantEntry[],
  manifest: { keys: string[]; kappas: string[]; shapes: string[]; dtypes: string[] },
  sources: KappaSources,
  onProgress: (line: string) => void,
): Promise<Map<string, Uint8Array>> {
  // Artifacts a saturated quota refuses to persist ride in memory for the
  // session (the resolver's first tier) — the journey never dead-ends.
  const inMemory = new Map<string, Uint8Array>();
  for (const entry of quant) {
    try {
      await tensorsDir.getFileHandle(`${entry.artifact}.bin`);
      continue;
    } catch {
      // Missing (evaporated under pressure): recover by deriving.
    }
    const idx = manifest.kappas.indexOf(entry.wide);
    if (idx < 0) throw new Error(`quant map names κ \`${entry.wide}\` outside the manifest`);
    let wide: Uint8Array;
    try {
      const handle = await tensorsDir.getFileHandle(`${entry.wide}.bin`);
      wide = new Uint8Array(await (await handle.getFile()).arrayBuffer());
    } catch {
      const source = sources[entry.wide];
      if (!source) {
        throw new Error(`κ \`${entry.wide}\` resolves nowhere: not cached, no recorded provenance`);
      }
      const res = await fetch(source.url, {
        headers: { Range: `bytes=${source.start}-${source.end - 1}` },
      });
      if (!res.ok && res.status !== 206) {
        throw new Error(`provenance fetch failed (HTTP ${res.status}) for κ \`${entry.wide}\``);
      }
      const body = new Uint8Array(await res.arrayBuffer());
      wide = res.status === 200 ? body.slice(source.start, source.end) : body;
    }
    // A head chunk derives from a BYTE RANGE of the wide head/embedding tensor,
    // not the whole tensor: slice to its range before deriving.
    if (entry.offset != null && entry.len != null) {
      wide = wide.slice(entry.offset, entry.offset + entry.len);
    }
    const artifact = await deriveQuantizedArtifact(wide, manifest.dtypes[idx], entry.out, entry.in);
    const kappa = await computeKappa(artifact);
    if (kappa !== entry.artifact) {
      throw new Error(
        `re-derived artifact κ \`${kappa}\` does not reproduce the recorded \`${entry.artifact}\``,
      );
    }
    // Crystallize before writing: a lingering wide blob is gas-phase and must
    // not hold the quota against its own artifact — EXCEPT a head chunk's wide
    // κ, which a tied head shares with the embedding Gather (and the sibling
    // chunks) and so stays load-bearing.
    if (entry.offset == null) {
      await tensorsDir.removeEntry(`${entry.wide}.bin`).catch(() => {});
    }
    try {
      const handle = await tensorsDir.getFileHandle(`${entry.artifact}.bin`, { create: true });
      const writable = await handle.createWritable();
      await writable.write(artifact as unknown as ArrayBufferView<ArrayBuffer>);
      await writable.close();
    } catch {
      await tensorsDir.removeEntry(`${entry.artifact}.bin`).catch(() => {});
      inMemory.set(entry.artifact, artifact);
      onProgress(
        `quantized artifact held in memory (quota refused the write): ${entry.artifact.slice(0, 24)}…`,
      );
      continue;
    }
    onProgress(`quantized artifact re-derived (derive-as-recovery): ${entry.artifact.slice(0, 24)}…`);
  }
  return inMemory;
}

/** Build ONE staged/decode session from a model's OPFS dir (manifest, κ-store,
 * quant tier, derived store, weight-tier paging) and return it with the OPFS
 * sync handles it holds — WITHOUT touching the warm cache. The shared back half
 * of `warmStagedSession`, so the target and its paired draft build identically. */
async function buildSession(
  modelDirName: string,
  tokenizerBytes: Uint8Array,
  onProgress: (line: string) => void,
  plan: "decode" | "window",
  weightBudget: number | undefined,
): Promise<{ session: StagedSession; handles: Map<string, OpenHandle> }> {
  const root = await navigator.storage.getDirectory();
  const models = await root.getDirectoryHandle("models");
  const modelDir = await models.getDirectoryHandle(modelDirName);
  const sources = await loadKappaSources(modelDirName);

  const configJson = await readModelText(modelDir, "config.json");
  const manifest = JSON.parse(await readModelText(modelDir, "manifest.json")) as {
    keys: string[];
    kappas: string[];
    shapes: string[];
    dtypes: string[];
  };
  const stagesMeta = JSON.parse(await readModelText(modelDir, "stages.json")) as {
    layersPerStage: number;
    contextLength?: number;
    quant?: QuantEntry[];
  };
  const derivedEntries = await loadDerivedEntries(modelDir);

  const tensorsDir = await root.getDirectoryHandle("tensors");
  // Quantized tier: every artifact must be resident before the session
  // opens sync handles — a missing artifact re-derives from the wide form
  // through recorded provenance (derive-as-recovery, fail-closed on its κ).
  const quant = stagesMeta.quant ?? [];
  let inMemoryArtifacts = new Map<string, Uint8Array>();
  if (quant.length) {
    inMemoryArtifacts = await ensureQuantArtifacts(tensorsDir, quant, manifest, sources, onProgress);
    onProgress(`quantized tier (int8): ${quant.length} projection artifact(s) bound`);
  }
  const kappaHandles = new Map<string, OpenHandle>();
  const sessionKappas = [...manifest.kappas, ...quant.map((q) => q.artifact)];
  for (const kappa of sessionKappas) {
    try {
      kappaHandles.set(kappa, await openSync(tensorsDir, `${kappa}.bin`));
    } catch {
      // Not in the local cache: the resolver falls back to the recorded
      // provenance; a κ that resolves nowhere aborts naming the label.
    }
  }
  // Weight-tier pager (row `lazy-constant-residency`): when a budget is set,
  // each stage loads paged against it, sizing paged constants from the open
  // sync handle's `getSize()` (an OPFS stat, never a body read).
  const sizeKappa = weightBudget
    ? (kappa: string) => kappaHandles.get(kappa)?.getSize()
    : undefined;
  if (weightBudget) {
    onProgress(`weight-tier paging: stages resident within ${(weightBudget / (1024 * 1024)).toFixed(0)} MB`);
  }

  const create = plan === "decode" ? createDecodeSession : createStagedSession;
  const session = await create(
    configJson,
    manifest,
    stagesMeta.contextLength,
    stagesMeta.layersPerStage,
    overlayResolver(inMemoryArtifacts, kappaResolver(kappaHandles, sources)),
    kappaInvalidator(kappaHandles, tensorsDir),
    overlayRangeResolver(inMemoryArtifacts, kappaRangeResolver(kappaHandles, sources)),
    quant.length ? quant : undefined,
    derivedStore(derivedEntries, modelDir),
    weightBudget,
    sizeKappa,
    tokenizerBytes,
    onProgress,
  );
  return { session, handles: kappaHandles };
}

async function warmStagedSession(
  modelDirName: string,
  tokenizerBytes: Uint8Array,
  onProgress: (line: string) => void,
  plan: "decode" | "window",
  weightBudget: number | undefined,
  draftModelDir: string | undefined,
): Promise<StagedSession> {
  if (
    warm &&
    warm.modelDir === modelDirName &&
    warm.plan === plan &&
    warm.draftModelDir === draftModelDir
  ) {
    onProgress("session warm — resident window carries across turns");
    return warm.session;
  }
  disposeWarm();
  onProgress(
    plan === "decode"
      ? "session cold — building the decode-plan session (one position per token)"
      : "session cold — building the staged session",
  );

  const { session, handles } = await buildSession(
    modelDirName,
    tokenizerBytes,
    onProgress,
    plan,
    weightBudget,
  );

  // Speculative draft pairing (row `speculative-draft-pairing`): build the
  // paired draft as a SECOND session and `attach_draft` it, so speculative
  // decode drafts from the paired model. The attach shares ONE residency ledger
  // across the pair (no over-commit) and refuses an incompatible vocabulary — a
  // failure at any point degrades to prompt-lookup, never a dead-ended turn. The
  // draft's tokenizer is irrelevant (it consumes the target's ids), so the
  // target's tokenizer bytes construct it. Only the decode plan speculates.
  let draftHandles: Map<string, OpenHandle> | undefined;
  if (draftModelDir && plan === "decode" && typeof session.attach_draft === "function") {
    try {
      onProgress(`draft model: building the paired speculative drafter (${draftModelDir})…`);
      const draft = await buildSession(
        draftModelDir,
        tokenizerBytes,
        onProgress,
        plan,
        weightBudget,
      );
      // `attach_draft` CONSUMES the draft session (its growable lives on inside
      // the target); keep its κ handles open for the target's lifetime.
      session.attach_draft(draft.session);
      draftHandles = draft.handles;
      onProgress("draft model attached — speculative decode drafts from the paired model");
    } catch (e) {
      closeHandles(draftHandles);
      draftHandles = undefined;
      onProgress(`draft model unavailable (drafting by prompt-lookup instead): ${e}`);
    }
  }

  warm = { modelDir: modelDirName, plan, session, handles, draftModelDir, draftHandles };
  return session;
}

self.onmessage = async (e) => {
  // Control message from the main thread: a pool worker failed to instantiate.
  // Record it so the readiness poll in `holo.initThreaded` fails fast and falls
  // back to single-threaded (ADR-0018 M2). Not a generation request.
  if (e.data && e.data.type === "pool-failed") {
    poolFailed = e.data.why || "pool worker failed";
    return;
  }
  const {
    holoBytes,
    modelDir,
    staged,
    prompt,
    genOpts,
    tokenizerBytes,
    decodePlan,
    weightBudget,
    draftModelDir,
  } = e.data;

  try {
    // Establish the wasm binding once and report the decode-pool status as a
    // progress line (recorded in `__hologram_status`) — a page-observable signal
    // that the threaded pool actually engaged, since dedicated-worker console is
    // not surfaced to the page. See ADR-0018 / the threaded probes.
    if (!poolReported) {
      poolReported = true;
      // Opt into the multi-threaded decode pool unless disabled by config
      // (`threads === false`, from `hologram_threads=0`) OR the window plan is in
      // use (`decodePlan === false`): the pool only parallelises the `m == 1`
      // per-token decode GEMV, so a window (m > 1) forward gains nothing and
      // should not pay N workers + shared memory (ADR-0018 m8). Must precede the
      // first `ensureReady()`. A no-op unless the page is cross-origin-isolated.
      //
      // The pool workers are spawned + owned by the MAIN thread (via `spawn`),
      // not here, so terminating THIS worker on cancel/error tears the pool down
      // instead of orphaning it (ADR-0018 C1). `module` + the shared `memory` are
      // structured-cloneable across postMessage. `teardown` cleans up on fallback
      // (M1); `failure` surfaces a pool worker's init failure to the poll (M2).
      preferThreadedPool(e.data.threads !== false && e.data.decodePlan !== false, {
        spawn: (module, memory, n, stackSize) =>
          self.postMessage({ type: "spawn-pool", module, memory, n, stackSize }),
        teardown: () => self.postMessage({ type: "pool-teardown" }),
        failure: () => poolFailed,
      });
      await ensureReady();
      const p = poolInfo();
      // Tell the main thread the pool is committed (past registration): a pool
      // worker that dies now is a mid-session fault, so the main thread must
      // ABORT the turn rather than fall back (ADR-0018 M3).
      if (p.threaded) self.postMessage({ type: "pool-committed" });
      self.postMessage({
        type: "progress",
        line: p.threaded
          ? `pool: multi-threaded decode active (${p.workers} workers)`
          : `pool: single-threaded decode`,
      });
    }
    if (staged) {
      if (!tokenizerBytes) {
        throw new Error("staged chat needs the model's tokenizer.json");
      }
      const session = await warmStagedSession(
        modelDir,
        tokenizerBytes,
        (line: string) => {
          self.postMessage({ type: "progress", line });
        },
        decodePlan === false ? "window" : "decode",
        typeof weightBudget === "number" && weightBudget > 0 ? weightBudget : undefined,
        typeof draftModelDir === "string" && draftModelDir ? draftModelDir : undefined,
      );
      const result = session.generate(prompt, genOpts as never, (text: string) => {
        self.postMessage({ type: "token", text });
      });
      self.postMessage({ type: "done", text: result });
      // Idle anneal (row `idle-derivation`): with the turn delivered, derive
      // the next window bucket's archives into the derived store — off the
      // per-token path, no weights moved. A later crossing resolves instead
      // of compiling. Failure is inert: speculation is never load-bearing.
      setTimeout(() => {
        try {
          const bucket = session.prederive_next_window();
          if (bucket !== undefined) {
            self.postMessage({ type: "progress", line: `idle: pre-derived the ${bucket}-token window` });
          }
        } catch {
          // abandoned speculation — the next crossing derives on demand
        }
      }, 0);
      return;
    }
    const material = await materializeFromOpfs(holoBytes, modelDir);
    const result = await wasmGenerate(
      material,
      prompt,
      genOpts,
      tokenizerBytes,
      (text: string) => {
        self.postMessage({ type: "token", text });
      },
    );
    self.postMessage({ type: "done", text: result });
  } catch (err) {
    self.postMessage({ type: "error", error: String(err) });
  }
};
