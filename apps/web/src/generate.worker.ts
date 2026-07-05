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
  createStagedSession,
  kappaRequirements,
  materialize,
  type StagedSession,
} from "./holo";

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
  session: StagedSession;
  handles: Map<string, OpenHandle>;
} | null = null;

function disposeWarm() {
  if (!warm) return;
  for (const handle of warm.handles.values()) {
    try {
      handle.close();
    } catch {
      // already closed by an invalidation — disposal proceeds
    }
  }
  try {
    warm.session.free();
  } catch {
    // wasm object already freed
  }
  warm = null;
}

async function warmStagedSession(
  modelDirName: string,
  tokenizerBytes: Uint8Array,
  onProgress: (line: string) => void,
): Promise<StagedSession> {
  if (warm && warm.modelDir === modelDirName) {
    onProgress("session warm — resident window carries across turns");
    return warm.session;
  }
  disposeWarm();
  onProgress("session cold — building the staged session");

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
  };
  const derivedEntries = await loadDerivedEntries(modelDir);

  const tensorsDir = await root.getDirectoryHandle("tensors");
  const kappaHandles = new Map<string, OpenHandle>();
  for (const kappa of manifest.kappas) {
    try {
      kappaHandles.set(kappa, await openSync(tensorsDir, `${kappa}.bin`));
    } catch {
      // Not in the local cache: the resolver falls back to the recorded
      // provenance; a κ that resolves nowhere aborts naming the label.
    }
  }
  const session = await createStagedSession(
    configJson,
    manifest,
    stagesMeta.contextLength,
    stagesMeta.layersPerStage,
    kappaResolver(kappaHandles, sources),
    kappaInvalidator(kappaHandles, tensorsDir),
    derivedStore(derivedEntries, modelDir),
    tokenizerBytes,
    onProgress,
  );
  warm = { modelDir: modelDirName, session, handles: kappaHandles };
  return session;
}

self.onmessage = async (e) => {
  const { holoBytes, modelDir, staged, prompt, genOpts, tokenizerBytes } = e.data;

  try {
    if (staged) {
      if (!tokenizerBytes) {
        throw new Error("staged chat needs the model's tokenizer.json");
      }
      const session = await warmStagedSession(modelDir, tokenizerBytes, (line: string) => {
        self.postMessage({ type: "progress", line });
      });
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
