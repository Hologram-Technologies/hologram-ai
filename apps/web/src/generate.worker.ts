// The generation worker (journey stages S3 + S4): materialize the k-form
// archive against the OPFS κ-store, then run the real generation loop,
// streaming tokens back to the page.
//
// Materialization is resolution, not loading: each κ-map entry is read from
// `tensors/{κ}.bin` through a sync access handle (worker-only API), re-hashed
// inside the pipeline, and patched into the archive's constants. A missing or
// corrupt κ aborts the turn with the label.
import { generate as wasmGenerate, generateStaged, kappaRequirements, materialize } from "./holo";

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

// Staged generation with a sequence-following window (row
// `staged-window-growth`): the session recompiles the weightless stage
// k-forms per geometric window bucket from the recorded manifest, so
// per-token compute scales with the actual sequence — never the model's full
// context. κs resolve on demand through pre-opened sync handles (falling
// back to recorded provenance); one stage's weights are resident at a time.
async function runStaged(
  staged: { modelDir: string },
  prompt: string,
  genOpts: unknown,
  tokenizerBytes: Uint8Array | undefined,
  onToken: (text: string) => void,
  onProgress: (line: string) => void,
): Promise<string> {
  const root = await navigator.storage.getDirectory();
  const models = await root.getDirectoryHandle("models");
  const modelDir = await models.getDirectoryHandle(staged.modelDir);
  const sources = await loadKappaSources(staged.modelDir);

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

  const tensorsDir = await root.getDirectoryHandle("tensors");
  const kappaHandles = new Map<string, OpenHandle>();
  try {
    for (const kappa of manifest.kappas) {
      try {
        kappaHandles.set(kappa, await openSync(tensorsDir, `${kappa}.bin`));
      } catch {
        // Not in the local cache: the resolver falls back to the recorded
        // provenance; a κ that resolves nowhere aborts naming the label.
      }
    }
    return await generateStaged(
      configJson,
      manifest,
      stagesMeta.contextLength,
      stagesMeta.layersPerStage,
      kappaResolver(kappaHandles, sources),
      kappaInvalidator(kappaHandles, tensorsDir),
      prompt,
      genOpts as never,
      tokenizerBytes,
      onToken,
      onProgress,
    );
  } finally {
    for (const handle of kappaHandles.values()) handle.close();
  }
}

self.onmessage = async (e) => {
  const { holoBytes, modelDir, staged, prompt, genOpts, tokenizerBytes } = e.data;

  try {
    if (staged) {
      const result = await runStaged(
        { modelDir },
        prompt,
        genOpts,
        tokenizerBytes,
        (text: string) => {
          self.postMessage({ type: "token", text });
        },
        (line: string) => {
          self.postMessage({ type: "progress", line });
        },
      );
      self.postMessage({ type: "done", text: result });
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
