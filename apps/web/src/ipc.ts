import { computeKappa, countTokens, validateModelConfig } from "./holo";
import { type GenOpts } from "./holo";
import GenerateWorker from "./generate.worker?worker";
import { cacheBudgetFromHeadroom, environmentBudgetBytes, estimateResources, formatBytes, measuredStorageHeadroomBytes } from "./resources";

export interface WorkspacePaths {
  root: string;
  modelsDir: string;
  outputDir: string;
  hologramAiBin: string | null;
}

export type Modality = "text-chat";

export interface KnownModelStatus {
  id: string;
  hfId: string;
  displayName: string;
  description: string;
  modality: Modality;
  size: string;
  approxArchiveMb: number;
  quantize: string;
  promptTemplate: string | null;
  stop: string[];
  chatTurnSeparator: string | null;
  maxTokens?: number;
  localDir: string | null;
  downloaded: boolean;
  compiledArchive: string | null;
}

export interface CompiledArchive {
  name: string;
  path: string;
  sizeBytes: number;
}

export interface LogEntry {
  timestampMs: number;
  level: string;
  target: string;
  message: string;
}

export interface LogsResponse {
  entries: LogEntry[];
  nextIndex: number;
}

export interface ProcessLine {
  stream: "stdout" | "stderr";
  line: string;
}

type CatalogueEntry = Omit<KnownModelStatus, "localDir" | "downloaded" | "compiledArchive">;

// The catalogue is DATA (public/catalogue.json), never code: the anti-hardcode
// gate forbids model identities in the app source. localStorage carries
// user-added entries on top of the shipped file.
let shippedCatalogue: CatalogueEntry[] | null = null;
async function loadShippedCatalogue(): Promise<CatalogueEntry[]> {
  if (shippedCatalogue) return shippedCatalogue;
  const base = import.meta.env.BASE_URL ?? "/";
  const res = await fetch(`${base}catalogue.json`);
  if (!res.ok) throw new Error(`catalogue.json failed to load: HTTP ${res.status}`);
  shippedCatalogue = (await res.json()) as CatalogueEntry[];
  return shippedCatalogue;
}

/** The HuggingFace endpoint — overridable via localStorage for hermetic
 * testing against a fixture server (the BDD harness sets it before load). */
export function hfBase(): string {
  return localStorage.getItem("hologram_hf_base") ?? "https://huggingface.co";
}

/** Whether a holospaces egress/auth extension is present (either beacon). */
export function extensionPresent(): boolean {
  const attr = document.documentElement.getAttribute("data-holospaces-egress");
  const flag = (window as unknown as Record<string, unknown>).__HOLOSPACES_EXTENSION_INSTALLED__;
  return attr !== null || flag === true;
}

let logs: LogEntry[] = [];
function addLog(level: string, target: string, message: string) {
  logs.push({ timestampMs: Date.now(), level, target, message });
}

export async function workspacePaths(): Promise<WorkspacePaths> {
  return {
    root: "/",
    modelsDir: "/models",
    outputDir: "/output",
    hologramAiBin: null,
  };
}

export async function getOpfsDir() {
  return await navigator.storage.getDirectory();
}

async function getOpfsDirIfExists(dir: FileSystemDirectoryHandle, name: string): Promise<FileSystemDirectoryHandle | null> {
  try {
    return await dir.getDirectoryHandle(name);
  } catch {
    return null;
  }
}

async function getCatalogue(): Promise<CatalogueEntry[]> {
  const shipped = await loadShippedCatalogue();
  const stored = localStorage.getItem("hologram_catalogue_custom");
  let custom: CatalogueEntry[] = [];
  if (stored) {
    try {
      custom = JSON.parse(stored);
    } catch {
      localStorage.removeItem("hologram_catalogue_custom");
    }
  }
  const merged = [...shipped];
  for (const entry of custom) {
    if (!merged.some(m => m.hfId === entry.hfId)) merged.push(entry);
  }
  return merged;
}

export async function addCustomModel(hfId: string): Promise<void> {
  const catalogue = await getCatalogue();
  if (catalogue.some(m => m.hfId === hfId)) return;
  const id = hfId.split("/").pop()?.toLowerCase() || hfId.toLowerCase();
  const stored = localStorage.getItem("hologram_catalogue_custom");
  let custom: CatalogueEntry[] = [];
  if (stored) {
    try {
      custom = JSON.parse(stored);
    } catch {}
  }
  custom.push({
    id,
    hfId,
    displayName: hfId,
    description: "Custom HuggingFace model.",
    modality: "text-chat",
    size: "?",
    approxArchiveMb: 0,
    quantize: "none",
    promptTemplate: null,
    stop: [],
    chatTurnSeparator: null,
  });
  localStorage.setItem("hologram_catalogue_custom", JSON.stringify(custom));
}

export async function listKnownModels(): Promise<KnownModelStatus[]> {
  const root = await getOpfsDir();
  const modelsDir = await root.getDirectoryHandle("models", { create: true });
  const catalogue = await getCatalogue();

  const results: KnownModelStatus[] = [];
  for (const model of catalogue) {
    const localName = model.hfId.split("/").pop() || model.hfId;
    const localDir = await getOpfsDirIfExists(modelsDir, localName);
    let downloaded = false;
    let compiledArchive: string | null = null;
    
    if (localDir) {
      // Find the .onnx file recursively in localDir
      async function hasOnnx(dir: FileSystemDirectoryHandle): Promise<boolean> {
        // @ts-ignore
        for await (const [name, handle] of dir.entries()) {
          if (handle.kind === 'file' && (name.endsWith('.onnx') || name.endsWith('.safetensors'))) return true;
          if (handle.kind === 'directory' && await hasOnnx(handle as FileSystemDirectoryHandle)) return true;
        }
        return false;
      }
      downloaded = await hasOnnx(localDir);
      
      // The compiled artifact is either a monolithic .holo or a staged bundle
      // (stages.json + stages/N.holo — windowed execution over k).
      // @ts-ignore
      for await (const [childName, childHandle] of localDir.entries()) {
        if (childHandle.kind === 'file' && (childName.endsWith('.holo') || childName === 'stages.json')) {
          compiledArchive = `models/${localName}/${childName}`;
          break;
        }
      }
    }
    
    results.push({
      ...model,
      localDir: localDir ? `models/${localName}` : null,
      downloaded,
      compiledArchive,
    });
  }
  return results;
}

export async function listCompiledArchives(): Promise<CompiledArchive[]> {
  const root = await getOpfsDir();
  const modelsDir = await root.getDirectoryHandle("models", { create: true });
  
  const archives: CompiledArchive[] = [];
  
  // @ts-ignore
  for await (const [name, handle] of modelsDir.entries()) {
    if (handle.kind === "directory") {
      const dirHandle = handle as FileSystemDirectoryHandle;
      // @ts-ignore
      for await (const [childName, childHandle] of dirHandle.entries()) {
        if (childHandle.kind === "file" && (childName.endsWith(".holo") || childName === "stages.json")) {
          const file = await childHandle.getFile();
          archives.push({
            name: childName === "stages.json" ? `${name} (staged)` : `${name}/${childName.replace(".holo", "")}`,
            path: `models/${name}/${childName}`,
            sizeBytes: file.size,
          });
        }
      }
    }
  }
  
  return archives;
}

type Listener = (line: ProcessLine) => void;
const listeners: Record<string, Listener[]> = {};

function emitLine(event: string, line: ProcessLine) {
  if (listeners[event]) {
    listeners[event].forEach(l => l(line));
  }
  addLog(line.stream === "stderr" ? "error" : "info", event, line.line);
}



let downloadWorker: Worker | null = null;
function getDownloadWorker(): Worker {
  if (!downloadWorker) {
    downloadWorker = new Worker(new URL('./download.worker.ts', import.meta.url), { type: 'module' });
  }
  return downloadWorker;
}

export async function downloadKnownModel(id: string): Promise<number> {
  const catalogue = await getCatalogue();
  const model = catalogue.find(m => m.id === id);
  if (!model) throw new Error("Unknown model");
  
  emitLine("models://download-line", { stream: "stdout", line: `Downloading and Compiling ${model.hfId}...` });
  
  const localName = model.hfId.split("/").pop() || model.hfId;
  const root = await getOpfsDir();
  const modelsDir = await root.getDirectoryHandle("models", { create: true });
  const localDir = await modelsDir.getDirectoryHandle(localName, { create: true });
  
  let info: any;
  let infoAttempts = 0;
  while (infoAttempts < 3) {
    infoAttempts++;
    try {
      const response = await fetch(`${hfBase()}/api/models/${model.hfId}?blobs=true`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      info = await response.json();
      break;
    } catch (err) {
      if (infoAttempts >= 3) {
        throw new Error(`Failed to fetch model info for ${model.hfId} after 3 attempts: ${err}`);
      }
      emitLine("models://download-line", { stream: "stdout", line: `Failed to fetch model info (attempt ${infoAttempts}/3). Retrying in ${1 << infoAttempts}s...` });
      await new Promise(r => setTimeout(r, (1 << infoAttempts) * 1000));
    }
  }
  const siblings = info.siblings || [];
  
  const onnxFiles = siblings.filter((f: any) => f.rfilename.endsWith('.onnx') || f.rfilename.endsWith('.onnx_data') || f.rfilename.endsWith('.onnx.data'));
  const safetensorsFiles = siblings.filter((f: any) => f.rfilename.endsWith('.safetensors'));
  
  if (onnxFiles.length === 0 && safetensorsFiles.length === 0) {
    throw new Error(`No ONNX or Safetensors export found in repository.`);
  }

  if (safetensorsFiles.length > 0) {
    // Safetensors flow
    const companionNames = ["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "special_tokens_map.json"];
    const companions = siblings.filter((f: any) => companionNames.includes(f.rfilename.split('/').pop()!));
  
    let configText = "";
    
    // Download companions to OPFS
    for (const file of companions) {
      const url = `${hfBase()}/${model.hfId}/resolve/main/${file.rfilename}`;
      emitLine("models://download-line", { stream: "stdout", line: `Fetching ${file.rfilename}...` });
      
      let response: Response | null = null;
      for (let attempts = 1; attempts <= 3; attempts++) {
        try {
          response = await fetch(url);
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          break;
        } catch (err) {
          if (attempts >= 3) throw new Error(`Failed to fetch ${file.rfilename}`);
          await new Promise(r => setTimeout(r, (1 << attempts) * 1000));
        }
      }
      
      const text = await response!.text();
      // Exact basename only: tokenizer_config.json / generation_config.json
      // also end with "config.json" and must never shadow the model config.
      if (file.rfilename.split('/').pop() === "config.json") {
        configText = text;
      }
      
      const parts = file.rfilename.split('/');
      let currentDir = localDir;
      for (let i = 0; i < parts.length - 1; i++) {
        currentDir = await currentDir.getDirectoryHandle(parts[i], { create: true });
      }
      const fileName = parts[parts.length - 1];
      const handle = await currentDir.getFileHandle(fileName, { create: true });
      const writable = await handle.createWritable();
      await writable.write(text);
      await writable.close();
    }
    
    if (!configText) {
      throw new Error("Missing config.json");
    }

    // Preflight step (a) — the family registry check comes first: an
    // unsupported architecture or malformed config rejects the journey here,
    // before the resource estimate and before any shard byte moves.
    emitLine("models://download-line", { stream: "stdout", line: `Preflight: validating ${model.hfId} config against the family registry...` });
    await validateModelConfig(configText);

    // The resource guard (S1 preflight step c): the ONLY rejection is genuine
    // storage shortfall — the κ-store bytes the model needs versus the
    // MEASURED OPFS quota. Model size never rejects: execution is windowed
    // over k, and the window/storage figures are surfaced as information.
    const config = JSON.parse(configText);
    const shardBytes = safetensorsFiles.reduce(
      (sum: number, f: { size?: number; lfs?: { size?: number } }) =>
        sum + (f.size ?? f.lfs?.size ?? 0),
      0,
    );
    const windowBudget = environmentBudgetBytes((navigator as { deviceMemory?: number }).deviceMemory);
    // The stage-granularity knob: a smaller stage-plan budget forces finer
    // staging (more, smaller stages) without shrinking the context window.
    const stageKnob = Number(localStorage.getItem("hologram_stage_window") ?? "");
    const stagePlanBudget = Number.isFinite(stageKnob) && stageKnob > 0 ? stageKnob : windowBudget;
    const estimate = estimateResources(config, shardBytes, windowBudget, stagePlanBudget);

    // The resource PROJECTION (information, never rejection): the κ-store is
    // a cache over recorded provenance — tensors beyond the local headroom
    // resolve at run time from their revision-pinned source. The cache-budget
    // knob (hologram_cache_budget) tunes local caching; a safety margin keeps
    // headroom for archives/companions.
    const headroom = await measuredStorageHeadroomBytes();
    const cacheKnob = localStorage.getItem("hologram_cache_budget");
    const knobValue = cacheKnob === null ? NaN : Number(cacheKnob);
    const cacheBudget = Number.isFinite(knobValue) && knobValue >= 0
      ? knobValue
      : cacheBudgetFromHeadroom(headroom);
    const coverage = shardBytes > 0 ? Math.min(1, cacheBudget / shardBytes) : 1;
    emitLine("models://download-line", {
      stream: "stdout",
      line:
        `Resource projection: κ-store ${formatBytes(shardBytes)}, measured local headroom ${formatBytes(headroom)}, ` +
        `cache coverage ~${Math.round(coverage * 100)}% (the remainder resolves from recorded κ-provenance); ` +
        `execution window ~${formatBytes(estimate.windowBytes)} ` +
        `(${estimate.stageCount === 1 ? "monolithic" : `${estimate.stageCount} stages, ${estimate.layersPerStage} layer(s) each`}), ` +
        `context ${estimate.contextLength}.`,
    });

    const worker = getDownloadWorker();
    const outcome = await new Promise<{ holoBytes?: Uint8Array; stageCount?: number }>(
      (resolve, reject) => {
        worker.onmessage = (e) => {
          if (e.data.type === "progress") {
            emitLine("models://download-progress", { stream: "stdout", line: e.data.line });
          } else if (e.data.type === "done") {
            resolve({ holoBytes: e.data.holoBytes });
          } else if (e.data.type === "done_staged") {
            resolve({ stageCount: e.data.stageCount });
          } else if (e.data.type === "error") {
            reject(new Error(e.data.error));
          }
        };
        worker.postMessage({
          type: "download_safetensors",
          payload: {
            modelId: model.hfId,
            configText,
            files: safetensorsFiles,
            contextLength: estimate.contextLength,
            layersPerStage: estimate.layersPerStage,
            stageCount: estimate.stageCount,
            localName,
            revision: info.sha,
            cacheBudgetBytes: cacheBudget,
            hfBase: hfBase()
          }
        });
      },
    );

    if (outcome.stageCount) {
      emitLine("models://compile-line", {
        stream: "stdout",
        line: `Compiled ${outcome.stageCount} stage archives (windowed execution over k).`,
      });
    } else {
      const holoBytes = outcome.holoBytes!;
      const kappa = await computeKappa(holoBytes);
      const holoName = `${kappa}.holo`;
      // The archive is crystalline structure; cached tensors are gas phase
      // (provenance-recoverable). Under quota pressure the cache evaporates
      // to make room — the journey is never refused for resources.
      for (;;) {
        try {
          const holoHandle = await localDir.getFileHandle(holoName, { create: true });
          const writable = await holoHandle.createWritable();
          await writable.write(holoBytes as any);
          await writable.close();
          break;
        } catch (e) {
          const tensorsDir = await root.getDirectoryHandle("tensors", { create: true });
          let evicted = false;
          const iter = (
            tensorsDir as unknown as { entries(): AsyncIterableIterator<[string, FileSystemHandle]> }
          ).entries();
          for await (const [name] of iter) {
            await tensorsDir.removeEntry(name).catch(() => {});
            evicted = true;
            break;
          }
          if (!evicted) throw e;
          emitLine("models://compile-line", {
            stream: "stdout",
            line: "κ-store pressure: evaporated a cached tensor to persist the archive.",
          });
        }
      }
      emitLine("models://compile-line", { stream: "stdout", line: `Compiled and saved to ${holoName} (${holoBytes.length} bytes).` });
    }
  } else {
    // ONNX flow - unsupported because it cannot be streamed over k
    emitLine("models://download-progress", { stream: "stderr", line: "Error: ONNX models are not supported in the web IDE because they cannot be streamed over the holospaces/k-representation." });
    emitLine("models://download-progress", { stream: "stderr", line: "Please use a model with safetensors." });
    throw new Error("Model lacks safetensors export.");
  }
  return 0;
}

export async function compileKnownModel(_id: string, _specificOnnx?: string): Promise<number> {
  // We unified download and compile into downloadKnownModel.
  // The UI might still call this, so just return success.
  return 0;
}

// The session window of a compiled model (from its model-meta), plus the
// tokenizer bytes for template-aware counting. Cached per model dir (row
// `session-window`).
const sessionInfoCache = new Map<string, { contextLength: number | null; tokenizer?: Uint8Array }>();

async function readModelFile(dirName: string, fileName: string): Promise<Uint8Array | null> {
  try {
    const root = await getOpfsDir();
    const models = await root.getDirectoryHandle("models");
    const dir = await models.getDirectoryHandle(dirName);
    async function find(
      handle: FileSystemDirectoryHandle,
      target: string,
    ): Promise<FileSystemFileHandle | null> {
      for await (const [n, h] of (handle as any).entries()) {
        if (h.kind === "file" && n === target) return h as FileSystemFileHandle;
        if (h.kind === "directory") {
          const found = await find(h as FileSystemDirectoryHandle, target);
          if (found) return found;
        }
      }
      return null;
    }
    const fh = await find(dir, fileName);
    if (!fh) return null;
    return new Uint8Array(await (await fh.getFile()).arrayBuffer());
  } catch {
    return null;
  }
}

/** Load a model's session bounds once (context window + its own tokenizer) for
 * template-aware trimming — the model's own limits are the only limits (row
 * `session-window`). Read off the hot path (a load-time effect), never per send. */
export async function loadSessionMeta(
  archivePath: string,
): Promise<{ contextLength: number | null; tokenizer?: Uint8Array }> {
  const dirName = archivePath.split("/")[1];
  const { contextLength } = await sessionInfo(archivePath);
  const tokenizer = (await readModelFile(dirName, "tokenizer.json")) ?? undefined;
  return { contextLength, tokenizer };
}

/** The compiled model's own context window (from model-meta.json / stages.json). */
export async function sessionInfo(archivePath: string): Promise<{ contextLength: number | null }> {
  const dirName = archivePath.split("/")[1];
  const cached = sessionInfoCache.get(dirName);
  if (cached) return { contextLength: cached.contextLength };
  let contextLength: number | null = null;
  const meta = await readModelFile(dirName, "model-meta.json");
  if (meta) {
    contextLength = JSON.parse(new TextDecoder().decode(meta)).contextLength ?? null;
  } else {
    const stages = await readModelFile(dirName, "stages.json");
    if (stages) contextLength = JSON.parse(new TextDecoder().decode(stages)).contextLength ?? null;
  }
  sessionInfoCache.set(dirName, { contextLength });
  return { contextLength };
}

/** Token count of `text` under the model's own tokenizer — the model's own
 * limits are the only limits (session trimming, row `session-window`). */
export async function countPromptTokens(archivePath: string, text: string): Promise<number> {
  const dirName = archivePath.split("/")[1];
  let entry = sessionInfoCache.get(dirName);
  if (!entry) {
    await sessionInfo(archivePath);
    entry = sessionInfoCache.get(dirName)!;
  }
  if (!entry.tokenizer) {
    const bytes = await readModelFile(dirName, "tokenizer.json");
    if (!bytes) throw new Error(`no tokenizer.json under models/${dirName} — cannot count tokens`);
    entry.tokenizer = bytes;
  }
  return countTokens(entry.tokenizer, text);
}

export interface GenerateOpts {
  archive: string;
  prompt: string;
  maxTokens?: number;
  temperature?: number;
  topK?: number;
  stop?: string[];
  promptTemplate?: string;
}

// Removed unused variable

let activeWorker: Worker | null = null;

export async function generate(opts: GenerateOpts): Promise<number> {
  // ... read holoBytes ...
  
  const archiveParts = opts.archive.split("/");
  const root = await getOpfsDir();
  const modelsDir = await root.getDirectoryHandle("models");
  const localDir = await modelsDir.getDirectoryHandle(archiveParts[1]);
  const staged = archiveParts[2] === "stages.json";
  let holoBytes: Uint8Array | undefined;
  let stageCount = 0;
  if (staged) {
    const metaHandle = await localDir.getFileHandle("stages.json");
    const meta = JSON.parse(await (await metaHandle.getFile()).text());
    stageCount = meta.stageCount;
  } else {
    const holoHandle = await localDir.getFileHandle(archiveParts[2]);
    const holoFile = await holoHandle.getFile();
    holoBytes = new Uint8Array(await holoFile.arrayBuffer());
  }
  
  async function findFileRecursive(dir: FileSystemDirectoryHandle, targetName: string): Promise<FileSystemFileHandle | null> {
    for await (const [name, handle] of (dir as any).entries()) {
      if (handle.kind === 'file' && name === targetName) {
        return handle as FileSystemFileHandle;
      }
      if (handle.kind === 'directory') {
        const found = await findFileRecursive(handle as FileSystemDirectoryHandle, targetName);
        if (found) return found;
      }
    }
    return null;
  }

  let tokenizerBytes: Uint8Array | undefined;
  try {
    const tokHandle = await findFileRecursive(localDir, "tokenizer.json");
    if (tokHandle) {
      const tokFile = await tokHandle.getFile();
      tokenizerBytes = new Uint8Array(await tokFile.arrayBuffer());
    }
  } catch {
    // optional
  }
  
  const genOpts: GenOpts = {
    prompt_template: opts.promptTemplate,
    max_tokens: opts.maxTokens,
    temperature: opts.temperature,
    top_k: opts.topK,
    stop: opts.stop,
  };
  
  emitLine("chat://line", { stream: "stdout", line: "" });
  
  return new Promise((resolve, reject) => {
    // Reuse the live worker: it holds the warm staged session. The worker
    // itself rebuilds the session when the model changes (its inputs are
    // the model's), so reuse across sends is always safe.
    activeWorker ??= new GenerateWorker();
    // A worker script/load failure would otherwise leave generation hanging
    // silently (onmessage never fires); surface it loudly.
    activeWorker.onerror = (ev) => {
      const msg = `generate worker failed: ${(ev as ErrorEvent).message ?? ev}`;
      emitLine("chat://line", { stream: "stderr", line: msg });
      if (activeWorker) { activeWorker.terminate(); activeWorker = null; }
      reject(new Error(msg));
    };
    
    activeWorker.onmessage = (e) => {
      if (e.data.type === 'token') {
        emitLine("chat://line", { stream: "stdout", line: e.data.text });
      } else if (e.data.type === 'progress') {
        // Narration of the honest work behind the first token (window
        // compiles, per-stage materialization) — a status channel, never
        // part of the completion text. Mirrored into a session log so the
        // hermetic suite can assert warm-turn behavior (row `warm-turn`).
        const g = globalThis as unknown as { __hologram_status?: string[] };
        (g.__hologram_status ??= []).push(e.data.line);
        emitLine("chat://status", { stream: "stdout", line: e.data.line });
      } else if (e.data.type === 'done') {
        const g = globalThis as unknown as {
          __hologram_completions?: { prompt: string; text: string }[];
        };
        (g.__hologram_completions ??= []).push({ prompt: opts.prompt, text: e.data.text });
        emitLine("chat://line", { stream: "stdout", line: e.data.text });
        // The worker stays alive: it holds the warm staged session (row
        // `warm-turn`) — the next turn reuses the resident window instead
        // of rebuilding it. Cancel/error still terminate.
        resolve(0);
      } else if (e.data.type === 'error') {
        emitLine("chat://line", { stream: "stderr", line: e.data.error });
        if (activeWorker) {
          activeWorker.terminate();
          activeWorker = null;
        }
        reject(new Error(e.data.error));
      }
    };
    
    activeWorker.postMessage({
      holoBytes,
      modelDir: archiveParts[1],
      staged: staged ? { stageCount } : undefined,
      prompt: opts.prompt,
      genOpts,
      tokenizerBytes,
    });
  });
}

export async function cancelGeneration(): Promise<boolean> {
  if (activeWorker) {
    activeWorker.terminate();
    activeWorker = null;
    emitLine("chat://line", { stream: "stdout", line: "\n[Generation cancelled]" });
    return true;
  }
  return false;
}

export async function recentLogs(since: number): Promise<LogsResponse> {
  const newLogs = logs.filter(l => l.timestampMs > since);
  return {
    entries: newLogs,
    nextIndex: Date.now(),
  };
}

export async function clearLogs(): Promise<void> {
  logs = [];
}

export function onProcessLine(
  event: string,
  cb: (line: ProcessLine) => void,
): Promise<() => void> {
  if (!listeners[event]) listeners[event] = [];
  listeners[event].push(cb);
  return Promise.resolve(() => {
    listeners[event] = listeners[event].filter(l => l !== cb);
  });
}
