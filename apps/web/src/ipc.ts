import { computeKappa } from "./holo";
import { type GenOpts } from "./holo";
import GenerateWorker from "./generate.worker?worker";
import { environmentBudgetBytes, estimateResources, formatBytes } from "./resources";

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
      
      // Find the compiled archive by looking for any .holo file (kappa cache collapse)
      // @ts-ignore
      for await (const [childName, childHandle] of localDir.entries()) {
        if (childHandle.kind === 'file' && childName.endsWith('.holo')) {
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
        if (childHandle.kind === "file" && childName.endsWith(".holo")) {
          const file = await childHandle.getFile();
          archives.push({
            name: `${name}/${childName.replace(".holo", "")}`,
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
    const companionNames = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"];
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
      if (file.rfilename.endsWith("config.json")) {
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

    // The memory guard (S1): a config-derived estimate gates the journey
    // BEFORE any shard bytes move. Parametric — a function of the model's own
    // config + manifest, never a per-model constant.
    const config = JSON.parse(configText);
    const shardBytes = safetensorsFiles.reduce(
      (sum: number, f: { size?: number; lfs?: { size?: number } }) =>
        sum + (f.size ?? f.lfs?.size ?? 0),
      0,
    );
    const budget = environmentBudgetBytes((navigator as { deviceMemory?: number }).deviceMemory);
    const estimate = estimateResources(config, shardBytes, budget);
    if (estimate.runtimeBytes > budget) {
      const msg =
        `Memory guard: ${model.hfId} needs ~${formatBytes(estimate.runtimeBytes)} at run time ` +
        `(weights ${formatBytes(shardBytes)} + activations at context ${estimate.contextLength}), ` +
        `but the environment budget is ${formatBytes(budget)}. Rejecting before transfer.`;
      emitLine("models://download-progress", { stream: "stderr", line: msg });
      throw new Error(msg);
    }
    emitLine("models://download-line", {
      stream: "stdout",
      line: `Memory guard: ~${formatBytes(estimate.runtimeBytes)} within budget ${formatBytes(budget)}; context length ${estimate.contextLength}.`,
    });

    const worker = getDownloadWorker();
    const holoBytes = await new Promise<Uint8Array>((resolve, reject) => {
      worker.onmessage = (e) => {
        if (e.data.type === "progress") {
          emitLine("models://download-progress", { stream: "stdout", line: e.data.line });
        } else if (e.data.type === "done") {
          resolve(e.data.holoBytes);
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
          hfBase: hfBase()
        }
      });
    });
    
    const kappa = await computeKappa(holoBytes);
    const holoName = `${kappa}.holo`;
    const holoHandle = await localDir.getFileHandle(holoName, { create: true });
    const writable = await holoHandle.createWritable();
    await writable.write(holoBytes as any);
    await writable.close();
    
    emitLine("models://compile-line", { stream: "stdout", line: `Compiled and saved to ${holoName} (${holoBytes.length} bytes).` });
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
  const holoHandle = await localDir.getFileHandle(archiveParts[2]);
  const holoFile = await holoHandle.getFile();
  const holoBytes = new Uint8Array(await holoFile.arrayBuffer());
  
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
    if (activeWorker) {
      activeWorker.terminate();
    }
    
    activeWorker = new GenerateWorker();
    
    activeWorker.onmessage = (e) => {
      if (e.data.type === 'token') {
        emitLine("chat://line", { stream: "stdout", line: e.data.text });
      } else if (e.data.type === 'done') {
        const g = globalThis as unknown as { __hologram_completions?: string[] };
        (g.__hologram_completions ??= []).push(e.data.text);
        emitLine("chat://line", { stream: "stdout", line: e.data.text });
        if (activeWorker) {
          activeWorker.terminate();
          activeWorker = null;
        }
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
