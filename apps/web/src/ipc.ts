import { compile, generate as wasmGenerate } from "./holo";
import { type GenOpts } from "./holo";

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

const DEFAULT_CATALOGUE: Omit<KnownModelStatus, "localDir" | "downloaded" | "compiledArchive">[] = [
  {
    id: "smollm2-135m-instruct",
    hfId: "HuggingFaceTB/SmolLM2-135M-Instruct",
    displayName: "SmolLM2 135M Instruct",
    description: "Lightweight chat model — fastest path to a working demo.",
    modality: "text-chat",
    size: "135M",
    approxArchiveMb: 150,
    quantize: "none",
    promptTemplate: "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
    stop: ["<|im_end|>"],
    chatTurnSeparator: "<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>\n<|im_start|>user\n",
  },
  {
    id: "qwen2.5-0.5b-instruct",
    hfId: "onnx-community/Qwen2.5-0.5B-Instruct",
    displayName: "Qwen2.5 0.5B Instruct",
    description: "Small chat-tuned model — follows instructions and answers questions.",
    modality: "text-chat",
    size: "0.5B",
    approxArchiveMb: 350,
    quantize: "none",
    promptTemplate:
      "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
    stop: ["<|im_end|>"],
    chatTurnSeparator:
      "<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>\n<|im_start|>user\n",
  },
];

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

async function getOpfsDir() {
  return await navigator.storage.getDirectory();
}

async function getOpfsFileIfExists(dir: FileSystemDirectoryHandle, name: string): Promise<FileSystemFileHandle | null> {
  try {
    return await dir.getFileHandle(name);
  } catch {
    return null;
  }
}

async function getOpfsDirIfExists(dir: FileSystemDirectoryHandle, name: string): Promise<FileSystemDirectoryHandle | null> {
  try {
    return await dir.getDirectoryHandle(name);
  } catch {
    return null;
  }
}

function getCatalogue(): Omit<KnownModelStatus, "localDir" | "downloaded" | "compiledArchive">[] {
  const stored = localStorage.getItem("hologram_catalogue");
  if (stored) {
    try {
      return JSON.parse(stored);
    } catch {}
  }
  localStorage.setItem("hologram_catalogue", JSON.stringify(DEFAULT_CATALOGUE));
  return DEFAULT_CATALOGUE;
}

export async function addCustomModel(hfId: string): Promise<void> {
  const catalogue = getCatalogue();
  if (catalogue.some(m => m.hfId === hfId)) return;
  const id = hfId.split("/").pop()?.toLowerCase() || hfId.toLowerCase();
  catalogue.push({
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
  localStorage.setItem("hologram_catalogue", JSON.stringify(catalogue));
}

export async function listKnownModels(): Promise<KnownModelStatus[]> {
  const root = await getOpfsDir();
  const modelsDir = await root.getDirectoryHandle("models", { create: true });
  const catalogue = getCatalogue();
  
  const results: KnownModelStatus[] = [];
  for (const model of catalogue) {
    const localName = model.hfId.split("/").pop() || model.hfId;
    const localDir = await getOpfsDirIfExists(modelsDir, localName);
    let downloaded = false;
    let compiledArchive: string | null = null;
    
    if (localDir) {
      downloaded = (await getOpfsFileIfExists(localDir, "model.onnx")) !== null;
      if (await getOpfsFileIfExists(localDir, `${model.id}.holo`)) {
        compiledArchive = `models/${localName}/${model.id}.holo`;
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

export async function downloadKnownModel(id: string): Promise<number> {
  const catalogue = getCatalogue();
  const model = catalogue.find(m => m.id === id);
  if (!model) throw new Error("Unknown model");
  
  emitLine("models://download-line", { stream: "stdout", line: `Downloading ${model.hfId} from HuggingFace...` });
  
  const localName = model.hfId.split("/").pop() || model.hfId;
  const root = await getOpfsDir();
  const modelsDir = await root.getDirectoryHandle("models", { create: true });
  const localDir = await modelsDir.getDirectoryHandle(localName, { create: true });
  
  const infoRes = await fetch(`https://huggingface.co/api/models/${model.hfId}`);
  if (!infoRes.ok) throw new Error(`Failed to fetch model info: ${infoRes.statusText}`);
  const info = await infoRes.json();
  const siblings = info.siblings || [];
  
  const onnxFiles = siblings.filter((f: any) => f.rfilename.endsWith('.onnx') || f.rfilename.endsWith('.onnx_data') || f.rfilename.endsWith('.onnx.data'));
  if (onnxFiles.length === 0) {
    throw new Error(`No ONNX export found in repository. The web version requires pre-exported models.`);
  }

  const companionNames = ["tokenizer.json", "config.json", "tokenizer_config.json", "special_tokens_map.json"];
  const companions = siblings.filter((f: any) => companionNames.includes(f.rfilename.split('/').pop()!));

  const filesToDownload = [...onnxFiles, ...companions];
  
  for (const file of filesToDownload) {
    const url = `https://huggingface.co/${model.hfId}/resolve/main/${file.rfilename}`;
    emitLine("models://download-line", { stream: "stdout", line: `Fetching ${url}...` });
    
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(`Failed to fetch ${file.rfilename}: ${res.statusText}`);
    }
    
    // Create subdirectories if necessary (OPFS requires creating directories one by one)
    const parts = file.rfilename.split('/');
    const fileName = parts.pop()!;
    let currentDir = localDir;
    for (const part of parts) {
      currentDir = await currentDir.getDirectoryHandle(part, { create: true });
    }

    const fileHandle = await currentDir.getFileHandle(fileName, { create: true });
    const writable = await fileHandle.createWritable();
    await res.body!.pipeTo(writable);
    
    emitLine("models://download-line", { stream: "stdout", line: `Saved ${file.rfilename}.` });
  }
  
  emitLine("models://download-line", { stream: "stdout", line: `Download complete.` });
  return 0;
}

export async function compileKnownModel(id: string, specificOnnx?: string): Promise<number> {
  const catalogue = getCatalogue();
  const model = catalogue.find(m => m.id === id);
  if (!model) throw new Error("Unknown model");
  
  emitLine("models://compile-line", { stream: "stdout", line: `Compiling ${model.id}...` });
  
  const localName = model.hfId.split("/").pop() || model.hfId;
  const root = await getOpfsDir();
  const modelsDir = await root.getDirectoryHandle("models", { create: true });
  const localDir = await modelsDir.getDirectoryHandle(localName);
  
  // Find the .onnx file recursively in localDir
  async function findOnnx(dir: FileSystemDirectoryHandle): Promise<FileSystemFileHandle | null> {
    for await (const [name, handle] of (dir as any).entries()) {
      if (handle.kind === 'file' && name.endsWith('.onnx')) {
        return handle as FileSystemFileHandle;
      }
      if (handle.kind === 'directory') {
        const found = await findOnnx(handle as FileSystemDirectoryHandle);
        if (found) return found;
      }
    }
    return null;
  }

  // Get a specific file handle by path relative to localDir
  async function getSpecificFile(dir: FileSystemDirectoryHandle, path: string): Promise<FileSystemFileHandle> {
    const parts = path.split('/');
    const fileName = parts.pop()!;
    let currentDir = dir;
    for (const part of parts) {
      currentDir = await currentDir.getDirectoryHandle(part);
    }
    return await currentDir.getFileHandle(fileName);
  }
  
  const onnxHandle = specificOnnx ? await getSpecificFile(localDir, specificOnnx) : await findOnnx(localDir);
  if (!onnxHandle) throw new Error(`Could not find .onnx file in downloaded model (${specificOnnx || "any"})`);
  
  const onnxFile = await onnxHandle.getFile();
  const onnxBytes = new Uint8Array(await onnxFile.arrayBuffer());
  
  emitLine("models://compile-line", { stream: "stdout", line: `Loaded ONNX (${onnxBytes.length} bytes). Compiling via wasm...` });
  
  const holoBytes = await compile(onnxBytes);
  
  const holoHandle = await localDir.getFileHandle(`${model.id}.holo`, { create: true });
  const writable = await holoHandle.createWritable();
  await writable.write(holoBytes as any);
  await writable.close();
  
  emitLine("models://compile-line", { stream: "stdout", line: `Compiled and saved to ${model.id}.holo (${holoBytes.length} bytes).` });
  
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

export async function generate(opts: GenerateOpts): Promise<number> {
  
  const archiveParts = opts.archive.split("/");
  const root = await getOpfsDir();
  const modelsDir = await root.getDirectoryHandle("models");
  const localDir = await modelsDir.getDirectoryHandle(archiveParts[1]);
  const holoHandle = await localDir.getFileHandle(archiveParts[2]);
  const holoFile = await holoHandle.getFile();
  const holoBytes = new Uint8Array(await holoFile.arrayBuffer());
  
  let tokenizerBytes: Uint8Array | undefined;
  try {
    const tokHandle = await localDir.getFileHandle("tokenizer.json");
    const tokFile = await tokHandle.getFile();
    tokenizerBytes = new Uint8Array(await tokFile.arrayBuffer());
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
  
  const result = await wasmGenerate(holoBytes, opts.prompt, genOpts, tokenizerBytes);
  
  emitLine("chat://line", { stream: "stdout", line: result });
  
  return 0;
}

export async function cancelGeneration(): Promise<boolean> {
  // TODO: abort running wasm generation if holo supports it
  return true;
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
