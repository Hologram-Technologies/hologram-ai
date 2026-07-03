import { compileOnnxWithData, computeKappa, compileSafetensorsStreamed, KappaHasher, ensureReady } from "./holo";
import { type GenOpts } from "./holo";
import GenerateWorker from "./generate.worker?worker";

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



export async function streamSafetensorsForCompile(url: string, fileName: string) {
  await ensureReady();
  let response: Response | null = null;
  let attempts = 0;
  while (attempts < 3) {
    attempts++;
    try {
      response = await fetch(url);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      break;
    } catch (err) {
      if (attempts >= 3) throw err;
      await new Promise(r => setTimeout(r, (1 << attempts) * 1000));
    }
  }
  if (!response || !response.body) throw new Error(`Failed to fetch ${fileName}`);

  const reader = response.body.getReader();
  let downloadedBytes = 0;
  const totalBytes = Number(response.headers.get("content-length")) || 0;
  let lastEmit = 0;
  
  let headerLengthBuf = new Uint8Array(8);
  let headerLengthRead = 0;
  let chunks: Uint8Array[] = [];
  
  while (headerLengthRead < 8) {
    const { done, value } = await reader.read();
    if (done) throw new Error("EOF before reading safetensors length");
    downloadedBytes += value.length;
    chunks.push(value);
    const needed = 8 - headerLengthRead;
    const toCopy = Math.min(needed, value.length);
    headerLengthBuf.set(value.slice(0, toCopy), headerLengthRead);
    headerLengthRead += toCopy;
  }
  
  const view = new DataView(headerLengthBuf.buffer);
  const headerLen = view.getUint32(0, true); 
  
  let headerBuf = new Uint8Array(headerLen);
  let headerRead = 0;
  let offsetInChunks = 8;
  
  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];
    if (offsetInChunks < chunk.length) {
      const remainingInChunk = chunk.length - offsetInChunks;
      const needed = headerLen - headerRead;
      if (needed <= 0) break;
      const toCopy = Math.min(needed, remainingInChunk);
      headerBuf.set(chunk.slice(offsetInChunks, offsetInChunks + toCopy), headerRead);
      headerRead += toCopy;
      offsetInChunks += toCopy;
    } else {
      offsetInChunks -= chunk.length;
    }
  }
  
  while (headerRead < headerLen) {
    const { done, value } = await reader.read();
    if (done) throw new Error("EOF before reading safetensors header");
    downloadedBytes += value.length;
    chunks.push(value);
    const needed = headerLen - headerRead;
    const toCopy = Math.min(needed, value.length);
    headerBuf.set(value.slice(0, toCopy), headerRead);
    headerRead += toCopy;
  }
  
  const headerStr = new TextDecoder().decode(headerBuf);
  const header = JSON.parse(headerStr);
  
  const tensors: any[] = [];
  for (const [key, meta] of Object.entries(header)) {
    if (key === "__metadata__") continue;
    tensors.push({ key, meta: meta as any });
  }
  tensors.sort((a, b) => a.meta.data_offsets[0] - b.meta.data_offsets[0]);
  
  const results = {
    keys: [] as string[],
    kappas: [] as string[],
    shapes: [] as string[],
    dtypes: [] as string[]
  };
  
  let currentTensorIdx = 0;
  let currentHasher = new KappaHasher();
  const baseOffset = 8 + headerLen;
  
  function processBytes(globalOffset: number, bytes: Uint8Array) {
    let localOffset = 0;
    while (localOffset < bytes.length && currentTensorIdx < tensors.length) {
      const tensor = tensors[currentTensorIdx];
      const tensorStart = baseOffset + tensor.meta.data_offsets[0];
      const tensorEnd = baseOffset + tensor.meta.data_offsets[1];
      const currentGlobal = globalOffset + localOffset;
      
      if (currentGlobal < tensorStart) {
        const toSkip = Math.min(tensorStart - currentGlobal, bytes.length - localOffset);
        localOffset += toSkip;
      } else if (currentGlobal < tensorEnd) {
        const toFeed = Math.min(tensorEnd - currentGlobal, bytes.length - localOffset);
        currentHasher.update(bytes.slice(localOffset, localOffset + toFeed));
        localOffset += toFeed;
        if (currentGlobal + toFeed === tensorEnd) {
          const kappa = currentHasher.finalize();
          results.keys.push(tensor.key);
          results.kappas.push(kappa);
          results.shapes.push(JSON.stringify(tensor.meta.shape));
          results.dtypes.push(tensor.meta.dtype);
          currentTensorIdx++;
          if (currentTensorIdx < tensors.length) {
            currentHasher.free();
            currentHasher = new KappaHasher();
          } else {
            currentHasher.free();
          }
        }
      } else {
        localOffset++;
      }
    }
  }
  
  let processedOffset = 0;
  for (const chunk of chunks) {
    processBytes(processedOffset, chunk);
    processedOffset += chunk.length;
  }
  
  while (currentTensorIdx < tensors.length) {
    const { done, value } = await reader.read();
    if (done) break;
    processBytes(downloadedBytes, value);
    downloadedBytes += value.length;
    const now = Date.now();
    if (now - lastEmit > 500) {
      lastEmit = now;
      if (totalBytes > 0) {
        const percent = Math.round((downloadedBytes / totalBytes) * 100);
        emitLine("models://download-progress", { stream: "stdout", line: `Streaming ${fileName}: ${percent}% (${(downloadedBytes / 1024 / 1024).toFixed(1)}MB)` });
      }
    }
  }
  
  if (currentTensorIdx < tensors.length) {
    throw new Error(`EOF before finishing tensor ${tensors[currentTensorIdx].key}`);
  }
  emitLine("models://download-line", { stream: "stdout", line: `Finished streaming ${fileName}.` });
  return results;
}

export async function downloadKnownModel(id: string): Promise<number> {
  const catalogue = getCatalogue();
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
      const response = await fetch(`https://huggingface.co/api/models/${model.hfId}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      info = await response.json();
      break;
    } catch (err) {
      if (infoAttempts >= 3) {
        throw new Error(`Failed to fetch model info after 3 attempts: ${err}`);
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
      const url = `https://huggingface.co/${model.hfId}/resolve/main/${file.rfilename}`;
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
  
    // Stream safetensors to compute hashes
    const allKeys: string[] = [];
    const allKappas: string[] = [];
    const allShapes: string[] = [];
    const allDtypes: string[] = [];
    
    for (const file of safetensorsFiles) {
      const url = `https://huggingface.co/${model.hfId}/resolve/main/${file.rfilename}`;
      emitLine("models://download-line", { stream: "stdout", line: `Streaming ${file.rfilename}...` });
      const res = await streamSafetensorsForCompile(url, file.rfilename);
      allKeys.push(...res.keys);
      allKappas.push(...res.kappas);
      allShapes.push(...res.shapes);
      allDtypes.push(...res.dtypes);
    }
    
    emitLine("models://compile-line", { stream: "stdout", line: `Compiling streamed tensors...` });
    
    const holoBytes = await compileSafetensorsStreamed(
      configText,
      allKeys,
      allKappas,
      allShapes,
      allDtypes
    );
    
    const kappa = await computeKappa(holoBytes);
    const holoName = `${kappa}.holo`;
    const holoHandle = await localDir.getFileHandle(holoName, { create: true });
    const writable = await holoHandle.createWritable();
    await writable.write(holoBytes as any);
    await writable.close();
    
    emitLine("models://compile-line", { stream: "stdout", line: `Compiled and saved to ${holoName} (${holoBytes.length} bytes).` });
  } else {
    // ONNX flow
    const companionNames = ["tokenizer.json", "config.json", "tokenizer_config.json", "special_tokens_map.json"];
    const companions = siblings.filter((f: any) => companionNames.includes(f.rfilename.split('/').pop()!));
  
    let selectedOnnxFiles: any[] = [];
    if (onnxFiles.length > 0) {
      const mainOnnxFiles = onnxFiles.filter((f: any) => f.rfilename.endsWith('.onnx'));
      let chosenMain = mainOnnxFiles[0];
      for (const f of mainOnnxFiles) {
        const name = f.rfilename.toLowerCase();
        if (name.includes('cpu') || name.includes('int4')) {
          chosenMain = f;
          if (name.includes('cpu') && name.includes('int4')) break;
        }
      }
      if (chosenMain) {
        selectedOnnxFiles.push(chosenMain);
        const expectedDataName = chosenMain.rfilename + '.data';
        const dataFile = onnxFiles.find((f: any) => f.rfilename === expectedDataName || f.rfilename === chosenMain.rfilename + '_data');
        if (dataFile) selectedOnnxFiles.push(dataFile);
      } else {
        selectedOnnxFiles = onnxFiles;
      }
    }
  
    const filesToDownload = [...selectedOnnxFiles, ...companions];
    
    for (const file of filesToDownload) {
      const url = `https://huggingface.co/${model.hfId}/resolve/main/${file.rfilename}`;
      emitLine("models://download-line", { stream: "stdout", line: `Fetching ${url}...` });
      
      let response: Response | null = null;
      let attempts = 0;
      while (attempts < 3) {
        attempts++;
        try {
          response = await fetch(url);
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          break; // Success
        } catch (err) {
          if (attempts >= 3) {
            throw new Error(`Failed to fetch ${file.rfilename} after 3 attempts: ${err}`);
          }
          emitLine("models://download-line", { stream: "stdout", line: `Download failed (attempt ${attempts}/3). Retrying in ${1 << attempts}s...` });
          await new Promise(r => setTimeout(r, (1 << attempts) * 1000));
        }
      }
      if (!response) throw new Error(`Failed to fetch ${file.rfilename}`);
      
      // Write to OPFS
      const parts = file.rfilename.split('/');
      let currentDir = localDir;
      for (let i = 0; i < parts.length - 1; i++) {
        currentDir = await currentDir.getDirectoryHandle(parts[i], { create: true });
      }
      const fileName = parts[parts.length - 1];
      
      const handle = await currentDir.getFileHandle(fileName, { create: true });
      const writable = await handle.createWritable();
      
      if (response.body) {
        const contentLength = response.headers.get("content-length");
        const total = contentLength ? parseInt(contentLength, 10) : 0;
        let loaded = 0;
        let lastEmit = 0;
        
        const reader = response.body.getReader();
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          await writable.write(value);
          loaded += value.length;
          
          const now = Date.now();
          if (now - lastEmit > 500) {
            lastEmit = now;
            if (total > 0) {
              const percent = Math.round((loaded / total) * 100);
              emitLine("models://download-progress", { stream: "stdout", line: `Downloading ${file.rfilename}: ${percent}% (${(loaded / 1024 / 1024).toFixed(1)}MB / ${(total / 1024 / 1024).toFixed(1)}MB)` });
            } else {
              emitLine("models://download-progress", { stream: "stdout", line: `Downloading ${file.rfilename}: ${(loaded / 1024 / 1024).toFixed(1)}MB` });
            }
          }
        }
      }
      await writable.close();
      emitLine("models://download-line", { stream: "stdout", line: `Saved ${file.rfilename}.` });
    }
    
    // NOW compile the ONNX files
    async function findOnnx(dir: FileSystemDirectoryHandle): Promise<{ file: FileSystemFileHandle, parent: FileSystemDirectoryHandle } | null> {
      for await (const [name, handle] of (dir as any).entries()) {
        if (handle.kind === 'file' && name.endsWith('.onnx')) {
          return { file: handle as FileSystemFileHandle, parent: dir };
        }
        if (handle.kind === 'directory') {
          const found = await findOnnx(handle as FileSystemDirectoryHandle);
          if (found) return found;
        }
      }
      return null;
    }
    
    const onnxResult = await findOnnx(localDir);
    if (!onnxResult) throw new Error("ONNX file not found after download");
    
    const onnxHandle = onnxResult.file;
    const parentDir = onnxResult.parent;
    const onnxFile = await onnxHandle.getFile();
    if (onnxFile.size > 2 * 1024 * 1024 * 1024) {
      throw new Error(`Model is too large (${(onnxFile.size / 1024 / 1024 / 1024).toFixed(1)}GB) to compile in the browser due to WebAssembly 32-bit memory limits (max 2-4GB). Please use the hologram-ai desktop or CLI for models larger than 2GB, or use a smaller/quantized model.`);
    }
    
    let onnxBytes: Uint8Array;
    try {
      onnxBytes = new Uint8Array(await onnxFile.arrayBuffer());
    } catch (err: any) {
      throw new Error(`Failed to read ONNX file into memory`);
    }
    
    let externalDataBytes: Uint8Array | null = null;
    try {
      let dataHandle: FileSystemFileHandle | null = null;
      try { dataHandle = await parentDir.getFileHandle(`${onnxHandle.name}.data`); } catch (e) {}
      if (!dataHandle) {
        try { dataHandle = await parentDir.getFileHandle(`${onnxHandle.name}_data`); } catch (e) {}
      }
      if (dataHandle) {
        const dataFile = await dataHandle.getFile();
        externalDataBytes = new Uint8Array(await dataFile.arrayBuffer());
      }
    } catch (e) {}

    let holoBytes: Uint8Array;
    if (externalDataBytes) {
      emitLine("models://compile-line", { stream: "stdout", line: `Loaded ONNX and external data. Compiling via wasm...` });
      holoBytes = await compileOnnxWithData(onnxBytes, externalDataBytes);
    } else {
      emitLine("models://compile-line", { stream: "stdout", line: `Loaded ONNX. Compiling via wasm...` });
      const { compile } = await import("./holo");
      holoBytes = await compile(onnxBytes);
    }
    
    const kappa = await computeKappa(holoBytes);
    const holoName = `${kappa}.holo`;
    const holoHandle = await localDir.getFileHandle(holoName, { create: true });
    const writable = await holoHandle.createWritable();
    await writable.write(holoBytes as any);
    await writable.close();
    
    emitLine("models://compile-line", { stream: "stdout", line: `Compiled and saved to ${holoName} (${holoBytes.length} bytes).` });
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
