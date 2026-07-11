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
  /** A paired speculative DRAFT model (row `speculative-draft-pairing`): an
   * hfId whose small model drafts continuations for this target under
   * speculative decode. DATA, never code — the pairing is a catalogue statement.
   * Downloading this target downloads the paired draft too, and selecting it
   * makes the drafter available. Same-family (shared tokenizer/vocab) so the
   * draft covers the target's vocabulary; a self-pairing (`= hfId`) is a valid
   * guaranteed-compatible draft. Absent ⇒ speculative decode uses prompt-lookup. */
  draftModel?: string;
  localDir: string | null;
  downloaded: boolean;
  compiledArchive: string | null;
  /** `true` for a shipped-catalogue suggestion (a FEATURED starting point the
   * user has not adopted), `false` for a model the user added themselves. The
   * Models page keeps featured suggestions OUT of "My Models" until the user
   * actually downloads one — a curated starter list is not the user's library. */
  featured: boolean;
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

type CatalogueEntry = Omit<
  KnownModelStatus,
  "localDir" | "downloaded" | "compiledArchive" | "featured"
>;

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

type TaggedEntry = CatalogueEntry & { featured: boolean };

async function getCatalogue(): Promise<TaggedEntry[]> {
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
  // Shipped entries are FEATURED suggestions; the user's own added entries are
  // not. A custom entry that shadows a shipped hfId wins (the user adopted it).
  const merged: TaggedEntry[] = shipped
    .filter((s) => !custom.some((c) => c.hfId === s.hfId))
    .map((s) => ({ ...s, featured: true }));
  for (const entry of custom) {
    merged.push({ ...entry, featured: false });
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
    quantize: "int8",
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

// One model's download+compile, keyed by hfId (structural `model`) — the shared
// core of `downloadKnownModel`, so a target and its paired draft download the
// identical parametric way.
async function downloadOne(model: { hfId: string; quantize: string }): Promise<void> {
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
        // The quantized tier is a per-model catalogue statement (data, never
        // code); the localStorage knob forces it for hermetic witnesses.
        const quantize = localStorage.getItem("hologram_quantize") ?? model.quantize;
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
            hfBase: hfBase(),
            quantize
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
}

export async function downloadKnownModel(id: string): Promise<number> {
  const catalogue = await getCatalogue();
  const model = catalogue.find(m => m.id === id);
  if (!model) throw new Error("Unknown model");

  await downloadOne(model);

  // Speculative draft pairing (row `speculative-draft-pairing`): a target that
  // names a `draftModel` downloads its paired draft too, the same parametric
  // way, so selecting the target makes the drafter available. A self-pairing
  // (`draftModel === hfId`) shares the target's own dir — no second download. A
  // draft-download failure is surfaced but NEVER fails the target: speculative
  // decode degrades to prompt-lookup, the journey never dead-ends.
  if (model.draftModel && model.draftModel !== model.hfId) {
    try {
      emitLine("models://download-line", {
        stream: "stdout",
        line: `Downloading paired draft model ${model.draftModel}...`,
      });
      await downloadOne({ hfId: model.draftModel, quantize: model.quantize });
    } catch (e) {
      emitLine("models://download-line", {
        stream: "stderr",
        line: `Paired draft model ${model.draftModel} did not download (speculative decode falls back to prompt-lookup): ${e}`,
      });
    }
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
/** A special token as it appears in `tokenizer_config.json`: a bare string, or
 * an `AddedToken` object with a `content` field. Anything else has no token. */
function tokenString(v: unknown): string | undefined {
  if (typeof v === "string") return v;
  if (v && typeof v === "object" && typeof (v as { content?: unknown }).content === "string") {
    return (v as { content: string }).content;
  }
  return undefined;
}

/** The model's chat config DERIVED from its own config files — never hard-coded
 * per model. `chatTemplate` is the model's Jinja template (rendered with the
 * conversation by the caller); `eosToken`/`bosToken` are its special-token
 * strings (from `tokenizer_config.json`); `eosTokenId` is its generation stop
 * id (from `generation_config.json`, else `config.json`). A model that ships no
 * `chat_template` (a base, non-chat model) returns none and the caller falls
 * back to the catalogue's template. */
export interface ModelChat {
  chatTemplate?: string;
  eosToken?: string;
  bosToken?: string;
  eosTokenId?: number;
}

/** The model's OWN end-of-sequence token id, read from a HF config object's
 * `eos_token_id`. HF allows a single id OR an array of ids (a model with
 * several terminators, e.g. Llama-3's `<|end_of_text|>` + `<|eot_id|>`). A
 * single `Option<u32>` engine stop cannot represent a set without picking one
 * arbitrarily, so we derive an id ONLY for the unambiguous single-number case;
 * for the array case we return none and let the rendered chat-template stop
 * strings (which carry every terminator) do the stopping — parametric, no
 * per-model guess. */
function parseEosTokenId(cfg: unknown): number | undefined {
  const raw = (cfg as { eos_token_id?: unknown } | null)?.eos_token_id;
  return typeof raw === "number" && Number.isInteger(raw) && raw >= 0 ? raw : undefined;
}

async function loadEosTokenId(dirName: string): Promise<number | undefined> {
  // generation_config.json is authoritative for generation; config.json is the
  // fallback. Exact basenames — never a substring guess.
  for (const file of ["generation_config.json", "config.json"]) {
    const raw = await readModelFile(dirName, file);
    if (!raw) continue;
    try {
      const id = parseEosTokenId(JSON.parse(new TextDecoder().decode(raw)));
      if (id !== undefined) return id;
    } catch {
      // Malformed config: try the next candidate rather than fail the load.
    }
  }
  return undefined;
}

async function loadModelChat(dirName: string): Promise<ModelChat> {
  const eosTokenId = await loadEosTokenId(dirName);
  const raw = await readModelFile(dirName, "tokenizer_config.json");
  if (!raw) return { eosTokenId };
  try {
    const cfg = JSON.parse(new TextDecoder().decode(raw));
    // `chat_template` is usually a string; some repos ship an array of named
    // templates ({name, template}) — take the `default`, else the first.
    let chatTemplate: string | undefined;
    if (typeof cfg.chat_template === "string") {
      chatTemplate = cfg.chat_template;
    } else if (Array.isArray(cfg.chat_template)) {
      const named = cfg.chat_template as Array<{ name?: string; template?: string }>;
      chatTemplate =
        named.find((t) => t.name === "default")?.template ?? named[0]?.template;
    }
    return {
      chatTemplate,
      eosToken: tokenString(cfg.eos_token),
      bosToken: tokenString(cfg.bos_token),
      eosTokenId,
    };
  } catch {
    return { eosTokenId };
  }
}

export async function loadSessionMeta(
  archivePath: string,
): Promise<{ contextLength: number | null; tokenizer?: Uint8Array } & ModelChat> {
  const dirName = archivePath.split("/")[1];
  const { contextLength } = await sessionInfo(archivePath);
  const tokenizer = (await readModelFile(dirName, "tokenizer.json")) ?? undefined;
  const chat = await loadModelChat(dirName);
  return { contextLength, tokenizer, ...chat };
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
  /** The model's OWN end-of-sequence token id (from its generation_config.json /
   * config.json). Passed so generation stops on the model's real eos rather than
   * a tokenizer default — parametric over any model. */
  eos?: number;
  /** Prewarm only (row `eager-prewarm`): spawn the decode pool and build the warm
   * staged session, then stop — do NOT generate. Moves the pool-spawn + session
   * build OFF the first real turn's TTFT. `prompt` is ignored. Resolves when the
   * worker reports `warmed`; the worker + pool stay live for the first turn. */
  warm?: boolean;
}

let activeWorker: Worker | null = null;

// The multi-threaded decode pool workers (ADR-0018). The MAIN thread owns them —
// not the execute (generate) worker that requests them — so that terminating the
// execute worker on cancel/error tears the pool down here too. Otherwise the pool
// workers orphan and each pins the whole (model-sized) shared linear memory,
// leaking one memory per cancel until the tab OOMs.
let poolWorkers: Worker[] = [];
// Whether the pool passed registration (the execute worker posted `pool-committed`).
// Before commit, a pool-worker failure means "fall back to single-threaded"; after
// commit it means a mid-session fault → abort the turn (ADR-0018 M2/M3).
let poolCommitted = false;
function terminatePool(): void {
  for (const w of poolWorkers) w.terminate();
  poolWorkers = [];
  poolCommitted = false;
  // Telemetry for the teardown witness (no orphan accumulation across cancels).
  (globalThis as unknown as { __hologram_pool_live?: number }).__hologram_pool_live = 0;
}

// Settlers for the in-flight generate() turn, if any. Module-level (not captured
// in a per-turn closure) because the pool workers' `onerror` handlers are set on
// the FIRST turn but must settle whichever turn is live now — a warm turn 2+ has
// different resolve/reject (ADR-0018 M3). `resolveActiveTurn` also lets cancel
// settle the turn: without it, `.terminate()` fires no message, the generate()
// promise never resolves, and the caller's `finally` (which re-enables the UI)
// never runs — the pre-ADR cancel-hang.
let resolveActiveTurn: ((v: number) => void) | null = null;
let rejectActiveTurn: ((e: Error) => void) | null = null;
function clearTurnSettlers(): void {
  resolveActiveTurn = null;
  rejectActiveTurn = null;
}

// A pool worker failed. Before the execute worker commits (registration passed),
// a failure means "fall back to single-threaded": tear the pool down and tell the
// execute worker (its readiness poll then fails fast). After commit, the execute
// worker is blocked in a synchronous decode over the now-broken pool (a dead
// worker never decrements the substrate's join count → the fork-join would hang
// forever), so ABORT the live turn: terminate everything and reject it.
function failPool(why: string): void {
  if (poolCommitted) {
    terminatePool();
    if (activeWorker) { activeWorker.terminate(); activeWorker = null; }
    const r = rejectActiveTurn;
    clearTurnSettlers();
    r?.(new Error(why));
  } else {
    terminatePool();
    activeWorker?.postMessage({ type: 'pool-failed', why });
  }
}

/** The local dir of the target's paired speculative DRAFT model (row
 * `speculative-draft-pairing`), if the catalogue pairs one AND it is compiled
 * locally. Returns undefined otherwise — the worker then drafts by prompt-lookup.
 * `targetDirName` is the target's model dir (its hfId's leaf). A draft is usable
 * only when STAGED (the decode-session drafter reads `stages.json`); a
 * self-pairing points at the target's own — compiled — dir. */
async function resolveDraftModelDir(targetDirName: string): Promise<string | undefined> {
  const catalogue = await getCatalogue();
  const target = catalogue.find((m) => (m.hfId.split("/").pop() || m.hfId) === targetDirName);
  const draftHfId = target?.draftModel;
  if (!draftHfId) return undefined;
  const draftDirName = draftHfId.split("/").pop() || draftHfId;
  try {
    const root = await getOpfsDir();
    const modelsDir = await root.getDirectoryHandle("models");
    const draftDir = await modelsDir.getDirectoryHandle(draftDirName);
    await draftDir.getFileHandle("stages.json");
    return draftDirName;
  } catch {
    return undefined;
  }
}

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
    eos: opts.eos,
  };
  // Speculative decode (row `speculative-decode`): opt-in via the
  // `hologram_speculative` knob = draft width K (>= 2). Works at ANY temperature
  // (the accept rule samples per absolute position — byte-identical to plain
  // decode at that temperature); a stale/absent verify pipeline falls back to
  // plain decode. Off by default — the verify pipeline costs residency and
  // prompt-lookup drafting only speeds up repetitive text, so it is a knob.
  const specKnob = Number(localStorage.getItem("hologram_speculative") ?? "");
  if (Number.isFinite(specKnob) && specKnob >= 2) {
    genOpts.speculative_draft = Math.floor(specKnob);
  }

  // Speculative draft pairing (row `speculative-draft-pairing`): only when
  // speculating, resolve the target's paired draft model dir (if the catalogue
  // pairs one AND it is compiled locally) so the worker builds and attaches it
  // as the drafter. Absent/uncompiled ⇒ the worker drafts by prompt-lookup —
  // never load-bearing, so this only ever adds speed.
  const draftModelDir = genOpts.speculative_draft
    ? await resolveDraftModelDir(archiveParts[1])
    : undefined;

  // A real turn opens an assistant line; a prewarm is invisible to the chat.
  if (!opts.warm) emitLine("chat://line", { stream: "stdout", line: "" });

  return new Promise((resolve, reject) => {
    // Reuse the live worker: it holds the warm staged session. The worker
    // itself rebuilds the session when the model changes (its inputs are
    // the model's), so reuse across sends is always safe.
    activeWorker ??= new GenerateWorker();
    // The live turn's settlers, for module-level `failPool`/`cancelGeneration`.
    resolveActiveTurn = resolve;
    rejectActiveTurn = reject;
    // A worker script/load failure would otherwise leave generation hanging
    // silently (onmessage never fires); surface it loudly.
    activeWorker.onerror = (ev) => {
      const msg = `generate worker failed: ${(ev as ErrorEvent).message ?? ev}`;
      emitLine("chat://line", { stream: "stderr", line: msg });
      terminatePool();
      if (activeWorker) { activeWorker.terminate(); activeWorker = null; }
      clearTurnSettlers();
      reject(new Error(msg));
    };

    activeWorker.onmessage = (e) => {
      if (e.data.type === 'spawn-pool') {
        // The execute worker created the shared memory + did the one-time data
        // init; spawn the pool workers here (main-owned) over that same memory.
        // Rebuild fresh — a stale pool from a prior turn must not linger.
        terminatePool();
        const { module, memory, n, stackSize } = e.data;
        for (let id = 0; id < n; id++) {
          const w = new Worker(new URL('./pool.worker.ts', import.meta.url), { type: 'module' });
          // A pool worker throws (bad instantiation, GEMV trap, OOM) → its
          // `onerror`/error message fires here; without this the execute side
          // would only notice via the readiness timeout (M2) or hang (M3).
          w.onerror = () => failPool(`decode pool worker ${id} crashed`);
          w.onmessage = (ev) => {
            if (ev.data && ev.data.error) failPool(`decode pool worker ${id} failed: ${ev.data.error}`);
          };
          w.postMessage({ module, memory, id, stackSize });
          poolWorkers.push(w);
        }
        (globalThis as unknown as { __hologram_pool_live?: number }).__hologram_pool_live = n;
      } else if (e.data.type === 'pool-committed') {
        poolCommitted = true;
      } else if (e.data.type === 'warmed') {
        // Prewarm done (row `eager-prewarm`): the pool is spawned and the staged
        // session is built. Keep the worker + pool LIVE — the first real turn
        // reuses both, paying neither on its TTFT. Settle without a completion.
        clearTurnSettlers();
        resolve(0);
      } else if (e.data.type === 'pool-teardown') {
        // The execute worker fell back to single-threaded — drop the pool it no
        // longer uses (M1), so it does not linger pinning the shared memory.
        terminatePool();
      } else if (e.data.type === 'token') {
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
        clearTurnSettlers();
        resolve(0);
      } else if (e.data.type === 'error') {
        emitLine("chat://line", { stream: "stderr", line: e.data.error });
        terminatePool();
        if (activeWorker) {
          activeWorker.terminate();
          activeWorker = null;
        }
        clearTurnSettlers();
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
      // The decode plan (row `decode-plan`) is the default per-token path —
      // one position per step, never a window-sized forward. The knob
      // reverts to the whole-window plan for witnesses and diagnosis.
      decodePlan: localStorage.getItem("hologram_decode_plan") !== "0",
      // Multi-threaded decode pool (ADR-0018): on unless `hologram_threads=0`.
      // The V&V flips it to compare threaded vs single-threaded byte-for-byte;
      // it is also a user escape hatch. A no-op unless cross-origin-isolated.
      threads: localStorage.getItem("hologram_threads") !== "0",
      // Pool worker count override (tuning/diagnosis); default = logical cores − 1.
      poolWorkers: (() => {
        const w = Number(localStorage.getItem("hologram_pool_workers") ?? "");
        return Number.isFinite(w) && w > 0 ? Math.floor(w) : undefined;
      })(),
      // The weight-tier pager (row `lazy-constant-residency`): opt-in via the
      // knob (MB), so a stage whose weights exceed the heap window pages them
      // from the OPFS κ-store rather than pinning the whole stage resident.
      weightBudget: (() => {
        const mb = Number(localStorage.getItem("hologram_weight_budget") ?? "");
        return Number.isFinite(mb) && mb > 0 ? Math.floor(mb * 1024 * 1024) : undefined;
      })(),
      // The paired speculative draft model's dir (row `speculative-draft-pairing`),
      // resolved above only when speculating and only when a compiled draft is
      // present — else undefined and the worker drafts by prompt-lookup.
      draftModelDir,
      // Prewarm (row `eager-prewarm`): the worker spawns the pool + builds the
      // session, then stops before generate. Off the first turn's TTFT.
      warm: opts.warm === true,
    });
  });
}

export async function cancelGeneration(): Promise<boolean> {
  if (activeWorker) {
    // Tear down the decode pool with the execute worker — a hard terminate is the
    // only way to interrupt the synchronous decode, and the pool workers pin the
    // shared memory, so they must go too (ADR-0018).
    terminatePool();
    activeWorker.terminate();
    activeWorker = null;
    // Settle the in-flight turn (resolve = graceful stop, keeping the partial
    // completion): a terminated worker emits no message, so without this the
    // caller's `await generate()` would hang and its `finally` (which re-enables
    // the composer) would never run — the composer would stay disabled.
    const r = resolveActiveTurn;
    clearTurnSettlers();
    r?.(0);
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
