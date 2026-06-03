import { invoke } from "@tauri-apps/api/core";
import { listen, UnlistenFn } from "@tauri-apps/api/event";

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
  /**
   * Per-model "between turns" template, with `{response}` substituted by
   * the prior assistant message. The desktop UI uses this to inject prior
   * turns into the `{prompt}` slot of `promptTemplate` for multi-turn chat.
   */
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

// Tauri serializes Rust struct fields in snake_case by default. Re-shape on
// the way in to keep TypeScript camelCase.
function camelize<T>(obj: unknown): T {
  if (obj === null || typeof obj !== "object") return obj as T;
  if (Array.isArray(obj)) return obj.map((x) => camelize<unknown>(x)) as T;
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(obj as Record<string, unknown>)) {
    const ck = k.replace(/_([a-z])/g, (_, c: string) => c.toUpperCase());
    out[ck] = camelize<unknown>(v);
  }
  return out as T;
}

export async function workspacePaths(): Promise<WorkspacePaths> {
  return camelize<WorkspacePaths>(await invoke("workspace_paths"));
}

export async function listKnownModels(): Promise<KnownModelStatus[]> {
  return camelize<KnownModelStatus[]>(await invoke("list_known_models"));
}

export async function listCompiledArchives(): Promise<CompiledArchive[]> {
  return camelize<CompiledArchive[]>(await invoke("list_compiled_archives"));
}

export async function downloadKnownModel(id: string): Promise<number> {
  return invoke<number>("download_known_model", { req: { id } });
}

export async function compileKnownModel(id: string): Promise<number> {
  return invoke<number>("compile_known_model", { req: { id } });
}

export interface GenerateOpts {
  archive: string;
  prompt: string;
  maxTokens?: number;
  temperature?: number;
  topK?: number;
  stop?: string[];
  /**
   * Chat prompt template with a `{prompt}` placeholder. Forwarded to the
   * CLI as `--prompt-template` so the model's native format wraps the
   * inner multi-turn content. Templates are no longer baked into `.holo`
   * archives (closed section set) — they must be supplied per invocation.
   */
  promptTemplate?: string;
}

export async function generate(opts: GenerateOpts): Promise<number> {
  return invoke<number>("generate", {
    req: {
      archive: opts.archive,
      prompt: opts.prompt,
      max_tokens: opts.maxTokens,
      temperature: opts.temperature,
      top_k: opts.topK,
      stop: opts.stop ?? [],
      prompt_template: opts.promptTemplate,
    },
  });
}

export async function cancelGeneration(): Promise<boolean> {
  return invoke<boolean>("cancel_generation");
}

export async function recentLogs(since: number): Promise<LogsResponse> {
  return camelize<LogsResponse>(await invoke("recent_logs", { since }));
}

export async function clearLogs(): Promise<void> {
  await invoke("clear_logs");
}

export function onProcessLine(
  event: string,
  cb: (line: ProcessLine) => void,
): Promise<UnlistenFn> {
  return listen<ProcessLine>(event, (e) => cb(e.payload));
}
