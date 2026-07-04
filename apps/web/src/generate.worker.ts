// The generation worker (journey stages S3 + S4): materialize the k-form
// archive against the OPFS κ-store, then run the real generation loop,
// streaming tokens back to the page.
//
// Materialization is resolution, not loading: each κ-map entry is read from
// `tensors/{κ}.bin` through a sync access handle (worker-only API), re-hashed
// inside the pipeline, and patched into the archive's constants. A missing or
// corrupt κ aborts the turn with the label.
import { generate as wasmGenerate, generateStaged, kappaRequirements, materialize } from "./holo";

async function materializeFromOpfs(holoBytes: Uint8Array): Promise<Uint8Array> {
  const required = await kappaRequirements(holoBytes);
  if (required.length === 0) return holoBytes;

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
        // Leave the κ unopened: the resolver returns undefined and the
        // pipeline aborts naming the label — the loud S3 failure contract.
      }
    }
    return await materialize(holoBytes, (kappa: string) => {
      const handle = handles.get(kappa);
      if (!handle) return undefined;
      const size = handle.getSize();
      const buf = new Uint8Array(size);
      handle.read(buf, { at: 0 });
      handle.close();
      handles.delete(kappa);
      return buf;
    });
  } finally {
    for (const handle of handles.values()) handle.close();
  }
}

interface OpenHandle {
  getSize(): number;
  read(buffer: Uint8Array, options?: { at: number }): number;
  close(): void;
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

// Staged generation (windowed execution over k): stage archives are read
// upfront (small k-forms — no weights); κs resolve on demand through
// pre-opened sync handles, re-readable per stage per pass. One stage's
// weights are resident at a time.
async function runStaged(
  staged: { modelDir: string; stageCount: number },
  prompt: string,
  genOpts: unknown,
  tokenizerBytes: Uint8Array | undefined,
  onToken: (text: string) => void,
): Promise<string> {
  const root = await navigator.storage.getDirectory();
  const models = await root.getDirectoryHandle("models");
  const modelDir = await models.getDirectoryHandle(staged.modelDir);
  const stagesDir = await modelDir.getDirectoryHandle("stages");

  const stageArchives: Uint8Array[] = [];
  const required = new Set<string>();
  for (let i = 0; i < staged.stageCount; i++) {
    const handle = await openSync(stagesDir, `${i}.holo`);
    const bytes = readAll(handle);
    handle.close();
    stageArchives.push(bytes);
    for (const kappa of await kappaRequirements(bytes)) required.add(kappa);
  }

  const tensorsDir = await root.getDirectoryHandle("tensors");
  const kappaHandles = new Map<string, OpenHandle>();
  try {
    for (const kappa of required) {
      try {
        kappaHandles.set(kappa, await openSync(tensorsDir, `${kappa}.bin`));
      } catch {
        // Unresolvable κ: the resolver returns undefined and the pipeline
        // aborts naming the label — the loud S3 failure contract.
      }
    }
    return await generateStaged(
      staged.stageCount,
      (i: number) => stageArchives[i],
      (kappa: string) => {
        const handle = kappaHandles.get(kappa);
        return handle ? readAll(handle) : undefined;
      },
      prompt,
      genOpts as never,
      tokenizerBytes,
      onToken,
    );
  } finally {
    for (const handle of kappaHandles.values()) handle.close();
  }
}

self.onmessage = async (e) => {
  const { holoBytes, staged, prompt, genOpts, tokenizerBytes } = e.data;

  try {
    if (staged) {
      const result = await runStaged(staged, prompt, genOpts, tokenizerBytes, (text: string) => {
        self.postMessage({ type: "token", text });
      });
      self.postMessage({ type: "done", text: result });
      return;
    }
    const material = await materializeFromOpfs(holoBytes);
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
