// The generation worker (journey stages S3 + S4): materialize the k-form
// archive against the OPFS κ-store, then run the real generation loop,
// streaming tokens back to the page.
//
// Materialization is resolution, not loading: each κ-map entry is read from
// `tensors/{κ}.bin` through a sync access handle (worker-only API), re-hashed
// inside the pipeline, and patched into the archive's constants. A missing or
// corrupt κ aborts the turn with the label.
import { generate as wasmGenerate, kappaRequirements, materialize } from "./holo";

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

self.onmessage = async (e) => {
  const { holoBytes, prompt, genOpts, tokenizerBytes } = e.data;

  try {
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
