// The persistent download worker (journey stage S1).
//
// Two phases, per docs/conceptual-model/02-user-journey.md:
//
// 1. PREFLIGHT — the model is validated before any shard byte moves: the
//    tensor manifest is read from the shards' safetensors headers alone
//    (ranged requests — kilobytes, not weights) and the parametric graph is
//    built from config + manifest inside wasm. An unsupported family, a
//    malformed config, or an unrealizable manifest rejects HERE, naming the
//    reason, with zero shard bytes transferred.
// 2. STREAM — each shard's tensors are walked from the already-parsed header,
//    κ-hashed incrementally, persisted to OPFS as tensors/{κ}.bin, and the
//    content is discarded as it is retrieved. The final step is mechanical:
//    bind the streamed κs into the already-validated graph and emit the
//    k-form archive — it cannot fail on model validity.
import {
  compileSafetensorsStaged,
  compileSafetensorsStreamed,
  validateModelConfig,
  validateStreamedManifest,
  KappaHasher,
  ensureReady,
} from "./holo";

self.onmessage = async (e) => {
  const { type, payload } = e.data;
  if (type === "download_safetensors") {
    await handleSafetensorsDownload(payload);
  } else if (type === "download_onnx") {
    await handleOnnxDownload(payload);
  }
};

function emitProgress(msg: string) {
  self.postMessage({ type: "progress", line: msg });
}

function emitDone(holoBytes: Uint8Array) {
  self.postMessage({ type: "done", holoBytes });
}

function emitDoneStaged(stageCount: number) {
  self.postMessage({ type: "done_staged", stageCount });
}

function emitError(err: string) {
  self.postMessage({ type: "error", error: err });
}

interface ShardTensor {
  key: string;
  dtype: string;
  shape: number[];
  data_offsets: [number, number];
}

interface ShardManifest {
  rfilename: string;
  url: string;
  headerLen: number;
  tensors: ShardTensor[];
}

/** Fetch `[start, endInclusive]` of `url`. Honors 206; if the server ignores
 * Range (200), reads only the needed prefix and cancels the stream — never
 * buffering a shard body. */
async function fetchRange(url: string, start: number, endInclusive: number): Promise<Uint8Array> {
  const response = await fetch(url, { headers: { Range: `bytes=${start}-${endInclusive}` } });
  if (!response.ok || !response.body) {
    throw new Error(`Failed ranged fetch of ${url} (HTTP ${response.status})`);
  }
  if (response.status === 206) {
    return new Uint8Array(await response.arrayBuffer());
  }
  if (start !== 0) {
    await response.body.cancel();
    throw new Error(`server ignored the Range request for ${url}`);
  }
  const wanted = endInclusive + 1;
  const reader = response.body.getReader();
  const out = new Uint8Array(wanted);
  let got = 0;
  while (got < wanted) {
    const { done, value } = await reader.read();
    if (done) throw new Error(`EOF before ${wanted} bytes of ${url}`);
    const take = Math.min(wanted - got, value.length);
    out.set(value.subarray(0, take), got);
    got += take;
  }
  await reader.cancel();
  return out;
}

/** Read a shard's safetensors header via ranged requests (8-byte u64 length +
 * JSON), returning its tensor manifest sorted by data offset. */
async function fetchShardManifest(url: string, rfilename: string): Promise<ShardManifest> {
  const lenBytes = await fetchRange(url, 0, 7);
  const headerLen = Number(
    new DataView(lenBytes.buffer, lenBytes.byteOffset, 8).getBigUint64(0, true),
  );
  if (!Number.isSafeInteger(headerLen) || headerLen === 0) {
    throw new Error(`${rfilename}: implausible safetensors header length ${headerLen}`);
  }
  const headerBytes = await fetchRange(url, 8, 8 + headerLen - 1);
  const header = JSON.parse(new TextDecoder().decode(headerBytes));
  const tensors: ShardTensor[] = [];
  for (const [key, meta] of Object.entries(header)) {
    if (key === "__metadata__") continue;
    const m = meta as { dtype: string; shape: number[]; data_offsets: [number, number] };
    tensors.push({ key, dtype: m.dtype, shape: m.shape, data_offsets: m.data_offsets });
  }
  tensors.sort((a, b) => a.data_offsets[0] - b.data_offsets[0]);
  return { rfilename, url, headerLen, tensors };
}

export async function handleSafetensorsDownload(
  {
    modelId,
    configText,
    files,
    contextLength,
    layersPerStage,
    stageCount,
    localName,
    revision,
    cacheBudgetBytes,
    hfBase,
  }: {
    modelId: string;
    configText: string;
    files: { rfilename: string }[];
    contextLength?: number;
    layersPerStage?: number;
    stageCount?: number;
    localName?: string;
    revision?: string;
    cacheBudgetBytes?: number;
    hfBase?: string;
  },
  emitProgressFn = emitProgress,
  emitDoneFn = emitDone,
  emitDoneStagedFn = emitDoneStaged,
  emitErrorFn = emitError,
) {
  try {
    await ensureReady();
    const base = hfBase ?? "https://huggingface.co";

    // ── Phase 1: preflight ─────────────────────────────────────────────────
    // Config-only first: an unsupported family or malformed config rejects
    // before even the shard HEADERS are touched.
    emitProgressFn(`Preflight: validating ${modelId} config against the family registry...`);
    await validateModelConfig(configText);
    emitProgressFn(`Preflight: reading shard headers for ${modelId} (ranged requests)...`);
    // Revision-pinned URLs: the recorded κ-provenance must be immutable.
    const rev = revision ?? "main";
    const manifests: ShardManifest[] = [];
    for (const file of files) {
      const url = `${base}/${modelId}/resolve/${rev}/${file.rfilename}`;
      manifests.push(await fetchShardManifest(url, file.rfilename));
    }
    const allKeys: string[] = [];
    const allShapes: string[] = [];
    const allDtypes: string[] = [];
    for (const manifest of manifests) {
      for (const tensor of manifest.tensors) {
        allKeys.push(tensor.key);
        allShapes.push(JSON.stringify(tensor.shape));
        allDtypes.push(tensor.dtype);
      }
    }
    emitProgressFn(`Preflight: validating ${modelId} (${allKeys.length} tensors) against the parametric family registry...`);
    await validateStreamedManifest(configText, allKeys, allShapes, allDtypes, contextLength);
    emitProgressFn("Preflight passed: the model is valid; streaming weights.");

    // ── Phase 2: stream over k ─────────────────────────────────────────────
    // The κ-store is a CACHE: tensors persist locally while the budget
    // allows; every tensor's provenance (revision-pinned URL + absolute byte
    // range) is recorded so the rest resolve at run time.
    const root = await navigator.storage.getDirectory();
    const tensorsDir = await root.getDirectoryHandle("tensors", { create: true });
    const allKappas: string[] = [];
    const kappaSources: Record<string, { url: string; start: number; end: number }> = {};
    const cacheBudget = cacheBudgetBytes ?? Number.POSITIVE_INFINITY;
    let cachedBytes = 0;

    for (const manifest of manifests) {
      emitProgressFn(`Streaming ${manifest.rfilename}...`);
      const response = await fetch(manifest.url);
      if (!response.ok || !response.body) {
        throw new Error(`Failed to fetch ${manifest.rfilename}`);
      }
      const reader = response.body.getReader();
      const totalBytes = Number(response.headers.get("content-length")) || 0;
      const baseOffset = 8 + manifest.headerLen;
      const tensors = manifest.tensors;

      let downloadedBytes = 0;
      let lastEmit = 0;
      let currentTensorIdx = 0;
      let currentHasher = new KappaHasher();
      let currentTensorBuffer: Uint8Array | null = null;
      let currentTensorOffset = 0;

      async function processBytes(globalOffset: number, bytes: Uint8Array) {
        let localOffset = 0;
        while (localOffset < bytes.length && currentTensorIdx < tensors.length) {
          const tensor = tensors[currentTensorIdx];
          const tensorStart = baseOffset + tensor.data_offsets[0];
          const tensorEnd = baseOffset + tensor.data_offsets[1];
          const tensorSize = tensorEnd - tensorStart;
          const currentGlobal = globalOffset + localOffset;

          if (currentGlobal < tensorStart) {
            localOffset += Math.min(tensorStart - currentGlobal, bytes.length - localOffset);
          } else if (currentGlobal < tensorEnd) {
            const toFeed = Math.min(tensorEnd - currentGlobal, bytes.length - localOffset);
            const slice = bytes.slice(localOffset, localOffset + toFeed);
            currentHasher.update(slice);

            if (!currentTensorBuffer) {
              currentTensorBuffer = new Uint8Array(tensorSize);
              currentTensorOffset = 0;
            }
            currentTensorBuffer.set(slice, currentTensorOffset);
            currentTensorOffset += slice.length;
            localOffset += toFeed;

            if (currentGlobal + toFeed === tensorEnd) {
              // The tensor is complete: record its provenance, cache it
              // within budget, and DISCARD the content — only the
              // k-representation flows forward.
              const kappa = currentHasher.finalize();
              kappaSources[kappa] = { url: manifest.url, start: tensorStart, end: tensorEnd };
              if (cachedBytes + tensorSize <= cacheBudget) {
                const binHandle = await tensorsDir.getFileHandle(`${kappa}.bin`, { create: true });
                if ("createSyncAccessHandle" in binHandle) {
                  const accessHandle = await (binHandle as any).createSyncAccessHandle();
                  accessHandle.write(currentTensorBuffer);
                  accessHandle.flush();
                  accessHandle.close();
                } else {
                  const writable = await (binHandle as any).createWritable();
                  await writable.write(currentTensorBuffer);
                  await writable.close();
                }
                cachedBytes += tensorSize;
              }
              allKappas.push(kappa);
              currentTensorIdx++;
              currentTensorBuffer = null;
              if (currentTensorIdx < tensors.length) {
                currentHasher = new KappaHasher();
              }
            }
          } else {
            localOffset++;
          }
        }
      }

      while (currentTensorIdx < tensors.length) {
        const { done, value } = await reader.read();
        if (done) break;
        await processBytes(downloadedBytes, value);
        downloadedBytes += value.length;

        const now = Date.now();
        if (now - lastEmit > 500) {
          lastEmit = now;
          if (totalBytes > 0) {
            const percent = Math.round((downloadedBytes / totalBytes) * 100);
            emitProgressFn(
              `Streaming ${manifest.rfilename}: ${percent}% (${(downloadedBytes / 1024 / 1024).toFixed(1)}MB)`,
            );
          }
        }
      }

      if (currentTensorIdx < tensors.length) {
        throw new Error(`EOF before finishing tensor ${tensors[currentTensorIdx].key}`);
      }
      emitProgressFn(`Finished streaming ${manifest.rfilename}.`);
    }

    // Record κ-provenance + model metadata (the session's context window)
    // under the model directory.
    if (localName) {
      const modelsDir = await root.getDirectoryHandle("models", { create: true });
      const localDir = await modelsDir.getDirectoryHandle(localName, { create: true });
      const srcHandle = await localDir.getFileHandle("kappa-sources.json", { create: true });
      const writable = await srcHandle.createWritable();
      await writable.write(JSON.stringify(kappaSources));
      await writable.close();
      const metaHandle = await localDir.getFileHandle("model-meta.json", { create: true });
      const metaWritable = await metaHandle.createWritable();
      await metaWritable.write(JSON.stringify({ contextLength }));
      await metaWritable.close();
      // The streamed manifest (name → κ/shape/dtype): chat recompiles stage
      // windows from it, so the generation window can follow the sequence.
      const manifestHandle = await localDir.getFileHandle("manifest.json", { create: true });
      const manifestWritable = await manifestHandle.createWritable();
      await manifestWritable.write(
        JSON.stringify({ keys: allKeys, kappas: allKappas, shapes: allShapes, dtypes: allDtypes }),
      );
      await manifestWritable.close();
    }

    // Mechanical: the graph was validated in preflight; this binds the
    // streamed κs and emits the k-form archive(s). Models beyond the
    // execution window compile as stage archives (windowed execution over k).
    if (stageCount && stageCount > 1 && layersPerStage && localName) {
      emitProgressFn(
        `Binding streamed κs into ${stageCount} stage archives (windowed execution over k)...`,
      );
      const stages = await compileSafetensorsStaged(
        configText,
        allKeys,
        allKappas,
        allShapes,
        allDtypes,
        contextLength,
        layersPerStage,
      );
      const modelsDir = await root.getDirectoryHandle("models", { create: true });
      const localDir = await modelsDir.getDirectoryHandle(localName, { create: true });
      const stagesDir = await localDir.getDirectoryHandle("stages", { create: true });
      for (let i = 0; i < stages.length; i++) {
        const handle = await stagesDir.getFileHandle(`${i}.holo`, { create: true });
        const writable = await handle.createWritable();
        await writable.write(stages[i] as unknown as ArrayBufferView<ArrayBuffer>);
        await writable.close();
      }
      const metaHandle = await localDir.getFileHandle("stages.json", { create: true });
      const writable = await metaHandle.createWritable();
      await writable.write(
        JSON.stringify({ stageCount: stages.length, layersPerStage, contextLength }),
      );
      await writable.close();
      emitDoneStagedFn(stages.length);
      return;
    }

    emitProgressFn("Binding streamed κs into the k-form archive...");
    const holoBytes = await compileSafetensorsStreamed(
      configText,
      allKeys,
      allKappas,
      allShapes,
      allDtypes,
      contextLength,
    );

    emitDoneFn(holoBytes);
  } catch (err) {
    emitErrorFn(String(err));
  }
}

// ONNX models cannot be streamed over k without a complex streaming protobuf parser in JS.
// Per architectural constraints, the implementation MUST operate over the k-representation
// and stream in every aspect to avoid contrived 32-bit WebAssembly limits.
// Therefore, ONNX downloads are not supported in the web IDE.
async function handleOnnxDownload(
  _args: unknown,
  _emitProgressFn = emitProgress,
  _emitDoneFn = emitDone,
  emitErrorFn = emitError,
) {
  emitErrorFn(
    "ONNX models are not supported in the web IDE because they cannot be streamed over the holospaces/k-representation. Please use safetensors.",
  );
}
