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
  compileSafetensorsStagedQuantized,
  compileSafetensorsStreamed,
  computeKappa,
  deriveQuantizedArtifact,
  optimalQuantTier,
  quantizableWeights,
  headQuantChunks,
  validateModelConfig,
  validateStreamedManifest,
  ensureReady,
  type QuantEntry,
} from "./holo";
// The download worker is single-threaded (it never decodes; the pool only
// parallelises the `m == 1` decode GEMV — ADR-0018), so it binds `KappaHasher`
// straight from the single-threaded glue rather than through `holo`, which now
// resolves its glue dynamically. Same module instance `holo.ensureReady()`
// initialises here — one wasm, one init.
import { KappaHasher } from "./wasm/hologram_ai_wasm.js";

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
  /** The shard's content pin (HTTP ETag — on the HF Hub this is the blob's
   * content hash). Identical pin ⇒ byte-identical shard ⇒ every recorded
   * (range → κ) of an earlier stream is valid (row `network-skip`). */
  etag: string | null;
}

/** Fetch `[start, endInclusive]` of `url`. Honors 206; if the server ignores
 * Range (200), reads only the needed prefix and cancels the stream — never
 * buffering a shard body. */
async function fetchRangeMeta(
  url: string,
  start: number,
  endInclusive: number,
): Promise<{ bytes: Uint8Array; etag: string | null }> {
  const response = await fetch(url, { headers: { Range: `bytes=${start}-${endInclusive}` } });
  const etag = response.headers.get("x-linked-etag") ?? response.headers.get("etag");
  const bytes = await readRangeBody(response, url, start, endInclusive);
  return { bytes, etag };
}

async function fetchRange(url: string, start: number, endInclusive: number): Promise<Uint8Array> {
  return (await fetchRangeMeta(url, start, endInclusive)).bytes;
}

async function readRangeBody(
  response: Response,
  url: string,
  start: number,
  endInclusive: number,
): Promise<Uint8Array> {
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
  const { bytes: lenBytes, etag } = await fetchRangeMeta(url, 0, 7);
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
  return { rfilename, url, headerLen, tensors, etag };
}

/** A shard's transit prior: absolute range → κ, recorded by an earlier
 * stream, valid only under the same content pin. */
type ShardPrior = Record<string, string>;

/** The prior's OPFS name for a pin (an ETag is quoted and may be weak). */
function pinFileName(etag: string): string {
  return `${etag.replace(/[^A-Za-z0-9._-]/g, "")}.json`;
}

async function loadShardPrior(
  provDir: FileSystemDirectoryHandle,
  etag: string,
): Promise<ShardPrior | null> {
  try {
    const handle = await provDir.getFileHandle(pinFileName(etag));
    return JSON.parse(await (await handle.getFile()).text()) as ShardPrior;
  } catch {
    return null;
  }
}

async function saveShardPrior(
  provDir: FileSystemDirectoryHandle,
  etag: string,
  prior: ShardPrior,
): Promise<void> {
  try {
    const handle = await provDir.getFileHandle(pinFileName(etag), { create: true });
    const writable = await handle.createWritable();
    await writable.write(JSON.stringify(prior));
    await writable.close();
  } catch {
    // The prior is an optimization; a lost write only costs a re-stream.
  }
}

/** Resolve a tensor's bytes for derivation: the local κ-store first, then
 * the recorded provenance (an async ranged fetch — content addressing makes
 * it exactly as trustworthy; the artifact's own κ pins the result). */
async function readTensorBytes(
  tensorsDir: FileSystemDirectoryHandle,
  kappa: string,
  sources: Record<string, { url: string; start: number; end: number }>,
): Promise<Uint8Array> {
  try {
    const handle = await tensorsDir.getFileHandle(`${kappa}.bin`);
    return new Uint8Array(await (await handle.getFile()).arrayBuffer());
  } catch {
    const source = sources[kappa];
    if (!source) throw new Error(`κ \`${kappa}\` resolves nowhere: not cached, no recorded provenance`);
    const res = await fetch(source.url, {
      headers: { Range: `bytes=${source.start}-${source.end - 1}` },
    });
    if (!res.ok && res.status !== 206) {
      throw new Error(`provenance fetch failed (HTTP ${res.status}) for κ \`${kappa}\``);
    }
    const body = new Uint8Array(await res.arrayBuffer());
    return res.status === 200 ? body.slice(source.start, source.end) : body;
  }
}

/** Write a load-bearing (crystalline) file, evaporating gas-phase tensor
 * cache entries to make room when the quota refuses (lifecycle σ-order:
 * cached tensors are provenance-recoverable; the structure is not). Fails
 * loud only when nothing is left to evaporate. */
async function writeEssential(
  tensorsDir: FileSystemDirectoryHandle,
  evictable: string[],
  emitProgressFn: (msg: string) => void,
  write: () => Promise<void>,
): Promise<void> {
  for (;;) {
    try {
      await write();
      return;
    } catch (e) {
      const kappa = evictable.pop();
      if (!kappa) throw e;
      await tensorsDir.removeEntry(`${kappa}.bin`).catch(() => {});
      emitProgressFn(
        `κ-store pressure: evaporated a cached tensor (gas phase, provenance-recoverable) to persist the model structure.`,
      );
    }
  }
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
    quantize,
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
    quantize?: string;
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
    // A staged plan validates its stage graphs (the head chunks at the
    // pipeline's own granularity); only a monolithic plan validates the
    // monolithic graph and its whole-head working set.
    const stagedPlan = stageCount && stageCount > 1 ? layersPerStage : undefined;
    await validateStreamedManifest(
      configText,
      allKeys,
      allShapes,
      allDtypes,
      contextLength,
      stagedPlan,
    );
    emitProgressFn("Preflight passed: the model is valid; streaming weights.");

    // ── Phase 2: stream over k ─────────────────────────────────────────────
    // The κ-store is a CACHE: tensors persist locally while the budget
    // allows; every tensor's provenance (revision-pinned URL + absolute byte
    // range) is recorded so the rest resolve at run time.
    const root = await navigator.storage.getDirectory();
    const tensorsDir = await root.getDirectoryHandle("tensors", { create: true });
    const provDir = await root.getDirectoryHandle("provenance", { create: true });
    const allKappas: string[] = [];
    const kappaSources: Record<string, { url: string; start: number; end: number }> = {};
    let cacheBudget = cacheBudgetBytes ?? Number.POSITIVE_INFINITY;
    let cachedBytes = 0;
    const cachedKappas: string[] = [];

    for (const manifest of manifests) {
      const baseOffset = 8 + manifest.headerLen;
      const tensors = manifest.tensors;

      // Exact-repeat transit prior (row `network-skip`): under the shard's
      // content pin, ranges whose κ an earlier stream recorded are KNOWN —
      // known = provenance-recorded κ, not cached bytes — and known content
      // never re-transits. No skipped byte is trusted: the κ only enters
      // the manifest; it verifies at first materialization, and a wrong
      // prior unpins and recovers from provenance (`saturation-residency`).
      const prior = manifest.etag ? await loadShardPrior(provDir, manifest.etag) : null;
      const known = new Map<number, string>();
      if (prior) {
        for (const [i, t] of tensors.entries()) {
          const kappa = prior[`${t.data_offsets[0]}-${t.data_offsets[1]}`];
          if (kappa) known.set(i, kappa);
        }
      }

      emitProgressFn(
        known.size > 0
          ? `Streaming ${manifest.rfilename} (${known.size}/${tensors.length} tensors known under the content pin)...`
          : `Streaming ${manifest.rfilename}...`,
      );

      const freshPrior: ShardPrior = {};
      let skippedBytes = 0;
      let lastEmit = 0;

      /** A completed tensor: record provenance, cache within budget, and
       * DISCARD the content — only the k-representation flows forward. The
       * κ-store is a CACHE: a refused write (quota, I/O) stops caching and
       * the journey continues on recorded provenance — measured headroom is
       * a projection, and the environment's real quota answers at the write
       * (row memory-guard: never refused for resources). */
      const finalizeTensor = async (
        tensor: ShardTensor,
        kappa: string,
        buffer: Uint8Array | null,
      ) => {
        const tensorStart = baseOffset + tensor.data_offsets[0];
        const tensorEnd = baseOffset + tensor.data_offsets[1];
        kappaSources[kappa] = { url: manifest.url, start: tensorStart, end: tensorEnd };
        freshPrior[`${tensor.data_offsets[0]}-${tensor.data_offsets[1]}`] = kappa;
        allKappas.push(kappa);
        const tensorSize = tensorEnd - tensorStart;
        if (buffer && cachedBytes + tensorSize <= cacheBudget) {
          try {
            const binHandle = await tensorsDir.getFileHandle(`${kappa}.bin`, { create: true });
            if ("createSyncAccessHandle" in binHandle) {
              const accessHandle = await (binHandle as any).createSyncAccessHandle();
              accessHandle.write(buffer);
              accessHandle.flush();
              accessHandle.close();
            } else {
              const writable = await (binHandle as any).createWritable();
              await writable.write(buffer);
              await writable.close();
            }
            cachedBytes += tensorSize;
            cachedKappas.push(kappa);
          } catch (e) {
            cacheBudget = 0; // the environment answered: cache no further
            void tensorsDir.removeEntry(`${kappa}.bin`).catch(() => {});
            void e;
            emitProgressFn(
              "κ-store quota reached — remaining tensors stream via recorded provenance (cache, not mirror).",
            );
          }
        }
      };

      /** Stream-hash the unknown run `tensors[from..to)` via one coalesced
       * ranged request — the transit is exactly the unknown set. */
      const streamRun = async (from: number, to: number) => {
        const runStart = baseOffset + tensors[from].data_offsets[0];
        const runEnd = baseOffset + tensors[to - 1].data_offsets[1];
        const response = await fetch(manifest.url, {
          headers: { Range: `bytes=${runStart}-${runEnd - 1}` },
        });
        if (!response.ok || !response.body) {
          throw new Error(`Failed to fetch ${manifest.rfilename}`);
        }
        // 206 starts at runStart; a server that ignores Range (200) starts
        // at 0 — the loop skips the prefix and cancels past the run.
        let globalPos = response.status === 206 ? runStart : 0;
        const reader = response.body.getReader();

        let idx = from;
        let hasher = new KappaHasher();
        let buffer: Uint8Array | null = null;
        let bufferOffset = 0;

        while (idx < to) {
          const { done, value } = await reader.read();
          if (done) break;
          let localOffset = 0;
          while (localOffset < value.length && idx < to) {
            const tensor = tensors[idx];
            const tensorStart = baseOffset + tensor.data_offsets[0];
            const tensorEnd = baseOffset + tensor.data_offsets[1];
            const currentGlobal = globalPos + localOffset;
            if (currentGlobal < tensorStart) {
              localOffset += Math.min(tensorStart - currentGlobal, value.length - localOffset);
              continue;
            }
            const toFeed = Math.min(tensorEnd - currentGlobal, value.length - localOffset);
            const slice = value.slice(localOffset, localOffset + toFeed);
            hasher.update(slice);
            if (!buffer) {
              buffer = new Uint8Array(tensorEnd - tensorStart);
              bufferOffset = 0;
            }
            buffer.set(slice, bufferOffset);
            bufferOffset += slice.length;
            localOffset += toFeed;
            if (currentGlobal + toFeed === tensorEnd) {
              await finalizeTensor(tensor, hasher.finalize(), buffer);
              idx++;
              buffer = null;
              if (idx < to) hasher = new KappaHasher();
            }
          }
          globalPos += value.length;

          const now = Date.now();
          if (now - lastEmit > 500) {
            lastEmit = now;
            const streamed = Math.min(globalPos, runEnd) - runStart;
            emitProgressFn(
              `Streaming ${manifest.rfilename}: ${(streamed / 1024 / 1024).toFixed(1)}MB (run ${from + 1}–${to}/${tensors.length})`,
            );
          }
          if (globalPos >= runEnd) break;
        }
        if (idx < to) {
          throw new Error(`EOF before finishing tensor ${tensors[idx].key}`);
        }
        await reader.cancel().catch(() => {});
      };

      let i = 0;
      while (i < tensors.length) {
        const kappa = known.get(i);
        if (kappa !== undefined) {
          const tensor = tensors[i];
          await finalizeTensor(tensor, kappa, null);
          skippedBytes += tensor.data_offsets[1] - tensor.data_offsets[0];
          i++;
          continue;
        }
        let j = i;
        while (j < tensors.length && !known.has(j)) j++;
        await streamRun(i, j);
        i = j;
      }

      if (manifest.etag) {
        await saveShardPrior(provDir, manifest.etag, freshPrior);
      }
      emitProgressFn(
        skippedBytes > 0
          ? `Finished ${manifest.rfilename}: skipped ${(skippedBytes / 1024 / 1024).toFixed(1)}MB already known under the content pin.`
          : `Finished streaming ${manifest.rfilename}.`,
      );
    }

    // Record κ-provenance + model metadata (the session's context window)
    // under the model directory.
    if (localName) {
      const modelsDir = await root.getDirectoryHandle("models", { create: true });
      const localDir = await modelsDir.getDirectoryHandle(localName, { create: true });
      await writeEssential(tensorsDir, cachedKappas, emitProgressFn, async () => {
        const srcHandle = await localDir.getFileHandle("kappa-sources.json", { create: true });
        const writable = await srcHandle.createWritable();
        await writable.write(JSON.stringify(kappaSources));
        await writable.close();
      });
      await writeEssential(tensorsDir, cachedKappas, emitProgressFn, async () => {
        const metaHandle = await localDir.getFileHandle("model-meta.json", { create: true });
        const metaWritable = await metaHandle.createWritable();
        await metaWritable.write(JSON.stringify({ contextLength }));
        await metaWritable.close();
      });
      // The streamed manifest (name → κ/shape/dtype): chat recompiles stage
      // windows from it, so the generation window can follow the sequence.
      await writeEssential(tensorsDir, cachedKappas, emitProgressFn, async () => {
        const manifestHandle = await localDir.getFileHandle("manifest.json", { create: true });
        const manifestWritable = await manifestHandle.createWritable();
        await manifestWritable.write(
          JSON.stringify({ keys: allKeys, kappas: allKappas, shapes: allShapes, dtypes: allDtypes }),
        );
        await manifestWritable.close();
      });
    }

    // Mechanical: the graph was validated in preflight; this binds the
    // streamed κs and emits the k-form archive(s). Models beyond the
    // execution window compile as stage archives (windowed execution over k).
    if (stageCount && stageCount > 1 && layersPerStage && localName) {
      // PARAMETRIC tier selection (never a user knob): a model is AUTOMATICALLY
      // compiled at its optimal tier. "auto" (the default) resolves to the tier
      // that best serves THIS model's footprint under the 4 GiB address space —
      // int8 for quality when it fits resident, int4 ONLY to keep a larger model
      // resident/interactive when int8 cannot. An explicit "int8"/"int4" is a
      // diagnostic override. The decision is narrated, never silent.
      if (quantize === undefined || quantize === "auto") {
        const params = allShapes.reduce((sum, s) => {
          const dims = JSON.parse(s) as number[];
          return sum + dims.reduce((p, d) => p * d, 1);
        }, 0);
        quantize = await optimalQuantTier(params);
        const b = (params / 1e9).toFixed(2);
        emitProgressFn(
          quantize === "int4"
            ? `Parametric tier: ${b}B params exceed int8-resident under 4 GiB — compiling int4 ` +
              `to keep it resident/interactive (reduced quality; a size-fit choice).`
            : `Parametric tier: ${b}B params fit resident at int8 — compiling int8 (full quality).`,
        );
      }
      const quantEntries: QuantEntry[] = [];
      // The resolved tier is recorded per entry so the binder declares the
      // matching weight dtype (int8 = one byte/code, int4 = packed nibbles).
      if (quantize === "int8" || quantize === "int4") {
        const eligible = new Set(
          await quantizableWeights(
            configText,
            allKeys,
            allKappas,
            allShapes,
            allDtypes,
            contextLength,
            layersPerStage,
          ),
        );
        emitProgressFn(
          `Quantized tier (${quantize}): deriving artifacts for ${eligible.size} projection weight(s)...`,
        );
        let wideTotal = 0;
        let quantTotal = 0;
        const artifactKappas: string[] = [];
        for (let i = 0; i < allKappas.length; i++) {
          const wideKappa = allKappas[i];
          if (!eligible.has(wideKappa) || quantEntries.some((e) => e.wide === wideKappa)) continue;
          const dims = JSON.parse(allShapes[i]) as number[];
          const wide = await readTensorBytes(tensorsDir, wideKappa, kappaSources);
          // The wide bytes are in hand: the blob is gas-phase NOW. Evaporate
          // it BEFORE the artifact writes — under a quota the wide download
          // already saturated, this keeps occupancy monotonically shrinking
          // for cached tensors. At real 1.5B scale the reverse order
          // evaporated freshly written artifacts while consumed wide blobs
          // held the quota.
          await tensorsDir.removeEntry(`${wideKappa}.bin`).catch(() => {});
          const wideIdx = cachedKappas.indexOf(wideKappa);
          if (wideIdx >= 0) cachedKappas.splice(wideIdx, 1);
          const artifact = await deriveQuantizedArtifact(wide, allDtypes[i], dims[0], dims[1], quantize);
          const artifactKappa = await computeKappa(artifact);
          // Artifacts are the crystallizing product of this download — never
          // on its own evictable list; remaining wide blobs evaporate first.
          // A refused write SATURATES the tier, never the journey (the
          // memory-guard law): this and every remaining projection stay on
          // the wide tier through recorded provenance, stated in narration.
          try {
            await writeEssential(tensorsDir, cachedKappas, emitProgressFn, async () => {
              const handle = await tensorsDir.getFileHandle(`${artifactKappa}.bin`, { create: true });
              const writable = await handle.createWritable();
              await writable.write(artifact as unknown as ArrayBufferView<ArrayBuffer>);
              await writable.close();
            });
          } catch {
            emitProgressFn(
              `Quantized tier saturated by the storage quota: ${quantEntries.length} projection(s) ` +
                `crystallized; the remaining weights stay on the wide tier via recorded provenance.`,
            );
            break;
          }
          artifactKappas.push(artifactKappa);
          wideTotal += wide.length;
          quantTotal += artifact.length;
          quantEntries.push({ wide: wideKappa, artifact: artifactKappa, out: dims[0], in: dims[1], tier: quantize });
        }
        emitProgressFn(
          `Quantized tier: ${quantEntries.length} artifact(s), ${(quantTotal / 1024 / 1024).toFixed(1)}MB ` +
            `derived; ${(wideTotal / 1024 / 1024).toFixed(1)}MB of wide forms gas-phase.`,
        );
        // Artifacts join the evictable set BEHIND every wide blob (σ-order:
        // re-derivable, but dearer than provenance-recoverable wide) so a
        // later structure write never dead-ends while gas remains.
        cachedKappas.unshift(...artifactKappas);

        // The LM head joins the int8 tier too (chunked head): each vocab-row
        // chunk gets its OWN per-chunk artifact, derived from a byte range of
        // the wide head/embedding tensor. Unlike a projection, the wide κ is
        // NOT evaporated — a tied head shares the embedding table's κ, which the
        // embedding Gather still needs, and re-derivation reads the same range.
        // A large bf16 head is otherwise a matmul whose whole-panel F32 image
        // thrashes residency; the int8 chunk is a dequant-fused matmul with no
        // F32 panel. Empty (a no-op) where the head is a single chunk.
        const headTargets = await headQuantChunks(
          configText,
          allKeys,
          allKappas,
          allShapes,
          allDtypes,
          contextLength,
          layersPerStage,
        );
        if (headTargets.length) {
          emitProgressFn(
            `Quantized tier (${quantize}): deriving ${headTargets.length} LM-head chunk artifact(s)...`,
          );
          const headArtifactKappas: string[] = [];
          const wideCache = new Map<string, Uint8Array>();
          for (const t of headTargets) {
            const idx = allKappas.indexOf(t.kappa);
            if (idx < 0) throw new Error(`head-chunk κ \`${t.kappa}\` is outside the manifest`);
            let wide = wideCache.get(t.kappa);
            if (!wide) {
              wide = await readTensorBytes(tensorsDir, t.kappa, kappaSources);
              wideCache.set(t.kappa, wide);
            }
            const slice = wide.slice(t.offset, t.offset + t.len);
            const artifact = await deriveQuantizedArtifact(slice, allDtypes[idx], t.out, t.in, quantize);
            const artifactKappa = await computeKappa(artifact);
            try {
              await writeEssential(tensorsDir, cachedKappas, emitProgressFn, async () => {
                const handle = await tensorsDir.getFileHandle(`${artifactKappa}.bin`, {
                  create: true,
                });
                const writable = await handle.createWritable();
                await writable.write(artifact as unknown as ArrayBufferView<ArrayBuffer>);
                await writable.close();
              });
            } catch {
              emitProgressFn(
                `Quantized tier saturated by the storage quota while crystallizing the LM head; ` +
                  `the remaining head chunks stay on the wide tier via recorded provenance.`,
              );
              break;
            }
            headArtifactKappas.push(artifactKappa);
            quantEntries.push({
              wide: t.kappa,
              artifact: artifactKappa,
              out: t.out,
              in: t.in,
              offset: t.offset,
              len: t.len,
              tier: quantize,
            });
          }
          cachedKappas.unshift(...headArtifactKappas);
          emitProgressFn(
            `Quantized tier: LM head crystallized into ${headArtifactKappas.length} ${quantize} chunk(s) ` +
              `(dequant-fused matmul — no whole-panel F32 image).`,
          );
        }
      }

      emitProgressFn(
        `Binding streamed κs into ${stageCount} stage archives (windowed execution over k)...`,
      );
      const stages = quantEntries.length
        ? await compileSafetensorsStagedQuantized(
            configText,
            allKeys,
            allKappas,
            allShapes,
            allDtypes,
            contextLength,
            layersPerStage,
            quantEntries,
          )
        : await compileSafetensorsStaged(
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
        await writeEssential(tensorsDir, cachedKappas, emitProgressFn, async () => {
          const handle = await stagesDir.getFileHandle(`${i}.holo`, { create: true });
          const writable = await handle.createWritable();
          await writable.write(stages[i] as unknown as ArrayBufferView<ArrayBuffer>);
          await writable.close();
        });
      }
      await writeEssential(tensorsDir, cachedKappas, emitProgressFn, async () => {
        const metaHandle = await localDir.getFileHandle("stages.json", { create: true });
        const writable = await metaHandle.createWritable();
        await writable.write(
          JSON.stringify({ stageCount: stages.length, layersPerStage, contextLength, quant: quantEntries }),
        );
        await writable.close();
      });
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
