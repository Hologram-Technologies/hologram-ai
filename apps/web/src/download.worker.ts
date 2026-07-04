import { compileSafetensorsStreamed, KappaHasher, ensureReady } from "./holo";

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

function emitError(err: string) {
  self.postMessage({ type: "error", error: err });
}

export async function handleSafetensorsDownload(
  {
    modelId,
    configText,
    files,
    contextLength,
    hfBase
  }: {
    modelId: string,
    configText: string,
    files: any[],
    contextLength?: number,
    hfBase?: string
  },
  emitProgressFn = emitProgress,
  emitDoneFn = emitDone,
  emitErrorFn = emitError
) {
  try {
    await ensureReady();
    const root = await navigator.storage.getDirectory();
    const tensorsDir = await root.getDirectoryHandle("tensors", { create: true });

    const allKeys: string[] = [];
    const allKappas: string[] = [];
    const allShapes: string[] = [];
    const allDtypes: string[] = [];

    for (const file of files) {
      const url = `${hfBase ?? "https://huggingface.co"}/${modelId}/resolve/main/${file.rfilename}`;
      emitProgressFn(`Streaming ${file.rfilename}...`);
      
      const response = await fetch(url);
      if (!response.ok || !response.body) throw new Error(`Failed to fetch ${file.rfilename}`);

      const reader = response.body.getReader();
      let downloadedBytes = 0;
      const totalBytes = Number(response.headers.get("content-length")) || 0;
      let lastEmit = 0;

      // Read header length
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
        tensors.push({ key, meta });
      }
      tensors.sort((a, b) => (a.meta as any).data_offsets[0] - (b.meta as any).data_offsets[0]);

      let currentTensorIdx = 0;
      let currentHasher = new KappaHasher();
      const baseOffset = 8 + headerLen;

      // Buffer for current tensor
      let currentTensorBuffer: Uint8Array | null = null;
      let currentTensorOffset = 0;

      async function processBytes(globalOffset: number, bytes: Uint8Array) {
        let localOffset = 0;
        while (localOffset < bytes.length && currentTensorIdx < tensors.length) {
          const tensor = tensors[currentTensorIdx];
          const tensorStart = baseOffset + tensor.meta.data_offsets[0];
          const tensorEnd = baseOffset + tensor.meta.data_offsets[1];
          const tensorSize = tensorEnd - tensorStart;
          const currentGlobal = globalOffset + localOffset;

          if (currentGlobal < tensorStart) {
            const toSkip = Math.min(tensorStart - currentGlobal, bytes.length - localOffset);
            localOffset += toSkip;
          } else if (currentGlobal < tensorEnd) {
            const toFeed = Math.min(tensorEnd - currentGlobal, bytes.length - localOffset);
            const slice = bytes.slice(localOffset, localOffset + toFeed);
            currentHasher.update(slice);
            
            // Allocate buffer if needed
            if (!currentTensorBuffer) {
              currentTensorBuffer = new Uint8Array(tensorSize);
              currentTensorOffset = 0;
            }
            currentTensorBuffer.set(slice, currentTensorOffset);
            currentTensorOffset += slice.length;

            localOffset += toFeed;

            if (currentGlobal + toFeed === tensorEnd) {
              const kappa = currentHasher.finalize();
              const binHandle = await tensorsDir.getFileHandle(`${kappa}.bin`, { create: true });
              if ('createSyncAccessHandle' in binHandle) {
                const accessHandle = await (binHandle as any).createSyncAccessHandle();
                accessHandle.write(currentTensorBuffer);
                accessHandle.flush();
                accessHandle.close();
              } else {
                const writable = await (binHandle as any).createWritable();
                await writable.write(currentTensorBuffer);
                await writable.close();
              }

              allKeys.push(tensor.key);
              allKappas.push(kappa);
              allShapes.push(JSON.stringify(tensor.meta.shape));
              allDtypes.push(tensor.meta.dtype);
              
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

      let processedOffset = 0;
      for (const chunk of chunks) {
        await processBytes(processedOffset, chunk);
        processedOffset += chunk.length;
      }
      chunks = []; // free memory

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
            emitProgressFn(`Streaming ${file.rfilename}: ${percent}% (${(downloadedBytes / 1024 / 1024).toFixed(1)}MB)`);
          }
        }
      }
      
      if (currentTensorIdx < tensors.length) {
        throw new Error(`EOF before finishing tensor ${tensors[currentTensorIdx].key}`);
      }
      emitProgressFn(`Finished streaming ${file.rfilename}.`);
    }

    emitProgressFn("Compiling streamed tensors...");
    const holoBytes = await compileSafetensorsStreamed(
      configText,
      allKeys,
      allKappas,
      allShapes,
      allDtypes,
      contextLength
    );

    emitDoneFn(holoBytes);
  } catch (err: any) {
    emitErrorFn(err.toString());
  }
}

// ONNX models cannot be streamed over k without a complex streaming protobuf parser in JS.
// Per architectural constraints, the implementation MUST operate over the k-representation
// and stream in every aspect to avoid contrived 32-bit WebAssembly limits.
// Therefore, ONNX downloads are not supported in the web IDE.
async function handleOnnxDownload(
  _args: any,
  _emitProgressFn = emitProgress,
  _emitDoneFn = emitDone,
  emitErrorFn = emitError
) {
  emitErrorFn("ONNX models are not supported in the web IDE because they cannot be streamed over the holospaces/k-representation. Please use safetensors.");
}
