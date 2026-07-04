import { compileSafetensorsStreamed, KappaHasher, compileOnnxWithData } from "./holo";

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

async function handleSafetensorsDownload({
  modelId,
  configText,
  files
}: {
  modelId: string,
  configText: string,
  files: any[]
}) {
  try {
    const root = await navigator.storage.getDirectory();
    const modelsDir = await root.getDirectoryHandle("models", { create: true });
    // In UOR, we store chunks by kappa in a global pool or per model. Let's do per model.
    const localName = modelId.split("/").pop() || modelId;
    const modelDir = await modelsDir.getDirectoryHandle(localName, { create: true });

    const allKeys: string[] = [];
    const allKappas: string[] = [];
    const allShapes: string[] = [];
    const allDtypes: string[] = [];

    for (const file of files) {
      const url = `https://huggingface.co/${modelId}/resolve/main/${file.rfilename}`;
      emitProgress(`Streaming ${file.rfilename}...`);
      
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
              
              // Write the completed tensor buffer to OPFS as <kappa>.bin
              const binHandle = await modelDir.getFileHandle(`${kappa}.bin`, { create: true });
              // Use SyncAccessHandle for blazing fast writes in worker
              const accessHandle = await (binHandle as any).createSyncAccessHandle();
              accessHandle.write(currentTensorBuffer);
              accessHandle.flush();
              accessHandle.close();

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
            emitProgress(`Streaming ${file.rfilename}: ${percent}% (${(downloadedBytes / 1024 / 1024).toFixed(1)}MB)`);
          }
        }
      }
      
      if (currentTensorIdx < tensors.length) {
        throw new Error(`EOF before finishing tensor ${tensors[currentTensorIdx].key}`);
      }
      emitProgress(`Finished streaming ${file.rfilename}.`);
    }

    emitProgress("Compiling streamed tensors...");
    const holoBytes = await compileSafetensorsStreamed(
      configText,
      allKeys,
      allKappas,
      allShapes,
      allDtypes
    );

    emitDone(holoBytes);
  } catch (err: any) {
    emitError(err.toString());
  }
}

async function handleOnnxDownload({
  modelId,
  files
}: {
  modelId: string,
  files: any[]
}) {
  try {
    const root = await navigator.storage.getDirectory();
    const modelsDir = await root.getDirectoryHandle("models", { create: true });
    const localName = modelId.split("/").pop() || modelId;
    const modelDir = await modelsDir.getDirectoryHandle(localName, { create: true });

    for (const file of files) {
      const url = `https://huggingface.co/${modelId}/resolve/main/${file.rfilename}`;
      emitProgress(`Downloading ${file.rfilename}...`);
      
      const response = await fetch(url);
      if (!response.ok || !response.body) throw new Error(`Failed to fetch ${file.rfilename}`);

      const handle = await modelDir.getFileHandle(file.rfilename, { create: true });
      const accessHandle = await (handle as any).createSyncAccessHandle();
      
      const reader = response.body.getReader();
      let downloadedBytes = 0;
      const totalBytes = Number(response.headers.get("content-length")) || 0;
      let lastEmit = 0;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        accessHandle.write(value);
        downloadedBytes += value.length;
        
        const now = Date.now();
        if (now - lastEmit > 500) {
          lastEmit = now;
          if (totalBytes > 0) {
            const percent = Math.round((downloadedBytes / totalBytes) * 100);
            emitProgress(`Downloading ${file.rfilename}: ${percent}% (${(downloadedBytes / 1024 / 1024).toFixed(1)}MB)`);
          } else {
            emitProgress(`Downloading ${file.rfilename}: ${(downloadedBytes / 1024 / 1024).toFixed(1)}MB`);
          }
        }
      }
      
      accessHandle.flush();
      accessHandle.close();
      emitProgress(`Finished downloading ${file.rfilename}.`);
    }

    emitProgress("Compiling ONNX model...");
    
    // Find the main ONNX file from the payload
    const mainOnnxFile = files.find(f => f.rfilename.endsWith('.onnx'));
    if (!mainOnnxFile) throw new Error("ONNX file not found in download list");
    const onnxFileName = mainOnnxFile.rfilename.split('/').pop()!;
    
    const onnxHandle = await modelDir.getFileHandle(onnxFileName);
    const onnxFile = await onnxHandle.getFile();
    
    if (onnxFile.size > 2 * 1024 * 1024 * 1024) {
      throw new Error(`Model is too large (${(onnxFile.size / 1024 / 1024 / 1024).toFixed(1)}GB) to compile in the browser due to WebAssembly 32-bit memory limits (max 2-4GB). Please use the hologram-ai desktop or CLI for models larger than 2GB, or use a smaller/quantized model.`);
    }
    
    const onnxBytes = new Uint8Array(await onnxFile.arrayBuffer());

    let dataBytes = new Uint8Array();
    try {
      const dataHandle = await modelDir.getFileHandle(onnxFileName + ".data");
      const dataFile = await dataHandle.getFile();
      dataBytes = new Uint8Array(await dataFile.arrayBuffer());
    } catch {
      try {
        const dataHandle2 = await modelDir.getFileHandle(onnxFileName + "_data");
        const dataFile2 = await dataHandle2.getFile();
        dataBytes = new Uint8Array(await dataFile2.arrayBuffer());
      } catch {
        // no data file
      }
    }

    let holoBytes: Uint8Array;
    if (dataBytes.length > 0) {
      holoBytes = await compileOnnxWithData(onnxBytes, dataBytes);
    } else {
      const { compile } = await import("./holo");
      holoBytes = await compile(onnxBytes);
    }
    emitDone(holoBytes);
  } catch (err: any) {
    emitError(err.toString());
  }
}
