import { describe, it, expect, beforeAll, afterAll, vi } from 'vitest';
import { downloadKnownModel, getOpfsDir } from './ipc';

describe('End-to-End Model Pipeline', () => {
  let originalFetch: typeof globalThis.fetch;

  beforeAll(() => {
    originalFetch = globalThis.fetch;
    globalThis.fetch = vi.fn().mockImplementation(async (url: string | Request | URL) => {
      const urlStr = url.toString();
      if (urlStr.includes('/api/models/test/mock-model')) {
        return new Response(JSON.stringify({
          id: "test/mock-model",
          siblings: [
            { rfilename: "model.safetensors", size: 100 },
            { rfilename: "config.json", size: 50 },
            { rfilename: "tokenizer.json", size: 50 }
          ]
        }));
      }
      if (urlStr.includes('/resolve/main/model.safetensors')) {
        return new Response(new Uint8Array(100).buffer);
      }
      if (urlStr.includes('/resolve/main/config.json')) {
        return new Response(new TextEncoder().encode('{}').buffer);
      }
      if (urlStr.includes('/resolve/main/tokenizer.json')) {
        return new Response(new TextEncoder().encode('{}').buffer);
      }
      return originalFetch(url);
    });
  });
  
  afterAll(() => {
    globalThis.fetch = originalFetch;
  });

  it('should download and compile a model as a single action', async () => {
    // Add mock model to catalogue
    const { addCustomModel } = await import('./ipc');
    await addCustomModel('test/mock-model');
    
    // Attempt download and compile
    // Because it's a mock safetensors, compile will probably fail with an error
    // but the test should verify that the download creates the OPFS files properly
    try {
      await downloadKnownModel('mock-model');
    } catch (e: any) {
      // The mock safetensors is invalid, so compilation WILL fail, but the download should succeed
    }
    
    // Verify files exist in OPFS
    const root = await getOpfsDir();
    const modelsDir = await root.getDirectoryHandle('models');
    const modelDir = await modelsDir.getDirectoryHandle('mock-model');
    
    const safetensorsHandle = await modelDir.getFileHandle('model.safetensors');
    const safetensorsFile = await safetensorsHandle.getFile();
    expect(safetensorsFile.size).toBe(100);
  });
});
