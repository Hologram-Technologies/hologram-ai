import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { fetchViaExtension } from './ipc';

describe('fetchViaExtension', () => {
  let mockPort: any;
  let onMessageListeners: any[] = [];
  let onDisconnectListeners: any[] = [];
  
  beforeEach(() => {
    onMessageListeners = [];
    onDisconnectListeners = [];
    mockPort = {
      onMessage: {
        addListener: (fn: any) => onMessageListeners.push(fn),
      },
      onDisconnect: {
        addListener: (fn: any) => onDisconnectListeners.push(fn),
      },
      postMessage: vi.fn(),
    };
    
    if (!(window as any).chrome) (window as any).chrome = {};
    (window as any).chrome.runtime = {
      connect: vi.fn().mockReturnValue(mockPort),
    } as any;
    
    document.documentElement.setAttribute("data-holospaces-egress", "mock-extension-id");
  });
  
  afterEach(() => {
    if ((window as any).chrome) {
      delete (window as any).chrome.runtime;
    }
    document.documentElement.removeAttribute("data-holospaces-egress");
    vi.restoreAllMocks();
  });
  
  it('should reject if chrome.runtime is missing', async () => {
    if ((window as any).chrome) {
      delete (window as any).chrome.runtime;
    }
    
    await expect(fetchViaExtension('https://example.com'))
      .rejects.toThrow(/Chrome extension not available/);
  });
  
  it('should connect to the correct extension ID and port', () => {
    // We don't await because it promises until end, just check the sync parts
    fetchViaExtension('https://example.com').catch(() => {});
    
    // @ts-ignore
    expect(globalThis.chrome.runtime.connect).toHaveBeenCalledWith(
      "mock-extension-id",
      { name: "holospaces-content" }
    );
  });
  
  it('should post a fetch message on the port', () => {
    fetchViaExtension('https://example.com').catch(() => {});
    
    expect(mockPort.postMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        type: 'fetch',
        url: 'https://example.com',
        method: 'GET'
      })
    );
  });
  
  it('should resolve with concatenated chunks when extension sends end', async () => {
    const promise = fetchViaExtension('https://example.com');
    
    // Grab the auto-generated ID from the postMessage call
    const postedMsg = mockPort.postMessage.mock.calls[0][0];
    const id = postedMsg.id;
    
    // Simulate extension response
    const listener = onMessageListeners[0];
    
    // 1. head
    listener({ id, type: 'head', status: 200, headers: {}, totalBytes: 8 });
    
    // 2. chunks
    listener({ id, type: 'chunk', bytes: [104, 101, 108, 108] }); // "hell"
    listener({ id, type: 'chunk', bytes: [111, 32, 119, 111] }); // "o wo"
    
    // 3. end
    listener({ id, type: 'end' });
    
    const result = await promise;
    
    expect(result).toBeInstanceOf(Uint8Array);
    expect(result.length).toBe(8);
    expect(Array.from(result)).toEqual([104, 101, 108, 108, 111, 32, 119, 111]);
  });
  
  it('should reject when HTTP status >= 400 is returned', async () => {
    const promise = fetchViaExtension('https://example.com');
    
    const postedMsg = mockPort.postMessage.mock.calls[0][0];
    const id = postedMsg.id;
    
    const listener = onMessageListeners[0];
    listener({ id, type: 'head', status: 404 });
    
    await expect(promise).rejects.toThrow(/HTTP 404/);
  });
  
  it('should reject when extension emits an error event', async () => {
    const promise = fetchViaExtension('https://example.com');
    
    const postedMsg = mockPort.postMessage.mock.calls[0][0];
    const id = postedMsg.id;
    
    const listener = onMessageListeners[0];
    listener({ id, type: 'error', error: 'Network failure' });
    
    await expect(promise).rejects.toThrow(/Network failure/);
  });
  
  it('should reject when extension port disconnects with lastError', async () => {
    const promise = fetchViaExtension('https://example.com');
    
    // Simulate chrome.runtime.lastError
    (window as any).chrome.runtime.lastError = { message: "Could not establish connection. Receiving end does not exist." };
    
    const listener = onDisconnectListeners[0];
    listener();
    
    await expect(promise).rejects.toThrow(/Could not establish connection. Receiving end does not exist./);
    
    delete (window as any).chrome.runtime.lastError;
  });
  
  it('should reject when extension port disconnects unexpectedly without lastError', async () => {
    const promise = fetchViaExtension('https://example.com');
    
    const listener = onDisconnectListeners[0];
    listener();
    
    await expect(promise).rejects.toThrow(/Chrome extension disconnected unexpectedly/);
  });
});
