// Pool worker — a participant in the substrate's embedder worker pool (ADR-0018).
//
// It instantiates the SAME threaded wasm module over the SHARED linear memory
// created by the execute worker, registers with the pool by calling the exported
// `hologram_worker_run(id)` (which blocks in the fork-join loop until
// `hologram_pool_shutdown`), and computes disjoint GEMV output-column tiles. It
// owns no session and touches no JS state — pure compute over shared memory, so
// the decode output is bit-identical to the single-threaded path (the substrate
// guarantees this: `parallel_gemv_matches_serial_bitwise`).
import { initSync } from "./wasm-threads/hologram_ai_wasm.js";

interface PoolMsg {
  module: WebAssembly.Module;
  memory: WebAssembly.Memory;
  id: number;
  stackSize: number;
}

self.onmessage = (e: MessageEvent<PoolMsg>) => {
  const { module, memory, id, stackSize } = e.data;
  // A non-zero, 64 KiB-aligned `thread_stack_size` marks this as a spawned
  // thread: wasm-bindgen gives it a fresh stack + TLS and does NOT re-run the
  // module's data init (that ran once on the execute/"main" instance). The
  // execute worker gates its first decode on `hologram_pool_workers()` reaching
  // the expected count, because late registration traps in the substrate.
  let exports: { hologram_worker_run: (id: number) => void };
  try {
    exports = initSync({ module, memory, thread_stack_size: stackSize }) as unknown as {
      hologram_worker_run: (id: number) => void;
    };
  } catch (err) {
    // Surface the failure so the main thread aborts the pool (else the execute
    // side would only notice via the readiness timeout). Also rethrow to trip
    // `onerror`, in case the message channel is not observed.
    self.postMessage({ error: String(err), id });
    throw err;
  }
  exports.hologram_worker_run(id); // registers (fetch_add), then blocks until shutdown
};
