// Node worker_threads participant for the pool-registration witness. Mirrors
// the browser pool worker: instantiate the SAME threaded module over the SHARED
// memory, then register by calling `hologram_worker_run` (blocks until shutdown).
import { workerData } from "node:worker_threads";
import { fileURLToPath } from "node:url";
import path from "node:path";

const glueUrl = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  "../src/wasm-threads/hologram_ai_wasm.js",
);
const glue = await import(glueUrl);
const { module, memory, id } = workerData;
const ex = glue.initSync({ module, memory, thread_stack_size: 2 * 1024 * 1024 });
ex.hologram_worker_run(id); // blocks in the pool loop until hologram_pool_shutdown
