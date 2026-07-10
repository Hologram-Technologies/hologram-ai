// Cheap, browser-free witness of the pool's core mechanic (ADR-0018): multiple
// wasm instances over ONE shared linear memory, each registering with the
// substrate pool via `hologram_worker_run`, and the count visible across
// instances through the shared `WORKERS` atomic. Uses Node worker_threads (which
// give real OS threads + SharedArrayBuffer), so it runs in CI without a browser.
//
// Fails-without: if the memory is not actually shared, or `thread_stack_size`
// does not distinguish worker-init from main-init (data-init corruption), or the
// pool export is missing, registration never reaches N and this fails. Needs the
// threaded build (`pnpm wasm:threads`).
import { Worker } from "node:worker_threads";
import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WASM = path.join(HERE, "../src/wasm-threads/hologram_ai_wasm_bg.wasm");
const GLUE = path.join(HERE, "../src/wasm-threads/hologram_ai_wasm.js");
const N = 3;

if (!existsSync(WASM)) {
  console.error(`✗ threaded wasm not built: ${WASM}\n  run: pnpm wasm:threads`);
  process.exit(1);
}

const glue = await import(GLUE);
const mod = new WebAssembly.Module(readFileSync(WASM));
const memory = new WebAssembly.Memory({ initial: 27, maximum: 65536, shared: true });
const ex = glue.initSync({ module: mod, memory });

if (ex.hologram_pool_workers() !== 0) {
  console.error(`✗ expected 0 workers before spawn, got ${ex.hologram_pool_workers()}`);
  process.exit(1);
}

const workers = [];
for (let id = 0; id < N; id++) {
  const w = new Worker(new URL("./pool-registration.worker.mjs", import.meta.url), {
    workerData: { module: mod, memory, id },
  });
  w.on("error", (e) => {
    console.error(`✗ pool worker ${id} errored: ${e.message}`);
    process.exit(2);
  });
  workers.push(w);
}

// The registrations happen on real OS threads; poll the shared counter.
const spin = new Int32Array(new SharedArrayBuffer(4));
const t0 = Date.now();
while (ex.hologram_pool_workers() < N && Date.now() - t0 < 10_000) Atomics.wait(spin, 0, 0, 25);

const got = ex.hologram_pool_workers();
ex.hologram_pool_shutdown();
await Promise.all(workers.map((w) => w.terminate()));

if (got === N) {
  console.log(`✓ ${N} pool workers registered over one shared memory (hologram_pool_workers === ${got})`);
  console.log("POOL REGISTRATION WITNESS: PASS");
  process.exit(0);
} else {
  console.error(`✗ only ${got}/${N} workers registered — shared memory / registration broken`);
  console.log("POOL REGISTRATION WITNESS: FAILED");
  process.exit(1);
}
