# ADR-0018: Multi-Threaded WASM Decode (Embedder Worker Pool)

**Status:** Accepted (correctness gates green; benchmarked 2.5–4× on realistic
chat-scale GEMVs — see Verification results)
**Date:** 2026-07-10
**Supersedes:** ADR-0017 §"Single-threaded" / §6 ("No headers required") — the
parallel-off, no-COOP/COEP stance. Everything else in ADR-0017 stands.
**Relates to:** CONFORMANCE class **NS** (multi-target runtime core), **PV**
(performance).

---

## Context

ADR-0017 (2026-05-27) chose a **single-threaded** browser build: "rayon cannot
spawn threads on `wasm32-unknown-unknown`", "no headers required (single-threaded,
no `SharedArrayBuffer`)." That decision was correct **for its date** — and it
predates the substrate capability that reopens it.

Substrate **v0.8.1** ships `hologram-backend/src/cpu/wasm_pool.rs` (plan 077 item
5): an **embedder-provided web-worker pool** that fans the decode GEMV across N
workers sharing one linear memory, with a **bit-identical** guarantee to the
serial kernel (`parallel_gemv_matches_serial_bitwise`, `simd.rs:6877`: asserts
`p.to_bits() == sv.to_bits()`). Its module doc states the design target
explicitly:

> wasm32 has no `std::thread`: parallelism comes from the **embedder**
> (hologram-ai serves its own COOP/COEP headers) instantiating this module on N
> web workers that share one linear memory (a `+atomics,+bulk-memory`
> shared-memory build), each calling the exported `hologram_worker_run` once.

hologram-ai **is** that embedder, named in the substrate's own contract. The
deployed browser decode is single-threaded-bound (~1.6 tok/s warm on SmolLM2-135M
int8 W8A8, measured by `bdd/probe-deployed-live.mjs`). The pool is the lever to
scale that across the user's cores — byte-for-byte identical output.

### The load-bearing constraint (why this is a real project, not a flag)

The pool blocks idle workers on a futex. That futex has two implementations,
selected by the backend's `std` feature (`wasm_pool.rs:38-53`, `197-224`):

- **`#[cfg(not(feature = "std"))]`** — imports `hologram_host_wait32` /
  `hologram_host_notify` (JS `Atomics.wait` / `Atomics.notify`). Real blocking:
  idle workers sleep.
- **`#[cfg(feature = "std")]`** — `wait_u32` spins + `std::thread::yield_now()`,
  `notify_all` is a no-op. Correct only under **preemptive OS threads**
  (`wasm32-wasip1-threads`/wasmtime, the substrate's test lane). In a browser it
  **busy-spins** — N cores pinned at 100% whenever idle. And the pool traps on
  late registration, so it cannot be torn down/rebuilt per turn to dodge the
  spin.

Today our wasm build compiles `hologram-backend` **with** `std` (the workspace
pins the host-shell crates at their `std`-on defaults). So a naive
`+atomics` build would get the busy-spin path. **The browser needs the backend
compiled `no_std`** to reach the `Atomics.wait` path — while `hologram-ai-wasm`
itself remains a `std` crate (std being *linked* ≠ the backend's `std` *feature*
being on; features are per-crate-compilation). This is the pivotal feasibility
question and is gated by a spike (below).

## Decision

Adopt the substrate embedder pool for browser decode, in four layers, each with
its own fails-without witness. The **single-threaded `+simd128` build stays the
default fallback** (byte-identical to today) and is selected automatically when
the page is not cross-origin-isolated.

### 1. A shared-memory, no_std-backend threaded wasm build (behind a feature)

- Feature surgery on **our** crates only (substrate is read-only): the host-shell
  substrate deps (`exec`/`archive`/`backend`/`compiler`/`host`) become
  `default-features = false` in `hologram-ai`, `hologram-ai-common`, and any
  other wasm-graph crate; a **`std`** feature (default-on, so native/CLI is
  unchanged) re-enables their `std`; a **`wasm-threads`** feature forwards to
  `hologram-backend/wasm-threads`.
- The threaded browser build enables `wasm-threads`, omits `std` → the no_std
  `Atomics` futex path. Toolchain: **nightly** + `-Zbuild-std=std,panic_abort`,
  `RUSTFLAGS=-Ctarget-feature=+simd128,+atomics,+bulk-memory,+mutable-globals`,
  linker `--shared-memory --import-memory --max-memory=<4GiB>`.
- The native lib/CLI/benches keep stable 1.97.0 + `std` + `parallel`. Only the
  wasm build step uses nightly — pinned identically in the deploy action and CI
  (gate == CI; see [[dark-gates]]).

### 2. A JS worker-pool embedder

- Create one `WebAssembly.Memory({ initial, maximum, shared: true })`; instantiate
  the module on **N+1 web workers** over it. N pool workers each call
  `hologram_worker_run(id)` **once, before the first execute** (late registration
  traps). Provide the two host imports as `Atomics.wait`/`Atomics.notify`.
- **Decode runs on a worker, not the main thread** (the join blocks on
  `Atomics.wait`, disallowed on the main thread). We already run decode in
  `generate.worker.ts`; that worker becomes the execute participant, plus N pool
  workers.
- Teardown → `hologram_pool_shutdown()`. Feature-detect: if `crossOriginIsolated`
  is false or the threaded module is absent, fall back to the single-worker
  single-threaded module (today's path).

### 3. Cross-origin isolation via `COEP: credentialless` + a service-worker shim

- GitHub Pages sets no headers → register a `coi-serviceworker` that injects
  `Cross-Origin-Opener-Policy: same-origin` + `Cross-Origin-Embedder-Policy:
  credentialless`. **`credentialless`, not `require-corp`**, so the only external
  origin — `huggingface.co` and its LFS CDN redirect — loads without needing a
  `Cross-Origin-Resource-Policy` header it does not send. (`require-corp` would
  break both model download and the inference-time provenance re-fetch at
  `generate.worker.ts:359`.)
- Mirror the two headers in vite `server.headers` + `preview.headers` so dev and
  the BDD journey gain isolation and match production. `bdd/fixture-server.mjs`
  (cross-origin, no CORP) is the **in-repo canary**: if `credentialless` is wrong,
  the hermetic journey gate fails before deploy, not in production.

### 4. Single shared session; residency unchanged

The pool shares ONE linear memory and ONE decode session; **workers only compute
GEMV output-column tiles — they never own or touch the session**. So the execute
thread's `Rc<RefCell>` session is safe (never crosses threads), and the 4 GiB
ceiling is unchanged (one shared memory). The `unsafe impl Send for SendStore`
("wasm32 is single-threaded") claim is re-examined: the sink is invoked only on
the execute thread, so the claim narrows from "single-threaded" to "single
*execute* thread" — documented, not deleted.

## V&V (external validators; every claim fails-without)

- **Byte-identical decode (substrate + ours).** The substrate guarantees the
  kernel (`parallel_gemv_matches_serial_bitwise`). We add a hologram-ai-level
  witness under the `wasm32-wasip1-threads`/wasmtime std test-lane: the same
  decode, threaded vs serial, **bit-identical logits/tokens** (fails-without: the
  pool mis-partitions → drift). This is the gate that our integration wires the
  pool correctly.
- **Browser, Playwright (the external validator):**
  1. `crossOriginIsolated === true` on the isolated build (fails-without: no
     `SharedArrayBuffer`, pool cannot start).
  2. Fixture handshake stays **byte-identical** to `reference-transcript.json`
     under isolation + threading (fails-without: COEP broke the journey, or
     threaded decode drifted).
  3. Live HF download completes under `credentialless` (fails-without: Risk 2 —
     isolation blocks HF).
  4. `hologram_pool_workers() === N` at decode time (fails-without: pool never
     registered → silent single-thread).
  5. Byte-identity holds for a LARGER model too (`probe-threads-live.mjs` with
     `HAI_PROBE_HF=Qwen/Qwen2.5-1.5B-Instruct`), and the pool speedup is
     benchmarked separately (`scripts/bench-pool-scaling.sh`) — the **payoff,
     quantified, not asserted**.
- **Dark-gate guards** ([[dark-gates]]): the threaded wasm builds in CI with the
  *same* nightly pin the deploy uses; the single-threaded fallback stays green;
  the fixture canary makes the COEP failure reproducible in the gate.

## Consequences

**Positive.** Multi-core browser decode, bit-identical to serial; the NS class
extends from "runs in the browser" to "runs multi-core in the browser"; PV gains
scale with the user's cores.

**Costs / honest constraints.**
- A **nightly** pin for the wasm build only (native stays stable 1.97.0). Two
  toolchains, both pinned, both == CI.
- A service worker: the first controlled navigation needs one reload to gain
  isolation (standard coi caveat) — handled by the shim's auto-reload.
- `credentialless` drops credentials on cross-origin subresources — fine (HF
  public files need none); private/gated models via token are out of scope.
- Safari supports only `require-corp`, not `credentialless` → Safari degrades to
  the single-threaded fallback (feature-detected, not broken).

## Risks & the gating spike

This ADR was **Proposed** pending one unproven assumption: that the host-shell
substrate stack (`compiler`/`exec`/`archive`/`backend`/`host`) builds **no_std**
for wasm32 (ADR-0017's spike built it `std`-on). The spike, in order of
decisiveness:

1. **Feature plumbing (fast, no build-std) — ✅ PASSED (2026-07-10).** After
   moving `default-features = false` + a default-on `std` feature into
   `[workspace.dependencies]` and `hologram-ai`/`hologram-ai-common`,
   `hologram-ai-wasm` compiles for `wasm32-unknown-unknown` with the host-shell
   stack no_std (1m07s). Verified by `cargo tree`: the wasm build resolves
   `hologram-backend` to `[cpu]` (std OFF → `Atomics.wait` futex),
   `--features wasm-threads` adds `wasm-threads`, and native is unchanged at
   `[cpu, parallel, std]`.
2. **Atomics/shared-memory (slow, build-std):** does the nightly build-std
   `+atomics,+bulk-memory` shared-memory build link, exporting
   `hologram_worker_run` and importing the `Atomics` futex? (in progress)
3. **Bit-identity + isolation + HF (Playwright):** the browser gates above.

All three passed — this is now **Accepted**.

## Verification results (2026-07-10)

Correctness — all green:
- **SPIKE 1/2:** the no_std host-shell stack builds for wasm32; the nightly
  build-std `+atomics` shared-memory build links; wasm-bindgen 0.2.126 emits the
  shared-memory thread glue (`initSync({module, memory})`). The pool's host futex
  is satisfied natively by `wasm_futex` (nightly `memory.atomic.wait32/notify`),
  so the artifact imports only `env.memory` — no JS futex shim.
- **Regression guard:** the (now no_std-backend) single-threaded build runs the
  hermetic journey byte-exact (28 scenarios / 170 steps).
- **Pool mechanic** (`bdd/probe-pool-registration.mjs`, Node worker_threads): N
  instances share one memory; `hologram_pool_workers()` reaches N.
- **Isolated browser** (`bdd/probe-threads-local.mjs`): `crossOriginIsolated`,
  the pool engages (7 workers), the fixture handshake stays byte-exact, and the
  cross-origin fixture (no CORP) loads — the credentialless-vs-HF canary.
- **Byte-identity** (`bdd/probe-threads-live.mjs`, real SmolLM2-135M above the
  pool floor): the threaded completion is **byte-for-byte identical** to the
  single-threaded one (`"The capital of France is Paris."`). The pool's fan-out
  preserves output end-to-end.

Speedup — the decode GEMV scales near-linearly for realistic chat models:
- Benchmark: `apps/web/scripts/bench-pool-scaling.sh` (the substrate's
  `wasm_threads_timing` under wasmtime, wasm32-wasip1-threads — std threads drive
  the SAME atomics fork-join queue the browser web workers do; representative for
  the compute-bound decode GEMV, and free of OPFS/download limits). It times the
  int8 M=1 decode GEMV SERIAL vs POOLED (3 workers + main = **4 participants**,
  matching the codespace's 4 physical cores) at chat-scale MLP shapes:

  | GEMV k×n | model scale | serial | pooled (4 part.) | speedup |
  |---|---|---|---|---|
  | 896×4864   | Qwen2.5-0.5B | 252 µs  | 99 µs   | **2.55×** |
  | 1536×8960  | Qwen2.5-1.5B | 755 µs  | 199 µs  | **3.79×** |
  | 3584×18944 | Qwen2.5-7B   | 4700 µs | 1371 µs | **3.43×** |
  | 3584×18944 (int4) | 7B int4 | 6147 µs | 1524 µs | **4.04×** |

  Near-linear on 4 participants (up to 4.04×), and it scales with the host's
  cores (a user's 8–16-core machine goes higher). The GB/s columns confirm it is
  memory-bandwidth-scaling, not a fixed-overhead artifact.
- The earlier **~1.0×** on SmolLM2-135M was NOT implementation overhead — it is
  the smallest model's floor: its GEMV tiles (hidden 576) split N ways fall to
  the fork-join break-even, and the substrate's 256 KiB pool floor exists exactly
  to skip GEMVs that small. 135M is well below what a user expects to chat with;
  the models that matter (Phi/Llama/Qwen/GLM, 0.5B–7B+) are all in the scaling
  regime above.
- The implementation adds **no per-token overhead** — the fork-join hot loop is
  entirely in the substrate; the embedder only spawns the pool once (a one-time
  first-turn cost that overlaps the model's own materialization). `wasm-opt`
  (with `--enable-threads --enable-simd`) keeps the threaded build no slower than
  the optimized single-threaded fallback.
- **No arbitrary cap.** The only size ceiling is the wasm32 4 GiB address space
  (a HOST law; `STRUCTURAL_CEILING`), and larger models use the substrate's
  weight-tier pager — the threading path adds no model/size/input cap
  (`context_length` is a passed-through model parameter, never clamped; the pool
  memory is `hardwareConcurrency−1 × 2 MiB` stacks, <1% of the budget for
  realistic core counts). Qwen2.5-1.5B downloads, quantizes, compiles, and
  decodes in the browser (164 s to Ready), threaded, on the codespace.

### Hardening (adversarial pass, 2026-07-10)

An adversarial review (a second agent + a manual pass against the substrate pool
source) drove these fixes; each is now covered:
- **Pool lifecycle (the load-bearing one).** The pool workers are spawned and
  OWNED by the MAIN thread, not the (nested) execute worker, so every path that
  hard-terminates the execute worker — `cancelGeneration`, the worker `onerror`,
  and a worker-reported `error` — tears the pool down with it. Without this, each
  cancel orphaned N workers that each pinned the whole model-sized shared memory
  (OOM after a few cancels). Witnessed by `bdd/probe-threads-teardown.mjs`: 3
  cancel cycles, the live-worker count returns to 0 each time and never exceeds N.
- **Fallback + failure detection.** On any fallback the execute worker signals
  the main thread to drop the pool (no lingering half-pool); a pool worker that
  fails to instantiate is surfaced (its `onerror`/error message) so the readiness
  poll fails FAST instead of waiting out the timeout; a pool worker that dies
  after commit aborts the turn (via a module-level current-turn settler) instead
  of hanging the fork-join forever. Readiness itself stays the race-free shared
  `hologram_pool_workers()` atomic (a worker's `registered` message is premature —
  it precedes the substrate's `fetch_add`).
- **Cancel now settles the turn** (resolve, keeping the partial completion), so
  the caller's `finally` runs and the composer re-enables — fixing a pre-ADR
  cancel-hang the teardown witness exposed.
- **Parametric.** The worker count is `hardwareConcurrency − 1` (no arbitrary cap;
  skip the pool below 2 participants) — the host's cores, never a model/size/input
  parameter. The pool is not spun up for the window plan (m > 1 gains nothing).
- **No silent build degradation.** The threaded build fails HARD if `wasm-opt` is
  absent and asserts the artifact actually has a shared memory.

### Verification trap: the fixture is below the pool floor

`fork_join_gemv` runs serial when `k·n < POOL_MIN_WEIGHT_BYTES` (256 KiB int8;
`wasm_pool.rs:195`). The hermetic `handshake-tiny` fixture's GEMVs are far below
that — so a fixture-only byte-exact test proves the **fallback**, not the pool,
and would stay green even with a broken pool (a dark gate; [[dark-gates]]). The
bit-identity witness must therefore use a model whose decode GEMV exceeds the
floor (e.g. SmolLM2-135M, hidden 576) **and** assert the pool actually fired
(`hologram_pool_workers() === N`, GEMV above floor) — otherwise it witnesses
nothing.

## Alternatives considered

- **`wasm-bindgen-rayon`.** Rejected: the substrate provides its *own* pool wired
  to `fork_join_gemv` with the bit-identity guarantee; layering rayon on top would
  duplicate the thread machinery and bypass that guarantee.
- **Whole-session-per-worker.** Rejected: decode is sequential (token *t+1* needs
  *t*); there is no session-level parallelism to exploit — only intra-GEMV.
- **`COEP: require-corp`.** Rejected: breaks the only external origin (HF) and the
  inference-time provenance re-fetch. `credentialless` isolates without CORP.
- **Std `+atomics` build (skip the no_std surgery).** Rejected: the busy-spin
  futex pins N cores when idle — a worse browser experience than single-threaded.
