# ADR-0018: Multi-Threaded WASM Decode (Embedder Worker Pool)

**Status:** Accepted (correctness gates green; on substrate v0.8.2 the pool covers
decode AND prefill — TTFT pools 3.2–3.7×, decode GEMV 2–3×; see Verification
results, incl. the honest browser end-to-end caveat)
**Date:** 2026-07-10 (v0.8.2 rebase 2026-07-11)
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

Per-model metrics, with units (`pnpm bench:pool` → `scripts/bench-pool-scaling.sh`
builds `scripts/pool-bench.rs` against the substrate's pub int8 GEMV + pool, runs
it under wasmtime/wasm32-wasip1-threads — std threads drive the SAME atomics
fork-join the browser web workers do). It composes each metric from the model's
REAL step from every int8 projection GEMV: **decode** = per-token step (M=1 GEMVs
across layers + LM head), **TTFT** = prefill (M=128 GEMM) + 1 decode step. SERIAL
vs POOLED (3 workers + main = **4 participants**, the codespace's 4 physical
cores). As of substrate **v0.8.2** the pool covers BOTH decode (M=1) and prefill
(M>1). Numbers are ±noise on the shared VM:

  | model | ~GB | decode tok/s (s→p) | sp | **TTFT ms (s→p)** | **sp** |
  |---|---|---|---|---|---|
  | Qwen2.5-0.5B | 0.5 | 31 → 61 | 2.0× | 2430 → 1087 | **2.2×** |
  | Llama-3.2-1B | 1.2 | 13 → 35 | 2.7× | 6201 → 1682 | **3.7×** |
  | Qwen2.5-1.5B | 1.5 | 11 → 30 | 2.8× | 8941 → 2533 | **3.5×** |
  | Qwen2.5-3B   | 3.1 | 5.1 → 15 | 3.0× | 18954 → 5360 | **3.5×** |
  | Phi-3-mini-3.8B | 3.7 | 5.1 → 14 | 2.8× | 22637 → 7033 | **3.2×** |
  | Qwen2.5-7B   | 7.1 | 2.2 → 4.7 | 2.1× | 56603 → 15334 | **3.7×** |
  | Llama-3.1-8B | 7.5 | 2.2 → 5.1 | 2.4× | 48148 → 13172 | **3.7×** |

Reading the metrics:
- **TTFT (prefill) pools 3.2–3.7× — the biggest real win.** Prefill is a
  compute-bound M=P GEMM; v0.8.2 pools it (work-keyed admission), so first-token
  latency drops with cores: 1.5B 8.9 s → 2.5 s, 7B 56.6 s → 15.3 s (at a 128-token
  prompt; ∝ prompt length). Prefill is GEMM-dominated, so unlike decode this
  speedup is NOT diluted by non-GEMV work — it lands end-to-end.
- **Decode pools the GEMV 2–3× (bandwidth-bound), but the END-TO-END browser
  decode gain is smaller.** The fork-join partitions output columns so each weight
  byte is read once — bandwidth-optimal, no embedder change beats it. BUT the real
  browser decode STEP is only ~25–50% GEMV; the rest is non-GEMV substrate work
  (attention, κ content-addressing, sampling) the pool cannot touch. By Amdahl a
  2–3× GEMV speedup is ~1.2–1.5× end-to-end, and the contended 4-core codespace
  (pool + vite + Chromium + Playwright) eroded a real browser 0.5B measurement to
  **~1.0×** (`bdd/bench-decode.mjs`: 0.98× at 7 workers, 1.02× at 3). The true
  real-machine decode gain (free cores, larger/more-GEMV-heavy models) is between
  1× and the GEMV's 2–3× and must be measured on a real host / the deployed
  instance — the codespace cannot isolate it.
- **Worker count `hardwareConcurrency−1` is validated** (the wasmtime sweep
  saturates at ~physical cores then plateaus; no degradation from HT). A runtime
  override `hologram_pool_workers` exists for tuning/diagnosis. No arbitrary cap.
- SmolLM2-135M's ~1.0× is the smallest model's floor (below the fork-join
  break-even), not implementation overhead; 135M is below chat scale.
- **No arbitrary cap.** The only ceiling is the wasm32 4 GiB address space (a HOST
  law; `STRUCTURAL_CEILING`); larger models use the substrate weight-tier pager.
  The threading path adds no model/size/input cap (`context_length` passed through,
  never clamped; pool stacks `hardwareConcurrency−1 × 2 MiB`, <1% of budget).
  Qwen2.5-1.5B downloads, quantizes, compiles, and decodes threaded in the browser
  (164 s to Ready).

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

## Follow-on optimizations (2026-07-11)

The per-model benchmark (`scripts/pool-bench.rs`) and the decode-step
decomposition it grew (`docs/notes/throughput-latency-analysis.md`) showed the
pool is not the whole story: at chat context lengths the serial attention + KV
path and the fixed weight-bandwidth dominate. Two follow-ons land here, each
substrate-proven and behind its own witness:

- **Eager pool prewarm.** The pool spawn + the staged-session build ran inside the
  FIRST turn's TTFT. A `warm` message (`generate({warm:true})`) runs the identical
  setup — `preferThreadedPool → ensureReady → warmStagedSession` — then STOPS before
  generate; the warm session + pool are cached, so the first real turn pays neither.
  `Chat.tsx` prewarms once per archive when the model's meta is ready and idle, and
  `onSend` AWAITS the prewarm promise before its own `generate` — the warm build and
  the first turn never build the session concurrently (which would race the shared
  residency ledger, the invariant the crash-fixes established). The pool stays
  main-owned, so cancel/error still tears it down: prewarm adds no lifecycle
  exception. Best-effort — a prewarm failure is swallowed and the first turn builds
  as before.

- **int4 weight tier + PARAMETRIC tier selection.** A selected model is
  AUTOMATICALLY compiled at its optimal tier — never a user knob.
  `QuantTier::optimal_for(params, address_space)` is the single law: int8 for
  quality whenever the model's int8 resident weights fit the ¾-of-4-GiB weight
  budget; int4 ONLY when int8 would not fit resident but int4 would (keeping a
  larger model resident + interactive); int8 (served by the weight-tier pager) for
  anything larger. The browser download computes the model footprint and resolves
  the tier automatically (`optimal_quant_tier` wasm binding), narrating the choice;
  the catalogue default is `"auto"`, and `hologram_quantize` is a diagnostic
  override only. int4 itself is wired end-to-end: `encode_int4_per_channel{,_omajor}`
  (per-channel symmetric, two's-complement nibbles low-first, bit-exact to the
  substrate's `I4_VALUES`); the tier rides in the `QuantMap` value so the κ-binder
  declares `DType::INT4` + halved ranges; both emit paths — κ-artifact (browser/
  staged) and inline `quantize_weights` (native) — decode, each substrate-proven
  (`omajor_i4_substrate_contract.rs` reproduces the exact i4 integer oracle;
  `int8_accuracy.rs` decodes inline i4 to cosine ≥ 0.97). **Honest cost:** the fused
  kernel is PER-CHANNEL (one scale per output channel), and at the MODEL level int4
  is severely, MODEL-DEPENDENTLY lossy — measured 0.66/0.96/0.66/0.72 logit cosine
  vs bf16 across Llama/Qwen2/Mistral/Phi3 (int8 is ≥0.99), the ~16% per-GEMV error
  compounding across layers + the head (`int4_decode_tracks_bf16_..._for_every_family`).
  So the parametric policy PREFERS int8, reaching for int4 only as the price of
  keeping a too-large model resident at all; quality is measured and stated, never
  silent, and never a user's manual burden.

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
