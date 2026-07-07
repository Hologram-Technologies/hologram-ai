# 04 — Resource model: maximum entropy in minimum k

Objective: viable LLM streaming in-browser at optimal performance. Means: hold
the maximum entropy in the minimum k-representation. The resource model is
minimal when the addressing equivalence coincides with the semantic
equivalence of the workload, and verification is placed at trust boundaries
only.

## Axes, floors, levers

| Axis | Floor | Lever | Ledger anchor |
|---|---|---|---|
| Rest | corpus entropy / addressing quotient | coarsen the quotient (canonical form) | `kappa-addressing`; candidate S0 canonical rows (open) |
| Transit | set-difference(remote, known) under the quotient; known = provenance-recorded κ, not cached bytes | κ-prior: provenance (exact repeat, under the shard's content pin — build, row `network-skip`), κ-manifest (cross-model — declared); coalesced ranges | `kappa-provenance-resolution`, `network-skip` |
| Structure | O(config) graph + 32B·#tensors(config) identity data | parametric generation; minimal rep = (family id, config, κ-manifest) | `parametric-graph`, `parametricity` |
| Residency | max stage + context window + resident prefix labels O(L·seq·d_kv), per environment | stage granularity — down to SUB-TENSOR: no tensor is atomic; the head partitions into vocab-row chunks via κ-range bindings (`chunked-head`); measured-headroom residency — admission leaves the model's own largest-stage transient bound free (a MODEL-derived margin, computed from the manifest before any byte moves); one stage is the floor; fallback to strict windowing, never refusal | `staged-execution`, `chunked-head`, `stage-residency-cache`, `memory-guard` |
| Generation | novel suffix cone only | window follows the sequence (geometric buckets, model context as ceiling); recursion through the known: resident labels re-derived, not re-executed (CE) | `staged-window-growth`, `decode-elision`, `structural-ce` |

## Critical path (in-browser)

What the user experiences is time-to-first-token and tokens/sec. Each cost is
owned by exactly one lever:

- **Wire:** transit prior (row `network-skip`): under a shard's content pin
  (its HTTP ETag — the Hub's blob hash), provenance-recorded ranges never
  re-transit and unknown runs move as coalesced ranges; a changed pin
  discards the prior wholesale. No skipped byte is trusted — the prior only
  asserts labels; first-touch verification and unpin-recovery govern.
  Dominates time-to-first-use on cold start; zero on warm start. The
  cross-model κ-manifest tier stays declared.
- **Compile:** weightless, O(config); per window bucket, reused while the
  window fits (`staged-window-growth`). Off the per-token path.
- **Materialization:** per stage per window; the session verified-κ set makes
  it read-only I/O after first touch (`session-verified-kappa`); the
  residency cache removes it from the per-token path entirely while the
  environment has measured headroom (`stage-residency-cache`). OPFS sync
  access handles on the worker path.
- **Decode:** two levers, one per factor of cost = cone size × per-element
  cost. Elision bounds the cone to the novel suffix; the Q0 kernel floor
  bounds per-element cost (see Kernel floor). Zero verification, zero
  recompute-to-check. This is the tokens/sec owner.
- **Sampler:** pinned state; negligible cost, but on the path and part of the
  derivation key.

The same rule extends to the per-TURN path (row `warm-turn`): the browser's
staged session — compiled window, resident stages, verified-κ set,
derived-artifact cache — survives across sends, so a warm turn pays decode
with zero recompiles and zero rematerializations; the session rebuilds on
model switch, and a cold turn has identical semantics — warmth is a
projection, never a meaning.

**A cost on the per-token path that is not decode or sampler is a defect.**

## Kernel floor (micro-architectural)

The rest/generation principle recurs at cache scale. A function over a finite
domain is its own table: at Q0 (Z/256) any operation's value-dependent state
is a 256-entry LUT (4 cache lines) plus a 64B psumbook (1 line), L1-resident
by construction. Hit rate on reused state is structurally 1; residual traffic
is single-pass sequential streaming (compulsory misses only, prefetch-hidden).
Fiber-ordered Q8 GEMM touches 1 L1 line per radix pass (substrate plan 033).

Scope: the bound is per tier. Q1 tables (`[u16;65536]`, 128 KB) are
L2-resident on typical L1d; the claim does not lift to Q1. Precision above Q0
routes through the carry chain only (see Totality); there is no float escape.

Grounding in the pinned substrate (read-only, inspected at `hologram-backend`):
the Q1 LUT tier is real and bit-identity-constructed — `cpu/lut.rs`
materializes 16-bit activations as `narrow(f(widen(bits)))` tables, "a pure
speedup, not an approximation" — and the Q0 byte-domain kernels with the
carry chain (CurvatureFlux) exist in `cpu/kernels.rs`. Witness: native
hardware counters (L1-dcache miss ratio partitioned into table vs streaming
accesses, ~0 on the former after warmup) as a performance-contract row. wasm
exposes no counters; in-browser the statement stays structural, native
measurement is the proxy.

## One principle, three scales

Recursion through the known (UOR-Atlas-UTQC: cache collapse subverts
expansion) instantiated at each residency tier:

| Scale | Known set | Reuse act | Store |
|---|---|---|---|
| Content | provenance-recorded κ | wire skip, materialize-once | OPFS κ-store |
| Compute | derived labels (CE) | decode elision of the prefix cone | session memo |
| Kernel | function values over finite domain | LUT lookup replaces recompute | L1 |

The wasm/native decode gap closes from both ends: elision shrinks the cone
(fewer elements), the Q0 floor bounds cost per element (table lookups, no
transcendentals, no thrashing). Neither lever requires threads; both are
admissible in wasm today. What remains after both is irreducible novel
arithmetic, which is the definition of the floor.

## Totality: no classical fallback

The normative posture: the execution path is single and total. Every graph
operation lowers to the quantum hierarchy (Q0→Q3 carry chain); precision is
not a mode switch to a float path but a carry lift, decided by curvature
(CurvatureFlux, statically promoted where the compiler proves it). The float
reference exists at gate time only: tables are built `narrow(f(widen(bits)))`,
bit-identical to the reference by construction, and the reference is then
retired from runtime. A runtime float escape is a defect of the same kind as
per-token verification: it reintroduces the cost structure the model
eliminates and forks semantics into two paths with two behaviors.

Distinguish resource fallback from semantic fallback. Strict windowing under
memory pressure is a projection within the k-model (same semantics, different
plan; never refused). A classical kernel path is a second semantics and is
inadmissible. The first degrades performance; the second forfeits the model.

**The measured present contradicts the posture, and the ledger says so.** The
pinned substrate's dispatch (`hologram-backend::cpu::kernels::dispatch`)
tries `try_dispatch_float` FIRST: any float-dtype kernel call runs in native
IEEE-754 kernels (`cpu/float_kernels.rs`) at runtime, and this repo's
f32/bf16 LLM workloads therefore run the float path today, pervasively. The
substrate is a read-only upstream dependency: retiring runtime float
dispatch for these workloads (the carry lift) is owned there. The row
`total-algebraic-path` is held **open** as a measured target — the probe
reports the float-dispatch fraction of the compiled plan (today ~all of it)
so the frontier is a number, not a hope. It flips to build only when the
number reaches zero and gate-time parity with the retired reference is
witnessed per (op, tier).

Arbitrary models: coverage is the totality of the lowering, measured by the
open row `arbitrary-architecture-coverage`. An op the hierarchy does not yet
express is a dictionary gap that halts loudly, never a license for a float
path. Arbitrary input: totality holds per tier by finiteness (any byte
stream is Q0-valid; higher tiers reached by carry), so novel input executes
on the same path as known input; reuse varies, semantics never.

Arbitrary ceilings are prohibited. Any guard whose bound derives from a
residency assumption is transitional and must be made unreachable by
structure, never kept as a rejection. The instrument is sub-tensor
κ-resolution: a stage binds a byte range of a κ, verified once against the
whole label; the tensor is then a term over ranges exactly as the model is a
term over κs. The head is the first application (row `chunked-head`, build):
vocab-chunked head stages sized to the pipeline's own layer-stage
granularity, logits concatenated across chunks, no whole-vocabulary image
ever resident — any vocab at any scale executes, and the staged preflight
floor has no reachable input (the guard survives only on the monolithic
plan, where it is true). `KappaStore::resolve_range` completes the transit
side: after one whole-κ verification, a ranged binding moves only its bytes.
The same instrument dissolves any future per-tensor ceiling: no single
tensor need ever fit anything.

## Lifecycle: saturation-derived residency (UOR-Framework #2)

Retention is derived from resolution state, not assigned by policy:
λ_eff = λ_base · T_ctx, where T_ctx falls to zero as an object's fibers pin;
σ = 1 means no decay. The pinning events are exactly the trust-boundary
crossings this model already defines — no new runtime work, only counters
off the hot path:

| Event (already occurring) | Pin weight | Tier affected |
|---|---|---|
| κ bound in the active compiled archive | σ = 1 (ground state) | κ-store: never evict while bound |
| First-touch verification (session set) | high | κ-store, session memo |
| Label recurs as operand in a derivation (CE reuse) | medium | session memo: prefix cone crystallizes |
| Materialization / read | low | κ-store |
| Verification failure / prior mismatch | unpin + T spike | evaporates, re-resolves from provenance |

**Build (row `saturation-residency`)** — the one mandatory eviction event,
the inverse of admission: a failed verification UNPINS. The corrupted cache
entry evaporates (`KappaStore::invalidate` — the OPFS entry and its open
handle in the browser, the `{κ}.bin` file natively) and resolution
re-resolves once through the deeper tier (recorded provenance),
re-verifying before anything executes. A wrong cache degrades to a stream,
never a dead end; without a deeper tier the failure stays loud, naming the
label. Only the failing entry is unpinned — bound content is never evicted
by another entry's failure. Witnessed natively (two-tier store: recovery,
evaporation, neighbors untouched, unverifiable-recovery rejection) and in
the hermetic browser journey (a corrupted κ-store tensor: the handshake
still matches the reference and the entry has evaporated).

**Build (σ-order at write pressure, row `memory-guard`)** — the first
pressure-driven eviction, ordered by σ: when the environment's REAL quota
refuses a write (measured headroom is a projection; the write is where the
environment answers), gas-phase content evaporates for crystalline
structure — a refused tensor-cache write stops caching and the journey
continues on recorded provenance; a refused structure write (archive,
provenance record) evaporates cached tensors (provenance-recoverable, gas
by definition) until it lands. Witnessed under a browser-enforced quota cap
between the structure size and the tensor bytes: the download completes,
provenance is fully recorded, and the chat still answers. The admission
margin is likewise model-derived now: a stage session joins the resident
set only while the environment leaves the pipeline's largest-stage
transient bound free (a fixed margin crashed a 1.5B head stage while
smaller stages held the room).

**Declared, open** — the graded remainder: full σ-ordered eviction of the
κ-store across models (unloaded models' tensors evaporate first), λ_base
pressure-scaling generalizing admission to a graded policy, and
session-memo decay derivation. Policy quality (hit rate vs an LRU baseline)
is measured when they land, never asserted. Discipline: hologram-ai needs only the eviction ordering σ
induces, not the thermodynamic vocabulary; FiberBudget/T_ctx stay upstream
primitives.

## Closure of the known set over derivation

The known set closes over deterministic derivation: any artifact computed
deterministically from κ inputs has a derived κ and is itself content,
persisted in the κ-store exactly like weights. Each expensive step runs once
per host and enters every later session's prior (the UTQC seeding pattern);
the warm browser resolves work instead of re-performing it.

Artifact classes this admits:

| Derived artifact | Derivation inputs | Cost it removes | Status |
|---|---|---|---|
| Compiled stage archives | model κ-manifest, config, window bucket, partition | streamed compile per window per session | **build** — row `derived-artifact-kappa` |
| Fused LUTs (Q0/Q1 unary chains) | op chain, dtype, tier | table build | declared (kernels upstream) |
| Quantized weight forms | tensor κ, quantization params | wide-form re-transit and re-materialization | **build** — row `quantized-transit`: the artifact is matmul-ready (transposed at derivation, per-channel symmetric int8, re-derivation bit-identical, ~3.8× smaller than f32 measured); stage graphs bind it as two ranged sub-tensor κ-bindings feeding `Dequantize` adjacent to its matmul (the substrate-fused shape); with wide blobs evicted the staged pipeline generates moving zero wide bytes and equals the quantized monolithic archive. The browser tier is row `quantized-rest` (S1): the download derives the artifacts in-browser and evaporates the wide blobs — the wide form transits once and rests nowhere; a missing artifact re-derives at session warm, fail-closed on its recorded κ. Quantization is a semantic tier, never silent — the catalogue states it per model (data, never code), the narration states it per session, and quality vs the wide tier is measured, never asserted |
| Curvature profiles | model κ-manifest, calibration set κ | per-layer lift decisions | declared (carry lift upstream) |
| Prefill cones (recurring prompt prefixes) | graph κ, template/prefix κ, bucket | prefill of the recurring prefix | declared — candidate `prefill-cone-reuse`, behind `structural-ce` |

Soundness is inherited, nothing new: a derived artifact persists under its
derivation key (a κ over the exact inputs the derivation is a function of)
with its recorded content-κs; a later session with identical inputs resolves
it, content-verified at load — once, off the per-token path; a corrupted
entry evaporates and the recovery is derivation itself, so a wrong prior
degrades to a compile, never a dead end, and never executes unverified
content. Any input change is a different key, never a reinterpretation.
Saturation composes: derived artifacts pin by the same events, so hot
derivations crystallize and rare ones evaporate — the κ-store is a
derivation cache ordered by use.

## Annealing: memo and table are one spectrum (declared)

A table is a total memo over a finite domain; a memo is a partial table over
an observed domain. The Q0 LUT, the prefill cone, and the session memo are
the same object at three densities: total, one point, sparse. A cone
tabulates when either bound is met — structural (finite input domain within
the table feasibility hierarchy) or statistical (σ over its observed domain
crosses the crystallization threshold; tabulation is what σ = 1 means
operationally). Tables tier by size across the residency hierarchy (Q0 op
tables in L1, fused chains in heap, vocab-scale cone tables in the OPFS
κ-store); tabulation is a density claim, never a residency claim. Both
directions are semantics-free: a table entry is `derive(inputs)` by
construction; eviction is melting, by the same lifecycle. Idle time feeds
the anneal (pre-deriving entailed work off the critical path; speculation is
the lowest-σ content, so pressure throttles it first). Candidate row `cone-tabulation` is held open (it depends on elision/memo
internals owned by the substrate — the session memo is inside the runner).
Row `idle-derivation` is **build** for its first clause: between turns the
session pre-derives the next window bucket's stage archives into the
derived store, off the per-token path (weightless — no weights move, the
resident window and its counters untouched), and a later crossing RESOLVES
the window instead of compiling on the critical path; abandoned speculation
is ordinary derived content, evaporable by the lifecycle. Continuation
cones and table densification stay declared with `cone-tabulation`.

## End state: two traffic classes (declared)

The annealed per-token access set is total over two classes: structurally
L1-resident reused state (hit rate 1 by the kernel floor) and single-pass
prefetchable streams (compulsory misses only) — weights in materialization
order, resident prefix labels under attention, activations. Elision removes
recomputation, tabulation removes derivation where use concentrates,
totality removes every access outside the two classes. The per-token floor
is then O(L·d²) table-lookup MACs + O(L·seq·d_kv) streamed label reads, both
bandwidth-shaped. Anything measured above the floor is attributable: a cone
not yet elided, a table not yet dense, or an access in neither class — a
defect by Totality. This sharpens the critical-path rule into an audit.
Predicated on the kernel floor and totality, both upstream-owned today (see
Totality's measured present); the resident-prefix-label term of the
residency floor becomes measurable when decode elision externalizes the
prefix labels.

## Benchmark: efficiency against floors (the `performance-contract` content)

The floors make benchmarking generic: report measured/floor per axis, never
absolute times — the ratios are the implementation's quality, comparable
across hosts, models, and inputs; a ratio of 1 is the ceiling for that axis.
Calibration first (measured stream bandwidth + lookup throughput, so floors
are stated in the environment's own units). Per-axis ratios: wire (bytes
fetched / set-difference entropy), rest (store bytes / corpus entropy under
the active quotient), structure (archive bytes / O(config) + 32B·#tensors),
residency (peak claimed heap / max stage + window + prefix labels), TTFT
(measured / wire+compile+prefill floors, each 0 on hit), decode (s/token /
bandwidth-shaped two-class floor). Coverage is parametric: a (family,
config) sweep with scale as a dimension — the claim is only witnessed where
the model exceeds what the environment holds, and a flat ratio across each
structural boundary IS the scaling claim, measured. Input sweeps an
entropy-controlled corpus across the reuse spectrum; the primary output is
the reuse curve. Attribution counters (elided vs executed, hits vs derives,
skipped vs fetched, asserted vs verified, in- vs out-of-class) map every
excess to one lever in this document; a residual mapping to no lever is a
model gap and becomes a row. The harness is itself a k-citizen: fixtures are
parametric derivations (never fully materialized anywhere), and everything
the harness produces enters the κ-store as derived content under the
lifecycle — a run's residual footprint is its report κ.

This is the content of the open `performance-contract` row: ratio thresholds
per axis, held open until measured, tightened as levers land, never
asserted.

## Verification placement

Verify at trust-boundary crossings, once per crossing, never per traversal.

- **Mint (network → runtime):** free; hashing is what produces κ.
- **Prior meets content:** manifest- or provenance-asserted κ verifies at
  first materialization. Once (`session-verified-kappa`).
- **Session cache:** the session-local verified-κ set; a κ verified this
  session materializes without re-hash. Staged execution re-materializes; it
  must not re-verify. The set is session-scoped by construction: a fresh
  session re-verifies at first touch and rejects corrupted content naming
  the label.
- **Write path:** cache integrity is write-once atomicity, not read-side
  re-hashing.
- **Elision path:** zero runtime verification. Derived labels are asserted in
  the hot path; their soundness is gate-time (`structural-*` witnesses, CI).
  Recompute-to-check deletes the advantage.

The prior accelerates; the posterior governs.

## Generation over the known

A decode step's derivation walk re-derives labels for the unchanged prefix
cone and executes only the novel suffix — the cache-collapse advantage
established by UOR-Atlas-UTQC, inherited through holospaces, instantiated as
decode elision with no KV-cache (`decode-elision`). Generated content is
itself κ-labeled on production, so generation extends the known set.
Elision keys on realized token ids: once emitted, a token is an input, and
the prefix cone derives from it regardless of how it was chosen. Pinned
sampler state (params + seed) enters the derivation key only where the walk
itself is reproduced: speculative continuation cones and cross-session
identical walks. The novel cone, precisely: one position's forward pass —
per token, O(L·d²) projection/MLP work at the new position plus attention
reads over the resident prefix labels (the K/V-class outputs of every prior
position), which are O(L·seq·d_kv) bytes, grow linearly with the sequence,
and belong in the residency floor. Derived-label reuse requires bit-exact kernel
determinism, witnessed per environment (`structural-ce`); cross-environment
reuse is open, not build.

## Soundness condition (all axes)

1. **Congruence.** A quotient is admissible only if it is a congruence with
   respect to the kernels: any two representatives of a class are
   execution-indistinguishable. Without congruence, representative
   substitution changes outputs while every hash check passes.
2. **Tagged verification.** Under quotients the check is
   canonicalize-then-hash, so every κ carries its quotient tag (byte-κ is the
   identity quotient). Migration between quotients is re-mint + rebind,
   never reinterpretation.
3. **Fail closed, recover by rebind.** An asserted κ either verifies or
   resolution rejects with the label; a rejected prior is recovered by
   fetching the current model's recorded range, minting κ from the bytes,
   and recompiling the weightless binding. A wrong prior degrades to a
   stream; it never dead-ends the journey and never executes on unverified
   content.

Each declared equivalence is its own dictionary row with its own witness and
its own measured cost; no aggregate canonicalization claim.

## Candidate rows (open — declared, not asserted)

- S0 `canonical-kappa-<eq>`, one row per declared equivalence: quotient
  decidable; canonicalize-then-hash reproduces a fixed point; congruence
  witnessed against execution parity on representative pairs; κ tagged with
  the quotient; fail-closed at materialization.
- S1 `network-skip` cross-model tier: a published κ-manifest as a
  content-addressed local prior across models. (The exact-repeat tier is
  build — see the Wire lever.)
- S3 `total-algebraic-path` (open, measured): every executed kernel is a
  hierarchy kernel; zero runtime float dispatch; parity with the retired
  reference witnessed at gate time per (op, tier). Held open by the pinned
  substrate's float-first dispatch for float dtypes (see Totality); the
  probe reports the measured float-dispatch fraction.
- Kernel-floor performance contract: native hardware-counter witness
  (L1-dcache table-access miss ratio ~0 after warmup) once Q0-tier kernels
  carry these workloads; structural-only until then.
- Measured, never asserted: dedup ratio per corpus; elision ratio per
  workload; stage-granularity optimum per environment.

## Measured (2026-07-05, addendum — Qwen2.5-1.5B bf16, headless Chromium)

The 1.5B staged journey drove four fixes, each measured against a real
trap: the fixed admission margin crashed its head stage (fixed:
model-derived margins); the codespace's REAL quota (enforced at the write
while estimate() reported 6.4 GB — the projection-vs-write distinction,
observed) killed the download (fixed: fail-soft + σ-eviction); the head
EXECUTION working set — two whole-vocabulary F32 images plus material
copies, 3.27 GB measured — structurally exceeded a 32-bit tab even fully
strict. The whole-vocabulary head was the last residency ASSUMPTION, and
it is now removed the same way whole-model residency was: the head
partitions into vocab-row chunks at the pipeline's own stage granularity
via **κ-range bindings** (row `chunked-head` — sub-tensor κ-resolution: a
chunk binds a byte range of the whole tensor's κ; verification covers the
whole content once; no whole-vocabulary image ever materializes). The
transit side matches the residency side: verification is the ONLY
whole-content read — once a session has verified a κ, a ranged binding
rematerializes through `KappaStore::resolve_range` (an OPFS `read({at})` or
a ranged GET inside the recorded provenance span), moving only its slice
(witnessed: a verified pass moves the ranged reads' exact tiling of the
tensor, never chunk-count × whole). Chunked
logits agree with the whole head within kernel reduction-order tolerance
(measured ≤ 4e-7 — the substrate's matmul tiling varies with output width)
with EXACT greedy-decode parity. A head within layer granularity is one
chunk: the classic head stage, byte-identical archives. The execution
working-set floor guard remains only where it is true: the MONOLITHIC
plan, which small models take; staged preflight validates the stage graphs
the plan will actually build.

## Measured (2026-07-05 — Qwen2.5-0.5B-Instruct bf16, headless Chromium, codespace)

Reported, never asserted:

- Download + in-browser streamed compile (14 stages, model's own 32k
  context): **~31–38 s**.
- First chat token: **~60–90 s** after send (window compile ~2 s, then one
  materialize+execute pass over 14 stages, narrated per stage).
- Per token thereafter: **~85 s** in wasm. Native, same code and κ-store:
  **~4 s** cached (residency budget holds the model) vs **15–25 s** strict —
  the residency cache is a measured **3–4×**; the session verified-κ set
  removes a full model of BLAKE3 from every strict pass.
- Enabling wasm SIMD128 moved end-to-end by only ~4%: LLVM does not
  reassociate float reductions, so the matmul inner loops stay effectively
  scalar. The remaining wasm/native gap (~20×) is substrate kernel speed —
  the substrate is a read-only dependency, so this axis is owned upstream.
- Content-addressed elision occasionally collapses a whole native decode
  pass to **~22 ms** (observed once per short generation); making the
  prefix-cone hit rate structural rather than incidental is the largest
  open tokens/sec lever in this repo's control (`decode-elision`).

## Measured (2026-07-06 — the decode ratio is a number)

The `performance-contract` target lane now runs the Benchmark section: it
calibrates the environment's stream bandwidth (22 GB/s on this codespace),
computes the fixture's decode floor from weights-per-pass (0.026 ms/token),
times staged decode steps under residency, and attributes each step. First
reading: **215–414× floor**; elision fires on the staged resident path
(51–64% of kernels elided; zero rematerialization after the first pass).
The attribution names the structure of the excess:

- **The window multiplies everything.** Layer kernels span the whole
  compiled window, so one changed position dispatches the whole kernel —
  per-step cost is a window-sized forward regardless of prefix length.
  Elision at kernel granularity cannot save a kernel whose input changed
  anywhere. The model's own answer is label-granular prefix-cone reuse,
  which needs per-position kernel decomposition (candidate row
  `positional-cones` — the recipe emits decode cones per position so the
  unchanged prefix elides structurally) or sub-kernel elision
  (substrate-owned, like `total-algebraic-path`). This is the decode
  frontier, now measured rather than suspected — and realized as row
  `decode-plan` (build; measured below).

  Feasibility, source-confirmed for the tractable realization — a DECODE
  PLAN beside the prefill plan: K and V are ordinary graph tensors (the
  `GroupedQueryAttention` op consumes them as inputs; nothing is locked
  inside a fused kernel), and for a single last-position query causal
  masking is vacuous, so a decode step needs only existing ops — seq-1
  projections, rotation at the absolute position as DATA — the canonical
  RoPE lowering bakes relative-position tables at compile time, so the
  decode plan takes cos/sin at the consumed position as runtime inputs
  (synthesized from the config's own rope base and head_dim, like every
  auxiliary) and rotates with plain arithmetic — Concat
  of the carried prefix K/V with the new position's K/V, GQA with q=1
  against the full carried span, MLP at one position, the
  single-position head with no gather. The carried K/V are not a mutable
  cache: each step's K/V output is DERIVED CONTENT of the realized tokens,
  flowing through named ports exactly as stage activations do — the
  "resident prefix labels" of the Generation axis, made explicit. Per-step
  compute drops from a window-sized forward to the decode floor's own
  O(L·d²) + attention reads. Prefill (the whole-window pass that seeds the
  carried K/V) remains the existing plan with per-layer K/V outputs.
- **The head was window-multiplied for a one-row read** — fixed: row
  `single-position-head` (build). The pipeline gathers the consumed
  position's hidden state after the final norm (`last_pos`, a runtime
  input the generation loop synthesizes by name like every auxiliary) and
  the head — whole or chunked — computes O(vocab·d) per step, never
  O(window·vocab·d). At a 152k vocabulary this removes roughly a quarter
  (0.5B, window 128) of every decode step; the reference-parity witness
  sweeps the gather over every position. Stage-archive derivation keys
  bump to v3 (the recipe is part of the derivation function).

## Measured (2026-07-06 — the window left the step: row `decode-plan`, build)

The decode plan is built and witnessed, native first. The decoder recipe is
emitted at seq = 1 and every fused attention node is decomposed into masked
past-attention over a fixed bucket of carried K/V rows — all existing ops,
strictly 2-D matmuls per kv-head group (the substrate's MatMul kernel is
2-D; nothing batched is assumed):

- **Carried K/V is derived content through named ports**, not a mutable
  cache: `past_k_l`/`past_v_l` enter as inputs, `k_new_l`/`v_new_l` leave
  as outputs, and the ENGINE splices each step's rows into its buffers
  between steps (`DecodeSession`). No scatter op exists in the graph.
- **Positions are runtime data.** `rope_cos`/`rope_sin` are synthesized at
  the token's absolute position from the config's own rope base and
  head_dim (the canonical RoPE lowering bakes relative tables at compile
  time, so rotation arrives as data and is applied with plain arithmetic —
  rotate-half, pair `j ± d/2`); the additive `decode_mask` erases
  unrealized bucket rows inside the softmax, so garbage bytes past the
  realized length never touch the numbers.
- **One compiled artifact serves every step.** Bucket exhaustion is a
  geometric recompile with a row copy (witnessed mid-sequence: 2 → 4 → 8
  with parity intact), never a ceiling; the model's own trained context is
  the only semantic bound.
- **Parity is the gate.** Per position, the decode plan's logit row equals
  the whole-window plan's row (reference witness sweeps every position;
  the BDD scenarios replay the fixture tokens through both plans).

Measured on the fixture (target lane, same calibrated floor as above):
staged whole-window steps run **3.8–8.4 ms/token (156–344× floor)**; the
decode plan runs **0.5–1.2 ms/token (20–50× floor)** — the window
multiplier is gone from the step, and the decode-plan cost is
window-INDEPENDENT: it stays flat as context grows while the whole-window
step scales with it. The residual multiplier at fixture scale is per-step
dispatch overhead (~145 tiny kernels); at real scale the matmuls dominate
and the ratio compresses toward the floor's own terms.

The staged/browser realization is built on the same partition contract:
the decode stage graphs cut at the whole-window plan's own boundaries
(identical κ-map coverage; each layer stage carries its layers' K/V ports
at absolute indices), the staged pipeline routes ports by NAME (the
archives' own port sections are the contract — shared position ports feed
every layer stage from one pipeline port, carried K/V surfaces as
trailing pipeline outputs), and the staged decode pipeline is
**byte-identical** to the monolithic decode plan per position. Greedy
completions are identical across the decode and whole-window plans
(witnessed), so the plan switch is invisible to the transcript. In the
browser, `DecodeChatSession` is the default chat path (knob
`hologram_decode_plan=0` reverts): every token — prompt prefill included —
is one single-position pass; buckets grow geometrically through the same
derived-artifact store under a decode-specific derivation key.

**Cross-turn K/V retention** (same cycle, forced by measurement): the
carried rows persist across turns, keyed by their realized tokens. A
prompt extending the session's realized sequence — a chat transcript
extends its own history — rewinds to the shared prefix and steps only its
novel suffix; witnessed exactly (turn 2 stepped 5× for a 10-token
transcript: its 2-token suffix plus generation), with the completion
equal to a fresh replay. The Generation axis's "resident prefix labels"
are now held across turns, not just across steps.

## Measured (2026-07-06 — Qwen2.5-0.5B int8, headless Chromium, DEPLOYED instance)

The decode plan at real scale, on the live Pages deploy (probe against the
shipped catalogue entry):

- Download + streamed compile + int8 derivation: **41.6 s** (unchanged).
- Turn 1 (cold, ~40-token templated prompt, 26 streamed samples):
  **~21 s/step average** (prefill steps + generated tokens), vs
  **~85 s/token** under the whole-window plan — the window multiplier left
  the step, and per-token cost no longer grows with context.
- The residual per-step cost is WEIGHTS-SIDE, window-independent: a
  single-position matmul streams the full weight set with no row reuse,
  so per-element cost × O(weights) elements dominates where the
  whole-window plan amortized weight traffic across positions. This is
  the kernel-floor axis (per-element cost, substrate-owned), now the
  measured owner of the browser gap — not the window, which is gone.
- Turn 2 under per-turn transcript replay measured CATASTROPHIC
  (>30 min for a ~130-step replay) — the measurement that forced
  cross-turn retention. With retention deployed and re-measured: turn 2
  completes in **1084 s ≈ suffix + generation only** (~23 tokens at
  ~46 s/step; a full replay would have been ~5850 s) — the retained
  prefix is verified at real scale. The bucket machinery narrated its
  design: 64 → 128 compiled at growth, 256 RESOLVED from the derived
  store (the idle prederive's hit).
- The remaining in-repo lever was prefill seeding — realized as row
  `chunked-prefill`: the decode plan generalized to seq = C. A prompt
  suffix seeds in ceil(n/C) passes instead of n (one weight stream per
  chunk, not per token; C = 32 in the browser); intra-chunk causality
  enters through the same additive mask that erases unrealized rows;
  rope tables arrive pre-expanded to the plan's head-major layout
  (exact-shape arithmetic, zero broadcast assumptions); a partial final
  chunk PADS — padded rows land above the realized length, unreachable
  by the mask until overwritten, sound by the same law as the fixed
  bucket itself. Witnessed: a chunk-seeded session is indistinguishable
  from a step-fed one at the sampler row and every subsequent position;
  ceil(n/chunk) passes counted exactly. The seeder resolves through the
  derived store per (bucket, chunk) and drops on bucket growth
  (re-installed lazily). The per-token GENERATION cost (~45–50 s at
  0.5B) remains the substrate's single-position kernel throughput in
  wasm — the kernel-floor axis, owned upstream.
