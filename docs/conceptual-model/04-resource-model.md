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
| Transit | set-difference(remote, known) under the quotient; known = provenance-recorded κ, not cached bytes | κ-prior: provenance (exact repeat, under the hf-hub revision-pin oracle), κ-manifest (cross-model); coalesced ranges | `kappa-provenance-resolution`; candidate S1 `network-skip` (open) |
| Structure | O(config) graph + 32B·#tensors(config) identity data | parametric generation; minimal rep = (family id, config, κ-manifest) | `parametric-graph`, `parametricity` |
| Residency | max stage + context window + resident prefix labels O(L·seq·d_kv), per environment | stage granularity; measured-headroom residency (stages stay resident while the environment measurably has room, one stage is the floor) | `staged-execution`, `stage-residency-cache`, `memory-guard` |
| Generation | novel suffix cone only | window follows the sequence (geometric buckets, model context as ceiling); recursion through the known: resident labels re-derived, not re-executed (CE) | `staged-window-growth`, `decode-elision`, `structural-ce` |

## Critical path (in-browser)

What the user experiences is time-to-first-token and tokens/sec. Each cost is
owned by exactly one lever:

- **Wire:** transit prior (skip known κ, coalesced ranges). Dominates
  time-to-first-use on cold start; zero on warm start.
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

**Declared, open** — the graded remainder: σ-ordered pressure eviction of
the κ-store (unbound gas-phase content of unloaded models evaporates first;
the active binding is crystalline by construction), λ_base pressure-scaling
generalizing the residency admission probe from binary to graded, and
session-memo decay derivation (prefix-cone labels re-pin every token).
These need an eviction-pressure mechanism that does not yet exist in-repo —
today the κ-store budget is write-time only and nothing else evicts. Policy
quality (hit rate vs an LRU baseline) is measured when they land, never
asserted. Discipline: hologram-ai needs only the eviction ordering σ
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
| Quantized weight forms | tensor κ, quantization params | wide-form re-transit and re-materialization | declared — candidate `quantized-transit` |
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
the lowest-σ content, so pressure throttles it first). Candidate rows
`cone-tabulation` and `idle-derivation` are held open: both depend on
elision/memo internals owned by the substrate (the session memo is inside
the runner) and on the derivation-closure machinery above, which is now the
build foundation they compose over.

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
- S1 `network-skip`: no skipped byte is trusted; every asserted κ verifies at
  materialization or resolution rejects and recovers by re-mint + rebind.
  Exact-repeat tier from recorded provenance (hf-hub revision-pin oracle);
  cross-model tier from a published κ-manifest.
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
