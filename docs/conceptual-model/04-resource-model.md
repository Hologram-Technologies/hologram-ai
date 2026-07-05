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
| Residency | max stage + context window, per environment | stage granularity; measured-headroom residency (stages stay resident while the environment measurably has room, one stage is the floor) | `staged-execution`, `stage-residency-cache`, `memory-guard` |
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
- **Decode:** elision bounds per-token compute to the novel suffix cone; zero
  verification, zero recompute-to-check. This is the tokens/sec owner.
- **Sampler:** pinned state; negligible cost, but on the path and part of the
  derivation key.

**A cost on the per-token path that is not decode or sampler is a defect.**

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
Cross-session reuse is conditioned on the full derivation key recurring:
identical graph, prompt cone, and pinned sampler state (params + seed); the
sampler is part of the walk. Derived-label reuse requires bit-exact kernel
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
