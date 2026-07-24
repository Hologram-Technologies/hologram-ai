# Upstream issue: v0.9.0 fused decode attention traps `unreachable` in wasm at production scale

**Status: RESOLVED in hologram v0.10.0 (`47a4955`), 2026-07-14.** Found the same
day via the deployed instance; substrate fix released and adopted the same day.
**Substrate:** Hologram-Technologies/hologram; the trap was present at v0.9.0
(`22b0ce1`), fixed at v0.10.0 (`47a4955`).

## Resolution

hologram **v0.10.0** makes the fused decode (κ119 `DecodeAttention` + κ120
`KvCacheWrite` move) sound on the wasm32 target across a staged carry-across-
eviction step. We adopted it (`Cargo.toml` pins all `hologram-*` crates to
`47a4955`) and **re-enabled the fused path on wasm** — the browser now ships the
same fast decode as native. Verified end to end BEFORE deploy, not after:

- the two hermetic `wasm-pack test --node` repros (bare κ119/κ120 over a
  realized past; the resident-KV carry/steal) — green on the wasm target;
- the staged head_dim-128 browser gate
  (`features/suites/s4_application/deep_model_journey.feature`, the staged
  scenario: staging + weight-budget eviction + multi-token carry) — green in
  real Chromium on the wasm build, where on v0.9.0 the fused path trapped and
  the legacy fallback could not even compile (see below).

The temporary mitigation (browser falls back to legacy) is **removed**: it was
not merely slower, it was defective — the legacy decomposition hit
`CompletenessFailure` compiling the MQA staged decode at head_dim 128 (a
legacy-only limit the fused form does not share), so a real staged model on the
fallback failed to start at all. v0.10.0 makes the fallback unnecessary.

The record below is retained for history.

---

**Original report (v0.9.0):**

## Symptom

On the deployed GitHub Pages instance, a real model (Qwen/Qwen2.5-1.5B-Instruct:
28 layers, head_dim 128, GQA kv_heads 2, vocab 151936, context 32768, int8,
34 staged archives) downloads, compiles, and streams its FIRST token, then
traps on the SECOND decode step:

```
… stage 34/34 materialized …
The                       ← first token streams
RuntimeError: unreachable ← second step traps
```

The first token comes from prefill / the first step, which runs the KV cache
write as an honest COPY (the cache was ingested from host bytes, not a retained
label). The SECOND step is the first that binds the RESIDENT K/V carry
(`carry = true`) — the first `KvCacheWrite` eligible for the in-place κ-MOVE
(steal), and the first `DecodeAttention` reading a carried past under labels.
The trap is deterministic across the pool being single- or multi-threaded, so it
is not the pool.

## What we established

We narrowed the fault by ELIMINATION with hermetic in-wasm tests that compile
AND execute the real substrate kernels on the wasm32 target the browser ships
(`crates/hologram-ai-wasm`, run by `wasm-pack test --node`). Two candidate
sub-paths were ruled OUT — both pass in wasm at production head_dim 128:

- `fused_decode_over_realized_past_in_wasm` — the bare κ119 `DecodeAttention`
  + κ120 `KvCacheWrite` step over a REALIZED past (pos ≥ 1, mask revealing real
  keys), at head_dim 16 and 128. **Passes.** So the κ119 read of a non-empty
  past and the κ120 write are sound in isolation on wasm.
- `fused_resident_carry_two_walks_in_wasm` — the resident-KV carry/steal over
  TWO `execute_kv_resident` walks (`carry = false` then `carry = true`, the
  first that `release_label`s the retained cache so κ120 does the in-place MOVE
  and κ119 reads the carried past by label), at head_dim 128. **Passes.** So the
  in-place move and the carried-label read are sound on wasm WITHOUT staging.

- **Native is correct at the full shape.** `hologram-ai`'s
  `decode_family_coverage` drives the int8-quantized STAGED decode at head_dim
  128 for every family, multi-step, INCLUDING under stage-eviction pressure (a
  1-byte residency budget that drops and re-materializes every stage between
  steps) — bit-identical, reproducible, no trap.

**By elimination the remaining differentiator is the STAGED carry across a
dropped-and-rematerialized stage**: when a stage is evicted under the residency
budget and later re-materialized from the κ-store, the resident K/V carry for
that stage is banked (`kv_shadow`) and restored, and the carried cache LABEL is
rebound to the freshly re-materialized stage. That rebind — a κ-label bound to a
cache that was evicted and re-created — is the one thing the two passing repros
do NOT do and the deployed (34-stage, budget-bounded) path does. It is sound on
native and traps `unreachable` on wasm32. The suspect is the substrate's wasm
handling of a κ120 in-place MOVE / κ119 read against a cache label whose backing
was re-materialized after eviction (a stale/aliased offset or a `usize`-32
index recomputed from the rebound stage geometry).

`RuntimeError: unreachable` is a wasm trap (panic-to-abort or an explicit
`unreachable`); it is NOT surfaced through our panic hook
(`console_error_panic_hook`/`last_panic`), so it originates below our std layer
— i.e. in the no_std substrate kernel/executor, not our binding.

## Repro

- **Hermetic, in CI, in seconds:** the two `wasm-pack test --node` tests above
  isolate the sub-paths that are sound. They give a substrate engineer a running
  in-wasm harness (compile-in-wasm → `execute_kv_resident`) to extend: add a
  THIRD walk that forces a stage eviction + re-materialization between walks 1
  and 2 at head_dim 128, and the trap should appear. (Our repros are mono
  single-stage, so they stop one step short of the eviction rebind.)
- **Hermetic browser gate:** `features/suites/s4_application/deep_model_journey.feature`
  (the staged scenario) drives a synthesized head_dim-128, int8, STAGED fixture
  under a forced weight budget so stages page and evict between decode steps —
  the full deployed shape, minus the multi-GB size. It runs the SHIPPED (legacy)
  decode, so it is green today; flip `FUSED_RESIDENT_DECODE` on for wasm and it
  reproduces the deployed trap.
- The deployed real model reproduces it every turn.

Happy to contribute a substrate-level wasm test at head_dim 128 that evicts and
re-materializes a stage across the carried-past `DecodeAttention` + `KvCacheWrite`
move.

## Ask

Make the fused decode path (κ119 `DecodeAttention` + κ120 `KvCacheWrite` move)
sound on the wasm32 target when the carried cache label was banked across a
stage eviction and rebound to a re-materialized stage at production head_dim —
or fail LOUD with a diagnosable `BackendError` (never an opaque `unreachable`).
A wasm-target test at head_dim 128 that evicts a stage between two carried steps
would lock it.
