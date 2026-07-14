# Decode performance & correctness plan — the deployed Qwen2.5-1.5B evidence

**Status: analysis complete (file:line-grounded), implementation staged by tier.**
**Trigger:** a deployed Qwen2.5-1.5B log shows ~2.4 tok/s, ~12–18 s re-materialization
per decode-window growth, and `RuntimeError: unreachable` recurring at the 256/512
window step. This note records the ROOT CAUSES and the parametric fixes.

> The log predates the v0.10.0 merge (main @ `8e5a13c`, 08:49 UTC). Re-verify the
> `unreachable` on the now-deployed build; the performance findings are
> independent of the decode kernel version (they are staging/residency issues).

## Root causes (confirmed)

### ① DOMINANT — a resident-capable model is shattered into 30–34 one-layer stages, then re-paged on every window growth

The log itself says "1.54B params fit resident at int8", yet it stages into 34
archives (one 445 MB embedding + 33× ~45 MB) and re-materializes all ~1.9 GB from
the κ-store on every window doubling (64→128→256→512), ~12–18 s each.

The cause is **not** the runtime KV budget — `decode_residency_budget`
(`crates/hologram-ai-wasm/src/lib.rs:1259`) correctly charges KV at the *current*
bucket (`lib.rs:1762`, `decode.rs:71`). The cause is the **compile-time estimate**,
`apps/web/src/resources.ts` `estimateResources`, which freezes `layersPerStage = 1`
from two wrong inputs:

- **Activation reserve at MAX context (32768)** — `resources.ts:157,159-163`:
  `perLayerActivation(ctx) = ctx·hidden·8·4`; at ctx=32768, hidden=1536 that is
  **1.61 GB per layer**, which dominates `layerBytes` and pins `layersPerStage=1`
  (`resources.ts:164`). But this term is a *prefill* (m=C) activation; the resident
  DECODE session runs chunk=1, whose activation is ~50 KB, not 1.61 GB.
- **F32 weight inflation (×4)** — `resources.ts:127-130` sizes weights at F32
  regardless of the int8 runtime tier, so the monolithic gate (`resources.ts:167`)
  can never pass.

That frozen `layers_per_stage=1` is threaded into every per-bucket decode recompile
(`crates/hologram-ai/src/staged.rs:2002` `decode_stages_for_bucket` →
`compile_chunk_stages`), and each window growth builds a **fresh runner with an
empty resident set** (`crates/hologram-ai/src/decode.rs:1137` `grow()` →
`wire_runner`, `staged.rs:1675`), so the first pass re-materializes ALL 34 stages.

**Arithmetic proving staging is unnecessary at chat context** (STRUCTURAL_CEILING
4 GiB, HOST_HEADROOM 1 GiB, `lib.rs:1237,1249`): at bucket 512, KV = 28 MB, budget
≈ 2.97 GB, int8 weights ≈ 1.9 GB → the model fits resident. Even at 32768, the
correct DECODE staging (chunk=1 activation, KV charged separately) is ~4–5 stages,
not 30 — the 30 comes entirely from reserving a 32768-token *prefill* activation in
the decode plan.

### ② Serial attention/softmax below the pool tile floor

The GEMV pool parallelizes only the `m==1` weight GEMV (`apps/web/src/holo.ts:66`),
~constant ~34 ms/token. Attention pools only above the substrate's 256 KiB f32 tile
floor (`holo.ts:121`), which one query row's scores don't cross until context ≈ 5 K,
so at chat contexts attention + the softmax reduction run **serially**. The pool
speedup therefore collapses with context (3.09×@128 → 1.14×@32 K) — pooled work is
fixed, serial work grows. Next ceiling after ①.

### ③ Eviction resurrects the O(context) KV re-hash

The fused κ120 in-place KV move is O(new rows) **only while a stage stays resident**.
When ① forces eviction, the carry is banked to host bytes (`runner.rs:377`, an
O(bucket) `resolve().to_vec()`) and **re-ingested via BLAKE3 over the whole cache**
next token (`staged.rs:1096` → `runner.rs:301`) — the ~28/110/442 ms/token @2K/8K/32K
tax v0.10.0 was meant to kill, riding along whenever the model streams stages.

### ④ Minor per-token waste
- **Double O(vocab) logits copy** — `runner.rs:352` (`resolve().to_vec()`) then
  `decode.rs:491` (`le_f32_vec`) → Qwen V=151936 ≈ 1.2 MB copied twice/token.
- **Per-token port-metadata clones** — `input_port_info()`/`output_port_info()`
  clone `Vec<PortInfo>` (String+Vec each) every token (`decode.rs:461,475`,
  `runner.rs:276,322`, `staged.rs:1264`). Ports are static for the session.

### ⑤ CORRECTNESS — `unreachable` on staged window-growth with resident-KV carry

Recurs at the 256/512 growth on a *staged, footprint-bounded* model carrying
resident-KV across multi-turn context. The `grow()` over-commit guard
(`decode.rs:1172-1178`: drop seeder + `evict_resident()` BEFORE `rebuild`) and the
KV-widen migration (`decode.rs:1197-1210`, bounds-safe) are present. The gap:
**no test drives a bucket regrow with the resident-KV carry on a staged,
footprint-bounded model under a realistic (non-∞) budget at head_dim 128** — the
exact deployed config. `decode_growth_residency.rs` uses `set_residency_budget(MAX)`
(nothing evicts) and checks only accounting; `parametric_reference.rs:602` checks
grow correctness but on a monolithic runner.

## The plan (parametric, by tier)

### ⚠ Refinement (found while implementing) — the F32 execution transient constrains staging

`stage_transient_bound` (`staged.rs:484`) = `3·weight_bytes + 8·elements`: a stage's
walk transiently widens its panel to **two full F32 images** (`8·elements`). For int8
(`weight_bytes ≈ elements`) that is ~11× the packed weight. A 13-layer int8 stage
(~611 MB packed) would transiently need ~6.7 GB — over the 4 GiB ceiling. So **fine
staging is partly REQUIRED by the F32 transient, not only by the activation reserve**,
and admission reserves the largest single walk (`max_walk`, `staged.rs:1172-1173`)
against the shared ledger. Consequences for the plan:

- The re-materialization the log shows is dominated by **window GROWTH** (each doubling
  builds a fresh empty runner → re-pages all stages), NOT per-token within a bucket
  (stages do stay resident within a bucket). So **Tier 2 (resident-across-growth) is
  the larger real win; Tier 1 must not simply coarsen** — a coarser stage whose F32
  transient exceeds the ceiling would OVER-COMMIT → the same `unreachable`.
- Tier 1's per-bucket recompute must therefore choose the COARSEST granularity whose
  worst-stage transient (`stage_transient_bound`) still fits the ceiling AND whose
  resident set + `max_walk` fits the budget — a tighter constraint than "weights fit".
  The `set_expected_stage_bytes` footprints are load-bearing and must be validated
  against the real model's measured stage bytes.

**Verification requirement (hard):** because a wrong granularity/footprint over-commits
into the exact trap, Tier 1/2 MUST be validated on the real 1.5B model in the browser
(measured stage bytes + no over-commit at each bucket). Native fixtures are monolithic
and cannot exercise this. Do NOT land Tier 1/2 on native tests alone.

### Tier 1 — recompute `layers_per_stage` per bucket (eliminates ① and most of ③)

The decode path already recompiles archives per bucket; make it recompute the
*granularity* too, from the current bucket's residency budget — so a model is
monolithic/coarse at chat context and re-stages finer only as context legitimately
approaches 32 K.

1. **`crates/hologram-ai/src/staged.rs` `decode_stages_for_bucket` (~:1994):**
   replace `self.layers_per_stage` with `self.decode_layers_per_stage(bucket)`.
   - `decode_layers_per_stage(bucket)`: if `residency_budget == 0` (native/unbounded)
     keep `self.layers_per_stage`; else `lps = max(1, min(L, budget_room /
     per_layer_int8))` where `per_layer_int8` is the tier-aware per-layer resident
     bytes and `budget_room` reserves the largest single-stage transient (mirror
     `stage_transient_bound`, `staged.rs:484`). Decode (chunk=1) activations are
     negligible and omitted.
   - **New helper `per_layer_resident_bytes()`**: sum the int8 artifact sizes
     (`self.quant` map + `self.shapes`) for one layer's projections (+ scales),
     and the embedding/head floors. This is the Rust twin of `weightDecomposition`
     at the *tier*, not F32.
   - **Cache key:** include the resolved `lps` in the `decode-archives:v2` key
     (`staged.rs:1973`) so a re-granularized plan is a distinct artifact.
2. **`apps/web/src/resources.ts` `estimateResources`:** make the COMPILE-TIME plan
   non-pathological too (the projection + the initial compile): size the activation
   reserve for the decode regime (chunk, not max context) and weights at the resolved
   tier. Add a `runtimeTierBytesPerParam` argument (1 for int8, resolved the same way
   as `QuantTier::optimal_for`). Keep max-context KV in the *budget* so the plan is
   safe at 32 K.
3. **Verify (native):** a new test forcing a small `residency_budget` that drives
   `decode_layers_per_stage` across buckets 64→…→32768, asserting (a) coarse/monolithic
   at small bucket, finer at large, (b) each stage's predicted footprint ≤ budget,
   (c) decode output bit-identical to the frozen-granularity path. Plus `resources.test.ts`
   for the estimate. **End-to-end tok/s must be confirmed on Qwen2.5-1.5B in the
   browser / on the deployed build** — the native fixtures are monolithic regardless.

Expected: 34-stage re-page → monolithic/coarse resident; per-growth re-materialization
from ~18 s to near-zero within a bucket; the eviction KV-re-hash (③) disappears at
chat context because stages stay resident.

### Tier 2 — don't re-materialize on growth
On `grow()`, migrate the resident stage sessions to the new bucket instead of
`wire_runner`'s empty set where the compiled graph is compatible; when eviction is
genuinely needed, bank/re-ingest the KV carry **by κ-label**, not by re-hashing bytes
(`runner.rs:377`/`staged.rs:1096`).

### Tier 3 — per-token micro-overhead (native-verifiable, low risk)
- Sample over the raw logit buffer — drop the second O(V) copy (`decode.rs:491` feeds
  `generate.rs:520` a `Vec<f32>`; sample over `&out.bytes` directly). Byte-identical.
- Cache `input_port_info()`/`output_port_info()` once per session, not per token.

### Tier 4 — attention (substrate/upstream)
Serial softmax reduction below the tile floor is the ceiling after Tier 1. A
flash-attention-style fused reduction, or lowering the attention pool threshold —
file an upstream ask against the substrate (κ119).

### Correctness gate (parallel, native-verifiable)
Add the missing hermetic test: a staged, footprint-bounded bucket-regrow
(64→128→256) with the resident-KV carry at head_dim 128 under a realistic budget —
reproduces and locks the `unreachable`. Re-verify Qwen2.5-1.5B on the deployed
v0.10.0.

## What is already optimal (audited — do not touch)
O(vocab) sampling (`generate.rs:309`), incremental detokenization
(`hologram-ai-tokenizer/src/streaming.rs`), delta streaming
(`generate.worker.ts:695`), incremental decode mask (`decode.rs:349`), the fused
κ120 in-place KV move on the resident path (`runner.rs:288`), pool worker count
parametric to cores (`holo.ts:124`).

## Verification constraint (honesty)
The dominant win (Tier 1) changes the core decode-residency path; its *real* payoff
is only observable on the large model that exhibits the problem (Qwen2.5-1.5B in the
browser). Native tests verify the granularity logic + decode correctness but are
monolithic at fixture scale, so they cannot measure the tok/s win. Land Tier 1 with
native correctness gates AND a deployed re-measure, never on native tests alone.
