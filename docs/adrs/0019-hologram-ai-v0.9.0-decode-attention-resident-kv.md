# ADR-0019: Adopt v0.9.0 fused decode-attention + resident KV (κ-move)

**Status:** Proposed — adoption gated on the v0.9.0 release (upstream PR #41,
in-flight). This ADR is the plan to *adopt*, not to reimplement; the primitives
are the substrate's.

**Date:** 2026-07-12. **Branch:** `feat/wasm-threads` (decode work continues here).
**Supersedes the "our-side capture" sketch in** `docs/notes/throughput-latency-analysis.md`
§2b / lever G — v0.9.0 provides the primitive, so we adopt rather than hand-roll a
ring-export against `execute_addressed`.

## Context

`docs/notes/throughput-latency-analysis.md` established that the GEMV worker pool
(ADR-0018) removes the *short*-context bottleneck but the pool speedup collapses
with context (3.09×@128 → 1.14×@32768 on Qwen2.5-1.5B) because attention, the KV
recopy, and — the finding of the 2026-07-12 pass — a **hidden per-token BLAKE3
re-hash of the resident K/V** are serial and O(L). Three substrate contracts, all
measured:

1. **Mask.** `AttentionCall.causal: bool` cannot express "attend to the first
   `realized` of `bucket` rows" (realized length is a *runtime* value over a fixed
   padded bucket).
2. **KV recopy.** The single-`k` kernel signature forces our `decode_plan.rs` to
   emit `Concat(past_k, k_new)` + `Transpose` over the whole bucket every step —
   ~440 ms/token at 32K (1.5B).
3. **KV re-hash.** Carrying `past_k`/`past_v` as host bytes through the byte
   `execute` BLAKE3-hashes the *entire* cache every token — a second O(bucket)
   cost `pool-bench` never saw because it never crosses the byte boundary.
   Measured (`examples/kv_rehash_cost.rs`, commit `4ebad94`; native SIMD, the
   optimistic bound): **1.5B 28 / 110 / 442 ms/tok @2K / 8K / 32K**.

Upstream **PR #41 (`feat(backend)!: fused, pooled, KV-cache-aware masked decode
attention (v0.9.0)`)** fixes all three as substrate primitives. Its own body cites
this repo's three-gap finding. The `!` marks a breaking change.

## The v0.9.0 contract we adopt

Extracted from the PR's public API snapshots (`api/hologram-{backend,ops,compiler,
exec}.txt`) and its executor-level witnesses (`crates/hologram-exec/examples/
addressed_decode_timing.rs`, `tests/decode_attention_e2e.rs`, `tests/kv_cache_write.rs`
— our byte-identical oracles).

### Fused decode attention — `OpKind::Attention`, 6 inputs (gaps 1 + 2)

A graph node `Attention(q, k_past, v_past, k_new, v_new, mask)` lowers to
`DecodeAttentionCall` (κ discriminant **119**):

- **Split KV.** `past ∥ new` is iterated *in place*; the concatenation is never
  materialized. The recopy is deleted **by construction** — not optimized away.
- **Required additive mask** `[q_rows, past+new]` f32: `-inf` erases a key exactly
  (`exp(-inf)=0`). The κ split is explicit — the *bucket* is structure in the
  signature; the *realized length* is content in the mask operand's bytes. This is
  the one masking authority: a 6-input node that also sets `AttentionAttrs::causal`
  is **refused at compile time** (`CompileError::GraphValidation`).
- **Pooled** on `(batch, head, query-row)` like the GEMMs (wasm fork-join,
  publisher-carried scratch; native row tiles).
- Decode (`m=1`), chunked prefill (`m=C`), and speculative verify (`m=K`) all ride
  the one kernel.
- `DecodeAttentionCall{ q, k_past, v_past, k_new, v_new, mask, output, past_len,
  new_len, q_rows, heads, kv_heads, head_dim, batch, dtype, scale_bits }`.

### Resident KV — `OpKind::KvCacheWrite` κ-move (gap 3)

A graph node `KvCacheWrite(cache, new, pos)` lowers to `KvCacheWriteCall{ cache,
new, pos, output, bucket_rows, new_rows, planes, row_bytes }` (κ discriminant
**120**):

- Fixed-bucket row write at a **runtime `pos`** operand (4 bytes, ring wrap) — so
  **one** compiled step-graph serves every step; the position is content, not
  structure.
- The kernel's contract is an honest O(bucket) copy, dtype-agnostic, validated.
- The **executor** realizes an *eligible* write as a κ **move**: the old cache
  label is retired (a moved value is never re-addressed), the buffer is mutated at
  **O(new_rows)**, and the result is retained under the derived output label — the
  decode loop binds it next step with **no hash, no copy**. Eligibility is proven,
  not assumed:
  - **Load-time analysis** re-derives it from the decoded plan (every other toucher
    of the cache slot scheduled strictly earlier); the compiler defers output-only
    writes to a trailing schedule level so real decode graphs qualify. A hand-built
    archive cannot spoof it.
  - **Steal-time ownership check**: the pool declines unless the buffer is owned by
    exactly this node's operands — view aliases, duplicate-label ports, pinned/lazy
    tiers all fall back to the honest copy.

### Executor + compiler surface

- Addressed loop: `intern_input(bytes) -> ContentLabel`, `execute_addressed(&[labels])
  -> Vec<ContentLabel>`; the cache labels ride `out[k_idx]`/`out[v_idx]` forward.
- `BufferArena::bindable_input(&ContentLabel) -> bool` + hardening: a refused
  `execute_addressed` validates bindability **before** any state change, so a
  failed call cannot age out the resident-KV labels a retrying decode loop still
  needs.
- Compile with `WittLevel::W32` (our 32-bit production law). `lower::ShapeArgs`
  gained `q_rows, past_len, new_len, kv_bucket_rows, kv_new_rows, kv_planes,
  kv_row_bytes`.

### κ-leases — residency by ownership (PR #41 head `f2f864b`)

The "κ-leases, pinned total, flow law, exact confinement" commit adds the residency
primitive our driver + residency ledger need — the substrate applying the UOR
driftless-torus discipline (`docs/numerics/invariance.md`) to the pool:

- `InferenceSession::retain_label(&ContentLabel) -> bool` / `release_label(...) ->
  bool` — **residency by ownership, not recency**: a leased value survives every
  walk until released (refcounted). The **ownership law**: a lease is a *borrow*,
  so `KvCacheWrite` on a leased cache **declines the in-place move and takes the
  honest copy** — the leased pre-image survives bit-intact. Releasing the last
  lease restores the move.
- `InferenceSession::pool_allocated_bytes() -> usize` — **the exact confinement
  metric.** A steady-state decode loop holds this *constant* (O(1) memory/step);
  we read it directly for the residency ledger instead of host-side estimation
  (the source of three prior over-commits). `leased_count()` for lease bookkeeping.
- This is precisely the two things our decode needs beyond one step's outputs:
  **speculative rollback** (lease pre-state → draft step by honest copy → accept
  ⇒ `release_label` and the next step moves; reject ⇒ re-step from the intact
  pre-image) and **draft-pairing** (a second session's KV parked across the main's
  walks — our `share_residency_with`, now a lease). Contract witnessed upstream by
  `tests/lease.rs`, `tests/confinement.rs`, `tests/flow_law.rs`.

### The invariance ladder — what "byte-identical" we may claim

Per `docs/numerics/invariance.md`, our witnesses obey the rung law:
- **Integer paths** are codec-invariant, schedule-independent, **and cross-lane
  machine-invariant** (exact i32 accumulation).
- **f32 within one lane** is structure-pinned bit-identity: pooled == serial,
  **split-KV == precatenated**, padded == tight, **moved == copied**, chunked ==
  sequential — all exact **on one target**. Our fused-vs-legacy decode witness is
  an f32 path, so it runs **natively on a single lane**; it is NOT a cross-lane
  claim.
- **f32 across lanes is a non-claim** (wasm SIMD128 has no FMA). No witness asserts
  cross-lane f32 bit-identity.

### The step graph (per layer), from `addressed_decode_timing.rs`

```
inputs:  q, k_cache, v_cache, k_new, v_new, mask, pos
attn  =  Attention(q, k_cache, v_cache, k_new, v_new, mask)   # -> output rows
k_out =  KvCacheWrite(k_cache, k_new, pos)                    # -> updated cache
v_out =  KvCacheWrite(v_cache, v_new, pos)                    # -> updated cache
outputs: attn, k_out, v_out
```

Driver: intern the caches once; each step intern only the small operands
(`q, k_new, v_new, mask, pos`), `execute_addressed`, carry `out[k_out]/out[v_out]`
labels into the next step. The O(bucket) K/V bytes are never re-hashed and never
copied.

## Decision

Rework hologram-ai's decode stack to **emit and drive the v0.9.0 form**. The host
stops owning K/V bytes; the cache is substrate-resident and updated by a κ-move.

## Consequences — the hologram-ai changes

### 1. IR + lowering (`crates/hologram-ai-common`)

- Add `AiOp::Attention` (the 6-input masked form) and `AiOp::KvCacheWrite` to the
  op IR, lowering to `OpKind::Attention` (6 inputs) and `OpKind::KvCacheWrite`.
  Thread the new `ShapeArgs` (`q_rows, past_len, new_len, kv_bucket_rows,
  kv_new_rows, kv_planes, kv_row_bytes`) from node shapes.
- Enforce our side of the single-authority rule: never set a `causal` attr on the
  6-input node (the substrate refuses it; we must not emit it).

### 2. `rewrite_decode_attention` (`opt/decode_plan.rs:282`)

Replace the decomposition — `Concat(pk,kn)` (`:429`), `Transpose` (`:432`),
`Concat(pv,vn)` (`:449`), the softmax-accumulate `Concat` (`:462`) and terminal
`Transpose` (`:477`) — with, **per layer**:

- one `AiOp::Attention` over `(q_roped, k_cache, v_cache, k_new_roped, v_new, mask)`;
- two `AiOp::KvCacheWrite` over `(k_cache, k_new_roped, pos)` and `(v_cache, v_new,
  pos)`, whose outputs are the layer's next cache.

`past_k`/`past_v` (`:370-371`, `[kv, bucket, dh]`) become **resident cache in/out
ports** rather than re-supplied host inputs. Add a single shared `pos` input
(`rank-1`, i32). Keep RoPE as-is (still runtime data). The `DECODE_MASK_PORT` mask
(`:360`) stays but is now the **sole** masking authority in the exact
`[q_rows, past+new]` additive shape the kernel requires (it already erases
unrealized rows + does causal-within-chunk; confirm the layout matches).

### 3. Decode driver (`crates/hologram-ai/src/decode.rs`)

- `DecodeState` carries **per-layer cache `ContentLabel`s** (`k_label[l]`,
  `v_label[l]`) instead of `past_k`/`past_v` `Vec<Vec<u8>>` (`:179-180`) and the
  host splice (`:387-392`).
- `pass()` (`:339`) drives `runner.execute_addressed(&labels)` (not `execute`,
  `:351`): intern the small per-step operands, bind the resident cache labels,
  advance `pos` (a 4-byte operand = `cur_len`), read back the updated cache labels.
- `feed()` (prefill, `m=C`) and `verify`/`draft_verify` (`m=K`) ride the same
  graph; verify does not advance the cache (bind the same labels, discard the
  write outputs) — the split-KV / KvCacheWrite eligibility must still hold.
- Extend `LmSession` (`engine.rs`) with the addressed method (both `HoloRunner`
  and `StagedRunner` already reach `execute_addressed`/`intern_input`).
- Use `bindable_input` to pre-validate before `execute_addressed`, matching the
  substrate hardening, so a mid-turn refusal cannot strand cache labels.

### 4. Residency ledger under the 32-bit law ([[no-heuristics-parametric]])

This is the crash-prone surface (three prior over-commits — see the project-state
memory). The change is a *net simplification*:

- The resident cache lives in the substrate `BufferArena`, **O(bucket) fixed** per
  layer × 2 planes — the κ-move mutates in place, so it does **not** grow with
  context and the `Concat` no longer doubles it transiently. **Read
  `pool_allocated_bytes()` for the ledger** — the exact substrate-reported resident
  footprint — instead of host-side estimation (the estimation error is what
  over-committed three times). The host `Vec` bytes it replaces are removed from
  the ledger.
- **Speculative verify / draft-pairing use κ-leases, not the host splice.** Verify
  (`m=K`) leases the pre-image cache labels, drafts by honest copy (pre-image
  intact), and on accept `release_label`s so the accepted step moves; on reject it
  re-steps from the intact pre-image — the substrate ownership law replaces our
  host-side accept/reject bookkeeping. Draft-pairing leases the parked second-model
  KV (superseding the ad-hoc `share_residency_with`).
- **Bucket-growth** (`decode.rs grow()`) simplifies: `KvCacheWrite` is fixed-bucket
  with ring wrap, so growth still rebuilds at a wider bucket, but it now evicts +
  re-interns the resident cache labels (retire old, intern widened) rather than
  reallocating and re-hashing host buffers — the same "evict-before-rebuild"
  discipline that fixed the 3rd crash (`decode_growth_residency.rs`), applied to
  labels.

### 5. wasm glue (`crates/hologram-ai-wasm`) + web

- The decode loop (`lib.rs:1888`) calls the addressed path. Prewarm/seeder
  lifecycle (ADR-0018) is unchanged in shape; the seeder rides the same fused
  graph at `m=C`.

## Verification — the honesty gate (fails-without witnesses)

No claim ships without a witness that fails if the claim is false ([[dark-gates]]).

1. **Byte-identical migration.** New fused-form decode logits == the legacy
   `Concat/Transpose/softmax` decode logits, bit-for-bit, per family (Llama /
   Qwen2 / Mistral / Phi3). This is the load-bearing "we changed the schedule, not
   the numbers" witness — extend `decode_family_coverage.rs`. **Rung 2: run on one
   native lane** (f32), never a cross-lane claim.
2. **Addressed == byte.** `execute_addressed` caches == `execute` caches bit-for-bit
   (mirror the substrate's `tests/kv_cache_write.rs` at our layer): the
   `KvCacheWrite` move produces the same bytes as the honest copy.
3. **Confinement (residency no-over-commit).** `pool_allocated_bytes()` is
   **constant** across a steady-state decode loop (O(1)/step) and the ledger never
   over-commits the 4 GiB space across seeder + step + verify + a bucket-growth
   transition — **fails-without** the eviction/lease discipline (spirit of
   `decode_growth_residency.rs` + the substrate's `confinement.rs`).
4. **Lease rollback.** Speculative reject re-steps from a bit-intact leased
   pre-image; accept releases and the next step moves (`last_dispatched()` flips
   copy→move) — the accept/reject primitive, witnessed at our layer.
5. **Re-hash removed, measured.** A driver-level bench (extend
   `kv_rehash_cost.rs` / port `addressed_decode_timing.rs`) showing the per-token
   O(bucket) hash is gone on the addressed path.
6. **Single masking authority.** We never emit a `causal` attr on the 6-input node
   (assert at emit); the compiler would reject it anyway.

## Pin-flip readiness

- Repin the 8 `hologram-*` deps from `f031e8b` (v0.8.2) to the **v0.9.0 tag** on
  release (or PR head `08ec60b` for pre-merge prep — a moving target; flip to the
  tag before any deploy). An explicit `rev`/tag cannot mis-resolve against a stale
  git db (the ADR-0018 pin trap).
- **Preserve the Cargo.lock split** with `holospaces` → `hologram-backend@18f553d`:
  the shared `hologram-substrate-core` / `hologram-realizations` must still resolve
  after the bump. Verify the 8×v0.9.0 + 12×18f553d split holds.
- The PR is a `!` breaking change: **dry-pin-check the workspace build** against the
  v0.9.0 rev *before* adopting the new ops, to surface the breaking surface early
  (our decode currently uses only primitive ops, so it may build unchanged).
  **Verified 2026-07-12** (PR head `08ec60b`, throwaway pin, reverted): the lock
  resolved our 8 deps to **v0.9.0** while `holospaces`' 12 stayed at `0.6.0`/`18f553d`
  (split intact, 8 + 12), and `cargo check --workspace` exited **0** — v0.9.0's
  breaking change does not break our current build, so adoption is additive on a
  green workspace.

## Adoption findings (2026-07-12, the flip)

Three findings from turning the fused path on in production, all resolved:

1. **Scale parity.** The κ119 kernel applies its scale as `dot / (1/sm)`
   (`scale_bits` = the multiplier `sm`; `0` ⇒ `√d` divisor), while the legacy
   decomposition pre-folds `1/√d` into q — `x·(1/√d)` ≠ `x/√d` in the last
   ulp. The fused emission now always folds the model's scale (declared or the
   `1/√dh` default) into q with the identical `Mul` legacy emits and declares
   `AttentionAttrs { causal: false, scale_bits: 1.0 }` (a new
   `AttrSpec::Attention`), making the kernel's own scaling an exact no-op —
   one scale placement for both forms, any model, any head_dim.
2. **Family-gate methodology.** The int8/int4 family gates compared the final
   logits of two *autonomous* generations; a single knife-edge argmax flip on
   a near-tied pair (a legal quantization outcome — observed on Phi3 when the
   fused kernel's equally-valid f32 schedule moved a tie by an ulp) diverges
   the contexts and collapses the cosine, gating on sequence luck. The gates
   now **teacher-force** the quantized leg on the bf16 reference tokens and
   assert per-position logit cosine under a **shared context** at every
   generation step — strictly stronger as a tracking measure and robust to
   ties. Measured post-fix: int8 ≥ 0.9994/position for every family (Phi3
   0.9996–0.9997); int4 0.958–0.988/position (the earlier ≈0.66 "model-level"
   figure was divergence-amplified, not numeric error).
3. **Substrate concurrency quarantine.** v0.9.0's pooled decode-attention
   publisher scratch is unsound across *concurrent* `InferenceSession` walks
   in one process — `RefCell already borrowed` at best, silent numeric
   corruption at worst (`upstream-issue-v090-pooled-decode-scratch.md`).
   Production already drives walks sequentially; `HoloRunner::execute{,_addressed}`
   now take a process-global `walk_lock()` making that contract explicit —
   concurrent callers (parallel tests, future servers) serialize instead of
   corrupting. Uncontended cost ≈ ns against multi-ms walks.

## Risks / honest framing

- The driver + residency changes touch the code that crashed three times. Mitigated
  by witness 3 (fails-without) and by the fact that the resident cache is
  *fixed*-size (no growth doubling) — but it is real surgery, gated on green
  witnesses, not merged blind.
- Absolute wasm tok/s is **deploy-measured** (the 4-core codespace is too contended
  for browser wall-clock); the witnesses prove correctness + residency + the
  removed hash, and the deploy confirms the speedup.
- Adoption is gated on v0.9.0 landing; until then this ADR is the plan and the
  upstream witnesses are the contract.
