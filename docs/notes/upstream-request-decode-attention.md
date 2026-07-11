# Upstream request: a pooled, KV-cache-aware, masked fused decode-attention kernel

**Status: OPEN (filed 2026-07-11).** Follows the same finding→request→fix loop as
`upstream-request-prefill-pooling.md`, which shipped as substrate **v0.8.2**.

**To:** the hologram substrate (Hologram-Technologies/hologram, `hologram-backend`).
**From:** hologram-ai (ADR-0018 follow-on; the throughput/latency analysis in
`throughput-latency-analysis.md`).

## Ask

Give decode attention a **single fused kernel call per layer** that (a) reads the
**resident KV cache in place** (no per-step recopy), (b) accepts an **additive
mask** (or a runtime realized-length) so a fixed padded bucket works, and (c)
**pools** across the worker pool like the GEMV/GEMM already do. Concretely, extend
`AttentionCall` (`kernel_call.rs:576`) so `hologram-ai`'s decode rewrite can lower
`GroupedQueryAttention` to it at `m = 1` (and `m = C` for chunked prefill).

The kernel exists (`attention_float`, `attention_w8`, `AttentionCall`) but at
decode we cannot use it, so `decode_plan.rs` (ours) decomposes each GQA node into
primitive ops. Three gaps block the fully-resident decode path:

1. **Masking.** `AttentionCall` offers only `causal: bool`. Decode uses a fixed
   padded past-bucket whose *realized length is a runtime value*; visibility is an
   additive `decode_mask` `[g·C, bucket+C]` that erases unrealized rows AND does
   causal-within-chunk. A `causal` bool over a fixed `seq` cannot express "attend
   to the first `realized` of `bucket` rows." **Add an optional additive-mask
   operand, or a runtime `valid_len`.**
2. **KV-cache append + residency.** The kernel takes one `k`/`v`. At decode the
   keys are `[resident past ∥ this step's new row]`. With no in-graph scatter, our
   rewrite emits `Concat(past_k, k_new)` + `Transpose` **every step** — it recopies
   (and transposes) the *entire* bucket to append one row. **Let the kernel take
   `past_k`/`past_v` + `k_new`/`v_new` (or read a resident cache with a write
   position), so the cache is read once, never recopied.**
3. **The K/V is re-hashed on the byte boundary every step — a SECOND O(bucket)
   cost, comparable to the recopy, that our benchmark never measured.** Our decode
   driver carries `past_k`/`past_v` as host `Vec<u8>` and runs them through the
   byte `InferenceSession::execute`, which content-addresses every input via
   `address_bytes` (BLAKE3). The per-port `input_cache` always misses on decode
   (the substrate's own comment: *"autoregressive decode (input changes every step,
   always a miss)"*), so the **entire** carried K/V is BLAKE3-hashed every token.
   Measured (`examples/kv_rehash_cost.rs`, native SIMD — the optimistic bound;
   deployed wasm32 hashes slower): **1.5B ⇒ 28 ms/tok @2K, 110 @8K, 442 @32K**;
   **7B ⇒ 55 / 225 / 884**. This is ON TOP of the kernel step `pool-bench` reports
   (which never crosses the byte boundary), so at 8K it ~doubles the step and at
   32K it adds a second ~440 ms. The substrate **already exposes the fix** and
   recommends it here — `execute_addressed` (`session.rs:874`) binds inputs by
   κ-label via `bind_resident`, *"nothing is rehashed … no hashing, no copy"*, and
   `HoloRunner::execute_addressed`/`intern_input` already re-export it. But there
   is **no append-in-place op on a resident KV**, and our cache is a fixed-bucket
   ring the host mutates each step, so we cannot feed the addressed path without
   either (a) this request's resident cache with a write position — which lets the
   updated cache be **retained by label** and bound next step with no hash — or (b)
   a graph+driver restructure that exports the updated `[bucket, dh]` cache as a
   retained output label per layer (the ring-write moved into graph ops). **(a) is
   this request; (b) is the our-side capture, tracked in the analysis note.**

## Why (measured)

`apps/web/scripts/pool-bench.rs` (`pnpm bench:pool`) decomposes the per-token step.
The GEMV pool (ADR-0018) is fixed in context; attention + KV-recopy are serial and
grow with context, so the pool speedup **collapses** and throughput with it —
Qwen2.5-1.5B, 4 participants:

| context L | tok/s | pool speedup | KV-recopy (ms/token) |
|---|---|---|---|
| 128 | 26 | 3.09× | ~0 |
| 8 192 | 7.7 | 1.66× | 19 |
| 32 768 | 1.3 | **1.14×** | **440** |

At 32K the `Concat`+`Transpose` recopy of the bucket is ~440 ms/token (1.5B) and
the QK^T/softmax/P·V run single-threaded — the weight GEMV is a rounding error.
This is the same story for every model (7B: 2.45×@128 → 1.15×@32768).

## Why the embedder cannot fix it

The recopy comes from `Concat(past_k, k_new)` — forced by the stateless,
content-addressed graph (no scatter op) plus `AttentionCall`'s single-`k`
signature. The custom mask is forced by `AttentionCall`'s `causal`-only masking
over a fixed `seq`. Both are substrate contracts; `decode_plan.rs` decomposes GQA
precisely *because* it cannot express the decode case through the fused kernel. We
can (and will) shave constant factors our-side, but eliminating the O(bucket)
per-step recopy and pooling the attention math needs the kernel.

## Impact

Removes the long-context throughput ceiling: the per-token cost drops to the
**inherent** O(context) KV *read* (bandwidth-bound, unavoidable for causal
attention) **divided across cores**, with no recopy, no serial softmax, and — via
a resident cache bound by label — **no per-step re-hash** (gap 3: another
~440 ms/tok at 32K on 1.5B removed byte-identically). That is the difference
between "caps out past a few K tokens" and "scales to arbitrary context at the
host's memory bandwidth" — the parametric, no-arbitrary-ceiling target. Decode and
chunked prefill share the one kernel; speculative verify (m=K) rides it too.

The re-hash is the UOR "a resident value is bound by its address, never
re-addressed" law applied to decode: the byte boundary re-hashes an object that is
already resident. `execute_addressed` is the substrate honoring that law; gap 3 is
the one place decode cannot yet reach it.
