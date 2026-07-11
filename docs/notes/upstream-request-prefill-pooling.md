# Upstream request: pool the m>1 (prefill) GEMM in the wasm worker pool

**Status: RESOLVED in substrate v0.8.2** (commit 0ac84a0 "pool the m>1 GEMM in the
wasm worker pool" + f031e8b "pool admission — work admits, width declines"). The
admission gate is now WORK-keyed (`m*k*n`) with a per-participant column-width
floor; decode admission (m=1) is bit-for-bit unchanged. Measured on 4 participants
(wasmtime): TTFT/prefill parallelises **3.2–3.7×** (e.g. Qwen2.5-1.5B first-token
8.9 s → 2.5 s; 7B 56.6 s → 15.3 s). This note is kept as the record of the
finding→request→fix loop.

**To:** the hologram substrate (Hologram-Technologies/hologram, `hologram-backend`).
**From:** hologram-ai (ADR-0018, multi-threaded browser decode).
**Date:** 2026-07-11.

## Ask

Extend the wasm embedder worker pool (`cpu/wasm_pool.rs`, `fork_join_gemv`) to
also parallelise the **m > 1** matmul path (prefill / batched verify), by the
same **output-column** partition it already uses for `m == 1` decode. Today the
pool is `m == 1` only; `matmul_i8_pc_omajor` (and the i4/e8cb variants) run the
serial cache-blocked kernel when `m > 1`.

## Why (benchmark)

ADR-0018 shipped multi-threaded browser **decode**: the m==1 GEMV pool gives
**3–5× on 4 participants**, driving decode to the host's memory-bandwidth ceiling
(`apps/web/scripts/pool-bench.rs`). But the same benchmark shows **time-to-first-
token is now the dominant cost for realistic models**, and the pool does not touch
it — prefill is the m>1 GEMM, which runs serially:

| model | TTFT (ms), prompt_len=128, serial prefill |
|---|---|
| Qwen2.5-0.5B | ~5 000 |
| Qwen2.5-1.5B | ~11 000 |
| Qwen2.5-3B   | ~36 000 |
| Qwen2.5-7B   | ~79 000 |

TTFT scales with prompt length and with model size, single-threaded. On a machine
with N cores, this is the one part of inference leaving N−1 of them idle.

## Why the embedder cannot fix it

Doing prefill as P separate pooled `m == 1` GEMVs reloads each weight P times — it
is **~10× slower** than the batched serial GEMM (this repo's measured "batched
prefill 10×" fact; the `decode_weights_fit_resident` seeder-skip was a 10× TTFT
regression). At 3–5× pool speedup that is still ~2–2.5× *slower* than serial
batched. So the batched serial GEMM is already the best the embedder can do; the
only lever is to parallelise the batched GEMM itself.

## Why it is a small, sound change

The existing `m == 1` partition (`pool_exec_gemv`, contiguous output-column ranges,
whole-output-per-participant, exact-i32 accumulation) already generalises to any
m: each participant computes its output-column range for **all m rows**, reading
its weight-column tile ONCE (no per-position reload) and the m activation rows.
Per-output reduction order is unchanged, so the bit-identity guarantee
(`parallel_gemv_matches_serial_bitwise`) extends to m>1 unchanged. The dispatch
sites in `simd.rs` would call `fork_join_gemv` (or a `fork_join_gemm`) in the m>1
branch above the same `POOL_MIN_WEIGHT_BYTES` floor.

## Impact

~3–5× lower TTFT for chat-scale models on multi-core browsers — turning the ADR's
"multi-threaded decode" into "multi-threaded inference". Decode already benefits;
this closes the remaining serial gap.
