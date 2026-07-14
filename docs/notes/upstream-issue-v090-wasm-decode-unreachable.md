# Upstream issue: v0.9.0 fused decode attention traps `unreachable` in wasm at production scale

**Status: OPEN (found 2026-07-14 via the deployed instance).**
**To:** the hologram substrate (Hologram-Technologies/hologram, v0.9.0 = `22b0ce1`).
**Our-side mitigation:** the browser build falls back to the legacy decode
decomposition (see below); native keeps the fused resident-KV path.

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

- **Native is correct at the same shape.** `hologram-ai`'s
  `decode_family_coverage` drives the int8-quantized STAGED decode pipeline at
  head_dim 128 for every family, multi-step, INCLUDING under stage-eviction
  pressure (a 1-byte residency budget that drops and re-materializes every stage
  between steps) — bit-identical, reproducible, no trap. So the graph, the
  resident-carry protocol, and the decode math are sound.
- The divergence is native-vs-**wasm**: the substrate's wasm decode kernels
  (`decode_attention_float` / the `KvCacheWrite` in-place move, and their SIMD
  inner loops) are the only thing that differs between the passing native run
  and the trapping wasm run at head_dim 128 on the carried-past step.
- `RuntimeError: unreachable` is a wasm trap — a Rust panic-to-abort or an
  explicit `unreachable`. It is NOT surfaced through our panic hook
  (`console_error_panic_hook`/`last_panic`), which suggests the trap does not
  unwind as a normal Rust panic on the wasm target (panic=abort), or it
  originates below the hook.

## Repro

The deployed real model reproduces it every turn. A smaller hermetic repro is
in progress (`apps/web/bdd/deep_model_journey`: a synthesized head_dim-128,
int8, staged fixture generating multiple tokens so it reaches the second
decode step). Happy to contribute a substrate-level wasm test at head_dim 128
that exercises the carried-past `DecodeAttention` + `KvCacheWrite` move.

## Ask

Make the fused decode path (κ119 `DecodeAttention` + κ120 `KvCacheWrite` move)
sound on the wasm32 target at production head_dim on the resident-carry step,
or fail LOUD with a diagnosable `BackendError` (never an opaque `unreachable`).
A wasm-target test at head_dim 128 with a non-empty carried past would lock it.
