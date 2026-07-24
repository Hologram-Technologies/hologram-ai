# Upstream issue: v0.9.0 pooled decode-attention scratch is unsound across concurrent sessions

**Status: OPEN (found 2026-07-12, during the ADR-0019 adoption).**
**To:** the hologram substrate (Hologram-Technologies/hologram, v0.9.0 = `22b0ce1`).
**Our-side quarantine shipped:** a process-global walk lock in `HoloRunner`
(`runner.rs::walk_lock`) serializes every `execute`/`execute_addressed` — see below.

## Symptom

With two or more `InferenceSession`s executing **concurrently in one process**
(e.g. `cargo test`'s parallel test threads, each driving its own sessions), the
v0.9.0 fused decode attention (κ119) fails in one of two ways:

1. **Panic**: `RefCell already borrowed` at
   `crates/hologram-backend/src/cpu/float_kernels.rs:64:47` — the pooled
   decode-attention's publisher-carried score scratch is re-entered.
2. **Silent corruption** (worse): the walk completes but the numbers are wrong
   and non-deterministic — observed as decode token sequences changing
   run-to-run (`[404, 288, 1125, 1125]` vs `[1487, 699, 1125, 1125]` for the
   same model + prompt) and logit cosines degrading, only while another
   session executes concurrently.

Sequential execution — any number of sessions, one walk at a time — is
correct and deterministic (all our single-test runs pass bit-identically).

## Repro

`hologram-ai` @ ADR-0019 (fused decode production path), then:

```
cargo test -p hologram-ai --test decode_family_coverage
```

The binary's three tests each build decode sessions and run in parallel
threads (cargo default). One or more fail with the panic or the
reproducibility assertion. `--test-threads=1` (or any serialization of the
walks) is reliably green.

## Why it matters

The scratch reuse is presumably per-thread by design (publisher-carried,
workers never allocate — the v0.9.0 PR's stated model), but the worker pool is
**process-global** while sessions are not: two concurrent publishers share the
pool, and the steal/carry paths appear to cross session scratch. Anyone
driving hologram from a multi-threaded host (parallel test harnesses, a
server handling two requests) hits it — and the corruption mode is silent.

## Our-side quarantine (shipped)

Production drives every session sequentially already (one browser worker; one
CLI generation loop), so the contract "one walk at a time per process" was
already true in deployment. `HoloRunner::execute`/`execute_addressed` now take
a process-global mutex, making the contract explicit and load-bearing: any
concurrent caller serializes (correct, slightly slower) instead of racing
(wrong). Uncontended cost is nanoseconds against multi-millisecond walks;
wasm32's single-threaded driver never contends.

## Ask

Make concurrent sessions either **safe** (per-publisher scratch keyed by
session, or pool-level isolation between concurrent walks) or **loudly
refused** (a documented single-walk contract enforced with a clear error).
Silent cross-session corruption is the one unacceptable outcome. Happy to
contribute the repro as a substrate test.
