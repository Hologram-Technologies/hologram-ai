# Plan 045: Variable-Length Execution Fix

**Status:** Open
**Created:** 2026-04-02
**Branch:** `feat/cpu-inference-perf`

## Problem

Models must be compiled with `--seq-len N` matching the exact prompt token count.
If compiled at seq=32 but run with 24 tokens, output is garbage because:
- Reshape/Expand ops use compiled dimensions (32)
- Softmax/RmsNorm use inferred dimensions from buffer length (24)
- This inconsistency corrupts shapes mid-graph

## Root Cause

`resolve_size()` in `float_dispatch/mod.rs` falls back to `n_floats` when
`n_floats % compiled_size != 0`. But some ops (Reshape, Expand) embed the
compiled seq_len in their parameters and don't use `resolve_size()` at all.

## Fix Strategy

### Option A: 0-sentinels for all seq-dependent dims (recommended)

During lowering, emit `0` for ALL dimensions that depend on sequence length:
- `FloatOp::MatMul { m: 0, k, n }` — m is seq-dependent
- `FloatOp::Reshape` target shape — seq dimension is 0
- `FloatOp::Softmax { size: 0 }` — when size is seq-dependent

The executor resolves 0 → infer from buffer length. All ops use the same
inference, so dimensions stay consistent.

**Changes:**
- `hologram-ai-common/src/lower/strategy.rs` — track which dims are seq-dependent
  from the `DimVarTable`, emit 0 instead of concrete values
- `hologram-exec/src/float_dispatch/mod.rs` — improve `resolve_size()` heuristics
- `hologram-exec/src/tape.rs` — ensure all kernels handle 0-sentinel dims

### Option B: Shape context projection (existing infrastructure)

The `ShapeContextGraph` already maps compiled shapes to runtime shapes.
Wire it into `execute_direct` so every instruction's output gets correct
shape metadata. The old `execute_inner` used `shape_overrides` — port this
to the new single-path executor.

## Testing

- Compile TinyLlama at seq=64, run with 24-token prompt → correct output
- Compile at seq=128, run with 10 tokens → correct output
- KV cache decode still works (seq=1)
- Non-LLM models (BERT, ResNet) unaffected
