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

## Status Update (2026-04-05)

### What Works Now
- **prompt <= compiled seq_len**: ShapeContextGraph + KV cache produces
  correct output when compiled at max context length (e.g., seq=2048) and
  prompted with any shorter sequence. Shape overrides flow through
  `input_metas` in `execute_direct`.
- **Compile at model context length**: Default compilation uses the model's
  full `context_length` (2048 for TinyLlama). Any prompt up to that length
  works with variable-length execution. This is the recommended path.

### What Was Attempted and Failed
- **Option A partial (Dynamic dims):** Setting seq-dependent dims to
  `Dim::Dynamic` after optimization breaks lowering — `concrete_last_dim`
  returns None for ALL seq dims, causing Softmax/RmsNorm/MatMul to get
  size=0 which the runtime can't resolve correctly.
- **Shape tensor i64 zeroing:** Zeroing `known_i64_values` at seq positions
  requires following Reshape data flow to map target tensor axes to shape
  tensor element indices. The axis-based `seq_dim_positions` set doesn't
  map to element indices of 1-D shape constants.

### Infrastructure Built (ready for use)
- `concretize_all_dims` now returns `seq_dim_positions: HashSet<(TensorId, usize)>`
  identifying which tensor dims are seq-dependent
- `retain_live_nodes()` on ShapeContextGraph prunes dead entries after fusion
- ShapeContextGraph wired into HoloRunner.execute/execute_with_kv
- `execute_direct` populates input_metas from shape_overrides
- `execute_tape_with_kv_shapes_cached` combines all three: KV + shapes + cache

### Remaining Work (true 0-sentinel lowering)
The correct fix requires:
1. **Find Reshape nodes** in the AiGraph whose shape tensor contains
   seq-dependent values (by tracing Shape→Gather→Concat→Reshape chains)
2. **Zero the specific elements** in the shape tensor constant's i64 values
   that correspond to seq-dependent positions
3. **Handle Expand** similarly — zero seq dims in target_shape
4. **Ensure runtime Reshape** handles 0 elements by inferring from total
   buffer element count and the non-zero target dims

This is a graph-analysis problem in the compiler, not a lowering strategy
change. The DeferredStrategy already handles 0-sentinels correctly for
MatMul/Softmax/RmsNorm — only Reshape/Expand need the targeted fix.

## Testing

- Compile TinyLlama at seq=64, run with 24-token prompt → correct output
- Compile at seq=128, run with 10 tokens → correct output
- KV cache decode still works (seq=1)
- Non-LLM models (BERT, ResNet) unaffected
