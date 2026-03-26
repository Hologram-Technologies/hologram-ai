# Plan 033 — Wire ShapeContextGraph into Runtime Execution

## Context

Variable-length KV decode is broken. When compiled at seq=32 but executed
at seq=6+1, decode produces cos_sim=0.076 against the correct logits
(test: `onnx_kv_decode_variable_length`). Same-seq decode works perfectly
(cos_sim=1.0, test: `onnx_kv_decode_matches_full_prefill`).

Root cause: the runtime infers shapes from buffer sizes using heuristics
(`resolve_matmul_dims`, `resolve_last_dim`, `infer_slice_axis_size`).
These heuristics are mathematically ambiguous — multiple shapes can
produce the same buffer byte count. During variable-length decode,
at least one op resolves to the wrong shape, corrupting downstream
computation through 22 transformer layers.

The fix is NOT another heuristic. The compiler already knows every shape.
The `ShapeContextGraph` — a per-node shape recipe computed during lowering
— is fully implemented, serialized in the archive, loaded by `HoloRunner`,
and never used. This plan wires it into the execution path.

## Design Principles

1. **The compiler knows all shapes.** No runtime guessing.
2. **One topological pass before execution** resolves every node's shape.
3. **One position mechanism.** Position is a dimension in the shape recipe,
   not a separate `ExecutionContext.position_offset`.
4. **Conformance-gated.** The existing failing test passes when this is done.

## What Exists (99% complete)

| Component | Location | Status |
|-----------|----------|--------|
| `ShapeContextGraph` struct | `exec_context.rs:202-360` | Done |
| `ShapeProjection` trait (100+ ops) | `shape_spec_bridge.rs:24-379` | Done |
| `resolve_spec()` function | `shape_spec_bridge.rs:402-580` | Done |
| `walk_shape_context()` function | `shape_spec_bridge.rs:835-929` | Done |
| `propagate_i64_values()` | `shape_spec_bridge.rs:937-1048` | Done |
| Builder integration (populates during lowering) | `builder.rs:105-607` | Done |
| Archive embedding (`SECTION_SHAPE_CONTEXT`) | `builder.rs:605-607` | Done |
| Archive reading (`read_shape_context_from_plan`) | `compiler.rs:1814-1845` | Done |
| `HoloRunner.shape_ctx` field | `compiler.rs:1600` | Loaded, unused |
| Unit tests (`walk_shape_context_*`) | `exec_conformance.rs` | Pass |

## What's Missing (the 1%)

The runtime never calls `walk_shape_context()`. The shape_map it produces
is never used to set `TensorMeta` on arena nodes before dispatch.

## Implementation

### Phase 1: Wire walk_shape_context into execute_tape (hologram-ai)

**File**: `crates/hologram-ai/src/compiler.rs`

Before calling `hologram::execute_tape()`, the `HoloRunner` should:

1. Collect runtime input shapes from `GraphInputs`
2. Call `walk_shape_context()` to produce `shape_map: HashMap<u32, Vec<usize>>`
3. Pass `shape_map` into the execution path

The cleanest integration point is `HoloRunner::execute()` and
`HoloRunner::execute_with_kv()`. Both currently delegate directly
to `hologram::execute_tape()`. Instead:

```rust
pub fn execute(&self, inputs: &GraphInputs) -> Result<GraphOutputs> {
    let shape_map = self.resolve_shapes(inputs);
    hologram::execute_tape_with_shapes(&self.tape, &self.plan, inputs, &shape_map)
}

fn resolve_shapes(&self, inputs: &GraphInputs) -> HashMap<u32, Vec<usize>> {
    let Some(ctx) = &self.shape_ctx else { return HashMap::new() };
    let runtime_inputs = collect_input_shapes(self.plan(), inputs);
    let mut shape_map = HashMap::new();
    walk_shape_context(ctx, &runtime_inputs, &HashMap::new(), &mut shape_map);
    shape_map
}
```

### Phase 2: New hologram base API — execute_tape_with_shapes (hologram base)

**File**: hologram `crates/hologram-exec/src/mmap/mod.rs`

Add a new execution function that accepts pre-computed shapes:

```rust
pub fn execute_tape_with_shapes(
    tape: &EnumTape,
    plan: &LoadedPlan,
    inputs: &GraphInputs,
    shape_map: &HashMap<u32, Vec<usize>>,
) -> ExecResult<GraphOutputs>
```

This function does everything `execute_tape` does, plus:
- After `seed_arena()`, iterate `shape_map` and set `TensorMeta` on
  every node that has a resolved shape
- This pre-populates the arena with correct N-D metadata for ALL nodes,
  not just constants and inputs

When `shape_map` is empty (no ShapeContextGraph available), falls back
to the current behavior (heuristic inference).

Similarly for `execute_tape_with_kv_and_shapes`.

### Phase 3: Remove heuristic fallbacks (hologram base)

Once all shapes flow through `shape_map`, the heuristic functions become
dead code for the ShapeContextGraph path. Don't delete them yet — keep
as fallback for archives without shape context (backward compat) — but
add metrics/logging that tracks how often the fallback fires. Goal: zero
fallback invocations for any model compiled with shape context.

**Files**:
- `crates/hologram-exec/src/shape_resolve.rs` — add fallback counter
- `crates/hologram-exec/src/float_dispatch/mod.rs` — `infer_slice_axis_size` logs warning

### Phase 4: Unify position encoding (hologram-ai + hologram base)

Position is a shape dimension, not a separate mechanism. The current dual
system (`position_ids` input + `ExecutionContext.position_offset`) should
collapse into one:

- The graph carries `position_ids` as an explicit input (ONNX path)
- The `run_cmd.rs` generation loop sets `position_ids` correctly for
  both prefill and decode (it already does this)
- Remove `ExecutionContext.position_offset` — RoPE for GGUF should read
  position from the same `position_ids` input, not a side-channel

This requires hologram-ai's GGUF lowering to inject `position_ids` as a
graph input (like the ONNX path already does via `PositionIdsInjection`).

**Files**:
- `crates/hologram-ai-gguf/src/arch/llama.rs` — inject position_ids input
- `crates/hologram-ai-common/src/lower/strategy.rs` — lower RoPE to use
  position_ids input instead of ExecutionContext
- hologram base `tape.rs` — `InlineRoPE` reads position from input tensor,
  not `tape_ctx.ctx.position_offset`
- hologram base `mmap/mod.rs` — remove `ExecutionContext` setup

### Phase 5: Conformance gate

The existing test `onnx_kv_decode_variable_length` must pass.
Add equivalent for GGUF: `gguf_kv_decode_variable_length`.

Both tests compile at seq=32, run at seq=6+1, compare prefill-all
logits against KV-decode logits.

## Execution Order

| # | Phase | Repo | Effort | Blocks |
|---|-------|------|--------|--------|
| 1 | Wire walk_shape_context | hologram-ai | S | — |
| 2 | execute_tape_with_shapes API | hologram base | S | Phase 1 |
| 3 | Remove heuristic fallbacks | hologram base | M | Phase 2 |
| 4 | Unify position encoding | both | M | Phase 2 |
| 5 | Conformance gate | hologram-ai | S | Phase 2 |

## Blocker Found: Pre-Fusion vs Post-Fusion Node IDs

The ShapeContextGraph is computed during lowering (pre-fusion graph).
`hologram::compile()` runs fusion passes (constant folding, unary chain
fusion, MatMul+Activation fusion, bias fusion) that ADD and REMOVE nodes,
changing the graph topology. The serialized (post-fusion) graph has
different node IDs than the pre-fusion graph.

Consequence: shape_map keys from `walk_shape_context` refer to pre-fusion
node indices. The tape executor's `instr.output_idx` uses post-fusion
indices. **They don't match.** Shape overrides are applied to wrong nodes.

### Solutions (in order of design quality):

**A. Compute ShapeContextGraph post-compilation** (best)
Move shape projection computation into hologram base's compiler, after
fusion. The `SerializedGraph` already carries `constant_shapes` and
`node_shapes` — extend with `ShapeProjectionEntry` per node. The
`ShapeProjection` trait would need to move to hologram base (or the
projection logic duplicated for `FloatOp` + fused ops).

**B. Maintain a pre→post node ID mapping** (moderate)
During `hologram::compile()`, track which pre-fusion node maps to which
post-fusion node. Pass this mapping through the compilation output.
`HoloRunner` applies the mapping before using shape overrides.

**C. Disable fusion when shape context is present** (stopgap)
If the archive has a ShapeContextGraph, skip the fusion stage. This
preserves node ID correspondence but loses fusion benefits.

Recommended: **Option A** — it aligns with the principle that the compiler
knows everything. The shape recipes should be computed on the graph the
runtime actually executes.

Phases 1+2 are implemented (API surface ready). Phase 3 is blocked on
the node ID mapping issue. Phase 4 is architectural. Phase 5 already exists.

## Verification

1. `cargo test -p hologram-ai --features e2e -- onnx_kv_decode` — both tests pass
2. `cargo test -p hologram-ai` — mini_transformer tests still pass
3. `cargo test -p hologram-ai --features e2e -- tinyllama` — all TinyLlama tests pass
4. `hologram-ai run ... --prompt "Tell me a joke"` — coherent English output
5. `cargo clippy -- -D warnings` — clean in both repos
6. No `.unwrap()`, no `println!`/`eprintln!` in committed code

## Critical Files

| File | Repo | Changes |
|------|------|---------|
| `crates/hologram-ai/src/compiler.rs` | hologram-ai | Wire walk_shape_context in HoloRunner |
| `crates/hologram-ai-common/src/lower/shape_spec_bridge.rs` | hologram-ai | walk_shape_context (already exists) |
| `crates/hologram-ai-common/src/exec_context.rs` | hologram-ai | ShapeContextGraph (already exists) |
| `crates/hologram-exec/src/mmap/mod.rs` | hologram base | execute_tape_with_shapes API |
| `crates/hologram-exec/src/shape_resolve.rs` | hologram base | Fallback tracking |
| `crates/hologram-ai/tests/mini_fixture.rs` | hologram-ai | Conformance tests (exist) |
