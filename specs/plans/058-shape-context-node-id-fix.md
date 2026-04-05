# Plan 058: ShapeContextGraph Node-ID Alignment

**Status:** Open
**Created:** 2026-04-05
**Supersedes:** Plan 033 (Phase 3 blocker), Plan 045 (Option B)

## Problem

The `ShapeContextGraph` is built during lowering (pre-compilation) but
executed against a post-compilation tape. `hologram::compile()` runs
fusion passes that **remove** nodes from the graph, creating gaps in the
node-ID space. The ShapeContextGraph references these removed nodes,
so `shape_overrides` entries for them never match any tape instruction.

Result: variable-length prefill produces garbage when prompt length
differs from compiled seq_len. The shape context activates but the
projected shapes don't reach the ops that need them.

## Root Cause (Verified)

Investigation of the compilation pipeline confirms:

1. **ShapeContextGraph** is populated during `lower()` in `builder.rs`
   using builder node indices (sequential `u32` values)
2. `hologram::compile()` calls `fuse()` which removes nodes via
   `graph.remove_node()` — their arena slots become free
3. **Surviving** nodes keep their original `NodeId.index()` values
4. `SerializedGraph::from_graph()` preserves original `NodeId`s
5. Tape builder sets `output_idx = NodeId.index()` for each instruction
6. At runtime, `shape_overrides[removed_node_id]` matches nothing

**Key insight:** Only references to *removed* nodes are broken. Shape
entries for surviving nodes already have correct IDs. The fix is to
prune dead entries from the ShapeContextGraph after compilation.

## Fix Strategy: Post-Compilation ShapeContextGraph Pruning

After `hologram::compile()`, the `CompilationOutput.archive` contains
the serialized graph with only live nodes. Prune the ShapeContextGraph
to remove entries referencing nodes that no longer exist in the archive.

This is simpler than Option A from Plan 033 (moving shape computation
into hologram base) because:
- No code moves between repos
- ShapeProjection trait stays in hologram-ai
- Projection entries for surviving nodes are already correct
- Only dead entries need removal

### Phase 1: Collect live node IDs from compiled archive

**File:** `hologram-ai/crates/hologram-ai/src/compiler.rs`

After `hologram::compile()`, unpack the archive and collect all live
node IDs from the `SerializedGraph`:

```rust
let compilation = hologram::compile(lower_out.graph)?;
let unpacked = unpack_archive(&compilation.archive)?;

// Collect live node IDs from the compiled (post-fusion) graph.
let live_ids: HashSet<u32> = unpacked.plan.graph().nodes
    .iter()
    .map(|n| n.id.index())
    .collect();
```

### Phase 2: Prune ShapeContextGraph

**File:** `hologram-ai-common/crates/hologram-ai-common/src/exec_context.rs`

Add a `retain_live_nodes()` method to `ShapeContextGraph`:

```rust
impl ShapeContextGraph {
    /// Remove entries referencing nodes that were eliminated by fusion.
    pub fn retain_live_nodes(&mut self, live_ids: &HashSet<u32>) {
        self.seeds.retain(|s| live_ids.contains(&s.node_id));
        self.projections.retain(|p| {
            live_ids.contains(&p.node_id)
                && p.input_node_ids.iter().all(|id| live_ids.contains(id))
        });
    }
}
```

### Phase 3: Wire pruning into compilation pipeline

**File:** `hologram-ai/crates/hologram-ai/src/compiler.rs`

In `compile_llm_component()` and `compile_single_graph()`, after
extracting the ShapeContextGraph from `lower_out.context`, prune it:

```rust
let mut shape_ctx = lower_out.context
    .get::<ShapeContextGraph>().ok().flatten();

if let Some(ref mut ctx) = shape_ctx {
    let live_ids: HashSet<u32> = unpacked.plan.graph().nodes
        .iter()
        .map(|n| n.id.index())
        .collect();
    ctx.retain_live_nodes(&live_ids);
}
```

### Phase 4: Re-enable variable-length for all prompt lengths

**File:** `hologram-ai/crates/hologram-ai/src/commands/run_cmd.rs`

In `resolve_seq_mode()`, remove the `prompt <= compiled_seq` guard.
With pruned shape context, variable-length works for any prompt length:

```rust
if runner.has_shape_context() {
    info!("shape context available — using variable-length execution");
    return SeqMode::Variable { max_seq };
}
```

### Phase 5: Conformance tests

**File:** `hologram-ai/crates/hologram-ai/tests/mini_fixture.rs`

Add a test that compiles TinyLlama at a small seq_len, then runs with
a different prompt length and verifies coherent output:

```rust
#[test]
fn variable_length_shape_context() {
    // Compile at seq=16
    // Run with 8-token prompt → shape context resolves correctly
    // Run with 24-token prompt → shape context resolves correctly
    // Both produce coherent top-5 tokens (not garbage)
}
```

## Critical Files

| File | Repo | Change |
|------|------|--------|
| `crates/hologram-ai-common/src/exec_context.rs` | hologram-ai | Add `retain_live_nodes()` |
| `crates/hologram-ai/src/compiler.rs` | hologram-ai | Prune shape context after compile |
| `crates/hologram-ai/src/commands/run_cmd.rs` | hologram-ai | Remove prompt-length guard |
| `crates/hologram-ai/tests/mini_fixture.rs` | hologram-ai | Conformance test |

## What Exists (no changes needed)

| Component | Status |
|-----------|--------|
| `walk_shape_context()` | Working |
| `ShapeProjection` trait (100+ ops) | Working |
| Archive embedding/reading | Working |
| `HoloRunner.resolve_shapes()` | Working (wired this session) |
| `execute_direct` input_metas from shape_overrides | Working (wired this session) |
| `execute_tape_with_kv_shapes_cached` | Working (added this session) |

## Testing

```bash
# Unit tests
cargo test -p hologram-ai-common
cargo test -p hologram-ai

# E2E: compile at seq=24, run with 36-token prompt
RUST_LOG=info cargo run --release -- run /tmp/tinyllama-prewarm/model.holo \
  --prompt "<|system|>You are a comedian.</s><|user|>Tell me a joke</s><|assistant|>" \
  --temperature 0.0

# Verify: "shape context available" in logs, coherent output, no garbage
```

## Risks

- **Orphaned projections:** If a projection's input was removed but its
  output survived (e.g., fusion absorbed the input into the output node),
  the projection entry is pruned. The output node falls back to heuristic
  shape resolution. This is safe but suboptimal — monitor with a counter.
- **Chain breakage:** Projections form a topological chain. Removing a
  middle entry means downstream entries can't resolve their inputs from
  the shape_map. `walk_shape_context` handles this gracefully (skips
  entries with missing inputs).
