# Plan 059: Post-Fusion ShapeContextGraph

**Status:** Open
**Created:** 2026-04-05
**Supersedes:** Plan 033 Option A, Plan 045, Plan 058

## Problem

Variable-length execution (runtime seq_len ≠ compiled seq_len) fails because
the `ShapeContextGraph` is computed pre-fusion but executed against a
post-fusion tape. Fusion removes ~28% of nodes, breaking shape projection
chains. The remaining ops have baked parameters (MatMul m/k/n, Softmax size)
containing the compiled seq_len that shape metadata alone can't override.

**Symptoms:**
- Compile at seq=24, run with 18 or 36 tokens → garbage output
- Compile at seq=2048, run with any length ≤ 2048 → works (heuristics succeed
  when buffer sizes are proportional to baked values)

## Root Cause

The ShapeContextGraph is computed during `lower()` in hologram-ai using
pre-fusion node IDs. `hologram::compile()` runs fusion passes that:
1. **Remove nodes** (constant folding, view fusion, CSE) — projection entries
   for those nodes are pruned, but downstream entries that depended on them
   can no longer resolve their inputs
2. **Replace ops** (float chain fusion, matmul+activation fusion) — the new
   fused op has the same node ID but different shape semantics
3. **No new projections** are created for fused ops

Result: `walk_shape_context` resolves 595 of 824 tape nodes (72%). The
remaining 229 nodes use heuristic shape resolution, which fails when
runtime shapes differ from compiled shapes.

## Solution: Compute ShapeContextGraph Post-Fusion

Move shape projection computation into hologram base's `emit_stage()`,
after fusion completes. The post-fusion graph has complete topology —
every node that will appear in the tape has a projection rule.

### Design Principles

1. **The compiler knows all shapes.** Shape projection is a compile-time
   computation, not a runtime heuristic.
2. **Every tape node has a projection.** No gaps, no fallbacks.
3. **Projection rules are simple.** Most ops are element-preserving
   (shape out = shape in). Only ~12 op categories need custom rules.
4. **Backward compatible.** Archives without shape projections fall back
   to existing heuristic behavior.

## Architecture

### New: `ShapeProjection` in hologram-core

**File:** `hologram-core/src/op/shape_projection.rs`

A pure function that computes output shape from input shapes and op
parameters. No traits needed — just a `match` on `GraphOp`:

```rust
/// Compute the output shape of a GraphOp given its input shapes.
///
/// Returns `None` for ops whose output shape can't be determined
/// statically (Custom ops, data-dependent ops like NonZero).
pub fn project_shape(
    op: &GraphOp,
    input_shapes: &[&[usize]],
) -> Option<Vec<usize>> {
    match op {
        // Element-preserving: output = input[0]
        GraphOp::Lut(_) | GraphOp::FusedView(_) | GraphOp::FusedView16(_)
        | GraphOp::Passthrough | GraphOp::Output => {
            input_shapes.first().map(|s| s.to_vec())
        }

        // Byte-domain binary: broadcast
        GraphOp::Prim(PrimOp::Add | PrimOp::Mul | ...) => {
            broadcast_shape(input_shapes.get(0)?, input_shapes.get(1)?)
        }

        // Float ops: delegate to float_project_shape
        GraphOp::Float(f) => float_project_shape(f, input_shapes),

        // Fused float chain: first op's shape (all unary, element-preserving)
        GraphOp::FusedFloatChain(_) => {
            input_shapes.first().map(|s| s.to_vec())
        }

        // Fused matmul variants: [m, n]
        GraphOp::FusedMatMulActivation { m, k: _, n, .. }
        | GraphOp::FusedMatMulBiasActivation { m, k: _, n, .. } => {
            Some(vec![*m as usize, *n as usize])
        }

        // Constants, inputs: shape from graph metadata (seeds)
        GraphOp::Constant(_) | GraphOp::Input => None, // seeded separately

        _ => None,
    }
}

fn float_project_shape(op: &FloatOp, inputs: &[&[usize]]) -> Option<Vec<usize>> {
    match op {
        // Unary element-preserving (~40 variants)
        FloatOp::Relu | FloatOp::Gelu | FloatOp::Silu | FloatOp::Sigmoid
        | FloatOp::Tanh | FloatOp::Exp | FloatOp::Log | FloatOp::Neg
        | FloatOp::Abs | FloatOp::Sqrt | FloatOp::Reciprocal
        | FloatOp::Cos | FloatOp::Sin | FloatOp::Sign | FloatOp::Floor
        | FloatOp::Ceil | FloatOp::Round | FloatOp::Erf | FloatOp::Clip
        | FloatOp::IsNaN | FloatOp::Not | FloatOp::Cast { .. }
        | FloatOp::Dequantize { .. } => {
            inputs.first().map(|s| s.to_vec())
        }

        // Binary broadcast
        FloatOp::Add | FloatOp::Sub | FloatOp::Mul | FloatOp::Div
        | FloatOp::Pow | FloatOp::Mod | FloatOp::Min | FloatOp::Max
        | FloatOp::Equal | FloatOp::Less | FloatOp::Greater
        | FloatOp::Where => {
            broadcast_shape(inputs.get(0)?, inputs.get(1)?)
        }

        // Norms, softmax: shape preserved
        FloatOp::Softmax { .. } | FloatOp::LogSoftmax { .. }
        | FloatOp::RmsNorm { .. } | FloatOp::LayerNorm { .. }
        | FloatOp::AddRmsNorm { .. } | FloatOp::InstanceNorm { .. }
        | FloatOp::GroupNorm { .. } | FloatOp::FusedSwiGLU => {
            inputs.first().map(|s| s.to_vec())
        }

        // MatMul: [*, m, k] × [*, k, n] → [*, m, n]
        FloatOp::MatMul { m, k: _, n } => {
            // Use baked m/n, but at runtime these will be overridden
            // by actual input shapes via input_metas
            Some(vec![*m as usize, *n as usize])
        }

        // Reshape: metadata-only, shape comes from ShapeContextGraph
        FloatOp::Reshape => None, // handled by seed/projection chain

        // Transpose: apply permutation
        FloatOp::Transpose { perm, ndim } => {
            let input = inputs.first()?;
            let n = *ndim as usize;
            if input.len() < n { return None; }
            let out: Vec<usize> = (0..n)
                .map(|i| input[perm[i] as usize])
                .collect();
            Some(out)
        }

        // Reductions: drop last dim
        FloatOp::ReduceSum { .. } | FloatOp::ReduceMean { .. }
        | FloatOp::ReduceMax { .. } | FloatOp::ReduceMin { .. }
        | FloatOp::ReduceProd { .. } => {
            let input = inputs.first()?;
            if input.len() <= 1 { return Some(vec![1]); }
            Some(input[..input.len() - 1].to_vec())
        }

        // Gather: output shape = indices shape × dims_after_axis
        FloatOp::Gather { dim, .. } => {
            // Complex — depends on indices shape
            None // fall back to heuristic
        }

        // Embed: [token_ids] → [len, dim]
        FloatOp::Embed { dim, .. } => {
            let indices = inputs.first()?;
            let len: usize = indices.iter().product();
            Some(vec![len, *dim as usize])
        }

        // Expand: target_shape
        FloatOp::Expand { ndim, target_shape } => {
            let n = *ndim as usize;
            Some(target_shape[..n].iter().map(|&d| d as usize).collect())
        }

        // Slice: modify one axis
        FloatOp::Slice { start, end, .. } => {
            let input = inputs.first()?;
            let mut out = input.to_vec();
            // Slice modifies one axis: size = end - start
            let slice_len = (*end as usize).saturating_sub(*start as usize);
            if let Some(last) = out.last_mut() {
                *last = slice_len;
            }
            Some(out)
        }

        // Concat: sum along concat axis
        FloatOp::Concat { size_a, size_b, .. } => {
            let a = inputs.get(0)?;
            let mut out = a.to_vec();
            // Replace the concat dim with size_a + size_b
            if let Some(last) = out.last_mut() {
                *last = (*size_a as usize) + (*size_b as usize);
            }
            Some(out)
        }

        // Conv2d: spatial transform
        FloatOp::Conv2d { kernel_h, kernel_w, stride_h, stride_w,
                          pad_h, pad_w, dilation_h, dilation_w, group: _, input_h, input_w } => {
            let input = inputs.first()?;
            if input.len() < 4 { return None; }
            let weight = inputs.get(1)?;
            let n = input[0];
            let c_out = weight[0];
            let h_out = (*input_h as usize + 2 * *pad_h as usize
                        - *dilation_h as usize * (*kernel_h as usize - 1) - 1)
                        / *stride_h as usize + 1;
            let w_out = (*input_w as usize + 2 * *pad_w as usize
                        - *dilation_w as usize * (*kernel_w as usize - 1) - 1)
                        / *stride_w as usize + 1;
            Some(vec![n, c_out, h_out, w_out])
        }

        // Attention: [seq, num_q_heads, head_dim]
        FloatOp::Attention { head_dim, num_q_heads, .. } => {
            let q = inputs.first()?;
            // Output matches Q shape
            Some(q.to_vec())
        }

        // KV ops: pass-through
        FloatOp::KvWrite { .. } | FloatOp::KvRead { .. } => {
            inputs.first().map(|s| s.to_vec())
        }

        // RoPE: element-preserving
        FloatOp::RotaryEmbedding { .. } => {
            inputs.first().map(|s| s.to_vec())
        }

        _ => None,
    }
}
```

### Modified: `emit_stage()` in hologram-compiler

**File:** `hologram-compiler/src/compiler/mod.rs`

After fusion, walk the post-fusion graph in topological order and build
a `Vec<(NodeId, Vec<usize>)>` of projected shapes. For each node:

1. Look up input shapes from the shape map (seeded from graph inputs and
   constants, then projected forward)
2. Call `project_shape(op, input_shapes)` to compute output shape
3. Insert into shape map

This is the same algorithm as `walk_shape_context` but runs once at
compile time on the post-fusion graph. The output is a complete
`node_shapes` map that covers 100% of tape nodes.

```rust
fn compute_post_fusion_shapes(
    graph: &Graph,
    schedule: &ExecutionSchedule,
) -> HashMap<NodeId, Vec<usize>> {
    let mut shape_map: HashMap<NodeId, Vec<usize>> = HashMap::new();

    // Seed from graph metadata (inputs, constants).
    for (&nid, shape) in graph.node_shapes() {
        shape_map.insert(nid, shape.clone());
    }
    for (&cid, shape) in graph.constant_shapes() {
        // Find the Constant node for this ConstantId and seed it.
        // ...
    }

    // Topological walk: project each node's output shape from inputs.
    for level in &schedule.levels {
        for &nid in level {
            let node = match graph.get(nid) {
                Some(n) => n,
                None => continue,
            };
            let input_shapes: Vec<Vec<usize>> = node.predecessors()
                .iter()
                .map(|&pred| shape_map.get(&pred).cloned().unwrap_or_default())
                .collect();
            let input_refs: Vec<&[usize]> = input_shapes.iter()
                .map(|s| s.as_slice())
                .collect();

            if let Some(out_shape) = project_shape(&node.op, &input_refs) {
                shape_map.insert(nid, out_shape);
            }
        }
    }

    shape_map
}
```

### Modified: `CompilationOutput`

Add the shape map to the output:

```rust
pub struct CompilationOutput {
    pub archive: Vec<u8>,
    pub stats: CompilationStats,
    pub schedule: ExecutionSchedule,
    pub qedl_boundaries: Vec<(NodeId, QedlBoundary, EncodingId)>,
    pub node_shapes: HashMap<NodeId, Vec<usize>>,  // NEW: post-fusion shapes
}
```

### Modified: Archive embedding

hologram-ai reads `CompilationOutput.node_shapes` and embeds them in the
`ShapeContextGraph` section of the archive. The node IDs now match the
post-fusion tape exactly — no pruning needed.

The `ShapeContextGraph` format simplifies: instead of projection rules
(seeds + projections), it's just a flat `HashMap<u32, Vec<usize>>` of
compiled shapes per node. At runtime, `walk_shape_context` is replaced
by proportional scaling keyed on `seq_dim_positions`.

Actually, a better design: keep the projection rules but compute them
post-fusion. This way the runtime can project shapes from actual input
shapes (not just scale compiled shapes). The projection rules for
post-fusion ops are:

| Op Category | Projection Rule |
|-------------|----------------|
| Unary / FusedFloatChain | SameAs(0) |
| Binary (Add, Mul, ...) | Broadcast(0, 1) |
| MatMul / Gemm / Fused variants | MatMul { k_hint } |
| Softmax, Norm, Activation | SameAs(0) |
| Reshape | Reshape (shape from i64 chain) |
| Transpose | Permute { perm } |
| Reduce* | DropLastDim |
| Concat | ConcatDim { axis } |
| Slice | SliceDim { axis, start, end } |
| Conv2d / Pool | Spatial { params } |
| Gather | GatherDim { dim } |
| Embed | EmbedDim { dim } |
| Constant / Input | Seed (from metadata) |
| KvWrite / KvRead | SameAs(0) |

### Modified: HoloRunner

`resolve_shapes()` calls `walk_shape_context()` on the post-fusion
graph. Since every tape node has a projection, the shape map covers
100% of nodes. No heuristic fallback needed.

## Implementation Phases

### Phase 1: `project_shape()` function (hologram-core)

Add `shape_projection.rs` to `hologram-core/src/op/`. Pure function,
~200 lines, covers all `FloatOp` and `GraphOp` variants with a `match`.
Test with unit tests for each op category.

**Files:**
- `hologram-core/src/op/shape_projection.rs` (new)
- `hologram-core/src/op/mod.rs` (add module)

### Phase 2: Post-fusion shape computation (hologram-compiler)

In `emit_stage()`, call `compute_post_fusion_shapes()` to walk the
post-fusion graph and produce `HashMap<NodeId, Vec<usize>>`. Add to
`CompilationOutput`.

**Files:**
- `hologram-compiler/src/compiler/mod.rs` (modify emit_stage)

### Phase 3: Embed in archive (hologram-archive)

Store the post-fusion shapes in `SerializedGraph.node_shapes` (which
already exists but currently uses pre-lowering shapes). Or add a new
section for post-fusion shape projections.

**Files:**
- `hologram-archive/src/format/graph.rs` (verify node_shapes populated)

### Phase 4: Wire into HoloRunner (hologram-ai)

Use the post-fusion `node_shapes` as `shape_overrides` directly. At
runtime, for each node, scale the compiled seq dim to runtime seq:

```rust
fn resolve_shapes_from_node_shapes(
    sg: &SerializedGraph,
    inputs: &GraphInputs,
) -> HashMap<u32, Vec<usize>> {
    let compiled_seq = detect_compiled_seq(sg);
    let runtime_seq = inputs.shape(0).and_then(|s| s.last().copied());

    let mut overrides = HashMap::new();
    if let (Some(cs), Some(rs)) = (compiled_seq, runtime_seq) {
        for (nid, shape) in &sg.node_shapes {
            let scaled = scale_seq_dim(shape, cs, rs, &seq_dim_positions);
            overrides.insert(nid.index(), scaled);
        }
    }
    overrides
}
```

But this still needs `seq_dim_positions` to know which dims to scale.
The post-fusion shapes alone aren't enough — we need the projection
rules to project from runtime inputs.

**Better approach:** Compute and embed `ShapeProjectionEntry` per node
(post-fusion), then use the existing `walk_shape_context` at runtime.
This is the same design as today's ShapeContextGraph, but computed
on the post-fusion graph so every node has an entry.

### Phase 5: Remove pre-fusion ShapeContextGraph

Once post-fusion projections are working:
- Remove `retain_live_nodes()` (no longer needed)
- Remove pre-fusion ShapeContextGraph from `lower_out.context`
- Remove `seq_dim_positions` collection in `concretize_all_dims`
- Remove prompt-length guard in `resolve_seq_mode()`

### Phase 6: Conformance test

- Compile TinyLlama at seq=24, run with 10, 18, 24, and 36 token prompts
- All produce coherent output (top-5 tokens match seq=2048 baseline)
- KV cache decode still works

## Critical Files

**hologram base:**
| File | Change |
|------|--------|
| `hologram-core/src/op/shape_projection.rs` | New: `project_shape()` function |
| `hologram-core/src/op/mod.rs` | Add module |
| `hologram-compiler/src/compiler/mod.rs` | Compute post-fusion shapes in `emit_stage()` |
| `hologram-archive/src/format/graph.rs` | Verify `node_shapes` population |

**hologram-ai:**
| File | Change |
|------|--------|
| `hologram-ai/src/compiler.rs` | Use post-fusion shapes; remove pre-fusion SCG |
| `hologram-ai/src/commands/run_cmd.rs` | Remove prompt-length guard |
| `hologram-ai-common/src/exec_context.rs` | Remove `retain_live_nodes()` |

## Effort Estimate

| Phase | Scope | Size |
|-------|-------|------|
| 1 | shape_projection.rs | M (200 lines, ~90 match arms) |
| 2 | emit_stage integration | S (20 lines) |
| 3 | Archive embedding | S (verify existing) |
| 4 | HoloRunner wiring | M (replace resolve_shapes) |
| 5 | Cleanup | S (remove dead code) |
| 6 | Conformance | S (1 test) |

Total: ~1-2 focused sessions.

## Risks

- **Fused ops shape rules:** FusedMatMulActivation, FusedConv2dActivation
  preserve the base op's output shape. FusedFloatChain is element-preserving.
  These are straightforward but must be exhaustively matched.
- **Reshape projection chain:** The Shape→Gather→Concat→Reshape pattern
  requires i64 value propagation through the post-fusion graph. If fusion
  folded the Shape/Gather/Concat nodes into constants, the Reshape target
  is a baked constant — the projection rule becomes `Seed` (read from
  archive), not `Reshape` (compute from chain). This is simpler but means
  the Reshape target shape contains the compiled seq_len.
- **The Reshape problem persists** if the target shape constant is baked.
  The post-fusion SCG projects correct shapes for 100% of nodes BUT the
  Reshape target constant still has `[1, 24, 32, 64]`. The projection
  says "output shape = target shape" which is `[1, 24, 32, 64]`, not
  `[1, 18, 32, 64]`. This is the same problem as before — just with
  100% coverage instead of 72%.

**Mitigation for Reshape:** The post-fusion `project_shape()` can
recognize Reshape nodes and instead of using the baked target, compute
the target from the input shape + known dimension structure (batch=1,
seq=runtime, heads=compiled, head_dim=compiled). This requires knowing
which target dim is seq-dependent — which is exactly the
`seq_dim_positions` information we already collect.

## Alternative: Hybrid Approach

Instead of full post-fusion ShapeProjection, combine:
1. **Post-fusion `node_shapes`** (already in archive) — provides 100%
   coverage with compiled shapes
2. **`seq_dim_positions`** (already collected) — identifies which dims
   in each tensor are seq-dependent
3. **Runtime scaling** — for each node, replace the seq-dependent dim
   with the runtime value

This avoids building a full projection system. The challenge is that
`seq_dim_positions` uses pre-lowering `TensorId` keys, not post-fusion
`NodeId` keys. A `TensorId → NodeId` mapping from the lowering output
(`tid_to_idx`) bridges this gap.

This hybrid approach is simpler than full post-fusion projection and
may be sufficient for LLM workloads where seq is the only variable dim.
