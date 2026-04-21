# Plan: Post-Lowering Quantization Pass

## Context

Quantization is currently smeared across 9+ locations in `builder.rs`, triggered
by Q4 eligibility checks, Gemm quant_b flags, fused op decomposition, and
on-demand quantization. Every new fusion pass (NormProjection, SwiGluProjection,
SharedInputProjection) needs its own quantization hooks. This doesn't scale —
adding Q8 fallback for small models required touching 5 different code paths and
still missed half the MatMul ops.

**The fix:** Move quantization out of lowering into a single post-lowering graph
pass. Lower everything as f32, then walk the Graph and convert eligible weights.

## Architecture

```
Import → Optimize → Lower (f32 only) → Quantize Graph → Compile → Archive
                                        ^^^^^^^^^^^^^^^^
                                        NEW: single pass
```

The Graph is fully mutable between `lower()` and `compile()`. It has:
- `graph.nodes()` — iterate all nodes
- `graph.replace_op(id, new_op)` — replace a node's GraphOp
- `graph.add_constant(data)` — register quantized weight constants
- `graph.predecessors(id)` / `graph.successors(id)` — walk edges

## Implementation

### Step 1: Create `quantize_graph()` function

New file: `crates/hologram-ai-common/src/lower/quantize_graph.rs`

```rust
pub fn quantize_graph(
    graph: &mut hologram::Graph,
    strategy: QuantStrategy,
    total_params: u64,
) -> anyhow::Result<QuantizeStats>
```

This function:
1. Computes adaptive error threshold from `total_params`
2. Auto-downgrades Q4 → Q8 for small models (<750M params)
3. Walks all nodes in the Graph
4. For each `GraphOp::Float(FloatOp::MatMul { .. })` or `GraphOp::Float(FloatOp::Gemm { .. })`:
   - Finds the weight input (predecessor that is a `Constant` node)
   - Reads the constant data (`graph.get_constant(cid)`)
   - Checks eligibility: 2D, dims >= 256, f32
   - Quantizes with appropriate algorithm (Q4 k-means or Q8 uniform)
   - Registers new constant: `graph.add_constant(ConstantData::Bytes(quantized))`
   - Replaces node: `graph.replace_op(node_id, GraphOp::MatMulLut4(new_cid))`
5. Returns stats (quantized count, skipped count, total weight savings)

### Step 2: Remove quantization from `lower()`

In `crates/hologram-ai-common/src/lower/builder.rs`:
- Remove `do_early_quant` block (lines 148-232)
- Remove `q4_eligible` set and all its checks
- Remove `early_quant_bytes` HashMap
- Remove `quantize_weight_on_demand()` function
- Remove `quantize_weight_q8_on_demand()` function
- Remove all `matmul_lut_4bit()` / `matmul_lut_8bit()` calls from node lowering
- Remove `try_convert_f32_to_lut4()` / `try_convert_f32_to_lut8()` interception
- Remove Q4/Q8 branches from FusedNormProjection and FusedSwiGluProjection lowering
- `LoweringOptions.quant_strategy` is no longer used during lowering — only by the new pass

Every MatMul/Gemm emits as `GraphOp::Float(FloatOp::MatMul)` with f32 weight constants.

### Step 3: Call `quantize_graph()` in compiler.rs

In `crates/hologram-ai/src/compiler.rs`, after each `lower()` call:

```rust
let mut lower_out = lower(&ai_graph, &kv_layout, &lowering_opts, &LowerPhase::Forward)?;

// Post-lowering quantization: single pass over the Graph.
let quant_stats = quantize_graph(
    &mut lower_out.graph,
    self.quant_strategy,
    total_params_approx,
)?;
info!(quantized = quant_stats.quantized, skipped = quant_stats.skipped, "quantize pass");

let compilation = hologram::compile(lower_out.graph)?;
```

This applies to all three LLM pipeline graphs (prefill, decode, verify) and single-graph models.

### Step 4: Handle weight access

The Graph's `ConstantStore` holds weight data as `ConstantData::Bytes(vec)`. For
f32 weights embedded during lowering, this is the raw f32 bytes. The quantize
pass reads these bytes, quantizes them, and registers the result as a new constant.

For mmap'd weights (streaming compilation), the `ConstantData::Deferred` variant
holds a byte offset into the weight file. The quantize pass needs to read from the
weight file — pass the weight source path or mmap handle alongside the graph.

### Step 5: Verify

- `cargo test` — all existing tests pass
- Qwen2 M=1: matches ORT (max diff < 2e-4)
- Qwen2 M=5: "The capital of France is Paris." 
- TinyLlama: "The capital of France is Paris." at 30+ tok/s (Q4)
- Qwen2 speed target: 10+ tok/s (Q8 with full coverage)

## Key Files

| File | Change |
|------|--------|
| `hologram-ai-common/src/lower/quantize_graph.rs` | NEW — the single quantization pass |
| `hologram-ai-common/src/lower/builder.rs` | SIMPLIFY — remove all 9 quantization hooks |
| `hologram-ai-common/src/lower/mod.rs` | Add `pub mod quantize_graph` |
| `hologram-ai/src/compiler.rs` | Call `quantize_graph()` after each `lower()` |

## Edge Cases

### Weight deduplication across pipeline graphs
The LLM pipeline compiles 3 graphs (prefill/decode/verify) sharing weights. The
quantize pass must share a cache so the same weight isn't quantized 3 times.
Pass a `&mut HashMap<ConstantId, ConstantId>` (old f32 cid → new quantized cid)
across graphs.

### Pre-quantized weights (Gemm quant_b)
Some models arrive with pre-quantized weights via `Gemm { quant_b: 1 }`. The
lowering currently intercepts these. Two options:
- **Keep the interception** in lowering for pre-quantized inputs (it's not a
  policy decision, it's respecting the model's existing format)
- Or emit as `Float(Gemm)` and let the quantize pass recognize `quant_b`

Recommended: keep the pre-quantized interception since it's format handling, not
a quantization policy decision.

### Streaming compilation (large models)
For models > 256 MB, weights are mmap'd to a temp file (`ConstantData::Deferred`).
The quantize pass receives the weight file path and reads weights on demand via
mmap. After quantization, the quantized bytes replace the deferred constant.

### Fused activation epilogues
`MatMulLut4Activation` fuses matmul + activation. The quantize pass checks if the
quantized MatMul's sole consumer is a ReLU/SiLU/GeLU and emits the fused variant
instead. This is a simple successor check in the Graph.

### Conv2d quantization
SD UNet quantizes Conv2d weights via `try_convert_conv2d_to_lut4`. Include Conv2d
in the pass's node scan — same pattern as MatMul but with reshaping the 4D weight
to 2D before quantizing.

### Per-node rollback
If quantization of one weight fails (error threshold exceeded), that node stays as
`Float(MatMul)` — no graph-wide rollback needed. Each node is independent.

## What This Enables

Once quantization is a single pass:
- Adding Q8 fallback for small models = 1 line change in the pass
- Per-layer sensitivity (protect first/last layers) = straightforward loop condition
- New fusion passes never need quantization hooks
- Mixed quantization (Q4 for FFN, Q8 for attention) = policy in one place
- Future quantization algorithms (GPTQ, AWQ) slot in cleanly
