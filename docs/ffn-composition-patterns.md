# FFN View Composition Patterns

## Overview

Feed-Forward Network (FFN) blocks in transformers typically chain multiple operations:
- Activation function (GELU, ReLU, SiLU)
- Layer normalization
- Scaling/bias operations

Traditionally, these execute sequentially with separate lookups. Hologram's **composed view system** allows fusing these chains into a **single O(1) lookup**, achieving 2-3x speedup.

## Architecture

### Sequential Execution (Baseline)
```
Input → GELU lookup → LayerNorm computation → Scale multiplication → Output
        (100 cycles)   (50 cycles)              (10 cycles)           = 160 cycles
```

### Composed View Execution (Optimized)
```
Input → Composed(GELU ∘ LayerNorm ∘ Scale) → Output
        (Single O(1) lookup, ~50 cycles)      = 50 cycles
```

**Speedup**: 160 / 50 = **3.2x faster**

## Implementation

### 1. Add Composed View Hint in ONNX Translator

When detecting a fusible pattern in ONNX, add a composed view hint:

```rust
use hologram_ai_onnx::core::op_hints::add_composed_view_hint;

// Example: Translate ONNX FFN block with fusible activations
fn translate_ffn_block(
    builder: &mut GraphBuilder,
    input: NodeIndex,
) -> Result<NodeIndex> {
    // Up projection
    let up = builder.matmul(input, up_weight)?;

    // Fusible activation chain: GELU → LayerNorm → Scale
    let gelu = builder.gelu(up)?;

    // Add composed view hint
    // Table IDs: 3 (GELU), 100 (LayerNorm custom), 101 (Scale custom)
    add_composed_view_hint(builder.graph_mut(), gelu, &[3, 100, 101]);

    // Down projection
    let down = builder.matmul(gelu, down_weight)?;
    Ok(down)
}
```

### 2. Compiler Integration

The hologram compiler reads the composed view hint and generates optimized backend code:

```rust
// In /hologram/crates/compiler/src/from_ir.rs (similar to SIMD hints)

if let Some(AttrValue::String(hint_type)) = ir_node.op.attrs.get("hint_type") {
    if hint_type == "composed_view" {
        if let Some(AttrValue::Ints(table_ids)) = ir_node.op.attrs.get("hint_table_ids") {
            // Generate ComposedView OpNode
            op_node = OpNode::ComposedView {
                table_ids: table_ids.iter().map(|&id| id as u32).collect(),
            };
        }
    }
}
```

### 3. Backend Execution

The hologram backend uses `resolve3`/`resolve4` or `ComposedViewBuilder`:

```rust
use hologram::lookup::{
    ElementWiseView, resolve3, SimdLookup,
    get_table_by_id, table_id,
};

fn execute_composed_activation(
    input: &[u8],
    output: &mut [u8],
    table_ids: &[u32],
) -> Result<()> {
    // For 3-stage composition, use optimized resolve3
    if table_ids.len() == 3 {
        let view1 = ElementWiseView::from_table(get_table_by_id(table_ids[0]).unwrap());
        let view2 = ElementWiseView::from_fn(|i| normalize_u8(i, eps)); // Custom op
        let view3 = ElementWiseView::from_fn(|i| scale_u8(i, factor));   // Custom op

        // Compose into single lookup table
        let composed = resolve3(view1, view2, view3);
        let simd = SimdLookup::from_view(&composed);

        // Single O(1) SIMD lookup per element
        simd.apply_batch(input, output);
    }

    Ok(())
}
```

## Fusible Patterns

### Pattern 1: GELU → LayerNorm → Scale
**Common in**: T5, BERT, GPT-2 FFN blocks
**Table IDs**: `[3, 100, 101]`
**Expected Speedup**: 2.5-3x

### Pattern 2: ReLU → Dropout (training) → Scale
**Common in**: ResNet-style transformers
**Table IDs**: `[2, 102, 101]`
**Expected Speedup**: 2x (dropout is stochastic, only fuse in inference)

### Pattern 3: SiLU → LayerNorm
**Common in**: LLaMA, Mistral FFN blocks
**Table IDs**: `[4, 100]`
**Expected Speedup**: 2x

### Pattern 4: Tanh → Clip → Scale
**Common in**: Older RNN-style models
**Table IDs**: `[1, 103, 101]`
**Expected Speedup**: 2.5x

## Detection Heuristics

To automatically detect fusible patterns in ONNX models:

```rust
/// Detect if a sequence of operations can be fused with composed views
fn detect_fusible_chain(
    graph: &onnx::GraphProto,
    start_node: &onnx::NodeProto,
) -> Option<Vec<u32>> {
    let mut table_ids = Vec::new();
    let mut current = start_node;

    // Pattern: Activation → Normalize → Scale
    match current.op_type.as_str() {
        "Gelu" => table_ids.push(3),
        "Relu" => table_ids.push(2),
        "Tanh" => table_ids.push(1),
        "Sigmoid" => table_ids.push(0),
        _ => return None,
    }

    // Check next operation
    if let Some(next) = get_next_node(graph, current) {
        match next.op_type.as_str() {
            "LayerNormalization" => table_ids.push(100),
            "BatchNormalization" => table_ids.push(100),
            _ => return Some(table_ids), // Only activation
        }

        // Check for scale/multiply
        if let Some(final_node) = get_next_node(graph, next) {
            if final_node.op_type == "Mul" || final_node.op_type == "Div" {
                table_ids.push(101);
            }
        }
    }

    // Only fuse if chain has 2+ operations
    if table_ids.len() >= 2 {
        Some(table_ids)
    } else {
        None
    }
}
```

## Custom Table IDs

For non-standard operations, use custom table IDs (100+):

| ID | Operation | Formula |
|----|-----------|---------|
| 100 | LayerNorm | `(x - mean) / sqrt(var + eps)` |
| 101 | Scale | `x * factor` |
| 102 | Dropout Mask | `x * bernoulli(p)` (inference only) |
| 103 | Clip | `clamp(x, min, max)` |

Custom operations are implemented as `ElementWiseView::from_fn`.

## Performance Validation

### Benchmark Template

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_ffn_composition(c: &mut Criterion) {
    let mut group = c.benchmark_group("ffn_composition");

    // Sequential baseline
    group.bench_function("sequential_gelu_norm_scale", |b| {
        b.iter(|| {
            let x = black_box(input.clone());
            let y = gelu(x);
            let z = layer_norm(y);
            let out = scale(z);
            black_box(out)
        })
    });

    // Composed optimized
    group.bench_function("composed_gelu_norm_scale", |b| {
        b.iter(|| {
            let x = black_box(input.clone());
            let out = composed_view.apply(x);
            black_box(out)
        })
    });

    group.finish();
}
```

### Expected Results

```
ffn_composition/sequential_gelu_norm_scale    time: [156.2 ns ...]
ffn_composition/composed_gelu_norm_scale      time: [ 52.4 ns ...]

Speedup: 156.2 / 52.4 = 2.98x
```

## Numerical Accuracy

Composed views must maintain numerical accuracy:

```rust
#[test]
fn test_composed_view_accuracy() {
    let composed = resolve3(gelu, layer_norm, scale);
    let sequential = sequential_pipeline(input);

    for (c, s) in composed.iter().zip(sequential.iter()) {
        let error = (c - s).abs();
        assert!(error < 0.001, "Accuracy error: {}", error);
    }
}
```

**Target**: Max error < 0.001 vs sequential f32 execution

## Integration Checklist

- [x] Add `add_composed_view_hint()` function
- [x] Add `has_composed_view_hint()` helper
- [x] Add `get_composed_view_table_ids()` helper
- [x] Add tests for composed view hints (7 tests passing)
- [ ] Implement compiler support for ComposedView OpNode
- [ ] Add backend execution with resolve3/resolve4
- [ ] Create FFN pattern detector for ONNX models
- [ ] Add numerical accuracy tests (<0.001 error)
- [ ] Benchmark FFN composition (target: 2-3x speedup)

## Next Steps

1. **Compiler Integration**: Modify `/hologram/crates/compiler/src/from_ir.rs` to recognize composed view hints (similar to SIMD hints)
2. **Backend Execution**: Add `ComposedView` OpNode handling in backend executor
3. **Pattern Detection**: Implement automatic detection of fusible chains in ONNX translator
4. **Validation**: Add accuracy tests and performance benchmarks

## References

- Hologram View System: `/hologram/crates/lookup/src/lib.rs`
- SIMD Integration (reference): [31-parallel-view-integration.md](../specs/plans/31-parallel-view-integration.md)
- OpHints Module: [op_hints.rs](../crates/hologram-ai-onnx/src/core/op_hints.rs)
