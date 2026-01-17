# DimExpr Support for All Kernel Parameters

## Goal

Enable dynamic shape support in kernel parameters by propagating `DimExpr` from IR through the compiler pipeline. This allows ONNX models with symbolic dimensions (batch size, sequence length, etc.) to resolve correctly at runtime.

## Background

### Current State

- `KernelParams` already has `dim_exprs: [Option<DimExpr>; 4]` field
- Only LayerNorm and RMSNorm populate `dim_exprs`
- All other ops store concrete `usize` in `params.dims`, losing symbolic info
- Runtime resolution infrastructure exists (`resolve_dims()`, `resolve_dims_with_predecessors()`)

### DimExpr Variants

Defined in `crates/ir/src/shape.rs`:

```rust
pub enum DimExpr {
    Static(usize),                           // Known constant value
    InputRef { input_id, dim_index },        // Reference input shape dimension
    TotalElements { input_id },              // Product of all input dims
    ProductOfDims { ... },                   // Product of two specific dims
    TotalElementsDiv { input_id, divisor },  // Total elements / constant
    DimDiv { input_id, dim_index, divisor }, // Single dim / constant
    PredecessorElementsDiv { slot, divisor }, // From predecessor buffer
}
```

## Design Decision

**Approach: Always use DimExpr**

Populate `dim_exprs` for every operation, using `DimExpr::Static(n)` for known values. This is simpler than conditional dual-mode and has negligible overhead.

```rust
// Every op handler does this:
params.dim_exprs[0] = Some(make_expr_or_static(graph, node_idx, ...));
params.dims[0] = static_fallback;  // Backward compat
```

Where `make_expr_or_static()` returns:
- `DimExpr::Static(n)` for compile-time known values
- `DimExpr::InputRef {...}` for symbolic input dimensions
- `DimExpr::TotalElements {...}` for total element counts
- `DimExpr::PredecessorElementsDiv {...}` for derived sizes

## Implementation

### Phase 1: Infrastructure

**File**: `crates/compiler/src/pipeline.rs`

Add helper functions:

```rust
/// Build input NodeId to runtime slot mapping.
fn build_input_slot_map(
    graph: &CompileGraph,
    layout_metadata: &LayoutMetadata,
) -> HashMap<NodeId, usize> {
    let mut map = HashMap::new();
    for (slot, &node_id) in layout_metadata.input_node_ids.iter().enumerate() {
        map.insert(node_id, slot);
    }
    map
}

/// Create DimExpr for an operation's element count.
/// Returns Static for known values, InputRef/TotalElements for dynamic.
fn make_size_expr(
    graph: &CompileGraph,
    node_idx: NodeId,
    ref_map: &HashMap<NodeId, BufferRef>,
    input_slot_map: &HashMap<NodeId, usize>,
    static_fallback: usize,
) -> DimExpr {
    // 1. Check if this node or predecessor is a model input
    let predecessors = graph.predecessors(node_idx);
    for pred in &predecessors {
        if let Some(&slot) = input_slot_map.get(pred) {
            // Predecessor is a model input - use TotalElements
            return DimExpr::TotalElements { input_id: slot };
        }
        // Check if predecessor was remapped to Input buffer
        if let Some(BufferRef::Input(id)) = ref_map.get(pred) {
            return DimExpr::TotalElements { input_id: *id };
        }
    }

    // 2. Check if predecessor has known workspace slot
    for pred in &predecessors {
        if let Some(BufferRef::Workspace(slot)) = ref_map.get(pred) {
            return DimExpr::PredecessorElementsDiv {
                predecessor_slot: *slot,
                divisor: 1,
            };
        }
    }

    // 3. Fall back to static value
    DimExpr::Static(static_fallback)
}
```

### Phase 2: Unary/Activation Operations

**Operations**: ReLU, Sigmoid, Tanh, GELU, Abs, Neg, Exp, Log, Sqrt, etc.

**Location**: `pipeline.rs` lines ~2571-2700

```rust
// Before:
params.dims[0] = total_size;

// After:
params.dim_exprs[0] = Some(make_size_expr(
    graph, node_idx, ref_map, &input_slot_map, total_size
));
params.dims[0] = total_size;
```

### Phase 3: Binary Elementwise Operations

**Operations**: Add, Sub, Mul, Div, Pow, etc.

**Dims layout**:
- `dims[0]` = output element count
- `dims[1]` = second input size (for broadcast)
- `dims[2]` = first input size (for broadcast)

```rust
// For broadcast, use the larger input's size expression
let (larger_pred, smaller_pred) = if first_size >= second_size {
    (predecessors[0], predecessors[1])
} else {
    (predecessors[1], predecessors[0])
};

params.dim_exprs[0] = Some(make_size_expr(
    graph, larger_pred, ref_map, &input_slot_map, output_size
));
params.dims[0] = output_size;
```

### Phase 4: MatMul Operations

**Location**: `pipeline.rs` lines ~2540-2549

**Dims layout**:
- `dims[0]` = M (batch * rows, typically from input)
- `dims[1]` = K (inner dimension, from weight)
- `dims[2]` = N (columns, from weight)
- `dims[3]` = output elements (M * N)

```rust
// M typically comes from dynamic input (batch * seq_len)
if let Some(&slot) = input_slot_map.get(&lhs_pred) {
    // LHS is model input - M is dynamic
    params.dim_exprs[0] = Some(DimExpr::TotalElementsDiv {
        input_id: slot,
        divisor: *k,  // Total elements / K = M
    });
    params.dim_exprs[3] = Some(DimExpr::ProductOfDims {
        input_id_a: slot,
        dim_a: 0,
        input_id_b: slot,
        dim_b: 1,
    });
}
params.dims[0] = *m;
params.dims[1] = *k;
params.dims[2] = *n;
params.dims[3] = m * n;
```

### Phase 5: Gather/Embedding Operations

**Location**: `pipeline.rs` lines ~2637-2671

```rust
// num_indices comes from indices tensor (model input)
if let Some(&slot) = input_slot_map.get(&indices_pred) {
    params.dim_exprs[2] = Some(DimExpr::TotalElements { input_id: slot });
}
params.dims[0] = axis_size;
params.dims[1] = inner_size;
params.dims[2] = num_indices;
params.dims[3] = num_indices * inner_size;
```

### Phase 6: Softmax Operations

**Location**: `pipeline.rs` lines ~3172-3304

```rust
// batch_size = total_elements / softmax_size
params.dim_exprs[0] = Some(DimExpr::TotalElementsDiv {
    input_id: input_slot,
    divisor: softmax_size,
});
params.dim_exprs[1] = Some(DimExpr::Static(softmax_size));
params.dims[0] = batch_size;
params.dims[1] = softmax_size;
```

### Phase 7: Reduce Operations

```rust
// Input size from predecessor
params.dim_exprs[0] = Some(make_size_expr(
    graph, node_idx, ref_map, &input_slot_map, input_size
));
```

### Phase 8: Conv2D Operations (Deferred)

Conv2D with dynamic spatial dimensions would need new `DimExpr` variants. This can be deferred if ONNX models typically have static spatial dimensions.

## Files to Modify

| File | Changes |
|------|---------|
| `crates/compiler/src/pipeline.rs` | Add helpers, update all op handlers |
| `crates/ir/src/shape.rs` | Possibly add ConvOutput variant |
| `crates/backend/src/core/executor.rs` | Verify resolution called for all ops |

## Runtime Resolution

The executor already handles resolution in `executor.rs`:

```rust
fn resolve_dim_expr(&self, expr: &DimExpr) -> BackendResult<usize> {
    match expr {
        DimExpr::Static(n) => Ok(*n),
        DimExpr::InputRef { input_id, dim_index } => {
            self.shape_registry
                .get(&(*input_id as u64))
                .and_then(|shape| shape.get(*dim_index).copied())
                .ok_or_else(|| ...)
        }
        DimExpr::TotalElements { input_id } => {
            self.shape_registry
                .get(&(*input_id as u64))
                .map(|shape| shape.iter().product())
                .ok_or_else(|| ...)
        }
        // ... other variants
    }
}
```

Ensure `resolve_dims()` is called before kernel execution for ops with dynamic dims.

## Verification

1. `cargo build -p hologram-compiler` - must succeed
2. `cargo clippy -p hologram-compiler -- -D warnings` - no warnings
3. `cargo test -p hologram-compiler` - all tests pass
4. Create test with symbolic batch dimension
5. Test with hologram-ai-onnx T5 encoder with varying batch sizes
6. Benchmark static models to verify no regression

## Migration Path

1. **Phase 1**: Add infrastructure helpers
2. **Phase 2-3**: Unary + Binary ops (covers most activations and residual adds)
3. **Phase 4**: MatMul (critical for attention layers)
4. **Phase 5-6**: Gather + Softmax (embedding and attention)
5. **Phase 7-8**: Reduce + Conv2D (lower priority)

Each phase is independently deployable and testable.
