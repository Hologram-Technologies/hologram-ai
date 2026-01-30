# OP[9] Compiler Bug Fix

**Bug**: OP[9] (Reduce operation) generates `dims=[262144, 512, 1, 0]` causing 134M element calculation (536MB) that exceeds 16MB workspace allocation.

**Root Cause**: Compiler sets dims[1]=Static(512) based on compile-time shape, which doesn't match runtime flattened tensor shape.

## Problem Analysis

### What Happens at Compile Time

File: `/hologram/crates/compiler/src/pipeline/mod.rs` lines 2106-2157

```rust
// For reduce operations:
let reduce_info = match graph.get_op(node_idx) {
    Some(OpNode::ReduceMean(inner)) => Some((inner.axis, &inner.input_shape)),
    // ...
};

if let Some((axis, input_shape)) = reduce_info {
    // Calculate based on compile-time input_shape
    let outer_size = input_shape.iter().take(norm_axis).product::<usize>().max(1);
    let reduce_size = input_shape.get(norm_axis).copied().unwrap_or(1);
    let inner_size = input_shape.iter().skip(norm_axis + 1).product::<usize>().max(1);

    params.dims[0] = outer_size;
    params.dims[1] = reduce_size;
    params.dims[2] = inner_size;

    // BUG: dims[1] and dims[2] are set to STATIC values
    params.dim_exprs[0] = Some(make_size_expr(...));  // Dynamic (from predecessor)
    params.dim_exprs[1] = Some(DimExpr::Static(reduce_size));  // ❌ STATIC!
    params.dim_exprs[2] = Some(DimExpr::Static(inner_size));   // ❌ STATIC!
}
```

### What Happens at Runtime

**Compile-time assumption**:
- input_shape = [batch, 512] (2D tensor)
- axis = -1 (reduce last dimension)
- outer_size = batch (dynamic)
- reduce_size = 512 (static)
- inner_size = 1 (static)

**Runtime reality**:
- Predecessor OP[8] outputs 262,144 **flattened** elements
- dims[0] resolves to 262144 ✅ (correct via PredecessorElementsDiv)
- dims[1] stays 512 ❌ (WRONG - should be 1 for flattened)
- dims[2] stays 1 ✅ (correct)
- **Result**: 262144 * 512 * 1 = 134M elements (536MB > 16MB) → OVERFLOW

### The Core Issue

The compiler uses **compile-time shape** (`input_shape` from the ONNX graph) to calculate `reduce_size`, but at runtime the tensor may have a **different shape** (flattened) due to upstream operations.

## The Fix

### Option 1: Make reduce_size Dynamic (Recommended)

When `outer_size` is dynamic (from predecessor), `reduce_size` should also be computed dynamically:

**File**: `/hologram/crates/compiler/src/pipeline/mod.rs` lines 2146-2157

**Replace**:
```rust
// Set DimExpr for dynamic shape support.
// For reduce, outer_size is typically dynamic (batch * seq),
// while reduce_size and inner_size are typically static.
params.dim_exprs[0] = Some(make_size_expr(
    graph,
    node_idx,
    &ref_map,
    &input_slot_map,
    outer_size,
));
params.dim_exprs[1] = Some(DimExpr::Static(reduce_size));
params.dim_exprs[2] = Some(DimExpr::Static(inner_size));
```

**With**:
```rust
// Set DimExpr for dynamic shape support.
// If outer_size is dynamic (from predecessor), compute reduce_size dynamically too.
let outer_expr = make_size_expr(graph, node_idx, &ref_map, &input_slot_map, outer_size);
params.dim_exprs[0] = Some(outer_expr.clone());

// Check if outer_size is dynamic
let outer_is_dynamic = matches!(
    outer_expr,
    DimExpr::PredecessorElementsDiv { .. } | DimExpr::TotalElements { .. }
);

if outer_is_dynamic && inner_size == 1 {
    // When outer_size is dynamic and inner_size=1, compute reduce_size dynamically:
    // total_elements = outer_size * reduce_size * inner_size
    // reduce_size = total_elements / outer_size / inner_size
    // Since inner_size=1: reduce_size = total_elements / outer_size

    // Get the same predecessor used for outer_size
    match outer_expr {
        DimExpr::PredecessorElementsDiv { predecessor_slot, .. } => {
            params.dim_exprs[1] = Some(DimExpr::PredecessorElementsDiv {
                predecessor_slot,
                divisor: outer_size,  // Divide total by outer to get reduce dimension
            });
        }
        DimExpr::TotalElements { input_id } => {
            // For input tensors, we'd need a different approach
            // For now, fall back to static
            params.dim_exprs[1] = Some(DimExpr::Static(reduce_size));
        }
        _ => {
            params.dim_exprs[1] = Some(DimExpr::Static(reduce_size));
        }
    }
    params.dim_exprs[2] = Some(DimExpr::Static(inner_size));
} else if outer_is_dynamic && reduce_size == 1 {
    // When reducing on a dimension of size 1, keep it static
    params.dim_exprs[1] = Some(DimExpr::Static(reduce_size));

    // inner_size might be dynamic: inner = total / outer / reduce
    // Since reduce=1: inner = total / outer
    match outer_expr {
        DimExpr::PredecessorElementsDiv { predecessor_slot, .. } => {
            params.dim_exprs[2] = Some(DimExpr::PredecessorElementsDiv {
                predecessor_slot,
                divisor: outer_size,
            });
        }
        _ => {
            params.dim_exprs[2] = Some(DimExpr::Static(inner_size));
        }
    }
} else {
    // Static case: shapes are fully known at compile time
    params.dim_exprs[1] = Some(DimExpr::Static(reduce_size));
    params.dim_exprs[2] = Some(DimExpr::Static(inner_size));
}

tracing::debug!(
    "build_plan_ops: Reduce ({:?}) axis={}, outer_is_dynamic={}, dim_exprs={:?}",
    decision.kernel_id,
    axis,
    outer_is_dynamic,
    params.dim_exprs
);
```

### Option 2: Use Total Element Count for Flattened Tensors

Add validation to detect when the actual runtime shape doesn't match compile-time assumptions:

**File**: `/hologram/crates/compiler/src/pipeline/mod.rs` after line 2157

**Add**:
```rust
// Validate: if outer_size is dynamic and we're reducing along the last dimension
// of what might be a flattened tensor, adjust the dims calculation
if outer_is_dynamic && norm_axis == input_shape.len() - 1 {
    // Check if this looks like a flattened reduction
    // (outer_size is dynamic, reducing last dim, inner_size=1)
    if inner_size == 1 {
        tracing::warn!(
            "build_plan_ops: Reduce may have flattened input. \
             axis={}, input_shape={:?}, outer={} (dynamic), reduce={}, inner={}. \
             Setting reduce_size as dynamic to handle runtime shape variation.",
            axis,
            input_shape,
            outer_size,
            reduce_size,
            inner_size
        );
    }
}
```

### Option 3: Runtime Validation and Adjustment (Safeguard)

Add runtime checking in the executor to detect and warn about this condition:

**File**: `/hologram/crates/backend/src/core/executor.rs` in `execute_kernel()`

After dims resolution, before validation (around line 1965):

```rust
// SAFEGUARD: Detect reduce operations with suspicious dims
if op.kernel_id.category() == crate::plan::KernelId::CATEGORY_REDUCE {
    let outer = resolved_params.dims[0];
    let reduce = resolved_params.dims[1];
    let inner = resolved_params.dims[2];

    // Get actual predecessor element count
    if !op.input_refs.is_empty() {
        if let Some(actual_elements) = self.execution_context.get_element_count(
            &match &op.input_refs[0] {
                BufferRef::Workspace(slot) =>
                    crate::core::context::BufferLocation::Workspace(*slot),
                BufferRef::Input(id) =>
                    crate::core::context::BufferLocation::Input(*id),
                _ => continue,
            }
        ) {
            let calculated = outer * reduce * inner;
            if calculated != actual_elements && calculated > actual_elements {
                tracing::warn!(
                    "OP[{}] Reduce: dims=[{}, {}, {}] multiply to {} but predecessor has {} elements. \
                     This indicates a compiler bug. Adjusting dims to match actual input.",
                    op_idx,
                    outer, reduce, inner,
                    calculated,
                    actual_elements
                );

                // Adjust: if inner=1, set reduce = actual / outer
                if inner == 1 && outer > 0 {
                    let corrected_reduce = actual_elements / outer;
                    resolved_params.dims[1] = corrected_reduce;
                    tracing::warn!(
                        "OP[{}] Corrected dims[1] from {} to {} (actual_elements={} / outer={})",
                        op_idx,
                        reduce,
                        corrected_reduce,
                        actual_elements,
                        outer
                    );
                }
            }
        }
    }
}
```

## Recommended Approach

**Implement all three options**:

1. **Option 1** (Compiler fix): Make reduce_size dynamic when outer_size is dynamic
   - Fixes the root cause
   - Handles dynamic shapes correctly
   - Most correct solution

2. **Option 2** (Compiler warning): Add validation and warnings
   - Helps detect similar issues
   - Provides better error messages

3. **Option 3** (Runtime safeguard): Add runtime correction
   - Safety net for existing compiled models
   - Prevents crashes until recompilation
   - Can be removed after all models are recompiled

## Testing

After implementing the fix, test with T5 decoder:

```bash
# Recompile with fixed compiler
cargo run --release -p hologram-ai -- compile \
    /workspace/models/t5-small/decoder_model.onnx \
    --output /tmp/decoder_fixed.holo

# Run test
export RUST_LOG=hologram_backend::core::executor=warn
cargo run --release -p hologram-ai -- run \
    --config examples/T5/t5.toml
```

**Expected result**:
- OP[9] should execute successfully
- dims should be [262144, 1, 1, 0] instead of [262144, 512, 1, 0]
- No buffer overflow error

## Related Issues

This fix addresses:
- OP[9] buffer overflow in T5 decoder
- General issue with reduce operations on flattened tensors
- Dynamic shape handling in reduction operations

## Files to Modify

1. `/hologram/crates/compiler/src/pipeline/mod.rs` - Lines 2146-2170 (Option 1 & 2)
2. `/hologram/crates/backend/src/core/executor.rs` - After line 1965 (Option 3)

---

**Priority**: HIGH - Blocks T5 execution
**Complexity**: Medium - Requires careful handling of dynamic dims
**Risk**: Low - Well-isolated change with clear test case
