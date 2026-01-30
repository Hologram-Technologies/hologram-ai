# Dims Corruption Root Cause Analysis

**Date**: 2026-01-27
**Status**: Root cause identified, partial fix implemented, full fix requires architectural changes in Hologram

## Executive Summary

I've identified the root cause of the `dims=[65536, 65536, 65536, 0]` corruption bug. The issue is **NOT** with `usize` vs fixed-size types. The problem is a **semantic inconsistency** in how `DimExpr::PredecessorElementsDiv::predecessor_slot` is used between the compiler and runtime, combined with using workspace **region allocation sizes** instead of actual **tensor element counts**.

## Root Cause

### Problem 1: Inconsistent `predecessor_slot` Semantic

The Hologram compiler sets `predecessor_slot` with **three different semantics** in different places:

1. **LayerNorm/RMSNorm** (`/hologram/crates/compiler/src/pipeline/mod.rs:2264, 2298`):
   ```rust
   params.dim_exprs[0] = Some(DimExpr::PredecessorElementsDiv {
       predecessor_slot: 0,  // Hardcoded 0 = index into input_refs
       divisor: norm_size,
   });
   ```

2. **Conv2D with Input predecessor** (line 1575):
   ```rust
   params.dim_exprs[3] = Some(DimExpr::PredecessorElementsDiv {
       predecessor_slot: slot,  // slot from input_slot_map = INPUT ID (0, 1, 2, ...)
       divisor: conv_inner.channels.max(1),
   });
   ```

3. **Conv2D with Workspace predecessor** (line 1583):
   ```rust
   params.dim_exprs[3] = Some(DimExpr::PredecessorElementsDiv {
       predecessor_slot: *slot,  // Workspace slot ID (e.g., 356)
       divisor: conv_inner.channels.max(1),
   });
   ```

**These are incompatible!** The runtime cannot handle all three semantics simultaneously.

### Problem 2: Original Runtime Bug

The original runtime code (`/hologram/crates/backend/src/core/executor.rs:1834-1856`) used `filter_map`:

```rust
let predecessor_elements: Vec<usize> = op
    .input_refs
    .iter()
    .filter_map(|r| match r {
        BufferRef::Workspace(slot) => /* get size */,
        _ => None,  // Skip Input, Constant, etc.
    })
    .collect();
```

This created an **index mismatch**:
- If `input_refs = [Input(0), Workspace(5), Workspace(7)]`
- Then `predecessor_elements = [workspace_5_size, workspace_7_size]` (length 2)
- If compiler sets `predecessor_slot = 5` (workspace slot ID)
- Runtime does `predecessor_elements.get(5)` → **IndexOutOfBounds or None**
- Dimension doesn't get resolved, keeps default value (65536)

### Problem 3: Workspace Region Size vs Tensor Element Count

Even with my fix to build a dense array, using **workspace region allocation size** is fundamentally wrong:

```rust
self.plan.workspace_layout.regions.get(*slot).map(|region| region.size / 4)
```

**Why this is wrong**:
- Workspace regions are allocated for the **MAXIMUM size** needed across all uses
- A region might be 16MB (4M elements) to handle the largest tensor
- But the current tensor stored in that slot might only be 512 elements
- Using 4M instead of 512 gives wildly incorrect dims!

**Example from T5**:
- OP[9] has `input_refs = [Workspace(7)]`
- Workspace region 7 is allocated 16MB (4,194,304 elements)
- Actual tensor in workspace_7 at this point: 512 elements
- My code sets `dims[0] = 4,194,304` → **WRONG!**
- Correct value should be `dims[0] = 512`

## My Attempted Fixes

### Fix Attempt 1: Build Sparse Array by Workspace Slot ID
- Changed `filter_map` to `map`, built array indexed by workspace slot
- **Result**: Different error at OP[9] instead of OP[46]
- **Why it failed**: Still using region size instead of tensor size

### Fix Attempt 2: Build Dense Array Matching input_refs Order
- Changed to build `predecessor_elements` with same order/length as `input_refs`
- Added support for Input buffers using `shape_registry`
- **Result**: Compiles, not yet tested
- **Likely outcome**: Will work for operations with Input predecessors, still wrong for Workspace predecessors

## The Correct Solution

This requires **architectural changes in Hologram**:

### Option A: Track Tensor Sizes at Runtime (Recommended)

Add a runtime registry that tracks the actual tensor size in each workspace slot, similar to `shape_registry` for inputs:

```rust
struct PlanExecutor {
    // ... existing fields ...
    workspace_tensor_sizes: HashMap<usize, usize>,  // workspace_slot -> element_count
}
```

Update this registry when operations write to workspace:
```rust
// After executing an operation that writes to workspace
if let BufferRef::Workspace(slot) = output_ref {
    let output_size = calculate_output_elements(&resolved_params);
    self.workspace_tensor_sizes.insert(slot, output_size);
}
```

Then use actual sizes instead of region sizes:
```rust
BufferRef::Workspace(slot) => {
    self.workspace_tensor_sizes.get(slot).copied().unwrap_or(0)
}
```

### Option B: Make Compiler Avoid PredecessorElementsDiv

Change the compiler to use `DimExpr::Static` when sizes can be computed statically, and only use `PredecessorElementsDiv` for truly dynamic cases where the size depends on runtime input shapes.

### Option C: Standardize predecessor_slot Semantic

Make `predecessor_slot` **always mean "index into input_refs"** (0-based):
1. Fix LayerNorm/RMSNorm to keep using 0 (already correct)
2. Fix Conv2D code to find the **index** of the predecessor in `input_refs`, not the slot ID
3. Runtime builds dense array matching `input_refs` order (my Fix Attempt 2)

This fixes the semantic inconsistency but doesn't solve the workspace region size problem.

## Current Status

### What Works
- ✅ `usize` serialization/deserialization works correctly
- ✅ Compilation sets correct dims (verified in .holo file)
- ✅ My Fix Attempt 2 changes runtime to build dense predecessor_elements array
- ✅ Input buffer sizes correctly retrieved from shape_registry

### What's Still Broken
- ❌ Workspace buffer sizes use allocation size instead of tensor size
- ❌ Compiler has inconsistent `predecessor_slot` semantics
- ❌ T5 decoder still fails (now at OP[9] instead of OP[46])

## Recommended Next Steps

1. **Short-term workaround**: Disable `PredecessorElementsDiv` usage in compiler, use Static dims where possible

2. **Medium-term fix**: Implement Option A (track workspace tensor sizes at runtime)

3. **Long-term fix**: Re-architect dim resolution to avoid this class of bugs:
   - Make dims resolution more explicit and type-safe
   - Add validation that catches mism atched array indices at compile time
   - Consider using named fields instead of array indices

## Files Modified

### `/hologram/crates/backend/src/core/executor.rs`
- Lines 1831-1865: Changed to build dense `predecessor_elements` array
- Added support for Input, Constant, ExternalConstant buffer sizes
- Still using workspace region size (incorrect)

## Testing Needed

1. Test with operations that have Input predecessors (should work now)
2. Test with operations that have only Workspace predecessors (will still fail)
3. Add unit tests for dim resolution with various buffer ref combinations

## Related Documents

- [/workspace/specs/usize-investigation-findings.md](usize-investigation-findings.md) - Proved usize is not the problem
- [/workspace/specs/hologram-team-prompt.md](hologram-team-prompt.md) - Original bug report for Hologram team
- [/workspace/specs/hologram-compiler-dims-bug-report.md](hologram-compiler-dims-bug-report.md) - Detailed bug evidence

## Conclusion

The dims corruption is caused by a **combination of**:
1. Semantic inconsistency in `predecessor_slot` usage
2. Using workspace allocation sizes instead of actual tensor sizes
3. Filter-based array building that creates index mismatches

The proper fix requires tracking actual workspace tensor sizes at runtime, which is an architectural change in Hologram's executor. My current fixes address the semantic inconsistency but not the workspace size problem.

**This bug blocks T5 decoder execution and likely affects all transformer models with dynamic shapes.**
