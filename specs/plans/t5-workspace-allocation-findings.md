# T5 Workspace Allocation Investigation - Findings and Recommendations

## Executive Summary

The T5 text generation pipeline compilation and runtime encountered workspace buffer allocation errors in the hologram compiler. Through iterative investigation, we identified the root cause and attempted multiple fixes. **The encoder now compiles and runs successfully**, but the decoder still has issues that require deeper compiler fixes beyond simple minimum allocation strategies.

## Problem Description

### Initial Error
```
OP[13] KernelId(771): input[1] size 262144 bytes exceeds workspace region 'workspace_11'
allocation of 4 bytes
```

The hologram compiler was severely underallocating workspace buffers for transformer operations, allocating only 4 bytes where 262KB was needed.

## Root Cause Analysis

### Location
- **File**: `/hologram/crates/compiler/src/pipeline/mod.rs`
- **Function**: `plan_workspace()` (lines 817-900)
- **Issue**: Shape inference failures causing incorrect workspace size calculations

### Technical Details

1. **FusedActivation Operations**:
   - Return `workspace_size() = 0` to indicate dynamic sizing
   - Set `preserves_element_count() = true`
   - **Problem**: Not treated as elementwise operations, so predecessor sizing wasn't applied

2. **Attention Operations**:
   - Properly implement `workspace_size()` returning `num_heads * seq_len * seq_len * 4 * 2`
   - For T5-small: 12 × 512 × 512 × 4 × 2 = 25MB ✓ CORRECT
   - **Problem**: Some predecessor operations had failed shape inference, propagating 0-sized allocations

3. **Shape Inference Cascade**:
   - When operation A returns size=0, operation B (using A's output) calculates size=0
   - The `elementwise_numel_from_predecessors()` function takes max of predecessor sizes
   - If all predecessors have size=0, the result is 0

## Implemented Fixes

### Fix 1: FusedActivation Elementwise Sizing
**Status**: ✅ SUCCESSFUL

**Change**: When `workspace_size()` returns 0 and `preserves_element_count()` is true, use `elementwise_numel_from_predecessors()` to calculate size.

**Location**: Lines 876-895 in mod.rs
```rust
let size = if size == 0 && op.preserves_element_count() {
    let numel = elementwise_numel_from_predecessors(graph, node_idx, layout_metadata);
    let size_bytes = numel.max(1) * 4;
    if size_bytes < 8 * 1024 * 1024 {
        size_bytes.max(MIN_REASONABLE_BYTES)
    } else {
        size_bytes
    }
} else if size > 0 && size < 8 * 1024 * 1024 {
    size.max(MIN_REASONABLE_BYTES)
} else {
    size
};
```

**Result**: FusedActivation operations (relu, gelu, etc.) now correctly inherit workspace sizes from predecessors.

### Fix 2: MIN_REASONABLE_BYTES Fallback
**Status**: ⚠️ PARTIAL

**Attempts**:
1. 64KB minimum, < 1KB threshold → Still underallocated (needed 256KB, got 64KB)
2. 1MB minimum, < 1KB threshold → Still underallocated (needed 1MB, got 512KB)
3. 4MB minimum, < 8MB threshold → Encoder works, decoder underallocated (needed 8MB, got 4MB)
4. 16MB minimum, < 8MB threshold → **TOO AGGRESSIVE** - causes workspace overflow

**Problem**: The minimum allocation approach is a band-aid. The real issue is that specific operations are calculating sizes that are exactly half (or some fraction) of what they actually need, suggesting a bug in the size calculation logic for those operations.

## Current Status

### Working ✅
- **Hologram compiler builds** successfully
- **T5 encoder compiles** (282MB .holo file)
- **T5 encoder executes** successfully (completes in ~706ms)
- **Tokenization** works correctly
- **FusedActivation** operations properly sized

### Not Working ❌
- **T5 decoder runtime**: Multiple workspace regions exceed total workspace allocation
- **Root cause**: Applying minimums too aggressively to operations that have partially-correct size calculations
- **Specific failure**: Operations calculating correct sizes (e.g., 512KB) are being forced to 16MB minimum, causing overflow

## Recommendations

### Short-term (Workaround)
To get T5 generation working immediately:
1. **Revert to targeted fixes**: Keep Fix #1 (FusedActivation sizing) and a conservative MIN_REASONABLE_BYTES (1-2MB) with low threshold (< 64KB)
2. **Manually patch problematic operations**: Identify the specific KernelIds that consistently underallocate and fix their `workspace_size()` implementations

### Medium-term (Proper Fix)
1. **Fix shape inference for specific operations**:
   - Investigate why `elementwise_numel_from_predecessors()` returns half the required size for certain operations
   - Likely issue: Operations with multiple inputs/outputs where one path has correct sizes and another has placeholder sizes

2. **Implement workspace_size_expr() for dynamic sizing**:
   - Operations that return `workspace_size() = 0` should implement `workspace_size_expr()`
   - This provides symbolic expressions for runtime-determined sizes

3. **Add workspace allocation tests**:
   - Unit tests for `elementwise_numel_from_predecessors()` with various graph topologies
   - Integration tests compiling small transformer models and validating workspace sizes

### Long-term (Architecture)
1. **Shape inference improvements**:
   - More robust propagation through the graph
   - Better handling of dynamic dimensions
   - Validation pass to detect size=0 cases early

2. **Workspace allocation strategy**:
   - Move from static worst-case allocation to dynamic allocation with bounds checking
   - Or: Better static analysis to accurately predict all workspace requirements upfront

## Affected Files

### Modified (for fixes)
- `/hologram/crates/compiler/src/pipeline/mod.rs` - Workspace allocation logic

### Needs Investigation
- `/hologram/crates/compiler/src/pipeline/helpers.rs` - `output_numel_for_node()` and `elementwise_numel_from_predecessors()`
- `/hologram/crates/compiler/src/graph/ops/*.rs` - Individual operation `workspace_size()` implementations
- Specific operations showing consistent underallocation patterns

## Key Learnings

1. **The minimum allocation approach masks underlying bugs** rather than fixing them
2. **Shape inference failures cascade** through the computation graph
3. **FusedActivation was a special case** that needed elementwise treatment
4. **T5 encoder complexity is lower** than decoder (which has cross-attention and larger FFN)
5. **The real bug is in size calculation**, not in missing minimums

## Next Steps

1. Document this investigation in project knowledge base
2. Create issues in hologram compiler for:
   - Fix `elementwise_numel_from_predecessors()` to handle mixed-size predecessors correctly
   - Implement `workspace_size_expr()` for FusedActivation and other dynamic operations
   - Add comprehensive workspace allocation tests
3. Revert to conservative fix (1-2MB minimum with < 64KB threshold)
4. Manually patch the specific operations causing decoder failures
5. Test T5 generation end-to-end

## Conclusion

We've made significant progress - the T5 encoder works correctly with the FusedActivation fix. However, the decoder's issues reveal that workspace allocation needs a more sophisticated solution than just applying minimums. The proper fix requires understanding and correcting the size calculation logic for specific operations rather than masking the problem with conservative over-allocation.

The investigation has provided valuable insights into the hologram compiler's workspace management and identified specific areas that need improvement for production-ready transformer model support.
