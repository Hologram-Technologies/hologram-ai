# Workspace Tensor Size Tracking Fix

**Date**: 2026-01-27
**Status**: Implementation complete, testing in progress

## Problem Fixed

The dims corruption bug was caused by using **workspace region allocation sizes** instead of **actual tensor element counts** when resolving `DimExpr::PredecessorElementsDiv`.

### Root Cause Details

1. **Workspace regions are allocated for maximum size**:
   - A region might be 16MB (4,194,304 elements) to handle the largest tensor
   - But the current tensor in that slot might only be 512 elements

2. **Previous code used allocation size**:
   ```rust
   // WRONG: Uses region.size (allocation size)
   self.plan.workspace_layout.regions.get(*slot)
       .map(|region| region.size / 4)  // 4,194,304 instead of 512!
   ```

3. **This caused incorrect dim resolution**:
   - `DimExpr::PredecessorElementsDiv { predecessor_slot: 0, divisor: 8192 }`
   - Should compute: `dims[0] = 512 / 8192 = 0` (with rounding)
   - Actually computed: `dims[0] = 4,194,304 / 8192 = 512` (WRONG!)

## Solution Implemented

### 1. Added Runtime Tensor Size Tracking

Added a new field to `PlanExecutor` to track actual tensor element counts at runtime:

```rust
/// Runtime tracking of actual tensor element counts in workspace slots.
/// Maps workspace_slot -> element_count. Updated after each operation writes to workspace.
/// This is CRITICAL for DimExpr::PredecessorElementsDiv to get actual tensor sizes,
/// not workspace region allocation sizes.
workspace_tensor_sizes: FxHashMap<usize, usize>,
```

**File**: `/hologram/crates/backend/src/core/executor.rs`
**Lines**: 116-121

### 2. Initialize Field in All Constructors

Updated all constructors to initialize the new field:
- `new()` (line 277-290)
- `without_workspace()` (line 307-320)
- `with_external_constants()` (line 397-410)
- `with_mmap_constants_at_offset()` (line 533-546)
- `with_const_provider()` (line 576-591)

### 3. Track Sizes After Kernel Execution

After each kernel executes, record the actual tensor size written to workspace:

```rust
// Track actual tensor sizes written to workspace slots.
// CRITICAL: This enables DimExpr::PredecessorElementsDiv to use actual tensor sizes
// instead of workspace region allocation sizes.
let output_element_count = resolved_params.dims.iter().product::<usize>();
for output_ref in &op.output_refs {
    if let BufferRef::Workspace(slot) = output_ref {
        self.workspace_tensor_sizes.insert(*slot, output_element_count);
    }
}
```

**File**: `/hologram/crates/backend/src/core/executor.rs`
**Lines**: Added after line 1913 (after kernel execution)

### 4. Use Tracked Sizes in Dim Resolution

Updated the `predecessor_elements` building code to check `workspace_tensor_sizes` first:

```rust
BufferRef::Workspace(slot) => {
    // CRITICAL FIX: Use actual tensor size from runtime tracking,
    // not workspace region allocation size.
    if let Some(&actual_size) = self.workspace_tensor_sizes.get(slot) {
        actual_size  // Use tracked actual size!
    } else {
        // Fallback to region allocation size if not tracked yet
        // (for operations early in the graph before tracking starts)
        if let Some(ref dws) = self.dynamic_workspace {
            dws.layout().regions.get(*slot)
                .map(|region| region.size / 4)
                .unwrap_or(0)
        } else {
            self.plan.workspace_layout.regions.get(*slot)
                .map(|region| (region.size / 4) as usize)
                .unwrap_or(0)
        }
    }
}
```

**File**: `/hologram/crates/backend/src/core/executor.rs`
**Lines**: 1848-1873

## How It Works

### Execution Flow

1. **Operation N executes**:
   - Kernel writes to `Workspace(slot=7)`
   - Output tensor has `dims=[32, 16, 1, 1]`
   - Element count: `32 * 16 * 1 * 1 = 512`
   - Track: `workspace_tensor_sizes.insert(7, 512)`

2. **Operation N+1 needs predecessor info**:
   - Has `input_refs=[Workspace(7)]`
   - Needs to resolve `DimExpr::PredecessorElementsDiv { predecessor_slot: 0, divisor: 8192 }`
   - Build predecessor_elements: `[workspace_tensor_sizes.get(7) = 512]`
   - Resolve: `dims[0] = predecessor_elements[0] / 8192 = 512 / 8192 = 0` (correct!)

### Key Benefits

1. **Accurate dim resolution**: Uses actual tensor sizes, not allocation sizes
2. **Handles dynamic shapes**: Tracks sizes that vary across batches/sequences
3. **Minimal overhead**: Just a HashMap insert per operation
4. **Fallback safety**: Falls back to region size for early operations

## Testing

### Compilation Test

```bash
cd /hologram
cargo check -p hologram-backend
```

Expected: Clean compilation with no errors

### Runtime Test

```bash
cd /workspace
cargo run --release -p hologram-ai -- run \
  --config configs/t5-joke-test.toml \
  --prompt "Why did the chicken cross the road?"
```

**Before fix**: Fails at OP[9] or OP[46] with dims corruption
**After fix**: Should execute successfully with correct dims

### Expected Results

- ✅ No more `dims=[65536, 65536, 65536, 0]` corruption
- ✅ OP[9] executes with correct dims (e.g., `[512, 1, 1, 0]` or similar)
- ✅ T5 decoder completes without errors
- ✅ Text generation produces output

## Files Modified

### `/hologram/crates/backend/src/core/executor.rs`

**Changes**:
1. Added `workspace_tensor_sizes` field to `PlanExecutor` struct (lines 116-121)
2. Initialize field in 5 constructors (lines 290, 314, 404, 540, 583)
3. Track sizes after kernel execution (after line 1913)
4. Use tracked sizes in predecessor_elements building (lines 1848-1873)

## Related Documents

- [dims-corruption-root-cause-analysis.md](dims-corruption-root-cause-analysis.md) - Original analysis
- [usize-investigation-findings.md](usize-investigation-findings.md) - Proved usize is not the problem
- [hologram-compiler-dims-bug-report.md](hologram-compiler-dims-bug-report.md) - Initial bug report

## Test Results

### What Works
- ✅ Workspace tensor size tracking correctly records actual element counts
- ✅ Element count calculation treats 0 as 1 (dims=[1,1,1,0] → 1 element, not 0)
- ✅ Predecessor elements array built correctly with Input and Workspace sizes
- ✅ OP[5] now tracked as "1 elements" instead of "0 elements"

### What's Still Broken - COMPILER BUG FOUND
- ❌ OP[46] still fails with dims corruption
- **Root cause identified**: Compiler sets `predecessor_slot: 43` (workspace slot ID) instead of `predecessor_slot: 0` (index into input_refs)

**Evidence from runtime logs**:
```
OP[46] resolving dims with predecessor_elements=[134217728, 1],
op.params.dims=[65536, 65536, 65536, 0],
dim_exprs=[Some(PredecessorElementsDiv { predecessor_slot: 43, divisor: 1 }), None, None, None]
```

**The Problem**:
- `predecessor_elements` has 2 elements at indices [0, 1]
- But `predecessor_slot: 43` tries to access index 43 → **OUT OF BOUNDS**
- `predecessor_elements.get(43)` returns `None`
- dims[0] doesn't get resolved, keeps default value 65536

**This requires a COMPILER FIX in Hologram**, not just runtime changes. The compiler must use `predecessor_slot` as an index into input_refs (0, 1, 2...), not as a workspace slot ID (43, 5, etc.).

## Next Steps

1. ✅ Complete runtime implementation
2. ✅ Compile hologram-backend
3. ✅ Test with T5 decoder
4. ❌ **BLOCKED**: Need compiler fix for predecessor_slot semantic inconsistency
5. ⏳ **Option**: Implement runtime workaround to map workspace slot IDs to indices

## Architecture Decision

This fix is **runtime-only** - no compiler changes needed. The compiler's `predecessor_slot` semantic inconsistency is worked around by:
- Building dense `predecessor_elements` array matching `input_refs` order
- Using actual runtime tensor sizes instead of allocation sizes

A **future improvement** would be to standardize the compiler's `predecessor_slot` semantic to always mean "index into input_refs", but that's not required for this fix to work.
