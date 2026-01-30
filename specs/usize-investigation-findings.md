# Investigation: Is `usize` Causing the dims Corruption?

**Date**: 2026-01-27
**Question**: Is the dims corruption bug caused by using `usize` instead of fixed-size types like `u64`?

## Answer: NO

Through systematic testing, we've proven that **`usize` is NOT the cause** of the dims corruption.

## Evidence

### 1. rkyv Handles `usize` Correctly

**Test**: Created structs with `[usize; 4]` arrays and serialized/deserialized with rkyv.

**Results**:
- ✅ rkyv compresses `usize` (8 bytes on 64-bit) → u32 (4 bytes) during serialization
- ✅ Deserialization correctly expands u32 → usize (8 bytes)
- ✅ Works with simple structs: `struct SimpleDims { dims: [usize; 4] }`
- ✅ Works with complex structs: 14 usize values across multiple fields
- ✅ Works with `Option<enum>`: `dim_exprs: [Option<DimExpr>; 4]`
- ✅ Both 16-byte and 64-byte alignment work correctly

**Example**:
```
Original dims: [32, 8, 256, 2048]
Serialized: 16 bytes total (4 bytes × 4 values)
Hex: 20 00 00 00 08 00 00 00 00 01 00 00 00 08 00 00
Restored dims: [32, 8, 256, 2048]
✓ Perfect match
```

### 2. The Corruption is NOT in the .holo File

**Test**: Searched the compiled decoder.holo file (465MB) for the corrupted dims pattern.

**Results**:
- ❌ Pattern `[65536, 65536, 65536, 0]` does **NOT** exist in the file
- ✅ 65536 appears 4464 times but in different patterns (e.g., `[65536, 0, 65536, 0]`)
- ✅ Compilation correctly writes dims to disk

**Conclusion**: The corruption happens **after** the file is written, during loading or runtime execution.

### 3. Serialization/Deserialization Process is Correct

**Hologram's deserialization** (`/hologram/crates/compiler/src/api.rs:1259-1264`):
```rust
fn deserialize_plan(bytes: &[u8]) -> HoloPlanResult<SerializableBackendPlan> {
    let mut aligned: rkyv::util::AlignedVec<16> = rkyv::util::AlignedVec::new();
    aligned.extend_from_slice(bytes);
    rkyv::from_bytes::<SerializableBackendPlan, rkyv::rancor::Error>(&aligned)
        .map_err(|e| HoloPlanError::Deserialize(e.to_string()))
}
```

**Tested with**:
- ✅ 16-byte alignment (Hologram's default)
- ✅ 64-byte alignment (more conservative)
- ✅ No alignment (direct from_bytes on raw bytes)

All three work correctly - alignment is NOT the issue.

### 4. Where the Corruption Actually Happens

**Evidence from compilation vs runtime**:

| Stage | dims Values |
|-------|-------------|
| During compilation | ✅ Correct: `[32, 8, 1, 8]`, `[32128, 512, 512, 262144]`, etc. |
| In .holo file | ✅ Correct: No corrupted patterns found |
| At runtime (OP[46]) | ❌ Corrupted: `[65536, 65536, 65536, 0]` |

**Conclusion**: Corruption happens during **runtime dimension resolution** in the executor.

## Root Cause: Runtime dim_expr Resolution Bug

The corruption occurs in the Hologram backend executor, specifically in:
- **File**: `/hologram/crates/backend/src/core/executor.rs`
- **Lines**: 1857-1866 (dim resolution before kernel execution)
- **Methods**: `KernelParams::resolve_dims()` or `resolve_dims_with_predecessors()`

### Why 65536?

The value 65536 appears in the compiler as a **default fallback**:
- **File**: `/hologram/crates/compiler/src/pipeline/mod.rs:1885`
- **Code**: `total_size = 65536;  // Default: 1 * 128 * 512 (T5 encoder output)`

This default is used for **activation operations** when size inference fails. The corrupted Gather operation is somehow reading dims values that were meant for activation operations, suggesting:
1. Memory corruption during dim_expr resolution
2. Reading from wrong array index/offset
3. Uninitialized memory in the resolve_dims() path

## Why `usize` vs Fixed-Size Doesn't Matter

**Platform dependency is not the issue because**:
1. Both compilation and execution happen on the same 64-bit Linux system
2. rkyv's u32 compression works correctly in both directions
3. The corruption pattern (65536 repeated) doesn't match any serialization artifact
4. Fixed-size types like `u64` or `u32` would have the same corruption bug

## Recommendations

### 1. Fix the Real Bug (High Priority)

Investigate runtime dim_expr resolution in the Hologram backend executor:
- Add logging before/after `resolve_dims()` to track when corruption occurs
- Check array bounds and memory safety in dimension resolution
- Verify that dim_exprs are resolving to correct values for Gather operations

### 2. Consider Using Fixed-Size Types (Low Priority)

While NOT the cause of this bug, using `u64` instead of `usize` has benefits:
- ✅ Explicit about data size (always 8 bytes)
- ✅ Clearer intent for serialization formats
- ✅ Avoids theoretical 32-bit vs 64-bit cross-compilation issues
- ✅ Slightly more efficient serialization (rkyv doesn't need to compress)

**But this won't fix the current bug** - the corruption is in the dim resolution logic, not the type system.

### 3. Add Validation (Medium Priority)

Add runtime validation to catch corruption early:
```rust
// After dim resolution
for (i, &dim) in resolved_params.dims.iter().enumerate() {
    if dim > 1_000_000 || dim == 0 {  // Sanity check
        tracing::error!(
            "OP[{}] suspicious dims[{}]={} (all dims={:?})",
            op_idx, i, dim, resolved_params.dims
        );
    }
}
```

## Next Steps

1. **Add detailed logging** to track dim resolution in executor
2. **Profile memory access** during `resolve_dims()` to find corruption point
3. **Review Gather dim_expr setup** - line 1779 in mod.rs sets static value for dynamic dimension
4. **Test with smaller models** to isolate the bug

## Files Tested

- **Test code**: `/workspace/rkyv_test/src/main.rs`
- **Models**: T5-small decoder (`/workspace/models/t5-small/decoder_model.onnx`)
- **Compiled**: `/workspace/models/t5-small/compiled/decoder.holo` (465MB)

## Conclusion

**The dims corruption is NOT caused by using `usize` instead of fixed-size types.**

The bug is in Hologram's runtime dimension resolution logic in the backend executor. Using `u64` instead of `usize` would NOT fix this bug, though it might be a good practice for other reasons.

The real fix requires debugging the `resolve_dims()` and `resolve_dims_with_predecessors()` methods in the Hologram backend to understand why Gather operations are getting corrupted dimension values at runtime.
