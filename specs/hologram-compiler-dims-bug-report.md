# Hologram Compiler Bug Report: Corrupted dims Array in BackendOp

**Date**: 2026-01-27
**Severity**: High - Blocks T5 decoder execution
**Component**: hologram-compiler (BackendOp dims initialization)

## Summary

The hologram compiler is generating corrupted `dims` arrays in `BackendOp.params.dims` during T5 decoder compilation. The corrupted values (`[65536, 65536, 65536, 0]`) cause runtime buffer overflow errors when the backend executor tries to validate workspace allocation.

## Reproduction Steps

1. **Compile T5 decoder**:
   ```bash
   cargo run --release -- compile \
     /workspace/models/t5-small/decoder_model.onnx \
     --output /tmp/decoder.holo
   ```

2. **Execute compiled model**:
   ```bash
   cargo run --release -- run \
     --config examples/T5/t5.toml \
     --prompt "translate English to French: Hello"
   ```

3. **Observe error at OP[97]**:
   ```
   ERROR: Kernel execution failed at OP[97] kernel=KernelId(769)
   dims=[65536, 65536, 65536, 0]
   input[1] size 262144 bytes exceeds workspace region 'workspace_22'
   allocation of 2048 bytes
   ```

## Expected Behavior

The `dims` array should contain the actual tensor dimensions for the operation:
- For a 1D tensor of 65536 elements: `[65536, 1, 1, 1]`
- For a 2D tensor: `[dim0, dim1, 1, 1]`
- All four elements should be valid, non-zero dimensions

## Actual Behavior

The `dims` array contains corrupted values:
- `dims=[65536, 65536, 65536, 0]` at OP[97] (Gather operation, KernelId 769)
- `dims=[65536, 0, 0, 0]` at multiple activation operations throughout the model
- The value `65536` (0x10000) appears to be an uninitialized sentinel value
- The fourth element is often `0`, which is invalid for a dimension

## Evidence

### Error Message
```
OP[97] KernelId(769): input[1] size 262144 bytes exceeds workspace region
'workspace_22' allocation of 2048 bytes. dims=[65536, 65536, 65536, 0].
This indicates a compiler bug in shape inference or buffer allocation.
```

### Patterns Observed During Execution
Multiple operations show suspicious dims patterns:
- Encoder: `dims=[65536, 0, 0, 0]` in activation operations (OP[2], OP[31], OP[39], OP[49], OP[54])
- Decoder: `dims=[65536, 65536, 65536, 0]` at OP[97] (failure point)
- Decoder: `dims=[65536, 0, 0, 0]` at OP[65], OP[74], OP[79]

### Buffer Size Calculation
The error shows:
- **Requested**: 262144 bytes = 65536 floats × 4 bytes/float
- **Allocated**: 2048 bytes
- **Actual need**: Input should be 65536 elements (1D), not 65536³ elements

This confirms dims[0] = 65536 is correct, but dims[1], dims[2], dims[3] are corrupted.

## Source Code Location

The corrupted dims are read from the compiled BackendPlan:

**File**: `/hologram/crates/backend/src/core/executor.rs:1316`
```rust
return Err(BackendError::invalid_config(format!(
    "OP[{}] {:?}: input size {} bytes exceeds workspace region '{}' \
     allocation of {} bytes. dims={:?}. \
     This indicates a compiler bug in shape inference or buffer allocation.",
    op_idx, op.kernel_id, expected_bytes, region_name, region_size,
    op.params.dims  // <-- Corrupted values come from here
)));
```

The dims are SET during compilation, likely in:

**File**: `/hologram/crates/compiler/src/pipeline/mod.rs`
- Function: `build_plan_ops()`
- Around lines where `BackendOp::params::dims` is assigned

## Root Cause Hypothesis

Based on the pattern of corruption:

1. **Hypothesis 1: Uninitialized Array**
   - The `dims: [usize; 4]` array is not fully initialized before use
   - Value `65536` might be a default/sentinel that isn't being overwritten
   - Elements beyond `dims[0]` are left uninitialized

2. **Hypothesis 2: Gather Operation Special Case**
   - KernelId(769) is a Gather operation
   - Gather might have special dims handling that doesn't initialize all 4 elements
   - The code path for Gather might differ from other operations

3. **Hypothesis 3: Activation Operations Default**
   - Many activation operations show `dims=[65536, 0, 0, 0]`
   - There might be a default value of `65536` used when shape inference fails
   - The remaining elements are set to `0` instead of `1`

## Suggested Investigation

### 1. Search for dims Initialization
```bash
# Find where dims arrays are created/initialized
grep -r "dims.*\[" /hologram/crates/compiler/src/pipeline/
grep -r "65536" /hologram/crates/compiler/
grep -r "\.dims = " /hologram/crates/compiler/
```

### 2. Add Logging to Track dims Values
In `/hologram/crates/compiler/src/pipeline/mod.rs`, add:
```rust
tracing::info!(
    "BUILD_OP[{}] kernel={:?} setting dims={:?}",
    op_idx, kernel_id, dims_array
);
```

### 3. Check Gather Operation Implementation
```bash
grep -A 20 "KernelId::GATHER" /hologram/crates/compiler/src/
```

### 4. Check Default Initialization
Look for where `BackendOp::params` is created:
```rust
pub struct BackendOpParams {
    pub dims: [usize; 4],  // How is this initialized?
    // ...
}
```

## Impact

**Current Status**:
- ✅ T5 encoder: Compiles and executes successfully
- ❌ T5 decoder: Compiles but fails at runtime (OP[97])
- ❌ T5 text generation: Blocked until this is fixed

**Affected Operations**:
- Gather operations (KernelId 769)
- Multiple activation operations throughout transformer models
- Likely affects all large transformer models (BERT, GPT, T5, etc.)

## Workaround Status

**Attempted**:
- ✅ Recompiled with latest hologram compiler - issue persists
- ✅ Removed dynamic workspace preallocation code - issue persists
- ❌ No working workaround found

**Conclusion**: This is a compiler bug that must be fixed in hologram before T5 can work.

## Related Documents

- Previous workspace allocation investigation: `/workspace/specs/plans/t5-workspace-allocation-findings.md`
- T5 workspace debug notes: `/workspace/specs/t5-workspace-debug.md`
- Workspace deep dive: `/workspace/specs/plans/t5-workspace-deep-dive-findings.md`

## Recommended Action

1. **Immediate**: Add comprehensive logging to track dims values during compilation
2. **Short-term**: Fix dims initialization in the compiler pipeline
3. **Long-term**: Add validation that all dims elements are valid (non-zero, < reasonable max)

## Contact

This bug report was generated during hologram-ai integration work for T5 model support.
