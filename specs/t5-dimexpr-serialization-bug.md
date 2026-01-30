# T5 DimExpr Serialization Bug

## Summary

The Hologram compiler correctly generates dynamic dimension expressions for Reduce operations with `divisor: 512`, but the runtime loads these operations with `divisor: 1`. This is a serialization/deserialization bug in the BackendPlan .holo file format.

## Symptoms

```
Compiler (correct):
dim_exprs=[Some(PredecessorElementsDiv { predecessor_slot: 27, divisor: 512 }), ...]

Runtime (incorrect):
dim_exprs=[Some(PredecessorElementsDiv { predecessor_slot: 7, divisor: 1 }), ...]
```

This causes buffer overflow errors at runtime:
```
ERROR: OP[9] KernelId(1540): input size 536870912 bytes exceeds workspace region
'workspace_7' allocation of 16777216 bytes. dims=[262144, 512, 1, 0]
```

## Root Cause Analysis

### Compiler Side (WORKING)

**File:** `/hologram/crates/compiler/src/pipeline/mod.rs`
**Lines:** 2146-2247

The compiler correctly:
1. Detects dynamic outer_size in Reduce operations
2. Sets `dim_exprs[0] = Some(DimExpr::PredecessorElementsDiv { predecessor_slot, divisor: reduce_size })`
3. Logs confirm: "FIX APPLIED: Set dim_exprs[0] divisor=512 (was 1), predecessor_slot=27"
4. Final values show: `dim_exprs=[Some(PredecessorElementsDiv { predecessor_slot: 27, divisor: 512 }), ...]`

**Evidence:** 19 Reduce operations compiled with divisor=512 in compilation logs.

### Serialization Side (SUSPECTED BUG)

**Structures have proper rkyv derives:**
- `DimExpr` enum in `/hologram/crates/ir/src/shape.rs:26-80`
  - Has `#[derive(Archive, Serialize, Deserialize, ...)]`
  - `PredecessorElementsDiv` variant at lines 74-79 with `divisor: usize` field

- `KernelParams` struct in `/hologram/crates/backend/src/core/plan.rs:467`
  - Has `#[derive(Debug, Clone, Default, Archive, RkyvSerialize, RkyvDeserialize)]`
  - Contains `dim_exprs: [Option<DimExpr>; 4]` field at line 484

**But runtime loads incorrect values:**
- Compiled .holo file: `/workspace/models/t5-small/compiled/decoder.holo` (444MB)
- Runtime shows: `dim_exprs=[Some(PredecessorElementsDiv { predecessor_slot: 7, divisor: 1 }), ...]`

### Hypothesis

Possible causes:
1. **rkyv serialization bug**: The `divisor` field in `PredecessorElementsDiv` might not be serializing correctly
2. **Default value initialization**: Deserialization might be using Default::default() which sets divisor to 1
3. **Version mismatch**: The BackendPlan format version might not match between compiler and runtime
4. **Buffer corruption**: The .holo file might be corrupted during write/read
5. **Workspace slot mismatch**: predecessor_slot changes from 27 (compile) to 7 (runtime), suggesting a different operation

## Test Case

### Input

T5-small decoder ONNX model with:
- 19 ReduceMean operations in layer norms
- Input shape: `[batch*seq, 512]` (dynamic first dimension)
- Reduce on axis=-1 (hidden dimension)

### Expected Behavior

For ReduceMean with input `[batch*seq, 512]`:
- `outer_size = batch*seq` (dynamic)
- `reduce_size = 512` (static, hidden dimension)
- `inner_size = 1` (keepdims)

Compiler should generate:
```rust
dim_exprs[0] = Some(PredecessorElementsDiv {
    predecessor_slot: X,
    divisor: 512  // outer = total / reduce
})
dim_exprs[1] = Some(Static(512))
dim_exprs[2] = Some(Static(1))
```

Runtime should resolve:
- `predecessor_elements[X] = 262144` (total elements)
- `resolved_dims[0] = 262144 / 512 = 512` (batch*seq at runtime)
- `resolved_dims[1] = 512` (hidden dim)
- `resolved_dims[2] = 1` (keepdims)

### Actual Behavior

Runtime loads:
```rust
dim_exprs[0] = Some(PredecessorElementsDiv {
    predecessor_slot: 7,
    divisor: 1  // WRONG!
})
```

Resolves to:
- `resolved_dims[0] = 262144 / 1 = 262144` (WRONG - should be 512)
- Causes buffer overflow: 262144 * 512 * 4 bytes = 536MB > 16MB workspace

## Reproduction Steps

1. Apply compiler fix in `/hologram/crates/compiler/src/pipeline/mod.rs` lines 2146-2247
2. Rebuild hologram compiler: `cargo build --release -p hologram`
3. Compile T5 decoder: `cargo run --release -p hologram-ai -- compile /workspace/models/t5-small/decoder_model.onnx --output decoder.holo`
4. Verify compiler logs show "FIX APPLIED: Set dim_exprs[0] divisor=512"
5. Run T5 pipeline: `cargo run --release -p hologram-ai -- run --config examples/T5/t5.toml`
6. Observe runtime error: "OP[9] ... dims=[262144, 512, 1, 0]" with divisor=1

## Workarounds

None identified. The bug is in the serialization layer, not the compiler logic.

## Next Steps

1. **Inspect .holo file binary**: Use hexdump to verify if divisor field contains 512 or 1
2. **Add serialization test**: Write unit test that serializes/deserializes KernelParams with PredecessorElementsDiv
3. **Check rkyv version**: Ensure compatible rkyv versions across all crates
4. **Add validation**: Check dim_exprs after deserialization in runtime
5. **Alternative fix**: Use different DimExpr variant that doesn't have this bug

## Files Modified

- `/hologram/crates/compiler/src/pipeline/mod.rs` (lines 2146-2247) - Compiler fix (WORKING)
- `/hologram/crates/backend/src/core/executor.rs` (lines 1945-1963) - Debug logging

## Related Issues

- ExecutionContext refactor (completed, working correctly)
- T5 pipeline testing (blocked by this bug)
- Dynamic shape inference (working in compiler, broken in runtime)

## Status

**BLOCKED**: Compiler generates correct code, but serialization loses the divisor value.
**Priority**: HIGH - Blocks all T5 and transformer model execution with dynamic shapes.
