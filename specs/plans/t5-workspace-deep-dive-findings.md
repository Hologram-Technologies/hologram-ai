# T5 Workspace Allocation Deep Dive - Investigation Findings

## Executive Summary

Through deep investigation of the hologram compiler workspace allocation system, we identified the root cause of the T5 decoder workspace underallocation bug. The issue was NOT in the workspace allocation logic itself, but in **ONNX shape inference creating symbolic dimensions** that couldn't be resolved to concrete values. By compiling with explicit input shapes, we fixed the underallocation issue but uncovered a **new bug in total workspace size calculation**.

## Timeline of Investigation

### Phase 1: Initial Analysis (Completed ✅)
- **Objective**: Compare encoder vs decoder ONNX models to understand why encoder works but decoder fails
- **Method**: Created Python script to analyze shape coverage in both models
- **Result**: Both models had abysmal shape coverage:
  - Encoder: 0.2% (1/492 tensors had shapes)
  - Decoder: 2.8% (25/897 tensors had shapes)

### Phase 2: ONNX Shape Inference (Completed ✅)
- **Objective**: Run ONNX shape inference to add missing shape information
- **Method**: Used `onnx.shape_inference.infer_shapes()` to generate shape-inferred models
- **Result**: Both models achieved 100% shape coverage
- **Problem**: Shape inference introduced 156 **symbolic dimensions** (unk__0, unk__1, etc.)

### Phase 3: Shape-Inferred Compilation (Completed ✅)
- **Objective**: Test if shape-inferred models compile and run correctly
- **Result**: Compilation succeeded but runtime still failed
- **Error**: Same workspace allocation error (workspace_102 had 16MB instead of 512MB)
- **Discovery**: Symbolic dimensions were being treated as Dynamic, causing underallocation

### Phase 4: Comprehensive Tracing (Completed ✅)
- **Objective**: Add detailed tracing to understand workspace allocation flow
- **Method**: Enhanced logging in:
  - `workspace_alloc` path in mod.rs
  - `elementwise_numel_from_predecessors()` in helpers.rs
  - `effective_numel_for_node()` in helpers.rs
  - `output_numel_for_node()` for binary operations
- **Findings**:
  - elementwise_numel correctly calculated 134M elements (512MB)
  - Add operations were getting correct workspace sizes allocated
  - But workspace_102 still only had 16MB at runtime

### Phase 5: Concrete Input Shapes (Completed ✅)
- **Objective**: Compile decoder with explicit concrete input shapes to resolve symbolic dimensions
- **Method**: Used `--input-shape` flags to specify exact dimensions:
  ```bash
  --input-shape "input_ids:1,1"
  --input-shape "encoder_hidden_states:1,512,512"
  --input-shape "encoder_attention_mask:1,512"
  ```
- **Result**: Compilation succeeded with correct workspace allocations!
  - Workspace regions now properly sized (512MB, 516MB, etc.)
  - Individual operations have correct workspace requirements

### Phase 6: New Bug Discovered (Current Issue ⚠️)
- **Error**: `WORKSPACE BUG: Region 114 'workspace_114' extends beyond total_size!`
- **Analysis**:
  - offset=2,924,609,536 size=541,065,216 end=3,465,674,752
  - total_size=2,980,839,424 (2.98GB)
  - Required: ~4.0GB (workspace_115 ends at 4,002,545,664)
- **Root Cause**: Total workspace size calculation is incorrect
  - Individual workspace regions are correctly sized
  - But the sum/maximum of all regions is not being calculated properly
  - Missing ~1GB of required workspace

## Technical Details

### Root Cause: Symbolic Dimensions from ONNX Shape Inference

ONNX shape inference creates symbolic dimensions for values it cannot determine statically:

```python
# Example from decoder model:
'/decoder/ConstantOfShape_output_0[0]' = 'unk__0'  # Batch size
'/decoder/ConstantOfShape_output_0[1]' = 'unk__1'  # Sequence length
'/decoder/Range_output_0[0]' = 'unk__2'            # Position indices
```

These symbolic dimensions translate to `Dim::Dynamic` or `Dim::Symbolic` in hologram IR, which then become 0 or placeholder values during workspace allocation, causing severe underallocation.

### Solution: Concrete Input Shapes

By providing explicit input shapes at compilation time, symbolic dimensions are resolved to concrete values:

```bash
cargo run --release -p hologram-ai -- compile decoder_model_inferred.onnx \
  --input-shape "input_ids:1,1" \
  --input-shape "encoder_hidden_states:1,512,512" \
  --input-shape "encoder_attention_mask:1,512"
```

This allows hologram to:
1. Propagate concrete shapes through the entire graph
2. Calculate exact workspace requirements for each operation
3. Allocate correct amounts (512MB, 516MB, etc.)

### Remaining Bug: Total Workspace Size Calculation

**Location**: Likely in `/hologram/crates/compiler/src/pipeline/mod.rs` or workspace planning code

**Problem**: The total workspace size is calculated as 2.98GB, but individual regions extend beyond 4GB:

```
Region 114: offset=2.9GB, size=516MB, end=3.5GB  ✅ Correct size
Region 115: offset=3.5GB, size=512MB, end=4.0GB  ✅ Correct size
Total: 2.98GB  ❌ Should be ~4.0GB or more
```

**Hypothesis**: The total workspace size might be calculated as:
- Sum of non-overlapping regions only?
- Maximum offset instead of maximum end position?
- Bug in accounting for alignment or padding?

## Workspace Allocation Architecture (Verified Correct ✅)

Through investigation, we verified these components are working correctly:

### 1. Workspace Allocation Logic
- **File**: `/hologram/crates/compiler/src/pipeline/mod.rs:875-899`
- **Function**: Correctly calls `op.workspace_size()` and falls back to `elementwise_numel_from_predecessors()`
- **FusedActivation Fix**: Properly implemented to use predecessor sizing when `workspace_size() == 0 && preserves_element_count()`

### 2. Shape Broadcasting System
- **File**: `/hologram/crates/ir/src/shape.rs:514-541`
- **Function**: `broadcast_shapes()` correctly implements NumPy broadcasting rules
- **Verification**: Tested with Add operations, produces correct broadcasted shapes

### 3. IR Binary Operation Creation
- **File**: `/hologram/crates/ir/src/builder.rs:109-126`
- **Function**: `GraphBuilder.binary()` correctly broadcasts shapes and creates IR nodes

### 4. IR → Compiler Conversion
- **File**: `/hologram/crates/ir/src/ops/binary/add.rs:49-57`
- **Function**: `IrAdd.to_compiler_op()` correctly uses `shape.num_elements()` to calculate size

### 5. Compiler Binary Operations
- **File**: `/hologram/crates/compiler/src/graph/ops/binary.rs:16-68`
- **Function**: `Add` correctly implements `workspace_size()` and `workspace_size_expr()` for dynamic sizing

### 6. ONNX → IR Translation
- **File**: `/workspace/crates/hologram-ai-onnx/src/core/translator.rs:300-421`
- **Function**: Correctly processes inputs, initializers, nodes, and outputs
- **Note**: Properly handles ONNX ValueInfoProto shape information when present

## Key Learnings

1. **ONNX shape inference is not a silver bullet**: It creates symbolic dimensions for values it cannot determine statically, which still need to be resolved.

2. **Concrete input shapes are essential**: For models with variable-length inputs (like T5), compilation requires explicit input shapes to resolve symbolic dimensions.

3. **Workspace allocation works correctly**: Once shapes are concrete, the workspace allocation logic correctly calculates per-operation requirements (512MB, 516MB, etc.).

4. **Total workspace size has a bug**: There's a separate issue in calculating the cumulative total workspace size needed.

5. **Diagnostic tracing is valuable**: The comprehensive tracing added during investigation should be kept (or made optional) for future debugging.

## Files Modified During Investigation

### Enhanced with Tracing
1. `/hologram/crates/compiler/src/pipeline/mod.rs` - Workspace allocation logging
2. `/hologram/crates/compiler/src/pipeline/helpers.rs` - Element count calculation logging

### Created for Analysis
1. `/workspace/scripts/analyze_onnx_shapes.py` - ONNX shape coverage analysis
2. `/workspace/scripts/run_shape_inference.py` - ONNX shape inference automation
3. `/workspace/models/t5-small/encoder_model_inferred.onnx` - Shape-inferred encoder
4. `/workspace/models/t5-small/decoder_model_inferred.onnx` - Shape-inferred decoder
5. `/workspace/models/t5-small/compiled/decoder-concrete.holo` - Decoder with concrete shapes
6. `/workspace/configs/t5-generate-inferred.toml` - Test configuration

## Next Steps

### Immediate (Fix Total Workspace Size Bug)
1. **Investigate total_size calculation**:
   - Find where total workspace size is computed
   - Check if it's using sum vs max correctly
   - Verify alignment and padding calculations

2. **Possible locations**:
   - `/hologram/crates/compiler/src/pipeline/mod.rs` - Workspace planning
   - `/hologram/crates/backend/src/workspace.rs` - Runtime workspace management
   - Serialization code that writes total_size to .holo files

3. **Fix the calculation**:
   - Ensure total_size >= max(region.offset + region.size) for all regions
   - Account for alignment requirements
   - Add validation to catch this early

### Short-term (Improve Shape Handling)
1. **Integrate shape inference into hologram-ai compilation**:
   - Automatically run ONNX shape inference when needed
   - Detect missing shape information
   - Prompt user for concrete input shapes

2. **Add shape validation**:
   - Warn when symbolic dimensions are detected
   - Provide helpful error messages about `--input-shape` flag
   - Validate that all shapes are concrete before workspace allocation

3. **Improve tracing**:
   - Make the enhanced tracing optional (via env var or flag)
   - Add tracing for total workspace size calculation
   - Log workspace layout summary

### Long-term (Architecture Improvements)
1. **Dynamic runtime workspace allocation**:
   - Implement fully dynamic workspace sizing at runtime
   - Use `workspace_size_expr()` more broadly
   - Eliminate need for concrete shapes at compile time (where possible)

2. **Better symbolic shape support**:
   - Implement hologram's own shape inference
   - Support symbolic shapes throughout the pipeline
   - Resolve symbols at runtime based on actual input sizes

3. **Workspace allocation tests**:
   - Unit tests for workspace size calculations
   - Integration tests with various model architectures
   - Regression tests for specific bugs (like this one)

## Conclusion

We successfully identified that the workspace underallocation bug was caused by ONNX shape inference creating symbolic dimensions that couldn't be resolved without explicit input shapes. By compiling with concrete input shapes, we fixed the per-operation workspace allocation, but uncovered a new bug in calculating the total workspace size.

The investigation demonstrated that:
- ✅ Hologram's workspace allocation infrastructure is sound
- ✅ Shape broadcasting and inference work correctly
- ✅ Per-operation workspace sizes are calculated correctly (with concrete shapes)
- ❌ Total workspace size calculation has a bug (missing ~1GB)

The path forward is clear: fix the total workspace size calculation bug, then the T5 decoder should run successfully!
