# Hologram Team Handoff: ResNet18 Shape Mismatch Bug

## Summary

ResNet18 ONNX model compilation fails with a shape mismatch error in the hologram compiler. The ONNX translation layer (hologram-ai-onnx) is working correctly - all 35 unit tests pass and the OperationGraph is constructed properly. However, when hologram's `compile()` function processes the graph, it produces:

```
ShapeMismatch { node_id: 273, expected: [1000], actual: [1, 512] }
```

## Test Case (No ONNX Required)

We've created a standalone test file that reproduces the bug using only hologram's native APIs:

**Location**: [/workspace/specs/hologram-team-resnet-test.rs](hologram-team-resnet-test.rs)

This test can be added directly to the hologram repository without any ONNX dependencies.

### Usage

```bash
# In hologram repository
cp /path/to/hologram-team-resnet-test.rs tests/resnet_shape_bug.rs

# Run the tests
cargo test --test resnet_shape_bug
```

### Test Functions

1. **`test_resnet18_final_layers()`** - Full pipeline that triggers the bug
   - Creates: Input → GlobalAveragePool → Flatten → MatMul → Add
   - Expected: Compilation success
   - Actual: ShapeMismatch error

2. **`test_flatten_shape_preservation()`** - Tests Hypothesis 1
   - Verifies: `Flatten { start_dim: 1 }` on `[1, 512, 1, 1]` produces `[1, 512]`
   - If this fails: Flatten is producing `[512]` instead of `[1, 512]`

3. **`test_matmul_with_batch_size_one()`** - Tests Hypothesis 2
   - Verifies: `MatMul { m: 1, k: 512, n: 1000 }` produces `[1, 1000]`
   - If this fails: MatMul is producing `[1000]` instead of `[1, 1000]`

4. **`test_add_broadcasting()`** - Tests Hypothesis 3
   - Verifies: `Add` supports broadcasting `[1, 1000] + [1000]`
   - If this fails: Add doesn't support broadcasting

## Detailed Bug Report

See: [/workspace/specs/plans/resnet18-shape-mismatch-bug-report.md](plans/resnet18-shape-mismatch-bug-report.md)

The bug report includes:
- Complete root cause analysis
- 3 hypotheses with test cases
- Suggested fixes for each scenario
- Impact analysis (blocks all CNN architectures)
- Questions for hologram team

## Three Hypotheses

### Hypothesis 1: Flatten Bug

**Issue**: `OpKind::Flatten { start_dim: 1 }` might be producing `[512]` instead of `[1, 512]`

**Expected behavior**:
```rust
// Input: [1, 512, 1, 1]
// Flatten with start_dim=1
let prefix: usize = input_shape[..1].iter().product();  // 1
let suffix: usize = input_shape[1..].iter().product();  // 512
// Output: [1, 512]  ← Correct
```

**Possible bug**:
```rust
// Output: [512]  ← Wrong (loses batch dimension)
```

### Hypothesis 2: MatMul Bug

**Issue**: `OpKind::MatMul { m: 1, k: 512, n: 1000 }` might be producing `[1000]` instead of `[1, 1000]`

**Expected behavior**:
```rust
// A: [1, 512], B: [512, 1000]
// Output: [1, 1000]  ← Correct (keeps batch dimension even when m=1)
```

**Possible bug**:
```rust
// Output: [1000]  ← Wrong (removes batch dimension when m=1)
```

### Hypothesis 3: Add Broadcasting Bug

**Issue**: `OpKind::Add` might not support broadcasting `[1, n] + [n]`

**Expected behavior**:
```rust
// A: [1, 1000], B: [1000]
// Output: [1, 1000]  ← Correct (broadcasts [1000] to [1, 1000])
```

**Possible bug**:
```rust
// Requires exact shape match, doesn't support broadcasting
```

## Impact

This bug blocks compilation of:
- ResNet (all variants: 18, 34, 50, 101, 152)
- VGG models
- EfficientNet
- MobileNet
- Any CNN using GlobalAveragePool → Flatten → MatMul/Gemm pattern

## Priority

**High** - Affects common production CNN architectures

## Contact

hologram-ai-onnx team via GitHub issues

## Verification

Once fixed, verify with:

```bash
# In hologram-ai-onnx repository
cargo test --test test_real_models -- --nocapture

# Should successfully compile ResNet18
```

## Additional Debug Tools (Optional)

If needed, we have additional debug utilities in the hologram-ai-onnx repository:

1. **Python ONNX analyzer**: `/workspace/scripts/analyze_resnet.py`
   - Inspects ONNX model structure
   - Lists all operations and their attributes
   - Shows initializer shapes

2. **Rust debug example**: `/workspace/crates/hologram-ai-onnx/examples/debug_resnet.rs`
   - Parses ONNX model using hologram-ai-onnx
   - Shows operation translation details

But these require ONNX files, so the standalone test case above is the recommended approach.
