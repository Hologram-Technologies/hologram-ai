# Hologram Team: ResNet18 Shape Mismatch Bug Report

## Summary
ResNet18 ONNX model compilation fails with `ShapeMismatch { node_id: 273, expected: [1000], actual: [1, 512] }` during hologram compilation phase. The ONNX translation is correct, but the hologram compiler rejects the graph.

## Environment
- hologram-ai-onnx: Successfully translates all ONNX operations
- Model: ResNet18-v1-7 from ONNX model zoo
- All 35 unit tests pass for operation translation

## Reproduction

### Step 1: Download ResNet18
```bash
curl -L "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet18-v1-7.onnx" -o resnet18.onnx
```

### Step 2: Compile with hologram-ai-onnx
```rust
let onnx_bytes = std::fs::read("resnet18.onnx")?;
let result = hologram_ai_onnx::compile_onnx(&onnx_bytes);
// Error: Compilation failed: ShapeMismatch { node_id: 273, expected: [1000], actual: [1, 512] }
```

## Root Cause Analysis

### Expected Behavior

ResNet18's final classification layer should work as follows:

1. **GlobalAveragePool**: `[1, 512, 7, 7]` → `[1, 512, 1, 1]`
   - Translates to: `OpKind::GlobalAveragePool`
   - Output shape: `vec![1, 512, 1, 1]` ✓

2. **Flatten** (axis=1): `[1, 512, 1, 1]` → `[1, 512]`
   - Translates to: `OpKind::Flatten { start_dim: 1 }`
   - Output shape calculation:
     ```rust
     let dim0 = shape[..1].iter().product(); // 1
     let dim1 = shape[1..].iter().product(); // 512 * 1 * 1 = 512
     vec![1, 512] ✓
     ```

3. **Gemm/MatMul**: `[1, 512]` × `[512, 1000]` → `[1, 1000]`
   - Translates to: `OpKind::MatMul { m: 1, k: 512, n: 1000 }`
   - Output shape: `vec![1, 1000]` ✓

### Observed Behavior

The error `ShapeMismatch { node_id: 273, expected: [1000], actual: [1, 512] }` suggests:
- Some node (273) expects input shape `[1000]` (1D)
- But receives shape `[1, 512]` (2D with batch dimension)

## Hypotheses

### Hypothesis 1: Flatten Produces Wrong Shape
**Possibility**: `OpKind::Flatten { start_dim: 1 }` might be flattening to `[512]` instead of `[1, 512]`

**Test Case**:
```rust
// Input: [1, 512, 1, 1]
let flatten_op = OpKind::Flatten { start_dim: 1 };
// Expected output: [1, 512]
// Actual output: [512] ?
```

**Fix**: Ensure Flatten preserves dimensions before `start_dim`

---

### Hypothesis 2: MatMul Produces Wrong Output Shape
**Possibility**: `OpKind::MatMul { m: 1, k: 512, n: 1000 }` might produce `[1000]` instead of `[1, 1000]`

**Test Case**:
```rust
// A: [1, 512], B: [512, 1000]
let matmul_op = OpKind::MatMul { m: 1, k: 512, n: 1000 };
// Expected output: [1, 1000]
// Actual output: [1000] ?
```

**Fix**: When m=1, MatMul should still produce `[1, n]` not `[n]`

---

### Hypothesis 3: Gemm Bias Addition Issue
**Possibility**: ONNX Gemm has an optional bias term of shape `[1000]`. The Add operation might expect matching shapes.

**Details**:
- ONNX Gemm: `Y = alpha * A * B + beta * C` where C is bias
- If MatMul produces `[1, 1000]` and bias is `[1000]`, broadcasting should work
- But if the compiler expects exact shape match, this could fail

**Test Case**:
```rust
// MatMul output: [1, 1000]
// Bias: [1000]
// Add operation with broadcasting
// Does hologram support shape broadcasting for Add?
```

**Fix**: Ensure Add supports broadcasting `[1, n] + [n] → [1, n]`

---

### Hypothesis 4: Node 273 is Not MatMul/Gemm
**Possibility**: Node 273 might be a different operation (Add, Squeeze, etc.) that's receiving unexpected input

**Investigation Needed**:
- What operation is node 273?
- What are its inputs?
- What shape does it expect vs receive?

## Minimal Test Case

```rust
use hologram::compiler::{OperationGraph, OpKind, OpNode, DType, compile, CompilerConfig};

fn test_resnet_final_layer() -> Result<(), Box<dyn std::error::Error>> {
    let mut graph = OperationGraph::default();

    // Input after GlobalAveragePool: [1, 512, 1, 1]
    let input = OpNode::new(0, OpKind::Input, vec![1, 512, 1, 1], DType::F32);
    graph.nodes.push(input);
    graph.inputs.push(0);

    // Flatten (axis=1): [1, 512, 1, 1] → [1, 512]
    let flatten = OpNode::new(1, OpKind::Flatten { start_dim: 1 }, vec![1, 512], DType::F32);
    flatten.inputs = vec![0];
    graph.nodes.push(flatten);

    // Weight (constant): [512, 1000]
    let weight = OpNode::new(2, OpKind::Constant, vec![512, 1000], DType::F32);
    graph.nodes.push(weight);

    // MatMul: [1, 512] × [512, 1000] → [1, 1000]
    let matmul = OpNode::new(
        3,
        OpKind::MatMul { m: 1, k: 512, n: 1000 },
        vec![1, 1000],
        DType::F32
    );
    matmul.inputs = vec![1, 2];
    graph.nodes.push(matmul);

    // Bias (constant): [1000]
    let bias = OpNode::new(4, OpKind::Constant, vec![1000], DType::F32);
    graph.nodes.push(bias);

    // Add: [1, 1000] + [1000] → [1, 1000] (with broadcasting)
    let add = OpNode::new(5, OpKind::Add, vec![1, 1000], DType::F32);
    add.inputs = vec![3, 4];
    graph.nodes.push(add);

    graph.outputs.push(5);

    // Compile
    let config = CompilerConfig::default();
    let plan = compile(&graph, &config)?;

    println!("✓ Compilation successful");
    Ok(())
}
```

## Questions for Hologram Team

1. **Flatten behavior**: Does `OpKind::Flatten { start_dim: 1 }` on input `[1, 512, 1, 1]` produce:
   - a) `[1, 512]` (preserves dims before start_dim) ← Expected
   - b) `[512]` (removes all dims before start_dim) ← Bug?

2. **MatMul with m=1**: Does `OpKind::MatMul { m: 1, k: 512, n: 1000 }` produce:
   - a) `[1, 1000]` (keeps batch dimension) ← Expected
   - b) `[1000]` (removes batch dimension) ← Bug?

3. **Add broadcasting**: Does `OpKind::Add` support broadcasting `[1, n] + [n]`?
   - If yes: Should automatically broadcast [1000] to [1, 1000]
   - If no: Need to insert explicit Unsqueeze before Add

4. **Node 273**: Can you provide debug info about:
   - What operation is node 273?
   - What are its input shapes?
   - What shape does it expect?

## Suggested Fix

**If Flatten is the issue**:
```rust
// In hologram compiler, ensure Flatten preserves prefix dimensions
impl Flatten {
    fn infer_output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        let prefix: usize = input_shape[..self.start_dim].iter().product();
        let suffix: usize = input_shape[self.start_dim..].iter().product();
        vec![prefix, suffix]  // Not just vec![suffix]
    }
}
```

**If MatMul is the issue**:
```rust
// Ensure MatMul output keeps batch dimension even when m=1
impl MatMul {
    fn output_shape(&self) -> Vec<usize> {
        vec![self.m, self.n]  // Not vec![self.n] when m==1
    }
}
```

**If Add broadcasting is the issue**:
```rust
// Support shape broadcasting in Add operation
// [1, n] + [n] should broadcast to [1, n]
```

## Impact

This bug blocks compilation of all ResNet models and likely affects other CNN architectures (VGG, EfficientNet, etc.) that use:
- GlobalAveragePool → Flatten → MatMul/Gemm pattern
- Bias addition with broadcasting

## Priority

**High** - Affects common CNN architectures used in production

## Contact

hologram-ai-onnx team via GitHub issues
