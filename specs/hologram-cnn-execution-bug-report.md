# Hologram Bug Report: Constant Folding Doesn't Compute Values

**Severity:** Critical
**Component:** `crates/compiler/src/fusion/constant.rs`
**Status:** Nodes folded to Constant have no data - buffer filled with zeros

---

## Summary

When the constant folding pass converts a node (like `Transpose`) to `OpKind::Constant`, **no actual data is computed**. The node is only *marked* as constant, but:

1. No constant value is calculated
2. No entry is added to `graph.constants`
3. The folded node is not in `constant_nodes_order`

Result: `serialize_constants()` can't find the data → fills buffer with zeros.

---

## Root Cause Analysis (Verified)

### The Constant Folding Pass

In `crates/compiler/src/fusion/constant.rs`:

```rust
// Lines 50-52: Only MARKS nodes as Constant, doesn't compute values
for id in to_fold {
    graph.replace_op(id, OpKind::Constant);  // No data!
}
```

The comment at lines 14-16 explicitly states:
> Note: Actual constant evaluation (computing values) is deferred to code generation.

**But code generation never does this either.**

### The Serialization Problem

In `crates/compiler/src/assemble.rs`:

```rust
// build_constant_index_map() uses constant_nodes_in_order()
fn build_constant_index_map(graph: &CompileGraph) -> Vec<(u32, usize)> {
    graph.constant_nodes_in_order()  // Only contains ORIGINAL constant nodes!
        .iter()
        .enumerate()
        ...
}
```

The `constant_nodes_order` is populated in `from_operation_graph()` **before** constant folding runs:

```rust
// traverse.rs:128-130 - Only captures nodes that START as Constant
if matches!(node.op, OpKind::Constant) {
    constant_nodes_order.push(node.id);
}
```

Nodes converted to Constant by folding are NOT added.

### The Data Flow

1. **Original constant** (e.g., FC weight): Node 50 → has data in `graph.constants[i]`
2. **Transpose(constant)**: Node 171 created dynamically
3. **Constant folding**: Node 171 marked as `OpKind::Constant`
4. **Buffer allocation**: Node 171 gets buffer 103 as `BufferType::Constant`
5. **Serialization**: `constant_index_map` has no entry for node 171 → zeros

---

## Evidence from Execution Trace

```
[INIT] Constant buffer analysis:
   Buffer 103: ALL ZEROS (2048000 bytes = 512x1000x4)  <-- FC weight buffer

[TRANSPOSE] Looking for Transpose instructions:
   (none!)  <-- Transpose was folded away, but data never computed

[MATMUL] MatMul a=171, b=103, c=172, m=1, k=512, n=1000
   a buf 171: Workspace, 2048 bytes     [OK - Flatten output]
   b buf 103: Constant, 2048000 bytes   [ALL ZEROS - should be transposed weights]

[NODE BUFFER MAP]:
   Node 171 -> Buffer 103  <-- Folded Transpose node, no data computed
```

---

## Reproduction Test Cases

### Test 1: Transpose of Constant (Minimal)

This is the **actual** bug trigger - Transpose(Constant) gets folded:

```rust
#[test]
fn test_transpose_constant_folding_produces_zeros() {
    use hologram_compiler::{compile, CompilerConfig, ConstantData, DType, OpKind, OpNode, OperationGraph};
    use hologram_backend::backends::cpu::CpuBackend;
    use hologram_backend::Backend;

    let mut graph = OperationGraph::new();

    // Input: [1, 4]
    graph.add_node(OpNode::new(0, OpKind::Input, vec![1, 4], DType::F32));
    graph.add_input("input", 0);

    // Constant weight: [4, 8] - will be transposed to [8, 4]
    graph.add_node(OpNode::new(1, OpKind::Constant, vec![4, 8], DType::F32));
    graph.add_constant(ConstantData::F32(vec![0.1; 32]));  // 4 * 8 = 32 values

    // Transpose the constant: [4, 8] -> [8, 4]
    // This gets folded to OpKind::Constant but DATA IS NOT COMPUTED
    graph.add_node(OpNode::new(
        2,
        OpKind::Transpose { perm: vec![1, 0] },
        vec![8, 4],
        DType::F32,
    ));
    graph.add_edge(1, 2);  // constant -> transpose

    // MatMul: [1, 4] @ [4, 8] -> [1, 8]
    // Uses transposed constant as weight
    graph.add_node(OpNode::new(
        3,
        OpKind::MatMul { m: 1, k: 4, n: 8 },
        vec![1, 8],
        DType::F32,
    ));
    graph.add_edge(0, 3);  // input -> matmul
    graph.add_edge(2, 3);  // transposed constant -> matmul

    // Output
    graph.add_node(OpNode::new(4, OpKind::Output, vec![1, 8], DType::F32));
    graph.add_edge(3, 4);
    graph.add_output("output", 4);

    // Compile
    let plan = compile(&graph, &CompilerConfig::default()).expect("compilation failed");

    // Execute
    let input_data: Vec<f32> = vec![1.0; 4];
    let input_bytes: Vec<u8> = bytemuck::cast_slice(&input_data).to_vec();

    let mut output_data: Vec<f32> = vec![0.0; 8];
    let output_bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut output_data);

    let backend = CpuBackend::new();
    backend.execute_plan(&plan, &[&input_bytes], &mut [output_bytes]).expect("execution failed");

    let non_zero = output_data.iter().filter(|&&x| x.abs() > 1e-10).count();
    eprintln!("Output: {}/8 non-zero: {:?}", non_zero, output_data);

    // BUG: This fails because transposed constant buffer contains zeros
    assert!(non_zero > 0, "Transpose(Constant) should produce non-zero output!");
}
```

### Test 2: Direct MatMul (Works - No Folding)

This test **passes** because there's no Transpose to fold:

```rust
#[test]
fn test_direct_matmul_constant_works() {
    // Same as above but WITHOUT Transpose - constant is used directly
    // MatMul input B is original constant, not folded Transpose
    // This WORKS because constant is in constant_nodes_order
}
```

---

## Suggested Fixes

### Option A: Compute Values During Constant Folding (Recommended)

Modify `fusion/constant.rs` to actually compute constant values:

```rust
fn fold_constants_once(graph: &mut CompileGraph, constants: &mut Vec<ConstantData>) -> usize {
    for id in to_fold {
        // 1. Get input constant data
        let input_data = get_predecessor_constant_data(graph, id, constants);

        // 2. Compute the operation result
        let result_data = match graph.get_node(id).op {
            OpKind::Transpose { perm } => compute_transpose(&input_data, &perm),
            OpKind::Reshape { shape } => input_data, // Just reinterpret
            // ... other foldable ops
        };

        // 3. Add to constants and update tracking
        let const_idx = constants.len();
        constants.push(result_data);
        graph.add_to_constant_nodes_order(id, const_idx);

        // 4. Mark as constant
        graph.replace_op(id, OpKind::Constant);
    }
}
```

### Option B: Skip Folding for Operations Without Evaluation

Don't fold operations that can't be evaluated:

```rust
fn is_foldable(op: &OpKind) -> bool {
    match op {
        // Only fold ops where we can compute the result
        OpKind::Reshape { .. } => true,  // No-op, just metadata
        OpKind::Transpose { .. } => false,  // Can't evaluate without runtime
        // ...
    }
}
```

### Option C: Track Folded Constants Separately

Add folded constants to the tracking list:

```rust
// In constant_folding()
for id in to_fold {
    graph.replace_op(id, OpKind::Constant);
    graph.add_folded_constant(id);  // NEW: track in separate list
}

// In serialize_constants()
// Include both original and folded constants
```

---

## Files to Modify

1. **`crates/compiler/src/fusion/constant.rs`**
   - `fold_constants_once()` - compute actual values

2. **`crates/compiler/src/graph/traverse.rs`**
   - Add method to track folded constants
   - Update `constant_nodes_order` after folding

3. **`crates/compiler/src/assemble.rs`**
   - `build_constant_index_map()` - include folded constants
   - `serialize_constants()` - serialize folded constant data

---

## Workaround for ONNX Frontend

Until the compiler is fixed, the ONNX frontend can avoid the bug by:

1. **Pre-transpose weights at load time** - Store weights in transposed form
2. **Don't use transB=1** - Transpose during model loading, not at runtime

```rust
// In ONNX builder, instead of creating Transpose node:
if trans_b {
    // Transpose the constant data directly
    let transposed_data = transpose_constant(&weight_data, &[1, 0]);
    // Use transposed data as the constant, no Transpose node needed
}
```

---

**Report from:** hologram-ai-onnx integration
**Date:** 2026-02-02
**Bug identified by:** Execution tracing showing no Transpose instruction + buffer 103 all zeros
