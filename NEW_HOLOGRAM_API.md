# New Hologram API Reference

This document shows how to use the rewritten hologram API (post-rewrite).

## Core Concepts

The new hologram API is much simpler:

1. **Build an `OperationGraph`** - just plain data structures
2. **Compile to `BackendPlan`** - single function call
3. **Serialize with `HolbWriter`** - standard format

### No More:
- ❌ `Dim`, `Shape` types (use `Vec<usize>` directly)
- ❌ `GraphBuilder`, `NodeIndex` (use plain vectors)
- ❌ Symbolic shapes (all shapes are concrete `usize`)
- ❌ `compile_ir_with_header` (use `compile()` + `HolbWriter`)

## Building a Graph

```rust
use hologram::compiler::{OperationGraph, OpNode, OpKind, DType};

let mut graph = OperationGraph::default();

// Add nodes
graph.nodes.push(OpNode::new(0, OpKind::Input, vec![2, 3], DType::F32));
graph.nodes.push(OpNode::new(1, OpKind::Relu, vec![2, 3], DType::F32));
graph.nodes.push(OpNode::new(2, OpKind::Output, vec![2, 3], DType::F32));

// Add edges (data flow)
graph.edges.push((0, 1));  // input → relu
graph.edges.push((1, 2));  // relu → output

// Register inputs/outputs
graph.inputs.push(("x".to_string(), 0));
graph.outputs.push(("y".to_string(), 2));

// Add constants (if needed)
graph.constants.push(ConstantData::F32(vec![1.0, 2.0, 3.0]));
```

## Compiling

```rust
use hologram::compiler::{compile, CompilerConfig};

let config = CompilerConfig::default();
let plan = compile(&graph, &config)?;
```

## Serialization

```rust
use hologram::holo::HolbWriter;

let mut writer = HolbWriter::new();
// Set the graph data (serialized plan)
writer.set_graph(&plan_bytes);
// Set weights (if any)
writer.set_weights(&weights_bytes);
// Build the .holb file
let holb_bytes = writer.build()?;
```

## Data Types

### OpKind Variants
- Arithmetic: `Add`, `Sub`, `Mul`, `Div`
- Activations: `Sigmoid`, `Tanh`, `Relu`, `Gelu`, `Silu`, `Softmax`
- Linear Algebra: `MatMul { m, k, n }`
- Reductions: `Sum`, `Max`, `Min`, `Mean`
- Special: `Input`, `Output`, `Constant`

### DType Variants
`F32`, `F64`, `I8`, `I16`, `I32`, `I64`, `U8`, `U16`, `U32`, `U64`, `Bool`

### ConstantData Variants
`F32(Vec<f32>)`, `F64(Vec<f64>)`, `I32(Vec<i32>)`, etc.

## For ONNX Compiler

### Strategy

1. **Parse ONNX protobuf** → extract ops, attributes, weights
2. **Build OperationGraph**:
   - Create `OpNode` for each ONNX op
   - Map ONNX tensor types → `DType`
   - Compute concrete output shapes (no symbolic dims)
   - Add edges based on ONNX value names
3. **Compile** → `BackendPlan`
4. **Serialize** → `.holb` file

### Handling Variable Shapes

Since the new API doesn't support symbolic shapes, we need to:
- Accept concrete batch size at compile time
- OR: Compile multiple versions for different batch sizes
- OR: Use dynamic dimensions = 1 as default

### Example ONNX → Hologram

```rust
// ONNX: Gemm(A, B, C) → Y
//   A: [M, K] f32
//   B: [K, N] f32
//   C: [N] f32
//   Y: [M, N] f32

// Hologram:
let gemm_node = OpNode::new(
    node_id,
    OpKind::MatMul { m: M, k: K, n: N },
    vec![M, N],  // output shape
    DType::F32,
);

// Add to graph
graph.nodes.push(gemm_node);
```

## Key Differences from Old API

| Old API | New API |
|---------|---------|
| `GraphBuilder::add_node()` | `graph.nodes.push(OpNode::new(...))` |
| `Dim::Static(n)`, `Dim::Symbolic("batch")` | Just `usize` |
| `Shape::new(vec![Dim::...])` | `Vec<usize>` |
| `NodeIndex(usize)` | `u32` (node ID) |
| `compile_ir_with_header()` | `compile()` + `HolbWriter` |
| `hologram::ir::GraphBuilder` | Plain `OperationGraph` struct |

## Migration Checklist

- [ ] Remove all `Dim`/`Shape` types → use `Vec<usize>`
- [ ] Remove `GraphBuilder` → build `OperationGraph` manually
- [ ] Remove `NodeIndex` → use `u32` node IDs
- [ ] Update `compile()` call signature
- [ ] Rewrite serialization using `HolbWriter`
- [ ] Handle shapes concretely (no symbolic dimensions)
