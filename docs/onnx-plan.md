# TODO: ONNX Integration Plan

This document describes how to implement ONNX support in a separate `hologram-onnx` repository.

## Overview

ONNX integration is implemented in a separate repository that uses hologram-compiler's `OperationGraph` types. This repo provides the infrastructure; the ONNX repo handles parsing and conversion.

## External ONNX Integration

When the separate ONNX repository is implemented, it can integrate in two ways:

### Option A: Direct OperationGraph (Recommended for ONNX)

```
model.onnx → [hologram-onnx repo] → OperationGraph + WeightData → .holo + .weights
                                                                        ↓
                                                                hologram run
```

- ONNX repo adds `hologram-compiler` as dependency (for `OperationGraph`, `WeightRef` types)
- ONNX repo directly builds `OperationGraph`:
  - `graph.input` → `NodeOp::Input`
  - `graph.initializer` → `NodeOp::WeightRef` (large) or `NodeOp::ConstantTensor` (small)
  - `graph.node` → `NodeOp::*` operations
- ONNX repo writes `.holo` (rkyv serialized OperationGraph) + `.weights` (raw bytes)
- `hologram run model.holo` executes it (already works today)

**Why recommended:** ONNX already has concrete shapes and weights - no need for inference.

### Option B: Via Common IR

```
model.onnx → [hologram-onnx repo] → Common IR → [hologram compile] → .holo + .weights
```

- ONNX repo produces Common IR with `WeightRef` nodes
- hologram compiler lowers to OperationGraph
- Useful if ONNX needs shape inference or transformations

### What This Repo Provides for ONNX

1. **`OperationGraph` types** - `NodeOp::WeightRef`, `NodeOp::ConstantTensor`, etc.
2. **Serialization** - `graph.to_bytes()` for .holo, `WeightData::write_to_file()` for .weights
3. **Execution** - `hologram run` loads and executes weighted graphs
4. **Weight loading** - `Model::load()` resolves WeightRef → actual tensors

### Integration Test

When ONNX repo is ready:

```bash
# In ONNX repo: compile model
hologram-onnx compile resnet.onnx -o resnet.holo
# Produces: resnet.holo + resnet.weights

# In this repo: run model
hologram run resnet.holo --input image.npy
# Loads weights, executes graph
```

## ONNX Implementation Guide (For Future hologram-onnx Repo)

This section documents how to implement ONNX support in a separate repository.

### Repository Structure

```
hologram-onnx/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── parser.rs       # ONNX protobuf parsing
│   ├── ops.rs          # ONNX op → NodeOp mapping
│   ├── weights.rs      # Initializer extraction
│   └── cli.rs          # hologram-onnx CLI
├── proto/
│   └── onnx.proto      # ONNX protobuf definitions
└── tests/
    └── models/         # Test ONNX models
```

### Dependencies

```toml
[dependencies]
hologram-compiler = { git = "https://github.com/UOR-Foundation/hologram" }
prost = "0.12"          # Protobuf parsing
prost-types = "0.12"
bytemuck = "1.14"       # Safe byte casting

[build-dependencies]
prost-build = "0.12"    # Compile onnx.proto
```

### Core Implementation

```rust
use hologram_compiler::{
    OperationGraph, GraphBuilder, NodeOp, WeightRef,
};

/// Parse ONNX model and produce OperationGraph + weights
pub fn compile_onnx(
    onnx_bytes: &[u8],
    config: &OnnxConfig,
) -> Result<(OperationGraph, Vec<u8>), OnnxError> {
    // 1. Parse protobuf
    let model = onnx::ModelProto::decode(onnx_bytes)?;
    let graph = model.graph.ok_or(OnnxError::NoGraph)?;

    let mut builder = GraphBuilder::new();
    let mut weight_buffer: Vec<u8> = Vec::new();
    let mut name_to_node: HashMap<String, NodeId> = HashMap::new();

    // 2. Create input nodes (exclude initializers)
    let initializer_names: HashSet<_> = graph.initializer
        .iter()
        .map(|i| i.name.as_str())
        .collect();

    for input in &graph.input {
        if !initializer_names.contains(input.name.as_str()) {
            let shape = parse_tensor_shape(&input.type_)?;
            let node = builder.add_input(&input.name, shape);
            name_to_node.insert(input.name.clone(), node);
        }
    }

    // 3. Process initializers (weights)
    for init in &graph.initializer {
        let data = extract_tensor_data(init)?;
        let shape: Vec<usize> = init.dims.iter().map(|&d| d as usize).collect();

        let node = if data.len() * 4 > config.weight_threshold {
            // Large weight → external file
            let offset = weight_buffer.len() as u64;
            weight_buffer.extend_from_slice(bytemuck::cast_slice(&data));

            builder.add_weight_ref(WeightRef {
                offset,
                length: data.len(),
                shape: shape.clone(),
            }, shape)
        } else {
            // Small weight → inline
            builder.add_constant_tensor(data, shape)
        };

        name_to_node.insert(init.name.clone(), node);
    }

    // 4. Process operations
    for node in &graph.node {
        let inputs: Vec<NodeId> = node.input
            .iter()
            .map(|name| name_to_node.get(name).copied())
            .collect::<Option<Vec<_>>>()
            .ok_or(OnnxError::MissingInput)?;

        let output_node = translate_onnx_op(
            &node.op_type,
            &inputs,
            &node.attribute,
            &mut builder,
        )?;

        // Map outputs
        for (i, output_name) in node.output.iter().enumerate() {
            name_to_node.insert(output_name.clone(), output_node);
        }
    }

    // 5. Mark outputs
    for output in &graph.output {
        if let Some(&node) = name_to_node.get(&output.name) {
            builder.mark_output(node);
        }
    }

    let graph = builder.build();
    Ok((graph, weight_buffer))
}
```

### ONNX Op Translation

```rust
fn translate_onnx_op(
    op_type: &str,
    inputs: &[NodeId],
    attrs: &[AttributeProto],
    builder: &mut GraphBuilder,
) -> Result<NodeId, OnnxError> {
    match op_type {
        // Matrix operations
        "MatMul" => Ok(builder.add_op(NodeOp::MatMul, vec![inputs[0], inputs[1]])),
        "Gemm" => {
            let alpha = get_attr_float(attrs, "alpha").unwrap_or(1.0);
            let beta = get_attr_float(attrs, "beta").unwrap_or(1.0);
            let trans_a = get_attr_int(attrs, "transA").unwrap_or(0) != 0;
            let trans_b = get_attr_int(attrs, "transB").unwrap_or(0) != 0;

            // A @ B + C with optional transposes and scaling
            let a = if trans_a { builder.add_op(NodeOp::Transpose, vec![inputs[0]]) } else { inputs[0] };
            let b = if trans_b { builder.add_op(NodeOp::Transpose, vec![inputs[1]]) } else { inputs[1] };
            let ab = builder.add_op(NodeOp::MatMul, vec![a, b]);

            if inputs.len() > 2 {
                Ok(builder.add_op(NodeOp::Add, vec![ab, inputs[2]]))
            } else {
                Ok(ab)
            }
        }

        // Element-wise binary
        "Add" => Ok(builder.add_op(NodeOp::Add, vec![inputs[0], inputs[1]])),
        "Sub" => Ok(builder.add_op(NodeOp::Sub, vec![inputs[0], inputs[1]])),
        "Mul" => Ok(builder.add_op(NodeOp::Mul, vec![inputs[0], inputs[1]])),
        "Div" => Ok(builder.add_op(NodeOp::Div, vec![inputs[0], inputs[1]])),

        // Activations
        "Relu" => Ok(builder.add_op(NodeOp::ReLU, vec![inputs[0]])),
        "Sigmoid" => Ok(builder.add_op(NodeOp::Sigmoid, vec![inputs[0]])),
        "Tanh" => Ok(builder.add_op(NodeOp::Tanh, vec![inputs[0]])),
        "Softmax" => {
            let axis = get_attr_int(attrs, "axis").unwrap_or(-1);
            Ok(builder.add_op(NodeOp::Softmax { axis }, vec![inputs[0]]))
        }

        // Shape operations
        "Reshape" => {
            // Shape comes from second input (constant)
            Ok(builder.add_op(NodeOp::Reshape, vec![inputs[0], inputs[1]]))
        }
        "Transpose" => {
            let perm = get_attr_ints(attrs, "perm");
            Ok(builder.add_op(NodeOp::Transpose, vec![inputs[0]]))
        }

        // Reductions
        "ReduceSum" => {
            let axes = get_attr_ints(attrs, "axes");
            let keepdims = get_attr_int(attrs, "keepdims").unwrap_or(1) != 0;
            Ok(builder.add_op(NodeOp::ReduceSum { axes, keepdims }, vec![inputs[0]]))
        }
        "ReduceMean" => {
            let axes = get_attr_ints(attrs, "axes");
            let keepdims = get_attr_int(attrs, "keepdims").unwrap_or(1) != 0;
            Ok(builder.add_op(NodeOp::ReduceMean { axes, keepdims }, vec![inputs[0]]))
        }

        _ => Err(OnnxError::UnsupportedOp(op_type.to_string())),
    }
}
```

### CLI

```rust
// hologram-onnx/src/cli.rs

fn main() -> Result<()> {
    let args = Args::parse();

    match args.command {
        Command::Compile { input, output } => {
            let onnx_bytes = fs::read(&input)?;
            let config = OnnxConfig::default();

            let (graph, weights) = compile_onnx(&onnx_bytes, &config)?;

            // Write .holo file (rkyv serialized graph)
            let holo_path = output.with_extension("holo");
            let graph_bytes = graph.to_bytes()?;
            fs::write(&holo_path, graph_bytes)?;

            // Write .weights file if non-empty
            if !weights.is_empty() {
                let weights_path = output.with_extension("weights");
                fs::write(&weights_path, &weights)?;
                println!("Compiled {} → {} + {}", input, holo_path, weights_path);
            } else {
                println!("Compiled {} → {}", input, holo_path);
            }
        }
    }

    Ok(())
}
```

### Usage

```bash
# In hologram-onnx repo
hologram-onnx compile resnet50.onnx -o resnet50

# Produces:
# resnet50.holo (OperationGraph, rkyv format)
# resnet50.weights (raw weight bytes)

# In hologram repo - run the model
hologram run resnet50.holo --input image.npy
```

### Testing

```rust
#[test]
fn test_simple_matmul() {
    let onnx = create_test_onnx_matmul();
    let (graph, weights) = compile_onnx(&onnx, &OnnxConfig::default()).unwrap();

    assert_eq!(graph.inputs().len(), 2);
    assert_eq!(graph.outputs().len(), 1);
    assert!(weights.is_empty()); // No initializers
}

#[test]
fn test_linear_with_weights() {
    // Load test model with weights
    let onnx = include_bytes!("../tests/models/linear.onnx");
    let (graph, weights) = compile_onnx(onnx, &OnnxConfig::default()).unwrap();

    assert!(!weights.is_empty());

    // Verify weight refs exist
    let has_weight_ref = graph.nodes().any(|n| matches!(n.op, NodeOp::WeightRef(_)));
    assert!(has_weight_ref);
}
```

### Priority ONNX Ops

Implement in this order for maximum model coverage:

1. **Core (covers ~80% of models):**

   - MatMul, Gemm, Add, Mul, Relu, Softmax
   - Reshape, Transpose, Concat

2. **Extended (covers ~95%):**

   - Conv, BatchNormalization, MaxPool, AveragePool
   - Sigmoid, Tanh, Gelu
   - ReduceSum, ReduceMean

3. **Full coverage:**
   - Attention ops: MultiHeadAttention (custom)
   - RNN ops: LSTM, GRU
   - Quantization: QuantizeLinear, DequantizeLinear

## Notes

- ONNX integration uses existing OperationGraph types - this guide is for the separate hologram-onnx repo
- Weight handling reuses existing `convert_tensors_to_refs()` pattern
- ONNX already has concrete shapes and weights, so shape inference is typically not needed
