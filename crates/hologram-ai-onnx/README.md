# hologram-ai-onnx

ONNX format support for [hologram-ai](../hologram-ai).

## Overview

This crate provides parsing and compilation of ONNX models to Hologram IR. It translates ONNX operations to Hologram's intermediate representation, which is then compiled to optimized backend plans using Hologram's compiler.

## Pure Hologram Architecture

**Everything runs through Hologram.**

This crate follows the **Pure Hologram Architecture** principle from CLAUDE.md:

- **All computation** flows through `hologram::compiler::compile_ir_with_header()`
- **All execution** uses `hologram::backend::PlanExecutor`
- **No external runtime dependencies** for core functionality
- **All operations** translate to Hologram IR

## Features

- **70+ ONNX Operations**: Comprehensive operation translator support
- **Symbolic Shapes**: Variable batch sizes and sequence lengths
- **Graph Partitioning**: Handle large models (>500 nodes) by splitting into partitions
- **Layer Detection**: Recognize transformer layer patterns for layer-wise compilation
- **Weight Management**: External and embedded weight storage options
- **Bundle Formats**: HOLB (unified bundle) and HOLM (pipeline bundle) support

## Usage

### Compiling ONNX Models

```rust
use hologram_ai_onnx::OnnxCompiler;
use std::fs;

// Create compiler
let compiler = OnnxCompiler::new();

// Compile ONNX to .holo format
let onnx_bytes = fs::read("model.onnx")?;
let (holo_bytes, weight_bytes) = compiler.compile(&onnx_bytes)?;

// Save compiled model
fs::write("model.holo", holo_bytes)?;
if !weight_bytes.is_empty() {
    fs::write("model.weights", weight_bytes)?;
}
```

### Loading and Executing

```rust
use hologram_ai::runtime::ModelExecutor;

// Load compiled model
let executor = ModelExecutor::from_holo_file("model.holo")?;

// Discover operations
let operations = executor.operations();
println!("Model has {} operations", operations.len());

// Get optimization report
let report = executor.optimization_report();
println!("SIMD level: {}", report.simd_level);
println!("Dynamic shapes: {}", report.dynamic_shapes);
```

## Weight Strategies

The compiler automatically selects weight storage strategies based on model size:

- **< 100MB**: Embedded in BackendPlan.constant_data (fast loading, RAM-resident)
- **100MB-1GB**: Page-aligned section in .holo file (memory-mapped)
- **> 1GB**: Separate .weights file (truly external)

See the [Integration Guide](../../specs/external-plans/hologram-integration.md) Section 6 for details.

## Symbolic Shapes

ONNX models with symbolic dimensions (e.g., `batch`, `seq_len`) are fully supported:

```rust
use hologram_ai_onnx::{Dim, SymbolicShape};

// Create a symbolic shape [batch, seq_len, 768]
let shape = SymbolicShape::new(vec![
    Dim::Symbolic("batch".into()),
    Dim::Symbolic("seq_len".into()),
    Dim::Static(768),
]);

assert_eq!(shape.rank(), 3);
assert!(!shape.is_fully_concrete());
```

This enables dynamic batch sizes and variable sequence lengths at runtime.

## Graph Partitioning

Large models (>500 nodes) can be partitioned to reduce memory usage during compilation:

```rust
use hologram_ai_onnx::GraphPartitioner;

// Create partitioner (500 nodes per partition)
let partitioner = GraphPartitioner::new();

// Partition large graph
let partitions = partitioner.partition(&graph)?;
println!("Split into {} partitions", partitions.len());

// Compile each partition independently
for partition in partitions {
    let compiled = compile_subgraph(&partition)?;
    // ...
}
```

## Layer Detection and Splitting

For transformer models, the crate can detect layer structure and enable layer-wise compilation:

```rust
use hologram_ai_onnx::core::{layer_detection, layer_splitter};

// Detect transformer layers
let layers = layer_detection::detect_transformer_layers(&graph)?;

if let Some(layers) = layers {
    println!("Found {} transformer layers", layers.len());

    // Split into per-layer models
    let layer_models = layer_splitter::split_by_layers(&model, &layers)?;

    // Compile each layer separately
    for (layer_name, layer_model) in layer_models {
        let holo_bytes = compile_layer(&layer_model)?;
        fs::write(format!("{}.holo", layer_name), holo_bytes)?;
    }
}
```

Supported layer naming patterns:
- BERT: `encoder.layer.N.*`, `decoder.layer.N.*`
- GPT-2: `transformer.h.N.*`
- LLaMA: `model.layers.N.*`
- T5: `encoder.block.N.*`, `decoder.block.N.*`

## Integration Guide

For detailed integration patterns and best practices, see:

- **specs/external-plans/hologram-integration.md** - Complete integration guide
  - Section 4: Operation Discovery
  - Section 5: Input Ordering
  - Section 6: Weight Handling Strategies
  - Section 7: Optimization Features
  - Section 8: Testing Recommendations

## Examples

See the integration checklist tests for working examples:

- **tests/integration_checklist.rs** - Comprehensive examples of:
  - Operation discovery
  - Weight strategy selection
  - Dynamic shape support
  - Optimization feature detection
  - End-to-end compilation and execution

## Supported Operations

The crate includes translators for 70+ ONNX operations across multiple categories:

- **Core**: Add, Sub, Mul, Div, Pow, Sqrt, Neg, Abs
- **Matrix**: MatMul, Gemm, Conv, ConvTranspose
- **Activation**: ReLU, Sigmoid, Tanh, GELU, SiLU, Swish
- **Normalization**: BatchNormalization, LayerNormalization, InstanceNormalization
- **Pooling**: MaxPool, AveragePool, GlobalAveragePool
- **Shape**: Reshape, Transpose, Squeeze, Unsqueeze, Concat, Split
- **Reduction**: ReduceSum, ReduceMean, ReduceMax, ReduceMin
- **Advanced**: Softmax, LogSoftmax, Cast, Gather, Where, Range
- **And many more...**

For the complete list, see the `src/translators/` directory.

## Architecture

```
ONNX Model (.onnx)
    ↓ [parse]
ONNX Protocol Buffer (GraphProto)
    ↓ [translate_graph_to_ir]
Hologram IR (OperationGraph)
    ↓ [hologram::compiler::compile_ir_with_header]
Backend Plan
    ↓ [serialize_backend_plan_with_header]
.holo File
    ↓ [ModelExecutor::from_holo_file]
PlanExecutor (execution)
```

Every step uses Hologram's native data structures and compiler. No external runtime dependencies.

## Contributing

When adding new ONNX operations:

1. **Implement in pure Rust** - No external runtime dependencies
2. **Write comprehensive tests** - Test all code paths and edge cases
3. **Support symbolic shapes** - Variable batch/seq_len dimensions
4. **Implement constant folding** - When all inputs are constants
5. **Document thoroughly** - Rustdoc for public APIs

See `CLAUDE.md` for detailed contribution guidelines.
