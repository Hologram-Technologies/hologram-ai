# hologram-onnx

An ONNX compiler and runtime using Hologram as the execution backend.

## Overview

hologram-onnx compiles ONNX models to Hologram's optimized `.holo` format, enabling high-performance inference with ISA-level optimizations including LOOP instructions, PhiCoordinate addressing, and ClassMap fusion.

## Features

- **39 ONNX Operations**: Full support for core, activation, shape, convolution, normalization, pooling, reduction, and advanced operations
- **Symbolic Shapes**: Variable batch sizes and sequence lengths for flexible deployment
- **ISA Optimizations**: Conv2D → Im2Col+GEMM decomposition, SIMD vectorization, O(1) space complexity
- **Weight Deduplication**: Automatic detection and deduplication of identical weights
- **Graph Partitioning**: Support for large models (3000+ nodes) with memory-efficient compilation

## Architecture

```
hologram-onnx/
├── crates/
│   ├── hologram-onnx-spec/     # ONNX protobuf definitions
│   ├── hologram-onnx-core/     # Parsing, validation, compilation
│   ├── hologram-onnx-ops/      # Operation translators (39 ops)
│   ├── hologram-onnx-config/   # Configuration and output handlers
│   └── hologram-onnx-cli/      # Command-line interface
```

### Crates

| Crate | Description |
|-------|-------------|
| `hologram-onnx-spec` | ONNX protobuf definitions compiled from official specification |
| `hologram-onnx-core` | Model parsing, validation, shape inference, and compilation |
| `hologram-onnx-ops` | Operation translators with symbolic shape support |
| `hologram-onnx-config` | Compilation configuration and output format handlers |
| `hologram-onnx-cli` | CLI for compiling and validating ONNX models |

## Quick Start

### Requirements

- Rust 2024 edition
- `protoc` (Protocol Buffers compiler)

### Building

```bash
cargo build
```

### CLI Usage

```bash
# Compile an ONNX model to .holo format
cargo run -- compile model.onnx -o model.holo

# Validate an ONNX model
cargo run -- validate model.onnx

# Show model information
cargo run -- info model.onnx
```

### Library Usage

```rust
use hologram_onnx_core::{OnnxCompiler, OnnxConfig, parse_model, validate_model};

// Parse and validate an ONNX model
let model_bytes = std::fs::read("model.onnx")?;
let model = parse_model(&model_bytes)?;
validate_model(&model)?;

// Compile to .holo format
let compiler = OnnxCompiler::new();
let (holo_bytes, weight_bytes) = compiler.compile(&model_bytes)?;
```

## Supported Operations

### Tier 1 (Core)
- **Core**: MatMul, Gemm, Add, Sub, Mul, Div, Pow
- **Activation**: Relu, Sigmoid, Tanh, Softmax, Gelu, Swish, Elu, Selu
- **Shape**: Reshape, Transpose, Squeeze, Unsqueeze, Concat, Split

### Tier 2 (CNN)
- **Convolution**: Conv, ConvTranspose
- **Normalization**: BatchNormalization, LayerNormalization, InstanceNormalization
- **Pooling**: MaxPool, AveragePool, GlobalAveragePool

### Tier 3 (Advanced)
- **Reduction**: ReduceSum, ReduceMean, ReduceMax, ReduceMin, ReduceProd
- **Attention**: Attention, MultiHeadAttention
- **RNN**: LSTM, GRU, RNN

## Development

```bash
# Build all crates
cargo build

# Run all tests
CARGO_NET_GIT_FETCH_WITH_CLI=true cargo test

# Run specific test suite
cargo test -p hologram-onnx-core --test decomposition_tests

# Check for issues
cargo clippy --all-targets

# Generate documentation
cargo doc --no-deps --open
```

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
