# hologram-onnx Tutorial

This tutorial provides a step-by-step guide for compiling and running ONNX models with hologram-onnx.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Compiling ONNX Models](#compiling-onnx-models)
3. [Pipeline Configuration](#pipeline-configuration)
4. [Output Handlers](#output-handlers)
5. [Symbolic Shapes](#symbolic-shapes)
6. [Large Model Handling](#large-model-handling)
7. [Troubleshooting](#troubleshooting)

## Getting Started

### Prerequisites

- Rust 2024 edition (nightly)
- `protoc` (Protocol Buffers compiler)
- ONNX model to compile

### Installation

```bash
# Clone and build
git clone https://github.com/anthropics/hologram-onnx.git
cd hologram-onnx
cargo build --release

# Verify installation
./target/release/hologram-onnx --version
```

### Download Test Models

```bash
# MNIST (26 KB)
mkdir -p crates/hologram-onnx-core/tests/fixtures
curl -L https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-12.onnx \
  -o crates/hologram-onnx-core/tests/fixtures/mnist-12.onnx

# ResNet50 (98 MB, optional)
mkdir -p models
curl -L https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v1-7.onnx \
  -o models/resnet50-v1-7.onnx
```

## Compiling ONNX Models

### Basic Compilation

```bash
# Compile a model
hologram-onnx compile model.onnx -o model.holo

# With verbose output
hologram-onnx compile model.onnx -o model.holo -v
```

This produces two files:
- `model.holo` - Compiled program (operations and structure)
- `model.holo.weights` - Model weights (optional, for weight deduplication)

### Model Information

```bash
# Show model structure
hologram-onnx info model.onnx

# Detailed node list
hologram-onnx info model.onnx --detailed
```

### Model Validation

```bash
# Validate without compiling
hologram-onnx validate model.onnx
```

## Pipeline Configuration

Pipeline configurations control compilation and inference behavior.

### Basic Configuration

Create `config.toml`:

```toml
[pipeline]
name = "mnist-classifier"
description = "MNIST digit classification"

[pipeline.model]
path = "models/mnist-12.onnx"
format = "onnx"

[pipeline.output]
path = "output/mnist.holo"
format = "holo"
```

### Compilation Options

```toml
[pipeline.compilation]
# Enable weight deduplication
deduplicate_weights = true

# Enable graph partitioning for large models
partition = false
partition_size = 100

# Memory budget (MB) for compilation
memory_budget = 4096

# Optimization level (0-3)
optimization_level = 2
```

### Input Preprocessing

```toml
[pipeline.preprocessing]
# Normalize inputs
normalize = true
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Resize input images
resize = [224, 224]

# Color channel order
channel_order = "rgb"  # or "bgr"
```

### Using Configuration

```bash
# Compile with config
hologram-onnx compile --config config.toml

# Override specific options
hologram-onnx compile --config config.toml --partition --partition-size 50
```

## Output Handlers

Output handlers process model outputs into domain-specific formats.

### Image Output Handler

For image generation/segmentation models:

```toml
[pipeline.handlers.segmentation]
handler_type = "image"
output = "segmentation_mask"

[pipeline.handlers.segmentation.config]
format = "png"
colormap = "viridis"
width = 512
height = 512
```

### Audio Output Handler

For speech synthesis/audio models:

```toml
[pipeline.handlers.speech]
handler_type = "audio"
output = "waveform"

[pipeline.handlers.speech.config]
format = "wav"
sample_rate = 22050
channels = 1
```

### Text Output Handler

For NLP/text generation models:

```toml
[pipeline.handlers.text]
handler_type = "text"
output = "tokens"

[pipeline.handlers.text.config]
tokenizer = "gpt2"
decode = true
```

### Enabling Features

Output handlers require feature flags:

```bash
# Build with all output handlers
cargo build --release --features "image-output,audio-output,text-output"

# Build with specific handlers
cargo build --release --features "image-output"
```

## Symbolic Shapes

hologram-onnx supports variable dimensions for flexible deployment.

### Variable Batch Size

Models can accept any batch size at runtime:

```rust
use hologram_onnx_core::SymbolicShape;
use hologram_compiler::shapes::Dim;

// Shape: [batch, 3, 224, 224] where batch is variable
let shape = SymbolicShape::new(vec![
    Dim::var("batch"),
    Dim::fixed(3),
    Dim::fixed(224),
    Dim::fixed(224),
]);
```

### Variable Sequence Length

For transformer models with variable input lengths:

```rust
// Shape: [batch, seq_len, hidden]
let shape = SymbolicShape::new(vec![
    Dim::var("batch"),
    Dim::var("seq_len"),
    Dim::fixed(768),
]);
```

### Shape Inference

Shapes propagate automatically through operations:

```rust
// Input: [batch, 64, 112, 112]
// Conv2D with 3x3 kernel, stride 2
// Output: [batch, 128, 55, 55] (computed automatically)
```

## Large Model Handling

### Graph Partitioning

For models with 3000+ nodes:

```bash
# Enable partitioning with 100 nodes per partition
hologram-onnx compile large_model.onnx -o output.holo \
    --partition \
    --partition-size 100
```

### Memory Budget

Set a memory limit for compilation:

```bash
# Limit to 4 GB
hologram-onnx compile large_model.onnx -o output.holo \
    --partition \
    --partition-size 100 \
    --memory-budget 4096
```

### Recommended Settings by Model Size

| Model Size | Partition | Partition Size | Memory Budget |
|------------|-----------|----------------|---------------|
| < 100 MB | No | N/A | Default |
| 100-500 MB | Optional | 200 | 4096 MB |
| 500 MB-2 GB | Yes | 100 | 8192 MB |
| > 2 GB | Yes | 50 | 16384 MB |

## Troubleshooting

### Common Errors

#### "Unsupported operation: XYZ"

The operation is not yet implemented. Check supported operations in README.md.

**Solution**: Use ONNX simplifier to convert to supported operations:

```bash
pip install onnx-simplifier
onnxsim model.onnx model_simplified.onnx
```

#### "Shape inference failed"

Input shapes couldn't be determined.

**Solution**: Ensure model has complete shape information:

```python
import onnx
from onnx import shape_inference
model = onnx.load("model.onnx")
model = shape_inference.infer_shapes(model)
onnx.save(model, "model_with_shapes.onnx")
```

#### "Out of memory during compilation"

Model is too large for available memory.

**Solution**: Enable partitioning:

```bash
hologram-onnx compile model.onnx -o output.holo \
    --partition --partition-size 50
```

#### "Dimension mismatch"

Operation input shapes don't match.

**Solution**: Check model with:

```bash
hologram-onnx validate model.onnx
hologram-onnx info model.onnx --detailed
```

#### "Feature not enabled: image-output"

Output handler requires feature flag.

**Solution**: Rebuild with feature:

```bash
cargo build --release --features "image-output"
```

### Debugging

#### Verbose Logging

```bash
# Enable debug logging
RUST_LOG=debug hologram-onnx compile model.onnx -o output.holo

# Trace-level for maximum detail
RUST_LOG=trace hologram-onnx compile model.onnx -o output.holo
```

#### Model Inspection

```bash
# List all operations
hologram-onnx info model.onnx --detailed

# Python inspection
python -c "import onnx; m = onnx.load('model.onnx'); print(m.graph)"
```

#### Shape Debugging

For shape inference issues:

```rust
use hologram_onnx_core::SymbolicShape;

let shape = SymbolicShape::concrete(vec![1, 3, 224, 224]);
println!("Shape: {:?}", shape);
println!("Dims: {:?}", shape.dims());
println!("Has variable: {}", shape.has_variable_dims());
```

### Performance Tips

1. **Use release builds**: `cargo build --release`
2. **Enable partitioning** for large models
3. **Preprocess offline**: Normalize/resize before inference
4. **Weight deduplication**: Reduces binary size for models with shared weights
5. **Profile with benchmarks**: `cargo bench` to identify bottlenecks

### Getting Help

- Check [benchmarks.md](benchmarks.md) for performance guidance
- Check [memory-analysis.md](memory-analysis.md) for memory profiling
- File issues at: https://github.com/anthropics/hologram-onnx/issues
