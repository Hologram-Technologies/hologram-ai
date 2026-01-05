# hologram-onnx Examples

This directory contains examples demonstrating various features of hologram-onnx.

## Examples

### 1. Basic Compilation (`basic_compilation.rs`)

**What it demonstrates:**
- Basic ONNX model creation using proto types
- Simple compilation using `compile_onnx()` function
- Writing output files to disk

**Run:**
```bash
cargo run --example basic_compilation
```

**Output:**
- `model.holo` - Compiled hologram IR graph
- `model.weights` - External weight data (if any)

---

### 2. Advanced Compilation (`advanced_compilation.rs`)

**What it demonstrates:**
- Custom compilation configuration using `OnnxConfig`
- Graph partitioning for large models
- Weight threshold configuration
- Conv2D decomposition options
- Memory budget limits

**Run:**
```bash
cargo run --example advanced_compilation
```

**Key features:**
- `OnnxCompiler::with_config()` - Custom compilation settings
- `enable_partitioning` - For models with 500+ nodes
- `decompose_conv2d` - Conv2D → Im2col+GEMM optimization
- `memory_budget` - RAM limit enforcement

---

### 3. Symbolic Shapes (`symbolic_shapes.rs`)

**What it demonstrates:**
- Variable batch sizes (essential for inference servers)
- Variable sequence lengths (NLP models like BERT)
- Multiple symbolic dimensions (adaptive pooling)
- Creating symbolic ONNX graphs

**Run:**
```bash
cargo run --example symbolic_shapes
```

**Use cases:**
- **Batch size:** Compile once, run with any batch size (1, 8, 32, 64, etc.)
- **Sequence length:** NLP models that handle variable text lengths
- **Dynamic dimensions:** Adaptive pooling, resizing operations

**Example shapes:**
```rust
// Variable batch
[batch, 3, 224, 224]  // batch is symbolic

// Variable sequence (BERT)
[batch, seq_len, 768]  // both batch and seq_len are symbolic

// Multiple symbolic
[batch, channels, height, width]  // all except channels are symbolic
```

---

### 4. T5 Text-to-Text (`t5_compilation.rs`)

**What it demonstrates:**
- Compiling Google's T5 encoder-decoder model
- Text-to-text generation architecture
- Handling large models with partitioning (600-800 nodes per component)
- Encoder-decoder workflow

**Run:**
```bash
cargo run --example t5_compilation
```

**Prerequisites:**
```bash
pip install optimum[exporters]
optimum-cli export onnx \
  --model google/t5-small \
  --task text2text-generation-with-past \
  /workspace/models/t5-small/
```

**Key features:**
- **Encoder:** Processes input text into hidden states
- **Decoder:** Generates output text token-by-token
- **Partitioning:** Handles 600-800 nodes per component efficiently
- **Text tasks:** Translation, summarization, question answering

**Output:**
- `encoder.holo` - Compiled T5 encoder
- `encoder.weights` - Encoder weights (if external)
- `decoder.holo` - Compiled T5 decoder
- `decoder.weights` - Decoder weights (if external)

---

## Quick Start

1. **Clone and build:**
   ```bash
   git clone <repo>
   cd hologram-onnx
   cargo build --examples
   ```

2. **Run an example:**
   ```bash
   cargo run --example basic_compilation
   ```

3. **Check outputs:**
   ```bash
   ls -lh *.holo *.weights
   ```

## Integration with hologram CLI

After compiling with hologram-onnx, you can execute the models using the hologram CLI:

```bash
# Compile ONNX → .holo
cargo run --example basic_compilation

# Execute with hologram
hologram run model.holo --weights model.weights --input input.bin --output output.bin
```

## Common Patterns

### Creating ONNX Models

```rust
use hologram_onnx::proto::*;

let graph = GraphProto {
    name: "my_model".to_string(),
    input: vec![/* inputs */],
    output: vec![/* outputs */],
    node: vec![/* operations */],
    initializer: vec![/* weights */],
    ..Default::default()
};

let model = ModelProto {
    ir_version: 8,
    opset_import: vec![],
    graph: Some(graph),
    ..Default::default()
};
```

### Compiling Models

```rust
use hologram_onnx::compile_onnx;

// Simple compilation
let (holo_bytes, weight_bytes) = compile_onnx(&onnx_bytes)?;

// With configuration
use hologram_onnx::{OnnxCompiler, OnnxConfig};

let config = OnnxConfig {
    weight_threshold: 4096,
    enable_partitioning: true,
    partition_size: 500,
    decompose_conv2d: true,
    decompose_pooling: true,
    memory_budget: Some(8 * 1024), // 8 GB
};

let compiler = OnnxCompiler::with_config(config);
let (holo_bytes, weight_bytes) = compiler.compile(&onnx_bytes)?;
```

### Symbolic Shapes

```rust
use tensor_shape_proto::Dimension;

// Static dimension
Dimension {
    value: Some(tensor_shape_proto::dimension::Value::DimValue(224)),
    ..Default::default()
}

// Symbolic dimension
Dimension {
    value: Some(tensor_shape_proto::dimension::Value::DimParam("batch".to_string())),
    ..Default::default()
}
```

## Supported Operations

hologram-onnx supports 50+ ONNX operations across multiple categories:

- **Core:** Add, Sub, Mul, Div, MatMul, Gemm
- **Activation:** Relu, Sigmoid, Tanh, Softmax, Gelu
- **Convolution:** Conv, ConvTranspose
- **Pooling:** MaxPool, AveragePool, GlobalAveragePool
- **Normalization:** LayerNorm, BatchNorm, GroupNorm, InstanceNorm
- **Shape:** Reshape, Transpose, Concat, Slice, Gather
- **Reduction:** ReduceSum, ReduceMean, ReduceMax, ReduceMin
- **Advanced:** Attention, LayerNormalization, Cast

See `/workspace/src/ops/translator.rs` for the complete list.

## Performance Tips

1. **Enable Conv2D decomposition** for better GEMM optimization:
   ```rust
   config.decompose_conv2d = true;
   ```

2. **Use partitioning for large models** (3000+ nodes):
   ```rust
   config.enable_partitioning = true;
   config.partition_size = 500;
   ```

3. **Set memory budget** to avoid OOM:
   ```rust
   config.memory_budget = Some(16 * 1024); // 16 GB
   ```

4. **External weights** for large models:
   ```rust
   config.weight_threshold = 1024; // Smaller = more external
   ```

## Troubleshooting

### Common Errors

**"Unsupported operation":**
- Check if the operation is listed in supported operations
- Some operations may require specific opset versions

**"Shape inference failed":**
- Ensure input shapes are properly specified
- Check that all intermediate shapes can be inferred

**"Memory budget exceeded":**
- Increase `memory_budget` in config
- Enable graph partitioning
- Use external weight storage

### Getting Help

- Check `/workspace/docs/` for detailed documentation
- Run tests: `cargo test --workspace`
- File issues: `<repo>/issues`

## License

See `/workspace/LICENSE` for licensing information.
