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
- **HOLM Pipeline Bundles**: Layer-wise compilation with memory-mapped execution for large transformer models

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

## T5 Model Compilation and Execution

hologram-ai provides seamless support for encoder-decoder transformer models like T5, with automatic tokenization and text generation.

### Prerequisites

Download or export your T5 model in ONNX format with encoder and decoder components:

```bash
# Example directory structure
models/t5-small/
├── encoder.onnx          # Encoder model (text → hidden states)
├── decoder.onnx          # Decoder model (hidden states → logits)
└── tokenizer.json        # SentencePiece tokenizer
```

### Compiling T5 Models

Use the `compile-pipeline` command to compile encoder, decoder, and tokenizer into a single optimized bundle:

```bash
# Compile T5 encoder, decoder, and tokenizer to a single .holo pipeline bundle
cargo run -p hologram-ai -- compile-pipeline \
  --encoder models/t5-small/encoder.onnx \
  --decoder models/t5-small/decoder.onnx \
  --tokenizer models/t5-small/tokenizer.json \
  -o models/t5-small/pipeline.holo
```

This creates a **HOLM pipeline bundle** (`pipeline.holo`) containing:
- Compiled encoder model (HOLB format)
- Compiled decoder model (HOLB format)
- SentencePiece tokenizer with vocabulary
- Memory-mapped weights for efficient loading

#### Compilation Options

```bash
# With graph partitioning for large models
cargo run -p hologram-ai -- compile-pipeline \
  --encoder encoder.onnx \
  --decoder decoder.onnx \
  --tokenizer tokenizer.json \
  --partition \
  --partition-size 500 \
  -o pipeline.holo

# Custom weight threshold (default: 4096 bytes)
cargo run -p hologram-ai -- compile-pipeline \
  --encoder encoder.onnx \
  --decoder decoder.onnx \
  --tokenizer tokenizer.json \
  --weight-threshold 8192 \
  -o pipeline.holo

# Keep intermediate files for debugging
cargo run -p hologram-ai -- compile-pipeline \
  --encoder encoder.onnx \
  --decoder decoder.onnx \
  --tokenizer tokenizer.json \
  --keep-intermediates \
  -o pipeline.holo
```

### Running T5 Text Generation

Execute the compiled pipeline bundle with a text prompt:

```bash
# Basic text generation
cargo run -p hologram-ai -- run-pipeline models/t5-small/pipeline.holo \
  --prompt "Tell me a joke"

# Translation
cargo run -p hologram-ai -- run-pipeline models/t5-small/pipeline.holo \
  --prompt "translate English to French: Hello, how are you?"

# Summarization
cargo run -p hologram-ai -- run-pipeline models/t5-small/pipeline.holo \
  --prompt "summarize: The quick brown fox jumps over the lazy dog"
```

#### Generation Parameters

Control text generation with advanced sampling parameters:

```bash
cargo run -p hologram-ai -- run-pipeline pipeline.holo \
  --prompt "Tell me a joke" \
  --max-tokens 100 \        # Maximum new tokens (default: 50)
  --min-tokens 12 \         # Minimum tokens before EOS (default: 12)
  --top-k 40 \              # Top-k sampling (default: 40)
  --temperature 0.9 \       # Sampling temperature (default: 0.9)
  --beam-size 1 \           # Beam search width (default: 1)
  --length-penalty 1.0 \    # Length penalty for beam search (default: 1.0)
  --no-repeat-ngram 3 \     # Block n-gram repetition (default: 3)
  --eos-prob-threshold 0.2  # Minimum EOS probability (default: 0.2)
```

#### Expected Output

```
=== T5 Pipeline Text Generation ===
Pipeline: models/t5-small/pipeline.holo
Prompt: "Tell me a joke"
Max new tokens: 50

Loading pipeline bundle...
✓ Encoder model loaded (12.3 MB)
✓ Decoder model loaded (18.7 MB)
✓ Tokenizer loaded (vocab size: 32128)

Tokenizing prompt...
✓ Input tokens: 5 tokens

Running encoder...
✓ Encoder complete (42.3ms)

Generating text...
Step 1/50 | Token: Why
Step 2/50 | Token: did
Step 3/50 | Token: the
...
✓ Generation complete (1.2s, 45 tokens)

=== Generated Output ===
Why did the chicken cross the road? To get to the other side!
```

### Categorical Instructions Support

The compiled `.holo` files automatically leverage hologram's **13 categorical instructions** for optimized execution:

| Instruction | Purpose | Performance Benefit |
|-------------|---------|---------------------|
| `CharProduct` | Monster group character table lookup | O(1) 194×194 matrix access |
| `OrbitClassify` | Classify bytes into 32 orbit classes | Parallel reduction optimization |
| `Lift` | Project to resonance space (mod 96) | Exact arithmetic for activations |
| `MultInverse` | Multiplicative inverse mod 256 | Fast normalization |
| `TorusMap` | Map byte to torus coordinates | Efficient embedding lookups |

These instructions are automatically used by the hologram backend when executing:

- Activation functions (sigmoid, tanh, GELU)
- Attention mechanisms (Q/K/V projections)
- Embedding table lookups
- Normalization layers

**No special flags needed** - categorical optimizations are automatically applied during execution based on the operations in your model.

### Alternative: Compile Individual Models

For advanced use cases, you can compile encoder and decoder separately:

```bash
# Compile encoder
cargo run -p hologram-ai -- compile models/t5-small/encoder.onnx \
  -o models/t5-small/encoder.holo

# Compile decoder
cargo run -p hologram-ai -- compile models/t5-small/decoder.onnx \
  -o models/t5-small/decoder.holo

# Run encoder with prompt (requires text-output feature)
cargo run -p hologram-ai --features text-output -- run \
  models/t5-small/encoder.holo \
  --prompt "Tell me a joke" \
  --tokenizer models/t5-small/tokenizer.json
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

## ISA Optimizations

hologram-onnx leverages Hologram's ISA-level optimizations for high performance:

### LOOP Instructions
- **O(1) instruction space**: Tensor operations expressed as single LOOP instructions
- **Hardware loop support**: Efficient iteration without branch overhead
- **Example**: 1M element ReLU = 1 LOOP instruction, not 1M separate operations

### PhiCoordinate Addressing
- **Zero-copy transpose**: Virtual memory layout changes without data movement
- **Stride-based access**: Efficient multi-dimensional array traversal
- **Example**: NCHW↔NHWC conversion with zero copy overhead

### ClassMap Fusion
- **Single-pass element-wise chains**: Fuse Add+ReLU, Mul+Add+Sigmoid, etc.
- **Reduced memory bandwidth**: One read/write pass for multiple operations
- **Example**: BatchNorm+ReLU fused into single kernel

### Im2Col + GEMM Decomposition
- **Optimal Conv2D**: Convolutions decomposed to matrix multiplication
- **SIMD vectorization**: 4-wide SIMD for matrix operations
- **Cache efficiency**: Tile-based GEMM with optimal memory access patterns

## HOLM Pipeline Archives

For large transformer models, hologram-onnx supports layer-wise compilation into HOLM (Hologram Layer Model) archives. This enables memory-efficient inference on models larger than available RAM.

### Features

- **Automatic Layer Detection**: Recognizes transformer patterns (BERT, T5, GPT-2, LLaMA, etc.)
- **Memory-Mapped Weights**: Weights accessed via mmap, never fully loaded into memory
- **Streaming Execution**: Prefetch layer N+1 while executing layer N, release layer N-1
- **Embedded Metadata**: Tokenizer vocabulary, generation config, and model metadata in single file

### Supported Architectures

| Pattern | Example Models |
|---------|---------------|
| `encoder.layer.N` | BERT, RoBERTa, DistilBERT |
| `decoder.block.N` | T5, BART |
| `transformer.h.N` | GPT-2, OPT |
| `model.layers.N` | LLaMA, Mistral |

### Usage

```bash
# Compile with layer-wise splitting
cargo run -p hologram-ai -- compile model.onnx -o model.holo --layer-wise

# Execute with streaming (automatic for HOLM files)
cargo run -p hologram-ai -- run model.holo --input input.json
```

### Format Layout

```
HOLM File:
├── Header (64 bytes) - magic "HOLM", model count
├── Index - per-layer offset, size, CRC32
├── [4KB alignment padding]
├── Layer 0 (HOLB) - graph + weights (page-aligned)
├── Layer 1 (HOLB) - graph + weights (page-aligned)
└── ...
```

Each layer is page-aligned (4KB) for efficient memory mapping.

## Performance

### Compilation Performance

| Model | Nodes | Compilation Time | Output Size |
|-------|-------|------------------|-------------|
| MNIST | 26 | 13ms | 304 B |
| ResNet50 | 122 | 263ms | 6.8 KB |
| ResNet50 (partitioned) | 122 | 168ms | 6.8 KB |

### Memory Efficiency

- **MNIST (26 KB)**: < 10 MB peak memory
- **ResNet50 (98 MB)**: ~400 MB peak memory
- **ResNet50 + partitioning**: ~200 MB peak memory (50% reduction)

### Graph Partitioning

For large models (3000+ nodes), enable partitioning:

```bash
hologram-onnx compile large_model.onnx -o output.holo \
    --partition --partition-size 100
```

## Benchmarks

Run performance benchmarks with:

```bash
# All benchmarks
cargo bench

# Compilation benchmarks (parsing, decomposition, full pipeline)
cargo bench --bench compilation_bench

# Execution benchmarks (Conv2D, MatMul, attention, elementwise)
cargo bench --bench execution_bench
```

See [docs/working/benchmarks.md](docs/working/benchmarks.md) for detailed benchmark documentation.

## Performance Profiling

Use `--profile` to identify bottlenecks in compilation and execution:

```bash
# Profile compilation
cargo run -p hologram-ai -- --profile compile model.onnx -o model.holo

# Profile execution
cargo run -p hologram-ai -- --profile run model.holo
```

This outputs timing for each phase:

```
INFO compile_onnx{onnx_size=12345678}: close time.busy=2.3s
INFO   parse_onnx: close time.busy=150ms
INFO   translate_to_ir{nodes=1234}: close time.busy=800ms
INFO   compile_ir: close time.busy=1.2s
INFO   serialize_holo: close time.busy=50ms
```

### Compilation Phases

| Phase | Description |
|-------|-------------|
| `parse_onnx` | Parse ONNX protobuf |
| `translate_to_ir` | Convert ONNX ops to hologram IR |
| `compile_ir` | Compile IR to BackendPlan |
| `serialize_holo` | Serialize to .holo format |
| `create_bundle` | Create unified bundle (if --bundle) |

### Execution Phases

| Phase | Description |
|-------|-------------|
| `load_unified_bundle` | Load and parse .holo bundle |
| `mmap_bundle` | Memory-map the file |
| `deserialize_graph` | Deserialize computation graph |
| `create_executor` | Create execution plan |
| `model_execute` | Full model execution |
| `execute_plan` | Run the computation graph |
| `input_mapping` | Upload inputs to backend |
| `download_outputs` | Download results from backend |

Without `--profile`, these spans have zero overhead and produce no output.

### Deep Profiling (Hologram Internals)

To see timing spans from inside the hologram backend (kernels, lookup tables, buffer operations), build with the `instrumentation` feature:

```bash
# Build with deep instrumentation
cargo build -p hologram-ai --features instrumentation

# Run with profiling to see all spans
cargo run -p hologram-ai --features instrumentation -- --profile run model.holo
```

This enables tracing in:
- `hologram-lookup` - Lookup table operations
- `hologram-backend` - Buffer allocation, kernel dispatch
- `hologram-compiler` - IR compilation phases

## Development

```bash
# Build all crates
cargo build --release

# Run all tests (lightweight only, ~5s)
cargo test --workspace --lib

# Check for issues
cargo clippy --all-targets

# Generate documentation
cargo doc --no-deps --open

# Run benchmarks
cargo bench
```

### Test Tiers

Tests are organized into tiers based on resource requirements:

**Lightweight tests** (run by default):
- Unit tests in `--lib`
- Mock model tests in `lightweight_tests.rs`
- Fast parsing tests (~0.5s each)

```bash
# Run all lightweight tests
cargo test --workspace --lib
```

**Heavyweight tests** (ignored by default):
- Full model compilation (BERT, T5, etc.)
- Require ~2GB memory
- Take ~17s per compilation
- Require model files in `models/` directory

```bash
# Run BERT compilation tests
cargo test -p hologram-ai-onnx --test bert_compilation -- --ignored

# Run BERT decoding test
cargo test -p hologram-ai-onnx --test bert_decode -- --ignored

# Run all ignored tests
cargo test --workspace -- --ignored
```

**Integration tests** (CI only):
- Download MNIST model from GitHub
- Full compilation and execution pipeline

```bash
# Run integration tests locally
mkdir -p crates/hologram-ai-onnx/tests/fixtures
curl -L https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-12.onnx \
  -o crates/hologram-ai-onnx/tests/fixtures/mnist-12.onnx
cargo test -p hologram-ai-onnx --test '*'
```

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
