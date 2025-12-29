# Performance Benchmarks

This document describes the performance benchmarks for the hologram-onnx compiler and runtime.

## Running Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark suite
cargo bench --bench compilation_bench
cargo bench --bench execution_bench

# Run with baseline comparison
cargo bench -- --baseline main

# Run specific benchmark
cargo bench -- mnist_parse
```

## Benchmark Suites

### Compilation Benchmarks (`benches/compilation_bench.rs`)

Measures the ONNX compilation pipeline performance.

| Benchmark | Description | Models |
|-----------|-------------|--------|
| `onnx_parsing` | Raw protobuf deserialization | MNIST, ResNet50 |
| `decomposition` | IR decomposition pass | Various Conv2D sizes |
| `full_compilation` | End-to-end compilation | MNIST, ResNet50 |
| `partitioned_compilation` | Compilation with graph partitioning | ResNet50 |
| `graph_size_scaling` | IR creation scalability | 10-500 operations |

#### Decomposition Size Configurations

| Name | Input Shape | Kernel Shape |
|------|-------------|--------------|
| `small` | [1, 3, 32, 32] | [16, 3, 3, 3] |
| `medium` | [1, 64, 112, 112] | [64, 64, 3, 3] |
| `large` | [1, 256, 56, 56] | [256, 256, 3, 3] |
| `resnet_block` | [1, 512, 28, 28] | [512, 512, 3, 3] |

### Execution Benchmarks (`benches/execution_bench.rs`)

Measures IR construction performance for various operations.

| Benchmark | Description | Configurations |
|-----------|-------------|----------------|
| `conv2d_execution` | Conv2D IR creation | Small to ResNet-sized |
| `matmul_execution` | MatMul IR creation | 64x64 to 1024x1024 |
| `batched_matmul` | Attention Q*K^T patterns | BERT, GPT-2 configs |
| `elementwise_ops` | ReLU, Add + ReLU fusion | 1K to 16M elements |
| `loop_instructions` | LOOP instruction scaling | 1K to 1M iterations |
| `softmax` | Softmax IR creation | Vocabulary sizes |
| `transpose` | Transpose IR creation | 2D and 4D layouts |

#### Conv2D Configurations

| Name | Input | Kernel | FLOPs |
|------|-------|--------|-------|
| `small_3x3` | [1, 3, 32, 32] | [16, 3, 3, 3] | ~125K |
| `medium_3x3` | [1, 64, 112, 112] | [64, 64, 3, 3] | ~1.4B |
| `resnet_first` | [1, 3, 224, 224] | [64, 3, 7, 7] | ~115M |
| `resnet_block` | [1, 256, 56, 56] | [256, 256, 3, 3] | ~3.5B |

#### MatMul Configurations

| Name | Dimensions | FLOPs |
|------|-----------|-------|
| `64x64` | 64 × 64 × 64 | 524K |
| `256x256` | 256 × 256 × 256 | 33.5M |
| `bert_hidden` | 768 × 768 × 768 | 906M |
| `gpt2_hidden` | 1024 × 1024 × 1024 | 2.1B |

#### Attention Configurations

| Name | Heads | Seq Len | Head Dim | FLOPs |
|------|-------|---------|----------|-------|
| `bert_base` | 12 | 512 | 64 | 402M |
| `bert_large` | 16 | 512 | 64 | 536M |
| `gpt2_small` | 12 | 1024 | 64 | 1.6B |
| `gpt2_medium` | 16 | 1024 | 64 | 2.1B |

## Benchmark Results

### Expected Performance Characteristics

Based on profiling (see `memory-analysis.md`):

| Model | Compilation Time |
|-------|------------------|
| MNIST (26 ops) | ~13ms |
| ResNet50 (122 ops) | ~263ms |
| ResNet50 + partitioning | ~168ms (36% faster) |

### ISA Optimization Impact

The benchmarks help measure the impact of ISA optimizations:

1. **LOOP Instructions**: O(1) instruction count regardless of tensor size
2. **PhiCoordinate Addressing**: Zero-copy transpose through virtual memory layout
3. **ClassMap Fusion**: Single-pass element-wise operation chains
4. **Im2col + GEMM**: Efficient convolution through matrix multiplication

### Throughput Metrics

Benchmarks report throughput in:
- **Bytes/sec**: For parsing and compilation (bytes of ONNX model)
- **Elements/sec**: For operation execution (tensor elements processed)
- **FLOPs/sec**: For compute-bound operations (Conv2D, MatMul)

## Test Models

### Required Models

Place test models in the following locations:

```
crates/hologram-onnx-core/tests/fixtures/
├── mnist-12.onnx          # MNIST digit recognition (required)
└── test-*.onnx            # Generated test models

models/
└── resnet50-v1-7.onnx     # ResNet50 (optional, for large model tests)
```

### Download MNIST Model

```bash
mkdir -p crates/hologram-onnx-core/tests/fixtures
curl -L https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-12.onnx \
  -o crates/hologram-onnx-core/tests/fixtures/mnist-12.onnx
```

### Download ResNet50 Model (Optional)

```bash
mkdir -p models
curl -L https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v1-7.onnx \
  -o models/resnet50-v1-7.onnx
```

## Analyzing Results

### Compare with Baseline

```bash
# Save baseline
cargo bench -- --save-baseline v1.0

# Compare against baseline
cargo bench -- --baseline v1.0
```

### Generate HTML Report

```bash
cargo bench -- --plotting-backend plotters
# Open target/criterion/report/index.html
```

### Profiling

For detailed profiling, use:

```bash
# With flamegraph
cargo flamegraph --bench compilation_bench -- --bench

# With perf (Linux)
perf record -g cargo bench --bench compilation_bench -- --profile-time 5
perf report
```

## Adding New Benchmarks

To add new benchmarks:

1. Add benchmark functions to appropriate file
2. Include in `criterion_group!` macro
3. Document configurations in this file
4. Run and verify with `cargo bench`

Example:

```rust
fn bench_new_operation(c: &mut Criterion) {
    let mut group = c.benchmark_group("new_operation");

    group.throughput(Throughput::Elements(1024));
    group.bench_function("example", |b| {
        b.iter(|| {
            // Benchmark code
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_new_operation,
    // ... other benchmarks
);
```
