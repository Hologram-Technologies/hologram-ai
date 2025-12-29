//! Benchmarks for ONNX compilation pipeline stages.
//!
//! This benchmark suite measures:
//! - ONNX protobuf parsing
//! - Decomposition pass
//! - Full compilation pipeline

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use prost::Message;
use std::path::Path;

// =============================================================================
// ONNX Parsing Benchmarks
// =============================================================================

fn bench_onnx_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("onnx_parsing");

    // Test with MNIST model if available
    let mnist_path = Path::new("crates/hologram-onnx-core/tests/fixtures/mnist-12.onnx");

    if mnist_path.exists() {
        let model_bytes = std::fs::read(mnist_path).expect("Failed to read MNIST model");
        group.throughput(Throughput::Bytes(model_bytes.len() as u64));

        group.bench_function("mnist_parse", |b| {
            b.iter(|| {
                hologram_onnx_spec::ModelProto::decode(black_box(model_bytes.as_slice()))
                    .expect("Failed to parse MNIST model")
            });
        });
    }

    // Test with ResNet if available
    let resnet_path = Path::new("models/resnet50-v1-7.onnx");

    if resnet_path.exists() {
        let model_bytes = std::fs::read(resnet_path).expect("Failed to read ResNet model");
        group.throughput(Throughput::Bytes(model_bytes.len() as u64));

        group.bench_function("resnet50_parse", |b| {
            b.iter(|| {
                hologram_onnx_spec::ModelProto::decode(black_box(model_bytes.as_slice()))
                    .expect("Failed to parse ResNet model")
            });
        });
    }

    group.finish();
}

// =============================================================================
// Decomposition Pass Benchmarks
// =============================================================================

fn bench_decomposition(c: &mut Criterion) {
    use hologram_compiler::ir::{DecomposeConfig, IRBuilder, ScalarType, Type, decompose_function};
    use hologram_compiler::shapes::Shape;

    let mut group = c.benchmark_group("decomposition");

    // Different sizes to benchmark
    let sizes: [(_, [usize; 4], [usize; 4]); 4] = [
        ("small", [1, 3, 32, 32], [16, 3, 3, 3]),
        ("medium", [1, 64, 112, 112], [64, 64, 3, 3]),
        ("large", [1, 256, 56, 56], [256, 256, 3, 3]),
        ("resnet_block", [1, 512, 28, 28], [512, 512, 3, 3]),
    ];

    for (name, input_size, kernel_size) in sizes {
        group.throughput(Throughput::Elements(1));

        group.bench_with_input(
            BenchmarkId::new("conv2d_decompose", name),
            &(input_size, kernel_size),
            |b, (input, kernel)| {
                // Create a function with a single Conv2D operation
                let mut builder = IRBuilder::new("bench");
                let input_ty = Type::tensor(ScalarType::F32, Shape::concrete(input.to_vec()));
                let kernel_ty = Type::tensor(ScalarType::F32, Shape::concrete(kernel.to_vec()));

                let x = builder.add_input("X", input_ty);
                let w = builder.add_input("W", kernel_ty);
                let result = builder.conv2d(x, w, None, (1, 1), (1, 1), (1, 1), 1);
                builder.set_output(result);
                let func = builder.build();

                b.iter(|| decompose_function(black_box(&func), &DecomposeConfig::default()));
            },
        );
    }

    group.finish();
}

// =============================================================================
// Full Compilation Pipeline Benchmarks
// =============================================================================

fn bench_full_compilation(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_compilation");

    // Test with MNIST model if available
    let mnist_path = Path::new("crates/hologram-onnx-core/tests/fixtures/mnist-12.onnx");

    if mnist_path.exists() {
        let model_bytes = std::fs::read(mnist_path).expect("Failed to read MNIST model");
        group.throughput(Throughput::Bytes(model_bytes.len() as u64));

        group.bench_function("mnist_full_compile", |b| {
            b.iter(|| {
                let compiler = hologram_onnx::OnnxCompiler::new();
                compiler
                    .compile(black_box(model_bytes.as_slice()))
                    .expect("Failed to compile MNIST model")
            });
        });
    }

    // Test with ResNet if available
    let resnet_path = Path::new("models/resnet50-v1-7.onnx");

    if resnet_path.exists() {
        let model_bytes = std::fs::read(resnet_path).expect("Failed to read ResNet model");
        group.throughput(Throughput::Bytes(model_bytes.len() as u64));

        group.bench_function("resnet50_full_compile", |b| {
            b.iter(|| {
                let compiler = hologram_onnx::OnnxCompiler::new();
                compiler
                    .compile(black_box(model_bytes.as_slice()))
                    .expect("Failed to compile ResNet model")
            });
        });
    }

    group.finish();
}

// =============================================================================
// Partitioned Compilation Benchmarks
// =============================================================================

fn bench_partitioned_compilation(c: &mut Criterion) {
    let mut group = c.benchmark_group("partitioned_compilation");

    // Test with ResNet using partitioning if available
    let resnet_path = Path::new("models/resnet50-v1-7.onnx");

    if resnet_path.exists() {
        let model_bytes = std::fs::read(resnet_path).expect("Failed to read ResNet model");

        let partition_sizes = [50, 100, 200];

        for size in partition_sizes {
            group.throughput(Throughput::Bytes(model_bytes.len() as u64));

            group.bench_with_input(
                BenchmarkId::new("resnet50_partitioned", size),
                &size,
                |b, &partition_size| {
                    b.iter(|| {
                        let config = hologram_onnx_core::OnnxConfig {
                            enable_partitioning: true,
                            partition_size,
                            ..Default::default()
                        };
                        let compiler = hologram_onnx::OnnxCompiler::with_config(config);
                        compiler
                            .compile(black_box(model_bytes.as_slice()))
                            .expect("Failed to compile with partitioning")
                    });
                },
            );
        }
    }

    group.finish();
}

// =============================================================================
// Graph Size Scaling Benchmarks
// =============================================================================

fn bench_graph_size_scaling(c: &mut Criterion) {
    use hologram_compiler::ir::{IRBuilder, ScalarType, Type};
    use hologram_compiler::shapes::Shape;

    let mut group = c.benchmark_group("graph_size_scaling");

    // Test how IR creation scales with graph size
    let sizes = [10, 50, 100, 200, 500];

    for num_ops in sizes {
        group.throughput(Throughput::Elements(num_ops as u64));

        group.bench_with_input(
            BenchmarkId::new("ir_creation", num_ops),
            &num_ops,
            |b, &num_ops| {
                b.iter(|| {
                    // Create a chain of Add operations
                    let mut builder = IRBuilder::new("bench");
                    let ty = Type::tensor(ScalarType::F32, Shape::concrete(vec![1, 64, 32, 32]));

                    let mut current = builder.add_input("X", ty.clone());
                    let constant = builder.add_input("C", ty);

                    for _ in 0..num_ops {
                        current = builder.add(current, constant);
                    }

                    builder.set_output(current);
                    black_box(builder.build())
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Benchmark Groups
// =============================================================================

criterion_group!(
    compilation_benches,
    bench_onnx_parsing,
    bench_decomposition,
    bench_full_compilation,
    bench_partitioned_compilation,
    bench_graph_size_scaling,
);

criterion_main!(compilation_benches);
