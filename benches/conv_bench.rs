//! Benchmarks for Conv2D operations.
//!
//! This benchmark suite measures:
//! - Conv2D IR node creation time
//! - Conv2D → Im2Col+GEMM decomposition time
//! - Shape inference for various input sizes
//! - ResNet-style block compilation

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use hologram_compiler::ir::{DecomposeConfig, IRBuilder, ScalarType, Type, decompose_function};
use hologram_compiler::shapes::Shape;
use hologram_onnx_core::SymbolicShape;
use hologram_onnx_ops::{infer_conv_output_shape, translate_conv};
use hologram_onnx_spec::AttributeProto;
use hologram_onnx_spec::attribute_proto::AttributeType;
use std::collections::HashMap;

// =============================================================================
// Helper Functions
// =============================================================================

fn f32_tensor(dims: &[usize]) -> Type {
    Type::tensor(ScalarType::F32, Shape::concrete(dims.to_vec()))
}

fn make_ints_attr(name: &str, values: Vec<i64>) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        ints: values,
        r#type: AttributeType::Ints as i32,
        ..Default::default()
    }
}

// =============================================================================
// Conv2D IR Creation Benchmarks
// =============================================================================

fn bench_conv2d_ir_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv2d_ir_creation");

    // Different input sizes to benchmark
    let sizes = [
        ("32x32", [1, 3, 32, 32]),
        ("64x64", [1, 3, 64, 64]),
        ("112x112", [1, 64, 112, 112]),
        ("224x224", [1, 3, 224, 224]),
    ];

    for (name, input_size) in sizes {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("create_node", name),
            &input_size,
            |b, size| {
                b.iter(|| {
                    let mut builder = IRBuilder::new("bench");
                    let input = builder.add_input("X", f32_tensor(size));
                    let kernel = builder.add_input("W", f32_tensor(&[64, size[1], 3, 3]));
                    let result = builder.conv2d(
                        black_box(input),
                        black_box(kernel),
                        None,
                        (1, 1),
                        (1, 1),
                        (1, 1),
                        1,
                    );
                    builder.set_output(result);
                    builder.build()
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// Conv2D Decomposition Benchmarks
// =============================================================================

fn bench_conv2d_decomposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv2d_decomposition");

    let sizes = [
        ("32x32", [1, 3, 32, 32]),
        ("64x64", [1, 3, 64, 64]),
        ("112x112", [1, 64, 112, 112]),
        ("224x224", [1, 3, 224, 224]),
    ];

    let config = DecomposeConfig::all();

    for (name, input_size) in sizes {
        // Pre-create the function to benchmark only decomposition
        let mut builder = IRBuilder::new("bench");
        let input = builder.add_input("X", f32_tensor(&input_size));
        let kernel = builder.add_input("W", f32_tensor(&[64, input_size[1], 3, 3]));
        let result = builder.conv2d(input, kernel, None, (1, 1), (1, 1), (1, 1), 1);
        builder.set_output(result);
        let func = builder.build();

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::new("decompose", name), &func, |b, func| {
            b.iter(|| decompose_function(black_box(func), &config))
        });
    }

    group.finish();
}

// =============================================================================
// Shape Inference Benchmarks
// =============================================================================

fn bench_conv2d_shape_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv2d_shape_inference");

    let attrs = vec![
        make_ints_attr("strides", vec![2, 2]),
        make_ints_attr("pads", vec![3, 3, 3, 3]),
        make_ints_attr("dilations", vec![1, 1]),
    ];

    let sizes = [
        ("32x32", vec![1, 3, 32, 32], vec![64, 3, 7, 7]),
        ("64x64", vec![1, 3, 64, 64], vec![64, 3, 7, 7]),
        ("112x112", vec![1, 64, 112, 112], vec![128, 64, 3, 3]),
        ("224x224", vec![1, 3, 224, 224], vec![64, 3, 7, 7]),
    ];

    for (name, input_dims, kernel_dims) in sizes {
        let input_shape = SymbolicShape::concrete(input_dims);
        let kernel_shape = SymbolicShape::concrete(kernel_dims);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("infer", name),
            &(&input_shape, &kernel_shape, &attrs),
            |b, (input, kernel, attrs)| {
                b.iter(|| infer_conv_output_shape(black_box(input), black_box(kernel), attrs))
            },
        );
    }

    group.finish();
}

// =============================================================================
// ResNet Block Benchmarks
// =============================================================================

fn bench_resnet_block(c: &mut Criterion) {
    let mut group = c.benchmark_group("resnet_block");

    // Benchmark different block configurations
    let configs = [
        ("basic_block", 2), // 2 conv layers
        ("bottleneck", 3),  // 3 conv layers
        ("deep_block", 5),  // 5 conv layers
    ];

    for (name, num_convs) in configs {
        group.throughput(Throughput::Elements(num_convs as u64));
        group.bench_with_input(
            BenchmarkId::new("create", name),
            &num_convs,
            |b, &num_convs| {
                b.iter(|| {
                    let mut builder = IRBuilder::new("resnet_block");
                    let mut current = builder.add_input("X", f32_tensor(&[1, 64, 56, 56]));

                    for i in 0..num_convs {
                        let kernel =
                            builder.add_input(&format!("W{}", i), f32_tensor(&[64, 64, 3, 3]));
                        current = builder.conv2d(current, kernel, None, (1, 1), (1, 1), (1, 1), 1);
                    }

                    builder.set_output(current);
                    builder.build()
                })
            },
        );
    }

    // Benchmark decomposition of ResNet blocks
    let config = DecomposeConfig::all();

    for (name, num_convs) in configs {
        // Pre-create the function
        let mut builder = IRBuilder::new("resnet_block");
        let mut current = builder.add_input("X", f32_tensor(&[1, 64, 56, 56]));

        for i in 0..num_convs {
            let kernel = builder.add_input(&format!("W{}", i), f32_tensor(&[64, 64, 3, 3]));
            current = builder.conv2d(current, kernel, None, (1, 1), (1, 1), (1, 1), 1);
        }

        builder.set_output(current);
        let func = builder.build();

        group.bench_with_input(BenchmarkId::new("decompose", name), &func, |b, func| {
            b.iter(|| decompose_function(black_box(func), &config))
        });
    }

    group.finish();
}

// =============================================================================
// Large Model Benchmarks
// =============================================================================

fn bench_large_conv_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_conv_chain");
    group.sample_size(10); // Reduce sample size for expensive benchmarks

    let chain_lengths = [5, 10, 20, 50];

    for num_convs in chain_lengths {
        group.throughput(Throughput::Elements(num_convs as u64));

        // Benchmark IR creation
        group.bench_with_input(
            BenchmarkId::new("create", num_convs),
            &num_convs,
            |b, &num_convs| {
                b.iter(|| {
                    let mut builder = IRBuilder::new("large_chain");
                    let mut current = builder.add_input("X", f32_tensor(&[1, 64, 56, 56]));

                    for i in 0..num_convs {
                        let kernel =
                            builder.add_input(&format!("W{}", i), f32_tensor(&[64, 64, 3, 3]));
                        current = builder.conv2d(current, kernel, None, (1, 1), (1, 1), (1, 1), 1);
                    }

                    builder.set_output(current);
                    builder.build()
                })
            },
        );
    }

    // Benchmark decomposition for large chains
    let config = DecomposeConfig::all();

    for num_convs in chain_lengths {
        // Pre-create the function
        let mut builder = IRBuilder::new("large_chain");
        let mut current = builder.add_input("X", f32_tensor(&[1, 64, 56, 56]));

        for i in 0..num_convs {
            let kernel = builder.add_input(&format!("W{}", i), f32_tensor(&[64, 64, 3, 3]));
            current = builder.conv2d(current, kernel, None, (1, 1), (1, 1), (1, 1), 1);
        }

        builder.set_output(current);
        let func = builder.build();

        group.bench_with_input(
            BenchmarkId::new("decompose", num_convs),
            &func,
            |b, func| b.iter(|| decompose_function(black_box(func), &config)),
        );
    }

    group.finish();
}

// =============================================================================
// ONNX Translation Benchmarks
// =============================================================================

fn bench_onnx_translation(c: &mut Criterion) {
    let mut group = c.benchmark_group("onnx_translation");

    let attrs = vec![
        make_ints_attr("strides", vec![1, 1]),
        make_ints_attr("pads", vec![1, 1, 1, 1]),
        make_ints_attr("dilations", vec![1, 1]),
    ];

    let sizes = [
        ("small", [1, 3, 32, 32]),
        ("medium", [1, 64, 56, 56]),
        ("large", [1, 3, 224, 224]),
    ];

    for (name, input_size) in sizes {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("translate_conv", name),
            &(input_size, &attrs),
            |b, (size, attrs)| {
                b.iter(|| {
                    let mut builder = IRBuilder::new("bench");
                    let input = builder.add_input("X", f32_tensor(size));
                    let kernel = builder.add_input("W", f32_tensor(&[64, size[1], 3, 3]));
                    translate_conv(&[input, kernel], attrs, &HashMap::new(), &mut builder)
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// Criterion Configuration
// =============================================================================

criterion_group!(
    benches,
    bench_conv2d_ir_creation,
    bench_conv2d_decomposition,
    bench_conv2d_shape_inference,
    bench_resnet_block,
    bench_large_conv_chain,
    bench_onnx_translation,
);

criterion_main!(benches);
