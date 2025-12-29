//! Benchmarks for shape inference operations.
//!
//! This benchmark suite measures:
//! - Symbolic shape creation
//! - Shape inference for various operations
//! - Broadcasting shape calculations
//! - MatMul shape inference

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use hologram_onnx_core::{Dim, SymbolicShape};
use hologram_onnx_ops::{infer_conv_output_shape, infer_pool_output_shape};
use hologram_onnx_spec::AttributeProto;
use hologram_onnx_spec::attribute_proto::AttributeType;

// =============================================================================
// Helper Functions
// =============================================================================

fn make_ints_attr(name: &str, values: Vec<i64>) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        ints: values,
        r#type: AttributeType::Ints as i32,
        ..Default::default()
    }
}

// =============================================================================
// Shape Creation Benchmarks
// =============================================================================

fn bench_shape_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("shape_creation");

    let ranks = [2, 4, 6, 8];

    for rank in ranks {
        let dims: Vec<usize> = (0..rank).map(|i| (i + 1) * 64).collect();

        group.throughput(Throughput::Elements(rank as u64));
        group.bench_with_input(BenchmarkId::new("concrete", rank), &dims, |b, dims| {
            b.iter(|| SymbolicShape::concrete(black_box(dims.clone())))
        });
    }

    // Symbolic shape creation
    for rank in ranks {
        let names: Vec<&str> = match rank {
            2 => vec!["batch", "features"],
            4 => vec!["batch", "channels", "height", "width"],
            6 => vec!["batch", "seq", "heads", "dim", "h", "w"],
            8 => vec!["a", "b", "c", "d", "e", "f", "g", "h"],
            _ => vec![],
        };

        group.bench_with_input(BenchmarkId::new("symbolic", rank), &names, |b, names| {
            b.iter(|| SymbolicShape::symbolic(black_box(names.clone())))
        });
    }

    group.finish();
}

// =============================================================================
// Binary Operation Shape Inference
// =============================================================================

fn bench_binary_op_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("binary_op_inference");

    // Same shape (no broadcasting)
    let same_shapes = [
        ("2D", vec![32, 64], vec![32, 64]),
        ("4D", vec![1, 64, 112, 112], vec![1, 64, 112, 112]),
    ];

    for (name, dims1, dims2) in same_shapes {
        let shape1 = SymbolicShape::concrete(dims1);
        let shape2 = SymbolicShape::concrete(dims2);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("same_shape", name),
            &(&shape1, &shape2),
            |b, (s1, s2)| b.iter(|| s1.infer_binary_op(black_box(s2))),
        );
    }

    // Broadcasting shapes
    let broadcast_shapes = [
        ("scalar", vec![1], vec![32, 64]),
        ("channel", vec![1, 64, 1, 1], vec![1, 64, 112, 112]),
        ("batch", vec![1, 64, 112, 112], vec![8, 64, 112, 112]),
    ];

    for (name, dims1, dims2) in broadcast_shapes {
        let shape1 = SymbolicShape::concrete(dims1);
        let shape2 = SymbolicShape::concrete(dims2);

        group.bench_with_input(
            BenchmarkId::new("broadcast", name),
            &(&shape1, &shape2),
            |b, (s1, s2)| b.iter(|| s1.infer_binary_op(black_box(s2))),
        );
    }

    group.finish();
}

// =============================================================================
// MatMul Shape Inference
// =============================================================================

fn bench_matmul_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_inference");

    let matmul_shapes = [
        ("small", vec![32, 64], vec![64, 128]),
        ("medium", vec![128, 512], vec![512, 256]),
        ("large", vec![1024, 1024], vec![1024, 1024]),
        ("batched", vec![8, 32, 64], vec![8, 64, 128]),
    ];

    for (name, dims1, dims2) in matmul_shapes {
        let shape1 = SymbolicShape::concrete(dims1);
        let shape2 = SymbolicShape::concrete(dims2);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("infer", name),
            &(&shape1, &shape2),
            |b, (s1, s2)| b.iter(|| s1.infer_matmul(black_box(s2))),
        );
    }

    // Symbolic batch dimension
    let symbolic_shapes = [
        ("symbolic_batch_2d", vec!["batch", "64"], vec!["64", "128"]),
        (
            "symbolic_batch_3d",
            vec!["batch", "seq", "64"],
            vec!["batch", "64", "128"],
        ),
    ];

    for (name, names1, names2) in symbolic_shapes {
        let shape1 = SymbolicShape::symbolic(names1);
        let shape2 = SymbolicShape::symbolic(names2);

        group.bench_with_input(
            BenchmarkId::new("symbolic", name),
            &(&shape1, &shape2),
            |b, (s1, s2)| b.iter(|| s1.infer_matmul(black_box(s2))),
        );
    }

    group.finish();
}

// =============================================================================
// Transpose Shape Inference
// =============================================================================

fn bench_transpose_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("transpose_inference");

    let shapes = [
        ("2D", vec![32, 64]),
        ("4D_NCHW_to_NHWC", vec![1, 64, 112, 112]),
        ("5D", vec![2, 3, 4, 5, 6]),
    ];

    let perms: [Option<&[i64]>; 3] = [
        None,                   // Default reverse
        Some(&[0, 2, 3, 1]),    // NCHW to NHWC
        Some(&[4, 3, 2, 1, 0]), // Full reverse
    ];

    for ((name, dims), perm) in shapes.iter().zip(perms.iter()) {
        let shape = SymbolicShape::concrete(dims.clone());
        let perm_owned = *perm;

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::new("infer", name), &shape, |b, shape| {
            b.iter(|| shape.infer_transpose(black_box(perm_owned)))
        });
    }

    group.finish();
}

// =============================================================================
// Reshape Shape Inference
// =============================================================================

fn bench_reshape_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("reshape_inference");

    let reshape_cases = [
        (
            "flatten",
            vec![1, 64, 7, 7],
            vec![Dim::Concrete(1), Dim::Concrete(3136)],
        ),
        (
            "expand",
            vec![1, 3136],
            vec![
                Dim::Concrete(1),
                Dim::Concrete(64),
                Dim::Concrete(7),
                Dim::Concrete(7),
            ],
        ),
        (
            "infer_dim",
            vec![2, 3, 4],
            vec![Dim::Concrete(6), Dim::Concrete(0)],
        ), // 0 means infer
    ];

    for (name, input_dims, target_dims) in reshape_cases {
        let input_shape = SymbolicShape::concrete(input_dims);
        let target_shape = SymbolicShape::new(target_dims);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("infer", name),
            &(&input_shape, &target_shape),
            |b, (input, target)| b.iter(|| input.infer_reshape(black_box(target.dims()))),
        );
    }

    group.finish();
}

// =============================================================================
// Conv Shape Inference
// =============================================================================

fn bench_conv_shape_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv_shape_inference");

    let conv_attrs = vec![
        make_ints_attr("strides", vec![2, 2]),
        make_ints_attr("pads", vec![3, 3, 3, 3]),
        make_ints_attr("dilations", vec![1, 1]),
    ];

    let conv_cases = [
        ("resnet_first", vec![1, 3, 224, 224], vec![64, 3, 7, 7]),
        ("resnet_block", vec![1, 64, 56, 56], vec![64, 64, 3, 3]),
        ("large_batch", vec![32, 64, 56, 56], vec![128, 64, 3, 3]),
    ];

    for (name, input_dims, kernel_dims) in conv_cases {
        let input_shape = SymbolicShape::concrete(input_dims);
        let kernel_shape = SymbolicShape::concrete(kernel_dims);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("infer", name),
            &(&input_shape, &kernel_shape, &conv_attrs),
            |b, (input, kernel, attrs)| {
                b.iter(|| infer_conv_output_shape(black_box(input), black_box(kernel), attrs))
            },
        );
    }

    // Symbolic batch
    let symbolic_input = SymbolicShape::symbolic(vec!["batch", "3", "224", "224"]);
    let kernel_shape = SymbolicShape::concrete(vec![64, 3, 7, 7]);

    group.bench_with_input(
        BenchmarkId::new("infer", "symbolic_batch"),
        &(&symbolic_input, &kernel_shape, &conv_attrs),
        |b, (input, kernel, attrs)| {
            b.iter(|| infer_conv_output_shape(black_box(input), black_box(kernel), attrs))
        },
    );

    group.finish();
}

// =============================================================================
// Pool Shape Inference
// =============================================================================

fn bench_pool_shape_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool_shape_inference");

    let pool_cases = [
        (
            "maxpool_3x3",
            vec![1, 64, 112, 112],
            vec![
                make_ints_attr("kernel_shape", vec![3, 3]),
                make_ints_attr("strides", vec![2, 2]),
                make_ints_attr("pads", vec![1, 1, 1, 1]),
            ],
        ),
        (
            "avgpool_2x2",
            vec![1, 256, 28, 28],
            vec![
                make_ints_attr("kernel_shape", vec![2, 2]),
                make_ints_attr("strides", vec![2, 2]),
                make_ints_attr("pads", vec![0, 0, 0, 0]),
            ],
        ),
    ];

    for (name, input_dims, attrs) in pool_cases {
        let input_shape = SymbolicShape::concrete(input_dims);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("infer", name),
            &(&input_shape, &attrs),
            |b, (input, attrs)| b.iter(|| infer_pool_output_shape(black_box(input), attrs)),
        );
    }

    group.finish();
}

// =============================================================================
// Shape Comparison Benchmarks
// =============================================================================

fn bench_shape_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("shape_comparison");

    let shapes = [
        ("2D_concrete", SymbolicShape::concrete(vec![32, 64])),
        (
            "4D_concrete",
            SymbolicShape::concrete(vec![1, 64, 112, 112]),
        ),
        (
            "4D_symbolic",
            SymbolicShape::symbolic(vec!["batch", "64", "112", "112"]),
        ),
    ];

    for (name, shape) in &shapes {
        group.throughput(Throughput::Elements(1));

        group.bench_with_input(
            BenchmarkId::new("is_fully_concrete", name),
            shape,
            |b, shape| b.iter(|| shape.is_fully_concrete()),
        );

        group.bench_with_input(
            BenchmarkId::new("is_partially_symbolic", name),
            shape,
            |b, shape| b.iter(|| shape.is_partially_symbolic()),
        );

        group.bench_with_input(BenchmarkId::new("rank", name), shape, |b, shape| {
            b.iter(|| shape.rank())
        });
    }

    group.finish();
}

// =============================================================================
// Criterion Configuration
// =============================================================================

criterion_group!(
    benches,
    bench_shape_creation,
    bench_binary_op_inference,
    bench_matmul_inference,
    bench_transpose_inference,
    bench_reshape_inference,
    bench_conv_shape_inference,
    bench_pool_shape_inference,
    bench_shape_comparison,
);

criterion_main!(benches);
