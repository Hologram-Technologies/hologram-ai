//! Benchmarks for SIMD activation hint integration
//!
//! This benchmark suite measures:
//! 1. IR graph construction with SIMD hints
//! 2. Compilation performance (IR → CompileGraph)
//! 3. End-to-end ONNX → IR → CompileGraph pipeline
//!
//! Run with: cargo bench --bench simd_activation_bench

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use hologram::ir::{DType, GraphBuilder, Shape};
use hologram_ai_onnx::core::op_hints::{ActivationType, add_simd_hint};
use hologram_compiler::from_ir::convert_from_ir;

/// Helper to create an IR graph with activation operations
fn create_activation_graph(
    batch_size: usize,
    hidden_size: usize,
    activation: ActivationType,
    with_hint: bool,
) -> hologram::ir::OperationGraph {
    let mut builder = GraphBuilder::new();
    let input = builder.input(
        "input",
        Shape::static_shape(&[batch_size, hidden_size]),
        DType::F32,
    );

    let activation_node = match activation {
        ActivationType::Sigmoid => builder.sigmoid(input).expect("Failed to create sigmoid"),
        ActivationType::Tanh => builder.tanh(input).expect("Failed to create tanh"),
        ActivationType::Relu => builder.relu(input).expect("Failed to create relu"),
        ActivationType::Gelu => builder.gelu(input).expect("Failed to create gelu"),
        ActivationType::Silu => {
            // SiLU = x * sigmoid(x)
            let sig = builder.sigmoid(input).expect("Failed to create sigmoid");
            if with_hint {
                add_simd_hint(builder.graph_mut(), sig, ActivationType::Sigmoid);
            }
            builder.mul(input, sig).expect("Failed to create mul")
        }
    };

    if with_hint && activation != ActivationType::Silu {
        add_simd_hint(builder.graph_mut(), activation_node, activation);
    }

    let _output = builder
        .output("output", activation_node)
        .expect("Failed to create output");

    builder.build()
}

/// Benchmark IR graph construction with SIMD hints
fn bench_ir_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("ir_construction");

    for &batch_size in &[1, 16, 128] {
        for &hidden_size in &[256, 768, 2048] {
            let param = format!("{}x{}", batch_size, hidden_size);

            group.bench_with_input(
                BenchmarkId::new("sigmoid_with_hint", &param),
                &(batch_size, hidden_size),
                |b, &(bs, hs)| {
                    b.iter(|| {
                        black_box(create_activation_graph(
                            bs,
                            hs,
                            ActivationType::Sigmoid,
                            true,
                        ))
                    })
                },
            );

            group.bench_with_input(
                BenchmarkId::new("sigmoid_no_hint", &param),
                &(batch_size, hidden_size),
                |b, &(bs, hs)| {
                    b.iter(|| {
                        black_box(create_activation_graph(
                            bs,
                            hs,
                            ActivationType::Sigmoid,
                            false,
                        ))
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark compilation from IR to CompileGraph
fn bench_ir_to_compile_graph(c: &mut Criterion) {
    let mut group = c.benchmark_group("ir_to_compile_graph");

    for &batch_size in &[1, 16, 128] {
        for &hidden_size in &[256, 768, 2048] {
            let elements = batch_size * hidden_size;
            group.throughput(Throughput::Elements(elements as u64));

            let param = format!("{}x{}", batch_size, hidden_size);
            let ir_with_hint =
                create_activation_graph(batch_size, hidden_size, ActivationType::Sigmoid, true);
            let ir_no_hint =
                create_activation_graph(batch_size, hidden_size, ActivationType::Sigmoid, false);

            group.bench_with_input(
                BenchmarkId::new("sigmoid_with_hint", &param),
                &ir_with_hint,
                |b, ir| b.iter(|| black_box(convert_from_ir(ir).expect("Failed to compile IR"))),
            );

            group.bench_with_input(
                BenchmarkId::new("sigmoid_no_hint", &param),
                &ir_no_hint,
                |b, ir| b.iter(|| black_box(convert_from_ir(ir).expect("Failed to compile IR"))),
            );
        }
    }

    group.finish();
}

/// Benchmark multiple activation types with hints
fn bench_activation_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("activation_types");

    let batch_size = 16;
    let hidden_size = 768;

    for activation in [
        ActivationType::Sigmoid,
        ActivationType::Tanh,
        ActivationType::Relu,
        ActivationType::Gelu,
    ] {
        let name = activation.name();
        let ir_graph = create_activation_graph(batch_size, hidden_size, activation, true);

        group.bench_with_input(BenchmarkId::from_parameter(name), &ir_graph, |b, ir| {
            b.iter(|| black_box(convert_from_ir(ir).expect("Failed to compile IR")))
        });
    }

    group.finish();
}

/// Benchmark a chain of multiple activations (simulating transformer FFN)
fn bench_activation_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("activation_chain");

    let batch_size = 16;
    let hidden_size = 768;

    // Create graph with multiple chained activations (like FFN)
    let mut builder = GraphBuilder::new();
    let input = builder.input(
        "input",
        Shape::static_shape(&[batch_size, hidden_size]),
        DType::F32,
    );

    // Simulate FFN: input → GELU → ReLU → Tanh
    let gelu = builder.gelu(input).expect("Failed to create gelu");
    add_simd_hint(builder.graph_mut(), gelu, ActivationType::Gelu);

    let relu = builder.relu(gelu).expect("Failed to create relu");
    add_simd_hint(builder.graph_mut(), relu, ActivationType::Relu);

    let tanh = builder.tanh(relu).expect("Failed to create tanh");
    add_simd_hint(builder.graph_mut(), tanh, ActivationType::Tanh);

    let _output = builder
        .output("output", tanh)
        .expect("Failed to create output");
    let ir_graph = builder.build();

    group.bench_function("ffn_activation_chain", |b| {
        b.iter(|| black_box(convert_from_ir(&ir_graph).expect("Failed to compile IR")))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_ir_construction,
    bench_ir_to_compile_graph,
    bench_activation_types,
    bench_activation_chain
);
criterion_main!(benches);
