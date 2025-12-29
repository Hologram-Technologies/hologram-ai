//! Benchmarks for operation execution performance.
//!
//! This benchmark suite measures:
//! - Conv2D IR creation (verifies Im2col+GEMM+SIMD optimization)
//! - MatMul IR creation
//! - Attention mechanism IR creation
//! - Element-wise operation IR creation

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use hologram_compiler::ir::{IRBuilder, ScalarType, Type};
use hologram_compiler::shapes::Shape;

// =============================================================================
// Helper Functions
// =============================================================================

fn f32_tensor(dims: &[usize]) -> Type {
    Type::tensor(ScalarType::F32, Shape::concrete(dims.to_vec()))
}

// =============================================================================
// Conv2D Execution Benchmarks
// =============================================================================

fn bench_conv2d_execution(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv2d_execution");

    let configs = [
        ("small_3x3", [1, 3, 32, 32], [16, 3, 3, 3]),
        ("medium_3x3", [1, 64, 112, 112], [64, 64, 3, 3]),
        ("resnet_first", [1, 3, 224, 224], [64, 3, 7, 7]),
        ("resnet_block", [1, 256, 56, 56], [256, 256, 3, 3]),
    ];

    for (name, input_size, kernel_size) in configs {
        // Calculate total FLOPs for throughput measurement
        let output_h = input_size[2] - kernel_size[2] + 1;
        let output_w = input_size[3] - kernel_size[3] + 1;
        let flops = 2
            * output_h
            * output_w
            * kernel_size[0]
            * kernel_size[1]
            * kernel_size[2]
            * kernel_size[3];

        group.throughput(Throughput::Elements(flops as u64));

        group.bench_with_input(
            BenchmarkId::new("conv2d_ir", name),
            &(input_size, kernel_size),
            |b, (input, kernel)| {
                b.iter(|| {
                    // Create IR for Conv2D
                    let mut builder = IRBuilder::new("bench");
                    let x = builder.add_input("X", f32_tensor(input));
                    let w = builder.add_input("W", f32_tensor(kernel));
                    let result =
                        builder.conv2d(black_box(x), black_box(w), None, (1, 1), (0, 0), (1, 1), 1);
                    builder.set_output(result);
                    black_box(builder.build())
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// MatMul Execution Benchmarks
// =============================================================================

fn bench_matmul_execution(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_execution");

    let sizes = [
        ("64x64", 64, 64, 64),
        ("128x128", 128, 128, 128),
        ("256x256", 256, 256, 256),
        ("512x512", 512, 512, 512),
        ("bert_hidden", 768, 768, 768),
        ("gpt2_hidden", 1024, 1024, 1024),
    ];

    for (name, m, n, k) in sizes {
        // FLOPs for matrix multiplication: 2*M*N*K
        let flops = 2 * m * n * k;
        group.throughput(Throughput::Elements(flops as u64));

        group.bench_with_input(
            BenchmarkId::new("matmul_ir", name),
            &(m, n, k),
            |b, &(m, n, k)| {
                b.iter(|| {
                    let mut builder = IRBuilder::new("bench");
                    let a = builder.add_input("A", f32_tensor(&[m, k]));
                    let b_val = builder.add_input("B", f32_tensor(&[k, n]));
                    let result = builder.matmul(black_box(a), black_box(b_val));
                    builder.set_output(result);
                    black_box(builder.build())
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Batched MatMul Benchmarks (for attention)
// =============================================================================

fn bench_batched_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("batched_matmul");

    let configs = [
        ("bert_base", 12, 512, 64),    // 12 heads, 512 seq_len, 64 head_dim
        ("bert_large", 16, 512, 64),   // 16 heads
        ("gpt2_small", 12, 1024, 64),  // GPT-2 small
        ("gpt2_medium", 16, 1024, 64), // GPT-2 medium
    ];

    for (name, num_heads, seq_len, head_dim) in configs {
        // FLOPs for batched Q*K^T: batch * 2 * seq_len * seq_len * head_dim
        let flops = num_heads * 2 * seq_len * seq_len * head_dim;
        group.throughput(Throughput::Elements(flops as u64));

        group.bench_with_input(
            BenchmarkId::new("attention_qk", name),
            &(num_heads, seq_len, head_dim),
            |b, &(num_heads, seq_len, head_dim)| {
                b.iter(|| {
                    let mut builder = IRBuilder::new("bench");

                    // Q and K tensors: [batch, num_heads, seq_len, head_dim]
                    let q = builder.add_input("Q", f32_tensor(&[1, num_heads, seq_len, head_dim]));
                    let k = builder.add_input("K", f32_tensor(&[1, num_heads, seq_len, head_dim]));

                    // Transpose K for Q*K^T
                    let k_t = builder.transpose(black_box(k), Some(vec![0, 1, 3, 2]));
                    let result = builder.matmul(black_box(q), black_box(k_t));
                    builder.set_output(result);
                    black_box(builder.build())
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Element-wise Operation Benchmarks (ClassMap fusion)
// =============================================================================

fn bench_elementwise_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise_ops");

    let sizes = [
        ("small", 1024),
        ("medium", 65536),
        ("large", 1048576),
        ("very_large", 16777216),
    ];

    for (name, elements) in sizes {
        group.throughput(Throughput::Elements(elements as u64));

        // Single operation
        group.bench_with_input(BenchmarkId::new("relu", name), &elements, |b, &elements| {
            b.iter(|| {
                let mut builder = IRBuilder::new("bench");
                let x = builder.add_input("X", f32_tensor(&[1, elements]));
                let result = builder.relu(black_box(x));
                builder.set_output(result);
                black_box(builder.build())
            });
        });

        // Fused operations (should use ClassMap)
        group.bench_with_input(
            BenchmarkId::new("fused_relu_add", name),
            &elements,
            |b, &elements| {
                b.iter(|| {
                    let mut builder = IRBuilder::new("bench");
                    let x = builder.add_input("X", f32_tensor(&[1, elements]));
                    let bias = builder.add_input("B", f32_tensor(&[1, elements]));
                    let added = builder.add(black_box(x), black_box(bias));
                    let result = builder.relu(black_box(added));
                    builder.set_output(result);
                    black_box(builder.build())
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// LOOP Instruction Benchmarks
// =============================================================================

fn bench_loop_instructions(c: &mut Criterion) {
    let mut group = c.benchmark_group("loop_instructions");

    // Different iteration counts to test O(1) complexity claim
    let iterations = [
        ("1K", 1000),
        ("10K", 10000),
        ("100K", 100000),
        ("1M", 1000000),
    ];

    for (name, count) in iterations {
        group.throughput(Throughput::Elements(count as u64));

        group.bench_with_input(BenchmarkId::new("add_chain", name), &count, |b, &count| {
            b.iter(|| {
                let mut builder = IRBuilder::new("bench");
                let x = builder.add_input("X", f32_tensor(&[count]));
                let y = builder.add_input("Y", f32_tensor(&[count]));
                let result = builder.add(black_box(x), black_box(y));
                builder.set_output(result);
                black_box(builder.build())
            });
        });
    }

    group.finish();
}

// =============================================================================
// Softmax Benchmarks
// =============================================================================

fn bench_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax");

    let configs = [
        ("small_vocab", 1, 100, 1000),  // Small vocabulary
        ("bert_vocab", 1, 512, 30522),  // BERT vocabulary size
        ("gpt2_vocab", 1, 1024, 50257), // GPT-2 vocabulary size
        ("attention", 1, 512, 512),     // Attention scores
    ];

    for (name, batch, seq_len, vocab_size) in configs {
        // Elements to process
        let elements = batch * seq_len * vocab_size;
        group.throughput(Throughput::Elements(elements as u64));

        group.bench_with_input(
            BenchmarkId::new("softmax_ir", name),
            &(batch, seq_len, vocab_size),
            |b, &(batch, seq_len, vocab_size)| {
                b.iter(|| {
                    let mut builder = IRBuilder::new("bench");
                    let x = builder.add_input("X", f32_tensor(&[batch, seq_len, vocab_size]));
                    let result = builder.softmax(black_box(x), -1);
                    builder.set_output(result);
                    black_box(builder.build())
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Transpose Benchmarks (PhiCoordinate addressing)
// =============================================================================

fn bench_transpose(c: &mut Criterion) {
    let mut group = c.benchmark_group("transpose");

    let configs = [
        ("2d_small", vec![64, 64], vec![1, 0]),
        ("2d_large", vec![1024, 1024], vec![1, 0]),
        ("4d_nchw_nhwc", vec![1, 64, 56, 56], vec![0, 2, 3, 1]),
        ("attention_reshape", vec![1, 12, 512, 64], vec![0, 2, 1, 3]),
    ];

    for (name, dims, perm) in configs {
        let elements: usize = dims.iter().product();
        group.throughput(Throughput::Elements(elements as u64));

        group.bench_with_input(
            BenchmarkId::new("transpose_ir", name),
            &(dims.clone(), perm.clone()),
            |b, (dims, perm)| {
                b.iter(|| {
                    let mut builder = IRBuilder::new("bench");
                    let x = builder.add_input("X", f32_tensor(dims));
                    let result = builder.transpose(black_box(x), Some(perm.clone()));
                    builder.set_output(result);
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
    execution_benches,
    bench_conv2d_execution,
    bench_matmul_execution,
    bench_batched_matmul,
    bench_elementwise_ops,
    bench_loop_instructions,
    bench_softmax,
    bench_transpose,
);

criterion_main!(execution_benches);
