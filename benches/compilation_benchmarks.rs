//! Compilation performance benchmarks.
//!
//! These benchmarks measure the performance of the ONNX→IR translation pipeline.
//!
//! Run with: `cargo bench`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use hologram_onnx::{compile_onnx, translate_graph_to_ir, proto::*};
use prost::Message;

/// Helper to create ONNX ValueInfoProto.
fn make_value_info(name: &str, dims: Vec<i64>, dtype: i32) -> ValueInfoProto {
    use tensor_shape_proto::Dimension;
    use type_proto::{Tensor, Value};

    ValueInfoProto {
        name: name.to_string(),
        r#type: Some(TypeProto {
            value: Some(Value::TensorType(Tensor {
                elem_type: dtype,
                shape: Some(TensorShapeProto {
                    dim: dims
                        .iter()
                        .map(|&d| Dimension {
                            value: Some(tensor_shape_proto::dimension::Value::DimValue(d)),
                            ..Default::default()
                        })
                        .collect(),
                }),
            })),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn make_ints_attr(name: &str, values: Vec<i64>) -> AttributeProto {
    use attribute_proto::AttributeType;
    AttributeProto {
        name: name.to_string(),
        ints: values,
        r#type: AttributeType::Ints as i32,
        ..Default::default()
    }
}

fn make_int_attr(name: &str, value: i64) -> AttributeProto {
    use attribute_proto::AttributeType;
    AttributeProto {
        name: name.to_string(),
        i: value,
        r#type: AttributeType::Int as i32,
        ..Default::default()
    }
}

/// Benchmark simple element-wise operations (Add).
fn bench_elementwise(c: &mut Criterion) {
    let graph = GraphProto {
        name: "add".to_string(),
        input: vec![
            make_value_info("input1", vec![256, 256], 1),
            make_value_info("input2", vec![256, 256], 1),
        ],
        output: vec![make_value_info("output", vec![256, 256], 1)],
        node: vec![NodeProto {
            input: vec!["input1".to_string(), "input2".to_string()],
            output: vec!["output".to_string()],
            op_type: "Add".to_string(),
            ..Default::default()
        }],
        initializer: vec![],
        ..Default::default()
    };

    c.bench_function("translate_add", |b| {
        b.iter(|| translate_graph_to_ir(black_box(&graph)))
    });
}

/// Benchmark convolution translation.
fn bench_convolution(c: &mut Criterion) {
    let mut group = c.benchmark_group("convolution");

    for size in [32, 64, 128].iter() {
        let graph = GraphProto {
            name: format!("conv_{}", size),
            input: vec![
                make_value_info("input", vec![1, 3, *size, *size], 1),
                make_value_info("weight", vec![64, 3, 3, 3], 1),
            ],
            output: vec![make_value_info("output", vec![1, 64, size - 2, size - 2], 1)],
            node: vec![NodeProto {
                input: vec!["input".to_string(), "weight".to_string()],
                output: vec!["output".to_string()],
                op_type: "Conv".to_string(),
                attribute: vec![make_ints_attr("kernel_shape", vec![3, 3])],
                ..Default::default()
            }],
            initializer: vec![],
            ..Default::default()
        };

        group.bench_with_input(BenchmarkId::from_parameter(size), &graph, |b, g| {
            b.iter(|| translate_graph_to_ir(black_box(g)))
        });
    }

    group.finish();
}

/// Benchmark matmul translation.
fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");

    for dim in [128, 512, 1024].iter() {
        let graph = GraphProto {
            name: format!("matmul_{}", dim),
            input: vec![
                make_value_info("input", vec![1, *dim], 1),
                make_value_info("weight", vec![*dim, *dim], 1),
            ],
            output: vec![make_value_info("output", vec![1, *dim], 1)],
            node: vec![NodeProto {
                input: vec!["input".to_string(), "weight".to_string()],
                output: vec!["output".to_string()],
                op_type: "MatMul".to_string(),
                ..Default::default()
            }],
            initializer: vec![],
            ..Default::default()
        };

        group.bench_with_input(BenchmarkId::from_parameter(dim), &graph, |b, g| {
            b.iter(|| translate_graph_to_ir(black_box(g)))
        });
    }

    group.finish();
}

/// Benchmark layer normalization.
fn bench_layernorm(c: &mut Criterion) {
    let graph = GraphProto {
        name: "layernorm".to_string(),
        input: vec![
            make_value_info("input", vec![1, 128, 768], 1),
            make_value_info("scale", vec![768], 1),
            make_value_info("bias", vec![768], 1),
        ],
        output: vec![make_value_info("output", vec![1, 128, 768], 1)],
        node: vec![NodeProto {
            input: vec!["input".to_string(), "scale".to_string(), "bias".to_string()],
            output: vec!["output".to_string()],
            op_type: "LayerNormalization".to_string(),
            attribute: vec![make_int_attr("axis", -1)],
            ..Default::default()
        }],
        initializer: vec![],
        ..Default::default()
    };

    c.bench_function("translate_layernorm", |b| {
        b.iter(|| translate_graph_to_ir(black_box(&graph)))
    });
}

/// Benchmark full compilation pipeline.
fn bench_e2e_compilation(c: &mut Criterion) {
    let graph = GraphProto {
        name: "conv_relu".to_string(),
        input: vec![
            make_value_info("input", vec![1, 3, 224, 224], 1),
            make_value_info("weight", vec![64, 3, 7, 7], 1),
        ],
        output: vec![make_value_info("output", vec![1, 64, 218, 218], 1)],
        node: vec![
            NodeProto {
                input: vec!["input".to_string(), "weight".to_string()],
                output: vec!["conv_out".to_string()],
                op_type: "Conv".to_string(),
                attribute: vec![make_ints_attr("kernel_shape", vec![7, 7])],
                ..Default::default()
            },
            NodeProto {
                input: vec!["conv_out".to_string()],
                output: vec!["output".to_string()],
                op_type: "Relu".to_string(),
                ..Default::default()
            },
        ],
        initializer: vec![],
        ..Default::default()
    };

    let model = ModelProto {
        ir_version: 8,
        opset_import: vec![],
        graph: Some(graph),
        ..Default::default()
    };

    let mut onnx_bytes = Vec::new();
    model.encode(&mut onnx_bytes).expect("Failed to encode");

    c.bench_function("e2e_conv_relu", |b| {
        b.iter(|| compile_onnx(black_box(&onnx_bytes)))
    });
}

criterion_group!(
    benches,
    bench_elementwise,
    bench_convolution,
    bench_matmul,
    bench_layernorm,
    bench_e2e_compilation
);
criterion_main!(benches);
