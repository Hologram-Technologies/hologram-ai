//! Tests for ONNX operations with symbolic/dynamic shapes.
//!
//! These tests verify that operations correctly handle:
//! - Variable batch sizes (Dim::Symbolic)
//! - Variable sequence lengths (Dim::Dynamic)
//! - Mixed static and symbolic dimensions

use hologram::ir::{GraphBuilder, DType, Shape, Dim};
use hologram_onnx::ops::*;
use hologram_onnx::proto::attribute_proto::AttributeType;
use hologram_onnx::proto::AttributeProto;

fn make_ints_attr(name: &str, values: Vec<i64>) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        ints: values,
        r#type: AttributeType::Ints as i32,
        ..Default::default()
    }
}

fn make_int_attr(name: &str, value: i64) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        i: value,
        r#type: AttributeType::Int as i32,
        ..Default::default()
    }
}

#[test]
fn test_matmul_with_variable_batch() {
    let mut builder = GraphBuilder::new();

    // Input with symbolic batch dimension: [batch, seq_len, hidden]
    let shape_a = Shape {
        dims: vec![Dim::Symbolic("batch".into()), Dim::Static(128), Dim::Static(768)],
    };
    let a = builder.input("a", shape_a, DType::F32);

    // Weight matrix: [hidden, hidden]
    let shape_b = Shape::static_shape(&[768, 768]);
    let b = builder.input("b", shape_b, DType::F32);

    let result = core::translate_gemm(&[a, b], &[], &mut builder);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 1);
}

#[test]
fn test_conv_with_dynamic_batch() {
    let mut builder = GraphBuilder::new();

    // Input with dynamic batch: [batch, channels, height, width]
    let shape = Shape {
        dims: vec![Dim::Dynamic, Dim::Static(3), Dim::Static(224), Dim::Static(224)],
    };
    let input = builder.input("input", shape, DType::F32);

    let weight = builder.input("weight", Shape::static_shape(&[64, 3, 7, 7]), DType::F32);

    let attrs = vec![
        make_ints_attr("kernel_shape", vec![7, 7]),
        make_ints_attr("strides", vec![2, 2]),
        make_ints_attr("pads", vec![3, 3, 3, 3]),
    ];

    let result = conv::translate_conv(&[input, weight], &attrs, &mut builder);
    assert!(result.is_ok());
}

#[test]
fn test_layer_norm_with_symbolic_dims() {
    let mut builder = GraphBuilder::new();

    // Input: [batch, seq_len, hidden_dim]
    let shape = Shape {
        dims: vec![
            Dim::Symbolic("batch".into()),
            Dim::Symbolic("seq_len".into()),
            Dim::Static(768),
        ],
    };
    let input = builder.input("input", shape, DType::F32);

    let attrs = vec![make_int_attr("axis", -1)];

    let result = norm::translate_layer_norm(&[input], &attrs, &mut builder);
    assert!(result.is_ok());
}

#[test]
fn test_softmax_with_variable_sequence_length() {
    let mut builder = GraphBuilder::new();

    // Input: [batch, num_heads, seq_len, seq_len]
    let shape = Shape {
        dims: vec![
            Dim::Symbolic("batch".into()),
            Dim::Static(12),
            Dim::Symbolic("seq_len".into()),
            Dim::Symbolic("seq_len".into()),
        ],
    };
    let input = builder.input("input", shape, DType::F32);

    let attrs = vec![make_int_attr("axis", -1)];

    let result = activation::translate_softmax(&[input], &attrs, &mut builder);
    assert!(result.is_ok());
}

#[test]
fn test_reduce_sum_with_dynamic_dims() {
    let mut builder = GraphBuilder::new();

    // Input with multiple dynamic dimensions
    let shape = Shape {
        dims: vec![Dim::Dynamic, Dim::Dynamic, Dim::Static(256)],
    };
    let input = builder.input("input", shape, DType::F32);

    let attrs = vec![
        make_ints_attr("axes", vec![2]),
        make_int_attr("keepdims", 1),
    ];

    let result = reduction::translate_reduce_sum(&[input], &attrs, &mut builder);
    assert!(result.is_ok());
}

#[test]
fn test_concat_with_symbolic_batch() {
    let mut builder = GraphBuilder::new();

    // Multiple inputs with same symbolic batch dimension
    let shape1 = Shape {
        dims: vec![Dim::Symbolic("batch".into()), Dim::Static(256)],
    };
    let shape2 = Shape {
        dims: vec![Dim::Symbolic("batch".into()), Dim::Static(512)],
    };

    let input1 = builder.input("input1", shape1, DType::F32);
    let input2 = builder.input("input2", shape2, DType::F32);

    let attrs = vec![make_int_attr("axis", 1)];

    let result = shape::translate_concat(&[input1, input2], &attrs, &mut builder);
    assert!(result.is_ok());
}

#[test]
#[ignore = "Reshape with zeros requires special handling for symbolic dimensions"]
fn test_reshape_with_dynamic_batch() {
    let mut builder = GraphBuilder::new();

    // Input: [batch, seq_len, hidden]
    let shape = Shape {
        dims: vec![
            Dim::Dynamic,
            Dim::Symbolic("seq_len".into()),
            Dim::Static(768),
        ],
    };
    let input = builder.input("input", shape, DType::F32);

    // Reshape to [batch, seq_len, num_heads, head_dim]
    let attrs = vec![make_ints_attr("shape", vec![0, 0, 12, 64])];

    let result = shape::translate_reshape(&[input], &attrs, &mut builder);
    assert!(result.is_ok());
}

#[test]
fn test_transpose_preserves_symbolic_dims() {
    let mut builder = GraphBuilder::new();

    // Input: [batch, seq_len, num_heads, head_dim]
    let shape = Shape {
        dims: vec![
            Dim::Symbolic("batch".into()),
            Dim::Symbolic("seq_len".into()),
            Dim::Static(12),
            Dim::Static(64),
        ],
    };
    let input = builder.input("input", shape, DType::F32);

    // Transpose to [batch, num_heads, seq_len, head_dim]
    let attrs = vec![make_ints_attr("perm", vec![0, 2, 1, 3])];

    let result = shape::translate_transpose(&[input], &attrs, &mut builder);
    assert!(result.is_ok());
}

#[test]
fn test_gather_with_dynamic_indices() {
    let mut builder = GraphBuilder::new();

    // Data: [vocab_size, embedding_dim]
    let data = builder.input("data", Shape::static_shape(&[50000, 768]), DType::F32);

    // Indices with dynamic shape: [batch, seq_len]
    let indices_shape = Shape {
        dims: vec![Dim::Dynamic, Dim::Symbolic("seq_len".into())],
    };
    let indices = builder.input("indices", indices_shape, DType::I64);

    let attrs = vec![make_int_attr("axis", 0)];

    let result = indexing::translate_gather(&[data, indices], &attrs, &mut builder);
    assert!(result.is_ok());
}

#[test]
fn test_pooling_with_variable_batch() {
    let mut builder = GraphBuilder::new();

    // Input: [batch, channels, height, width]
    let shape = Shape {
        dims: vec![Dim::Dynamic, Dim::Static(512), Dim::Static(7), Dim::Static(7)],
    };
    let input = builder.input("input", shape, DType::F32);

    let result = pool::translate_global_average_pool(&[input], &[], &mut builder);
    assert!(result.is_ok());
}
