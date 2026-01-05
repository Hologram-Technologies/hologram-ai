//! Integration tests for full ONNX→IR translation pipeline.
//!
//! These tests verify end-to-end translation of complete ONNX models.

use hologram_onnx::translate_graph_to_ir;
use hologram_onnx::proto::{GraphProto, NodeProto, ValueInfoProto, TypeProto};
use hologram_onnx::proto::type_proto::{Value, Tensor};
use hologram_onnx::proto::tensor_shape_proto::Dimension;
use hologram_onnx::proto::TensorShapeProto;
use hologram_onnx::proto::attribute_proto::AttributeType;
use hologram_onnx::proto::AttributeProto;

fn make_value_info(name: &str, dims: Vec<i64>, dtype: i32) -> ValueInfoProto {
    ValueInfoProto {
        name: name.to_string(),
        r#type: Some(TypeProto {
            value: Some(Value::TensorType(Tensor {
                elem_type: dtype,
                shape: Some(TensorShapeProto {
                    dim: dims.iter().map(|&d| Dimension {
                        value: Some(hologram_onnx::proto::tensor_shape_proto::dimension::Value::DimValue(d)),
                        ..Default::default()
                    }).collect(),
                }),
            })),
            ..Default::default()
        }),
        ..Default::default()
    }
}

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
fn test_simple_add_model() {
    // Create a simple graph: output = input1 + input2
    let graph = GraphProto {
        name: "simple_add".to_string(),
        input: vec![
            make_value_info("input1", vec![1, 3, 224, 224], 1), // FLOAT
            make_value_info("input2", vec![1, 3, 224, 224], 1),
        ],
        output: vec![
            make_value_info("output", vec![1, 3, 224, 224], 1),
        ],
        node: vec![
            NodeProto {
                input: vec!["input1".to_string(), "input2".to_string()],
                output: vec!["output".to_string()],
                op_type: "Add".to_string(),
                ..Default::default()
            },
        ],
        initializer: vec![],
        ..Default::default()
    };

    let result = translate_graph_to_ir(&graph);
    assert!(result.is_ok(), "Translation failed: {:?}", result.err());
}

#[test]
fn test_conv_relu_model() {
    // Create graph: output = ReLU(Conv(input, weight))
    let graph = GraphProto {
        name: "conv_relu".to_string(),
        input: vec![
            make_value_info("input", vec![1, 3, 224, 224], 1),
            make_value_info("weight", vec![64, 3, 3, 3], 1),
        ],
        output: vec![
            make_value_info("output", vec![1, 64, 222, 222], 1),
        ],
        node: vec![
            NodeProto {
                input: vec!["input".to_string(), "weight".to_string()],
                output: vec!["conv_output".to_string()],
                op_type: "Conv".to_string(),
                attribute: vec![make_ints_attr("kernel_shape", vec![3, 3])],
                ..Default::default()
            },
            NodeProto {
                input: vec!["conv_output".to_string()],
                output: vec!["output".to_string()],
                op_type: "Relu".to_string(),
                ..Default::default()
            },
        ],
        initializer: vec![],
        ..Default::default()
    };

    let result = translate_graph_to_ir(&graph);
    assert!(result.is_ok(), "Translation failed: {:?}", result.err());
}

#[test]
fn test_matmul_add_model() {
    // Create graph: output = MatMul(input, weight) + bias
    let graph = GraphProto {
        name: "matmul_add".to_string(),
        input: vec![
            make_value_info("input", vec![1, 128, 768], 1),
            make_value_info("weight", vec![768, 768], 1),
            make_value_info("bias", vec![768], 1),
        ],
        output: vec![
            make_value_info("output", vec![1, 128, 768], 1),
        ],
        node: vec![
            NodeProto {
                input: vec!["input".to_string(), "weight".to_string()],
                output: vec!["matmul_output".to_string()],
                op_type: "MatMul".to_string(),
                ..Default::default()
            },
            NodeProto {
                input: vec!["matmul_output".to_string(), "bias".to_string()],
                output: vec!["output".to_string()],
                op_type: "Add".to_string(),
                ..Default::default()
            },
        ],
        initializer: vec![],
        ..Default::default()
    };

    let result = translate_graph_to_ir(&graph);
    assert!(result.is_ok(), "Translation failed: {:?}", result.err());
}

#[test]
fn test_layernorm_model() {
    // Create graph: output = LayerNormalization(input, scale, bias)
    let graph = GraphProto {
        name: "layernorm".to_string(),
        input: vec![
            make_value_info("input", vec![1, 128, 768], 1),
            make_value_info("scale", vec![768], 1),
            make_value_info("bias", vec![768], 1),
        ],
        output: vec![
            make_value_info("output", vec![1, 128, 768], 1),
        ],
        node: vec![
            NodeProto {
                input: vec!["input".to_string(), "scale".to_string(), "bias".to_string()],
                output: vec!["output".to_string()],
                op_type: "LayerNormalization".to_string(),
                attribute: vec![make_int_attr("axis", -1)],
                ..Default::default()
            },
        ],
        initializer: vec![],
        ..Default::default()
    };

    let result = translate_graph_to_ir(&graph);
    assert!(result.is_ok(), "Translation failed: {:?}", result.err());
}

#[test]
fn test_softmax_model() {
    // Create graph: output = Softmax(input)
    let graph = GraphProto {
        name: "softmax".to_string(),
        input: vec![
            make_value_info("input", vec![1, 12, 128, 128], 1),
        ],
        output: vec![
            make_value_info("output", vec![1, 12, 128, 128], 1),
        ],
        node: vec![
            NodeProto {
                input: vec!["input".to_string()],
                output: vec!["output".to_string()],
                op_type: "Softmax".to_string(),
                attribute: vec![make_int_attr("axis", -1)],
                ..Default::default()
            },
        ],
        initializer: vec![],
        ..Default::default()
    };

    let result = translate_graph_to_ir(&graph);
    assert!(result.is_ok(), "Translation failed: {:?}", result.err());
}
