//! Mock ONNX model generators for lightweight integration testing.
//!
//! This module provides functions to generate small, valid ONNX models programmatically.
//! These models are designed to test the compilation pipeline without requiring downloads
//! of large real-world models like MNIST or BERT.
//!
//! # Model Types
//!
//! - **Identity**: Single identity op (simplest possible model)
//! - **Linear**: MatMul + Add for basic classification (MLP layer)
//! - **Conv**: Convolution network for testing CNN compilation
//! - **Multi-op**: Chains of operations for graph traversal testing
//! - **Attention**: Simplified attention pattern for transformer testing

use hologram_ai_onnx::proto::{
    AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto,
    TensorShapeProto, TypeProto, ValueInfoProto, tensor_shape_proto,
    tensor_shape_proto::dimension::Value as DimValue, type_proto::Value as TypeValue,
};
use prost::Message;

/// ONNX data types
pub mod dtype {
    pub const FLOAT: i32 = 1;
    pub const INT64: i32 = 7;
}

/// Create a tensor shape proto from dimensions.
fn make_shape(dims: &[i64]) -> TensorShapeProto {
    TensorShapeProto {
        dim: dims
            .iter()
            .map(|&d| tensor_shape_proto::Dimension {
                value: Some(DimValue::DimValue(d)),
                ..Default::default()
            })
            .collect(),
    }
}

/// Create a symbolic dimension.
fn symbolic_dim(name: &str) -> tensor_shape_proto::Dimension {
    tensor_shape_proto::Dimension {
        value: Some(DimValue::DimParam(name.to_string())),
        ..Default::default()
    }
}

/// Create a value info proto for a tensor.
fn make_value_info(name: &str, elem_type: i32, shape: TensorShapeProto) -> ValueInfoProto {
    ValueInfoProto {
        name: name.to_string(),
        r#type: Some(TypeProto {
            value: Some(TypeValue::TensorType(
                hologram_ai_onnx::proto::type_proto::Tensor {
                    elem_type,
                    shape: Some(shape),
                },
            )),
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// Create a float tensor initializer with the given shape and data.
fn make_float_initializer(name: &str, dims: &[i64], data: Vec<f32>) -> TensorProto {
    TensorProto {
        name: name.to_string(),
        dims: dims.to_vec(),
        data_type: dtype::FLOAT,
        float_data: data,
        ..Default::default()
    }
}

/// Create an int64 tensor initializer with the given shape and data.
fn make_int64_initializer(name: &str, dims: &[i64], data: Vec<i64>) -> TensorProto {
    TensorProto {
        name: name.to_string(),
        dims: dims.to_vec(),
        data_type: dtype::INT64,
        int64_data: data,
        ..Default::default()
    }
}

/// Create an attribute proto for an integer.
fn int_attr(name: &str, value: i64) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: 2, // INT
        i: value,
        ..Default::default()
    }
}

/// Create an attribute proto for a list of integers.
fn ints_attr(name: &str, values: Vec<i64>) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: 7, // INTS
        ints: values,
        ..Default::default()
    }
}

/// Create a simple node.
fn make_node(op_type: &str, inputs: Vec<&str>, outputs: Vec<&str>) -> NodeProto {
    NodeProto {
        op_type: op_type.to_string(),
        input: inputs.into_iter().map(|s| s.to_string()).collect(),
        output: outputs.into_iter().map(|s| s.to_string()).collect(),
        ..Default::default()
    }
}

/// Create a node with attributes.
fn make_node_with_attrs(
    op_type: &str,
    inputs: Vec<&str>,
    outputs: Vec<&str>,
    attrs: Vec<AttributeProto>,
) -> NodeProto {
    NodeProto {
        op_type: op_type.to_string(),
        input: inputs.into_iter().map(|s| s.to_string()).collect(),
        output: outputs.into_iter().map(|s| s.to_string()).collect(),
        attribute: attrs,
        ..Default::default()
    }
}

/// Wrap a graph in a model proto with the default opset.
fn wrap_model(graph: GraphProto, opset_version: i64) -> ModelProto {
    ModelProto {
        ir_version: 8,
        opset_import: vec![OperatorSetIdProto {
            domain: "".to_string(),
            version: opset_version,
        }],
        graph: Some(graph),
        producer_name: "hologram-ai-mock".to_string(),
        producer_version: "1.0".to_string(),
        ..Default::default()
    }
}

/// Generate the simplest possible valid ONNX model: a single Identity operation.
///
/// Input: [1, 10] float tensor
/// Output: [1, 10] float tensor (same as input)
pub fn identity_model() -> Vec<u8> {
    let graph = GraphProto {
        name: "identity_graph".to_string(),
        input: vec![make_value_info("input", dtype::FLOAT, make_shape(&[1, 10]))],
        output: vec![make_value_info(
            "output",
            dtype::FLOAT,
            make_shape(&[1, 10]),
        )],
        node: vec![make_node("Identity", vec!["input"], vec!["output"])],
        ..Default::default()
    };

    wrap_model(graph, 13).encode_to_vec()
}

/// Generate a simple linear layer model: MatMul + Add.
///
/// This represents a single fully-connected layer with bias.
/// Input: [batch, 16] float tensor
/// Output: [batch, 8] float tensor
///
/// Parameters:
/// - weights: [16, 8] (128 floats)
/// - bias: [8] (8 floats)
pub fn linear_model() -> Vec<u8> {
    // Initialize weights with small values
    let weights: Vec<f32> = (0..128).map(|i| (i as f32 * 0.01) - 0.64).collect();
    let bias: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();

    let graph = GraphProto {
        name: "linear_graph".to_string(),
        input: vec![make_value_info(
            "input",
            dtype::FLOAT,
            TensorShapeProto {
                dim: vec![
                    symbolic_dim("batch"),
                    tensor_shape_proto::Dimension {
                        value: Some(DimValue::DimValue(16)),
                        ..Default::default()
                    },
                ],
            },
        )],
        output: vec![make_value_info(
            "output",
            dtype::FLOAT,
            TensorShapeProto {
                dim: vec![
                    symbolic_dim("batch"),
                    tensor_shape_proto::Dimension {
                        value: Some(DimValue::DimValue(8)),
                        ..Default::default()
                    },
                ],
            },
        )],
        node: vec![
            make_node("MatMul", vec!["input", "weights"], vec!["matmul_out"]),
            make_node("Add", vec!["matmul_out", "bias"], vec!["output"]),
        ],
        initializer: vec![
            make_float_initializer("weights", &[16, 8], weights),
            make_float_initializer("bias", &[8], bias),
        ],
        ..Default::default()
    };

    wrap_model(graph, 13).encode_to_vec()
}

/// Generate a multi-layer perceptron model with ReLU activations.
///
/// Architecture: Linear(16, 32) -> ReLU -> Linear(32, 8) -> Softmax
///
/// This tests multi-operation graph compilation and activation functions.
pub fn mlp_model() -> Vec<u8> {
    // Layer 1: [16, 32]
    let w1: Vec<f32> = (0..512).map(|i| (i as f32 * 0.002) - 0.512).collect();
    let b1: Vec<f32> = (0..32).map(|i| i as f32 * 0.01).collect();

    // Layer 2: [32, 8]
    let w2: Vec<f32> = (0..256).map(|i| (i as f32 * 0.003) - 0.384).collect();
    let b2: Vec<f32> = (0..8).map(|i| i as f32 * 0.02).collect();

    let graph = GraphProto {
        name: "mlp_graph".to_string(),
        input: vec![make_value_info(
            "input",
            dtype::FLOAT,
            TensorShapeProto {
                dim: vec![
                    symbolic_dim("batch"),
                    tensor_shape_proto::Dimension {
                        value: Some(DimValue::DimValue(16)),
                        ..Default::default()
                    },
                ],
            },
        )],
        output: vec![make_value_info(
            "output",
            dtype::FLOAT,
            TensorShapeProto {
                dim: vec![
                    symbolic_dim("batch"),
                    tensor_shape_proto::Dimension {
                        value: Some(DimValue::DimValue(8)),
                        ..Default::default()
                    },
                ],
            },
        )],
        node: vec![
            make_node("MatMul", vec!["input", "w1"], vec!["fc1"]),
            make_node("Add", vec!["fc1", "b1"], vec!["fc1_bias"]),
            make_node("Relu", vec!["fc1_bias"], vec!["relu1"]),
            make_node("MatMul", vec!["relu1", "w2"], vec!["fc2"]),
            make_node("Add", vec!["fc2", "b2"], vec!["fc2_bias"]),
            make_node_with_attrs(
                "Softmax",
                vec!["fc2_bias"],
                vec!["output"],
                vec![int_attr("axis", -1)],
            ),
        ],
        initializer: vec![
            make_float_initializer("w1", &[16, 32], w1),
            make_float_initializer("b1", &[32], b1),
            make_float_initializer("w2", &[32, 8], w2),
            make_float_initializer("b2", &[8], b2),
        ],
        ..Default::default()
    };

    wrap_model(graph, 13).encode_to_vec()
}

/// Generate a simple 2D convolution model (conv only, no pooling).
///
/// Architecture: Conv2D(1, 4, kernel=3x3) -> ReLU
///
/// This tests convolution compilation with proper attribute handling.
/// Input: [batch, 1, 8, 8] (8x8 grayscale image)
/// Output: [batch, 4, 6, 6] (4 feature maps after 3x3 conv)
pub fn conv_model() -> Vec<u8> {
    // Conv weights: [out_channels, in_channels, kernel_h, kernel_w] = [4, 1, 3, 3]
    let conv_weights: Vec<f32> = (0..36).map(|i| (i as f32 * 0.05) - 0.9).collect();
    let conv_bias: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];

    let graph = GraphProto {
        name: "conv_graph".to_string(),
        input: vec![make_value_info(
            "input",
            dtype::FLOAT,
            TensorShapeProto {
                dim: vec![
                    symbolic_dim("batch"),
                    tensor_shape_proto::Dimension {
                        value: Some(DimValue::DimValue(1)),
                        ..Default::default()
                    },
                    tensor_shape_proto::Dimension {
                        value: Some(DimValue::DimValue(8)),
                        ..Default::default()
                    },
                    tensor_shape_proto::Dimension {
                        value: Some(DimValue::DimValue(8)),
                        ..Default::default()
                    },
                ],
            },
        )],
        output: vec![make_value_info(
            "output",
            dtype::FLOAT,
            TensorShapeProto {
                dim: vec![
                    symbolic_dim("batch"),
                    tensor_shape_proto::Dimension {
                        value: Some(DimValue::DimValue(4)),
                        ..Default::default()
                    },
                    tensor_shape_proto::Dimension {
                        value: Some(DimValue::DimValue(6)),
                        ..Default::default()
                    },
                    tensor_shape_proto::Dimension {
                        value: Some(DimValue::DimValue(6)),
                        ..Default::default()
                    },
                ],
            },
        )],
        node: vec![
            make_node_with_attrs(
                "Conv",
                vec!["input", "conv_w", "conv_b"],
                vec!["conv_out"],
                vec![
                    ints_attr("kernel_shape", vec![3, 3]),
                    ints_attr("strides", vec![1, 1]),
                    ints_attr("pads", vec![0, 0, 0, 0]),
                ],
            ),
            make_node("Relu", vec!["conv_out"], vec!["output"]),
        ],
        initializer: vec![
            make_float_initializer("conv_w", &[4, 1, 3, 3], conv_weights),
            make_float_initializer("conv_b", &[4], conv_bias),
        ],
        ..Default::default()
    };

    wrap_model(graph, 13).encode_to_vec()
}

/// Generate a model with Gather and shape operations.
///
/// This tests dynamic indexing and shape manipulation ops commonly found
/// in modern architectures.
pub fn gather_shape_model() -> Vec<u8> {
    // Embedding table: [100, 16] (100 tokens, 16-dim embeddings)
    let embeddings: Vec<f32> = (0..1600).map(|i| (i as f32 * 0.001) - 0.8).collect();

    let graph = GraphProto {
        name: "gather_shape_graph".to_string(),
        input: vec![make_value_info(
            "indices",
            dtype::INT64,
            TensorShapeProto {
                dim: vec![symbolic_dim("batch"), symbolic_dim("seq_len")],
            },
        )],
        output: vec![make_value_info(
            "output",
            dtype::FLOAT,
            TensorShapeProto {
                dim: vec![
                    symbolic_dim("batch"),
                    symbolic_dim("seq_len"),
                    tensor_shape_proto::Dimension {
                        value: Some(DimValue::DimValue(16)),
                        ..Default::default()
                    },
                ],
            },
        )],
        node: vec![make_node_with_attrs(
            "Gather",
            vec!["embeddings", "indices"],
            vec!["output"],
            vec![int_attr("axis", 0)],
        )],
        initializer: vec![make_float_initializer("embeddings", &[100, 16], embeddings)],
        ..Default::default()
    };

    wrap_model(graph, 13).encode_to_vec()
}

/// Generate a model with Transpose and Reshape operations.
///
/// Tests tensor manipulation operations for attention-like patterns.
/// Input: [batch, seq, 4, 8] -> Transpose -> [batch, 4, seq, 8] -> Reshape -> [batch*4, seq, 8]
pub fn transpose_reshape_model() -> Vec<u8> {
    let graph = GraphProto {
        name: "transpose_reshape_graph".to_string(),
        input: vec![make_value_info(
            "input",
            dtype::FLOAT,
            TensorShapeProto {
                dim: vec![
                    symbolic_dim("batch"),
                    symbolic_dim("seq"),
                    tensor_shape_proto::Dimension {
                        value: Some(DimValue::DimValue(4)),
                        ..Default::default()
                    },
                    tensor_shape_proto::Dimension {
                        value: Some(DimValue::DimValue(8)),
                        ..Default::default()
                    },
                ],
            },
        )],
        output: vec![make_value_info(
            "output",
            dtype::FLOAT,
            TensorShapeProto {
                dim: vec![
                    tensor_shape_proto::Dimension {
                        value: Some(DimValue::DimValue(-1)),
                        ..Default::default()
                    },
                    symbolic_dim("seq"),
                    tensor_shape_proto::Dimension {
                        value: Some(DimValue::DimValue(8)),
                        ..Default::default()
                    },
                ],
            },
        )],
        node: vec![
            make_node_with_attrs(
                "Transpose",
                vec!["input"],
                vec!["transposed"],
                vec![ints_attr("perm", vec![0, 2, 1, 3])],
            ),
            make_node("Reshape", vec!["transposed", "new_shape"], vec!["output"]),
        ],
        initializer: vec![make_int64_initializer("new_shape", &[3], vec![-1, 0, 8])],
        ..Default::default()
    };

    wrap_model(graph, 13).encode_to_vec()
}

/// Generate a tiny "MNIST-like" classifier model.
///
/// This provides a minimal MLP model for testing classification pipelines.
/// Uses a simple architecture without convolutions for reliability.
///
/// Input: [batch, 16] (flat 16-dim input)
/// Output: [batch, 10] (10-class probabilities)
pub fn mini_classifier_model() -> Vec<u8> {
    // FC1: [16, 32] = 512 params
    let fc1_w: Vec<f32> = (0..512).map(|i| (i as f32 * 0.002) - 0.512).collect();
    let fc1_b: Vec<f32> = (0..32).map(|i| i as f32 * 0.01).collect();

    // FC2: [32, 10] = 320 params
    let fc2_w: Vec<f32> = (0..320).map(|i| (i as f32 * 0.003) - 0.48).collect();
    let fc2_b: Vec<f32> = (0..10).map(|i| i as f32 * 0.02).collect();

    let graph = GraphProto {
        name: "mini_classifier".to_string(),
        input: vec![make_value_info(
            "input",
            dtype::FLOAT,
            TensorShapeProto {
                dim: vec![
                    symbolic_dim("batch"),
                    tensor_shape_proto::Dimension {
                        value: Some(DimValue::DimValue(16)),
                        ..Default::default()
                    },
                ],
            },
        )],
        output: vec![make_value_info(
            "output",
            dtype::FLOAT,
            TensorShapeProto {
                dim: vec![
                    symbolic_dim("batch"),
                    tensor_shape_proto::Dimension {
                        value: Some(DimValue::DimValue(10)),
                        ..Default::default()
                    },
                ],
            },
        )],
        node: vec![
            make_node("MatMul", vec!["input", "fc1_w"], vec!["fc1"]),
            make_node("Add", vec!["fc1", "fc1_b"], vec!["fc1_bias"]),
            make_node("Relu", vec!["fc1_bias"], vec!["relu1"]),
            make_node("MatMul", vec!["relu1", "fc2_w"], vec!["fc2"]),
            make_node("Add", vec!["fc2", "fc2_b"], vec!["logits"]),
            make_node_with_attrs(
                "Softmax",
                vec!["logits"],
                vec!["output"],
                vec![int_attr("axis", -1)],
            ),
        ],
        initializer: vec![
            make_float_initializer("fc1_w", &[16, 32], fc1_w),
            make_float_initializer("fc1_b", &[32], fc1_b),
            make_float_initializer("fc2_w", &[32, 10], fc2_w),
            make_float_initializer("fc2_b", &[10], fc2_b),
        ],
        ..Default::default()
    };

    wrap_model(graph, 13).encode_to_vec()
}

/// Generate a model with element-wise operations.
///
/// Tests: Add, Sub, Mul, Div with broadcasting.
pub fn elementwise_model() -> Vec<u8> {
    let scale = vec![1.0_f32, 2.0, 0.5, 1.5];
    let bias = vec![0.1_f32, -0.1, 0.2, -0.2];

    let graph = GraphProto {
        name: "elementwise_graph".to_string(),
        input: vec![make_value_info(
            "input",
            dtype::FLOAT,
            TensorShapeProto {
                dim: vec![
                    symbolic_dim("batch"),
                    tensor_shape_proto::Dimension {
                        value: Some(DimValue::DimValue(4)),
                        ..Default::default()
                    },
                ],
            },
        )],
        output: vec![make_value_info(
            "output",
            dtype::FLOAT,
            TensorShapeProto {
                dim: vec![
                    symbolic_dim("batch"),
                    tensor_shape_proto::Dimension {
                        value: Some(DimValue::DimValue(4)),
                        ..Default::default()
                    },
                ],
            },
        )],
        node: vec![
            make_node("Mul", vec!["input", "scale"], vec!["scaled"]),
            make_node("Add", vec!["scaled", "bias"], vec!["output"]),
        ],
        initializer: vec![
            make_float_initializer("scale", &[4], scale),
            make_float_initializer("bias", &[4], bias),
        ],
        ..Default::default()
    };

    wrap_model(graph, 13).encode_to_vec()
}

/// Generate a model with reduction operations.
///
/// Tests: ReduceMean, ReduceSum
pub fn reduction_model() -> Vec<u8> {
    let graph = GraphProto {
        name: "reduction_graph".to_string(),
        input: vec![make_value_info(
            "input",
            dtype::FLOAT,
            TensorShapeProto {
                dim: vec![
                    symbolic_dim("batch"),
                    tensor_shape_proto::Dimension {
                        value: Some(DimValue::DimValue(8)),
                        ..Default::default()
                    },
                    tensor_shape_proto::Dimension {
                        value: Some(DimValue::DimValue(16)),
                        ..Default::default()
                    },
                ],
            },
        )],
        output: vec![make_value_info(
            "output",
            dtype::FLOAT,
            TensorShapeProto {
                dim: vec![
                    symbolic_dim("batch"),
                    tensor_shape_proto::Dimension {
                        value: Some(DimValue::DimValue(8)),
                        ..Default::default()
                    },
                ],
            },
        )],
        node: vec![make_node_with_attrs(
            "ReduceMean",
            vec!["input"],
            vec!["output"],
            vec![ints_attr("axes", vec![-1]), int_attr("keepdims", 0)],
        )],
        ..Default::default()
    };

    wrap_model(graph, 13).encode_to_vec()
}

/// Generate a model with Concat operation.
///
/// Tests concatenation along different axes.
pub fn concat_model() -> Vec<u8> {
    let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let b: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];

    let graph = GraphProto {
        name: "concat_graph".to_string(),
        input: vec![make_value_info(
            "input",
            dtype::FLOAT,
            TensorShapeProto {
                dim: vec![
                    symbolic_dim("batch"),
                    tensor_shape_proto::Dimension {
                        value: Some(DimValue::DimValue(4)),
                        ..Default::default()
                    },
                ],
            },
        )],
        output: vec![make_value_info(
            "output",
            dtype::FLOAT,
            TensorShapeProto {
                dim: vec![
                    symbolic_dim("batch"),
                    tensor_shape_proto::Dimension {
                        value: Some(DimValue::DimValue(12)),
                        ..Default::default()
                    },
                ],
            },
        )],
        node: vec![make_node_with_attrs(
            "Concat",
            vec!["input", "const_a", "const_b"],
            vec!["output"],
            vec![int_attr("axis", -1)],
        )],
        initializer: vec![
            make_float_initializer("const_a", &[1, 4], a),
            make_float_initializer("const_b", &[1, 4], b),
        ],
        ..Default::default()
    };

    wrap_model(graph, 13).encode_to_vec()
}

/// Generate a model with Split operation.
///
/// Tests splitting tensors.
pub fn split_model() -> Vec<u8> {
    let graph = GraphProto {
        name: "split_graph".to_string(),
        input: vec![make_value_info(
            "input",
            dtype::FLOAT,
            TensorShapeProto {
                dim: vec![
                    symbolic_dim("batch"),
                    tensor_shape_proto::Dimension {
                        value: Some(DimValue::DimValue(12)),
                        ..Default::default()
                    },
                ],
            },
        )],
        output: vec![
            make_value_info(
                "out_a",
                dtype::FLOAT,
                TensorShapeProto {
                    dim: vec![
                        symbolic_dim("batch"),
                        tensor_shape_proto::Dimension {
                            value: Some(DimValue::DimValue(4)),
                            ..Default::default()
                        },
                    ],
                },
            ),
            make_value_info(
                "out_b",
                dtype::FLOAT,
                TensorShapeProto {
                    dim: vec![
                        symbolic_dim("batch"),
                        tensor_shape_proto::Dimension {
                            value: Some(DimValue::DimValue(4)),
                            ..Default::default()
                        },
                    ],
                },
            ),
            make_value_info(
                "out_c",
                dtype::FLOAT,
                TensorShapeProto {
                    dim: vec![
                        symbolic_dim("batch"),
                        tensor_shape_proto::Dimension {
                            value: Some(DimValue::DimValue(4)),
                            ..Default::default()
                        },
                    ],
                },
            ),
        ],
        node: vec![make_node_with_attrs(
            "Split",
            vec!["input"],
            vec!["out_a", "out_b", "out_c"],
            vec![int_attr("axis", -1), ints_attr("split", vec![4, 4, 4])],
        )],
        ..Default::default()
    };

    wrap_model(graph, 13).encode_to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_model_is_valid_protobuf() {
        let bytes = identity_model();
        assert!(!bytes.is_empty());
        let model = ModelProto::decode(bytes.as_slice()).expect("Should decode as valid protobuf");
        assert!(model.graph.is_some());
    }

    #[test]
    fn test_linear_model_is_valid_protobuf() {
        let bytes = linear_model();
        let model = ModelProto::decode(bytes.as_slice()).expect("Should decode as valid protobuf");
        let graph = model.graph.unwrap();
        assert_eq!(graph.node.len(), 2); // MatMul + Add
        assert_eq!(graph.initializer.len(), 2); // weights + bias
    }

    #[test]
    fn test_mlp_model_is_valid_protobuf() {
        let bytes = mlp_model();
        let model = ModelProto::decode(bytes.as_slice()).expect("Should decode as valid protobuf");
        let graph = model.graph.unwrap();
        assert_eq!(graph.node.len(), 6); // 2x(MatMul + Add + activation)
    }

    #[test]
    fn test_conv_model_is_valid_protobuf() {
        let bytes = conv_model();
        let model = ModelProto::decode(bytes.as_slice()).expect("Should decode as valid protobuf");
        let graph = model.graph.unwrap();
        assert!(graph.node.iter().any(|n| n.op_type == "Conv"));
    }

    #[test]
    fn test_mini_classifier_is_valid_protobuf() {
        let bytes = mini_classifier_model();
        let model = ModelProto::decode(bytes.as_slice()).expect("Should decode as valid protobuf");
        let graph = model.graph.unwrap();
        // MLP-based mini classifier has MatMul and Softmax
        assert!(graph.node.iter().any(|n| n.op_type == "MatMul"));
        assert!(graph.node.iter().any(|n| n.op_type == "Softmax"));
    }
}
