//! End-to-end compilation test for ONNX models.

use hologram_ai_onnx::proto;
use prost::Message;

/// Create a simple ONNX model: Input([1,10]) -> ReLU -> Output([1,10])
fn create_simple_relu_model() -> Vec<u8> {
    let model = proto::ModelProto {
        ir_version: 8,
        opset_import: vec![proto::OperatorSetIdProto {
            domain: String::new(),
            version: 17,
        }],
        graph: Some(proto::GraphProto {
            name: "test_graph".to_string(),
            // Define input
            input: vec![proto::ValueInfoProto {
                name: "input".to_string(),
                r#type: Some(proto::TypeProto {
                    value: Some(proto::type_proto::Value::TensorType(
                        proto::type_proto::Tensor {
                            elem_type: 1, // FLOAT
                            shape: Some(proto::TensorShapeProto {
                                dim: vec![
                                    proto::tensor_shape_proto::Dimension {
                                        value: Some(
                                            proto::tensor_shape_proto::dimension::Value::DimValue(
                                                1,
                                            ),
                                        ),
                                        denotation: String::new(),
                                    },
                                    proto::tensor_shape_proto::Dimension {
                                        value: Some(
                                            proto::tensor_shape_proto::dimension::Value::DimValue(
                                                10,
                                            ),
                                        ),
                                        denotation: String::new(),
                                    },
                                ],
                            }),
                        },
                    )),
                    denotation: String::new(),
                }),
                ..Default::default()
            }],
            // Define ReLU node
            node: vec![proto::NodeProto {
                input: vec!["input".to_string()],
                output: vec!["relu_out".to_string()],
                name: "relu_node".to_string(),
                op_type: "Relu".to_string(),
                ..Default::default()
            }],
            // Define output
            output: vec![proto::ValueInfoProto {
                name: "relu_out".to_string(),
                r#type: Some(proto::TypeProto {
                    value: Some(proto::type_proto::Value::TensorType(
                        proto::type_proto::Tensor {
                            elem_type: 1, // FLOAT
                            shape: Some(proto::TensorShapeProto {
                                dim: vec![
                                    proto::tensor_shape_proto::Dimension {
                                        value: Some(
                                            proto::tensor_shape_proto::dimension::Value::DimValue(
                                                1,
                                            ),
                                        ),
                                        denotation: String::new(),
                                    },
                                    proto::tensor_shape_proto::Dimension {
                                        value: Some(
                                            proto::tensor_shape_proto::dimension::Value::DimValue(
                                                10,
                                            ),
                                        ),
                                        denotation: String::new(),
                                    },
                                ],
                            }),
                        },
                    )),
                    denotation: String::new(),
                }),
                ..Default::default()
            }],
            ..Default::default()
        }),
        producer_name: "hologram-ai-onnx-test".to_string(),
        producer_version: "0.1.0".to_string(),
        ..Default::default()
    };

    // Serialize to bytes
    let mut bytes = Vec::new();
    model.encode(&mut bytes).expect("Failed to encode model");
    bytes
}

#[test]
fn test_compile_simple_relu_model() {
    // Create a simple ONNX model
    let onnx_bytes = create_simple_relu_model();

    // Compile it
    let result = hologram_ai_onnx::compile_onnx(&onnx_bytes);

    // Should succeed
    assert!(result.is_ok(), "Compilation failed: {:?}", result.err());

    // Get the .holb bytes
    let holb_bytes = result.unwrap();

    // Should produce non-empty output
    assert!(!holb_bytes.is_empty(), "Compiled output is empty");

    // Should start with HOLB magic bytes
    assert!(
        holb_bytes.len() >= 4,
        "Output too small to contain magic bytes"
    );
    assert_eq!(
        &holb_bytes[0..4],
        b"HOLB",
        "Output does not start with HOLB magic bytes"
    );

    println!("✅ Successfully compiled simple ReLU model");
    println!("   ONNX size: {} bytes", onnx_bytes.len());
    println!("   HOLB size: {} bytes", holb_bytes.len());
}

#[test]
fn test_compile_matmul_model() {
    // Create a simple MatMul model: Input([2,3]) @ Weight([3,4]) -> Output([2,4])
    let model = proto::ModelProto {
        ir_version: 8,
        opset_import: vec![proto::OperatorSetIdProto {
            domain: String::new(),
            version: 17,
        }],
        graph: Some(proto::GraphProto {
            name: "matmul_graph".to_string(),
            input: vec![proto::ValueInfoProto {
                name: "input".to_string(),
                r#type: Some(proto::TypeProto {
                    value: Some(proto::type_proto::Value::TensorType(
                        proto::type_proto::Tensor {
                            elem_type: 1, // FLOAT
                            shape: Some(proto::TensorShapeProto {
                                dim: vec![
                                    proto::tensor_shape_proto::Dimension {
                                        value: Some(
                                            proto::tensor_shape_proto::dimension::Value::DimValue(
                                                2,
                                            ),
                                        ),
                                        denotation: String::new(),
                                    },
                                    proto::tensor_shape_proto::Dimension {
                                        value: Some(
                                            proto::tensor_shape_proto::dimension::Value::DimValue(
                                                3,
                                            ),
                                        ),
                                        denotation: String::new(),
                                    },
                                ],
                            }),
                        },
                    )),
                    denotation: String::new(),
                }),
                ..Default::default()
            }],
            // Weight tensor as initializer
            initializer: vec![proto::TensorProto {
                dims: vec![3, 4],
                data_type: 1, // FLOAT
                name: "weight".to_string(),
                // 12 floats (3x4 matrix) all set to 1.0
                float_data: vec![1.0; 12],
                ..Default::default()
            }],
            node: vec![proto::NodeProto {
                input: vec!["input".to_string(), "weight".to_string()],
                output: vec!["output".to_string()],
                name: "matmul_node".to_string(),
                op_type: "MatMul".to_string(),
                ..Default::default()
            }],
            output: vec![proto::ValueInfoProto {
                name: "output".to_string(),
                r#type: Some(proto::TypeProto {
                    value: Some(proto::type_proto::Value::TensorType(
                        proto::type_proto::Tensor {
                            elem_type: 1, // FLOAT
                            shape: Some(proto::TensorShapeProto {
                                dim: vec![
                                    proto::tensor_shape_proto::Dimension {
                                        value: Some(
                                            proto::tensor_shape_proto::dimension::Value::DimValue(
                                                2,
                                            ),
                                        ),
                                        denotation: String::new(),
                                    },
                                    proto::tensor_shape_proto::Dimension {
                                        value: Some(
                                            proto::tensor_shape_proto::dimension::Value::DimValue(
                                                4,
                                            ),
                                        ),
                                        denotation: String::new(),
                                    },
                                ],
                            }),
                        },
                    )),
                    denotation: String::new(),
                }),
                ..Default::default()
            }],
            ..Default::default()
        }),
        producer_name: "hologram-ai-onnx-test".to_string(),
        producer_version: "0.1.0".to_string(),
        ..Default::default()
    };

    let mut onnx_bytes = Vec::new();
    model
        .encode(&mut onnx_bytes)
        .expect("Failed to encode model");

    // Compile it
    let result = hologram_ai_onnx::compile_onnx(&onnx_bytes);
    assert!(result.is_ok(), "Compilation failed: {:?}", result.err());

    let holb_bytes = result.unwrap();
    assert!(!holb_bytes.is_empty());
    assert_eq!(&holb_bytes[0..4], b"HOLB");

    println!("✅ Successfully compiled MatMul model");
    println!("   ONNX size: {} bytes", onnx_bytes.len());
    println!("   HOLB size: {} bytes", holb_bytes.len());
}
