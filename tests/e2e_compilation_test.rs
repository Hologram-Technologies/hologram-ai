//! End-to-end compilation tests.
//!
//! These tests verify the complete ONNX → IR → .holo pipeline.

use hologram_onnx::{OnnxCompiler, compile_onnx};
use hologram_onnx::proto::{GraphProto, NodeProto, ValueInfoProto, TypeProto};
use hologram_onnx::proto::type_proto::{Value, Tensor};
use hologram_onnx::proto::tensor_shape_proto::Dimension;
use hologram_onnx::proto::TensorShapeProto;
use hologram_onnx::proto::attribute_proto::AttributeType;
use hologram_onnx::proto::AttributeProto;
use hologram_onnx::proto::ModelProto;
use prost::Message;

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

fn make_int_attr(name: &str, value: i64) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        i: value,
        r#type: AttributeType::Int as i32,
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

/// Test compilation of a simple Add model.
#[test]
fn test_compile_simple_add() {
    let graph = GraphProto {
        name: "simple_add".to_string(),
        input: vec![
            make_value_info("input1", vec![2, 3], 1), // FLOAT
            make_value_info("input2", vec![2, 3], 1),
        ],
        output: vec![
            make_value_info("output", vec![2, 3], 1),
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

    let model = ModelProto {
        ir_version: 8,
        opset_import: vec![],
        graph: Some(graph),
        ..Default::default()
    };

    // Serialize the ONNX model to bytes
    let mut onnx_bytes = Vec::new();
    model.encode(&mut onnx_bytes).expect("Failed to encode ONNX model");

    // Compile using the convenience function
    let result = compile_onnx(&onnx_bytes);
    assert!(result.is_ok(), "Compilation failed: {:?}", result.err());

    let (holo_bytes, _weight_bytes) = result.unwrap();
    assert!(!holo_bytes.is_empty(), "Holo bytes should not be empty");
    // Weights may be empty for this simple model
}

/// Test compilation using OnnxCompiler.
#[test]
fn test_compile_with_compiler() {
    let graph = GraphProto {
        name: "matmul".to_string(),
        input: vec![
            make_value_info("input", vec![2, 4], 1),
            make_value_info("weight", vec![4, 8], 1),
        ],
        output: vec![
            make_value_info("output", vec![2, 8], 1),
        ],
        node: vec![
            NodeProto {
                input: vec!["input".to_string(), "weight".to_string()],
                output: vec!["output".to_string()],
                op_type: "MatMul".to_string(),
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
    model.encode(&mut onnx_bytes).expect("Failed to encode ONNX model");

    // Compile using OnnxCompiler
    let compiler = OnnxCompiler::new();
    let result = compiler.compile(&onnx_bytes);
    assert!(result.is_ok(), "Compilation failed: {:?}", result.err());

    let (holo_bytes, _weight_bytes) = result.unwrap();
    assert!(!holo_bytes.is_empty(), "Holo bytes should not be empty");
}

/// Test compilation of a Conv+ReLU model.
#[test]
fn test_compile_conv_relu() {
    let graph = GraphProto {
        name: "conv_relu".to_string(),
        input: vec![
            make_value_info("input", vec![1, 3, 32, 32], 1),
            make_value_info("weight", vec![16, 3, 3, 3], 1),
        ],
        output: vec![
            make_value_info("output", vec![1, 16, 30, 30], 1),
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

    let model = ModelProto {
        ir_version: 8,
        opset_import: vec![],
        graph: Some(graph),
        ..Default::default()
    };

    let mut onnx_bytes = Vec::new();
    model.encode(&mut onnx_bytes).expect("Failed to encode ONNX model");

    let result = compile_onnx(&onnx_bytes);
    assert!(result.is_ok(), "Compilation failed: {:?}", result.err());

    let (holo_bytes, _weight_bytes) = result.unwrap();
    assert!(!holo_bytes.is_empty(), "Holo bytes should not be empty");
}

/// Test compilation of LayerNorm model.
#[test]
fn test_compile_layernorm() {
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

    let model = ModelProto {
        ir_version: 8,
        opset_import: vec![],
        graph: Some(graph),
        ..Default::default()
    };

    let mut onnx_bytes = Vec::new();
    model.encode(&mut onnx_bytes).expect("Failed to encode ONNX model");

    let result = compile_onnx(&onnx_bytes);
    assert!(result.is_ok(), "Compilation failed: {:?}", result.err());

    let (holo_bytes, _weight_bytes) = result.unwrap();
    assert!(!holo_bytes.is_empty(), "Holo bytes should not be empty");
}

/// Test deserialization of compiled output.
#[test]
fn test_compile_and_deserialize() {
    let graph = GraphProto {
        name: "simple".to_string(),
        input: vec![
            make_value_info("x", vec![4], 1),
        ],
        output: vec![
            make_value_info("y", vec![4], 1),
        ],
        node: vec![
            NodeProto {
                input: vec!["x".to_string()],
                output: vec!["y".to_string()],
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
    model.encode(&mut onnx_bytes).expect("Failed to encode ONNX model");

    // Compile
    let (holo_bytes, _) = compile_onnx(&onnx_bytes).expect("Compilation failed");

    // Verify output format - .holo files start with HOLO_MAGIC
    use hologram::compiler::HOLO_MAGIC;
    assert!(holo_bytes.len() >= 12, "Holo bytes too short");
    assert_eq!(&holo_bytes[0..4], &HOLO_MAGIC, "Invalid magic bytes");
    let version = u32::from_le_bytes([holo_bytes[4], holo_bytes[5], holo_bytes[6], holo_bytes[7]]);
    assert_eq!(
        version,
        hologram::backend::plan::PLAN_FORMAT_VERSION,
        "Unexpected plan format version"
    );
    let header_len = u32::from_le_bytes([holo_bytes[8], holo_bytes[9], holo_bytes[10], holo_bytes[11]]);
    assert!(header_len > 0, "Missing layer header");
}
