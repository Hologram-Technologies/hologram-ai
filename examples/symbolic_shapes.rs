//! Symbolic shapes example - variable batch sizes and sequence lengths.
//!
//! This example demonstrates how hologram-onnx handles symbolic (dynamic) shapes,
//! which is essential for:
//! - Variable batch sizes
//! - Variable sequence lengths (NLP models)
//! - Dynamic input dimensions
//!
//! Run with: `cargo run --example symbolic_shapes`

use hologram_onnx::{compile_onnx, proto::*};
use prost::Message;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Symbolic Shapes Example ===\n");

    // Example 1: Variable batch size
    compile_variable_batch()?;

    // Example 2: Variable sequence length (NLP)
    compile_variable_sequence()?;

    // Example 3: Multiple symbolic dimensions
    compile_multi_symbolic()?;

    Ok(())
}

/// Example 1: Model with variable batch size.
fn compile_variable_batch() -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Variable Batch Size");
    println!("   Input shape: [batch, 3, 224, 224] where batch is symbolic\n");

    let graph = GraphProto {
        name: "resnet_batch".to_string(),
        input: vec![
            make_symbolic_value_info("input", "batch", vec![3, 224, 224], 1),
            make_value_info("weight", vec![64, 3, 7, 7], 1),
        ],
        output: vec![make_symbolic_value_info("output", "batch", vec![64, 218, 218], 1)],
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
    model.encode(&mut onnx_bytes)?;

    let (holo_bytes, _) = compile_onnx(&onnx_bytes)?;
    println!("   ✓ Compiled: {} bytes\n", holo_bytes.len());

    Ok(())
}

/// Example 2: NLP model with variable sequence length.
fn compile_variable_sequence() -> Result<(), Box<dyn std::error::Error>> {
    println!("2. Variable Sequence Length (NLP)");
    println!("   Input shape: [batch, seq_len, 768] where both are symbolic\n");

    let graph = GraphProto {
        name: "bert_encoder".to_string(),
        input: vec![
            make_double_symbolic_value_info("input", vec!["batch", "seq_len", "768"]),
            make_value_info("scale", vec![768], 1),
            make_value_info("bias", vec![768], 1),
        ],
        output: vec![make_double_symbolic_value_info("output", vec!["batch", "seq_len", "768"])],
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

    let model = ModelProto {
        ir_version: 8,
        opset_import: vec![],
        graph: Some(graph),
        ..Default::default()
    };

    let mut onnx_bytes = Vec::new();
    model.encode(&mut onnx_bytes)?;

    let (holo_bytes, _) = compile_onnx(&onnx_bytes)?;
    println!("   ✓ Compiled: {} bytes\n", holo_bytes.len());

    Ok(())
}

/// Example 3: Multiple symbolic dimensions.
fn compile_multi_symbolic() -> Result<(), Box<dyn std::error::Error>> {
    println!("3. Multiple Symbolic Dimensions");
    println!("   Demonstrates: batch, height, width all symbolic\n");

    let graph = GraphProto {
        name: "adaptive_pool".to_string(),
        input: vec![make_triple_symbolic_value_info(
            "input",
            vec!["batch", "channels", "height", "width"],
        )],
        output: vec![make_triple_symbolic_value_info(
            "output",
            vec!["batch", "channels", "pool_h", "pool_w"],
        )],
        node: vec![NodeProto {
            input: vec!["input".to_string()],
            output: vec!["output".to_string()],
            op_type: "GlobalAveragePool".to_string(),
            ..Default::default()
        }],
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
    model.encode(&mut onnx_bytes)?;

    let (holo_bytes, _) = compile_onnx(&onnx_bytes)?;
    println!("   ✓ Compiled: {} bytes\n", holo_bytes.len());

    Ok(())
}

// Helper functions for creating symbolic ValueInfoProto

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

fn make_symbolic_value_info(name: &str, batch_name: &str, static_dims: Vec<i64>, dtype: i32) -> ValueInfoProto {
    use tensor_shape_proto::Dimension;
    use type_proto::{Tensor, Value};

    let mut dims = vec![Dimension {
        value: Some(tensor_shape_proto::dimension::Value::DimParam(batch_name.to_string())),
        ..Default::default()
    }];

    dims.extend(static_dims.iter().map(|&d| Dimension {
        value: Some(tensor_shape_proto::dimension::Value::DimValue(d)),
        ..Default::default()
    }));

    ValueInfoProto {
        name: name.to_string(),
        r#type: Some(TypeProto {
            value: Some(Value::TensorType(Tensor {
                elem_type: dtype,
                shape: Some(TensorShapeProto { dim: dims }),
            })),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn make_double_symbolic_value_info(name: &str, dim_names: Vec<&str>) -> ValueInfoProto {
    use tensor_shape_proto::Dimension;
    use type_proto::{Tensor, Value};

    let dims = dim_names
        .iter()
        .map(|&name| {
            if let Ok(val) = name.parse::<i64>() {
                Dimension {
                    value: Some(tensor_shape_proto::dimension::Value::DimValue(val)),
                    ..Default::default()
                }
            } else {
                Dimension {
                    value: Some(tensor_shape_proto::dimension::Value::DimParam(name.to_string())),
                    ..Default::default()
                }
            }
        })
        .collect();

    ValueInfoProto {
        name: name.to_string(),
        r#type: Some(TypeProto {
            value: Some(Value::TensorType(Tensor {
                elem_type: 1,
                shape: Some(TensorShapeProto { dim: dims }),
            })),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn make_triple_symbolic_value_info(name: &str, dim_names: Vec<&str>) -> ValueInfoProto {
    make_double_symbolic_value_info(name, dim_names)
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
