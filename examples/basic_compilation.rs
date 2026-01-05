//! Basic ONNX compilation example.
//!
//! This example demonstrates how to compile a simple ONNX model to .holo format.
//!
//! Run with: `cargo run --example basic_compilation`

use hologram_onnx::{compile_onnx, proto::*};
use prost::Message;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a simple ONNX model: output = input1 + input2
    let graph = GraphProto {
        name: "simple_add".to_string(),
        input: vec![
            make_value_info("input1", vec![2, 3], 1), // FLOAT
            make_value_info("input2", vec![2, 3], 1),
        ],
        output: vec![make_value_info("output", vec![2, 3], 1)],
        node: vec![NodeProto {
            input: vec!["input1".to_string(), "input2".to_string()],
            output: vec!["output".to_string()],
            op_type: "Add".to_string(),
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

    // Serialize ONNX model to bytes
    let mut onnx_bytes = Vec::new();
    model.encode(&mut onnx_bytes)?;

    println!("Compiling ONNX model ({} bytes)...", onnx_bytes.len());

    // Compile to .holo format
    let (holo_bytes, weight_bytes) = compile_onnx(&onnx_bytes)?;

    println!("✓ Compilation successful!");
    println!("  - Holo file: {} bytes", holo_bytes.len());
    println!("  - Weights file: {} bytes", weight_bytes.len());

    // Optionally write to disk
    std::fs::write("model.holo", &holo_bytes)?;
    if !weight_bytes.is_empty() {
        std::fs::write("model.weights", &weight_bytes)?;
    }

    println!("✓ Files written to disk");

    Ok(())
}

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
                            value: Some(
                                tensor_shape_proto::dimension::Value::DimValue(d),
                            ),
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
