//! Advanced ONNX compilation with custom configuration.
//!
//! This example demonstrates:
//! - Custom compilation configuration
//! - Graph partitioning for large models
//! - Weight thresholding
//!
//! Run with: `cargo run --example advanced_compilation`

use hologram_onnx::{OnnxCompiler, OnnxConfig, proto::*};
use prost::Message;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a more complex model with convolution and activation
    let graph = GraphProto {
        name: "conv_network".to_string(),
        input: vec![
            make_value_info("input", vec![1, 3, 224, 224], 1),
            make_value_info("conv_weight", vec![64, 3, 7, 7], 1),
        ],
        output: vec![make_value_info("output", vec![1, 64, 218, 218], 1)],
        node: vec![
            NodeProto {
                input: vec!["input".to_string(), "conv_weight".to_string()],
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

    println!("=== Advanced ONNX Compilation ===\n");

    // Configure custom compilation settings
    let config = OnnxConfig {
        weight_threshold: 1024,          // Smaller threshold for external weights
        enable_partitioning: true,        // Enable for large models
        partition_size: 500,              // Nodes per partition
        decompose_conv2d: true,           // Conv2D → Im2col+GEMM
        decompose_pooling: true,          // Pooling decomposition
        pack_weights: true,               // Pack weights for faster runtime
        memory_budget: Some(16 * 1024),   // 16 GB memory limit
        enable_resize_upscaling: true,    // Enable Resize upscaling
    };

    println!("Configuration:");
    println!("  - Weight threshold: {} bytes", config.weight_threshold);
    println!("  - Partitioning: {}", if config.enable_partitioning { "enabled" } else { "disabled" });
    println!("  - Partition size: {} nodes", config.partition_size);
    println!("  - Conv2D decomposition: {}", config.decompose_conv2d);
    println!("  - Memory budget: {} MB\n", config.memory_budget.unwrap_or(0));

    // Create compiler with custom config
    let compiler = OnnxCompiler::with_config(config);

    println!("Compiling...");
    let (holo_bytes, weight_bytes) = compiler.compile(&onnx_bytes)?;

    println!("✓ Compilation successful!");
    println!("  - Holo file: {} bytes", holo_bytes.len());
    println!("  - Weights file: {} bytes", weight_bytes.len());

    // Write files with custom names
    let model_name = "conv_network";
    std::fs::write(format!("{}.holo", model_name), &holo_bytes)?;
    if !weight_bytes.is_empty() {
        std::fs::write(format!("{}.weights", model_name), &weight_bytes)?;
    }

    println!("\n✓ Files written:");
    println!("  - {}.holo", model_name);
    if !weight_bytes.is_empty() {
        println!("  - {}.weights", model_name);
    }

    Ok(())
}

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

fn make_ints_attr(name: &str, values: Vec<i64>) -> AttributeProto {
    use attribute_proto::AttributeType;

    AttributeProto {
        name: name.to_string(),
        ints: values,
        r#type: AttributeType::Ints as i32,
        ..Default::default()
    }
}
