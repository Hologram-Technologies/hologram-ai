//! Display ONNX and HOLO model information.
//!
//! This module provides functionality to inspect ONNX and .holo models and display
//! their structure, inputs, outputs, and operations.

use crate::core::{extract_opset_version, parse_model};
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use tracing::{debug, info};

/// Display information about an ONNX or .holo model.
///
/// # Arguments
///
/// * `model_path` - Path to ONNX or .holo model file
/// * `detailed` - Show detailed operation list
///
/// # Returns
///
/// Returns Ok(()) on success, or an error if the model cannot be read/parsed.
pub fn info_command(model_path: &Path, detailed: bool) -> Result<()> {
    // Check file extension to determine format
    if let Some(ext) = model_path.extension()
        && ext == "holo"
    {
        return info_holo_command(model_path);
    }

    info!("Reading ONNX model: {}", model_path.display());

    // Read model file
    let onnx_bytes = fs::read(model_path)
        .with_context(|| format!("Failed to read ONNX model from {}", model_path.display()))?;

    info!("Model file size: {} bytes", onnx_bytes.len());

    // Parse model
    debug!("Parsing ONNX protobuf...");
    let model = parse_model(&onnx_bytes).context("Failed to parse ONNX model")?;

    // Extract metadata
    let opset_version = extract_opset_version(&model);
    let graph = model.graph.as_ref().context("Model has no graph")?;

    // Display model information
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║              ONNX Model Information                        ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    println!("\n📄 Model Metadata:");
    if model.model_version > 0 {
        println!("  Version: {}", model.model_version);
    }
    if !model.producer_name.is_empty() {
        println!("  Producer: {}", model.producer_name);
    }
    if !model.producer_version.is_empty() {
        println!("  Producer Version: {}", model.producer_version);
    }
    if !model.doc_string.is_empty() {
        println!("  Description: {}", model.doc_string);
    }
    println!("  Opset Version: {}", opset_version);

    // Graph information
    println!("\n📊 Graph: {}", graph.name);
    println!("  Nodes: {}", graph.node.len());
    println!("  Inputs: {}", graph.input.len());
    println!("  Outputs: {}", graph.output.len());
    println!("  Initializers: {}", graph.initializer.len());

    // Display inputs
    if !graph.input.is_empty() {
        println!("\n📥 Inputs:");
        for input in &graph.input {
            let shape_str = get_tensor_shape_string(input);
            let type_str = get_tensor_type_string(input);
            println!("  - {} : {} {}", input.name, type_str, shape_str);
        }
    }

    // Display outputs
    if !graph.output.is_empty() {
        println!("\n📤 Outputs:");
        for output in &graph.output {
            let shape_str = get_tensor_shape_string(output);
            let type_str = get_tensor_type_string(output);
            println!("  - {} : {} {}", output.name, type_str, shape_str);
        }
    }

    // Display operation statistics
    if !graph.node.is_empty() {
        println!("\n⚙️  Operations:");
        let mut op_counts: HashMap<String, usize> = HashMap::new();
        for node in &graph.node {
            *op_counts.entry(node.op_type.clone()).or_insert(0) += 1;
        }

        let mut ops: Vec<_> = op_counts.iter().collect();
        ops.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));

        for (op_type, count) in ops {
            println!("  - {:<20} : {} node(s)", op_type, count);
        }
    }

    // Display detailed node list if requested
    if detailed && !graph.node.is_empty() {
        println!("\n📋 Detailed Node List:");
        for (i, node) in graph.node.iter().enumerate() {
            println!("\n  Node {}: {} ({})", i, node.name, node.op_type);
            if !node.input.is_empty() {
                println!("    Inputs: {}", node.input.join(", "));
            }
            if !node.output.is_empty() {
                println!("    Outputs: {}", node.output.join(", "));
            }
            if !node.attribute.is_empty() {
                println!("    Attributes: {} attribute(s)", node.attribute.len());
            }
        }
    }

    println!("\n");
    Ok(())
}

/// Get tensor shape as a string
fn get_tensor_shape_string(value_info: &crate::proto::ValueInfoProto) -> String {
    use crate::proto::type_proto::Value;

    if let Some(type_proto) = &value_info.r#type
        && let Some(Value::TensorType(tensor_type)) = &type_proto.value
        && let Some(shape) = &tensor_type.shape
    {
        let dims: Vec<String> = shape
            .dim
            .iter()
            .map(|d| {
                if let Some(value) = &d.value {
                    use crate::proto::tensor_shape_proto::dimension::Value as DimValue;
                    match value {
                        DimValue::DimValue(v) => v.to_string(),
                        DimValue::DimParam(p) => p.clone(),
                    }
                } else {
                    "?".to_string()
                }
            })
            .collect();
        return format!("[{}]", dims.join(", "));
    }
    "[]".to_string()
}

/// Get tensor data type as a string
fn get_tensor_type_string(value_info: &crate::proto::ValueInfoProto) -> String {
    use crate::proto::type_proto::Value;

    if let Some(type_proto) = &value_info.r#type
        && let Some(value) = &type_proto.value
    {
        return match value {
            Value::TensorType(tensor_type) => {
                use crate::proto::tensor_proto::DataType;
                match DataType::try_from(tensor_type.elem_type) {
                    Ok(DataType::Undefined) => "undefined",
                    Ok(DataType::Float) => "float32",
                    Ok(DataType::Uint8) => "uint8",
                    Ok(DataType::Int8) => "int8",
                    Ok(DataType::Uint16) => "uint16",
                    Ok(DataType::Int16) => "int16",
                    Ok(DataType::Int32) => "int32",
                    Ok(DataType::Int64) => "int64",
                    Ok(DataType::String) => "string",
                    Ok(DataType::Bool) => "bool",
                    Ok(DataType::Float16) => "float16",
                    Ok(DataType::Double) => "float64",
                    Ok(DataType::Uint32) => "uint32",
                    Ok(DataType::Uint64) => "uint64",
                    Ok(DataType::Complex64) => "complex64",
                    Ok(DataType::Complex128) => "complex128",
                    Ok(DataType::Bfloat16) => "bfloat16",
                    _ => "unknown",
                }
                .to_string()
            }
            _ => "tensor".to_string(),
        };
    }
    "tensor".to_string()
}

/// Display information about a compiled .holo model.
fn info_holo_command(model_path: &Path) -> Result<()> {
    use hologram_ir::OperationGraph;

    info!("Reading HOLO model: {}", model_path.display());

    // Read and parse the rkyv-serialized .holo file
    let holo_bytes = fs::read(model_path)
        .with_context(|| format!("Failed to read HOLO model from {}", model_path.display()))?;

    let graph =
        OperationGraph::from_bytes(&holo_bytes).with_context(|| "Failed to parse .holo file")?;

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║              HOLO Model Information                        ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    println!("\n📄 Model Metadata:");
    println!("  File: {}", model_path.display());
    println!("  Size: {} bytes", holo_bytes.len());
    println!("  Format: OperationGraph (rkyv binary)");

    // Graph introspection requires accessing the deserialized structure
    println!("\n📊 Graph:");
    println!("  Nodes: (introspection requires implementation)");
    println!("\n⚙️  Operations:");
    println!("  (operation details require implementation)");
    println!("\n📥 Inputs:");
    println!("  (input introspection requires implementation)");
    println!("\n📤 Outputs:");
    println!("  (output introspection requires implementation)");

    // Stub to avoid unused variable warning
    let _ = graph;

    println!("\n");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_info_command_missing_file() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("missing.onnx");

        let result = info_command(&model_path, false);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Failed to read ONNX model")
        );
    }
}
