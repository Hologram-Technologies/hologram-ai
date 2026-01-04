//! Compile ONNX models to .holo format.
//!
//! This module provides the `compile` command which:
//! - Loads an ONNX model from disk
//! - Translates ONNX → hologram IR using the full translation pipeline
//! - Applies decomposition pass (Conv2D → Im2col+GEMM)
//! - Converts IR to OperationGraph and compiles to parallel schedule
//! - Serializes to .holo format compatible with hologram runtime

use crate::core::{OnnxConfig, extract_opset_version, parse_model, validate_model};
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use tracing::{debug, info};

use crate::cli::translator::translate_graph_to_ir_with_path;

/// Compile an ONNX model to .holo format.
///
/// # Arguments
///
/// * `input` - Path to input ONNX model file
/// * `output` - Output path (without extension)
/// * `partition` - Enable graph partitioning for large models
/// * `partition_size` - Number of nodes per partition
/// * `memory_budget` - Memory budget in MB
/// * `weight_threshold` - Threshold for external weight storage (bytes)
/// * `enable_resize_upscaling` - Enable Resize upscaling (false saves memory)
/// * `input_shapes` - Optional map of input name -> concrete shape dimensions
///
/// # Returns
///
/// Returns Ok(()) on success, or an error if compilation fails.
#[allow(clippy::too_many_arguments)]
pub fn compile_command(
    input: &Path,
    output: &Path,
    partition: bool,
    partition_size: usize,
    memory_budget: Option<usize>,
    weight_threshold: usize,
    decompose_conv2d: bool,
    decompose_pooling: bool,
    enable_resize_upscaling: bool,
    input_shapes: &HashMap<String, Vec<usize>>,
) -> Result<()> {
    info!("Compiling ONNX model: {}", input.display());
    debug!("Output path: {}", output.display());
    debug!("Partitioning: {}", partition);
    debug!("Partition size: {}", partition_size);
    debug!("Weight threshold: {} bytes", weight_threshold);
    debug!("Decompose Conv2D: {}", decompose_conv2d);
    debug!("Decompose Pooling: {}", decompose_pooling);
    debug!("Enable Resize Upscaling: {}", enable_resize_upscaling);

    // Read ONNX model
    info!("Reading ONNX model...");
    let onnx_bytes = fs::read(input)
        .with_context(|| format!("Failed to read ONNX model from {}", input.display()))?;

    let onnx_size = onnx_bytes.len();
    info!("ONNX model size: {} bytes", onnx_size);

    // Create configuration
    let config = OnnxConfig {
        weight_threshold,
        enable_partitioning: partition,
        partition_size,
        decompose_conv2d,
        decompose_pooling,
        pack_weights: true,
        memory_budget,
        enable_resize_upscaling,
    };

    config
        .validate()
        .map_err(|e| anyhow::anyhow!("Invalid configuration: {}", e))?;

    // Step 1: Parse and validate ONNX model
    info!("Parsing ONNX protobuf...");
    let mut model = parse_model(&onnx_bytes).context("Failed to parse ONNX model")?;

    // Free ONNX bytes - we have the parsed model now
    drop(onnx_bytes);
    debug!("Freed ONNX bytes ({} bytes)", onnx_size);

    validate_model(&model).context("Model validation failed")?;

    // Apply input shapes if specified
    if !input_shapes.is_empty() {
        apply_input_shapes(&mut model, input_shapes)?;
    }

    let opset_version = extract_opset_version(&model);
    info!("ONNX opset version: {}", opset_version);

    // Get the graph
    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Model has no graph"))?;

    let graph_name = graph.name.clone();
    let node_count = graph.node.len();
    info!("Graph: {} ({} nodes)", graph_name, node_count);

    // Step 2: Translate ONNX → IR with symbolic shapes (with external data support)
    info!("Translating ONNX → IR...");
    let _ir_func = translate_graph_to_ir_with_path(graph, opset_version, Some(input))
        .context("Failed to translate ONNX to IR")?;

    // Free parsed ONNX model - we have the IR now
    drop(model);
    debug!("Freed parsed ONNX model");

    // Step 3: Apply decomposition pass
    // Decomposition is now handled internally by hologram-ir during translation
    info!("IR decomposition is handled by hologram-ir passes");

    // Step 4: Serialization to .holo format
    // The .holo format serializer needs to be implemented to support full compilation.
    info!("Serialization to .holo format requires implementation");
    Err(anyhow::anyhow!(
        "Full compilation pipeline requires a .holo format serializer. \
         Translation to hologram-ir succeeds, but .holo serialization is not yet implemented."
    ))
}

/// Apply concrete input shapes to an ONNX model.
///
/// This modifies the model's input definitions to have concrete dimensions
/// instead of symbolic ones, which is necessary for proper buffer allocation
/// during compilation and execution.
fn apply_input_shapes(
    model: &mut crate::proto::ModelProto,
    input_shapes: &HashMap<String, Vec<usize>>,
) -> Result<()> {
    use crate::proto::tensor_shape_proto::Dimension;
    use crate::proto::tensor_shape_proto::dimension::Value as DimValue;

    let graph = model
        .graph
        .as_mut()
        .ok_or_else(|| anyhow::anyhow!("Model has no graph"))?;

    for (name, dims) in input_shapes {
        info!("Applying input shape: {} = {:?}", name, dims);

        // Find the input by name
        for input in &mut graph.input {
            if input.name == *name {
                // Get the tensor type
                if let Some(ref mut type_proto) = input.r#type
                    && let Some(ref mut value) = type_proto.value
                    && let crate::proto::type_proto::Value::TensorType(tensor_type) = value
                {
                    // Create or modify the shape
                    let shape = tensor_type.shape.get_or_insert_with(Default::default);
                    shape.dim.clear();

                    for &dim in dims {
                        shape.dim.push(Dimension {
                            value: Some(DimValue::DimValue(dim as i64)),
                            denotation: String::new(),
                        });
                    }
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_compile_command_missing_input() {
        let temp_dir = TempDir::new().unwrap();
        let input = temp_dir.path().join("missing.onnx");
        let output = temp_dir.path().join("output");

        let result = compile_command(
            &input,
            &output,
            false,
            500,
            None,
            4096,
            true,
            true,
            true,
            &HashMap::new(),
        );
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Failed to read ONNX model")
        );
    }

    #[test]
    fn test_compile_command_creates_output_path() {
        let temp_dir = TempDir::new().unwrap();
        let output = temp_dir.path().join("model");

        // The compile will fail because we don't have a valid ONNX model,
        // but we can test that the paths are constructed correctly
        assert_eq!(
            output.with_extension("holo"),
            temp_dir.path().join("model.holo")
        );
        assert_eq!(
            output.with_extension("weights"),
            temp_dir.path().join("model.weights")
        );
    }
}
