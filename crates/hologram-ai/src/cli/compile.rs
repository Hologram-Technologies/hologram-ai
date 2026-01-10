//! Compile ONNX models to .holo format.
//!
//! This module provides the `compile` command which:
//! - Loads an ONNX model from disk
//! - Translates ONNX → hologram IR using the full translation pipeline
//! - Applies decomposition pass (Conv2D → Im2col+GEMM)
//! - Converts IR to OperationGraph and compiles to parallel schedule
//! - Serializes to .holo format compatible with hologram runtime

#[cfg(feature = "onnx")]
use hologram_ai_onnx::core::{OnnxConfig, parse_model};
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use tracing::{debug, info};

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
/// * `bundle` - Create a unified bundle with embedded weights (HOLB format)
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
    bundle: bool,
) -> Result<()> {
    info!("Compiling ONNX model: {}", input.display());
    debug!("Output path: {}", output.display());
    debug!("Output format: {}", if bundle { "unified bundle (HOLB)" } else { "separate files (HOLP + weights)" });
    debug!("Partitioning: {}", partition);
    debug!("Partition size: {}", partition_size);
    debug!("Weight threshold: {} bytes", weight_threshold);
    debug!("Decompose Conv2D: {}", decompose_conv2d);
    debug!("Decompose Pooling: {}", decompose_pooling);
    debug!("Enable Resize Upscaling: {}", enable_resize_upscaling);

    // Read ONNX model
    info!("Reading ONNX model...");
    let mut onnx_bytes = fs::read(input)
        .with_context(|| format!("Failed to read ONNX model from {}", input.display()))?;

    let onnx_size = onnx_bytes.len();
    info!("ONNX model size: {} bytes", onnx_size);

    // Apply input shapes if specified (requires modifying the model before compilation)
    if !input_shapes.is_empty() {
        info!("Applying concrete input shapes...");
        let mut model = parse_model(&onnx_bytes).context("Failed to parse ONNX model")?;
        apply_input_shapes(&mut model, input_shapes)?;

        // Re-serialize the modified model
        use prost::Message;
        onnx_bytes.clear();
        model.encode(&mut onnx_bytes)
            .context("Failed to re-encode ONNX model with concrete shapes")?;

        info!("Model updated with concrete input shapes");
    }

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

    // Compile using the OnnxCompiler API
    info!("Starting compilation pipeline...");
    let compiler = hologram_ai_onnx::OnnxCompiler::with_config(config);

    if bundle {
        // Compile to unified bundle (HOLB format)
        let bundle_bytes = compiler
            .compile_to_bundle(&onnx_bytes)
            .context("ONNX compilation to bundle failed")?;

        info!("Compilation successful: {} bytes unified bundle", bundle_bytes.len());

        // Write .holo bundle file
        let holo_path = output.with_extension("holo");
        fs::write(&holo_path, &bundle_bytes)
            .with_context(|| format!("Failed to write bundle to {}", holo_path.display()))?;
        info!("Written: {} (unified bundle)", holo_path.display());
    } else {
        // Compile to separate files (HOLP + weights)
        let (holo_bytes, weight_bytes) = compiler
            .compile(&onnx_bytes)
            .context("ONNX compilation failed")?;

        info!(
            "Compilation successful: {} bytes .holo, {} bytes .weights",
            holo_bytes.len(),
            weight_bytes.len()
        );

        // Write .holo file
        let holo_path = output.with_extension("holo");
        fs::write(&holo_path, &holo_bytes)
            .with_context(|| format!("Failed to write .holo file to {}", holo_path.display()))?;
        info!("Written: {}", holo_path.display());

        // Write .weights file if not empty
        if !weight_bytes.is_empty() {
            let weights_path = output.with_extension("weights");
            fs::write(&weights_path, &weight_bytes)
                .with_context(|| format!("Failed to write .weights file to {}", weights_path.display()))?;
            info!("Written: {}", weights_path.display());
        } else {
            debug!("No external weights to write");
        }
    }

    info!("Compilation complete!");
    Ok(())
}

/// Apply concrete input shapes to an ONNX model.
///
/// This modifies the model's input definitions to have concrete dimensions
/// instead of symbolic ones, which is necessary for proper buffer allocation
/// during compilation and execution.
fn apply_input_shapes(
    model: &mut hologram_ai_onnx::proto::ModelProto,
    input_shapes: &HashMap<String, Vec<usize>>,
) -> Result<()> {
    use hologram_ai_onnx::proto::tensor_shape_proto::Dimension;
    use hologram_ai_onnx::proto::tensor_shape_proto::dimension::Value as DimValue;

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
                    && let hologram_ai_onnx::proto::type_proto::Value::TensorType(tensor_type) = value
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
            false, // bundle
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
