//! Compile ONNX models to .holo format.
//!
//! This module provides the `compile` command which:
//! - Loads an ONNX model from disk
//! - Translates ONNX → hologram IR using the full translation pipeline
//! - Applies decomposition pass (Conv2D → Im2col+GEMM)
//! - Serializes the IR to .holo format for execution
//! - Writes the resulting .holo and .weights files

use anyhow::{Context, Result};
use hologram_onnx_core::{
    OnnxConfig, extract_opset_version, parse_model, validate_model,
    serialize_ir_function,
};
use std::fs;
use std::path::Path;
use tracing::{debug, info};

use crate::translator::{apply_ir_decomposition, translate_graph_to_ir};

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
///
/// # Returns
///
/// Returns Ok(()) on success, or an error if compilation fails.
pub fn compile_command(
    input: &Path,
    output: &Path,
    partition: bool,
    partition_size: usize,
    memory_budget: Option<usize>,
    weight_threshold: usize,
    decompose_conv2d: bool,
    decompose_pooling: bool,
) -> Result<()> {
    info!("Compiling ONNX model: {}", input.display());
    debug!("Output path: {}", output.display());
    debug!("Partitioning: {}", partition);
    debug!("Partition size: {}", partition_size);
    debug!("Weight threshold: {} bytes", weight_threshold);
    debug!("Decompose Conv2D: {}", decompose_conv2d);
    debug!("Decompose Pooling: {}", decompose_pooling);

    // Read ONNX model
    info!("Reading ONNX model...");
    let onnx_bytes = fs::read(input)
        .with_context(|| format!("Failed to read ONNX model from {}", input.display()))?;

    info!("ONNX model size: {} bytes", onnx_bytes.len());

    // Create configuration
    let config = OnnxConfig {
        weight_threshold,
        enable_partitioning: partition,
        partition_size,
        decompose_conv2d,
        decompose_pooling,
        pack_weights: true,
        memory_budget,
    };

    config
        .validate()
        .map_err(|e| anyhow::anyhow!("Invalid configuration: {}", e))?;

    // Step 1: Parse and validate ONNX model
    info!("Parsing ONNX protobuf...");
    let model = parse_model(&onnx_bytes).context("Failed to parse ONNX model")?;

    validate_model(&model).context("Model validation failed")?;

    let opset_version = extract_opset_version(&model);
    info!("ONNX opset version: {}", opset_version);

    // Get the graph
    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Model has no graph"))?;

    info!("Graph: {} ({} nodes)", graph.name, graph.node.len());

    // Step 2: Translate ONNX → IR with symbolic shapes
    info!("Translating ONNX → IR...");
    let ir_func =
        translate_graph_to_ir(graph, opset_version).context("Failed to translate ONNX to IR")?;

    // Step 3: Apply decomposition pass
    info!("Applying decomposition pass...");
    let decomposed = apply_ir_decomposition(ir_func, &config).context("Decomposition failed")?;

    info!("Decomposition complete: {} IR nodes", decomposed.body.len());

    // Step 4: Serialize IR to .holo format
    info!("Serializing to .holo format...");
    let (holo_bytes, weights_bytes) =
        serialize_ir_function(&decomposed, weight_threshold, config.pack_weights)
        .map_err(|e| anyhow::anyhow!("Serialization failed: {}", e))?;

    info!("Compilation successful!");
    info!("  .holo size: {} bytes", holo_bytes.len());
    info!("  .weights size: {} bytes", weights_bytes.len());

    // Write .holo file
    let holo_path = output.with_extension("holo");
    info!("Writing .holo file: {}", holo_path.display());
    fs::write(&holo_path, &holo_bytes)
        .with_context(|| format!("Failed to write .holo file to {}", holo_path.display()))?;

    // Write .weights file if there are external weights
    if !weights_bytes.is_empty() {
        let weights_path = output.with_extension("weights");
        info!("Writing .weights file: {}", weights_path.display());
        fs::write(&weights_path, &weights_bytes)
            .with_context(|| format!("Failed to write .weights file to {}", weights_path.display()))?;
        info!("  Output: {}.weights", output.display());
    }

    info!("✓ Compilation complete!");
    info!("  Output: {}.holo", output.display());

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

        let result = compile_command(&input, &output, false, 500, None, 4096, true, true);
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
