//! Compile ONNX models to .holo format.
//!
//! This module provides the `compile` command which:
//! - Loads an ONNX model from disk
//! - Compiles it using hologram-onnx-core (with hologram-compiler)
//! - Writes the resulting .holo and .weights files

use anyhow::{Context, Result};
use hologram_onnx_core::{OnnxCompiler, OnnxConfig};
use std::fs;
use std::path::Path;
use tracing::{info, debug};

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
///
/// # Example
///
/// ```no_run
/// use std::path::Path;
/// # use anyhow::Result;
/// # fn main() -> Result<()> {
/// hologram_onnx_cli::compile::compile_command(
///     Path::new("model.onnx"),
///     Path::new("model"),
///     false,
///     500,
///     None,
///     4096,
/// )?;
/// # Ok(())
/// # }
/// ```
pub fn compile_command(
    input: &Path,
    output: &Path,
    partition: bool,
    partition_size: usize,
    memory_budget: Option<usize>,
    weight_threshold: usize,
) -> Result<()> {
    info!("Compiling ONNX model: {}", input.display());
    debug!("Output path: {}", output.display());
    debug!("Partitioning: {}", partition);
    debug!("Partition size: {}", partition_size);
    debug!("Weight threshold: {} bytes", weight_threshold);

    // Read ONNX model
    info!("Reading ONNX model...");
    let onnx_bytes = fs::read(input)
        .with_context(|| format!("Failed to read ONNX model from {}", input.display()))?;

    info!("ONNX model size: {} bytes", onnx_bytes.len());

    // Create compiler with configuration
    let config = OnnxConfig {
        weight_threshold,
        enable_partitioning: partition,
        partition_size,
        decompose_conv2d: true,
        decompose_pooling: true,
        memory_budget,
    };

    config.validate()
        .map_err(|e| anyhow::anyhow!("Invalid configuration: {}", e))?;

    let compiler = OnnxCompiler::with_config(config);

    // Compile
    info!("Compiling ONNX → .holo format...");
    let (holo_bytes, weight_bytes) = compiler.compile(&onnx_bytes)
        .context("Failed to compile ONNX model")?;

    info!("Compilation successful!");
    info!("  .holo size: {} bytes", holo_bytes.len());
    info!("  .weights size: {} bytes", weight_bytes.len());

    // Write .holo file
    let holo_path = output.with_extension("holo");
    info!("Writing .holo file: {}", holo_path.display());
    fs::write(&holo_path, holo_bytes)
        .with_context(|| format!("Failed to write .holo file to {}", holo_path.display()))?;

    // Write .weights file if non-empty
    let has_weights = !weight_bytes.is_empty();
    if has_weights {
        let weights_path = output.with_extension("weights");
        info!("Writing .weights file: {}", weights_path.display());
        fs::write(&weights_path, &weight_bytes)
            .with_context(|| format!("Failed to write .weights file to {}", weights_path.display()))?;
    } else {
        info!("No external weights (all weights embedded in .holo file)");
    }

    info!("✓ Compilation complete!");
    info!("  Output: {}.holo", output.display());
    if has_weights {
        info!("          {}.weights", output.display());
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

        let result = compile_command(&input, &output, false, 500, None, 4096);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Failed to read ONNX model"));
    }

    #[test]
    fn test_compile_command_creates_output_path() {
        let temp_dir = TempDir::new().unwrap();
        let output = temp_dir.path().join("model");

        // The compile will fail because we don't have a valid ONNX model,
        // but we can test that the paths are constructed correctly
        assert_eq!(output.with_extension("holo"), temp_dir.path().join("model.holo"));
        assert_eq!(output.with_extension("weights"), temp_dir.path().join("model.weights"));
    }
}
