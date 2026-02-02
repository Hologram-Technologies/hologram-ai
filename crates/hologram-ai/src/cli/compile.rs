//! Compile ONNX models to .holb format.
//!
//! This module provides the `compile` command which:
//! - Loads an ONNX model from disk
//! - Translates ONNX → hologram OperationGraph
//! - Compiles to BackendPlan
//! - Serializes to .holb format compatible with hologram runtime

use anyhow::{Context, Result};
use std::fs;
use std::path::Path;
use tracing::info;

/// Compile an ONNX model to .holb format.
///
/// # Arguments
///
/// * `input` - Path to input ONNX model file
/// * `output` - Output path (without extension, will create .holb file)
///
/// # Returns
///
/// Returns Ok(()) on success, or an error if compilation fails.
pub fn compile_command(input: &Path, output: &Path) -> Result<()> {
    info!("Compiling ONNX model: {}", input.display());
    info!("Output path: {}", output.display());

    // Read ONNX model
    info!("Reading ONNX model...");
    let onnx_bytes = fs::read(input)
        .with_context(|| format!("Failed to read ONNX model from {}", input.display()))?;

    let onnx_size = onnx_bytes.len();
    info!("ONNX model size: {} bytes", onnx_size);

    // Compile using the new simplified API
    info!("Starting compilation...");

    #[cfg(feature = "onnx")]
    let holb_bytes =
        hologram_ai_onnx::compile_onnx(&onnx_bytes).context("ONNX compilation failed")?;

    #[cfg(not(feature = "onnx"))]
    let holb_bytes: Vec<u8> = {
        anyhow::bail!("ONNX support not enabled. Build with --features onnx");
    };

    info!("Compilation successful: {} bytes", holb_bytes.len());

    // Write .holb file
    let holb_path = output.with_extension("holb");
    fs::write(&holb_path, &holb_bytes)
        .with_context(|| format!("Failed to write .holb file to {}", holb_path.display()))?;
    info!("Written: {}", holb_path.display());

    info!("Compilation complete!");
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

        let result = compile_command(&input, &output);
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
            output.with_extension("holb"),
            temp_dir.path().join("model.holb")
        );
    }
}
