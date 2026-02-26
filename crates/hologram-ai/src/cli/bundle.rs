//! Bundle multiple .holb files into a single distributable file.
//!
//! Uses hologram-holo's HOLM format for pipeline bundles.

use anyhow::{Context, Result};
use std::fs;
use std::path::Path;
use tracing::{info, warn};

use hologram::holo::pipeline::HolmWriter;

/// Bundle multiple .holo files into a single bundle.
///
/// STUB: This function is temporarily disabled during API migration.
pub fn bundle_command(
    _inputs: &[std::path::PathBuf],
    _output: &Path,
    _names: Option<&[String]>,
) -> Result<()> {
    warn!("Bundle command is temporarily disabled during API migration");
    anyhow::bail!("Bundle command is temporarily disabled. Use individual .holo files.")
}

/// Bundle models from a unified config file.
///
/// STUB: This function is temporarily disabled during API migration.
pub fn bundle_from_config(_config_path: &Path, _output: &Path) -> Result<()> {
    warn!("Bundle command is temporarily disabled during API migration");
    anyhow::bail!("Bundle command is temporarily disabled. Use individual .holo files.")
}

/// Extract models from a bundle to a directory.
///
/// STUB: This function is temporarily disabled during API migration.
pub fn extract_command(_bundle_path: &Path, _output_dir: &Path) -> Result<()> {
    warn!("Extract command is temporarily disabled during API migration");
    anyhow::bail!("Extract command is temporarily disabled.")
}

/// Create a pipeline bundle (HOLM format) from HOLB model bundles.
///
/// Each entry in `inputs` is a tuple of (name, path) where:
/// - `name` is the model name (e.g., "encoder", "decoder")
/// - `path` is the path to the .holb file
///
/// # Arguments
/// * `inputs` - List of (name, path) pairs for models to bundle
/// * `output` - Output path for the HOLM pipeline bundle
///
/// # Returns
/// Returns Ok(()) on success, or error if:
/// - Any input file cannot be read
/// - Any input file is not a valid HOLB bundle
/// - The output file cannot be written
pub fn bundle_pipeline_command(inputs: &[(&str, &Path)], output: &Path) -> Result<()> {
    if inputs.is_empty() {
        anyhow::bail!("At least one model is required for pipeline bundle");
    }

    info!("Creating HOLM pipeline bundle with {} models", inputs.len());
    for (name, path) in inputs {
        info!("  - {}: {}", name, path.display());
    }

    let mut writer = HolmWriter::new();

    for (name, path) in inputs {
        let data = fs::read(path)
            .with_context(|| format!("Failed to read model '{}' from: {}", name, path.display()))?;

        info!("  Adding '{}' ({} bytes)", name, data.len());

        writer
            .add_model(name, &data)
            .with_context(|| format!("Failed to add model '{}' to pipeline", name))?;
    }

    info!("Building pipeline bundle...");
    writer
        .write_to_file(output)
        .with_context(|| format!("Failed to write pipeline bundle to: {}", output.display()))?;

    let output_size = fs::metadata(output).map(|m| m.len()).unwrap_or(0);

    info!(
        "Successfully created pipeline bundle: {} ({} bytes)",
        output.display(),
        output_size
    );

    Ok(())
}

/// Create a pipeline bundle from a unified config file.
///
/// STUB: This function is temporarily disabled during API migration.
pub fn bundle_pipeline_from_config(_config_path: &Path, _output: &Path) -> Result<()> {
    warn!("Pipeline bundle command is temporarily disabled during API migration");
    anyhow::bail!("Pipeline bundle command is temporarily disabled.")
}

/// List models in a bundle.
///
/// STUB: This function is temporarily disabled during API migration.
pub fn list_pipeline_command(_bundle_path: &Path) -> Result<()> {
    warn!("List pipeline command is temporarily disabled during API migration");
    anyhow::bail!("List pipeline command is temporarily disabled.")
}
