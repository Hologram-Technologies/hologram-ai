//! Bundle multiple .holo files into a single distributable file.
//!
//! This module provides the `bundle` command which:
//! - Combines multiple compiled .holo models into one file
//! - Supports bundling from individual files or a config
//! - Extracts bundles back to individual files
//! - Creates pipeline bundles (HOLM format) with embedded weights

use anyhow::{Context, Result};
use crate::core::{BundleBuilder, HoloBundle, HoloFormat, PipelineBundleReader, PipelineBundleWriter, UnifiedBundleReader};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use tracing::{debug, info};

/// Bundle multiple .holo files into a single bundle.
///
/// # Arguments
///
/// * `inputs` - Paths to .holo files to bundle
/// * `output` - Output path for the bundle
/// * `names` - Optional custom names for each model (parallel to inputs)
///
/// # Returns
///
/// Returns Ok(()) on success, or an error if bundling fails.
pub fn bundle_command(
    inputs: &[std::path::PathBuf],
    output: &Path,
    names: Option<&[String]>,
) -> Result<()> {
    info!("Creating bundle with {} models", inputs.len());

    if inputs.is_empty() {
        anyhow::bail!("No input files specified");
    }

    let mut builder = BundleBuilder::new();

    for (i, input_path) in inputs.iter().enumerate() {
        // Determine model name
        let name = if let Some(names) = names {
            names.get(i).cloned()
        } else {
            None
        };

        let name = name.unwrap_or_else(|| {
            input_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("model")
                .to_string()
        });

        info!("  Adding model '{}': {}", name, input_path.display());

        builder
            .add_model_from_file(&name, input_path)
            .with_context(|| format!("Failed to add model from {}", input_path.display()))?;
    }

    let bundle = builder.build().context("Failed to build bundle")?;

    info!("Bundle statistics:");
    info!("  Models: {}", bundle.model_count());
    info!("  Total data size: {} bytes", bundle.total_data_size());

    for entry in &bundle.entries {
        debug!(
            "    {} - {} bytes (checksum: {:08x})",
            entry.name, entry.data_size, entry.checksum
        );
    }

    // Write bundle
    info!("Writing bundle to: {}", output.display());
    bundle
        .write_to_file(output)
        .with_context(|| format!("Failed to write bundle to {}", output.display()))?;

    info!("Bundle created successfully!");
    Ok(())
}

/// Bundle models from a unified config file.
///
/// This reads compiled .holo files referenced in the config and bundles them.
///
/// # Arguments
///
/// * `config_path` - Path to the unified config file
/// * `output` - Output path for the bundle
///
/// # Returns
///
/// Returns Ok(()) on success, or an error if bundling fails.
pub fn bundle_from_config(config_path: &Path, output: &Path) -> Result<()> {
    use crate::config::UnifiedConfig;

    info!("Loading config from: {}", config_path.display());

    let config = UnifiedConfig::from_file(config_path)
        .map_err(|e| anyhow::anyhow!("Failed to load config: {}", e))?;

    let config_dir = config_path.parent().unwrap_or_else(|| Path::new("."));

    if config.models.is_empty() {
        anyhow::bail!("No models specified in config");
    }

    info!(
        "Creating bundle '{}' with {} models",
        config.name.as_deref().unwrap_or("unnamed"),
        config.models.len()
    );

    let mut builder = BundleBuilder::new();

    for (name, model_def) in &config.models {
        // Get the .holo path (either precompiled or derived from .onnx path)
        let holo_path = if let Some(precompiled) = model_def.precompiled() {
            let path = Path::new(precompiled);
            if path.is_absolute() {
                path.to_path_buf()
            } else {
                config_dir.join(path)
            }
        } else {
            let onnx_path = model_def.path();
            let holo_name = if let Some(stripped) = onnx_path.strip_suffix(".onnx") {
                format!("{}.holo", stripped)
            } else {
                format!("{}.holo", onnx_path)
            };
            let path = Path::new(&holo_name);
            if path.is_absolute() {
                path.to_path_buf()
            } else {
                config_dir.join(path)
            }
        };

        if !holo_path.exists() {
            anyhow::bail!(
                "Compiled model not found: {} (for model '{}'). Run 'hologram-onnx compile --config {}' first.",
                holo_path.display(),
                name,
                config_path.display()
            );
        }

        info!("  Adding model '{}': {}", name, holo_path.display());
        builder
            .add_model_from_file(name, &holo_path)
            .with_context(|| format!("Failed to add model '{}' from {}", name, holo_path.display()))?;
    }

    let bundle = builder.build().context("Failed to build bundle")?;

    info!("Bundle statistics:");
    info!("  Models: {}", bundle.model_count());
    info!("  Total data size: {} bytes", bundle.total_data_size());

    // Write bundle
    info!("Writing bundle to: {}", output.display());
    bundle
        .write_to_file(output)
        .with_context(|| format!("Failed to write bundle to {}", output.display()))?;

    info!("Bundle created successfully!");
    Ok(())
}

/// Extract models from a bundle to a directory.
///
/// # Arguments
///
/// * `bundle_path` - Path to the bundle file
/// * `output_dir` - Directory to extract models to
///
/// # Returns
///
/// Returns Ok(()) on success, or an error if extraction fails.
pub fn extract_command(bundle_path: &Path, output_dir: &Path) -> Result<()> {
    info!("Loading bundle from: {}", bundle_path.display());

    let bundle = HoloBundle::from_file(bundle_path)
        .with_context(|| format!("Failed to load bundle from {}", bundle_path.display()))?;

    info!("Bundle contains {} models:", bundle.model_count());
    for name in bundle.model_names() {
        if let Some(entry) = bundle.get_entry(name) {
            info!("  {} - {} bytes", name, entry.data_size);
        }
    }

    info!("Extracting to: {}", output_dir.display());
    bundle
        .extract_to_dir(output_dir)
        .with_context(|| format!("Failed to extract to {}", output_dir.display()))?;

    info!("Extraction complete!");
    Ok(())
}

/// Format a byte size for display.
fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Create a pipeline bundle (HOLM format) from multiple HOLB bundle files.
///
/// Pipeline bundles package multiple models with their embedded weights into
/// a single file where each model's weights section remains page-aligned for
/// efficient memory-mapping.
///
/// # Arguments
///
/// * `inputs` - Paths to HOLB bundle files to include
/// * `names` - Names for each model (parallel to inputs)
/// * `output` - Output path for the pipeline bundle
///
/// # Example
///
/// ```bash
/// hologram-onnx bundle-pipeline \
///     --encoder models/encoder_bundle.holo \
///     --decoder models/decoder_bundle.holo \
///     --output models/pipeline.holo
/// ```
pub fn bundle_pipeline_command(
    inputs: &[(&str, &Path)], // (name, path) pairs
    output: &Path,
) -> Result<()> {
    info!(
        "Creating pipeline bundle with {} models",
        inputs.len()
    );

    if inputs.is_empty() {
        anyhow::bail!("No input files specified");
    }

    let mut writer = PipelineBundleWriter::new();

    for (name, path) in inputs {
        // Read the HOLB bundle file
        let mut file = File::open(path)
            .with_context(|| format!("Failed to open model file: {}", path.display()))?;

        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .with_context(|| format!("Failed to read model file: {}", path.display()))?;

        // Verify it's a valid HOLB bundle
        if bytes.len() < 4 {
            anyhow::bail!(
                "File too small to be a HOLB bundle: {} ({} bytes)",
                path.display(),
                bytes.len()
            );
        }

        let format = HoloFormat::detect(&bytes[..4]);
        if !format.is_bundle() {
            anyhow::bail!(
                "File is not a HOLB bundle: {} (detected format: {:?}). \
                 Only HOLB bundles (single model with embedded weights) can be combined into a pipeline.",
                path.display(),
                format
            );
        }

        // Verify the bundle is valid
        let reader = UnifiedBundleReader::from_bytes(&bytes)
            .with_context(|| format!("Invalid HOLB bundle: {}", path.display()))?;

        if !reader.verify_checksums() {
            anyhow::bail!(
                "Checksum verification failed for: {}",
                path.display()
            );
        }

        info!(
            "  Adding model '{}': {} ({} graph, {} weights)",
            name,
            path.display(),
            format_size(reader.graph_bytes().len() as u64),
            format_size(reader.weights_bytes().len() as u64)
        );

        writer
            .add_model(name, bytes)
            .with_context(|| format!("Failed to add model '{}' to pipeline", name))?;
    }

    // Write the pipeline bundle
    info!("Writing pipeline bundle to: {}", output.display());
    let bytes_written = writer
        .write_to_file(output)
        .with_context(|| format!("Failed to write pipeline bundle to {}", output.display()))?;

    info!(
        "Pipeline bundle created successfully: {} total",
        format_size(bytes_written as u64)
    );

    Ok(())
}

/// Create a pipeline bundle from a unified config file.
///
/// Reads HOLB bundle files referenced in the config and creates a pipeline bundle.
///
/// # Arguments
///
/// * `config_path` - Path to the unified config file
/// * `output` - Output path for the pipeline bundle
pub fn bundle_pipeline_from_config(config_path: &Path, output: &Path) -> Result<()> {
    use crate::config::UnifiedConfig;

    info!("Loading config from: {}", config_path.display());

    let config = UnifiedConfig::from_file(config_path)
        .map_err(|e| anyhow::anyhow!("Failed to load config: {}", e))?;

    let config_dir = config_path.parent().unwrap_or_else(|| Path::new("."));

    if config.models.is_empty() {
        anyhow::bail!("No models specified in config");
    }

    info!(
        "Creating pipeline bundle '{}' with {} models",
        config.name.as_deref().unwrap_or("unnamed"),
        config.models.len()
    );

    // Collect paths as owned values
    let model_paths: Vec<(String, std::path::PathBuf)> = config
        .models
        .iter()
        .map(|(name, model_def)| {
            // Look for HOLB bundle file (model_bundle.holo pattern)
            let bundle_path = if let Some(precompiled) = model_def.precompiled() {
                let precompiled_path = Path::new(precompiled);
                // Check for _bundle.holo variant
                let bundle_name = precompiled_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .map(|s| format!("{}_bundle.holo", s))
                    .unwrap_or_else(|| format!("{}_bundle.holo", precompiled));

                let bundle_path = precompiled_path.parent().unwrap_or(Path::new(".")).join(&bundle_name);
                if precompiled_path.is_absolute() {
                    bundle_path
                } else {
                    config_dir.join(bundle_path)
                }
            } else {
                let onnx_path = model_def.path();
                let bundle_name = if let Some(stripped) = onnx_path.strip_suffix(".onnx") {
                    format!("{}_bundle.holo", stripped)
                } else {
                    format!("{}_bundle.holo", onnx_path)
                };
                config_dir.join(&bundle_name)
            };
            (name.clone(), bundle_path)
        })
        .collect();

    // Check all bundle files exist
    for (name, bundle_path) in &model_paths {
        if !bundle_path.exists() {
            anyhow::bail!(
                "HOLB bundle not found: {} (for model '{}'). \
                 Run 'hologram-onnx compile --config {} --bundle true' first.",
                bundle_path.display(),
                name,
                config_path.display()
            );
        }
    }

    // Convert to slice of (&str, &Path) pairs
    let inputs: Vec<(&str, &Path)> = model_paths
        .iter()
        .map(|(name, path)| (name.as_str(), path.as_path()))
        .collect();

    bundle_pipeline_command(&inputs, output)
}

/// List models in a bundle (supports both HOLO v2 and HOLM pipeline formats).
///
/// # Arguments
///
/// * `bundle_path` - Path to the bundle file
///
/// # Returns
///
/// Returns Ok(()) on success, or an error if listing fails.
pub fn list_pipeline_command(bundle_path: &Path) -> Result<()> {
    // Read magic bytes to determine format
    let mut file = File::open(bundle_path)
        .with_context(|| format!("Failed to open bundle: {}", bundle_path.display()))?;

    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)
        .with_context(|| "Failed to read magic bytes")?;

    let format = HoloFormat::detect(&magic);

    match format {
        HoloFormat::Pipeline => {
            // HOLM format: multi-model with embedded weights
            drop(file);
            let mut file = File::open(bundle_path)?;
            let mut bytes = Vec::new();
            file.read_to_end(&mut bytes)?;

            let reader = PipelineBundleReader::from_bytes(&bytes)
                .with_context(|| "Failed to parse pipeline bundle")?;

            println!("Pipeline Bundle: {}", bundle_path.display());
            println!("Format: HOLM (multi-model with embedded weights)");
            println!("Models: {}", reader.model_count());
            println!();
            println!(
                "{:<20} {:>12} {:>12} {:>12}",
                "Name", "Total Size", "Graph", "Weights"
            );
            println!("{}", "-".repeat(58));

            for name in reader.model_names() {
                if let Some(entry) = reader.get_entry(name) {
                    // Parse the HOLB to get graph/weights sizes
                    if let Some(model_reader) = reader.get_model(name) {
                        println!(
                            "{:<20} {:>12} {:>12} {:>12}",
                            name,
                            format_size(entry.size),
                            format_size(model_reader.graph_bytes().len() as u64),
                            format_size(model_reader.weights_bytes().len() as u64)
                        );
                    } else {
                        println!(
                            "{:<20} {:>12} {:>12} {:>12}",
                            name,
                            format_size(entry.size),
                            "-",
                            "-"
                        );
                    }
                }
            }

            let total_size = std::fs::metadata(bundle_path)?.len();
            println!();
            println!("Total bundle size: {}", format_size(total_size));
        }
        HoloFormat::Legacy => {
            // HOLO v2 format: multi-model, no embedded weights
            let bundle = HoloBundle::from_file(bundle_path)
                .with_context(|| format!("Failed to load bundle from {}", bundle_path.display()))?;

            println!("Bundle: {}", bundle_path.display());
            println!("Format: HOLO v2 (multi-model, no embedded weights)");
            println!("Version: {}", bundle.header.version);
            println!("Models: {}", bundle.model_count());
            println!("Total data size: {}", format_size(bundle.total_data_size()));
            println!();
            println!("{:<20} {:>12} {:>12}", "Name", "Size", "Checksum");
            println!("{}", "-".repeat(46));

            for entry in &bundle.entries {
                println!(
                    "{:<20} {:>12} {:>12}",
                    entry.name,
                    format_size(entry.data_size),
                    format!("{:08x}", entry.checksum)
                );
            }
        }
        HoloFormat::Bundle => {
            // HOLB format: single model with embedded weights
            println!("File: {}", bundle_path.display());
            println!("Format: HOLB (single model with embedded weights)");
            println!();

            drop(file);
            let mut file = File::open(bundle_path)?;
            let mut bytes = Vec::new();
            file.read_to_end(&mut bytes)?;

            let reader = UnifiedBundleReader::from_bytes(&bytes)
                .with_context(|| "Failed to parse HOLB bundle")?;

            println!("Graph size: {}", format_size(reader.graph_bytes().len() as u64));
            println!("Weights size: {}", format_size(reader.weights_bytes().len() as u64));
            let total_size = std::fs::metadata(bundle_path)?.len();
            println!("Total size: {}", format_size(total_size));
        }
        _ => {
            anyhow::bail!(
                "Unknown bundle format: {:?}. Expected HOLB, HOLO v2, or HOLM pipeline.",
                format
            );
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(500), "500 B");
        assert_eq!(format_size(1024), "1.00 KB");
        assert_eq!(format_size(1536), "1.50 KB");
        assert_eq!(format_size(1024 * 1024), "1.00 MB");
        assert_eq!(format_size(1024 * 1024 * 1024), "1.00 GB");
    }

    #[test]
    fn test_bundle_command_empty() {
        let temp_dir = TempDir::new().unwrap();
        let output = temp_dir.path().join("empty.holo");

        let result = bundle_command(&[], &output, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_bundle_command_missing_input() {
        let temp_dir = TempDir::new().unwrap();
        let output = temp_dir.path().join("test.holo");
        let missing = temp_dir.path().join("missing.holo");

        let result = bundle_command(&[missing], &output, None);
        assert!(result.is_err());
    }
}
