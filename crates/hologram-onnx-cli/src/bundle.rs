//! Bundle multiple .holo files into a single distributable file.
//!
//! This module provides the `bundle` command which:
//! - Combines multiple compiled .holo models into one file
//! - Supports bundling from individual files or a config
//! - Extracts bundles back to individual files

use anyhow::{Context, Result};
use hologram_onnx_core::{BundleBuilder, HoloBundle};
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
    use hologram_onnx_config::UnifiedConfig;

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

/// List models in a bundle.
///
/// # Arguments
///
/// * `bundle_path` - Path to the bundle file
///
/// # Returns
///
/// Returns Ok(()) on success, or an error if listing fails.
pub fn list_command(bundle_path: &Path) -> Result<()> {
    let bundle = HoloBundle::from_file(bundle_path)
        .with_context(|| format!("Failed to load bundle from {}", bundle_path.display()))?;

    println!("Bundle: {}", bundle_path.display());
    println!("Version: {}", bundle.header.version);
    println!("Models: {}", bundle.model_count());
    println!("Total size: {} bytes", bundle.total_data_size());
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
