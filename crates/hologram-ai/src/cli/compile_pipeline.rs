//! Compile ONNX models and tokenizers directly into a pipeline bundle.
//!
//! This module provides the `compile-pipeline` command which:
//! - Takes ONNX model files and/or tokenizer JSON files as input
//! - Compiles each to intermediate HOLB bundles in a temp directory
//! - Bundles all HOLB files into a single HOLM pipeline bundle
//! - Cleans up intermediate files (unless --keep-intermediates)
//!
//! This combines the `compile` and `bundle-pipeline` commands into one step.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::info;

use super::bundle::bundle_pipeline_command;
use super::compile::compile_command;

/// Source type for a model in the pipeline.
#[derive(Debug, Clone)]
pub enum ModelSource {
    /// ONNX model file
    Onnx(PathBuf),
    /// Tokenizer JSON file (tokenizer.json format)
    Tokenizer(PathBuf),
}

/// Compile ONNX models and tokenizer into a single pipeline bundle.
///
/// # Arguments
///
/// * `encoder` - Optional path to encoder ONNX model
/// * `decoder` - Optional path to decoder ONNX model
/// * `tokenizer` - Optional path to tokenizer JSON file
/// * `models` - Additional models as "name=path" or "name=tokenizer:path" strings
/// * `config_path` - Optional config file to load models from
/// * `output` - Output path for the pipeline bundle
/// * `weight_threshold` - Weight threshold for external storage (bytes)
/// * `partition` - Enable graph partitioning
/// * `partition_size` - Number of nodes per partition
/// * `memory_budget` - Memory budget in MB
/// * `keep_intermediates` - Keep intermediate HOLB files
///
/// # Example
///
/// ```bash
/// hologram-onnx compile-pipeline \
///     --encoder models/encoder.onnx \
///     --decoder models/decoder.onnx \
///     --tokenizer models/tokenizer.json \
///     -o pipeline.holo
/// ```
#[allow(clippy::too_many_arguments)]
pub fn compile_pipeline_command(
    encoder: Option<&Path>,
    decoder: Option<&Path>,
    tokenizer: Option<&Path>,
    models: &[String],
    config_path: Option<&Path>,
    output: &Path,
    weight_threshold: usize,
    partition: bool,
    partition_size: usize,
    memory_budget: Option<usize>,
    keep_intermediates: bool,
) -> Result<()> {
    // Collect all model sources
    let mut sources: Vec<(String, ModelSource)> = Vec::new();

    // Handle config-based compilation
    if let Some(config) = config_path {
        return compile_pipeline_from_config(
            config,
            output,
            weight_threshold,
            partition,
            partition_size,
            memory_budget,
            keep_intermediates,
        );
    }

    // Add encoder if specified
    if let Some(path) = encoder {
        sources.push(("encoder".to_string(), ModelSource::Onnx(path.to_path_buf())));
    }

    // Add decoder if specified
    if let Some(path) = decoder {
        sources.push(("decoder".to_string(), ModelSource::Onnx(path.to_path_buf())));
    }

    // Add tokenizer if specified
    if let Some(path) = tokenizer {
        sources.push((
            "tokenizer".to_string(),
            ModelSource::Tokenizer(path.to_path_buf()),
        ));
    }

    // Parse additional models from --model args
    for model_spec in models {
        let (name, source) = parse_model_spec(model_spec)?;
        sources.push((name, source));
    }

    if sources.is_empty() {
        anyhow::bail!(
            "No models specified. Use --encoder, --decoder, --tokenizer, or --model name=path"
        );
    }

    compile_pipeline_from_sources(
        &sources,
        output,
        weight_threshold,
        partition,
        partition_size,
        memory_budget,
        keep_intermediates,
    )
}

/// Parse a model specification string into a name and source.
///
/// Formats:
/// - `name=path.onnx` - ONNX model
/// - `name=tokenizer:path.json` - Tokenizer
fn parse_model_spec(spec: &str) -> Result<(String, ModelSource)> {
    let parts: Vec<&str> = spec.splitn(2, '=').collect();
    if parts.len() != 2 {
        anyhow::bail!(
            "Invalid model specification '{}'. Expected format: name=path or name=tokenizer:path",
            spec
        );
    }

    let name = parts[0].to_string();
    let path_spec = parts[1];

    // Check for tokenizer: prefix
    if let Some(path) = path_spec.strip_prefix("tokenizer:") {
        Ok((name, ModelSource::Tokenizer(PathBuf::from(path))))
    } else {
        Ok((name, ModelSource::Onnx(PathBuf::from(path_spec))))
    }
}

/// Compile pipeline from a list of model sources.
#[allow(clippy::too_many_arguments)]
fn compile_pipeline_from_sources(
    sources: &[(String, ModelSource)],
    output: &Path,
    weight_threshold: usize,
    partition: bool,
    partition_size: usize,
    memory_budget: Option<usize>,
    keep_intermediates: bool,
) -> Result<()> {
    info!(
        "Compiling pipeline with {} models to {}",
        sources.len(),
        output.display()
    );

    // Create temp directory for intermediate files
    let (temp_dir, _temp_guard) = if keep_intermediates {
        (
            output
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf(),
            None,
        )
    } else {
        let dir = tempfile::tempdir().context("Failed to create temp directory")?;
        let path = dir.path().to_path_buf();
        // Keep the TempDir guard to clean up on drop (unless we keep())
        (path, Some(dir))
    };

    info!("Intermediate files: {}", temp_dir.display());

    // Compile each model to HOLB
    let mut holb_paths: Vec<(String, PathBuf)> = Vec::new();

    for (name, source) in sources {
        let holb_path = temp_dir.join(format!("{}_bundle.holo", name));

        match source {
            ModelSource::Onnx(onnx_path) => {
                info!("Compiling ONNX model '{}': {}", name, onnx_path.display());

                // Compile ONNX to HOLB bundle
                let output_base = holb_path.with_extension("");
                compile_command(
                    onnx_path,
                    &output_base,
                    partition,
                    partition_size,
                    memory_budget,
                    weight_threshold,
                    true, // decompose_conv2d
                    true, // decompose_pooling
                    true, // enable_resize_upscaling
                    &HashMap::new(),
                    true, // bundle = true (create HOLB)
                )
                .with_context(|| format!("Failed to compile ONNX model '{}'", name))?;
            }
            ModelSource::Tokenizer(json_path) => {
                info!("Compiling tokenizer '{}': {}", name, json_path.display());

                // Compile tokenizer to HOLB bundle
                let tokenizer_config = crate::tokenizers::TokenizerConfig {
                    tokenizer_type: "sentencepiece".to_string(),
                    vocab_path: json_path.display().to_string(),
                    max_length: 512,
                    pad_token_id: 0,
                    eos_token_id: 1,
                    unk_token_id: 2,
                    ..Default::default()
                };
                crate::tokenizers::compile_tokenizer_to_bundle(&tokenizer_config, &holb_path)
                    .with_context(|| format!("Failed to compile tokenizer '{}'", name))?;
            }
        }

        // Verify the HOLB file was created
        if !holb_path.exists() {
            anyhow::bail!(
                "Compilation produced no output for '{}': expected {}",
                name,
                holb_path.display()
            );
        }

        holb_paths.push((name.clone(), holb_path));
    }

    // Bundle all HOLB files into HOLM pipeline
    info!("Bundling {} models into pipeline...", holb_paths.len());

    let input_refs: Vec<(&str, &Path)> = holb_paths
        .iter()
        .map(|(name, path)| (name.as_str(), path.as_path()))
        .collect();

    bundle_pipeline_command(&input_refs, output)
        .with_context(|| "Failed to create pipeline bundle")?;

    // Log cleanup behavior
    if keep_intermediates {
        info!("Keeping intermediate files in: {}", temp_dir.display());
        for (name, path) in &holb_paths {
            info!("  {}: {}", name, path.display());
        }
    } else {
        info!("Intermediate files will be cleaned up automatically");
    }

    // TempDir guard will clean up automatically when dropped (if not keeping intermediates)
    drop(_temp_guard);

    info!("Pipeline compilation complete: {}", output.display());

    Ok(())
}

/// Compile pipeline from a config file.
#[allow(clippy::too_many_arguments)]
fn compile_pipeline_from_config(
    config_path: &Path,
    output: &Path,
    weight_threshold: usize,
    partition: bool,
    partition_size: usize,
    memory_budget: Option<usize>,
    keep_intermediates: bool,
) -> Result<()> {
    use crate::config::UnifiedConfig;

    info!("Loading config from: {}", config_path.display());

    let config = UnifiedConfig::from_file(config_path)
        .map_err(|e| anyhow::anyhow!("Failed to load config: {}", e))?;

    let config_dir = config_path.parent().unwrap_or_else(|| Path::new("."));

    if config.models.is_empty() {
        anyhow::bail!("No models specified in config");
    }

    // Collect model sources from config
    let mut sources: Vec<(String, ModelSource)> = Vec::new();

    for (name, model_def) in &config.models {
        let path = model_def.path();
        let full_path = if Path::new(path).is_absolute() {
            PathBuf::from(path)
        } else {
            config_dir.join(path)
        };

        // Determine source type based on path or model_def type
        let source = if let Some(model_type) = model_def.model_type() {
            match model_type.to_lowercase().as_str() {
                "tokenizer" | "sentencepiece" => ModelSource::Tokenizer(full_path),
                "onnx" => ModelSource::Onnx(full_path),
                _ => ModelSource::Onnx(full_path), // Default to ONNX for unknown types
            }
        } else {
            // Infer from file extension
            if path.ends_with(".json") || path.contains("tokenizer") {
                ModelSource::Tokenizer(full_path)
            } else {
                ModelSource::Onnx(full_path)
            }
        };

        sources.push((name.clone(), source));
    }

    info!(
        "Compiling pipeline '{}' with {} models",
        config.name.as_deref().unwrap_or("unnamed"),
        sources.len()
    );

    compile_pipeline_from_sources(
        &sources,
        output,
        weight_threshold,
        partition,
        partition_size,
        memory_budget,
        keep_intermediates,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_model_spec_onnx() {
        let (name, source) = parse_model_spec("encoder=models/encoder.onnx").unwrap();
        assert_eq!(name, "encoder");
        match source {
            ModelSource::Onnx(path) => assert_eq!(path, PathBuf::from("models/encoder.onnx")),
            _ => panic!("Expected ONNX source"),
        }
    }

    #[test]
    fn test_parse_model_spec_tokenizer() {
        let (name, source) = parse_model_spec("tokenizer=tokenizer:models/tokenizer.json").unwrap();
        assert_eq!(name, "tokenizer");
        match source {
            ModelSource::Tokenizer(path) => {
                assert_eq!(path, PathBuf::from("models/tokenizer.json"))
            }
            _ => panic!("Expected Tokenizer source"),
        }
    }

    #[test]
    fn test_parse_model_spec_invalid() {
        let result = parse_model_spec("invalid-no-equals");
        assert!(result.is_err());
    }

    #[test]
    fn test_compile_pipeline_no_models() {
        let temp_dir = tempfile::tempdir().unwrap();
        let output = temp_dir.path().join("pipeline.holo");

        let result = compile_pipeline_command(
            None,
            None,
            None,
            &[],
            None,
            &output,
            4096,
            false,
            500,
            None,
            false,
        );

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("No models specified")
        );
    }
}
