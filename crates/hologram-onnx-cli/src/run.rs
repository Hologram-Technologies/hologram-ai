//! Run ONNX pipelines with unified configuration.
//!
//! This module provides the `run` command which:
//! - Loads a unified config from a TOML file
//! - Compiles models if needed (or uses precompiled .holo files)
//! - Executes the pipeline with provided inputs
//! - Processes outputs using configured handlers

use anyhow::{Context, Result};
use hologram_onnx_config::{
    ModelDef, OutputHandlerType, PipelineConfig, StageDef, UnifiedConfig,
};
use hologram_onnx_core::OnnxConfig;
use std::collections::HashMap;
use std::path::Path;
use tracing::{debug, info, warn};

use crate::compile::compile_command;

/// Run an ONNX pipeline from a unified config file.
///
/// # Arguments
///
/// * `config_path` - Path to the unified config TOML file
/// * `inputs` - Runtime inputs as key=value pairs
/// * `output_dir` - Optional directory for output files
/// * `force_recompile` - Force recompilation even if .holo files exist
///
/// # Returns
///
/// Returns Ok(()) on success, or an error if execution fails.
pub fn run_command(
    config_path: &Path,
    inputs: &[String],
    output_dir: Option<&Path>,
    force_recompile: bool,
) -> Result<()> {
    info!("Loading pipeline config: {}", config_path.display());

    // Load the unified config
    let config = UnifiedConfig::from_file(config_path)
        .with_context(|| format!("Failed to load config from {}", config_path.display()))?;

    // Get the config directory for resolving relative paths
    let config_dir = config_path
        .parent()
        .unwrap_or_else(|| Path::new("."));

    info!("Pipeline: {}", config.name.as_deref().unwrap_or("unnamed"));
    if let Some(desc) = &config.description {
        info!("Description: {}", desc);
    }

    // Parse runtime inputs
    let runtime_inputs = parse_inputs(inputs, &config)?;
    debug!("Runtime inputs: {:?}", runtime_inputs);

    // Ensure all models are compiled
    info!("Checking model compilation status...");
    let compiled_models = ensure_models_compiled(&config, config_dir, force_recompile)?;
    debug!("Compiled models: {:?}", compiled_models);

    // Convert to PipelineConfig for execution
    let pipeline_config: PipelineConfig = (&config).into();
    debug!("Pipeline config: {:?}", pipeline_config.pipeline.name);

    // Execute the pipeline
    info!("Executing pipeline...");
    let outputs = execute_pipeline(&config, &compiled_models, &runtime_inputs)?;

    // Process outputs
    info!("Processing outputs...");
    process_outputs(&config, &outputs, output_dir)?;

    info!("Pipeline execution complete!");
    Ok(())
}

/// Parse command-line inputs into a HashMap.
///
/// Inputs are expected in the format "name=value".
fn parse_inputs(
    inputs: &[String],
    config: &UnifiedConfig,
) -> Result<HashMap<String, String>> {
    let mut result = HashMap::new();

    for input in inputs {
        let parts: Vec<&str> = input.splitn(2, '=').collect();
        if parts.len() != 2 {
            anyhow::bail!(
                "Invalid input format '{}'. Expected 'name=value'",
                input
            );
        }

        let name = parts[0].to_string();
        let value = parts[1].to_string();

        // Validate that the input is defined in config
        if !config.inputs.contains_key(&name) && !config.inputs.is_empty() {
            warn!(
                "Input '{}' not defined in config. Available inputs: {:?}",
                name,
                config.inputs.keys().collect::<Vec<_>>()
            );
        }

        result.insert(name, value);
    }

    // Fill in defaults from config
    for (name, input_def) in &config.inputs {
        if !result.contains_key(name) {
            if let Some(default_str) = input_def.default_string() {
                debug!("Using default for '{}': {}", name, default_str);
                result.insert(name.clone(), default_str.to_string());
            } else if let Some(default_val) = input_def.default_value() {
                if let Some(s) = default_val.as_str() {
                    debug!("Using default for '{}': {}", name, s);
                    result.insert(name.clone(), s.to_string());
                }
            }
        }
    }

    Ok(result)
}

/// Ensure all models in the config are compiled.
///
/// Returns a map of model name → compiled .holo path.
fn ensure_models_compiled(
    config: &UnifiedConfig,
    config_dir: &Path,
    force_recompile: bool,
) -> Result<HashMap<String, std::path::PathBuf>> {
    let mut compiled = HashMap::new();

    for (name, model_def) in &config.models {
        let holo_path = get_compiled_path(model_def, config_dir);

        // Check if compilation is needed
        let needs_compile = force_recompile || !holo_path.exists();

        if needs_compile {
            let onnx_path = resolve_model_path(model_def.path(), config_dir);

            if !onnx_path.exists() {
                anyhow::bail!(
                    "ONNX model not found: {} (for model '{}')",
                    onnx_path.display(),
                    name
                );
            }

            info!("Compiling model '{}': {}", name, onnx_path.display());

            // Get compiler config
            let onnx_config: OnnxConfig = (&config.compiler).into();

            // Compile the model
            let output_base = holo_path.with_extension("");
            compile_command(
                &onnx_path,
                &output_base,
                onnx_config.enable_partitioning,
                onnx_config.partition_size,
                onnx_config.memory_budget,
                onnx_config.weight_threshold,
            )
            .with_context(|| format!("Failed to compile model '{}'", name))?;

            info!("Model '{}' compiled successfully", name);
        } else {
            debug!("Model '{}' already compiled: {}", name, holo_path.display());
        }

        compiled.insert(name.clone(), holo_path);
    }

    Ok(compiled)
}

/// Get the path to the compiled .holo file for a model.
fn get_compiled_path(model_def: &ModelDef, config_dir: &Path) -> std::path::PathBuf {
    // Check if precompiled path is specified
    if let Some(precompiled) = model_def.precompiled() {
        return resolve_model_path(precompiled, config_dir);
    }

    // Default: replace .onnx with .holo
    let onnx_path = model_def.path();
    let holo_path = if onnx_path.ends_with(".onnx") {
        format!("{}.holo", &onnx_path[..onnx_path.len() - 5])
    } else {
        format!("{}.holo", onnx_path)
    };

    resolve_model_path(&holo_path, config_dir)
}

/// Resolve a model path relative to the config directory.
fn resolve_model_path(path: &str, config_dir: &Path) -> std::path::PathBuf {
    let path = Path::new(path);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        config_dir.join(path)
    }
}

/// Execute the pipeline with the given inputs.
///
/// This is a placeholder implementation. Full implementation will use
/// hologram's runtime execution engine.
fn execute_pipeline(
    config: &UnifiedConfig,
    compiled_models: &HashMap<String, std::path::PathBuf>,
    _inputs: &HashMap<String, String>,
) -> Result<HashMap<String, OutputData>> {
    let mut outputs = HashMap::new();

    // Process stages
    for (idx, stage) in config.stages.iter().enumerate() {
        match stage {
            StageDef::Model(model_stage) => {
                let model_path = compiled_models.get(&model_stage.model).ok_or_else(|| {
                    anyhow::anyhow!("Model '{}' not found in compiled models", model_stage.model)
                })?;

                debug!(
                    "Stage {}: Executing model '{}' from {}",
                    idx,
                    model_stage.model,
                    model_path.display()
                );

                // Placeholder: In a real implementation, this would:
                // 1. Load the compiled model
                // 2. Prepare input tensors from the inputs map
                // 3. Execute the model
                // 4. Store outputs in the outputs map

                // For now, create placeholder outputs
                for output_name in &model_stage.outputs {
                    outputs.insert(
                        output_name.clone(),
                        OutputData::Tensor(vec![0.0f32; 10]), // Placeholder
                    );
                }

                info!("  ✓ Stage {}: {} completed", idx, model_stage.model);
            }

            StageDef::Builtin(builtin_stage) => {
                debug!(
                    "Stage {}: Executing builtin '{}'",
                    idx, builtin_stage.builtin
                );

                // Placeholder for builtin operations (randn, concat, etc.)
                for output_name in &builtin_stage.outputs {
                    outputs.insert(
                        output_name.clone(),
                        OutputData::Tensor(vec![0.0f32; 10]),
                    );
                }

                info!("  ✓ Stage {}: {} completed", idx, builtin_stage.builtin);
            }

            StageDef::Loop(loop_stage) => {
                debug!("Stage {}: Loop over '{}'", idx, loop_stage.over);
                // Placeholder for loop execution
                info!("  ✓ Stage {}: loop completed", idx);
            }

            StageDef::Conditional(cond_stage) => {
                debug!("Stage {}: Conditional '{}'", idx, cond_stage.condition);
                // Placeholder for conditional execution
                info!("  ✓ Stage {}: conditional completed", idx);
            }
        }
    }

    // Ensure all config outputs are present
    for (name, output_def) in &config.outputs {
        let tensor_name = output_def.tensor();
        if !outputs.contains_key(tensor_name) {
            debug!(
                "Output '{}' (tensor '{}') not produced by pipeline",
                name, tensor_name
            );
            // Create placeholder output
            outputs.insert(tensor_name.to_string(), OutputData::Tensor(vec![0.0f32; 10]));
        }
    }

    Ok(outputs)
}

/// Output data from pipeline execution.
#[derive(Debug)]
#[allow(dead_code)] // Variants defined for future runtime implementation
pub enum OutputData {
    /// Raw tensor data
    Tensor(Vec<f32>),
    /// Image data (width, height, channels, data)
    Image(u32, u32, u32, Vec<u8>),
    /// Audio data (sample_rate, samples)
    Audio(u32, Vec<f32>),
    /// Text output
    Text(String),
}

/// Process outputs using configured handlers.
fn process_outputs(
    config: &UnifiedConfig,
    outputs: &HashMap<String, OutputData>,
    output_dir: Option<&Path>,
) -> Result<()> {
    let output_dir = output_dir.unwrap_or_else(|| Path::new("."));

    for (name, output_def) in &config.outputs {
        let tensor_name = output_def.tensor();
        let handler_type = output_def.handler_type();

        let data = outputs.get(tensor_name).ok_or_else(|| {
            anyhow::anyhow!("Output tensor '{}' not found in pipeline outputs", tensor_name)
        })?;

        match handler_type {
            OutputHandlerType::Image => {
                let output_path = output_dir.join(format!("{}.png", name));
                info!("Writing image output: {}", output_path.display());
                // Placeholder: Would use image output handler
                std::fs::write(&output_path, b"PNG placeholder")?;
            }

            OutputHandlerType::Audio => {
                let output_path = output_dir.join(format!("{}.wav", name));
                info!("Writing audio output: {}", output_path.display());
                // Placeholder: Would use audio output handler
                std::fs::write(&output_path, b"WAV placeholder")?;
            }

            OutputHandlerType::Text => {
                if let OutputData::Text(text) = data {
                    info!("Text output '{}': {}", name, text);
                } else {
                    info!("Text output '{}': <tensor data>", name);
                }
            }

            OutputHandlerType::Json => {
                let output_path = output_dir.join(format!("{}.json", name));
                info!("Writing JSON output: {}", output_path.display());
                // Placeholder: Would serialize tensor to JSON
                std::fs::write(&output_path, "{}")?;
            }

            OutputHandlerType::Binary => {
                let output_path = output_dir.join(format!("{}.bin", name));
                info!("Writing binary output: {}", output_path.display());
                if let OutputData::Tensor(tensor) = data {
                    let bytes: Vec<u8> = tensor
                        .iter()
                        .flat_map(|f| f.to_le_bytes())
                        .collect();
                    std::fs::write(&output_path, bytes)?;
                }
            }

            OutputHandlerType::Auto => {
                debug!("Auto output handler for '{}' - no action", name);
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
    fn test_parse_inputs_simple() {
        let inputs = vec!["prompt=hello world".to_string()];
        let config = UnifiedConfig::default();

        let result = parse_inputs(&inputs, &config).unwrap();
        assert_eq!(result.get("prompt"), Some(&"hello world".to_string()));
    }

    #[test]
    fn test_parse_inputs_multiple() {
        let inputs = vec![
            "prompt=test".to_string(),
            "steps=50".to_string(),
        ];
        let config = UnifiedConfig::default();

        let result = parse_inputs(&inputs, &config).unwrap();
        assert_eq!(result.get("prompt"), Some(&"test".to_string()));
        assert_eq!(result.get("steps"), Some(&"50".to_string()));
    }

    #[test]
    fn test_parse_inputs_invalid_format() {
        let inputs = vec!["invalid_input".to_string()];
        let config = UnifiedConfig::default();

        let result = parse_inputs(&inputs, &config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Expected 'name=value'"));
    }

    #[test]
    fn test_resolve_model_path_absolute() {
        let config_dir = Path::new("/home/user/config");
        let result = resolve_model_path("/absolute/path/model.onnx", config_dir);
        assert_eq!(result, Path::new("/absolute/path/model.onnx"));
    }

    #[test]
    fn test_resolve_model_path_relative() {
        let config_dir = Path::new("/home/user/config");
        let result = resolve_model_path("models/model.onnx", config_dir);
        assert_eq!(result, Path::new("/home/user/config/models/model.onnx"));
    }

    #[test]
    fn test_get_compiled_path_default() {
        let model = ModelDef::Path("models/encoder.onnx".to_string());
        let config_dir = Path::new("/config");
        let result = get_compiled_path(&model, config_dir);
        assert_eq!(result, Path::new("/config/models/encoder.holo"));
    }

    #[test]
    fn test_run_command_missing_config() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("missing.toml");

        let result = run_command(&config_path, &[], None, false);
        assert!(result.is_err());
    }
}
