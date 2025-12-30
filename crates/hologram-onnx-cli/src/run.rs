//! Run ONNX pipelines with unified configuration.
//!
//! This module provides the `run` command which:
//! - Loads a unified config from a TOML file
//! - Compiles models if needed (or uses precompiled .holo files)
//! - Executes the pipeline with provided inputs using hologram runtime
//! - Processes outputs using configured handlers

use anyhow::{Context, Result};
use hologram_onnx_config::{
    ModelDef, OutputDef, OutputHandlerType, PipelineConfig, StageDef, UnifiedConfig,
};
use hologram_onnx_core::serialization::{DimSpec, SerNode, SerNodeKind};
use hologram_onnx_core::{load_holo_file, Interpreter, OnnxConfig};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;
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
            } else if let Some(default_val) = input_def.default_value()
                && let Some(s) = default_val.as_str()
            {
                debug!("Using default for '{}': {}", name, s);
                result.insert(name.clone(), s.to_string());
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
    let holo_path = if let Some(stripped) = onnx_path.strip_suffix(".onnx") {
        format!("{}.holo", stripped)
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

/// Execute the pipeline with the given inputs using the interpreter.
///
/// This function loads compiled .holo files and executes them using the
/// CPU-based interpreter which supports all operations.
fn execute_pipeline(
    config: &UnifiedConfig,
    compiled_models: &HashMap<String, std::path::PathBuf>,
    inputs: &HashMap<String, String>,
) -> Result<HashMap<String, OutputData>> {
    let mut outputs: HashMap<String, OutputData> = HashMap::new();
    let mut tensor_cache: HashMap<String, Arc<Vec<f32>>> = HashMap::new();

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

                // Load the compiled model using the new format
                let holo_model = load_holo_file(model_path)
                    .with_context(|| format!("Failed to load .holo file: {}", model_path.display()))?;

                debug!("Loaded HoloModel '{}' with {} nodes", holo_model.metadata.name, holo_model.graph.nodes.len());

                // Create interpreter for this model
                let mut interpreter = Interpreter::new(&holo_model)
                    .map_err(|e| anyhow::anyhow!("Failed to create interpreter: {}", e))?;

                let model_inputs: HashMap<&str, &SerNode> = holo_model
                    .graph
                    .nodes
                    .iter()
                    .filter_map(|node| {
                        if let SerNodeKind::Input { name } = &node.node {
                            Some((name.as_str(), node))
                        } else {
                            None
                        }
                    })
                    .collect();
                let mut set_inputs: HashSet<String> = HashSet::new();

                // Map stage inputs to tensors
                for (input_name, tensor_expr) in &model_stage.inputs {
                    let input_node = model_inputs.get(input_name.as_str()).ok_or_else(|| {
                        anyhow::anyhow!(
                            "Model '{}' has no input named '{}'",
                            model_stage.model,
                            input_name
                        )
                    })?;

                    // Extract string reference from expression
                    let tensor_ref = tensor_expr.as_str().unwrap_or_default();

                    // First check if it's in the tensor cache from previous stages
                    let input_data = if let Some(cached) = tensor_cache.get(tensor_ref) {
                        cached.as_ref().clone()
                    } else if let Some(input_str) = inputs.get(tensor_ref) {
                        // Parse runtime input as tensor
                        parse_input_tensor(input_str)?
                    } else {
                        let shape = input_shape_for_node(input_node)?;
                        warn!(
                            "Input '{}' (tensor '{}') not found, using zeros with shape {:?}",
                            input_name, tensor_ref, shape
                        );
                        default_tensor_for_node(input_node)?
                    };

                    // Set the input in interpreter
                    interpreter
                        .set_input(input_name, &input_data)
                        .with_context(|| format!("Failed to set input '{}'", input_name))?;
                    set_inputs.insert(input_name.clone());
                }

                for (input_name, input_spec) in &model_inputs {
                    if !set_inputs.contains(*input_name) {
                        let shape = input_shape_for_node(input_spec)?;
                        warn!(
                            "Input '{}' not provided, using zeros with shape {:?}",
                            input_name, shape
                        );
                        let input_data = default_tensor_for_node(input_spec)?;
                        interpreter
                            .set_input(input_name, &input_data)
                            .with_context(|| format!("Failed to set input '{}'", input_name))?;
                    }
                }

                // Execute the model
                interpreter.run()
                    .map_err(|e| anyhow::anyhow!("Execution failed: {}", e))?;

                // Get outputs
                let output_tensors = interpreter.get_outputs()
                    .map_err(|e| anyhow::anyhow!("Failed to get outputs: {}", e))?;

                // Store outputs
                for (i, output_name) in model_stage.outputs.iter().enumerate() {
                    if let Some(tensor) = output_tensors.get(i) {
                        let data = tensor.to_vec();
                        tensor_cache.insert(output_name.clone(), Arc::new(data.clone()));
                        outputs.insert(output_name.clone(), OutputData::Tensor(data));
                        debug!("Output '{}': {} elements, shape {:?}", output_name, tensor.data.len(), tensor.shape());
                    } else {
                        warn!("Expected output '{}' not found in execution results", output_name);
                    }
                }

                info!("  ✓ Stage {}: {} completed ({} outputs)", idx, model_stage.model, output_tensors.len());
            }

            StageDef::Builtin(builtin_stage) => {
                debug!(
                    "Stage {}: Executing builtin '{}'",
                    idx, builtin_stage.builtin
                );

                // Execute builtin operations
                let builtin_outputs = execute_builtin(
                    &builtin_stage.builtin,
                    &builtin_stage.args,
                    &builtin_stage.outputs,
                    &tensor_cache,
                    inputs,
                )?;

                for (name, tensor) in builtin_outputs {
                    tensor_cache.insert(name.clone(), Arc::new(tensor.clone()));
                    outputs.insert(name, OutputData::Tensor(tensor));
                }

                info!("  ✓ Stage {}: {} completed", idx, builtin_stage.builtin);
            }

            StageDef::Loop(loop_stage) => {
                debug!("Stage {}: Loop over '{}'", idx, loop_stage.over);
                // TODO: Implement loop execution for iterative pipelines
                warn!("Loop stages not yet implemented");
                info!("  ✓ Stage {}: loop completed", idx);
            }

            StageDef::Conditional(cond_stage) => {
                debug!("Stage {}: Conditional '{}'", idx, cond_stage.condition);
                // TODO: Implement conditional execution
                warn!("Conditional stages not yet implemented");
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
            if let Some(cached) = tensor_cache.get(tensor_name) {
                outputs.insert(tensor_name.to_string(), OutputData::Tensor(cached.as_ref().clone()));
            }
        }
    }

    Ok(outputs)
}

/// Parse an input string as a tensor.
fn parse_input_tensor(input_str: &str) -> Result<Vec<f32>> {
    // Handle file path inputs (e.g., image files)
    if input_str.ends_with(".png") || input_str.ends_with(".jpg") || input_str.ends_with(".jpeg") {
        return load_image_as_tensor(input_str);
    }

    // Handle JSON array format
    if input_str.starts_with('[') {
        let values: Vec<f32> = serde_json::from_str(input_str)
            .with_context(|| format!("Failed to parse input as JSON array: {}", input_str))?;
        return Ok(values);
    }

    // Handle comma-separated values
    if input_str.contains(',') {
        let values: Result<Vec<f32>, _> = input_str
            .split(',')
            .map(|s| s.trim().parse::<f32>())
            .collect();
        return values.with_context(|| format!("Failed to parse comma-separated values: {}", input_str));
    }

    // Single scalar value
    let value: f32 = input_str.parse()
        .with_context(|| format!("Failed to parse scalar value: {}", input_str))?;
    Ok(vec![value])
}

fn concrete_shape_from_dims(dims: &[DimSpec]) -> Vec<usize> {
    dims.iter()
        .map(|dim| match dim {
            DimSpec::Concrete(size) => *size,
            DimSpec::Symbolic(name) => {
                debug!("Symbolic dim '{}' defaulting to 1", name);
                1
            }
        })
        .collect()
}

fn input_shape_for_node(node: &SerNode) -> Result<Vec<usize>> {
    let dims = node.shape.as_ref().ok_or_else(|| {
        anyhow::anyhow!("Input node {} has no shape information", node.id)
    })?;
    Ok(concrete_shape_from_dims(dims))
}

fn default_tensor_for_node(node: &SerNode) -> Result<Vec<f32>> {
    let shape = input_shape_for_node(node)?;
    let size: usize = shape.iter().product();
    Ok(vec![0.0f32; size])
}

/// Load an image file as a tensor.
fn load_image_as_tensor(path: &str) -> Result<Vec<f32>> {
    // Basic image loading - returns flattened RGB values normalized to [0, 1]
    let _img_bytes = std::fs::read(path)
        .with_context(|| format!("Failed to read image: {}", path))?;

    // For now, return placeholder - actual image decoding would use image crate
    warn!("Image loading not fully implemented, using placeholder for: {}", path);
    Ok(vec![0.5f32; 224 * 224 * 3]) // Placeholder: 224x224x3 image
}

/// Execute a builtin operation.
fn execute_builtin(
    name: &str,
    args: &HashMap<String, hologram_onnx_config::Expr>,
    output_names: &[String],
    tensor_cache: &HashMap<String, Arc<Vec<f32>>>,
    runtime_inputs: &HashMap<String, String>,
) -> Result<HashMap<String, Vec<f32>>> {
    let mut outputs = HashMap::new();

    // Helper to get arg as string
    let get_arg_str = |key: &str| -> Option<&str> {
        args.get(key).and_then(|expr| expr.as_str())
    };

    // Helper to get arg as i64 array (for shape)
    let get_arg_shape = |key: &str| -> Option<Vec<usize>> {
        args.get(key).and_then(|expr| {
            match expr {
                hologram_onnx_config::Expr::Literal(v) => {
                    v.as_array().map(|arr| {
                        arr.iter().filter_map(|v| v.as_i64().map(|i| i as usize)).collect()
                    })
                }
            }
        })
    };

    let default_output = output_names.first()
        .map(|s| s.as_str())
        .unwrap_or("output");

    match name {
        "randn" | "random_normal" => {
            // Generate random normal tensor
            let shape = get_arg_shape("shape")
                .unwrap_or_else(|| vec![1, 4, 64, 64]); // Default latent shape

            let size: usize = shape.iter().product();
            use std::f32::consts::PI;

            // Box-Muller transform for random normal
            let mut rng_state = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(42);

            let mut tensor = Vec::with_capacity(size);
            for _ in 0..size {
                // Simple LCG random
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let u1 = (rng_state as f32) / (u64::MAX as f32);
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let u2 = (rng_state as f32) / (u64::MAX as f32);

                let z = (-2.0 * u1.max(1e-10).ln()).sqrt() * (2.0 * PI * u2).cos();
                tensor.push(z);
            }

            outputs.insert(default_output.to_string(), tensor);
        }

        "concat" | "concatenate" => {
            // Concatenate tensors
            let mut result = Vec::new();
            for tensor_expr in args.values() {
                if let Some(tensor_ref) = tensor_expr.as_str()
                    && let Some(cached) = tensor_cache.get(tensor_ref)
                {
                    result.extend(cached.as_ref().iter().copied());
                }
            }

            outputs.insert(default_output.to_string(), result);
        }

        "encode_text" | "tokenize" => {
            // Text encoding placeholder
            let text_ref = get_arg_str("text").unwrap_or("");
            let text = runtime_inputs.get(text_ref)
                .map(|s| s.as_str())
                .unwrap_or("");

            debug!("Encoding text: {}", text);

            // Return placeholder embeddings
            outputs.insert(default_output.to_string(), vec![0.0f32; 77 * 768]); // CLIP-style embeddings
        }

        _ => {
            warn!("Unknown builtin operation: {}", name);
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

    // Create output dir if it doesn't exist
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("Failed to create output directory: {}", output_dir.display()))?;

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

                if let OutputData::Tensor(tensor) = data {
                    // Get image dimensions from config or infer from tensor
                    let (width, height, channels) = infer_image_dimensions(tensor.len(), output_def);

                    // Convert tensor to u8 image data
                    let image_data = tensor_to_image_data(tensor, width, height, channels)?;

                    // Write using the image crate
                    write_image_data(&image_data, width, height, channels, &output_path)?;

                    info!("  Image saved: {}x{} ({} channels)", width, height, channels);
                } else {
                    warn!("Image output '{}' has non-tensor data", name);
                }
            }

            OutputHandlerType::Audio => {
                let output_path = output_dir.join(format!("{}.wav", name));
                info!("Writing audio output: {}", output_path.display());

                if let OutputData::Tensor(tensor) = data {
                    // Write raw audio samples
                    write_audio_output(tensor, &output_path)?;
                } else {
                    warn!("Audio output '{}' has non-tensor data", name);
                }
            }

            OutputHandlerType::Text => {
                if let OutputData::Text(text) = data {
                    info!("Text output '{}': {}", name, text);
                } else if let OutputData::Tensor(tensor) = data {
                    info!("Text output '{}': {} values", name, tensor.len());
                }
            }

            OutputHandlerType::Json => {
                let output_path = output_dir.join(format!("{}.json", name));
                info!("Writing JSON output: {}", output_path.display());

                if let OutputData::Tensor(tensor) = data {
                    let json = serde_json::to_string_pretty(tensor)?;
                    std::fs::write(&output_path, json)?;
                }
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

/// Infer image dimensions from tensor size and config.
fn infer_image_dimensions(
    tensor_size: usize,
    _output_def: &OutputDef,
) -> (u32, u32, u8) {
    // Default to 512x512 RGB (common for SD)
    let default_width = 512u32;
    let default_height = 512u32;
    let default_channels = 3u8;

    let expected_size = (default_width * default_height * default_channels as u32) as usize;
    if tensor_size == expected_size {
        return (default_width, default_height, default_channels);
    }

    // Try to infer from tensor size
    // Assume RGB (3 channels) by default
    let inferred_channels = 3u8;
    let pixel_count = tensor_size / inferred_channels as usize;
    let dim = (pixel_count as f64).sqrt() as u32;
    if dim * dim * inferred_channels as u32 == tensor_size as u32 {
        return (dim, dim, inferred_channels);
    }

    // Try 4 channels (RGBA)
    let inferred_channels = 4u8;
    let pixel_count = tensor_size / inferred_channels as usize;
    let dim = (pixel_count as f64).sqrt() as u32;
    if dim * dim * inferred_channels as u32 == tensor_size as u32 {
        return (dim, dim, inferred_channels);
    }

    // Fall back to default with warning
    debug!("Could not infer image dimensions from tensor size {}", tensor_size);
    (default_width, default_height, default_channels)
}

/// Convert tensor data to image u8 data.
///
/// Handles NCHW layout (default for diffusion models) and normalizes from [-1, 1] to [0, 255].
fn tensor_to_image_data(tensor: &[f32], width: u32, height: u32, channels: u8) -> Result<Vec<u8>> {
    let expected_size = (width * height * channels as u32) as usize;

    // Check if we have batch dimension (NCHW layout)
    let data = if tensor.len() == expected_size {
        tensor
    } else if tensor.len() >= expected_size {
        // Take first image from batch
        &tensor[..expected_size]
    } else {
        anyhow::bail!(
            "Tensor size {} doesn't match expected {}x{}x{}={}",
            tensor.len(), width, height, channels, expected_size
        );
    };

    // Detect if NCHW layout (most diffusion models use this)
    // For NCHW with 3 channels: tensor is [C, H, W] = [3, H, W]
    let channel_stride = (width * height) as usize;
    let is_nchw = tensor.len() >= channel_stride * channels as usize;

    let mut result = vec![0u8; expected_size];

    for h in 0..height as usize {
        for w in 0..width as usize {
            for c in 0..channels as usize {
                let src_idx = if is_nchw && channels > 1 {
                    // NCHW: [channel][height][width]
                    c * channel_stride + h * width as usize + w
                } else {
                    // NHWC: [height][width][channel]
                    h * width as usize * channels as usize + w * channels as usize + c
                };

                let dst_idx = h * width as usize * channels as usize + w * channels as usize + c;

                // Normalize from [-1, 1] to [0, 255]
                let value = if src_idx < data.len() {
                    let v = data[src_idx];
                    // Clamp and scale: [-1, 1] -> [0, 255]
                    ((v + 1.0) / 2.0 * 255.0).clamp(0.0, 255.0) as u8
                } else {
                    0
                };

                result[dst_idx] = value;
            }
        }
    }

    Ok(result)
}

/// Write image data to file.
fn write_image_data(data: &[u8], width: u32, height: u32, channels: u8, path: &Path) -> Result<()> {
    use image::{ImageBuffer, Rgb, Rgba, Luma};

    match channels {
        1 => {
            let img: ImageBuffer<Luma<u8>, Vec<u8>> =
                ImageBuffer::from_raw(width, height, data.to_vec())
                    .ok_or_else(|| anyhow::anyhow!("Failed to create grayscale image buffer"))?;
            img.save(path)?;
        }
        3 => {
            let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
                ImageBuffer::from_raw(width, height, data.to_vec())
                    .ok_or_else(|| anyhow::anyhow!("Failed to create RGB image buffer"))?;
            img.save(path)?;
        }
        4 => {
            let img: ImageBuffer<Rgba<u8>, Vec<u8>> =
                ImageBuffer::from_raw(width, height, data.to_vec())
                    .ok_or_else(|| anyhow::anyhow!("Failed to create RGBA image buffer"))?;
            img.save(path)?;
        }
        _ => anyhow::bail!("Unsupported channel count: {}", channels),
    }

    Ok(())
}

/// Write audio output to WAV file.
fn write_audio_output(samples: &[f32], path: &Path) -> Result<()> {
    // Simple WAV writer (mono, 44100 Hz)
    use std::io::Write;

    let sample_rate = 44100u32;
    let num_channels = 1u16;
    let bits_per_sample = 16u16;
    let byte_rate = sample_rate * num_channels as u32 * bits_per_sample as u32 / 8;
    let block_align = num_channels * bits_per_sample / 8;

    // Convert f32 samples to i16
    let samples_i16: Vec<i16> = samples
        .iter()
        .map(|&s| (s.clamp(-1.0, 1.0) * 32767.0) as i16)
        .collect();

    let data_size = (samples_i16.len() * 2) as u32;
    let file_size = 36 + data_size;

    let mut file = std::fs::File::create(path)?;

    // RIFF header
    file.write_all(b"RIFF")?;
    file.write_all(&file_size.to_le_bytes())?;
    file.write_all(b"WAVE")?;

    // fmt chunk
    file.write_all(b"fmt ")?;
    file.write_all(&16u32.to_le_bytes())?; // chunk size
    file.write_all(&1u16.to_le_bytes())?; // PCM format
    file.write_all(&num_channels.to_le_bytes())?;
    file.write_all(&sample_rate.to_le_bytes())?;
    file.write_all(&byte_rate.to_le_bytes())?;
    file.write_all(&block_align.to_le_bytes())?;
    file.write_all(&bits_per_sample.to_le_bytes())?;

    // data chunk
    file.write_all(b"data")?;
    file.write_all(&data_size.to_le_bytes())?;
    for sample in samples_i16 {
        file.write_all(&sample.to_le_bytes())?;
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

    #[test]
    fn test_concrete_shape_from_spec_symbolic_defaults() {
        let input = SerNode {
            id: 0,
            node: SerNodeKind::Input {
                name: "tokens".to_string(),
            },
            dtype: None,
            shape: Some(vec![DimSpec::Concrete(2), DimSpec::Symbolic("n".to_string())]),
        };

        let shape = input_shape_for_node(&input).unwrap();
        assert_eq!(shape, vec![2, 1]);
    }

    #[test]
    fn test_default_tensor_for_input_size() {
        let input = SerNode {
            id: 1,
            node: SerNodeKind::Input {
                name: "tokens".to_string(),
            },
            dtype: None,
            shape: Some(vec![DimSpec::Concrete(1), DimSpec::Concrete(77)]),
        };

        let data = default_tensor_for_node(&input).unwrap();
        assert_eq!(data.len(), 77);
    }
}
