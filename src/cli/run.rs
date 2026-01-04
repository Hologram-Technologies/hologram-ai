//! Run ONNX pipelines with unified configuration.
//!
//! This module provides the `run` command which:
//! - Loads a unified config from a TOML file
//! - Loads pre-compiled .holo files (compile with `hologram-onnx compile` first)
//! - Executes the pipeline with provided inputs using parallel scheduler
//! - Processes outputs using configured handlers
//! - Supports loop stages for diffusion model denoising

use anyhow::{Context, Result};
// Parallel execution functions are not available in simplified version
// use hologram_compiler::{execute_schedule_layerwise, execute_schedule_rayon, ParallelCompiledGraph};
use crate::config::{
    ModelDef, OutputDef, OutputHandlerType, RuntimeConfig, StageDef, UnifiedConfig,
};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info, warn};

// =============================================================================
// Diffusion Scheduler
// =============================================================================

/// DDIM scheduler for diffusion models.
/// Implements Denoising Diffusion Implicit Models scheduling.
#[derive(Clone)]
pub struct DdimScheduler {
    /// Number of inference steps
    num_inference_steps: usize,
    /// Timesteps for each step
    timesteps: Vec<i64>,
    /// Alpha cumulative products
    alphas_cumprod: Vec<f32>,
    /// Current step index
    current_step: usize,
}

impl DdimScheduler {
    /// Create a new DDIM scheduler with the given number of steps.
    pub fn new(num_inference_steps: usize) -> Self {
        let beta_start = 0.00085f32;
        let beta_end = 0.012f32;
        let num_train_timesteps = 1000usize;

        // Linear beta schedule
        let betas: Vec<f32> = (0..num_train_timesteps)
            .map(|i| {
                let t = i as f32 / (num_train_timesteps - 1) as f32;
                (beta_start.sqrt() + t * (beta_end.sqrt() - beta_start.sqrt())).powi(2)
            })
            .collect();

        // Compute alphas and cumulative products
        let alphas: Vec<f32> = betas.iter().map(|b| 1.0 - b).collect();
        let mut alphas_cumprod = Vec::with_capacity(num_train_timesteps);
        let mut cumprod = 1.0f32;
        for alpha in &alphas {
            cumprod *= alpha;
            alphas_cumprod.push(cumprod);
        }

        // Compute timesteps (evenly spaced from num_train_timesteps-1 to 0)
        let step_ratio = num_train_timesteps / num_inference_steps;
        let timesteps: Vec<i64> = (0..num_inference_steps)
            .rev()
            .map(|i| (i * step_ratio) as i64)
            .collect();

        Self {
            num_inference_steps,
            timesteps,
            alphas_cumprod,
            current_step: 0,
        }
    }

    /// Get the current timestep value.
    pub fn current_timestep(&self) -> i64 {
        self.timesteps.get(self.current_step).copied().unwrap_or(0)
    }

    /// Perform one DDIM denoising step.
    ///
    /// # Arguments
    /// * `latents` - Current latent tensor [B, C, H, W]
    /// * `noise_pred` - Predicted noise from UNet [B, C, H, W]
    /// * `step_index` - Current step index
    ///
    /// # Returns
    /// Updated latents after one denoising step.
    pub fn step(&self, latents: &[f32], noise_pred: &[f32], step_index: usize) -> Vec<f32> {
        let timestep = self.timesteps[step_index] as usize;

        // Get alpha values
        let alpha_prod_t = self.alphas_cumprod[timestep];
        let alpha_prod_t_prev = if step_index + 1 < self.num_inference_steps {
            let prev_timestep = self.timesteps[step_index + 1] as usize;
            self.alphas_cumprod[prev_timestep]
        } else {
            1.0 // Final step
        };

        let beta_prod_t = 1.0 - alpha_prod_t;

        // Compute predicted original sample (x0)
        // x0 = (latents - sqrt(1-alpha) * noise_pred) / sqrt(alpha)
        let sqrt_alpha = alpha_prod_t.sqrt();
        let sqrt_one_minus_alpha = beta_prod_t.sqrt();

        let mut pred_original: Vec<f32> = latents.iter()
            .zip(noise_pred.iter())
            .map(|(x, n)| (x - sqrt_one_minus_alpha * n) / sqrt_alpha.max(1e-8))
            .collect();

        // Clamp predicted original sample for stability
        for v in &mut pred_original {
            *v = v.clamp(-1.0, 1.0);
        }

        // Compute the previous sample (DDIM formula)
        // x_{t-1} = sqrt(alpha_{t-1}) * x0 + sqrt(1 - alpha_{t-1}) * noise_pred
        let sqrt_alpha_prev = alpha_prod_t_prev.sqrt();
        let sqrt_one_minus_alpha_prev = (1.0 - alpha_prod_t_prev).sqrt();

        pred_original.iter()
            .zip(noise_pred.iter())
            .map(|(x0, n)| sqrt_alpha_prev * x0 + sqrt_one_minus_alpha_prev * n)
            .collect()
    }
}

/// Apply classifier-free guidance to noise predictions.
///
/// # Arguments
/// * `noise_pred_uncond` - Unconditional noise prediction
/// * `noise_pred_cond` - Conditional noise prediction
/// * `guidance_scale` - CFG scale (typically 7.5)
///
/// # Returns
/// Guided noise prediction.
pub fn apply_cfg(noise_pred_uncond: &[f32], noise_pred_cond: &[f32], guidance_scale: f32) -> Vec<f32> {
    noise_pred_uncond.iter()
        .zip(noise_pred_cond.iter())
        .map(|(u, c)| u + guidance_scale * (c - u))
        .collect()
}

// =============================================================================
// Pipeline Execution Context
// =============================================================================

/// Execution context for pipeline stages.
/// Holds all state needed during pipeline execution.
struct PipelineContext<'a> {
    /// Compiled .holo model paths
    holo_models: &'a HashMap<String, std::path::PathBuf>,
    /// Runtime string inputs from command line
    runtime_inputs: &'a HashMap<String, String>,
    /// Cached tensors from previous stages
    tensor_cache: HashMap<String, Arc<Vec<f32>>>,
    /// Loop variables (name -> current value)
    loop_vars: HashMap<String, i64>,
    /// Diffusion scheduler (if initialized)
    scheduler: Option<DdimScheduler>,
    /// Guidance scale for CFG
    guidance_scale: f32,
    /// Random seed
    seed: u64,
    /// Runtime configuration
    runtime_config: &'a RuntimeConfig,
}

// serde_json is used for parsing input tensors via serde_json::from_str

/// Run an ONNX pipeline from a unified config file.
///
/// # Arguments
///
/// * `config_path` - Path to the unified config TOML file
/// * `inputs` - Runtime inputs as key=value pairs
/// * `output_dir` - Optional directory for output files
///
/// # Returns
///
/// Returns Ok(()) on success, or an error if execution fails.
///
/// # Note
///
/// Models must be pre-compiled with `hologram-onnx compile` before running.
/// The .holo files are loaded and executed using the parallel scheduler (rayon).
pub fn run_command(
    config_path: &Path,
    inputs: &[String],
    output_dir: Option<&Path>,
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

    // Get compiled .holo model paths
    let holo_models = get_holo_model_paths(&config, config_dir)?;

    // Execute using parallel scheduler
    info!("Executing pipeline with parallel scheduler...");
    let outputs = execute_pipeline(&config, &holo_models, &runtime_inputs)?;

    // Process outputs
    info!("Processing outputs...");
    process_outputs(&config, &outputs, output_dir)?;

    info!("Pipeline execution complete!");
    Ok(())
}

/// Get compiled .holo model paths from config.
fn get_holo_model_paths(
    config: &UnifiedConfig,
    config_dir: &Path,
) -> Result<HashMap<String, std::path::PathBuf>> {
    let mut holo_paths = HashMap::new();

    for (name, model_def) in &config.models {
        let holo_path = get_compiled_path(model_def, config_dir);

        if !holo_path.exists() {
            anyhow::bail!(
                "Compiled .holo file not found: {} (for model '{}'). \
                 Run 'hologram-onnx compile' first.",
                holo_path.display(),
                name
            );
        }

        holo_paths.insert(name.clone(), holo_path);
    }

    Ok(holo_paths)
}

/// Execute pipeline using the parallel scheduler (rayon).
///
/// This loads pre-compiled .holo files and executes with rayon parallelism.
fn execute_pipeline(
    config: &UnifiedConfig,
    holo_models: &HashMap<String, std::path::PathBuf>,
    inputs: &HashMap<String, String>,
) -> Result<HashMap<String, OutputData>> {
    // Parse guidance scale and seed from inputs
    let guidance_scale = inputs.get("guidance_scale")
        .and_then(|s| s.parse().ok())
        .unwrap_or(7.5f32);

    let seed = inputs.get("seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(42)
        });

    // Initialize context
    let mut ctx = PipelineContext {
        holo_models,
        runtime_inputs: inputs,
        tensor_cache: HashMap::new(),
        loop_vars: HashMap::new(),
        scheduler: None,
        guidance_scale,
        seed,
        runtime_config: &config.runtime,
    };

    // Execute all stages
    execute_stages(&config.stages, &mut ctx, 0)?;

    // Convert tensor cache to output data
    let mut outputs: HashMap<String, OutputData> = HashMap::new();
    for (name, tensor) in &ctx.tensor_cache {
        outputs.insert(name.clone(), OutputData::Tensor(tensor.as_ref().clone()));
    }

    // Ensure all config outputs are present
    for output_def in config.outputs.values() {
        let tensor_name = output_def.tensor();
        if !outputs.contains_key(tensor_name)
            && let Some(cached) = ctx.tensor_cache.get(tensor_name)
        {
            outputs.insert(tensor_name.to_string(), OutputData::Tensor(cached.as_ref().clone()));
        }
    }

    Ok(outputs)
}

/// Execute a list of stages recursively.
/// This allows loop stages to execute their nested stages.
fn execute_stages(
    stages: &[StageDef],
    ctx: &mut PipelineContext,
    depth: usize,
) -> Result<()> {
    let indent = "  ".repeat(depth);

    for (idx, stage) in stages.iter().enumerate() {
        match stage {
            StageDef::Model(model_stage) => {
                let holo_path = ctx.holo_models.get(&model_stage.model).ok_or_else(|| {
                    anyhow::anyhow!("Model '{}' not found in compiled models", model_stage.model)
                })?;

                info!(
                    "{}▶ Stage {}: running model '{}'",
                    indent, idx, model_stage.model
                );

                let _stage_start = Instant::now();

                // TODO: Parallel execution is not implemented in simplified version
                // The parallel execution runtime (ParallelCompiledGraph, execute_schedule_rayon,
                // execute_schedule_layerwise) has been removed in the simplified version.
                // This needs to be reimplemented using the new execution model.
                let _ = holo_path; // Silence unused warning
                return Err(anyhow::anyhow!(
                    "Model execution is not implemented in the simplified version. \
                     Parallel execution runtime needs to be reimplemented."
                ));

                // Stage completion (timing disabled in stub)
                info!(
                    "{}✓ Stage {}: {} (not executed - stub)",
                    indent,
                    idx,
                    model_stage.model
                );
            }

            StageDef::Builtin(builtin_stage) => {
                debug!(
                    "{}Stage {}: Executing builtin '{}'",
                    indent, idx, builtin_stage.builtin
                );

                let builtin_outputs = execute_builtin_with_context(
                    &builtin_stage.builtin,
                    &builtin_stage.args,
                    &builtin_stage.outputs,
                    ctx,
                )?;

                for (name, tensor) in builtin_outputs {
                    ctx.tensor_cache.insert(name, Arc::new(tensor));
                }

                info!("{}✓ Stage {}: {} completed", indent, idx, builtin_stage.builtin);
            }

            StageDef::Loop(loop_stage) => {
                info!("{}▶ Loop stage: {} as '{}'", indent, loop_stage.over, loop_stage.as_var);

                // Parse the iteration count from the "over" expression
                let iterations = parse_loop_range(&loop_stage.over, ctx)?;

                info!("{}  Running {} iterations", indent, iterations);

                let loop_start = Instant::now();

                for i in 0..iterations {
                    // Set loop variable
                    ctx.loop_vars.insert(loop_stage.as_var.clone(), i as i64);

                    // Update scheduler step if applicable
                    if let Some(ref mut scheduler) = ctx.scheduler {
                        scheduler.current_step = i;
                    }

                    if i % 5 == 0 || i == iterations - 1 {
                        info!("{}  Step {}/{}", indent, i + 1, iterations);
                    }

                    // Execute nested stages
                    execute_stages(&loop_stage.stages, ctx, depth + 1)?;
                }

                // Clean up loop variable
                ctx.loop_vars.remove(&loop_stage.as_var);

                info!(
                    "{}✓ Loop completed in {:?}",
                    indent,
                    loop_start.elapsed()
                );
            }

            StageDef::Conditional(cond_stage) => {
                debug!("{}Stage {}: Conditional '{}'", indent, idx, cond_stage.condition);

                // Evaluate condition
                let condition_met = evaluate_condition(&cond_stage.condition, ctx)?;

                if condition_met {
                    info!("{}  Condition '{}' is true, executing then branch", indent, cond_stage.condition);
                    execute_stages(&cond_stage.then_stages, ctx, depth + 1)?;
                } else if !cond_stage.else_stages.is_empty() {
                    info!("{}  Condition '{}' is false, executing else branch", indent, cond_stage.condition);
                    execute_stages(&cond_stage.else_stages, ctx, depth + 1)?;
                }
            }
        }
    }

    Ok(())
}

/// Resolve a tensor reference to actual data.
/// Handles cached tensors, runtime inputs, loop variables, and special expressions.
fn resolve_tensor_ref(tensor_ref: &str, ctx: &PipelineContext) -> Result<Vec<f32>> {
    // Check tensor cache first
    if let Some(cached) = ctx.tensor_cache.get(tensor_ref) {
        return Ok(cached.as_ref().clone());
    }

    // Check loop variables (return as single-element tensor)
    if let Some(&value) = ctx.loop_vars.get(tensor_ref) {
        return Ok(vec![value as f32]);
    }

    // Check for timestep reference from scheduler
    if (tensor_ref == "timestep" || tensor_ref == "t")
        && let Some(ref scheduler) = ctx.scheduler {
            return Ok(vec![scheduler.current_timestep() as f32]);
        }

    // Check runtime inputs
    if let Some(input_str) = ctx.runtime_inputs.get(tensor_ref) {
        return parse_input_tensor(input_str);
    }

    // Not found - return empty with warning
    warn!("Tensor reference '{}' not found, using empty tensor", tensor_ref);
    Ok(vec![0.0f32; 1])
}

/// Parse a loop range expression like "range(steps)" or "range(20)".
fn parse_loop_range(expr: &str, ctx: &PipelineContext) -> Result<usize> {
    let expr = expr.trim();

    // Handle range(n) syntax
    if let Some(inner) = expr.strip_prefix("range(").and_then(|s| s.strip_suffix(')')) {
        let inner = inner.trim();

        // Try parsing as a number
        if let Ok(n) = inner.parse::<usize>() {
            return Ok(n);
        }

        // Try as a runtime input reference
        if let Some(value_str) = ctx.runtime_inputs.get(inner)
            && let Ok(n) = value_str.parse::<usize>() {
                return Ok(n);
            }

        // Try as a loop variable
        if let Some(&n) = ctx.loop_vars.get(inner) {
            return Ok(n as usize);
        }

        anyhow::bail!("Cannot resolve loop range variable: {}", inner);
    }

    // Try parsing as a plain number
    if let Ok(n) = expr.parse::<usize>() {
        return Ok(n);
    }

    anyhow::bail!("Invalid loop range expression: {}", expr);
}

/// Evaluate a simple condition expression.
fn evaluate_condition(condition: &str, ctx: &PipelineContext) -> Result<bool> {
    let condition = condition.trim();

    // Handle "step > N" patterns
    if let Some((var, rest)) = condition.split_once('>') {
        let var = var.trim();
        let threshold: i64 = rest.trim().parse()
            .with_context(|| format!("Invalid threshold in condition: {}", condition))?;

        if let Some(&val) = ctx.loop_vars.get(var) {
            return Ok(val > threshold);
        }
    }

    // Handle "step < N" patterns
    if let Some((var, rest)) = condition.split_once('<') {
        let var = var.trim();
        let threshold: i64 = rest.trim().parse()
            .with_context(|| format!("Invalid threshold in condition: {}", condition))?;

        if let Some(&val) = ctx.loop_vars.get(var) {
            return Ok(val < threshold);
        }
    }

    // Handle "step == N" patterns
    if let Some((var, rest)) = condition.split_once("==") {
        let var = var.trim();
        let value: i64 = rest.trim().parse()
            .with_context(|| format!("Invalid value in condition: {}", condition))?;

        if let Some(&val) = ctx.loop_vars.get(var) {
            return Ok(val == value);
        }
    }

    // Default to false for unknown conditions
    warn!("Cannot evaluate condition: {}", condition);
    Ok(false)
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

/// Load an image file as a tensor.
fn load_image_as_tensor(path: &str) -> Result<Vec<f32>> {
    // Basic image loading - returns flattened RGB values normalized to [0, 1]
    let _img_bytes = std::fs::read(path)
        .with_context(|| format!("Failed to read image: {}", path))?;

    // For now, return placeholder - actual image decoding would use image crate
    warn!("Image loading not fully implemented, using placeholder for: {}", path);
    Ok(vec![0.5f32; 224 * 224 * 3]) // Placeholder: 224x224x3 image
}

/// Execute a builtin operation with full pipeline context.
fn execute_builtin_with_context(
    name: &str,
    args: &HashMap<String, crate::config::Expr>,
    output_names: &[String],
    ctx: &mut PipelineContext,
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
                crate::config::Expr::Literal(v) => {
                    v.as_array().map(|arr| {
                        arr.iter().filter_map(|v| v.as_i64().map(|i| i as usize)).collect()
                    })
                }
            }
        })
    };

    // Helper to get arg as f32
    let get_arg_f32 = |key: &str| -> Option<f32> {
        args.get(key).and_then(|expr| {
            match expr {
                crate::config::Expr::Literal(v) => v.as_f64().map(|f| f as f32),
            }
        })
    };

    // Helper to get arg as i64
    let get_arg_i64 = |key: &str| -> Option<i64> {
        args.get(key).and_then(|expr| {
            match expr {
                crate::config::Expr::Literal(v) => v.as_i64(),
            }
        })
    };

    let default_output = output_names.first()
        .map(|s| s.as_str())
        .unwrap_or("output");

    match name {
        "randn" | "random_normal" => {
            // Generate random normal tensor with optional seed
            let shape = get_arg_shape("shape")
                .unwrap_or_else(|| vec![1, 4, 64, 64]); // Default latent shape

            let size: usize = shape.iter().product();
            use std::f32::consts::PI;

            // Use seed from args or context
            let seed = get_arg_i64("seed")
                .map(|s| s as u64)
                .unwrap_or(ctx.seed);

            let mut rng_state = seed;

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

            info!("  Generated random latent: {:?} ({} elements)", shape, size);
            outputs.insert(default_output.to_string(), tensor);
        }

        "concat" | "concatenate" => {
            // Concatenate tensors along batch dimension
            let mut result = Vec::new();
            for tensor_expr in args.values() {
                if let Some(tensor_ref) = tensor_expr.as_str()
                    && let Some(cached) = ctx.tensor_cache.get(tensor_ref)
                {
                    result.extend(cached.as_ref().iter().copied());
                }
            }

            outputs.insert(default_output.to_string(), result);
        }

        "scheduler_init" | "init_scheduler" => {
            // Initialize the diffusion scheduler
            let num_steps = get_arg_i64("steps")
                .or_else(|| ctx.runtime_inputs.get("steps").and_then(|s| s.parse().ok()))
                .unwrap_or(20) as usize;

            info!("  Initializing DDIM scheduler with {} steps", num_steps);
            ctx.scheduler = Some(DdimScheduler::new(num_steps));

            // Output the timesteps for debugging
            if let Some(ref scheduler) = ctx.scheduler {
                outputs.insert(default_output.to_string(),
                    scheduler.timesteps.iter().map(|&t| t as f32).collect());
            }
        }

        "scheduler_step" | "denoise_step" => {
            // Perform one DDIM denoising step
            let latent_ref = get_arg_str("latent").or(get_arg_str("sample")).unwrap_or("latent");
            let noise_ref = get_arg_str("noise_pred").or(get_arg_str("noise")).unwrap_or("noise_pred");

            let latent = ctx.tensor_cache.get(latent_ref)
                .map(|t| t.as_ref().clone())
                .unwrap_or_else(|| {
                    warn!("Latent tensor '{}' not found for scheduler step", latent_ref);
                    vec![0.0f32; 1]
                });

            let noise_pred = ctx.tensor_cache.get(noise_ref)
                .map(|t| t.as_ref().clone())
                .unwrap_or_else(|| {
                    warn!("Noise prediction '{}' not found for scheduler step", noise_ref);
                    vec![0.0f32; latent.len()]
                });

            if let Some(ref scheduler) = ctx.scheduler {
                let step_idx = scheduler.current_step;
                let updated_latent = scheduler.step(&latent, &noise_pred, step_idx);
                outputs.insert(default_output.to_string(), updated_latent);
            } else {
                warn!("Scheduler not initialized, cannot perform denoise step");
                outputs.insert(default_output.to_string(), latent);
            }
        }

        "cfg" | "classifier_free_guidance" => {
            // Apply classifier-free guidance
            let uncond_ref = get_arg_str("uncond").or(get_arg_str("unconditional")).unwrap_or("noise_uncond");
            let cond_ref = get_arg_str("cond").or(get_arg_str("conditional")).unwrap_or("noise_cond");
            let scale = get_arg_f32("scale").unwrap_or(ctx.guidance_scale);

            let uncond = ctx.tensor_cache.get(uncond_ref)
                .map(|t| t.as_ref().clone())
                .unwrap_or_else(|| vec![0.0f32; 1]);

            let cond = ctx.tensor_cache.get(cond_ref)
                .map(|t| t.as_ref().clone())
                .unwrap_or_else(|| vec![0.0f32; 1]);

            let guided = apply_cfg(&uncond, &cond, scale);
            outputs.insert(default_output.to_string(), guided);
        }

        "get_timestep" | "timestep" => {
            // Get current timestep from scheduler
            if let Some(ref scheduler) = ctx.scheduler {
                let t = scheduler.current_timestep();
                outputs.insert(default_output.to_string(), vec![t as f32]);
            } else {
                // Use loop variable if available
                let step = ctx.loop_vars.get("step").copied().unwrap_or(0);
                outputs.insert(default_output.to_string(), vec![step as f32]);
            }
        }

        "scale_latent" | "vae_scale" => {
            // Scale latent for VAE decoder (SD uses 1/0.18215)
            let latent_ref = get_arg_str("latent").unwrap_or("latent");
            let scale = get_arg_f32("scale").unwrap_or(1.0 / 0.18215);

            if let Some(latent) = ctx.tensor_cache.get(latent_ref) {
                let scaled: Vec<f32> = latent.iter().map(|&v| v * scale).collect();
                outputs.insert(default_output.to_string(), scaled);
            } else {
                warn!("Latent '{}' not found for scaling", latent_ref);
            }
        }

        "tokenize" | "encode_text" => {
            // CLIP text tokenization
            // For now, returns a simple token sequence
            let text_ref = get_arg_str("text").unwrap_or("prompt");
            let text = ctx.runtime_inputs.get(text_ref)
                .map(|s| s.as_str())
                .unwrap_or("a photo");

            let max_length = get_arg_i64("max_length").unwrap_or(77) as usize;

            info!("  Tokenizing text: \"{}\"", text);

            // Simple tokenization: convert chars to token IDs
            // In a real implementation, this would use a proper BPE tokenizer
            let mut tokens: Vec<f32> = Vec::with_capacity(max_length);

            // Start token (49406 for CLIP)
            tokens.push(49406.0);

            // Convert text to pseudo-tokens (simplified)
            for (i, c) in text.chars().take(max_length - 2).enumerate() {
                // Simple mapping - in reality would use BPE vocabulary
                let token_id = match c {
                    'a'..='z' => 320 + (c as u32 - 'a' as u32),
                    'A'..='Z' => 320 + (c as u32 - 'A' as u32),
                    ' ' => 267,
                    '.' => 269,
                    ',' => 268,
                    _ => 259, // Unknown token
                };
                tokens.push(token_id as f32);
                if i >= max_length - 2 {
                    break;
                }
            }

            // End token (49407 for CLIP)
            tokens.push(49407.0);

            // Pad to max_length
            while tokens.len() < max_length {
                tokens.push(49407.0); // Pad with end token
            }

            outputs.insert(default_output.to_string(), tokens);
        }

        "zeros" => {
            // Generate zeros tensor
            let shape = get_arg_shape("shape").unwrap_or_else(|| vec![1]);
            let size: usize = shape.iter().product();
            outputs.insert(default_output.to_string(), vec![0.0f32; size]);
        }

        "ones" => {
            // Generate ones tensor
            let shape = get_arg_shape("shape").unwrap_or_else(|| vec![1]);
            let size: usize = shape.iter().product();
            outputs.insert(default_output.to_string(), vec![1.0f32; size]);
        }

        "copy" | "clone" => {
            // Copy a tensor
            let src_ref = get_arg_str("src").or(get_arg_str("input")).unwrap_or("");
            if let Some(src) = ctx.tensor_cache.get(src_ref) {
                outputs.insert(default_output.to_string(), src.as_ref().clone());
            }
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
#[cfg(feature = "image-output")]
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

/// Write image data to file (stub for when image feature is disabled).
#[cfg(not(feature = "image-output"))]
fn write_image_data(_data: &[u8], _width: u32, _height: u32, _channels: u8, _path: &Path) -> Result<()> {
    anyhow::bail!("Image output is not enabled. Rebuild with --features image-output")
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

        let result = run_command(&config_path, &[], None);
        assert!(result.is_err());
    }
}
