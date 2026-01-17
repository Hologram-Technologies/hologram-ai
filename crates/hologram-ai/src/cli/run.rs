//! Run ONNX pipelines with unified configuration.
//!
//! This module provides the `run` command which:
//! - Loads a unified config from a TOML file
//! - Loads pre-compiled .holo files (compile with `hologram-onnx compile` first)
//! - Executes the pipeline with provided inputs using parallel scheduler
//! - Processes outputs using configured handlers
//! - Supports loop stages for diffusion model denoising
//!
//! Also provides `run_direct_command` for simple .holo execution with --prompt flag
//! for T5 and other text models.

use crate::config::{
    ModelDef, OutputDef, OutputHandlerType, RuntimeConfig, StageDef, UnifiedConfig,
};
use crate::runtime::Tensor;
use crate::tokenizers::Tokenizer;
use anyhow::{Context, Result};
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

        let mut pred_original: Vec<f32> = latents
            .iter()
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

        pred_original
            .iter()
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
pub fn apply_cfg(
    noise_pred_uncond: &[f32],
    noise_pred_cond: &[f32],
    guidance_scale: f32,
) -> Vec<f32> {
    noise_pred_uncond
        .iter()
        .zip(noise_pred_cond.iter())
        .map(|(u, c)| u + guidance_scale * (c - u))
        .collect()
}

// =============================================================================
// Pipeline Execution Context
// =============================================================================

/// Execution context for pipeline stages.
/// Holds all state needed during pipeline execution.
#[allow(dead_code)] // Some fields used only in full implementation
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
    /// Cached model executors (path -> executor) to avoid reloading
    executor_cache: HashMap<std::path::PathBuf, crate::runtime::ModelExecutor>,
    /// Loaded tokenizer (if configured)
    tokenizer: Option<Box<dyn crate::tokenizers::Tokenizer>>,
    /// Guidance scale for CFG
    guidance_scale: f32,
    /// Random seed
    seed: u64,
    /// Runtime configuration
    runtime_config: &'a RuntimeConfig,
    /// Generated outputs (e.g., detokenized text) for output handlers
    generated_outputs: HashMap<String, String>,
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
pub fn run_command(config_path: &Path, inputs: &[String], output_dir: Option<&Path>) -> Result<()> {
    info!("Loading pipeline config: {}", config_path.display());

    // Load the unified config
    let config = UnifiedConfig::from_file(config_path)
        .with_context(|| format!("Failed to load config from {}", config_path.display()))?;

    // Get the config directory for resolving relative paths
    let config_dir = config_path.parent().unwrap_or_else(|| Path::new("."));

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
    let outputs = execute_pipeline(&config, &holo_models, &runtime_inputs, config_dir)?;

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
    config_dir: &Path,
) -> Result<HashMap<String, OutputData>> {
    // Parse guidance scale and seed from inputs
    let guidance_scale = inputs
        .get("guidance_scale")
        .and_then(|s| s.parse().ok())
        .unwrap_or(7.5f32);

    let seed = inputs
        .get("seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(42)
        });

    // Initialize context
    // Load tokenizer if configured
    let tokenizer = if let Some(tokenizer_config) = &config.tokenizer {
        let mut resolved = tokenizer_config.clone();
        let vocab_path = Path::new(&resolved.vocab_path);
        if !vocab_path.is_absolute() && !vocab_path.exists() {
            resolved.vocab_path = config_dir.join(vocab_path).display().to_string();
        }

        if let Some(ref merges_path) = resolved.merges_path {
            let merges = Path::new(merges_path);
            if !merges.is_absolute() && !merges.exists() {
                resolved.merges_path = Some(config_dir.join(merges).display().to_string());
            }
        }

        match crate::tokenizers::load_tokenizer(&resolved) {
            Ok(tok) => {
                info!(
                    "Loaded {} tokenizer (vocab size: {})",
                    tok.tokenizer_type(),
                    tok.vocab_size()
                );
                Some(tok)
            }
            Err(e) => {
                warn!(
                    "Failed to load tokenizer: {}. Using fallback tokenization.",
                    e
                );
                None
            }
        }
    } else {
        None
    };

    let mut ctx = PipelineContext {
        holo_models,
        runtime_inputs: inputs,
        tensor_cache: HashMap::new(),
        loop_vars: HashMap::new(),
        scheduler: None,
        executor_cache: HashMap::new(),
        tokenizer,
        guidance_scale,
        seed,
        runtime_config: &config.runtime,
        generated_outputs: HashMap::new(),
    };

    // Execute all stages
    execute_stages(&config.stages, &mut ctx, 0)?;

    // Convert tensor cache and generated outputs to output data
    let mut outputs: HashMap<String, OutputData> = HashMap::new();

    // First add text outputs (generated by decode builtin)
    for (name, text) in &ctx.generated_outputs {
        outputs.insert(name.clone(), OutputData::Text(text.clone()));
    }

    // Then add tensor outputs (only if not already present as text)
    for (name, tensor) in &ctx.tensor_cache {
        if !outputs.contains_key(name) {
            outputs.insert(name.clone(), OutputData::Tensor(tensor.as_ref().clone()));
        }
    }

    // Ensure all config outputs are present
    for output_def in config.outputs.values() {
        let tensor_name = output_def.tensor();
        if !outputs.contains_key(tensor_name)
            && let Some(cached) = ctx.tensor_cache.get(tensor_name)
        {
            outputs.insert(
                tensor_name.to_string(),
                OutputData::Tensor(cached.as_ref().clone()),
            );
        }
    }

    Ok(outputs)
}

/// Execute a list of stages recursively.
/// This allows loop stages to execute their nested stages.
fn execute_stages(stages: &[StageDef], ctx: &mut PipelineContext, depth: usize) -> Result<()> {
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

                let stage_start = Instant::now();

                // Execute model using hologram runtime
                let outputs =
                    execute_model_stage(holo_path, &model_stage.inputs, &model_stage.outputs, ctx)?;

                // Cache outputs
                for (name, tensor) in outputs {
                    ctx.tensor_cache.insert(name, Arc::new(tensor));
                }

                info!(
                    "{}✓ Stage {}: {} completed in {:?}",
                    indent,
                    idx,
                    model_stage.model,
                    stage_start.elapsed()
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

                info!(
                    "{}✓ Stage {}: {} completed",
                    indent, idx, builtin_stage.builtin
                );
            }

            StageDef::Loop(loop_stage) => {
                info!(
                    "{}▶ Loop stage: {} as '{}'",
                    indent, loop_stage.over, loop_stage.as_var
                );

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

                info!("{}✓ Loop completed in {:?}", indent, loop_start.elapsed());
            }

            StageDef::Conditional(cond_stage) => {
                debug!(
                    "{}Stage {}: Conditional '{}'",
                    indent, idx, cond_stage.condition
                );

                // Evaluate condition
                let condition_met = evaluate_condition(&cond_stage.condition, ctx)?;

                if condition_met {
                    info!(
                        "{}  Condition '{}' is true, executing then branch",
                        indent, cond_stage.condition
                    );
                    execute_stages(&cond_stage.then_stages, ctx, depth + 1)?;
                } else if !cond_stage.else_stages.is_empty() {
                    info!(
                        "{}  Condition '{}' is false, executing else branch",
                        indent, cond_stage.condition
                    );
                    execute_stages(&cond_stage.else_stages, ctx, depth + 1)?;
                }
            }
        }
    }

    Ok(())
}

/// Resolve a tensor reference to actual data.
/// Handles cached tensors, runtime inputs, loop variables, and special expressions.
#[allow(dead_code)] // Used in full runtime implementation
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
        && let Some(ref scheduler) = ctx.scheduler
    {
        return Ok(vec![scheduler.current_timestep() as f32]);
    }

    // Check runtime inputs
    if let Some(input_str) = ctx.runtime_inputs.get(tensor_ref) {
        return parse_input_tensor(input_str);
    }

    // Not found - return empty with warning
    warn!(
        "Tensor reference '{}' not found, using empty tensor",
        tensor_ref
    );
    Ok(vec![0.0f32; 1])
}

/// Parse a loop range expression like "range(steps)" or "range(20)".
fn parse_loop_range(expr: &str, ctx: &PipelineContext) -> Result<usize> {
    let expr = expr.trim();

    // Handle range(n) syntax
    if let Some(inner) = expr
        .strip_prefix("range(")
        .and_then(|s| s.strip_suffix(')'))
    {
        let inner = inner.trim();

        // Try parsing as a number
        if let Ok(n) = inner.parse::<usize>() {
            return Ok(n);
        }

        // Try as a runtime input reference
        if let Some(value_str) = ctx.runtime_inputs.get(inner)
            && let Ok(n) = value_str.parse::<usize>()
        {
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
        let threshold: i64 = rest
            .trim()
            .parse()
            .with_context(|| format!("Invalid threshold in condition: {}", condition))?;

        if let Some(&val) = ctx.loop_vars.get(var) {
            return Ok(val > threshold);
        }
    }

    // Handle "step < N" patterns
    if let Some((var, rest)) = condition.split_once('<') {
        let var = var.trim();
        let threshold: i64 = rest
            .trim()
            .parse()
            .with_context(|| format!("Invalid threshold in condition: {}", condition))?;

        if let Some(&val) = ctx.loop_vars.get(var) {
            return Ok(val < threshold);
        }
    }

    // Handle "step == N" patterns
    if let Some((var, rest)) = condition.split_once("==") {
        let var = var.trim();
        let value: i64 = rest
            .trim()
            .parse()
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
fn parse_inputs(inputs: &[String], config: &UnifiedConfig) -> Result<HashMap<String, String>> {
    let mut result = HashMap::new();

    for input in inputs {
        let parts: Vec<&str> = input.splitn(2, '=').collect();
        if parts.len() != 2 {
            anyhow::bail!("Invalid input format '{}'. Expected 'name=value'", input);
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
#[allow(dead_code)] // Used in full runtime implementation
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
        return values
            .with_context(|| format!("Failed to parse comma-separated values: {}", input_str));
    }

    // Single scalar value
    let value: f32 = input_str
        .parse()
        .with_context(|| format!("Failed to parse scalar value: {}", input_str))?;
    Ok(vec![value])
}

/// Load an image file as a tensor.
#[allow(dead_code)] // Used in full runtime implementation
fn load_image_as_tensor(path: &str) -> Result<Vec<f32>> {
    #[cfg(feature = "image-output")]
    {
        use image::ImageReader;

        let img = ImageReader::open(path)
            .with_context(|| format!("Failed to open image: {}", path))?
            .decode()
            .with_context(|| format!("Failed to decode image: {}", path))?;

        let rgb = img.to_rgb8();
        let mut data = Vec::with_capacity(rgb.len());
        for byte in rgb.as_raw() {
            data.push(*byte as f32 / 255.0);
        }
        Ok(data)
    }
    #[cfg(not(feature = "image-output"))]
    {
        anyhow::bail!(
            "Image input loading requires the `image-output` feature: {}",
            path
        );
    }
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
    let get_arg_str = |key: &str| -> Option<&str> { args.get(key).and_then(|expr| expr.as_str()) };

    // Helper to get arg as i64 array (for shape)
    let get_arg_shape = |key: &str| -> Option<Vec<usize>> {
        args.get(key).and_then(|expr| match expr {
            crate::config::Expr::Literal(v) => v.as_array().map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_i64().map(|i| i as usize))
                    .collect()
            }),
        })
    };

    // Helper to get arg as f32
    let get_arg_f32 = |key: &str| -> Option<f32> {
        args.get(key).and_then(|expr| match expr {
            crate::config::Expr::Literal(v) => v.as_f64().map(|f| f as f32),
        })
    };

    // Helper to get arg as i64
    let get_arg_i64 = |key: &str| -> Option<i64> {
        args.get(key).and_then(|expr| match expr {
            crate::config::Expr::Literal(v) => v.as_i64(),
        })
    };

    let default_output = output_names.first().map(|s| s.as_str()).unwrap_or("output");

    match name {
        "randn" | "random_normal" => {
            // Generate random normal tensor with optional seed
            let shape = get_arg_shape("shape").unwrap_or_else(|| vec![1, 4, 64, 64]); // Default latent shape

            let size: usize = shape.iter().product();
            use std::f32::consts::PI;

            // Use seed from args or context
            let seed = get_arg_i64("seed").map(|s| s as u64).unwrap_or(ctx.seed);

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
                outputs.insert(
                    default_output.to_string(),
                    scheduler.timesteps.iter().map(|&t| t as f32).collect(),
                );
            }
        }

        "scheduler_step" | "denoise_step" => {
            // Perform one DDIM denoising step
            let latent_ref = get_arg_str("latent")
                .or(get_arg_str("sample"))
                .unwrap_or("latent");
            let noise_ref = get_arg_str("noise_pred")
                .or(get_arg_str("noise"))
                .unwrap_or("noise_pred");

            let latent = ctx
                .tensor_cache
                .get(latent_ref)
                .map(|t| t.as_ref().clone())
                .unwrap_or_else(|| {
                    warn!(
                        "Latent tensor '{}' not found for scheduler step",
                        latent_ref
                    );
                    vec![0.0f32; 1]
                });

            let noise_pred = ctx
                .tensor_cache
                .get(noise_ref)
                .map(|t| t.as_ref().clone())
                .unwrap_or_else(|| {
                    warn!(
                        "Noise prediction '{}' not found for scheduler step",
                        noise_ref
                    );
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
            let uncond_ref = get_arg_str("uncond")
                .or(get_arg_str("unconditional"))
                .unwrap_or("noise_uncond");
            let cond_ref = get_arg_str("cond")
                .or(get_arg_str("conditional"))
                .unwrap_or("noise_cond");
            let scale = get_arg_f32("scale").unwrap_or(ctx.guidance_scale);

            let uncond = ctx
                .tensor_cache
                .get(uncond_ref)
                .map(|t| t.as_ref().clone())
                .unwrap_or_else(|| vec![0.0f32; 1]);

            let cond = ctx
                .tensor_cache
                .get(cond_ref)
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
            // Text tokenization using loaded tokenizer or fallback
            let text_ref = get_arg_str("text").unwrap_or("prompt");
            let text = ctx
                .runtime_inputs
                .get(text_ref)
                .map(|s| s.as_str())
                .unwrap_or("a photo");

            let max_length = get_arg_i64("max_length").unwrap_or(77) as usize;

            info!("  Tokenizing text: \"{}\"", text);

            // Use real tokenizer if available
            if let Some(ref tokenizer) = ctx.tokenizer {
                // Use the loaded tokenizer
                let token_ids = tokenizer.encode(text, max_length)?;
                let tokens: Vec<f32> = token_ids.iter().map(|&t| t as f32).collect();

                // Create attention mask: 1 for non-padding, 0 for padding
                let pad_token_id = tokenizer.pad_token_id();
                let attention_mask: Vec<f32> = token_ids
                    .iter()
                    .map(|&t| if t != pad_token_id { 1.0 } else { 0.0 })
                    .collect();

                // Output both input_ids and attention_mask if requested
                if output_names.len() >= 2 {
                    outputs.insert(output_names[0].clone(), tokens);
                    outputs.insert(output_names[1].clone(), attention_mask);
                } else {
                    outputs.insert(default_output.to_string(), tokens);
                }

                info!(
                    "  Tokenized to {} tokens (vocab size: {})",
                    token_ids.len(),
                    tokenizer.vocab_size()
                );
                info!(
                    "  First 10 input tokens: {:?}",
                    &token_ids[..token_ids.len().min(10)]
                );
            } else {
                // Fallback to simple character-based tokenization
                warn!("No tokenizer loaded, using fallback character-based tokenization");

                let mut tokens: Vec<f32> = Vec::with_capacity(max_length);
                let is_t5 = max_length > 77 || output_names.contains(&"attention_mask".to_string());

                if is_t5 {
                    // T5-style fallback
                    for c in text.chars().take(max_length) {
                        let token_id = match c {
                            'a'..='z' => 100 + (c as u32 - 'a' as u32),
                            'A'..='Z' => 100 + (c as u32 - 'A' as u32),
                            ' ' => 3,
                            _ => 3,
                        };
                        tokens.push(token_id as f32);
                    }

                    let actual_len = tokens.len();
                    while tokens.len() < max_length {
                        tokens.push(0.0);
                    }

                    let attention_mask: Vec<f32> = (0..max_length)
                        .map(|i| if i < actual_len { 1.0 } else { 0.0 })
                        .collect();

                    if output_names.len() >= 2 {
                        outputs.insert(output_names[0].clone(), tokens);
                        outputs.insert(output_names[1].clone(), attention_mask);
                    } else {
                        outputs.insert(default_output.to_string(), tokens);
                    }
                } else {
                    // CLIP-style fallback
                    tokens.push(49406.0);
                    for c in text.chars().take(max_length - 2) {
                        let token_id = match c {
                            'a'..='z' => 320 + (c as u32 - 'a' as u32),
                            'A'..='Z' => 320 + (c as u32 - 'A' as u32),
                            ' ' => 267,
                            _ => 259,
                        };
                        tokens.push(token_id as f32);
                    }
                    tokens.push(49407.0);
                    while tokens.len() < max_length {
                        tokens.push(49407.0);
                    }
                    outputs.insert(default_output.to_string(), tokens);
                }
            }
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

        "init_decoder_input" => {
            // Create initial decoder input with start token
            let start_token = get_arg_i64("start_token_id").unwrap_or(0);
            let batch_size = get_arg_i64("batch_size").unwrap_or(1) as usize;

            let decoder_input = vec![start_token as f32; batch_size];
            outputs.insert(default_output.to_string(), decoder_input);
            info!("  Initialized decoder with start token: {}", start_token);
        }

        "sample_token" => {
            // Sample next token from logits (greedy argmax for now)
            let logits_ref = get_arg_str("logits").unwrap_or("logits");
            let temperature = get_arg_f32("temperature").unwrap_or(1.0);

            let logits = ctx
                .tensor_cache
                .get(logits_ref)
                .map(|t| t.as_ref().clone())
                .unwrap_or_else(|| vec![0.0f32]);

            // Get last token's logits (assuming shape [batch, seq_len, vocab_size])
            let vocab_size = ctx
                .tokenizer
                .as_ref()
                .map(|t| t.vocab_size())
                .unwrap_or(32128); // Fallback if no tokenizer
            let last_logits = if logits.len() >= vocab_size {
                &logits[logits.len() - vocab_size..]
            } else {
                &logits[..]
            };

            // Apply temperature
            let scaled: Vec<f32> = last_logits.iter().map(|&x| x / temperature).collect();

            // Softmax
            let max = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scaled.iter().map(|&x| (x - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let _probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();

            // Argmax (greedy sampling for MVP)
            let next_token_id = scaled
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            debug!("  Sampled token: {}", next_token_id);
            outputs.insert(default_output.to_string(), vec![next_token_id as f32]);
        }

        "append_token" => {
            // Append token to sequence
            let sequence_ref = get_arg_str("sequence").unwrap_or("decoder_input_ids");
            let token_ref = get_arg_str("token").unwrap_or("next_token");

            let mut sequence = ctx
                .tensor_cache
                .get(sequence_ref)
                .map(|t| t.as_ref().clone())
                .unwrap_or_default();

            let token = ctx
                .tensor_cache
                .get(token_ref)
                .and_then(|t| t.first().copied())
                .unwrap_or(0.0);

            sequence.push(token);
            debug!(
                "  Appended token {}, sequence length: {}",
                token,
                sequence.len()
            );
            outputs.insert(default_output.to_string(), sequence);
        }

        #[cfg(feature = "text-output")]
        "detokenize" => {
            // Detokenize token IDs to text
            let token_ids_ref = get_arg_str("token_ids").unwrap_or("decoder_input_ids");
            let tokenizer_path = get_arg_str("tokenizer_path")
                .ok_or_else(|| anyhow::anyhow!("detokenize requires tokenizer_path"))?;

            let token_ids_f32 = ctx
                .tensor_cache
                .get(token_ids_ref)
                .map(|t| t.as_ref().clone())
                .unwrap_or_default();

            let token_ids: Vec<u32> = token_ids_f32.iter().map(|&f| f as u32).collect();

            // Load tokenizer
            use tokenizers::Tokenizer;
            let tokenizer = Tokenizer::from_file(tokenizer_path)
                .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

            // Decode
            let text = tokenizer
                .decode(&token_ids, true)
                .map_err(|e| anyhow::anyhow!("Detokenization failed: {}", e))?;

            info!("  Generated text: \"{}\"", text);

            // Store text in generated outputs for output handler
            ctx.generated_outputs
                .insert("generated_text".to_string(), text.clone());

            // Return text length as tensor (for compatibility)
            outputs.insert(default_output.to_string(), vec![text.len() as f32]);
        }

        "generate" => {
            // Auto-regressive text generation
            //
            // This builtin implements the full generation loop for encoder-decoder models.
            // It repeatedly calls the decoder to generate tokens one by one.
            //
            // Required args:
            //   - model: decoder model name
            //   - encoder_hidden_states: encoder outputs
            //   - encoder_attention_mask: attention mask
            //
            // Optional args:
            //   - max_new_tokens: maximum number of tokens to generate (default: 50)
            //   - start_token_id: initial token (default: 0 - PAD token for T5)
            //   - eos_token_id: end-of-sequence token to stop generation (default: 1 - EOS)
            //   - pad_token_id: padding token for decoder input (default: start_token_id)
            //   - temperature: sampling temperature (default: 1.0, greedy)

            let model_name = get_arg_str("model")
                .ok_or_else(|| anyhow::anyhow!("generate requires 'model' argument"))?;

            let max_new_tokens = get_arg_i64("max_new_tokens").unwrap_or(50) as usize;
            let start_token_id = get_arg_i64("start_token_id").unwrap_or(0) as u32;
            let eos_token_id = get_arg_i64("eos_token_id").unwrap_or(1) as u32;
            let pad_token_id = get_arg_i64("pad_token_id").unwrap_or(start_token_id as i64) as u32;
            let temperature = get_arg_f32("temperature").unwrap_or(1.0);

            // Get encoder outputs from context
            let encoder_hidden_states_ref = get_arg_str("encoder_hidden_states")
                .ok_or_else(|| anyhow::anyhow!("generate requires 'encoder_hidden_states'"))?;
            let encoder_attention_mask_ref = get_arg_str("encoder_attention_mask")
                .ok_or_else(|| anyhow::anyhow!("generate requires 'encoder_attention_mask'"))?;

            let encoder_hidden_states = ctx
                .tensor_cache
                .get(encoder_hidden_states_ref)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Encoder hidden states '{}' not found",
                        encoder_hidden_states_ref
                    )
                })?
                .clone();

            let encoder_attention_mask = ctx
                .tensor_cache
                .get(encoder_attention_mask_ref)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Encoder attention mask '{}' not found",
                        encoder_attention_mask_ref
                    )
                })?
                .clone();

            let decoder_seq_len = encoder_attention_mask.len();
            if decoder_seq_len == 0 {
                return Err(anyhow::anyhow!("Encoder attention mask is empty"));
            }
            let max_new_tokens = max_new_tokens.min(decoder_seq_len.saturating_sub(1));
            if max_new_tokens == 0 {
                return Err(anyhow::anyhow!(
                    "Decoder sequence length {} is too small for generation",
                    decoder_seq_len
                ));
            }

            info!("  Starting auto-regressive generation:");
            info!("    Model: {}", model_name);
            info!("    Max new tokens: {}", max_new_tokens);
            info!("    Start token: {}", start_token_id);
            info!("    EOS token: {}", eos_token_id);
            info!("    Pad token: {}", pad_token_id);

            // Initialize decoder input with start token
            let mut generated_tokens: Vec<u32> = vec![start_token_id];

            // Generation loop
            for step in 0..max_new_tokens {
                debug!("  Generation step {}/{}", step + 1, max_new_tokens);

                if generated_tokens.len() > decoder_seq_len {
                    return Err(anyhow::anyhow!(
                        "Generated token length {} exceeds decoder sequence length {}",
                        generated_tokens.len(),
                        decoder_seq_len
                    ));
                }

                // Build fixed-size decoder input padded to encoder sequence length.
                let mut decoder_input_ids: Vec<f32> = vec![pad_token_id as f32; decoder_seq_len];
                for (idx, token) in generated_tokens.iter().enumerate() {
                    decoder_input_ids[idx] = *token as f32;
                }

                // Store inputs in context for model execution
                ctx.tensor_cache.insert(
                    "_gen_decoder_input_ids".to_string(),
                    Arc::new(decoder_input_ids),
                );
                ctx.tensor_cache.insert(
                    "_gen_encoder_hidden_states".to_string(),
                    encoder_hidden_states.clone(),
                );
                ctx.tensor_cache.insert(
                    "_gen_encoder_attention_mask".to_string(),
                    encoder_attention_mask.clone(),
                );

                // Prepare decoder inputs
                let mut decoder_inputs = HashMap::new();
                decoder_inputs.insert(
                    "input_ids".to_string(),
                    crate::config::Expr::string("_gen_decoder_input_ids"),
                );
                decoder_inputs.insert(
                    "encoder_hidden_states".to_string(),
                    crate::config::Expr::string("_gen_encoder_hidden_states"),
                );
                decoder_inputs.insert(
                    "encoder_attention_mask".to_string(),
                    crate::config::Expr::string("_gen_encoder_attention_mask"),
                );

                // Get decoder model path
                let decoder_model = ctx
                    .holo_models
                    .get(model_name)
                    .ok_or_else(|| anyhow::anyhow!("Decoder model '{}' not found", model_name))?;

                // Execute decoder
                // Note: .holo files don't store output names, so we get generic names
                // "output_0" = logits, "output_1..24" = key-value caches
                let decoder_outputs = execute_model_stage(
                    decoder_model,
                    &decoder_inputs,
                    &["output_0".to_string()], // Request first output (logits)
                    ctx,
                )?;

                // Get logits (first output)
                let logits = decoder_outputs
                    .get("output_0")
                    .ok_or_else(|| anyhow::anyhow!("Decoder did not produce logits (output_0)"))?;

                // Debug: Check logits shape and values
                if step == 0 {
                    info!("    Logits total length: {}", logits.len());
                    info!("    First 10 logits: {:?}", &logits[..logits.len().min(10)]);
                    info!(
                        "    Last 10 logits: {:?}",
                        &logits[logits.len().saturating_sub(10)..]
                    );

                    // Check for non-zero values
                    let non_zero_count = logits.iter().filter(|&&x| x != 0.0).count();
                    let (min_val, max_val) = logits
                        .iter()
                        .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &val| {
                            (min.min(val), max.max(val))
                        });
                    info!("    Non-zero values: {}/{}", non_zero_count, logits.len());
                    info!("    Range: [{:.4}, {:.4}]", min_val, max_val);
                }

                // Sample next token from last position's logits
                // Logits shape: [batch=1, seq_len, vocab_size]
                let vocab_size = ctx
                    .tokenizer
                    .as_ref()
                    .map(|t| t.vocab_size())
                    .unwrap_or(32128); // Fallback if no tokenizer
                let seq_len = generated_tokens.len();

                // Get logits for the last token
                let last_token_start = (seq_len - 1) * vocab_size;
                let last_token_logits = if logits.len() >= last_token_start + vocab_size {
                    &logits[last_token_start..last_token_start + vocab_size]
                } else {
                    // Fallback: use last vocab_size elements
                    &logits[logits.len().saturating_sub(vocab_size)..]
                };

                // Apply temperature and sample
                let scaled: Vec<f32> = last_token_logits.iter().map(|&x| x / temperature).collect();

                // Debug: Show top-5 predicted tokens
                if step < 3 {
                    // Only for first 3 steps to avoid spam
                    let mut top_tokens: Vec<(usize, f32)> = scaled
                        .iter()
                        .enumerate()
                        .map(|(idx, &val)| (idx, val))
                        .collect();
                    top_tokens.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                    info!("    Top 5 predictions:");
                    for (rank, (token_id, logit)) in top_tokens.iter().take(5).enumerate() {
                        let token_str = if let Some(ref tokenizer) = ctx.tokenizer {
                            tokenizer
                                .decode(&[*token_id as u32])
                                .unwrap_or_else(|_| format!("token_{}", token_id))
                        } else {
                            format!("token_{}", token_id)
                        };
                        info!(
                            "      {}. Token {} ({}): logit={:.2}",
                            rank + 1,
                            token_id,
                            token_str,
                            logit
                        );
                    }
                }

                // Greedy sampling (argmax)
                let next_token_id = scaled
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx as u32)
                    .unwrap_or(0);

                debug!("    Sampled token: {}", next_token_id);

                // Check for EOS
                if next_token_id == eos_token_id {
                    info!("  Generation stopped at step {} (EOS token)", step + 1);
                    break;
                }

                // Append to sequence
                generated_tokens.push(next_token_id);
            }

            info!("  Generated {} tokens total", generated_tokens.len());

            // Convert to f32 for output
            let output_tokens: Vec<f32> = generated_tokens.iter().map(|&t| t as f32).collect();

            outputs.insert(default_output.to_string(), output_tokens);
        }

        #[cfg(not(feature = "text-output"))]
        "detokenize" | "decode" => {
            // Detokenize token IDs back to text
            let tokens_ref = get_arg_str("tokens")
                .ok_or_else(|| anyhow::anyhow!("decode requires 'tokens' argument"))?;

            if let Some(ref tokenizer) = ctx.tokenizer {
                let tokens_f32 = ctx
                    .tensor_cache
                    .get(tokens_ref)
                    .ok_or_else(|| anyhow::anyhow!("Tokens '{}' not found", tokens_ref))?;

                // Convert f32 back to u32
                let tokens_u32: Vec<u32> = tokens_f32.iter().map(|&t| t as u32).collect();

                // Debug: show first 10 tokens
                info!(
                    "  First 10 generated tokens: {:?}",
                    &tokens_u32[..tokens_u32.len().min(10)]
                );

                // Decode to text
                let text = tokenizer.decode(&tokens_u32)?;

                info!("  Decoded: \"{}\" (from {} tokens)", text, tokens_u32.len());

                // Store as generated output for text output handler
                ctx.generated_outputs
                    .insert(default_output.to_string(), text.clone());

                // Also store as tensor (ASCII bytes) for compatibility
                let bytes: Vec<f32> = text.bytes().map(|b| b as f32).collect();
                outputs.insert(default_output.to_string(), bytes);
            } else {
                return Err(anyhow::anyhow!("No tokenizer loaded for decoding"));
            }
        }

        _ => {
            warn!("Unknown builtin operation: {}", name);
        }
    }

    Ok(outputs)
}

/// Execute a compiled .holo model with given inputs/outputs.
///
/// This function:
/// 1. Loads and compiles the .holo file to BackendPlan
/// 2. Resolves input tensors from pipeline context
/// 3. Executes the model using ModelExecutor
/// 4. Returns output tensors
fn execute_model_stage(
    holo_path: &Path,
    input_mapping: &HashMap<String, crate::config::Expr>,
    output_names: &[String],
    ctx: &mut PipelineContext,
) -> Result<HashMap<String, Vec<f32>>> {
    use crate::runtime::{ModelExecutor, Tensor, infer_tensor_dtype, infer_tensor_shape};

    // Check executor cache first to avoid reloading
    let mut executor = if let Some(cached) = ctx.executor_cache.remove(holo_path) {
        debug!("  Using cached executor for: {}", holo_path.display());
        cached
    } else {
        info!("  Loading model: {}", holo_path.display());
        ModelExecutor::from_holo_file(holo_path)
            .with_context(|| format!("Failed to load model from {}", holo_path.display()))?
    };

    // Prepare input tensors
    let mut input_tensors = HashMap::new();
    for (model_input_name, tensor_ref_expr) in input_mapping {
        // Extract string reference from Expr
        let tensor_ref = tensor_ref_expr
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Input mapping must be string references"))?;

        // Resolve tensor from cache
        let tensor_data = resolve_tensor_ref(tensor_ref, ctx)?;

        // Infer shape from input name
        let shape = infer_tensor_shape(&tensor_data, model_input_name)?;

        let mut tensor = Tensor::new(tensor_data, shape);
        tensor.dtype = infer_tensor_dtype(model_input_name);
        let numel = tensor.numel();
        input_tensors.insert(model_input_name.clone(), tensor);

        debug!("  Input '{}': {} elements", model_input_name, numel);
    }

    // Execute model
    info!("  Executing model with {} inputs", input_tensors.len());
    let output_tensors = executor
        .execute(input_tensors)
        .with_context(|| "Model execution failed")?;

    // Debug: Check if first output has valid data RIGHT after execution
    let first_output_name = output_tensors.keys().next().map(|s| s.as_str());
    if let Some(output_name) = first_output_name
        && let Some(tensor) = output_tensors.get(output_name)
    {
        let f32_data = tensor.to_f32();
        let data_vec = f32_data.to_vec();
        let non_zero = data_vec
            .iter()
            .filter(|&&x| x != 0.0 && !x.is_nan())
            .count();
        let has_nan = data_vec.iter().any(|x| x.is_nan());
        let (min_val, max_val) = data_vec
            .iter()
            .filter(|x| !x.is_nan())
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &val| {
                (min.min(val), max.max(val))
            });
        debug!(
            "  RAW {}: {} elements, {} non-zero, {} NaN, range [{:.6}, {:.6}]",
            output_name,
            data_vec.len(),
            non_zero,
            if has_nan { "HAS" } else { "no" },
            min_val,
            max_val
        );

        // Show sample values
        if data_vec.len() >= 20 {
            debug!("    First 10: {:?}", &data_vec[..10]);
        }
    }

    // Debug: Log available outputs
    debug!(
        "  Available outputs from model: {:?}",
        output_tensors.keys().collect::<Vec<_>>()
    );

    // Extract outputs
    let mut outputs = HashMap::new();
    for output_name in output_names {
        if let Some(tensor) = output_tensors.get(output_name) {
            outputs.insert(output_name.clone(), tensor.to_f32().to_vec());
            debug!("  Output '{}': {} elements", output_name, tensor.numel());
        } else if let Some(tensor) = output_tensors.get("output") {
            // Fallback to default output name
            outputs.insert(output_name.clone(), tensor.to_f32().to_vec());
            debug!(
                "  Output '{}' (default): {} elements",
                output_name,
                tensor.numel()
            );
        } else {
            warn!("  Output '{}' not found in model outputs!", output_name);
        }
    }

    info!("  Model execution completed successfully");

    // Store executor back in cache for reuse
    ctx.executor_cache.insert(holo_path.to_path_buf(), executor);

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
    std::fs::create_dir_all(output_dir).with_context(|| {
        format!(
            "Failed to create output directory: {}",
            output_dir.display()
        )
    })?;

    for (name, output_def) in &config.outputs {
        let tensor_name = output_def.tensor();
        let handler_type = output_def.handler_type();

        let data = outputs.get(tensor_name).ok_or_else(|| {
            anyhow::anyhow!(
                "Output tensor '{}' not found in pipeline outputs",
                tensor_name
            )
        })?;

        match handler_type {
            OutputHandlerType::Image => {
                let output_path = output_dir.join(format!("{}.png", name));
                info!("Writing image output: {}", output_path.display());

                if let OutputData::Tensor(tensor) = data {
                    // Get image dimensions from config or infer from tensor
                    let (width, height, channels) =
                        infer_image_dimensions(tensor.len(), output_def);

                    // Convert tensor to u8 image data
                    let image_data = tensor_to_image_data(tensor, width, height, channels)?;

                    // Write using the image crate
                    write_image_data(&image_data, width, height, channels, &output_path)?;

                    info!(
                        "  Image saved: {}x{} ({} channels)",
                        width, height, channels
                    );
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

                match data {
                    OutputData::Text(text) => {
                        // Write text as JSON string
                        let json = serde_json::to_string_pretty(text)?;
                        std::fs::write(&output_path, json)?;
                        info!("  Text output: {}", text);
                    }
                    OutputData::Tensor(tensor) => {
                        let json = serde_json::to_string_pretty(tensor)?;
                        std::fs::write(&output_path, json)?;
                    }
                    OutputData::Image(width, height, channels, data) => {
                        let img_data = serde_json::json!({
                            "width": width,
                            "height": height,
                            "channels": channels,
                            "data_length": data.len()
                        });
                        let json = serde_json::to_string_pretty(&img_data)?;
                        std::fs::write(&output_path, json)?;
                        info!("  Image output: {}x{}x{}", width, height, channels);
                    }
                    OutputData::Audio(sample_rate, samples) => {
                        let audio_data = serde_json::json!({
                            "sample_rate": sample_rate,
                            "samples": samples
                        });
                        let json = serde_json::to_string_pretty(&audio_data)?;
                        std::fs::write(&output_path, json)?;
                        info!(
                            "  Audio output: {} Hz, {} samples",
                            sample_rate,
                            samples.len()
                        );
                    }
                }
            }

            OutputHandlerType::Binary => {
                let output_path = output_dir.join(format!("{}.bin", name));
                info!("Writing binary output: {}", output_path.display());

                if let OutputData::Tensor(tensor) = data {
                    let bytes: Vec<u8> = tensor.iter().flat_map(|f| f.to_le_bytes()).collect();
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
fn infer_image_dimensions(tensor_size: usize, _output_def: &OutputDef) -> (u32, u32, u8) {
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
    debug!(
        "Could not infer image dimensions from tensor size {}",
        tensor_size
    );
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
            tensor.len(),
            width,
            height,
            channels,
            expected_size
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
    use image::{ImageBuffer, Luma, Rgb, Rgba};

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
fn write_image_data(
    _data: &[u8],
    _width: u32,
    _height: u32,
    _channels: u8,
    _path: &Path,
) -> Result<()> {
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

// =============================================================================
// Direct .holo Execution for T5 Models
// =============================================================================

/// Run a .holo model directly with a text prompt (for T5 and text models).
///
/// # Arguments
///
/// * `model_path` - Path to compiled .holo file (encoder or decoder)
/// * `prompt` - Optional text prompt to process
/// * `tokenizer_path` - Path to tokenizer.json file
/// * `max_length` - Maximum sequence length for tokenization
/// * `output_dir` - Optional output directory
///
/// # Returns
///
/// Returns Ok(()) on success, or an error if execution fails.
///
/// # Example
///
/// ```bash
/// hologram-onnx run encoder.holo --prompt "Tell me a joke" --tokenizer tokenizer.json
/// ```
#[cfg(feature = "text-output")]
pub fn run_direct_command(
    model_path: &Path,
    prompt: Option<&str>,
    tokenizer_path: &Path,
    max_length: usize,
    _output_dir: Option<&Path>,
    execution_mode: super::CliExecutionMode,
) -> Result<()> {
    // Log execution mode
    info!("Execution mode: {:?}", execution_mode);
    use tokenizers::Tokenizer;

    info!("Direct model execution mode");
    info!("Model: {}", model_path.display());
    info!("Tokenizer: {}", tokenizer_path.display());

    if !model_path.exists() {
        anyhow::bail!(
            "Model file not found: {}. Run 'hologram-onnx compile' first.",
            model_path.display()
        );
    }

    // Get prompt from argument or use default
    let prompt_text = prompt.unwrap_or("Translate English to French: Hello, how are you?");
    info!("Prompt: \"{}\"", prompt_text);

    // Load tokenizer
    info!("Loading tokenizer...");
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| {
        anyhow::anyhow!(
            "Failed to load tokenizer from {}: {}",
            tokenizer_path.display(),
            e
        )
    })?;

    // Tokenize input
    info!("Tokenizing input...");
    let encoding = tokenizer
        .encode(prompt_text, true)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

    let input_ids = encoding.get_ids();
    info!("Input tokens: {} tokens", input_ids.len());
    debug!("Token IDs: {:?}", &input_ids[..input_ids.len().min(20)]);

    // Truncate or pad to max_length
    let mut padded_ids = Vec::with_capacity(max_length);
    for &id in input_ids.iter().take(max_length) {
        padded_ids.push(id);
    }
    // Pad with tokenizer's padding token (usually 0)
    while padded_ids.len() < max_length {
        padded_ids.push(0);
    }

    info!("Padded to {} tokens", padded_ids.len());

    // Create attention mask (1 for real tokens, 0 for padding)
    let _attention_mask: Vec<i64> = padded_ids
        .iter()
        .map(|&id| if id == 0 { 0 } else { 1 })
        .collect();

    // Prepare batch dimension (batch_size=1)
    let batch_size = 1usize;
    let seq_len = padded_ids.len();

    info!("Input shape: [batch={}, seq_len={}]", batch_size, seq_len);
    info!(
        "Attention mask shape: [batch={}, seq_len={}]",
        batch_size, seq_len
    );

    // Load the .holo model using ModelExecutor
    info!("Loading model: {}", model_path.display());
    let mut executor = crate::runtime::ModelExecutor::from_holo_file(model_path)
        .with_context(|| format!("Failed to load model from {}", model_path.display()))?;

    // Create attention mask (1 for real tokens, 0 for padding)
    let attention_mask: Vec<f32> = padded_ids
        .iter()
        .map(|&id| if id == 0 { 0.0 } else { 1.0 })
        .collect();

    // Convert input_ids to f32 for the executor
    let input_ids_f32: Vec<f32> = padded_ids.iter().map(|&id| id as f32).collect();

    // Create input tensors
    let mut inputs = std::collections::HashMap::new();
    inputs.insert(
        "input_ids".to_string(),
        crate::runtime::Tensor::new(input_ids_f32, vec![batch_size, seq_len]),
    );
    inputs.insert(
        "attention_mask".to_string(),
        crate::runtime::Tensor::new(attention_mask, vec![batch_size, seq_len]),
    );

    info!("");
    info!("=== Executing Encoder ===");
    info!("Running encoder with {} tokens...", seq_len);

    let start = std::time::Instant::now();
    let outputs = executor
        .execute(inputs)
        .with_context(|| "Failed to execute encoder")?;
    let elapsed = start.elapsed();

    info!("Encoder execution completed in {:?}", elapsed);

    // Display output information
    info!("");
    info!("=== Output ===");
    for (name, tensor) in &outputs {
        info!(
            "Output '{}': shape {:?}, {} elements",
            name,
            tensor.shape,
            tensor.numel()
        );

        // Check for NaN/Inf values
        let nan_count = tensor.data.iter().filter(|x| x.is_nan()).count();
        let inf_count = tensor.data.iter().filter(|x| x.is_infinite()).count();
        if nan_count > 0 || inf_count > 0 {
            warn!(
                "  WARNING: {} NaN values, {} Inf values",
                nan_count, inf_count
            );
        }

        // Show some sample values
        let sample_size = tensor.data.len().min(5);
        info!(
            "  First {} values: {:?}",
            sample_size,
            &tensor.data[..sample_size]
        );
    }

    info!("");
    info!("=== Encoder Execution Complete ===");
    info!("To generate text, use both encoder and decoder with a config file.");

    Ok(())
}

/// Stub for when text-output feature is disabled.
#[cfg(not(feature = "text-output"))]
pub fn run_direct_command(
    _model_path: &Path,
    _prompt: Option<&str>,
    _tokenizer_path: &Path,
    _max_length: usize,
    _output_dir: Option<&Path>,
    _execution_mode: super::CliExecutionMode,
) -> Result<()> {
    anyhow::bail!(
        "Direct .holo execution with --prompt requires the 'text-output' feature.\n\
         Rebuild with: cargo build --features text-output"
    )
}

// =============================================================================
// Pipeline Bundle Execution for T5 Models
// =============================================================================

/// Run a T5 pipeline bundle (HOLM format) for text generation.
///
/// This function handles encoder-decoder text generation using a pre-compiled
/// pipeline bundle containing encoder, decoder, and tokenizer models.
///
/// # Arguments
///
/// * `pipeline_path` - Path to the .holo pipeline bundle (HOLM format)
/// * `prompt` - Text prompt for generation
/// * `max_new_tokens` - Maximum number of tokens to generate
///
/// # Returns
///
/// Returns the generated text, or an error if generation fails.
#[allow(clippy::too_many_arguments)]
pub fn run_pipeline_bundle_command(
    pipeline_path: &Path,
    prompt: &str,
    max_new_tokens: usize,
    min_new_tokens: usize,
    top_k: usize,
    temperature: f32,
    beam_size: usize,
    length_penalty: f32,
    no_repeat_ngram: usize,
    eos_prob_threshold: f32,
) -> Result<String> {
    use crate::runtime::{ModelExecutor, Tensor, load_pipeline_bundle};

    info!("=== T5 Pipeline Text Generation ===");
    info!("Pipeline: {}", pipeline_path.display());
    info!("Prompt: \"{}\"", prompt);
    info!("Max new tokens: {}", max_new_tokens);
    info!("Min new tokens: {}", min_new_tokens);
    info!("Top-k: {}", top_k);
    info!("Temperature: {:.3}", temperature);
    info!("Beam size: {}", beam_size);
    info!("Length penalty: {:.3}", length_penalty);
    info!("No-repeat ngram: {}", no_repeat_ngram);
    info!("EOS prob threshold: {:.3}", eos_prob_threshold);

    // Load pipeline bundle
    info!("");
    info!("Loading pipeline bundle...");
    let pipeline = load_pipeline_bundle(pipeline_path)?;

    let model_names = pipeline.model_names();
    info!("Available models: {:?}", model_names);

    // Verify required models exist
    let has_encoder = model_names.iter().any(|n| n.contains("encoder"));
    let has_decoder = model_names.iter().any(|n| n.contains("decoder"));

    if !has_encoder {
        anyhow::bail!("Pipeline bundle must contain an encoder model");
    }
    if !has_decoder {
        anyhow::bail!("Pipeline bundle must contain a decoder model");
    }

    // Load encoder early to infer the compiled sequence length.
    let (encoder_plan, encoder_backend, encoder_inputs) =
        pipeline.load_model_with_inputs("encoder")?;
    let max_length = infer_sequence_length_from_plan(encoder_plan.plan()).unwrap_or_else(|| {
        warn!("  Falling back to max_length=512 (unable to infer from model plan)");
        512
    });
    let mut token_max_length = max_length;
    if let Ok(value) = std::env::var("HOLOGRAM_T5_MAX_LENGTH")
        && let Ok(override_len) = value.parse::<usize>()
        && override_len > 0
    {
        token_max_length = override_len.min(max_length);
        info!(
            "  Overriding token max_length with HOLOGRAM_T5_MAX_LENGTH={} (plan max_length={})",
            token_max_length, max_length
        );
    }

    // Load SentencePiece tokenizer from tokenizer.json adjacent to pipeline file
    info!("");
    info!("Loading tokenizer...");
    let tokenizer_path = pipeline_path.parent().and_then(|dir| {
        let direct = dir.join("tokenizer.json");
        if direct.exists() {
            return Some(direct);
        }
        dir.parent().and_then(|parent| {
            let fallback = parent.join("tokenizer.json");
            if fallback.exists() {
                Some(fallback)
            } else {
                None
            }
        })
    });

    let tokenizer: Option<crate::tokenizers::sentencepiece::SentencePieceTokenizer> =
        tokenizer_path.as_ref().and_then(|path| {
            use crate::tokenizers::Tokenizer;
            match crate::tokenizers::sentencepiece::SentencePieceTokenizer::from_file(path) {
                Ok(tok) => {
                    info!(
                        "  Loaded SentencePiece tokenizer (vocab size: {})",
                        tok.vocab_size()
                    );
                    Some(tok)
                }
                Err(e) => {
                    warn!("  Failed to load tokenizer from {:?}: {}", path, e);
                    None
                }
            }
        });

    // Tokenize input
    info!("");
    info!("Tokenizing input...");
    info!(
        "  Using model max_length: {} (token max_length: {})",
        max_length, token_max_length
    );

    let (mut input_ids, actual_len) = if let Some(ref tok) = tokenizer {
        use crate::tokenizers::Tokenizer;
        match tok.encode(prompt, token_max_length) {
            Ok(tokens) => {
                let actual_len = tokens
                    .iter()
                    .take_while(|&&t| t != tok.pad_token_id())
                    .count()
                    .max(1);
                info!(
                    "  Tokenized with SentencePiece: {} tokens (non-pad)",
                    actual_len
                );
                info!("  First tokens: {:?}", &tokens[..actual_len.min(10)]);
                (tokens, actual_len)
            }
            Err(e) => {
                warn!("  Tokenization failed: {}, using fallback", e);
                fallback_tokenize(prompt, token_max_length)
            }
        }
    } else {
        warn!("  No tokenizer available - using fallback tokenization");
        fallback_tokenize(prompt, token_max_length)
    };

    let pad_token_id_for_input = tokenizer
        .as_ref()
        .map(|tok| tok.pad_token_id())
        .unwrap_or(0);
    if input_ids.len() < max_length {
        input_ids.resize(max_length, pad_token_id_for_input);
    }

    // Create attention mask
    // T5 ONNX export uses inverted mask: 0 for valid positions, 1 for masked (padding)
    // The model computes: (1 + inverted_mask) * -inf = -2*inf for valid, -inf for padding
    // This is wrong - need to use standard mask format
    // Standard: 1 for valid, 0 for padding - model should compute (1 - mask) * -inf
    let attention_mask: Vec<u32> = (0..max_length)
        .map(|i| if i < actual_len { 1 } else { 0 })
        .collect();

    // Debug: print first 10 mask values
    info!(
        "  Attention mask first 10: {:?}",
        &attention_mask[..10.min(max_length)]
    );

    // Load and run encoder using ModelExecutor
    info!("");
    info!("Running encoder...");
    let mut encoder = match encoder_inputs {
        Some(order) => {
            ModelExecutor::from_plan_executor_with_inputs(encoder_plan, encoder_backend, order)
        }
        None => ModelExecutor::from_plan_executor(encoder_plan, encoder_backend),
    };

    let attention_mask_f32: Vec<f32> = attention_mask.iter().map(|&v| v as f32).collect();
    let input_ids_f32: Vec<f32> = input_ids.iter().map(|&v| v as f32).collect();
    let mut encoder_inputs = std::collections::HashMap::new();
    encoder_inputs.insert(
        "attention_mask".to_string(),
        Tensor::new(attention_mask_f32.clone(), vec![1, max_length]),
    );
    encoder_inputs.insert(
        "input_ids".to_string(),
        Tensor::new(input_ids_f32, vec![1, max_length]),
    );

    let encoder_start = std::time::Instant::now();
    let encoder_outputs = encoder.execute(encoder_inputs)?;
    info!("  Encoder completed in {:?}", encoder_start.elapsed());

    // Debug: show available output keys and their shapes
    info!("  Available encoder outputs:");
    for (key, tensor) in &encoder_outputs {
        let non_zero = tensor.to_f32().iter().filter(|&&x| x != 0.0).count();
        info!(
            "    '{}': shape={:?}, non-zero={}/{}",
            key,
            tensor.shape,
            non_zero,
            tensor.to_f32().len()
        );
    }

    // Get encoder hidden states
    let encoder_hidden_states = encoder_outputs
        .get("output")
        .or_else(|| encoder_outputs.values().next())
        .ok_or_else(|| anyhow::anyhow!("Encoder produced no output"))?;

    let hidden_states_data = encoder_hidden_states.to_f32().to_vec();
    let non_zero = hidden_states_data.iter().filter(|&&x| x != 0.0).count();
    let nan_count = hidden_states_data.iter().filter(|&&x| x.is_nan()).count();
    let inf_count = hidden_states_data
        .iter()
        .filter(|&&x| x.is_infinite())
        .count();
    let min_val = hidden_states_data
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    let max_val = hidden_states_data
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    info!(
        "  Encoder output: {} elements, {} non-zero, {} NaN, {} Inf, range=[{:.6e}, {:.6e}]",
        hidden_states_data.len(),
        non_zero,
        nan_count,
        inf_count,
        min_val,
        max_val,
    );
    // Show first 10 non-zero values
    let first_nonzero: Vec<_> = hidden_states_data
        .iter()
        .filter(|&&x| x != 0.0)
        .take(10)
        .collect();
    debug!("  First 10 non-zero values: {:?}", first_nonzero);

    // Clean encoder hidden states: replace NaN/Inf with 0.0 to prevent decoder corruption
    let hidden_states_data: Vec<f32> = if nan_count > 0 || inf_count > 0 {
        warn!(
            "  Cleaning {} NaN and {} Inf values from encoder output",
            nan_count, inf_count
        );
        hidden_states_data
            .into_iter()
            .map(|x| {
                if x.is_nan() || x.is_infinite() {
                    0.0
                } else {
                    x
                }
            })
            .collect()
    } else {
        hidden_states_data
    };

    // Drop encoder to free memory
    drop(encoder);

    // Load decoder for generation
    info!("");
    info!("Running decoder (auto-regressive generation)...");
    let (decoder_plan, decoder_backend, decoder_inputs) =
        pipeline.load_model_with_inputs("decoder")?;
    let mut decoder = match decoder_inputs {
        Some(order) => {
            ModelExecutor::from_plan_executor_with_inputs(decoder_plan, decoder_backend, order)
        }
        None => ModelExecutor::from_plan_executor(decoder_plan, decoder_backend),
    };
    let decoder_input_count = decoder.plan().layout_metadata.num_inputs;
    let decoder_output_count = decoder.plan().layout_metadata.num_outputs;
    let kv_layers = infer_t5_kv_layers(decoder_output_count);
    let decoder_supports_kv = decoder_input_count > 3 && kv_layers.is_some();
    let kv_input_names = if decoder_supports_kv {
        Some(t5_decoder_input_names(kv_layers.unwrap_or(0)))
    } else {
        None
    };
    let kv_input_sizes = if let Some(ref names) = kv_input_names {
        Some(build_input_size_map(decoder.plan(), names)?)
    } else {
        None
    };
    if decoder_supports_kv {
        info!(
            "  Decoder KV-cache enabled: {} layers ({} inputs, {} outputs)",
            kv_layers.unwrap_or(0),
            decoder_input_count,
            decoder_output_count
        );
    } else if decoder_input_count > 3 {
        warn!(
            "  Decoder expects {} inputs but KV-cache outputs could not be inferred ({} outputs).",
            decoder_input_count, decoder_output_count
        );
    } else {
        info!(
            "  Decoder KV-cache disabled ({} inputs, {} outputs).",
            decoder_input_count, decoder_output_count
        );
    }

    // T5 decoder vocab size
    let vocab_size = 32128usize;

    // Initialize decoder with start token
    let mut generated_tokens: Vec<u32> = vec![0]; // Start with pad token
    let mut kv_cache: Option<DecoderKvCache> = None;

    let (pad_token_id, eos_token_id, unk_token_id) = if let Some(ref tok) = tokenizer {
        (tok.pad_token_id(), tok.eos_token_id(), tok.unk_token_id())
    } else {
        (0, 1, 2)
    };
    let special_token_ids = tokenizer
        .as_ref()
        .map(|tok| tok.special_token_ids())
        .unwrap_or(&[]);

    let mut rng = XorShift64::seeded();
    let gen_start = std::time::Instant::now();

    let mut run_decoder_step = |tokens: &[u32],
                                use_cache: bool,
                                kv: Option<DecoderKvCache>|
     -> Result<(Tensor, Option<DecoderKvCache>)> {
        let mut decoder_input_ids = vec![0u32; max_length];
        for (i, &tok) in tokens.iter().enumerate() {
            if i < max_length {
                decoder_input_ids[i] = tok;
            }
        }

        let decoder_input_ids_f32: Vec<f32> = decoder_input_ids.iter().map(|&v| v as f32).collect();
        let mut decoder_inputs = std::collections::HashMap::new();
        decoder_inputs.insert(
            "encoder_attention_mask".to_string(),
            Tensor::new(attention_mask_f32.clone(), vec![1, max_length]),
        );
        decoder_inputs.insert(
            "encoder_hidden_states".to_string(),
            Tensor::new(
                hidden_states_data.clone(),
                encoder_hidden_states.shape.clone(),
            ),
        );
        decoder_inputs.insert(
            "input_ids".to_string(),
            Tensor::new(decoder_input_ids_f32, vec![1, max_length]),
        );

        if decoder_supports_kv {
            let size_map = kv_input_sizes
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Decoder KV-cache size map was not initialized"))?;
            let cache = if use_cache {
                if let Some(cache) = kv {
                    cache
                } else {
                    DecoderKvCache::zeros(kv_layers.unwrap_or(0), size_map)?
                }
            } else {
                DecoderKvCache::zeros(kv_layers.unwrap_or(0), size_map)?
            };

            for (name, tensor) in cache.as_inputs() {
                decoder_inputs.insert(name, tensor);
            }

            let branch_size = size_map
                .get("use_cache_branch")
                .copied()
                .ok_or_else(|| anyhow::anyhow!("Missing size metadata for 'use_cache_branch'"))?;
            decoder_inputs.insert(
                "use_cache_branch".to_string(),
                tensor_from_size(branch_size, if use_cache { 1.0 } else { 0.0 })?,
            );
        }

        let decoder_outputs = decoder.execute(decoder_inputs)?;
        let step_outputs =
            extract_decoder_step_outputs(decoder_outputs, kv_layers, decoder_supports_kv)?;
        Ok((step_outputs.logits, step_outputs.kv_cache))
    };

    if beam_size > 1 {
        let mut beams = vec![BeamState::new(vec![pad_token_id], 0.0)];
        for step in 0..max_new_tokens {
            let mut candidates: Vec<BeamState> = Vec::new();

            for beam in &beams {
                if beam.finished {
                    candidates.push(beam.clone());
                    continue;
                }

                let (logits_tensor, _kv_out) = run_decoder_step(&beam.tokens, false, None)?;
                let logits = logits_tensor.to_f32();
                let seq_pos = beam.tokens.len().saturating_sub(1);
                let logits_start = seq_pos * vocab_size;
                let logits_end = logits_start + vocab_size;
                let last_logits = if logits_end <= logits.len() {
                    &logits[logits_start..logits_end]
                } else {
                    &logits[logits.len().saturating_sub(vocab_size)..]
                };

                let eos_prob = softmax_prob(last_logits, eos_token_id as usize, temperature);
                let allow_eos = (step + 1) >= min_new_tokens && eos_prob >= eos_prob_threshold;

                let filtered_logits = apply_token_filters(
                    last_logits,
                    step,
                    pad_token_id,
                    unk_token_id,
                    eos_token_id,
                    allow_eos,
                    special_token_ids,
                );
                let log_probs = log_softmax(&filtered_logits, temperature);
                let top = top_k_log_probs(&log_probs, top_k);

                for (token, logp) in top {
                    if no_repeat_ngram > 0
                        && violates_no_repeat_ngram(&beam.tokens, token, no_repeat_ngram)
                    {
                        continue;
                    }

                    let mut new_tokens = beam.tokens.clone();
                    new_tokens.push(token);
                    let mut new_beam = BeamState::new(new_tokens, beam.score + logp);
                    if token == eos_token_id {
                        new_beam.finished = true;
                    }
                    candidates.push(new_beam);
                }
            }

            if candidates.is_empty() {
                break;
            }

            candidates.sort_by(|a, b| {
                let score_a = length_penalized(a.score, a.tokens.len(), length_penalty);
                let score_b = length_penalized(b.score, b.tokens.len(), length_penalty);
                score_b
                    .partial_cmp(&score_a)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            candidates.truncate(beam_size);
            beams = candidates;

            if beams.iter().all(|b| b.finished) {
                break;
            }
        }

        if let Some(best) = beams.into_iter().max_by(|a, b| {
            let score_a = length_penalized(a.score, a.tokens.len(), length_penalty);
            let score_b = length_penalized(b.score, b.tokens.len(), length_penalty);
            score_a
                .partial_cmp(&score_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            generated_tokens = best.tokens;
        }
    } else {
        for step in 0..max_new_tokens {
            let (logits_tensor, kv_out) =
                run_decoder_step(&generated_tokens, kv_cache.is_some(), kv_cache.clone())?;
            let logits = logits_tensor.to_f32();
            kv_cache = kv_out;
            if step == 0 && std::env::var("HOLOGRAM_TRACE_LOGITS_SHAPE").is_ok() {
                info!("  Decoder logits shape: {:?}", logits_tensor.shape);
            }

            let seq_pos = generated_tokens.len().saturating_sub(1);
            let logits_start = seq_pos * vocab_size;
            let logits_end = logits_start + vocab_size;
            let last_logits = if logits_end <= logits.len() {
                &logits[logits_start..logits_end]
            } else {
                &logits[logits.len().saturating_sub(vocab_size)..]
            };

            let eos_prob = softmax_prob(last_logits, eos_token_id as usize, temperature);
            let allow_eos = (step + 1) >= min_new_tokens && eos_prob >= eos_prob_threshold;
            let filtered_logits = apply_token_filters(
                last_logits,
                step,
                pad_token_id,
                unk_token_id,
                eos_token_id,
                allow_eos,
                special_token_ids,
            );
            if step == 0
                && let Some(limit) = std::env::var("HOLOGRAM_TRACE_TOPK")
                    .ok()
                    .and_then(|val| val.parse::<usize>().ok())
                    .filter(|val| *val > 0)
            {
                let log_probs = log_softmax(&filtered_logits, temperature);
                let top = top_k_log_probs(&log_probs, limit);
                let mut rendered = Vec::with_capacity(top.len());
                for (token, logp) in top {
                    let text = tokenizer
                        .as_ref()
                        .and_then(|tok| tok.decode(&[token]).ok())
                        .unwrap_or_else(|| format!("<{}>", token));
                    rendered.push(format!("{}:{:.3}({})", token, logp, text));
                }
                info!("  TopK tokens step0: [{}]", rendered.join(", "));
            }

            let next_token = if top_k <= 1 || !temperature.is_finite() || temperature <= 0.0 {
                select_argmax_with_repeat(&filtered_logits, &generated_tokens, no_repeat_ngram)
            } else {
                select_sample_with_repeat(
                    &filtered_logits,
                    top_k,
                    temperature,
                    &generated_tokens,
                    no_repeat_ngram,
                    &mut rng,
                )
            };

            if next_token == eos_token_id && allow_eos {
                info!("  Generation stopped at step {} (EOS token)", step + 1);
                break;
            }

            generated_tokens.push(next_token);

            if generated_tokens.len() >= max_length {
                info!("  Reached max length {}", max_length);
                break;
            }
        }
    }

    info!(
        "  Generation completed in {:?} ({} tokens)",
        gen_start.elapsed(),
        generated_tokens.len()
    );

    // Decode tokens to text
    info!("");
    info!("Decoding output tokens...");
    info!(
        "  Generated tokens: {:?}",
        &generated_tokens[..generated_tokens.len().min(20)]
    );

    let decoded_text = decode_tokens_with_tokenizer(&generated_tokens, tokenizer.as_ref());

    info!("");
    info!("=== Generated Text ===");
    info!("{}", decoded_text);

    Ok(decoded_text)
}

fn argmax_logits(logits: &[f32]) -> u32 {
    let mut max_idx = 0usize;
    let mut max_val = f32::NEG_INFINITY;
    for (idx, &value) in logits.iter().enumerate() {
        if value.is_nan() {
            continue;
        }
        if value > max_val {
            max_val = value;
            max_idx = idx;
        }
    }
    max_idx as u32
}

fn apply_token_filters(
    logits: &[f32],
    step: usize,
    pad_id: u32,
    unk_id: u32,
    eos_id: u32,
    allow_eos: bool,
    special_ids: &[u32],
) -> Vec<f32> {
    let mut filtered = logits.to_vec();
    if (pad_id as usize) < filtered.len() && step > 0 {
        filtered[pad_id as usize] = f32::NEG_INFINITY;
    }
    if (unk_id as usize) < filtered.len() {
        filtered[unk_id as usize] = f32::NEG_INFINITY;
    }
    if !allow_eos && (eos_id as usize) < filtered.len() {
        filtered[eos_id as usize] = f32::NEG_INFINITY;
    }
    for &token_id in special_ids {
        if token_id == eos_id && allow_eos {
            continue;
        }
        let idx = token_id as usize;
        if idx < filtered.len() {
            filtered[idx] = f32::NEG_INFINITY;
        }
    }
    filtered
}

fn log_softmax(logits: &[f32], temperature: f32) -> Vec<f32> {
    let temp = if temperature.is_finite() && temperature > 0.0 {
        temperature
    } else {
        1.0
    };
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f64;
    for &value in logits {
        if value.is_finite() {
            sum += (((value - max_logit) / temp) as f64).exp();
        }
    }
    let log_sum = if sum > 0.0 {
        sum.ln() as f32
    } else {
        f32::INFINITY
    };
    logits
        .iter()
        .map(|&value| {
            if value.is_finite() && log_sum.is_finite() {
                (value - max_logit) / temp - log_sum
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect()
}

fn top_k_log_probs(log_probs: &[f32], k: usize) -> Vec<(u32, f32)> {
    let k = k.min(log_probs.len()).max(1);
    let mut indexed: Vec<(usize, f32)> =
        log_probs.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed
        .into_iter()
        .take(k)
        .map(|(idx, logp)| (idx as u32, logp))
        .collect()
}

fn softmax_prob(logits: &[f32], token_id: usize, temperature: f32) -> f32 {
    if token_id >= logits.len() {
        return 0.0;
    }
    let temp = if temperature.is_finite() && temperature > 0.0 {
        temperature
    } else {
        1.0
    };
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f64;
    for &value in logits {
        if value.is_finite() {
            sum += (((value - max_logit) / temp) as f64).exp();
        }
    }
    if sum <= 0.0 {
        return 0.0;
    }
    let value = logits[token_id];
    if !value.is_finite() {
        return 0.0;
    }
    let numerator = (((value - max_logit) / temp) as f64).exp();
    (numerator / sum) as f32
}

fn length_penalized(score: f32, length: usize, penalty: f32) -> f32 {
    if penalty <= 0.0 {
        return score;
    }
    let len = length.max(1) as f32;
    let norm = ((5.0 + len) / 6.0).powf(penalty);
    score / norm
}

fn violates_no_repeat_ngram(tokens: &[u32], next_token: u32, n: usize) -> bool {
    if n == 0 || tokens.len() + 1 < n {
        return false;
    }
    if tokens.len() < n {
        return false;
    }
    let start = tokens.len() + 1 - n;
    let mut new_ngram = Vec::with_capacity(n);
    new_ngram.extend_from_slice(&tokens[start..tokens.len()]);
    new_ngram.push(next_token);

    for idx in 0..=tokens.len().saturating_sub(n) {
        if tokens[idx..idx + n] == new_ngram[..] {
            return true;
        }
    }

    false
}

fn select_argmax_with_repeat(logits: &[f32], tokens: &[u32], no_repeat_ngram: usize) -> u32 {
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (idx, _) in indexed {
        let token = idx as u32;
        if !violates_no_repeat_ngram(tokens, token, no_repeat_ngram) {
            return token;
        }
    }
    argmax_logits(logits)
}

fn select_sample_with_repeat(
    logits: &[f32],
    top_k: usize,
    temperature: f32,
    tokens: &[u32],
    no_repeat_ngram: usize,
    rng: &mut XorShift64,
) -> u32 {
    let log_probs = log_softmax(logits, temperature);
    let mut candidates = top_k_log_probs(&log_probs, top_k);
    candidates.retain(|(token, _)| !violates_no_repeat_ngram(tokens, *token, no_repeat_ngram));
    if candidates.is_empty() {
        return select_argmax_with_repeat(logits, tokens, no_repeat_ngram);
    }

    let max_logp = candidates
        .iter()
        .map(|(_, v)| *v)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut weights: Vec<f64> = candidates
        .iter()
        .map(|(_, v)| ((*v - max_logp) as f64).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return candidates[0].0;
    }
    let mut target = rng.next_f64() * total;
    for ((token, _), weight) in candidates.iter().zip(weights.drain(..)) {
        if target <= weight {
            return *token;
        }
        target -= weight;
    }
    candidates[0].0
}

#[derive(Debug, Clone)]
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn seeded() -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9E3779B97F4A7C15);
        Self::new(seed)
    }

    fn new(seed: u64) -> Self {
        let mut state = seed;
        if state == 0 {
            state = 0x9E3779B97F4A7C15;
        }
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f64(&mut self) -> f64 {
        let value = self.next_u64() >> 11;
        (value as f64) / ((1u64 << 53) as f64)
    }
}

fn infer_sequence_length_from_plan(plan: &hologram::backend::BackendPlan) -> Option<usize> {
    plan.layout_metadata
        .input_shapes
        .iter()
        .filter_map(|shape| {
            let seq_len = shape[1];
            if shape[2] == 1 && shape[3] == 1 && seq_len > 1 {
                Some(seq_len)
            } else {
                None
            }
        })
        .max()
}

#[derive(Debug)]
struct DecoderStepOutputs {
    logits: Tensor,
    kv_cache: Option<DecoderKvCache>,
}

#[derive(Debug, Clone)]
struct DecoderKvCache {
    values: HashMap<String, Tensor>,
}

impl DecoderKvCache {
    fn from_present_outputs(
        mut outputs: HashMap<String, Tensor>,
        num_layers: usize,
    ) -> Result<Self> {
        let mut values = HashMap::new();
        let mut present_count = 0usize;
        for layer in 0..num_layers {
            let present_names = [
                format!("present.{layer}.decoder.key"),
                format!("present.{layer}.decoder.value"),
                format!("present.{layer}.encoder.key"),
                format!("present.{layer}.encoder.value"),
            ];
            for name in present_names {
                let tensor = outputs
                    .remove(&name)
                    .ok_or_else(|| anyhow::anyhow!("Missing decoder KV output '{}'", name))?;
                let past_name = format!("past_key_values.{}", &name["present.".len()..]);
                values.insert(past_name, tensor);
                present_count += 1;
            }
        }

        if present_count != num_layers * 4 {
            anyhow::bail!(
                "Expected {} KV cache tensors, found {}",
                num_layers * 4,
                present_count
            );
        }

        Ok(Self { values })
    }

    fn as_inputs(&self) -> HashMap<String, Tensor> {
        self.values
            .iter()
            .map(|(name, tensor)| (name.clone(), tensor.clone()))
            .collect()
    }

    fn zeros(num_layers: usize, size_map: &HashMap<String, usize>) -> Result<Self> {
        let mut values = HashMap::new();
        for name in t5_decoder_kv_input_names(num_layers) {
            let size = size_map
                .get(&name)
                .copied()
                .ok_or_else(|| anyhow::anyhow!("Missing size metadata for '{}'", name))?;
            values.insert(name, tensor_from_size(size, 0.0)?);
        }
        Ok(Self { values })
    }
}

#[derive(Debug, Clone)]
struct BeamState {
    tokens: Vec<u32>,
    score: f32,
    finished: bool,
}

impl BeamState {
    fn new(tokens: Vec<u32>, score: f32) -> Self {
        Self {
            tokens,
            score,
            finished: false,
        }
    }
}

fn infer_t5_kv_layers(num_outputs: usize) -> Option<usize> {
    if num_outputs <= 1 {
        None
    } else if (num_outputs - 1).is_multiple_of(4) {
        Some((num_outputs - 1) / 4)
    } else {
        None
    }
}

fn t5_decoder_output_names(num_layers: usize) -> Vec<String> {
    let mut names = Vec::with_capacity(1 + num_layers * 4);
    names.push("logits".to_string());
    for layer in 0..num_layers {
        names.push(format!("present.{layer}.decoder.key"));
        names.push(format!("present.{layer}.decoder.value"));
        names.push(format!("present.{layer}.encoder.key"));
        names.push(format!("present.{layer}.encoder.value"));
    }
    names
}

fn t5_decoder_input_names(num_layers: usize) -> Vec<String> {
    let mut names = Vec::with_capacity(4 + num_layers * 4);
    names.push("encoder_attention_mask".to_string());
    names.push("encoder_hidden_states".to_string());
    names.push("input_ids".to_string());
    names.push("use_cache_branch".to_string());
    names.extend(t5_decoder_kv_input_names(num_layers));
    names
}

fn t5_decoder_kv_input_names(num_layers: usize) -> Vec<String> {
    let mut names = Vec::with_capacity(num_layers * 4);
    for layer in 0..num_layers {
        names.push(format!("past_key_values.{layer}.decoder.key"));
        names.push(format!("past_key_values.{layer}.decoder.value"));
        names.push(format!("past_key_values.{layer}.encoder.key"));
        names.push(format!("past_key_values.{layer}.encoder.value"));
    }
    names
}

fn build_input_size_map(
    plan: &hologram::backend::BackendPlan,
    input_names: &[String],
) -> Result<HashMap<String, usize>> {
    if plan.layout_metadata.num_inputs != input_names.len() {
        anyhow::bail!(
            "Decoder input count mismatch: plan expects {}, but names list has {}",
            plan.layout_metadata.num_inputs,
            input_names.len()
        );
    }

    if plan.layout_metadata.input_sizes.len() < input_names.len() {
        anyhow::bail!(
            "Decoder input size metadata has {} entries, expected {}",
            plan.layout_metadata.input_sizes.len(),
            input_names.len()
        );
    }

    let mut names_sorted: Vec<String> = input_names.to_vec();
    names_sorted.sort();

    let mut sizes = HashMap::with_capacity(input_names.len());
    for (idx, name) in names_sorted.into_iter().enumerate() {
        sizes.insert(name, plan.layout_metadata.input_sizes[idx]);
    }

    Ok(sizes)
}

fn tensor_from_size(size_bytes: usize, fill: f32) -> Result<Tensor> {
    if size_bytes == 0 {
        return Ok(Tensor::new(vec![fill], vec![1]));
    }

    if !size_bytes.is_multiple_of(std::mem::size_of::<f32>()) {
        anyhow::bail!("Input size {} is not aligned to f32", size_bytes);
    }

    let numel = size_bytes / std::mem::size_of::<f32>();
    Ok(Tensor::new(vec![fill; numel], vec![numel]))
}

fn extract_decoder_step_outputs(
    outputs: HashMap<String, Tensor>,
    kv_layers: Option<usize>,
    kv_enabled: bool,
) -> Result<DecoderStepOutputs> {
    let num_outputs = outputs.len();
    if num_outputs == 1 {
        let logits = outputs
            .get("output")
            .or_else(|| outputs.values().next())
            .ok_or_else(|| anyhow::anyhow!("Decoder produced no output"))?
            .clone();
        return Ok(DecoderStepOutputs {
            logits,
            kv_cache: None,
        });
    }

    if kv_enabled {
        let num_layers = kv_layers.ok_or_else(|| {
            anyhow::anyhow!("Decoder outputs do not match expected KV-cache layout")
        })?;
        let mut output_names = t5_decoder_output_names(num_layers);
        output_names.sort();

        let mut ordered = Vec::with_capacity(num_outputs);
        for idx in 0..num_outputs {
            let key = format!("output_{}", idx);
            let tensor = outputs
                .get(&key)
                .ok_or_else(|| anyhow::anyhow!("Missing decoder output '{}'", key))?
                .clone();
            ordered.push(tensor);
        }

        let mut named_outputs = HashMap::new();
        for (name, tensor) in output_names.into_iter().zip(ordered.into_iter()) {
            named_outputs.insert(name, tensor);
        }

        let logits = named_outputs
            .remove("logits")
            .ok_or_else(|| anyhow::anyhow!("Decoder outputs missing logits"))?;
        let kv_cache = Some(DecoderKvCache::from_present_outputs(
            named_outputs,
            num_layers,
        )?);
        return Ok(DecoderStepOutputs { logits, kv_cache });
    }

    let logits = outputs
        .get("output_0")
        .or_else(|| outputs.get("output"))
        .ok_or_else(|| anyhow::anyhow!("Decoder produced no output_0"))?
        .clone();

    Ok(DecoderStepOutputs {
        logits,
        kv_cache: None,
    })
}

/// Fallback tokenization when no tokenizer is available.
fn fallback_tokenize(prompt: &str, max_length: usize) -> (Vec<u32>, usize) {
    let mut input_ids: Vec<u32> = Vec::new();

    for word in prompt.split_whitespace() {
        let token = match word.to_lowercase().as_str() {
            "tell" => 129,
            "me" => 140,
            "a" => 3,
            "joke" => 5765,
            "in" => 16,
            "english" => 1566,
            "translate" => 13959,
            "hello" => 8774,
            "how" => 149,
            "are" => 33,
            "you" => 25,
            "what" => 125,
            "is" => 19,
            "the" => 8,
            "weather" => 1969,
            "today" => 469,
            _ => {
                let sum: u32 = word.bytes().map(|b| b as u32).sum();
                100 + (sum % 30000)
            }
        };
        input_ids.push(token);
    }

    input_ids.push(1); // Add EOS token
    let actual_len = input_ids.len();

    // Pad to max_length
    while input_ids.len() < max_length {
        input_ids.push(0);
    }
    input_ids.truncate(max_length);

    (input_ids, actual_len)
}

/// Decode tokens using the tokenizer if available, otherwise show raw tokens.
fn decode_tokens_with_tokenizer(
    tokens: &[u32],
    tokenizer: Option<&crate::tokenizers::sentencepiece::SentencePieceTokenizer>,
) -> String {
    use crate::tokenizers::Tokenizer;

    if let Some(tok) = tokenizer {
        match tok.decode(tokens) {
            Ok(text) => {
                let text = text.trim().to_string();
                if !text.is_empty() {
                    return text;
                }
            }
            Err(e) => {
                warn!("Tokenizer decode error: {}", e);
            }
        }
    }

    // Fallback: show raw token IDs
    format!(
        "[Raw tokens: {:?}]",
        tokens
            .iter()
            .filter(|&&t| t != 0 && t != 1)
            .collect::<Vec<_>>()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram::backend::{BackendPlan, BackendType};
    use std::collections::HashMap;
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
        let inputs = vec!["prompt=test".to_string(), "steps=50".to_string()];
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
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Expected 'name=value'")
        );
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

    #[test]
    fn test_infer_sequence_length_from_plan() {
        let mut plan = BackendPlan::new(BackendType::Cpu);
        plan.layout_metadata.input_shapes = vec![[1, 128, 1, 1], [1, 128, 1, 1]];

        assert_eq!(infer_sequence_length_from_plan(&plan), Some(128));
    }

    #[test]
    fn test_infer_sequence_length_from_plan_missing_seq() {
        let mut plan = BackendPlan::new(BackendType::Cpu);
        plan.layout_metadata.input_shapes = vec![[1, 1, 1, 1], [1, 64, 512, 1]];

        assert_eq!(infer_sequence_length_from_plan(&plan), None);
    }

    #[test]
    fn test_infer_t5_kv_layers() {
        assert_eq!(infer_t5_kv_layers(1), None);
        assert_eq!(infer_t5_kv_layers(5), Some(1));
        assert_eq!(infer_t5_kv_layers(25), Some(6));
        assert_eq!(infer_t5_kv_layers(6), None);
    }

    #[test]
    fn test_t5_decoder_input_names() {
        let names = t5_decoder_input_names(1);
        assert!(names.contains(&"encoder_attention_mask".to_string()));
        assert!(names.contains(&"encoder_hidden_states".to_string()));
        assert!(names.contains(&"input_ids".to_string()));
        assert!(names.contains(&"use_cache_branch".to_string()));
        assert!(names.contains(&"past_key_values.0.decoder.key".to_string()));
        assert!(names.contains(&"past_key_values.0.decoder.value".to_string()));
        assert!(names.contains(&"past_key_values.0.encoder.key".to_string()));
        assert!(names.contains(&"past_key_values.0.encoder.value".to_string()));
    }

    #[test]
    fn test_build_input_size_map() {
        let mut plan = BackendPlan::new(BackendType::Cpu);
        plan.layout_metadata.num_inputs = 4;
        plan.layout_metadata.input_sizes = vec![4, 8, 12, 16];

        let names = vec![
            "b".to_string(),
            "a".to_string(),
            "d".to_string(),
            "c".to_string(),
        ];

        let sizes = build_input_size_map(&plan, &names).unwrap();
        assert_eq!(sizes.get("a"), Some(&4));
        assert_eq!(sizes.get("b"), Some(&8));
        assert_eq!(sizes.get("c"), Some(&12));
        assert_eq!(sizes.get("d"), Some(&16));
    }

    #[test]
    fn test_argmax_logits() {
        let logits = vec![0.1, 0.2, 0.3, 0.4];
        let token = argmax_logits(&logits);
        assert_eq!(token, 3);
    }

    #[test]
    fn test_select_sample_with_repeat() {
        let logits = vec![0.1, 2.0, 0.5, 1.5];
        let mut rng = XorShift64::new(42);
        let token = select_sample_with_repeat(&logits, 2, 1.0, &[], 0, &mut rng);
        assert!(token == 1 || token == 3);
    }

    #[test]
    fn test_select_argmax_with_repeat() {
        let logits = vec![0.1, 2.0, 0.5, 1.5];
        let tokens = vec![1, 2, 1];
        let token = select_argmax_with_repeat(&logits, &tokens, 2);
        assert_ne!(token, 2);
    }

    #[test]
    fn test_apply_token_filters_masks_specials() {
        let logits = vec![0.1, 0.9, 0.2, 1.1];
        let special_ids = vec![3];
        let filtered = apply_token_filters(&logits, 1, 0, 2, 1, false, &special_ids);
        assert!(filtered[0].is_infinite() && filtered[0].is_sign_negative());
        assert!(filtered[1].is_infinite() && filtered[1].is_sign_negative());
        assert!(filtered[2].is_infinite() && filtered[2].is_sign_negative());
        assert!(filtered[3].is_infinite() && filtered[3].is_sign_negative());
    }

    #[test]
    fn test_extract_decoder_step_outputs_with_kv_cache() {
        let mut outputs = HashMap::new();
        outputs.insert("output_0".to_string(), Tensor::new(vec![0.1], vec![1]));
        outputs.insert("output_1".to_string(), Tensor::new(vec![1.0], vec![1]));
        outputs.insert("output_2".to_string(), Tensor::new(vec![2.0], vec![1]));
        outputs.insert("output_3".to_string(), Tensor::new(vec![3.0], vec![1]));
        outputs.insert("output_4".to_string(), Tensor::new(vec![4.0], vec![1]));

        let step_outputs = extract_decoder_step_outputs(outputs, Some(1), true).unwrap();
        assert_eq!(step_outputs.logits.to_f32(), &[0.1]);

        let cache = step_outputs.kv_cache.expect("expected kv cache");
        let inputs = cache.as_inputs();
        assert_eq!(inputs.len(), 4);
        assert!(inputs.contains_key("past_key_values.0.decoder.key"));
        assert!(inputs.contains_key("past_key_values.0.decoder.value"));
        assert!(inputs.contains_key("past_key_values.0.encoder.key"));
        assert!(inputs.contains_key("past_key_values.0.encoder.value"));
    }
}
