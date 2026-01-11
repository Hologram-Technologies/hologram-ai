//! Unified configuration for ONNX compilation and execution.
//!
//! This module provides a single configuration format that supports:
//! - Minimal configs for simple models (4 lines)
//! - Complex multi-model pipelines (Stable Diffusion, Whisper)
//! - Convention over configuration (auto-inference of names, handlers)
//!
//! # Example
//!
//! ## Minimal Config
//! ```toml
//! model = "mnist.onnx"
//! ```
//!
//! ## Full Pipeline Config
//! ```toml
//! [inputs]
//! prompt = { type = "text", default = "a photo of a cat" }
//!
//! [models]
//! encoder = "models/encoder.onnx"
//! decoder = "models/decoder.onnx"
//!
//! [compiler]
//! enable_partitioning = true
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use super::error::ConfigError;

// =============================================================================
// Default Value Functions
// =============================================================================

fn default_weight_threshold() -> usize {
    4096
}

fn default_partition_size() -> usize {
    500
}

fn default_true() -> bool {
    true
}

// =============================================================================
// Core Types
// =============================================================================

/// Unified configuration for ONNX compilation and execution.
///
/// Supports both minimal shorthand syntax and full pipeline specification.
/// Missing fields are auto-inferred where possible.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct UnifiedConfig {
    /// Shorthand: single model path (auto-generates model entry)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// Shorthand: single input definition
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<InputDef>,

    /// Shorthand: single output name
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,

    /// Pipeline inputs with types and defaults
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub inputs: HashMap<String, InputDef>,

    /// Models to load (auto-named from paths if not specified)
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub models: HashMap<String, ModelDef>,

    /// Compilation settings
    #[serde(default)]
    pub compiler: CompilerConfig,

    /// Runtime execution settings
    #[serde(default)]
    pub runtime: RuntimeConfig,

    /// Pipeline bundle configuration (for HOLM format)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pipeline: Option<PipelineBundleConfig>,

    /// Tokenizer configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokenizer: Option<crate::tokenizers::TokenizerConfig>,

    /// Execution stages (auto-generated from models if not specified)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stages: Vec<StageDef>,

    /// Output handlers (auto-inferred from tensor shapes)
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub outputs: HashMap<String, OutputDef>,

    /// Pipeline metadata
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Pipeline version
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,

    /// Pipeline description
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

// =============================================================================
// Input Types
// =============================================================================

/// Input definition with type, default, and constraints.
///
/// Supports two forms:
/// - Simple: just a string default value
/// - Full: typed input with constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum InputDef {
    /// Simple string default
    Simple(String),

    /// Typed input with constraints
    Full(InputSpec),
}

impl InputDef {
    /// Create a simple string input.
    pub fn simple(default: impl Into<String>) -> Self {
        Self::Simple(default.into())
    }

    /// Create a typed input.
    pub fn typed(dtype: InputType) -> InputSpec {
        InputSpec {
            dtype,
            default: None,
            shape: None,
        }
    }

    /// Get the input type, defaulting to Text for simple strings.
    pub fn input_type(&self) -> InputType {
        match self {
            Self::Simple(_) => InputType::Text,
            Self::Full(spec) => spec.dtype.clone(),
        }
    }

    /// Get the default value if any.
    pub fn default_value(&self) -> Option<&serde_json::Value> {
        match self {
            Self::Simple(_) => None, // Simple strings ARE the default
            Self::Full(spec) => spec.default.as_ref(),
        }
    }

    /// Get the default string for simple inputs.
    pub fn default_string(&self) -> Option<&str> {
        match self {
            Self::Simple(s) => Some(s.as_str()),
            Self::Full(_) => None,
        }
    }
}

/// Full input specification with type, default, and constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputSpec {
    /// Input data type
    #[serde(rename = "type")]
    pub dtype: InputType,

    /// Default value (type depends on dtype)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<serde_json::Value>,

    /// Shape constraints for tensor inputs
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape: Option<Vec<DimExpr>>,
}

/// Supported input types.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum InputType {
    /// Text input (string)
    #[default]
    Text,
    /// Image input (path or base64)
    Image,
    /// Audio input (path or raw samples)
    Audio,
    /// Raw tensor input
    Tensor,
    /// Integer input
    Int,
    /// Float input
    Float,
    /// File path input
    Path,
}

// =============================================================================
// Dimension Expressions
// =============================================================================

/// Dimension expression for dynamic shapes.
///
/// Supports:
/// - Fixed dimensions: `512`
/// - Symbolic dimensions: `"batch"`, `"seq_len"`
/// - Expressions: `"height/8"`, `"width*2"`
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum DimExpr {
    /// Fixed dimension
    Fixed(i64),
    /// Symbolic dimension or expression
    Symbolic(String),
}

impl DimExpr {
    /// Create a fixed dimension.
    pub fn fixed(n: i64) -> Self {
        Self::Fixed(n)
    }

    /// Create a symbolic dimension.
    pub fn symbolic(name: impl Into<String>) -> Self {
        Self::Symbolic(name.into())
    }

    /// Evaluate the expression with variable bindings.
    ///
    /// For fixed dimensions, returns the value directly.
    /// For symbolic dimensions, looks up in bindings or parses simple expressions.
    pub fn evaluate(&self, bindings: &HashMap<String, i64>) -> Result<i64, ConfigError> {
        match self {
            Self::Fixed(n) => Ok(*n),
            Self::Symbolic(expr) => {
                // First try direct lookup
                if let Some(&val) = bindings.get(expr) {
                    return Ok(val);
                }

                // Try to parse simple expressions like "height/8" or "batch*2"
                if let Some((lhs, op, rhs)) = Self::parse_binary_expr(expr) {
                    let lhs_val = bindings.get(&lhs).ok_or_else(|| {
                        ConfigError::invalid_value(
                            "dimension",
                            format!("unknown variable: {}", lhs),
                        )
                    })?;
                    let rhs_val: i64 = rhs.parse().map_err(|_| {
                        ConfigError::invalid_value(
                            "dimension",
                            format!("invalid number in expression: {}", rhs),
                        )
                    })?;

                    match op {
                        '/' => Ok(lhs_val / rhs_val),
                        '*' => Ok(lhs_val * rhs_val),
                        '+' => Ok(lhs_val + rhs_val),
                        '-' => Ok(lhs_val - rhs_val),
                        _ => Err(ConfigError::invalid_value(
                            "dimension",
                            format!("unknown operator: {}", op),
                        )),
                    }
                } else {
                    Err(ConfigError::invalid_value(
                        "dimension",
                        format!("cannot evaluate expression: {}", expr),
                    ))
                }
            }
        }
    }

    /// Parse a simple binary expression like "height/8".
    fn parse_binary_expr(expr: &str) -> Option<(String, char, String)> {
        for op in ['/', '*', '+', '-'] {
            if let Some(pos) = expr.find(op) {
                let lhs = expr[..pos].trim().to_string();
                let rhs = expr[pos + 1..].trim().to_string();
                if !lhs.is_empty() && !rhs.is_empty() {
                    return Some((lhs, op, rhs));
                }
            }
        }
        None
    }

    /// Check if this is a fixed (known) dimension.
    pub fn is_fixed(&self) -> bool {
        matches!(self, Self::Fixed(_))
    }

    /// Get the fixed value if this is a fixed dimension.
    pub fn as_fixed(&self) -> Option<i64> {
        match self {
            Self::Fixed(n) => Some(*n),
            Self::Symbolic(_) => None,
        }
    }
}

// =============================================================================
// Model Types
// =============================================================================

/// Model definition.
///
/// Supports two forms:
/// - Simple: just a path string (name inferred from filename)
/// - Full: explicit specification with optional precompiled path
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ModelDef {
    /// Simple path string (name inferred from filename)
    Path(String),

    /// Full model specification
    Full(ModelSpec),
}

impl ModelDef {
    /// Get the model path.
    pub fn path(&self) -> &str {
        match self {
            Self::Path(p) => p,
            Self::Full(spec) => &spec.path,
        }
    }

    /// Get the precompiled path if specified.
    pub fn precompiled(&self) -> Option<&str> {
        match self {
            Self::Path(_) => None,
            Self::Full(spec) => spec.precompiled.as_deref(),
        }
    }

    /// Get the opset version if specified.
    pub fn opset(&self) -> Option<u32> {
        match self {
            Self::Path(_) => None,
            Self::Full(spec) => spec.opset,
        }
    }

    /// Infer model name from path.
    ///
    /// Extracts the filename stem, e.g., "models/encoder.onnx" -> "encoder"
    pub fn infer_name(&self) -> String {
        let path = self.path();
        Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model")
            .to_string()
    }

    /// Get the model type if specified.
    ///
    /// Returns "onnx" or "tokenizer"/"sentencepiece" to indicate how the
    /// model should be compiled.
    pub fn model_type(&self) -> Option<&str> {
        match self {
            Self::Path(_) => None,
            Self::Full(spec) => spec.model_type.as_deref(),
        }
    }
}

/// Full model specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    /// Path to the ONNX model file
    pub path: String,

    /// Optional path to precompiled .holo file
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub precompiled: Option<String>,

    /// Optional ONNX opset version
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opset: Option<u32>,

    /// Model type: "onnx" (default) or "tokenizer"
    ///
    /// Used by compile-pipeline to determine how to compile the model.
    /// - "onnx": Standard ONNX model compilation
    /// - "tokenizer" or "sentencepiece": Tokenizer JSON compilation
    #[serde(default, skip_serializing_if = "Option::is_none", rename = "type")]
    pub model_type: Option<String>,
}

// =============================================================================
// Compiler Configuration
// =============================================================================

/// Compiler configuration for ONNX models.
///
/// Controls weight storage, graph partitioning, and ISA optimizations.
/// Maps directly to `OnnxConfig` from hologram-onnx-core.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilerConfig {
    /// Threshold (in bytes) for storing weights externally
    #[serde(default = "default_weight_threshold")]
    pub weight_threshold: usize,

    /// Enable graph partitioning for large models
    #[serde(default)]
    pub enable_partitioning: bool,

    /// Number of nodes per partition
    #[serde(default = "default_partition_size")]
    pub partition_size: usize,

    /// Enable Conv2D decomposition
    #[serde(default = "default_true")]
    pub decompose_conv2d: bool,

    /// Enable pooling decomposition
    #[serde(default = "default_true")]
    pub decompose_pooling: bool,

    /// Enable Resize upscaling (set to false to save memory, outputs at input resolution)
    #[serde(default = "default_true")]
    pub enable_resize_upscaling: bool,

    /// Enable packed weight serialization for fast runtime execution
    #[serde(default = "default_true")]
    pub pack_weights: bool,

    /// Memory budget in megabytes (MB)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_budget: Option<usize>,

    /// Target backend (cpu, cuda, metal, webgpu)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,

    /// Enable aggressive fusion patterns for better performance
    ///
    /// When enabled, the compiler applies advanced pattern matching:
    /// - MatMul + Bias → FusedMatMulBias
    /// - MatMul + Activation → FusedMatMulActivation
    /// - Div(x, Sqrt(y)) → Mul(x, Rsqrt(y))
    /// - Consecutive Add/Mul → vectorized operations
    #[serde(default)]
    pub aggressive_fusion: bool,

    /// Optimization level (0-3)
    ///
    /// - 0: No optimization
    /// - 1: Basic algebraic rewrites
    /// - 2: + Auto fusion of elementwise chains (default)
    /// - 3: + Dead code elimination, aggressive patterns
    #[serde(default = "default_opt_level")]
    pub opt_level: u8,

    /// Enable automatic fusion of element-wise operations
    #[serde(default = "default_true")]
    pub auto_fuse: bool,

    /// Enable FP16 (half precision) execution for reduced memory and faster compute.
    ///
    /// When enabled, weights and activations are converted to float16 format.
    /// This reduces memory usage by ~50% and can improve performance on
    /// hardware with native FP16 support.
    ///
    /// Note: May slightly reduce numerical precision.
    #[serde(default)]
    pub use_fp16: bool,

    /// Enable INT8 quantization for maximum compression.
    ///
    /// When enabled, weights are quantized to 8-bit integers with scale factors.
    /// This reduces memory by ~75% compared to FP32 and enables faster integer
    /// operations on compatible hardware.
    ///
    /// Requires calibration data for best accuracy (dynamic quantization used otherwise).
    #[serde(default)]
    pub use_int8: bool,

    /// Quantization mode for weight compression.
    ///
    /// - "none": No quantization (default, full FP32)
    /// - "dynamic": Dynamic range quantization at runtime
    /// - "static": Static quantization (requires calibration)
    #[serde(default = "default_quantization_mode")]
    pub quantization_mode: String,
}

fn default_quantization_mode() -> String {
    "none".to_string()
}

fn default_opt_level() -> u8 {
    2
}

impl Default for CompilerConfig {
    fn default() -> Self {
        Self {
            weight_threshold: default_weight_threshold(),
            enable_partitioning: false,
            partition_size: default_partition_size(),
            decompose_conv2d: true,
            decompose_pooling: true,
            enable_resize_upscaling: true,
            pack_weights: true,
            memory_budget: None,
            backend: None,
            aggressive_fusion: false,
            opt_level: 2,
            auto_fuse: true,
            use_fp16: false,
            use_int8: false,
            quantization_mode: "none".to_string(),
        }
    }
}

// =============================================================================
// Runtime Configuration
// =============================================================================

/// Runtime execution configuration.
///
/// Controls how models are executed at runtime, including memory management
/// strategies and performance optimizations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct RuntimeConfig {
    /// Enable layer-wise buffer allocation for memory efficiency.
    ///
    /// When enabled, buffers are allocated only when first needed and freed
    /// as soon as their last use completes. This significantly reduces peak
    /// memory usage for large models at the cost of slightly higher allocation
    /// overhead.
    ///
    /// Recommended for:
    /// - Models with many intermediate tensors (transformers, diffusion)
    /// - Memory-constrained environments
    /// - Models over ~100MB
    ///
    /// Default: false (pre-allocate all buffers for maximum performance)
    #[serde(default)]
    pub layerwise_execution: bool,

    /// Enable verbose execution logging (shows per-level timing).
    #[serde(default)]
    pub verbose: bool,
}

// =============================================================================
// Pipeline Bundle Configuration
// =============================================================================

/// Pipeline bundle configuration for HOLM format.
///
/// When a pipeline bundle is specified, models are loaded from the single
/// HOLM bundle file instead of individual .holo files. This enables:
/// - Single-file deployment (~300MB for full T5 pipeline)
/// - Per-model memory-mapped weights
/// - Simpler distribution and caching
///
/// # Example
///
/// ```toml
/// [pipeline]
/// bundle = "models/t5-pipeline.holo"
///
/// [models.encoder]
/// # No path needed - loaded from bundle by name
///
/// [models.decoder]
/// # No path needed - loaded from bundle by name
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PipelineBundleConfig {
    /// Path to the pipeline bundle (HOLM format).
    ///
    /// When specified, models are loaded from this bundle instead of
    /// individual .holo files. Model names in the config must match
    /// the names stored in the bundle.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bundle: Option<String>,

    /// Verify bundle checksums on load (default: true).
    ///
    /// Disabling this can speed up loading but removes integrity checks.
    #[serde(default = "default_true")]
    pub verify_checksums: bool,
}

impl Default for PipelineBundleConfig {
    fn default() -> Self {
        Self {
            bundle: None,
            verify_checksums: true,
        }
    }
}

// =============================================================================
// Stage Types
// =============================================================================

/// Stage definition for execution pipeline.
///
/// Supports multiple stage types:
/// - Model: invoke a model
/// - Builtin: invoke a built-in operation
/// - Loop: iterate over a range
/// - Conditional: branch based on condition
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StageDef {
    /// Simple model invocation
    Model(ModelStage),

    /// Built-in operation
    Builtin(BuiltinStage),

    /// Loop stage
    Loop(LoopStage),

    /// Conditional stage
    Conditional(ConditionalStage),
}

impl StageDef {
    /// Create a simple model stage.
    pub fn model(name: impl Into<String>) -> Self {
        Self::Model(ModelStage {
            model: name.into(),
            inputs: HashMap::new(),
            outputs: Vec::new(),
        })
    }

    /// Create a builtin operation stage.
    pub fn builtin(name: impl Into<String>) -> Self {
        Self::Builtin(BuiltinStage {
            builtin: name.into(),
            args: HashMap::new(),
            outputs: Vec::new(),
        })
    }
}

/// Model invocation stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStage {
    /// Model name (references a model in the models section)
    pub model: String,

    /// Input mappings (model input name → expression)
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub inputs: HashMap<String, Expr>,

    /// Output tensor names to capture
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub outputs: Vec<String>,
}

/// Built-in operation stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuiltinStage {
    /// Built-in operation name
    pub builtin: String,

    /// Operation arguments
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub args: HashMap<String, Expr>,

    /// Output names
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub outputs: Vec<String>,
}

/// Loop stage for iterative execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopStage {
    /// Iterator expression (e.g., "range(steps)")
    pub over: String,

    /// Loop variable name
    #[serde(rename = "as")]
    pub as_var: String,

    /// Stages to execute in each iteration
    pub stages: Vec<StageDef>,
}

/// Conditional stage for branching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditionalStage {
    /// Condition expression
    #[serde(rename = "if")]
    pub condition: String,

    /// Stages to execute if condition is true
    #[serde(rename = "then")]
    pub then_stages: Vec<StageDef>,

    /// Stages to execute if condition is false
    #[serde(default, rename = "else", skip_serializing_if = "Vec::is_empty")]
    pub else_stages: Vec<StageDef>,
}

// =============================================================================
// Expression Types
// =============================================================================

/// Expression for dynamic values.
///
/// Used in stage inputs to reference pipeline variables,
/// call functions, or provide literal values.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Expr {
    /// Literal value (string, number, etc.)
    Literal(serde_json::Value),
}

impl Expr {
    /// Create a string literal expression.
    pub fn string(s: impl Into<String>) -> Self {
        Self::Literal(serde_json::Value::String(s.into()))
    }

    /// Create a numeric literal expression.
    pub fn number(n: impl Into<serde_json::Number>) -> Self {
        Self::Literal(serde_json::Value::Number(n.into()))
    }

    /// Check if this is a reference expression (starts with variable name).
    pub fn is_reference(&self) -> bool {
        match self {
            Self::Literal(serde_json::Value::String(s)) => {
                // References are strings that look like "var.output" or "var"
                !s.starts_with('"') && s.parse::<f64>().is_err()
            }
            _ => false,
        }
    }

    /// Get as string if this is a string value.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Literal(serde_json::Value::String(s)) => Some(s),
            _ => None,
        }
    }
}

// =============================================================================
// Output Types
// =============================================================================

/// Output definition.
///
/// Supports two forms:
/// - Simple: just a tensor name
/// - Full: tensor with handler specification
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OutputDef {
    /// Simple tensor name
    Simple(String),

    /// Full output specification
    Full(OutputSpec),
}

impl OutputDef {
    /// Create a simple output.
    pub fn simple(tensor: impl Into<String>) -> Self {
        Self::Simple(tensor.into())
    }

    /// Create a typed output with handler.
    pub fn with_handler(tensor: impl Into<String>, handler: OutputHandlerType) -> Self {
        Self::Full(OutputSpec {
            tensor: tensor.into(),
            handler,
            options: HashMap::new(),
        })
    }

    /// Get the tensor name.
    pub fn tensor(&self) -> &str {
        match self {
            Self::Simple(s) => s,
            Self::Full(spec) => &spec.tensor,
        }
    }

    /// Get the handler type, defaulting to Auto.
    pub fn handler_type(&self) -> OutputHandlerType {
        match self {
            Self::Simple(_) => OutputHandlerType::Auto,
            Self::Full(spec) => spec.handler.clone(),
        }
    }
}

/// Full output specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputSpec {
    /// Tensor name to process
    pub tensor: String,

    /// Handler type
    pub handler: OutputHandlerType,

    /// Handler-specific options
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub options: HashMap<String, serde_json::Value>,
}

/// Output handler types.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum OutputHandlerType {
    /// Image output (PNG, JPEG)
    Image,
    /// Audio output (WAV)
    Audio,
    /// Text output (decoded tokens)
    Text,
    /// JSON output
    Json,
    /// Binary output (raw tensor data)
    Binary,
    /// Auto-detect based on tensor shape
    #[default]
    Auto,
}

// =============================================================================
// Config Loading and Normalization
// =============================================================================

impl UnifiedConfig {
    /// Load config from a TOML file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let contents = std::fs::read_to_string(path)?;
        Self::from_str(&contents)
    }

    /// Parse config from a TOML string.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(toml: &str) -> Result<Self, ConfigError> {
        let mut config: UnifiedConfig = toml::from_str(toml)?;
        config.normalize()?;
        config.validate()?;
        Ok(config)
    }

    /// Normalize the config by expanding shorthands.
    ///
    /// This handles:
    /// - Single `model` field → `models` map
    /// - Single `input` field → `inputs` map
    /// - Single `output` field → `outputs` map
    fn normalize(&mut self) -> Result<(), ConfigError> {
        // Expand single model to models map
        if let Some(model_path) = self.model.take() {
            let name = Path::new(&model_path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("model")
                .to_string();
            if self.models.is_empty() {
                self.models.insert(name, ModelDef::Path(model_path));
            }
        }

        // Expand single input to inputs map
        if let Some(input) = self.input.take()
            && self.inputs.is_empty()
        {
            self.inputs.insert("input".to_string(), input);
        }

        // Expand single output to outputs map
        if let Some(output) = self.output.take()
            && self.outputs.is_empty()
        {
            self.outputs
                .insert("output".to_string(), OutputDef::Simple(output));
        }

        // Auto-generate stages if empty and we have models
        if self.stages.is_empty() && !self.models.is_empty() {
            for name in self.models.keys() {
                self.stages.push(StageDef::Model(ModelStage {
                    model: name.clone(),
                    inputs: HashMap::new(),
                    outputs: Vec::new(),
                }));
            }
        }

        Ok(())
    }

    /// Validate the config structure.
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Validate compiler config
        if self.compiler.weight_threshold == 0 {
            return Err(ConfigError::invalid_value(
                "compiler.weight_threshold",
                "must be greater than 0",
            ));
        }

        if self.compiler.partition_size < 10 {
            return Err(ConfigError::invalid_value(
                "compiler.partition_size",
                "must be >= 10",
            ));
        }

        if let Some(budget) = self.compiler.memory_budget
            && budget < 100
        {
            return Err(ConfigError::invalid_value(
                "compiler.memory_budget",
                "must be >= 100 MB",
            ));
        }

        // Validate stage model references
        for stage in &self.stages {
            if let StageDef::Model(ms) = stage
                && !self.models.contains_key(&ms.model)
                && !self.models.is_empty()
            {
                return Err(ConfigError::invalid_value(
                    "stages.model",
                    format!("unknown model: {}", ms.model),
                ));
            }
        }

        Ok(())
    }

    /// Save config to a TOML file.
    pub fn to_file(&self, path: impl AsRef<Path>) -> Result<(), ConfigError> {
        let toml_str = toml::to_string_pretty(self)?;
        std::fs::write(path, toml_str)?;
        Ok(())
    }

    /// Check if this is a minimal single-model config.
    pub fn is_single_model(&self) -> bool {
        self.models.len() == 1 && self.stages.len() <= 1
    }

    /// Get the single model if this is a single-model config.
    pub fn single_model(&self) -> Option<(&String, &ModelDef)> {
        if self.is_single_model() {
            self.models.iter().next()
        } else {
            None
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimal_config() {
        let toml = r#"
            model = "mnist.onnx"
        "#;

        let config = UnifiedConfig::from_str(toml).unwrap();
        assert!(config.is_single_model());
        assert_eq!(config.models.len(), 1);
        assert!(config.models.contains_key("mnist"));
    }

    #[test]
    fn test_config_with_input() {
        let toml = r#"
            model = "mnist.onnx"
            input = { type = "image", default = "digit.png" }
            output = "classes"
        "#;

        let config = UnifiedConfig::from_str(toml).unwrap();
        assert_eq!(config.inputs.len(), 1);
        assert!(config.inputs.contains_key("input"));
        assert_eq!(config.outputs.len(), 1);
    }

    #[test]
    fn test_full_pipeline_config() {
        let toml = r#"
            name = "stable-diffusion"
            version = "1.0"

            [inputs]
            prompt = { type = "text", default = "a photo" }
            steps = { type = "int", default = 50 }

            [models]
            text_encoder = "models/text_encoder.onnx"
            unet = "models/unet.onnx"
            vae_decoder = "models/vae_decoder.onnx"

            [compiler]
            enable_partitioning = true
            partition_size = 1000
        "#;

        let config = UnifiedConfig::from_str(toml).unwrap();
        assert_eq!(config.name, Some("stable-diffusion".to_string()));
        assert_eq!(config.inputs.len(), 2);
        assert_eq!(config.models.len(), 3);
        assert!(config.compiler.enable_partitioning);
        assert_eq!(config.compiler.partition_size, 1000);
    }

    #[test]
    fn test_compiler_defaults() {
        let toml = r#"
            model = "test.onnx"
        "#;

        let config = UnifiedConfig::from_str(toml).unwrap();
        assert_eq!(config.compiler.weight_threshold, 4096);
        assert!(!config.compiler.enable_partitioning);
        assert_eq!(config.compiler.partition_size, 500);
        assert!(config.compiler.decompose_conv2d);
        assert!(config.compiler.decompose_pooling);
        assert!(config.compiler.pack_weights);
    }

    #[test]
    fn test_dim_expr_fixed() {
        let expr = DimExpr::Fixed(512);
        let bindings = HashMap::new();
        assert_eq!(expr.evaluate(&bindings).unwrap(), 512);
    }

    #[test]
    fn test_dim_expr_symbolic() {
        let expr = DimExpr::Symbolic("batch".to_string());
        let mut bindings = HashMap::new();
        bindings.insert("batch".to_string(), 32);
        assert_eq!(expr.evaluate(&bindings).unwrap(), 32);
    }

    #[test]
    fn test_dim_expr_binary() {
        let expr = DimExpr::Symbolic("height/8".to_string());
        let mut bindings = HashMap::new();
        bindings.insert("height".to_string(), 512);
        assert_eq!(expr.evaluate(&bindings).unwrap(), 64);
    }

    #[test]
    fn test_model_def_path() {
        let model = ModelDef::Path("models/encoder.onnx".to_string());
        assert_eq!(model.path(), "models/encoder.onnx");
        assert_eq!(model.infer_name(), "encoder");
    }

    #[test]
    fn test_input_def_simple() {
        let input = InputDef::Simple("hello".to_string());
        assert_eq!(input.input_type(), InputType::Text);
        assert_eq!(input.default_string(), Some("hello"));
    }

    #[test]
    fn test_input_def_full() {
        let input = InputDef::Full(InputSpec {
            dtype: InputType::Image,
            default: Some(serde_json::Value::String("test.png".to_string())),
            shape: None,
        });
        assert_eq!(input.input_type(), InputType::Image);
    }

    #[test]
    fn test_validation_invalid_weight_threshold() {
        let toml = r#"
            model = "test.onnx"
            [compiler]
            weight_threshold = 0
        "#;

        let result = UnifiedConfig::from_str(toml);
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_invalid_partition_size() {
        let toml = r#"
            model = "test.onnx"
            [compiler]
            partition_size = 5
        "#;

        let result = UnifiedConfig::from_str(toml);
        assert!(result.is_err());
    }

    #[test]
    fn test_stage_auto_generation() {
        let toml = r#"
            [models]
            encoder = "encoder.onnx"
            decoder = "decoder.onnx"
        "#;

        let config = UnifiedConfig::from_str(toml).unwrap();
        assert_eq!(config.stages.len(), 2);
    }

    #[test]
    fn test_output_def() {
        let output = OutputDef::Simple("logits".to_string());
        assert_eq!(output.tensor(), "logits");
        assert_eq!(output.handler_type(), OutputHandlerType::Auto);

        let output = OutputDef::with_handler("image", OutputHandlerType::Image);
        assert_eq!(output.handler_type(), OutputHandlerType::Image);
    }

    #[test]
    fn test_round_trip() {
        let toml = r#"
            model = "test.onnx"
        "#;

        let config = UnifiedConfig::from_str(toml).unwrap();
        let serialized = toml::to_string(&config).unwrap();
        let reparsed = UnifiedConfig::from_str(&serialized).unwrap();
        assert_eq!(config.models.len(), reparsed.models.len());
    }
}
