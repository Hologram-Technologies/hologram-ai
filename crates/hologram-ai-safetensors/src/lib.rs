//! SafeTensors format support for hologram-ai.
//!
//! This crate provides parsing and compilation of SafeTensors model directories
//! (typically from HuggingFace) to hologram IR.
//!
//! ## Directory Structure
//!
//! A SafeTensors model directory typically contains:
//! - `config.json` - Model configuration
//! - `*.safetensors` - Weight files (may be sharded)
//! - `tokenizer.json` - Tokenizer configuration (optional)
//!
//! ## Example
//!
//! ```ignore
//! use hologram_ai_safetensors::SafeTensorsCompiler;
//!
//! let compiler = SafeTensorsCompiler::new();
//! let (holo_bytes, weight_bytes) = compiler.compile_dir("./model/")?;
//! ```

#![deny(missing_docs)]
#![warn(clippy::all)]

pub mod config;
pub mod error;
pub mod parser;

pub use config::HfConfig;
pub use error::{Result, SafeTensorsError};
pub use parser::SafeTensorsParser;

use hologram_ai_common::{TransformerConfig, WeightMap};

/// SafeTensors model compiler.
///
/// Compiles SafeTensors model directories to hologram .holo format.
pub struct SafeTensorsCompiler {
    /// Whether to convert weights to F32 (default: true).
    pub convert_to_f32: bool,

    /// Weight storage strategy (auto-selected if None).
    pub weight_strategy: Option<hologram_ai_common::WeightStrategy>,
}

impl SafeTensorsCompiler {
    /// Create a new SafeTensors compiler with default settings.
    ///
    /// Weight strategy is auto-selected based on model size.
    pub fn new() -> Self {
        Self {
            convert_to_f32: true,
            weight_strategy: None,
        }
    }

    /// Create a compiler with explicit weight strategy.
    pub fn with_strategy(strategy: hologram_ai_common::WeightStrategy) -> Self {
        Self {
            convert_to_f32: true,
            weight_strategy: Some(strategy),
        }
    }

    /// Compile a SafeTensors model directory to hologram format.
    ///
    /// # Arguments
    /// * `path` - Path to the model directory containing config.json and *.safetensors files
    ///
    /// # Returns
    /// Tuple of (holo_bytes, weight_bytes)
    pub fn compile_dir(&self, path: &str) -> Result<(Vec<u8>, Vec<u8>)> {
        use std::path::Path;

        let dir = Path::new(path);
        if !dir.is_dir() {
            return Err(SafeTensorsError::NotADirectory(path.to_string()));
        }

        // Load config.json
        let config_path = dir.join("config.json");
        let hf_config = HfConfig::load(&config_path)?;
        let config = hf_config.to_transformer_config()?;

        // Load weights from SafeTensors files
        let mut parser = SafeTensorsParser::open_dir(dir)?;
        let weights = parser.load_all_weights(self.convert_to_f32)?;

        // Build the IR graph
        self.compile_with_config(&config, &weights)
    }

    /// Compile with explicit config and weights.
    pub fn compile_with_config(
        &self,
        config: &TransformerConfig,
        weights: &WeightMap,
    ) -> Result<(Vec<u8>, Vec<u8>)> {
        use hologram_ai_common::GenericTransformerBuilder;

        // Build the transformer graph
        let builder = GenericTransformerBuilder::new();
        let graph = builder
            .build(config, weights)
            .map_err(|e| SafeTensorsError::GraphBuildError(e.to_string()))?;

        // Compile to backend plan
        let backend_type = hologram::BackendType::Cpu;
        let (plan, header) = hologram::compiler::compile_ir_with_header(&graph, backend_type)
            .map_err(|e| SafeTensorsError::CompilationError(format!("{:?}", e)))?;

        // Select weight strategy (auto-select if not specified)
        let strategy = self.weight_strategy.unwrap_or_else(|| {
            // Auto-select based on constant_data size
            let weight_size = plan.constant_data.len();
            hologram_ai_common::WeightStrategy::auto_select(weight_size)
        });

        // Serialize the plan with selected weight strategy
        let (holo_bytes, weight_bytes) =
            hologram_ai_common::serialize_backend_plan_with_header(&plan, &header, strategy)
                .map_err(|e| SafeTensorsError::SerializationError(e.to_string()))?;

        Ok((holo_bytes, weight_bytes))
    }
}

impl Default for SafeTensorsCompiler {
    fn default() -> Self {
        Self::new()
    }
}
