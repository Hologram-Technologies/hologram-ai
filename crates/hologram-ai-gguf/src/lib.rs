//! GGUF format support for hologram-ai.
//!
//! This crate provides parsing and compilation of GGUF model files to hologram IR.
//! GGUF is the weight storage format used by llama.cpp and related projects.
//!
//! ## Features
//!
//! - Parse GGUF file headers and metadata
//! - Extract transformer configuration from metadata
//! - Dequantize weights (Q4_K, Q8_0, F16, etc.) to F32
//! - Build IR graphs using GenericTransformerBuilder
//!
//! ## Example
//!
//! ```ignore
//! use hologram_ai_gguf::GgufCompiler;
//!
//! let compiler = GgufCompiler::new();
//! let (holo_bytes, weight_bytes) = compiler.compile_file("model.gguf")?;
//! ```

#![deny(missing_docs)]
#![warn(clippy::all)]

pub mod dequant;
pub mod error;
pub mod metadata;
pub mod parser;

pub use error::{GgufError, Result};
pub use metadata::GgufMetadata;
pub use parser::GgufParser;

use hologram_ai_common::{TransformerConfig, WeightMap};

/// GGUF model compiler.
///
/// Compiles GGUF models to hologram .holo format.
pub struct GgufCompiler {
    /// Whether to dequantize weights to F32.
    pub dequantize: bool,

    /// Weight storage strategy (auto-selected if None).
    pub weight_strategy: Option<hologram_ai_common::WeightStrategy>,
}

impl GgufCompiler {
    /// Create a new GGUF compiler with default settings.
    ///
    /// Weight strategy is auto-selected based on model size.
    pub fn new() -> Self {
        Self {
            dequantize: true,
            weight_strategy: None,
        }
    }

    /// Create a compiler with explicit weight strategy.
    pub fn with_strategy(strategy: hologram_ai_common::WeightStrategy) -> Self {
        Self {
            dequantize: true,
            weight_strategy: Some(strategy),
        }
    }

    /// Compile a GGUF file to hologram format.
    ///
    /// # Arguments
    /// * `path` - Path to the GGUF file
    ///
    /// # Returns
    /// Tuple of (holo_bytes, weight_bytes)
    pub fn compile_file(&self, path: &str) -> Result<(Vec<u8>, Vec<u8>)> {
        // Parse the GGUF file
        let mut parser = GgufParser::open(path)?;
        let metadata = parser.metadata()?;
        let weights = parser.load_weights(self.dequantize)?;

        // Convert metadata to TransformerConfig
        let config = metadata.to_transformer_config()?;

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
            .map_err(|e| GgufError::GraphBuildError(e.to_string()))?;

        // Compile to backend plan
        let backend_type = hologram::BackendType::Cpu;
        let (plan, header) = hologram::compiler::compile_ir_with_header(&graph, backend_type)
            .map_err(|e| GgufError::CompilationError(format!("{:?}", e)))?;

        // Select weight strategy (auto-select if not specified)
        let strategy = self.weight_strategy.unwrap_or_else(|| {
            // Auto-select based on constant_data size
            let weight_size = plan.constant_data.len();
            hologram_ai_common::WeightStrategy::auto_select(weight_size)
        });

        // Serialize the plan with selected weight strategy
        let (holo_bytes, weight_bytes) =
            hologram_ai_common::serialize_backend_plan_with_header(&plan, &header, strategy)
                .map_err(|e| GgufError::SerializationError(e.to_string()))?;

        Ok((holo_bytes, weight_bytes))
    }
}

impl Default for GgufCompiler {
    fn default() -> Self {
        Self::new()
    }
}
