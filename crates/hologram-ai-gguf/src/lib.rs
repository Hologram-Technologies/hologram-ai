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
}

impl GgufCompiler {
    /// Create a new GGUF compiler with default settings.
    pub fn new() -> Self {
        Self { dequantize: true }
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
        let mut plan = hologram::compiler::compile_ir(&graph, backend_type)
            .map_err(|e| GgufError::CompilationError(format!("{:?}", e)))?;

        // Extract weights for external storage
        let weight_bytes = std::mem::take(&mut plan.constant_data);

        // Serialize the plan
        let serializable = plan.to_serializable();
        let plan_bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&serializable)
            .map(|b| b.to_vec())
            .map_err(|e| GgufError::SerializationError(e.to_string()))?;

        // Add magic bytes
        let mut holo_bytes = Vec::with_capacity(4 + plan_bytes.len());
        holo_bytes.extend_from_slice(&hologram::compiler::HOLO_MAGIC);
        holo_bytes.extend_from_slice(&plan_bytes);

        Ok((holo_bytes, weight_bytes))
    }
}

impl Default for GgufCompiler {
    fn default() -> Self {
        Self::new()
    }
}
