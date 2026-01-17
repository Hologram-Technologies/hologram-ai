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
        let (mut plan, header) = hologram::compiler::compile_ir_with_header(&graph, backend_type)
            .map_err(|e| GgufError::CompilationError(format!("{:?}", e)))?;

        // Extract weights for external storage
        let weight_bytes = std::mem::take(&mut plan.constant_data);

        // Serialize the plan with layer header
        let holo_bytes = serialize_backend_plan_with_header(&plan, &header)?;

        Ok((holo_bytes, weight_bytes))
    }
}

impl Default for GgufCompiler {
    fn default() -> Self {
        Self::new()
    }
}

fn serialize_backend_plan_with_header(
    plan: &hologram::backend::BackendPlan,
    header: &hologram::compiler::format::LayerHeaderData,
) -> Result<Vec<u8>> {
    let serializable = plan.to_serializable();
    let plan_bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&serializable)
        .map(|b| b.to_vec())
        .map_err(|e| GgufError::SerializationError(e.to_string()))?;
    let header_bytes = rkyv::to_bytes::<rkyv::rancor::Error>(header)
        .map(|b| b.to_vec())
        .map_err(|e| GgufError::SerializationError(e.to_string()))?;
    let header_len = u32::try_from(header_bytes.len())
        .map_err(|_| GgufError::SerializationError("LayerHeader too large".to_string()))?;

    let mut holo_bytes = Vec::with_capacity(12 + header_bytes.len() + plan_bytes.len());
    holo_bytes.extend_from_slice(&hologram::compiler::HOLO_MAGIC);
    holo_bytes.extend_from_slice(&hologram::backend::plan::PLAN_FORMAT_VERSION.to_le_bytes());
    holo_bytes.extend_from_slice(&header_len.to_le_bytes());
    holo_bytes.extend_from_slice(&header_bytes);
    holo_bytes.extend_from_slice(&plan_bytes);

    Ok(holo_bytes)
}
