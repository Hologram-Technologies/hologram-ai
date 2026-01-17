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
}

impl SafeTensorsCompiler {
    /// Create a new SafeTensors compiler with default settings.
    pub fn new() -> Self {
        Self {
            convert_to_f32: true,
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
        let (mut plan, header) =
            hologram::compiler::compile_ir_with_header(&graph, backend_type)
                .map_err(|e| SafeTensorsError::CompilationError(format!("{:?}", e)))?;

        // Extract weights for external storage
        let weight_bytes = std::mem::take(&mut plan.constant_data);

        // Serialize the plan with layer header
        let holo_bytes = serialize_backend_plan_with_header(&plan, &header)?;

        Ok((holo_bytes, weight_bytes))
    }
}

impl Default for SafeTensorsCompiler {
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
        .map_err(|e| SafeTensorsError::SerializationError(e.to_string()))?;
    let header_bytes = rkyv::to_bytes::<rkyv::rancor::Error>(header)
        .map(|b| b.to_vec())
        .map_err(|e| SafeTensorsError::SerializationError(e.to_string()))?;
    let header_len = u32::try_from(header_bytes.len())
        .map_err(|_| SafeTensorsError::SerializationError("LayerHeader too large".to_string()))?;

    let mut holo_bytes = Vec::with_capacity(12 + header_bytes.len() + plan_bytes.len());
    holo_bytes.extend_from_slice(&hologram::compiler::HOLO_MAGIC);
    holo_bytes.extend_from_slice(&hologram::backend::plan::PLAN_FORMAT_VERSION.to_le_bytes());
    holo_bytes.extend_from_slice(&header_len.to_le_bytes());
    holo_bytes.extend_from_slice(&header_bytes);
    holo_bytes.extend_from_slice(&plan_bytes);

    Ok(holo_bytes)
}
