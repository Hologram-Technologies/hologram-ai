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

// TEMPORARILY DISABLED: transformer module is disabled in hologram-ai-common
// use hologram_ai_common::{TransformerConfig, WeightMap};

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

    // TEMPORARILY DISABLED: transformer module is disabled in hologram-ai-common
    // The compile_file and compile_with_config methods require TransformerConfig
    // and GenericTransformerBuilder which are not currently available.
    //
    // /// Compile a GGUF file to hologram format.
    // pub fn compile_file(&self, path: &str) -> Result<(Vec<u8>, Vec<u8>)> { ... }
    //
    // /// Compile with explicit config and weights.
    // pub fn compile_with_config(&self, config: &TransformerConfig, weights: &WeightMap) -> Result<(Vec<u8>, Vec<u8>)> { ... }
}

impl Default for GgufCompiler {
    fn default() -> Self {
        Self::new()
    }
}
