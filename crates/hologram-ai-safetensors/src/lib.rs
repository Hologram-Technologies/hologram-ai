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

// TEMPORARILY DISABLED: transformer module is disabled in hologram-ai-common
// use hologram_ai_common::{TransformerConfig, WeightMap};

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

    // TEMPORARILY DISABLED: transformer module is disabled in hologram-ai-common
    // The compile_dir and compile_with_config methods require TransformerConfig
    // and GenericTransformerBuilder which are not currently available.
    //
    // /// Compile a SafeTensors model directory to hologram format.
    // pub fn compile_dir(&self, path: &str) -> Result<(Vec<u8>, Vec<u8>)> { ... }
    //
    // /// Compile with explicit config and weights.
    // pub fn compile_with_config(&self, config: &TransformerConfig, weights: &WeightMap) -> Result<(Vec<u8>, Vec<u8>)> { ... }
}

impl Default for SafeTensorsCompiler {
    fn default() -> Self {
        Self::new()
    }
}
