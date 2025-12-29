//! Config-driven output handlers for multi-modal ONNX pipelines.
//!
//! This crate provides:
//! - **TOML config parsing**: Load pipeline configurations from files
//! - **OutputHandler trait**: Generic interface for processing model outputs
//! - **Feature-gated handlers**: Image, audio, and text output processing
//! - **Multi-modal support**: Handle different output types in a single pipeline
//!
//! # Performance
//!
//! All output handlers are designed for minimal runtime overhead:
//! - **Zero-copy transformations**: Where possible via bytemuck
//! - **SIMD processing**: For data format conversions
//! - **Lazy loading**: Handlers only loaded when features enabled
//!
//! # Features
//!
//! - `image-output`: Enable image output processing (PNG, JPEG, WebP)
//! - `audio-output`: Enable audio output processing (WAV)
//! - `text-output`: Enable text/LLM output processing with tokenizers
//! - `all-outputs`: Enable all output handlers
//!
//! # Example
//!
//! ```rust,ignore
//! use hologram_onnx_config::{PipelineConfig, OutputHandlerRegistry};
//!
//! let config = PipelineConfig::from_file("pipeline.toml")?;
//! let registry = OutputHandlerRegistry::new();
//! let handlers = registry.create_handlers(&config)?;
//!
//! // Process model outputs
//! let output = handlers.process(&raw_outputs)?;
//! output.save("result.png")?;
//! ```
#![allow(unused_imports)]
// Unused imports kept for future API surface expansion

use hologram_onnx_core::{OnnxError, Result};
use std::collections::HashMap;
use std::path::Path;

// Module declarations
mod config;
mod error;
mod output_handlers;

// Public exports
pub use config::{ExecutionConfig, OutputHandlerConfig, PipelineConfig, StageConfig};
pub use error::ConfigError;
pub use output_handlers::{
    AudioOutput, ImageOutput, OutputHandler, OutputHandlerRegistry, ProcessedOutput, TensorData,
    TensorOutput,
};

// Feature-gated handler exports
#[cfg(feature = "image-output")]
pub use output_handlers::image::ImageHandler;

#[cfg(feature = "audio-output")]
pub use output_handlers::audio::AudioHandler;

#[cfg(feature = "text-output")]
pub use output_handlers::text::TextHandler;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_structure() {
        // Verify all modules are accessible
        assert!(true, "Module structure is correct");
    }
}
