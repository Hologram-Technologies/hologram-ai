//! ONNX operation translators.
//!
//! This module provides trait-based translators for converting ONNX operations
//! to hologram IR. Each operation type has its own translator struct that
//! implements the `OnnxTranslator` trait.
//!
//! # Architecture
//!
//! The translation system is organized as follows:
//!
//! - **traits.rs**: Core traits (`OnnxTranslator`, `OnnxAttributes`, `InputRequirement`)
//! - **error.rs**: Structured error types for translation failures
//! - **registry.rs**: Central registry that maps op types to translators
//! - **Category modules**: Individual translator implementations grouped by operation type
//!
//! # Usage
//!
//! The primary entry point is the `TranslatorRegistry`:
//!
//! ```ignore
//! use crate::translators::TranslatorRegistry;
//!
//! let registry = TranslatorRegistry::new();
//! let outputs = registry.translate(&node, &inputs, &mut builder)?;
//! ```
//!
//! # Adding New Operations
//!
//! To add a new ONNX operation translator:
//!
//! 1. Create a new file in the appropriate category directory
//! 2. Implement the `OnnxTranslator` trait for a new struct
//! 3. Register the translator in `registry.rs`
//! 4. Add tests for valid/invalid inputs and constant folding

mod error;
mod registry;
mod traits;

// Translator modules by category
pub mod activation;
pub mod advanced;
pub mod binary;
pub mod constant;
pub mod conv;
pub mod indexing;
pub mod logical;
pub mod matmul;
pub mod norm;
pub mod pad;
pub mod pool;
pub mod reduce;
pub mod resize;
pub mod shape;
pub mod unary;

// Re-export core types
pub use error::TranslationError;
pub use registry::TranslatorRegistry;
pub use traits::{InputRequirement, OnnxAttributes, OnnxTranslator};
