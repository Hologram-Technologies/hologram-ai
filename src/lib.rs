//! # hologram-onnx
//!
//! Production ONNX runtime for hologram with full ISA optimization support.
//!
//! This crate provides a complete ONNX compilation pipeline that leverages hologram's
//! Instruction Set Architecture (ISA) for maximum performance:
//!
//! - **LOOP instructions**: O(1) space complexity (5,461x instruction reduction)
//! - **PhiCoordinate addressing**: Cache-resident boundary pool addressing for 5-10x speedup
//! - **ClassMap fusion**: O(1) element-wise operation composition using 96-byte lookup tables
//! - **SIMD vectorization**: Provided by hologram-backend
//! - **Im2col + GEMM decomposition**: Conv2D optimization via hologram's decomposition pass
//!
//! ## Architecture
//!
//! ```text
//! ONNX ModelProto
//!     ↓ [Parser]
//! ONNX Graph + Initializers
//!     ↓ [Translator]
//! IR Function (with symbolic shapes)
//!     ↓ [Decomposition Pass] ← Leverages hologram ISA optimizations
//! IR Function (Conv2D → Im2col+GEMM, etc.)
//!     ↓ [Lower to OperationGraph] ← Uses hologram ISA builder
//! OperationGraph + WeightData
//!     ↓ [Serialize]
//! model.holo + model.weights
//! ```
//!
//! ## Features
//!
//! - **Symbolic shapes**: Full support for variable batch sizes and sequence lengths
//! - **Memory efficient**: Weight streaming and graph partitioning for large models (3000+ nodes)
//! - **Config-driven**: TOML pipeline configs for multi-modal outputs (text→image, text→audio, text→text)
//! - **Feature-gated handlers**: `image-output`, `audio-output`, `text-output`
//!
//! ## Usage
//!
//! ### Basic Compilation
//!
//! ```no_run
//! use hologram_onnx::compile_onnx;
//! use std::fs;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Load ONNX model
//! let onnx_bytes = fs::read("model.onnx")?;
//!
//! // Compile to .holo format
//! let (holo_bytes, weight_bytes) = compile_onnx(&onnx_bytes)?;
//!
//! // Write output files
//! fs::write("model.holo", holo_bytes)?;
//! if !weight_bytes.is_empty() {
//!     fs::write("model.weights", weight_bytes)?;
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ### Advanced Compilation with Config
//!
//! ```no_run
//! use hologram_onnx::core::{OnnxCompiler, OnnxConfig};
//! use std::fs;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = OnnxConfig {
//!     weight_threshold: 4096,        // External storage threshold
//!     enable_partitioning: true,     // For large models
//!     partition_size: 500,            // Nodes per partition
//!     decompose_conv2d: true,         // Conv2D → Im2col+GEMM
//!     decompose_pooling: true,        // Pooling decomposition
//!     memory_budget: Some(8 * 1024),  // 8 GB limit
//! };
//!
//! let compiler = OnnxCompiler::with_config(config);
//! let onnx_bytes = fs::read("large_model.onnx")?;
//! let (holo_bytes, weight_bytes) = compiler.compile(&onnx_bytes)?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Execution with Config-Driven Output Handlers
//!
//! Execution is handled by the `hologram` CLI with config files:
//!
//! ```bash
//! # Compile with hologram-onnx
//! hologram-onnx compile model.onnx -o model
//!
//! # Run with hologram CLI using pipeline config
//! hologram run model.holo --config pipeline.toml --input prompt="cat"
//! ```
//!
//! ## CLI Workflow
//!
//! - `hologram-onnx compile`: ONNX → .holo compilation (this crate)
//! - `hologram run`: .holo execution (hologram CLI)
//! - Config-driven output handlers integrated into hologram runtime
//!
//! ## ISA Optimizations
//!
//! This crate fully leverages hologram's ISA for performance:
//!
//! ### LOOP Instructions
//! - Conv2D uses LOOP for Im2col transformation
//! - Broadcasting operations use LOOP
//! - Attention mechanisms use LOOP for O(1) space complexity
//! - RNN unrolling uses LOOP
//!
//! ### PhiCoordinate Addressing
//! - Conv2D output indexing uses PhiCoordinate (5-10x speedup)
//! - Pooling operations use PhiCoordinate
//! - Transposed convolutions use PhiCoordinate
//!
//! ### ClassMap Fusion
//! - Element-wise activation chains use ClassMap
//! - Normalization + activation fusions use ClassMap
//! - 96-byte lookup table generation
//!
//! ### SIMD Vectorization
//! - MatMul uses SIMD (via hologram-backend)
//! - Conv2D GEMM uses SIMD
//! - Element-wise operations use SIMD

#![deny(missing_docs)]
#![warn(clippy::all)]

// Translation pipeline module
mod translator;

// Re-export translator functions
pub use translator::{translate_graph_to_ir, apply_ir_decomposition};

// Re-export ONNX protobuf types
pub use hologram_onnx_spec as spec;

// Re-export core types and functionality
pub mod core {
    //! Core ONNX parsing, translation, and compilation.
    pub use hologram_onnx_core::*;
}

// Re-export operation translators
pub mod ops {
    //! ONNX operation implementations with symbolic shape inference.
    pub use hologram_onnx_ops::*;
}

// Re-export config and output handlers
pub mod config {
    //! Config-driven pipeline execution and output handlers.
    pub use hologram_onnx_config::*;
}

// Re-export common types at top level for convenience
pub use hologram_onnx_core::{OnnxCompiler, OnnxConfig, OnnxError};

/// Convenience function to compile ONNX model to .holo format.
///
/// This function provides a simple interface for basic ONNX compilation
/// with default settings. For advanced usage with custom configuration,
/// use [`OnnxCompiler::with_config`].
///
/// # Arguments
///
/// * `onnx_bytes` - Raw ONNX model bytes (protobuf format)
///
/// # Returns
///
/// A tuple of `(holo_bytes, weight_bytes)`:
/// - `holo_bytes`: Serialized OperationGraph for the .holo file
/// - `weight_bytes`: External weight data for the .weights file (may be empty)
///
/// # Errors
///
/// Returns [`OnnxError`] if:
/// - ONNX protobuf parsing fails
/// - Unsupported operations are encountered
/// - Shape inference fails
/// - Symbolic shape validation fails
/// - Memory budget is exceeded during compilation
///
/// # Examples
///
/// ```no_run
/// use hologram_onnx::compile_onnx;
/// use std::fs;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let onnx_bytes = fs::read("model.onnx")?;
/// let (holo_bytes, weight_bytes) = compile_onnx(&onnx_bytes)?;
///
/// fs::write("model.holo", holo_bytes)?;
/// if !weight_bytes.is_empty() {
///     fs::write("model.weights", weight_bytes)?;
/// }
/// # Ok(())
/// # }
/// ```
///
/// # Performance
///
/// This function leverages all hologram ISA optimizations:
/// - LOOP instructions for O(1) space complexity
/// - PhiCoordinate addressing for 5-10x speedup
/// - ClassMap fusion for element-wise operations
/// - SIMD vectorization via hologram-backend
pub fn compile_onnx(onnx_bytes: &[u8]) -> Result<(Vec<u8>, Vec<u8>), OnnxError> {
    let compiler = OnnxCompiler::new();
    compiler.compile(onnx_bytes)
}
