//! Core ONNX parsing, translation, and compilation for hologram.
//!
//! This crate provides the fundamental infrastructure for compiling ONNX models
//! to hologram's `.holo` format with full ISA optimization support.
//!
//! # Architecture
//!
//! The compilation pipeline consists of several stages:
//!
//! 1. **Parsing**: ONNX protobuf → validated ModelProto
//! 2. **Translation**: ONNX GraphProto → IR Function (with symbolic shapes)
//! 3. **Decomposition**: High-level ops → ISA-optimized primitives
//! 4. **Lowering**: IR Function → OperationGraph
//! 5. **Serialization**: OperationGraph + WeightData → .holo + .weights files
//!
//! # ISA Integration
//!
//! This crate leverages hologram's ISA for maximum performance:
//!
//! - **LOOP instructions**: O(1) space complexity for nested loops
//! - **PhiCoordinate addressing**: 5-10x speedup for boundary pool access
//! - **ClassMap fusion**: O(1) element-wise operation composition
//! - **SIMD vectorization**: Provided by hologram-backend
//!
//! # Symbolic Shapes
//!
//! All tensor types support symbolic dimensions for variable batch sizes
//! and sequence lengths. Shape inference propagates symbolic dimensions
//! throughout the compilation pipeline.
//!
//! # Memory Efficiency
//!
//! - **Weight streaming**: Weights are extracted without loading entire model
//! - **Graph partitioning**: Large models (>500 nodes) are split into chunks
//! - **Threshold-based storage**: Weights >4KB stored in external .weights file

#![deny(missing_docs)]
#![warn(clippy::all)]

mod config;
mod error;
mod parser;
mod partitioning;
mod shapes;
mod translator;
mod weights;

// Re-export public API
pub use config::OnnxConfig;
pub use error::{OnnxError, Result};
pub use parser::{extract_opset_version, get_tensor_shape, parse_model, validate_model};
pub use partitioning::{GraphPartition, GraphPartitioner};
pub use shapes::{Dim, Shape, SymbolicShape};
pub use translator::{lower_to_operation_graph, OperationGraph};
pub use weights::WeightData;

// Note: OnnxCompiler has been moved to the top-level hologram-onnx crate
// because it requires both core (parsing) and ops (translators).
// Due to the dependency structure (ops → core), keeping it here would
// create a cyclic dependency.
//
// For compilation, use:
// ```
// use hologram_onnx::{compile_onnx, OnnxCompiler};
// ```
