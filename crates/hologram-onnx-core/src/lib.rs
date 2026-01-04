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

pub mod bundle;
mod config;
mod error;
mod ir_to_graph;
mod parser;
mod partitioning;
pub mod serialization;
mod shapes;
mod translator;
mod weights;

// Re-export public API
pub use bundle::{BundleBuilder, BundleHeader, HoloBundle, ModelIndexEntry, is_bundle};
pub use config::OnnxConfig;
pub use error::{OnnxError, Result};
pub use ir_to_graph::{ir_to_operation_graph, ir_to_operation_graph_streaming, ir_to_operation_graph_streaming_with_options, ConversionOptions, DEFAULT_WEIGHT_THRESHOLD_ELEMENTS};
pub use parser::{extract_opset_version, get_tensor_shape, parse_model, validate_model};
pub use partitioning::{GraphPartition, GraphPartitioner};
pub use shapes::{Dim, Shape, SymbolicShape};
pub use translator::{OperationGraph, lower_to_operation_graph};
pub use weights::WeightData;

// Legacy serialization types (kept for backwards compatibility with info/validate commands)
// The new .holo format uses OperationGraph JSON serialization from hologram-compiler
pub use serialization::{
    HoloHeader, HoloMetadata, PackedWeightEntry, PackedWeightKind, WeightEntry,
    HOLO_MAGIC, FORMAT_VERSION, HEADER_SIZE, FLAG_EXTERNAL_WEIGHTS,
};

// Note: OnnxCompiler has been moved to the top-level hologram-onnx crate
// because it requires both core (parsing) and ops (translators).
// Due to the dependency structure (ops → core), keeping it here would
// create a cyclic dependency.
//
// For compilation, use:
// ```
// use hologram_onnx::{compile_onnx, OnnxCompiler};
// ```
