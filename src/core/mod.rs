//! Core ONNX parsing, translation, and compilation.
//!
//! This module provides the fundamental infrastructure for ONNX model processing:
//! - Parsing ONNX protobuf files
//! - Symbolic shape handling
//! - IR translation with hologram-ir
//! - Weight extraction and management
//! - Graph serialization to .holo format
//! - Graph partitioning for large models

// Core modules
mod bundle;
mod error;
mod ir_to_graph;
mod parser;
mod partitioning;
mod serialization;
mod shapes;
mod translator;
mod weights;

// ONNX configuration (being replaced by config module, but keep for now)
mod config;

// Re-export public API
pub use bundle::{BundleBuilder, BundleHeader, HoloBundle, ModelIndexEntry, is_bundle};
pub use error::{OnnxError, Result};
pub use ir_to_graph::{ConversionOptions, DEFAULT_WEIGHT_THRESHOLD_ELEMENTS, ir_to_operation_graph_streaming_with_options};
pub use parser::{extract_opset_version, get_tensor_shape, parse_model, validate_model};
pub use partitioning::{GraphPartition, GraphPartitioner};
pub use serialization::{
    HoloHeader, HoloMetadata, PackedWeightEntry, PackedWeightKind, WeightEntry,
    HOLO_MAGIC, FORMAT_VERSION, HEADER_SIZE, FLAG_EXTERNAL_WEIGHTS,
};
pub use shapes::{Dim, Shape, SymbolicShape};
pub use translator::{apply_ir_decomposition, translate_graph_to_ir, lower_to_operation_graph, OperationGraph as TranslatorOperationGraph};
pub use weights::WeightData;

// Re-export config for backward compatibility
pub use config::OnnxConfig;

// Re-export hologram-ir types for convenience
pub use hologram_ir::{GraphBuilder, NodeIndex, OperationGraph};
