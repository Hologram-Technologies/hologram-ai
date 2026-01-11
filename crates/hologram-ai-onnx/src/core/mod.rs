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
mod weights_format;

// ONNX configuration (being replaced by config module, but keep for now)
mod config;

// Re-export public API
pub use bundle::{BundleBuilder, BundleHeader, HoloBundle, ModelIndexEntry, is_bundle};
// Unified bundle format (HOLB) - single model with embedded weights
pub use bundle::{UnifiedBundleReader, UnifiedBundleWriter, read_unified_bundle_file};
// Pipeline bundle format (HOLM) - multi-model with embedded weights
pub use bundle::{PipelineBundleReader, PipelineBundleWriter};
pub use error::{OnnxError, Result};
pub use ir_to_graph::{
    ConversionOptions, DEFAULT_WEIGHT_THRESHOLD_ELEMENTS,
    ir_to_operation_graph_streaming_with_options,
};
pub use parser::{extract_opset_version, get_tensor_shape, parse_model, validate_model};
pub use partitioning::{GraphPartition, GraphPartitioner};
pub use serialization::{
    FLAG_EXTERNAL_WEIGHTS, FORMAT_VERSION, HEADER_SIZE, HOLO_MAGIC, HoloHeader, HoloMetadata,
    PackedWeightEntry, PackedWeightKind, WeightEntry,
};
// Unified bundle format header and detection
pub use serialization::{
    BUNDLE_HEADER_SIZE, BUNDLE_VERSION, HOLB_MAGIC, HOLP_MAGIC, HoloBundleHeader, HoloFormat,
};
// Pipeline bundle format (HOLM) - multi-model with embedded weights
pub use serialization::{
    HOLM_MAGIC, HoloPipelineHeader, PIPELINE_HEADER_SIZE, PIPELINE_VERSION, PipelineModelEntry,
};
pub use shapes::{Dim, Shape, SymbolicShape};
pub use translator::{
    OperationGraph as TranslatorOperationGraph, lower_to_operation_graph, translate_graph_to_ir,
};
pub use weights::{MmapWeightEntry, WeightData, WeightRef};
pub use weights_format::{
    PAGE_SIZE, WEIGHTS_MAGIC, WEIGHTS_VERSION, WeightDType, WeightIndexEntry, WeightsFileReader,
    WeightsFileWriter, WeightsHeader,
};

// Re-export config for backward compatibility
pub use config::OnnxConfig;

// Re-export hologram IR types for convenience
pub use hologram::ir::{GraphBuilder, NodeIndex, OperationGraph};
