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
pub mod activation_fusion;
pub mod attention_detection;
mod bundle;
mod error;
mod ir_to_graph;
pub mod layer_detection;
pub mod layer_splitter;
pub mod op_hints;
mod parser;
mod partitioning;
pub mod sections;
mod serialization;
mod shapes;
mod translator;
mod weights;
mod weights_format;

// ONNX configuration
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
// V2 bundle format with sections support
pub use serialization::{
    BUNDLE_HEADER_SIZE_V2, BUNDLE_VERSION_V2, HoloBundleHeaderV2, SectionTableEntry,
    deserialize_sections_table, detect_bundle_version, serialize_sections_table,
};
pub use shapes::{Dim, Shape, SymbolicShape};
pub use translator::{
    OperationGraph as TranslatorOperationGraph, lower_to_operation_graph, translate_graph_to_ir,
    translate_graph_to_ir_with_groups,
};
pub use weights::{MmapWeightEntry, WeightData, WeightRef};
pub use weights_format::{
    PAGE_SIZE, WEIGHTS_MAGIC, WEIGHTS_VERSION, WeightDType, WeightIndexEntry, WeightsFileReader,
    WeightsFileWriter, WeightsHeader,
};

// Re-export config types
pub use config::{EmbeddedFileConfig, OnnxConfig, SectionType};

// Re-export operation hints
pub use op_hints::{
    ActivationType, add_composed_view_hint, add_parallel_hint, add_simd_hint,
    get_composed_view_table_ids, get_parallel_group_id, get_simd_table_id, has_composed_view_hint,
    has_parallel_hint, has_simd_hint,
};

// Re-export activation fusion
pub use activation_fusion::{ActivationChain, chain_name, detect_activation_chains};

// Re-export section traits and types
pub use sections::{
    EmbedError, EmbedResult, EmbeddableSection, FromEmbeddedSection, GenerationConfigSection,
    ModelConfigSection, PreprocessorConfigSection, RawFileSection, SentencePieceSection,
    SpecialTokensSection, TokenizerConfigSection, VocabularySection,
};

// Re-export hologram IR types for convenience
pub use hologram::ir::{GraphBuilder, NodeIndex, OperationGraph};
