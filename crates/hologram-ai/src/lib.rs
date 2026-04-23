//! hologram-ai: AI model compiler for the hologram O(1) LUT runtime.
//!
//! This is the top-level facade crate. It re-exports the public API and
//! depends on hologram with the `compiler` feature for `hologram::compile()`.
//!
//! hologram-ai is a **compiler**, not a runtime. It produces `.holo` archives
//! that are executed via the standard hologram APIs (see ADR-0016).

pub mod commands;
pub mod compiler;
pub mod download;
pub mod runner;
pub mod speculative;
pub mod validate;

// Flat re-exports.
pub use compiler::{CompileStats, CompiledModel, DebugMap, HoloArchive, ModelCompiler, ModelMetadata, ModelSource};
pub use runner::{read_shape_context_from_archive, rebuild_archive_with_section, run_with_shape_context, HoloRunner};
pub use hologram_ai_common::{AiGraph, AiNode, AiOp, DType, NodeId, Shape, TensorId, TensorInfo};
pub use hologram_ai_onnx::{import_onnx, import_onnx_path, OnnxImportOptions};
pub use hologram_ai_quant::{QuantDescriptor, QuantScheme};
