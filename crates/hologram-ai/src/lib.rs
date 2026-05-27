//! hologram-ai: AI model compiler for the hologram O(1) LUT runtime.
//!
//! This is the top-level facade crate. It re-exports the public API and
//! depends on hologram with the `compiler` feature for `hologram::compile()`.
//!
//! hologram-ai is a **compiler**, not a runtime. It produces `.holo` archives
//! that are executed via the standard hologram APIs (see ADR-0016).

pub mod address;
pub mod commands;
pub mod compiler;
/// Model downloader — native only (HTTP + tokio + local toolchain). Absent from
/// the wasm build (`--no-default-features`).
#[cfg(feature = "native")]
pub mod download;
pub mod runner;
pub mod validate;

// Flat re-exports.
pub use address::{
    component_kappa, compose_model, compose_models, model_kappa, model_kappa_label, ModelFormat,
    ModelOutcome,
};
pub use compiler::{
    CompileStats, CompiledModel, DebugMap, HoloArchive, ModelCompiler, ModelMetadata, ModelSource,
};
pub use hologram_ai_common::{AiGraph, AiNode, AiOp, DType, NodeId, Shape, TensorId, TensorInfo};
pub use hologram_ai_onnx::{import_onnx, import_onnx_path, OnnxImportOptions};
pub use hologram_ai_quant::{QuantDescriptor, QuantScheme};
pub use hologram_archive::ContentLabel;
pub use runner::HoloRunner;
