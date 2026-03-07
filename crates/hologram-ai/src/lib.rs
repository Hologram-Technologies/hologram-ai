//! hologram-ai: AI model inference via the hologram O(1) LUT runtime.
//!
//! This is the top-level facade crate. It re-exports the public API and
//! depends on hologram with the `compiler` feature for `hologram::compile()`.

pub mod session;
pub mod stream;
pub mod validate;

// Flat re-exports.
pub use session::{
    CompileOptions, CompiledModel, InferenceSession, ModelCompiler, ModelMetadata, ModelSource,
};
pub use hologram_ai_common::{
    AiGraph, AiNode, AiOp, TensorInfo, DType, Shape, TensorId, NodeId,
};
pub use hologram_ai_quant::{QuantDescriptor, QuantScheme};
pub use hologram_ai_onnx::import_onnx;
pub use hologram_ai_gguf::{import_gguf, GgufImportOptions};
