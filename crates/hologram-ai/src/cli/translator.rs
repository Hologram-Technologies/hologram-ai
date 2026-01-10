//! Full ONNX to hologram IR translation pipeline.
//!
//! This module provides the high-level translation API that combines parsing,
//! operator translation, and IR decomposition. The actual implementation has
//! been moved to `hologram_onnx_core::translator` and these functions serve
//! as compatibility wrappers for the CLI.

#[cfg(feature = "onnx")]
use hologram_ai_onnx::core::OnnxError;
#[cfg(feature = "onnx")]
use hologram_ai_onnx::proto::GraphProto;
#[cfg(feature = "onnx")]
use hologram::ir::OperationGraph as IRFunction;

#[cfg(feature = "onnx")]
type Result<T> = std::result::Result<T, OnnxError>;

/// Translate ONNX graph to hologram IR.
///
/// This is a compatibility wrapper. The actual translation is implemented
/// in `hologram_onnx_core::translator::translate_graph_to_ir`.
///
/// # Arguments
/// * `graph` - ONNX graph proto to translate
/// * `_opset_version` - ONNX opset version (currently unused)
///
/// # Returns
/// Translated hologram IR function, or an error if translation is not supported.
///
/// The opset version parameter is preserved for API compatibility but is not
/// currently used in translation.
#[allow(dead_code)] // Used for API compatibility
pub fn translate_graph_to_ir(graph: &GraphProto, _opset_version: i64) -> Result<IRFunction> {
    hologram_ai_onnx::translate_graph_to_ir(graph)
}

/// Translate ONNX graph to hologram IR with external data support.
///
/// This variant supports ONNX models with external weight data stored in
/// separate files (typical for large models).
///
/// # Arguments
/// * `graph` - ONNX graph proto to translate
/// * `_opset_version` - ONNX opset version (currently unused)
/// * `_model_path` - Path to the ONNX model file (for resolving external data)
///
/// # Returns
/// Translated hologram IR function, or an error if translation is not supported.
///
/// External data support is not yet implemented. This function currently
/// behaves the same as `translate_graph_to_ir`.
#[allow(dead_code)]
pub fn translate_graph_to_ir_with_path(
    graph: &GraphProto,
    _opset_version: i64,
    _model_path: Option<&std::path::Path>,
) -> Result<IRFunction> {
    // External data support is not yet implemented
    hologram_ai_onnx::translate_graph_to_ir(graph)
}

/// Apply IR-level decompositions.
///
/// This applies high-level IR transformations such as decomposing complex
/// operations into simpler primitives.
///
/// # Arguments
/// * `_ir_func` - IR function to decompose (modified in place)
///
/// # Returns
/// Result indicating success or failure.
///
/// # Note
/// IR decomposition is handled by hologram-ir in the new architecture.
/// This function is a no-op compatibility wrapper.
#[allow(dead_code)] // Used for API compatibility
pub fn apply_ir_decomposition(_ir_func: &mut IRFunction) -> Result<()> {
    // Decomposition is now handled by hologram-ir passes.
    // This is a compatibility wrapper that does nothing.
    Ok(())
}
