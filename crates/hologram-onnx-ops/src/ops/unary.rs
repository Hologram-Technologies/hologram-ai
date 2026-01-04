//! ONNX unary math operations.
//!
//! All unary operations in this module:
//! - Are **element-wise** (output shape = input shape)
//! - Support **symbolic shapes** (variable batch sizes, sequence lengths)
//! - Use **SIMD vectorization** via hologram-backend
//! - Can be **fused with ClassMap** for chained element-wise operations
//!
//! # Operators
//!
//! - **Sqrt**: Square root
//! - **Exp**: Exponential (e^x)
//! - **Log**: Natural logarithm
//! - **Neg**: Negation (-x)
//! - **Abs**: Absolute value
//! - **Reciprocal**: 1/x

use hologram_compiler::ir::{IRBuilder, NodeId};
use hologram_onnx_core::{OnnxError, Result, SymbolicShape};
use hologram_onnx_spec::AttributeProto;
use std::collections::HashMap;
use tracing::{debug, trace};

/// Translate ONNX Sqrt operation.
///
/// Sqrt: Y = sqrt(X) (element-wise square root)
///
/// # Performance
///
/// - **SIMD vectorization**: Uses optimized vectorized implementation
/// - **ClassMap fusion**: Can fuse with adjacent element-wise operations
/// - **Symbolic shapes**: Full support for dynamic dimensions
pub fn translate_sqrt(
    inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.len() != 1 {
        return Err(OnnxError::InvalidModel(format!(
            "Sqrt expects 1 input, got {}",
            inputs.len()
        )));
    }

    let input = inputs[0];
    debug!("Translating Sqrt operation");
    trace!("Sqrt input: {:?}", input);

    let result = builder.sqrt(input);

    trace!("Created Sqrt node: {:?}", result);
    Ok(result)
}

/// Translate ONNX Exp operation.
///
/// Exp: Y = e^X (element-wise exponential)
///
/// # Performance
///
/// - **SIMD vectorization**: Uses vectorized exp implementation
/// - **ClassMap fusion**: Can fuse with adjacent element-wise operations
/// - Common in softmax, sigmoid, and other activations
pub fn translate_exp(
    inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.len() != 1 {
        return Err(OnnxError::InvalidModel(format!(
            "Exp expects 1 input, got {}",
            inputs.len()
        )));
    }

    let input = inputs[0];
    debug!("Translating Exp operation");
    trace!("Exp input: {:?}", input);

    let result = builder.exp(input);

    trace!("Created Exp node: {:?}", result);
    Ok(result)
}

/// Translate ONNX Log operation.
///
/// Log: Y = ln(X) (element-wise natural logarithm)
///
/// # Performance
///
/// - **SIMD vectorization**: Uses vectorized log implementation
/// - Used in log-softmax, cross-entropy loss, etc.
pub fn translate_log(
    inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.len() != 1 {
        return Err(OnnxError::InvalidModel(format!(
            "Log expects 1 input, got {}",
            inputs.len()
        )));
    }

    let input = inputs[0];
    debug!("Translating Log operation");
    trace!("Log input: {:?}", input);

    let result = builder.log(input);

    trace!("Created Log node: {:?}", result);
    Ok(result)
}

/// Translate ONNX Neg operation.
///
/// Neg: Y = -X (element-wise negation)
///
/// # Performance
///
/// - **SIMD vectorization**: Uses vectorized negation
/// - **ClassMap fusion**: Trivial to fuse with other operations
pub fn translate_neg(
    inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.len() != 1 {
        return Err(OnnxError::InvalidModel(format!(
            "Neg expects 1 input, got {}",
            inputs.len()
        )));
    }

    let input = inputs[0];
    debug!("Translating Neg operation");
    trace!("Neg input: {:?}", input);

    let result = builder.neg(input);

    trace!("Created Neg node: {:?}", result);
    Ok(result)
}

/// Translate ONNX Abs operation.
///
/// Abs: Y = |X| (element-wise absolute value)
///
/// # Performance
///
/// - **SIMD vectorization**: Uses vectorized abs
/// - Used in L1 loss, regularization, etc.
pub fn translate_abs(
    inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.len() != 1 {
        return Err(OnnxError::InvalidModel(format!(
            "Abs expects 1 input, got {}",
            inputs.len()
        )));
    }

    let input = inputs[0];
    debug!("Translating Abs operation");
    trace!("Abs input: {:?}", input);

    let result = builder.abs(input);

    trace!("Created Abs node: {:?}", result);
    Ok(result)
}

/// Translate ONNX Reciprocal operation.
///
/// Reciprocal: Y = 1/X (element-wise reciprocal)
///
/// # Performance
///
/// - **SIMD vectorization**: Uses vectorized divide
/// - Alternative to division in some cases
pub fn translate_reciprocal(
    inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.len() != 1 {
        return Err(OnnxError::InvalidModel(format!(
            "Reciprocal expects 1 input, got {}",
            inputs.len()
        )));
    }

    let input = inputs[0];
    debug!("Translating Reciprocal operation");
    trace!("Reciprocal input: {:?}", input);

    // Reciprocal(x) = 1 / x
    let one = builder.add_f32(1.0);
    let result = builder.div(one, input);

    trace!("Created Reciprocal node: {:?}", result);
    Ok(result)
}

/// Translate ONNX Sin operation.
///
/// Sin: Y = sin(X) (element-wise sine)
///
/// # Performance
///
/// - **SIMD vectorization**: Uses vectorized sin implementation
/// - Common in positional embeddings for transformers
pub fn translate_sin(
    inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.len() != 1 {
        return Err(OnnxError::InvalidModel(format!(
            "Sin expects 1 input, got {}",
            inputs.len()
        )));
    }

    let input = inputs[0];
    debug!("Translating Sin operation");
    trace!("Sin input: {:?}", input);

    let result = builder.sin(input);

    trace!("Created Sin node: {:?}", result);
    Ok(result)
}

/// Translate ONNX Cos operation.
///
/// Cos: Y = cos(X) (element-wise cosine)
///
/// # Performance
///
/// - **SIMD vectorization**: Uses vectorized cos implementation
/// - Common in positional embeddings for transformers
pub fn translate_cos(
    inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.len() != 1 {
        return Err(OnnxError::InvalidModel(format!(
            "Cos expects 1 input, got {}",
            inputs.len()
        )));
    }

    let input = inputs[0];
    debug!("Translating Cos operation");
    trace!("Cos input: {:?}", input);

    let result = builder.cos(input);

    trace!("Created Cos node: {:?}", result);
    Ok(result)
}

/// Translate ONNX Erf operation.
///
/// Erf: Y = erf(X) (element-wise error function)
///
/// The error function is defined as:
/// erf(x) = (2/√π) ∫₀ˣ e^(-t²) dt
///
/// # Performance
///
/// - **SIMD vectorization**: Uses optimized vectorized implementation
/// - Common in GELU activation and transformer models
pub fn translate_erf(
    inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.len() != 1 {
        return Err(OnnxError::InvalidModel(format!(
            "Erf expects 1 input, got {}",
            inputs.len()
        )));
    }

    let input = inputs[0];
    debug!("Translating Erf operation");
    trace!("Erf input: {:?}", input);

    let result = builder.erf(input);

    trace!("Created Erf node: {:?}", result);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::f32_tensor;
    use hologram_compiler::ir::IRBuilder;

    fn make_builder() -> IRBuilder {
        IRBuilder::new("test")
    }

    // ========================================================================
    // Sqrt Tests
    // ========================================================================

    #[test]
    fn test_translate_sqrt() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[2, 3, 4]));

        let result = translate_sqrt(&[input], &[], &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_sqrt_symbolic() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[]));

        let result = translate_sqrt(&[input], &[], &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_sqrt_wrong_inputs() {
        let mut builder = make_builder();
        let result = translate_sqrt(&[], &[], &HashMap::new(), &mut builder);
        assert!(result.is_err());
    }

    // ========================================================================
    // Exp Tests
    // ========================================================================

    #[test]
    fn test_translate_exp() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[2, 3, 4]));

        let result = translate_exp(&[input], &[], &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_exp_symbolic() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[]));

        let result = translate_exp(&[input], &[], &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    // ========================================================================
    // Log Tests
    // ========================================================================

    #[test]
    fn test_translate_log() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[2, 3, 4]));

        let result = translate_log(&[input], &[], &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_log_symbolic() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[]));

        let result = translate_log(&[input], &[], &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    // ========================================================================
    // Neg Tests
    // ========================================================================

    #[test]
    fn test_translate_neg() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[2, 3, 4]));

        let result = translate_neg(&[input], &[], &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_neg_symbolic() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[]));

        let result = translate_neg(&[input], &[], &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    // ========================================================================
    // Abs Tests
    // ========================================================================

    #[test]
    fn test_translate_abs() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[2, 3, 4]));

        let result = translate_abs(&[input], &[], &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_abs_symbolic() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[]));

        let result = translate_abs(&[input], &[], &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    // ========================================================================
    // Reciprocal Tests
    // ========================================================================

    #[test]
    fn test_translate_reciprocal() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[2, 3, 4]));

        let result = translate_reciprocal(&[input], &[], &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_reciprocal_symbolic() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[]));

        let result = translate_reciprocal(&[input], &[], &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }
}
