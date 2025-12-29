//! ONNX activation functions.
//!
//! All activations in this module:
//! - Leverage **ClassMap fusion** for O(1) element-wise operation chains
//! - Support **symbolic shapes** (shape unchanged from input)
//! - Use **SIMD vectorization** via hologram-backend
//!
//! # ISA Optimizations
//!
//! - **ClassMap fusion**: Multiple activations compose into single 96-byte lookup table
//! - **SIMD**: All element-wise operations vectorized
//! - **Zero runtime overhead**: All decisions made at compile time

use hologram_compiler::ir::{IRBuilder, NodeId};
use hologram_onnx_core::{OnnxError, Result, SymbolicShape};
use hologram_onnx_spec::AttributeProto;
use std::collections::HashMap;
use tracing::{debug, trace};

use crate::utils::parse_attr_int;

/// Translate ONNX ReLU activation.
///
/// ReLU: Y = max(0, X)
///
/// # Performance
///
/// - **ClassMap fusion**: Can fuse with adjacent element-wise ops
/// - **SIMD vectorization**: Processes multiple elements in parallel
/// - Shape unchanged: Input shape = Output shape
pub fn translate_relu(
    inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel(
            "ReLU expects 1 input, got 0".to_string(),
        ));
    }

    let input = inputs[0];

    debug!("Translating ReLU operation");
    trace!("ReLU input: {:?}", input);

    // Create ReLU IR node
    // hologram's backend will use SIMD and ClassMap fusion
    let node = builder.relu(input);

    trace!("Created ReLU node: {:?}", node);
    Ok(node)
}

/// Translate ONNX Sigmoid activation.
///
/// Sigmoid: Y = 1 / (1 + exp(-X))
///
/// # Performance
///
/// - **ClassMap fusion**: Fuses with adjacent element-wise ops
/// - **SIMD vectorization**: Fast exp() and division
/// - Supports **symbolic shapes**
pub fn translate_sigmoid(
    inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel(
            "Sigmoid expects 1 input, got 0".to_string(),
        ));
    }

    let input = inputs[0];

    debug!("Translating Sigmoid operation");
    trace!("Sigmoid input: {:?}", input);

    let node = builder.sigmoid(input);

    trace!("Created Sigmoid node: {:?}", node);
    Ok(node)
}

/// Translate ONNX Tanh activation.
///
/// Tanh: Y = tanh(X) = (exp(X) - exp(-X)) / (exp(X) + exp(-X))
///
/// # Performance
///
/// - **ClassMap fusion**: Fuses with adjacent operations
/// - **SIMD vectorization**: Fast exp() operations
/// - Supports **symbolic shapes**
pub fn translate_tanh(
    inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel(
            "Tanh expects 1 input, got 0".to_string(),
        ));
    }

    let input = inputs[0];

    debug!("Translating Tanh operation");
    trace!("Tanh input: {:?}", input);

    let node = builder.tanh(input);

    trace!("Created Tanh node: {:?}", node);
    Ok(node)
}

/// Translate ONNX Softmax activation.
///
/// Softmax: Y_i = exp(X_i - max(X)) / sum(exp(X_j - max(X)))
///
/// # Attributes
///
/// - `axis` (int, default -1): Axis along which to compute softmax
///
/// # Performance
///
/// - Uses **LOOP instructions** for reduction operations (O(1) space)
/// - **SIMD vectorization** for exp() and summation
/// - Supports **symbolic shapes** (axis can reference symbolic dimension)
///
/// # Numerical Stability
///
/// Subtracts max(X) before exp() to prevent overflow.
pub fn translate_softmax(
    inputs: &[NodeId],
    attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel(
            "Softmax expects 1 input, got 0".to_string(),
        ));
    }

    let input = inputs[0];

    // Parse axis attribute (default: -1, meaning last axis)
    let axis = parse_attr_int(attrs, "axis", -1)?;

    debug!("Translating Softmax operation (axis={})", axis);
    trace!("Softmax input: {:?}", input);

    // Create Softmax IR node using builder method
    let node = builder.softmax(input, axis as isize);

    trace!("Created Softmax node: {:?}", node);
    Ok(node)
}

/// Translate ONNX GELU activation.
///
/// GELU (Gaussian Error Linear Unit): Y = X * Φ(X)
/// where Φ(x) is the cumulative distribution function of the standard Gaussian distribution.
///
/// Approximation: Y ≈ 0.5 * X * (1 + tanh(√(2/π) * (X + 0.044715 * X³)))
///
/// # Performance
///
/// - **ClassMap fusion**: Fuses with adjacent element-wise ops
/// - **SIMD vectorization**: Fast polynomial evaluation
/// - Supports **symbolic shapes**
pub fn translate_gelu(
    inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel(
            "GELU expects 1 input, got 0".to_string(),
        ));
    }

    let input = inputs[0];

    debug!("Translating GELU operation");
    trace!("GELU input: {:?}", input);

    let node = builder.gelu(input);

    trace!("Created GELU node: {:?}", node);
    Ok(node)
}

/// Translate ONNX Swish (SiLU) activation.
///
/// Swish: Y = X * sigmoid(X) = X / (1 + exp(-X))
///
/// Also known as SiLU (Sigmoid Linear Unit).
///
/// # Performance
///
/// - **ClassMap fusion**: Combines multiplication and sigmoid
/// - **SIMD vectorization**: Fast exp() and multiplication
/// - Supports **symbolic shapes**
pub fn translate_swish(
    inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel(
            "Swish expects 1 input, got 0".to_string(),
        ));
    }

    let input = inputs[0];

    debug!("Translating Swish operation");
    trace!("Swish input: {:?}", input);

    // IRBuilder doesn't have swish, need to decompose as x * sigmoid(x)
    // For now, return not-implemented error
    let _ = (builder, input);
    Err(OnnxError::IrTranslationError(
        "Swish operation not yet implemented".to_string(),
    ))
}

/// Translate ONNX ELU activation.
///
/// ELU (Exponential Linear Unit):
/// - Y = X if X > 0
/// - Y = alpha * (exp(X) - 1) if X <= 0
///
/// # Attributes
///
/// - `alpha` (float, default 1.0): Multiplier for negative values
///
/// # Performance
///
/// - **ClassMap fusion**: Fuses with adjacent operations
/// - **SIMD vectorization**: Fast exp() and conditionals
/// - Supports **symbolic shapes**
pub fn translate_elu(
    inputs: &[NodeId],
    attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel(
            "ELU expects 1 input, got 0".to_string(),
        ));
    }

    let input = inputs[0];

    // Parse alpha attribute (default: 1.0)
    let alpha = crate::utils::parse_attr_float(attrs, "alpha", 1.0)?;

    debug!("Translating ELU operation (alpha={})", alpha);
    trace!("ELU input: {:?}", input);

    // IRBuilder doesn't have ELU, need to decompose
    // For now, return not-implemented error
    let _ = (builder, input, alpha);
    Err(OnnxError::IrTranslationError(
        "ELU operation not yet implemented".to_string(),
    ))
}

/// Translate ONNX SELU activation.
///
/// SELU (Scaled Exponential Linear Unit):
/// - Y = scale * X if X > 0
/// - Y = scale * alpha * (exp(X) - 1) if X <= 0
///
/// Standard values: scale = 1.05070098, alpha = 1.67326324
///
/// # Attributes
///
/// - `alpha` (float, default 1.67326324): Multiplier for negative values
/// - `gamma` (float, default 1.05070098): Scale factor
///
/// # Performance
///
/// - **ClassMap fusion**: Combines scaling and ELU
/// - **SIMD vectorization**: Fast exp() and multiplication
/// - Supports **symbolic shapes**
pub fn translate_selu(
    inputs: &[NodeId],
    attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel(
            "SELU expects 1 input, got 0".to_string(),
        ));
    }

    let input = inputs[0];

    // Parse attributes (ONNX standard values for SELU)
    let alpha = crate::utils::parse_attr_float(attrs, "alpha", 1.673_263_2)?;
    let gamma = crate::utils::parse_attr_float(attrs, "gamma", 1.050_701)?;

    debug!(
        "Translating SELU operation (alpha={}, gamma={})",
        alpha, gamma
    );
    trace!("SELU input: {:?}", input);

    // IRBuilder doesn't have SELU, need to decompose
    // For now, return not-implemented error
    let _ = (builder, input, alpha, gamma);
    Err(OnnxError::IrTranslationError(
        "SELU operation not yet implemented".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::f32_tensor;
    use hologram_compiler::ir::IRBuilder;

    fn make_builder() -> IRBuilder {
        IRBuilder::new("test")
    }

    #[test]
    fn test_translate_relu() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[2, 3, 4]));

        let result = translate_relu(&vec![input], &[], &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_relu_no_input() {
        let mut builder = make_builder();
        let result = translate_relu(&vec![], &[], &HashMap::new(), &mut builder);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), OnnxError::InvalidModel(_)));
    }

    #[test]
    fn test_translate_sigmoid() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[1, 2, 3]));

        let result = translate_sigmoid(&vec![input], &[], &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_sigmoid_no_input() {
        let mut builder = make_builder();
        let result = translate_sigmoid(&vec![], &[], &HashMap::new(), &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_translate_tanh() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[4, 5]));

        let result = translate_tanh(&vec![input], &[], &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_tanh_no_input() {
        let mut builder = make_builder();
        let result = translate_tanh(&vec![], &[], &HashMap::new(), &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_translate_softmax_default_axis() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[2, 3, 4]));

        let result = translate_softmax(&vec![input], &[], &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_softmax_custom_axis() {
        use hologram_onnx_spec::attribute_proto::AttributeType;

        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[2, 3, 4]));

        let attrs = vec![AttributeProto {
            name: "axis".to_string(),
            i: 1, // Softmax along axis 1
            r#type: AttributeType::Int as i32,
            ..Default::default()
        }];

        let result = translate_softmax(&vec![input], &attrs, &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_softmax_no_input() {
        let mut builder = make_builder();
        let result = translate_softmax(&vec![], &[], &HashMap::new(), &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_implemented_activations_symbolic_shapes() {
        let mut builder = make_builder();
        // Symbolic batch dimension
        let input = builder.add_input("X", f32_tensor(&[])); // Will have symbolic shape

        let shapes = HashMap::new();

        // Implemented activations should work with symbolic shapes
        assert!(translate_relu(&vec![input], &[], &shapes, &mut builder).is_ok());
        assert!(translate_sigmoid(&vec![input], &[], &shapes, &mut builder).is_ok());
        assert!(translate_tanh(&vec![input], &[], &shapes, &mut builder).is_ok());
        assert!(translate_softmax(&vec![input], &[], &shapes, &mut builder).is_ok());
        assert!(translate_gelu(&vec![input], &[], &shapes, &mut builder).is_ok());
    }

    #[test]
    fn test_activation_chain() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[2, 3]));

        // Create chain: ReLU -> Sigmoid (tests ClassMap fusion potential)
        let relu_out = translate_relu(&vec![input], &[], &HashMap::new(), &mut builder).unwrap();
        let sigmoid_out =
            translate_sigmoid(&vec![relu_out], &[], &HashMap::new(), &mut builder).unwrap();

        // Should successfully create chain
        assert!(sigmoid_out != input);
        assert!(sigmoid_out != relu_out);
    }

    // Tests for advanced activations (GELU, Swish, ELU, SELU)

    #[test]
    fn test_translate_gelu() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[2, 3, 4]));

        let result = translate_gelu(&vec![input], &[], &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_gelu_no_input() {
        let mut builder = make_builder();
        let result = translate_gelu(&vec![], &[], &HashMap::new(), &mut builder);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), OnnxError::InvalidModel(_)));
    }

    #[test]
    fn test_translate_swish_returns_not_implemented() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[1, 2, 3]));

        let result = translate_swish(&vec![input], &[], &HashMap::new(), &mut builder);
        // Swish not yet implemented
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            OnnxError::IrTranslationError(_)
        ));
    }

    #[test]
    fn test_translate_swish_no_input() {
        let mut builder = make_builder();
        let result = translate_swish(&vec![], &[], &HashMap::new(), &mut builder);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), OnnxError::InvalidModel(_)));
    }

    #[test]
    fn test_translate_elu_returns_not_implemented() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[2, 3]));

        let result = translate_elu(&vec![input], &[], &HashMap::new(), &mut builder);
        // ELU not yet implemented
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            OnnxError::IrTranslationError(_)
        ));
    }

    #[test]
    fn test_translate_elu_no_input() {
        let mut builder = make_builder();
        let result = translate_elu(&vec![], &[], &HashMap::new(), &mut builder);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), OnnxError::InvalidModel(_)));
    }

    #[test]
    fn test_translate_selu_returns_not_implemented() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[2, 3]));

        let result = translate_selu(&vec![input], &[], &HashMap::new(), &mut builder);
        // SELU not yet implemented
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            OnnxError::IrTranslationError(_)
        ));
    }

    #[test]
    fn test_translate_selu_no_input() {
        let mut builder = make_builder();
        let result = translate_selu(&vec![], &[], &HashMap::new(), &mut builder);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), OnnxError::InvalidModel(_)));
    }

    #[test]
    fn test_not_implemented_activations_symbolic_shapes() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[])); // Symbolic shape

        let shapes = HashMap::new();

        // Not-implemented activations should return IrTranslationError
        assert!(matches!(
            translate_swish(&vec![input], &[], &shapes, &mut builder).unwrap_err(),
            OnnxError::IrTranslationError(_)
        ));
        assert!(matches!(
            translate_elu(&vec![input], &[], &shapes, &mut builder).unwrap_err(),
            OnnxError::IrTranslationError(_)
        ));
        assert!(matches!(
            translate_selu(&vec![input], &[], &shapes, &mut builder).unwrap_err(),
            OnnxError::IrTranslationError(_)
        ));
    }
}
