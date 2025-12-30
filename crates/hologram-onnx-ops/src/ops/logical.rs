//! ONNX logical and comparison operations.
//!
//! Operations for logical and conditional operations:
//! - **Where**: Conditional element selection
//! - **Equal**, **Less**, **Greater**, etc.: Comparison operators
//!
//! # Usage in Stable Diffusion
//!
//! - **Where**: Conditional image generation, masking
//! - **Comparisons**: Attention masking, thresholding

use hologram_compiler::ir::{IRBuilder, NodeId};
use hologram_onnx_core::{OnnxError, Result, SymbolicShape};
use hologram_onnx_spec::AttributeProto;
use std::collections::HashMap;
use tracing::{debug, trace};

/// Translate ONNX Where operation.
///
/// Where: Conditional element selection based on condition tensor.
///
/// # Inputs
///
/// - Input 0: condition - Boolean tensor
/// - Input 1: X - Values where condition is true
/// - Input 2: Y - Values where condition is false
///
/// # Output
///
/// Output[i] = X[i] if condition[i] else Y[i]
///
/// # Performance
///
/// - **SIMD vectorization**: Vectorized selection
/// - Supports **broadcasting** for inputs
///
/// # Implementation
///
/// Uses a Call node to `onnx.Where` which the runtime handles.
pub fn translate_where(
    inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.len() != 3 {
        return Err(OnnxError::InvalidModel(format!(
            "Where expects 3 inputs, got {}",
            inputs.len()
        )));
    }

    debug!("Translating Where operation");
    trace!("Where inputs: {:?}", inputs);

    let result = builder.call("onnx.Where", inputs.to_vec());

    trace!("Created Where call node: {:?}", result);
    Ok(result)
}

/// Translate ONNX Equal comparison.
///
/// Equal: Y = (X == Y) element-wise
///
/// # Inputs
///
/// - Input 0: A - First input tensor
/// - Input 1: B - Second input tensor
///
/// # Output
///
/// Boolean tensor with True where A == B
///
/// # Implementation
///
/// Uses a Call node to `onnx.Equal` which the runtime handles.
pub fn translate_equal(
    inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.len() != 2 {
        return Err(OnnxError::InvalidModel(format!(
            "Equal expects 2 inputs, got {}",
            inputs.len()
        )));
    }

    debug!("Translating Equal operation");
    trace!("Equal inputs: {:?}", inputs);

    let result = builder.call("onnx.Equal", inputs.to_vec());

    trace!("Created Equal call node: {:?}", result);
    Ok(result)
}

/// Translate ONNX Less comparison.
///
/// Less: Y = (A < B) element-wise
///
/// # Implementation
///
/// Uses a Call node to `onnx.Less` which the runtime handles.
pub fn translate_less(
    inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.len() != 2 {
        return Err(OnnxError::InvalidModel(format!(
            "Less expects 2 inputs, got {}",
            inputs.len()
        )));
    }

    debug!("Translating Less operation");
    trace!("Less inputs: {:?}", inputs);

    let result = builder.call("onnx.Less", inputs.to_vec());

    trace!("Created Less call node: {:?}", result);
    Ok(result)
}

/// Translate ONNX Greater comparison.
///
/// Greater: Y = (A > B) element-wise
///
/// # Implementation
///
/// Uses a Call node to `onnx.Greater` which the runtime handles.
pub fn translate_greater(
    inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.len() != 2 {
        return Err(OnnxError::InvalidModel(format!(
            "Greater expects 2 inputs, got {}",
            inputs.len()
        )));
    }

    debug!("Translating Greater operation");
    trace!("Greater inputs: {:?}", inputs);

    let result = builder.call("onnx.Greater", inputs.to_vec());

    trace!("Created Greater call node: {:?}", result);
    Ok(result)
}

/// Translate ONNX LessOrEqual comparison.
///
/// LessOrEqual: Y = (A <= B) element-wise
///
/// # Implementation
///
/// Uses a Call node to `onnx.LessOrEqual` which the runtime handles.
pub fn translate_less_or_equal(
    inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.len() != 2 {
        return Err(OnnxError::InvalidModel(format!(
            "LessOrEqual expects 2 inputs, got {}",
            inputs.len()
        )));
    }

    debug!("Translating LessOrEqual operation");
    trace!("LessOrEqual inputs: {:?}", inputs);

    let result = builder.call("onnx.LessOrEqual", inputs.to_vec());

    trace!("Created LessOrEqual call node: {:?}", result);
    Ok(result)
}

/// Translate ONNX GreaterOrEqual comparison.
///
/// GreaterOrEqual: Y = (A >= B) element-wise
///
/// # Implementation
///
/// Uses a Call node to `onnx.GreaterOrEqual` which the runtime handles.
pub fn translate_greater_or_equal(
    inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.len() != 2 {
        return Err(OnnxError::InvalidModel(format!(
            "GreaterOrEqual expects 2 inputs, got {}",
            inputs.len()
        )));
    }

    debug!("Translating GreaterOrEqual operation");
    trace!("GreaterOrEqual inputs: {:?}", inputs);

    let result = builder.call("onnx.GreaterOrEqual", inputs.to_vec());

    trace!("Created GreaterOrEqual call node: {:?}", result);
    Ok(result)
}

/// Translate ONNX Not operation.
///
/// Not: Y = !X (element-wise logical NOT)
///
/// # Implementation
///
/// Uses a Call node to `onnx.Not` which the runtime handles.
pub fn translate_not(
    inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.len() != 1 {
        return Err(OnnxError::InvalidModel(format!(
            "Not expects 1 input, got {}",
            inputs.len()
        )));
    }

    debug!("Translating Not operation");
    trace!("Not input: {:?}", inputs[0]);

    let result = builder.call("onnx.Not", inputs.to_vec());

    trace!("Created Not call node: {:?}", result);
    Ok(result)
}

/// Translate ONNX And operation.
///
/// And: Y = A && B (element-wise logical AND)
///
/// # Implementation
///
/// Uses a Call node to `onnx.And` which the runtime handles.
pub fn translate_and(
    inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.len() != 2 {
        return Err(OnnxError::InvalidModel(format!(
            "And expects 2 inputs, got {}",
            inputs.len()
        )));
    }

    debug!("Translating And operation");
    trace!("And inputs: {:?}", inputs);

    let result = builder.call("onnx.And", inputs.to_vec());

    trace!("Created And call node: {:?}", result);
    Ok(result)
}

/// Translate ONNX Or operation.
///
/// Or: Y = A || B (element-wise logical OR)
///
/// # Implementation
///
/// Uses a Call node to `onnx.Or` which the runtime handles.
pub fn translate_or(
    inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.len() != 2 {
        return Err(OnnxError::InvalidModel(format!(
            "Or expects 2 inputs, got {}",
            inputs.len()
        )));
    }

    debug!("Translating Or operation");
    trace!("Or inputs: {:?}", inputs);

    let result = builder.call("onnx.Or", inputs.to_vec());

    trace!("Created Or call node: {:?}", result);
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
    // Where Tests
    // ========================================================================

    #[test]
    fn test_translate_where() {
        let mut builder = make_builder();
        let condition = builder.add_input("condition", f32_tensor(&[2, 3]));
        let x = builder.add_input("X", f32_tensor(&[2, 3]));
        let y = builder.add_input("Y", f32_tensor(&[2, 3]));

        let result =
            translate_where(&[condition, x, y], &[], &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_where_wrong_inputs() {
        let mut builder = make_builder();
        let x = builder.add_input("X", f32_tensor(&[2, 3]));

        let result = translate_where(&[x], &[], &HashMap::new(), &mut builder);
        assert!(result.is_err());
    }

    // ========================================================================
    // Comparison Tests
    // ========================================================================

    #[test]
    fn test_translate_equal() {
        let mut builder = make_builder();
        let a = builder.add_input("A", f32_tensor(&[2, 3]));
        let b = builder.add_input("B", f32_tensor(&[2, 3]));

        let result = translate_equal(&[a, b], &[], &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_equal_wrong_inputs() {
        let mut builder = make_builder();
        let a = builder.add_input("A", f32_tensor(&[2, 3]));

        let result = translate_equal(&[a], &[], &HashMap::new(), &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_translate_less() {
        let mut builder = make_builder();
        let a = builder.add_input("A", f32_tensor(&[2, 3]));
        let b = builder.add_input("B", f32_tensor(&[2, 3]));

        let result = translate_less(&[a, b], &[], &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_greater() {
        let mut builder = make_builder();
        let a = builder.add_input("A", f32_tensor(&[2, 3]));
        let b = builder.add_input("B", f32_tensor(&[2, 3]));

        let result = translate_greater(&[a, b], &[], &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_less_or_equal() {
        let mut builder = make_builder();
        let a = builder.add_input("A", f32_tensor(&[2, 3]));
        let b = builder.add_input("B", f32_tensor(&[2, 3]));

        let result = translate_less_or_equal(&[a, b], &[], &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_greater_or_equal() {
        let mut builder = make_builder();
        let a = builder.add_input("A", f32_tensor(&[2, 3]));
        let b = builder.add_input("B", f32_tensor(&[2, 3]));

        let result = translate_greater_or_equal(&[a, b], &[], &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    // ========================================================================
    // Logical Tests
    // ========================================================================

    #[test]
    fn test_translate_not() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[2, 3]));

        let result = translate_not(&[input], &[], &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_not_wrong_inputs() {
        let mut builder = make_builder();
        let result = translate_not(&[], &[], &HashMap::new(), &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_translate_and() {
        let mut builder = make_builder();
        let a = builder.add_input("A", f32_tensor(&[2, 3]));
        let b = builder.add_input("B", f32_tensor(&[2, 3]));

        let result = translate_and(&[a, b], &[], &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_or() {
        let mut builder = make_builder();
        let a = builder.add_input("A", f32_tensor(&[2, 3]));
        let b = builder.add_input("B", f32_tensor(&[2, 3]));

        let result = translate_or(&[a, b], &[], &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }
}
