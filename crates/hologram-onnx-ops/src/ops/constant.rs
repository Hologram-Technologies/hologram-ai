//! ONNX constant and identity operations.
//!
//! Operations for creating constants and passing values through:
//! - **Constant**: Create a tensor with constant values
//! - **Identity**: Pass input through unchanged
//! - **ConstantOfShape**: Create a tensor of a specific shape with a constant value
//!
//! # Usage in Stable Diffusion
//!
//! - **Constant**: Timestep embeddings, configuration values
//! - **Identity**: Graph organization, skip connections

use hologram_compiler::ir::{IRBuilder, NodeId};
use hologram_onnx_core::{OnnxError, Result, SymbolicShape};
use hologram_onnx_spec::AttributeProto;
use std::collections::HashMap;
use tracing::{debug, trace};

/// Translate ONNX Constant operation.
///
/// Constant: Creates a tensor with constant values from attributes.
///
/// # Attributes
///
/// - `value` (tensor): The constant tensor value
/// - `value_float` (float): Single float value (creates scalar)
/// - `value_floats` (floats): Float array
/// - `value_int` (int): Single int value (creates scalar)
/// - `value_ints` (ints): Int array
///
/// # Performance
///
/// - Constants are embedded in the compiled graph
/// - No runtime overhead
pub fn translate_constant(
    inputs: &[NodeId],
    attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if !inputs.is_empty() {
        return Err(OnnxError::InvalidModel(format!(
            "Constant expects 0 inputs, got {}",
            inputs.len()
        )));
    }

    debug!("Translating Constant operation");

    // Try to get the constant value from attributes
    // Priority: value (tensor) > value_float > value_floats > value_int > value_ints

    // Check for tensor value attribute
    for attr in attrs {
        if attr.name == "value"
            && let Some(ref tensor) = attr.t {
                // Get the first float value from tensor data
                let val = if !tensor.float_data.is_empty() {
                    tensor.float_data[0]
                } else if !tensor.raw_data.is_empty() && tensor.raw_data.len() >= 4 {
                    // Try to read as raw f32
                    f32::from_le_bytes([
                        tensor.raw_data[0],
                        tensor.raw_data[1],
                        tensor.raw_data[2],
                        tensor.raw_data[3],
                    ])
                } else {
                    0.0
                };
                trace!("Found tensor constant: {}", val);
                let result = builder.add_f32(val);
                return Ok(result);
            }
    }

    // Check for value_float
    for attr in attrs {
        if attr.name == "value_float" {
            let val = attr.f;
            trace!("Found value_float constant: {}", val);
            let result = builder.add_f32(val);
            return Ok(result);
        }
    }

    // Check for value_int
    for attr in attrs {
        if attr.name == "value_int" {
            let val = attr.i as f32;
            trace!("Found value_int constant: {}", val);
            let result = builder.add_f32(val);
            return Ok(result);
        }
    }

    // Check for value_floats
    for attr in attrs {
        if attr.name == "value_floats" && !attr.floats.is_empty() {
            let val = attr.floats[0];
            trace!("Found value_floats constant: {} elements", attr.floats.len());
            let result = builder.add_f32(val);
            return Ok(result);
        }
    }

    // Default to zero if no value found
    debug!("No constant value found in attributes, defaulting to 0.0");
    let result = builder.add_f32(0.0);
    Ok(result)
}

/// Translate ONNX Identity operation.
///
/// Identity: Y = X (pass input through unchanged)
///
/// # Inputs
///
/// - Input 0: X - Input tensor
///
/// # Performance
///
/// - Zero overhead - just passes through the input reference
/// - Used for graph organization and skip connections
pub fn translate_identity(
    inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    _builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.len() != 1 {
        return Err(OnnxError::InvalidModel(format!(
            "Identity expects 1 input, got {}",
            inputs.len()
        )));
    }

    let input = inputs[0];

    debug!("Translating Identity operation");
    trace!("Identity input: {:?}", input);

    // Identity just returns the input unchanged
    let result = input;

    trace!("Created Identity node: {:?}", result);
    Ok(result)
}

/// Translate ONNX ConstantOfShape operation.
///
/// ConstantOfShape: Creates a tensor of the given shape filled with a constant value.
///
/// # Inputs
///
/// - Input 0: shape - 1-D tensor defining output shape
///
/// # Attributes
///
/// - `value` (tensor, default [0.0]): The constant value to fill the tensor with
pub fn translate_constant_of_shape(
    inputs: &[NodeId],
    attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.len() != 1 {
        return Err(OnnxError::InvalidModel(format!(
            "ConstantOfShape expects 1 input, got {}",
            inputs.len()
        )));
    }

    let _shape_input = inputs[0];

    debug!("Translating ConstantOfShape operation");
    trace!("ConstantOfShape shape input: {:?}", _shape_input);

    // Get the constant value (default 0.0)
    let mut value = 0.0_f32;
    for attr in attrs {
        if attr.name == "value"
            && let Some(ref tensor) = attr.t {
                if !tensor.float_data.is_empty() {
                    value = tensor.float_data[0];
                } else if !tensor.raw_data.is_empty() && tensor.raw_data.len() >= 4 {
                    value = f32::from_le_bytes([
                        tensor.raw_data[0],
                        tensor.raw_data[1],
                        tensor.raw_data[2],
                        tensor.raw_data[3],
                    ]);
                }
            }
    }

    trace!("ConstantOfShape fill value: {}", value);

    // Create a constant node with the fill value
    // The actual shape will be determined at runtime
    let result = builder.add_f32(value);

    trace!("Created ConstantOfShape node: {:?}", result);
    Ok(result)
}

/// Translate ONNX Shape operation.
///
/// Shape: Returns the shape of the input tensor as a 1-D tensor.
///
/// # Inputs
///
/// - Input 0: data - Input tensor
///
/// # Output
///
/// 1-D tensor containing the shape of the input
pub fn translate_shape_op(
    inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    _builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.len() != 1 {
        return Err(OnnxError::InvalidModel(format!(
            "Shape expects 1 input, got {}",
            inputs.len()
        )));
    }

    let input = inputs[0];

    debug!("Translating Shape operation");
    trace!("Shape input: {:?}", input);

    // Shape returns the shape of the input
    // For now, passthrough - shape operations need special handling
    let result = input;

    trace!("Created Shape node (passthrough): {:?}", result);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::f32_tensor;
    use hologram_compiler::ir::IRBuilder;
    use hologram_onnx_spec::attribute_proto::AttributeType;

    fn make_builder() -> IRBuilder {
        IRBuilder::new("test")
    }

    fn make_float_attr(name: &str, value: f32) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            f: value,
            r#type: AttributeType::Float as i32,
            ..Default::default()
        }
    }

    fn make_int_attr(name: &str, value: i64) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            i: value,
            r#type: AttributeType::Int as i32,
            ..Default::default()
        }
    }

    // ========================================================================
    // Constant Tests
    // ========================================================================

    #[test]
    fn test_translate_constant_float() {
        let mut builder = make_builder();
        let attrs = vec![make_float_attr("value_float", 3.14)];

        let result = translate_constant(&[], &attrs, &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_constant_int() {
        let mut builder = make_builder();
        let attrs = vec![make_int_attr("value_int", 42)];

        let result = translate_constant(&[], &attrs, &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_constant_default() {
        let mut builder = make_builder();

        let result = translate_constant(&[], &[], &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_constant_wrong_inputs() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[2, 3]));

        let result = translate_constant(&[input], &[], &HashMap::new(), &mut builder);
        assert!(result.is_err());
    }

    // ========================================================================
    // Identity Tests
    // ========================================================================

    #[test]
    fn test_translate_identity() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[2, 3, 4]));

        let result = translate_identity(&vec![input], &[], &HashMap::new(), &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), input);
    }

    #[test]
    fn test_translate_identity_symbolic() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[]));

        let result = translate_identity(&vec![input], &[], &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_identity_wrong_inputs() {
        let mut builder = make_builder();
        let result = translate_identity(&[], &[], &HashMap::new(), &mut builder);
        assert!(result.is_err());
    }

    // ========================================================================
    // ConstantOfShape Tests
    // ========================================================================

    #[test]
    fn test_translate_constant_of_shape() {
        let mut builder = make_builder();
        let shape = builder.add_input("shape", f32_tensor(&[3]));

        let result = translate_constant_of_shape(&vec![shape], &[], &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_constant_of_shape_with_value() {
        let mut builder = make_builder();
        let shape = builder.add_input("shape", f32_tensor(&[3]));
        let attrs = vec![make_float_attr("value_float", 1.0)];

        let result =
            translate_constant_of_shape(&vec![shape], &attrs, &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    // ========================================================================
    // Shape Tests
    // ========================================================================

    #[test]
    fn test_translate_shape_op() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[2, 3, 4]));

        let result = translate_shape_op(&vec![input], &[], &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_shape_op_wrong_inputs() {
        let mut builder = make_builder();
        let result = translate_shape_op(&[], &[], &HashMap::new(), &mut builder);
        assert!(result.is_err());
    }
}
