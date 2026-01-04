//! ONNX shape manipulation operations.
//!
//! Implements translators for Reshape, Transpose, Concat, Squeeze, Unsqueeze, etc.

use hologram_ir::{GraphBuilder, NodeIndex};
use crate::core::{OnnxError, Result};
use crate::proto::AttributeProto;
use crate::ops::utils::{parse_attr_int, parse_attr_ints};

/// Translate ONNX Reshape to IR.
pub fn translate_reshape(
    inputs: &[NodeIndex],
    _attrs: &[AttributeProto],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.len() < 2 {
        return Err(OnnxError::InvalidModel("Reshape requires 2 inputs (data, shape)".into()));
    }

    // Get shape from second input (should be constant)
    // The shape is provided as a tensor input, which hologram-ir will handle
    let result = builder.unary(hologram_ir::NodeOp::Reshape { new_shape: vec![] }, inputs[0])?;
    Ok(vec![result])
}

/// Translate ONNX Transpose to IR.
pub fn translate_transpose(
    inputs: &[NodeIndex],
    attrs: &[AttributeProto],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel("Transpose requires 1 input".into()));
    }

    let perm = parse_attr_ints(attrs, "perm", vec![])?;
    let perm_i32: Vec<i32> = perm.iter().map(|&x| x as i32).collect();

    let result = if perm_i32.is_empty() {
        // Default: reverse all axes
        builder.unary(hologram_ir::NodeOp::Transpose { perm: vec![] }, inputs[0])?
    } else {
        builder.transpose(inputs[0], perm_i32)?
    };

    Ok(vec![result])
}

/// Translate ONNX Concat to IR.
pub fn translate_concat(
    inputs: &[NodeIndex],
    attrs: &[AttributeProto],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel("Concat requires at least 1 input".into()));
    }

    let axis = parse_attr_int(attrs, "axis", 0)? as i32;
    let result = builder.concat(inputs, axis)?;

    Ok(vec![result])
}

/// Translate ONNX Squeeze to IR.
pub fn translate_squeeze(
    inputs: &[NodeIndex],
    _attrs: &[AttributeProto],
    _builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel("Squeeze requires 1 input".into()));
    }

    // Squeeze removes dimensions of size 1
    // For now, return identity (will be fixed with proper shape inference)
    Ok(vec![inputs[0]])
}

/// Translate ONNX Unsqueeze to IR.
pub fn translate_unsqueeze(
    inputs: &[NodeIndex],
    _attrs: &[AttributeProto],
    _builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel("Unsqueeze requires 1 input".into()));
    }

    // Unsqueeze adds dimensions of size 1
    // For now, return identity (will be fixed with proper shape inference)
    Ok(vec![inputs[0]])
}

/// Translate ONNX Flatten to IR.
pub fn translate_flatten(
    inputs: &[NodeIndex],
    attrs: &[AttributeProto],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel("Flatten requires 1 input".into()));
    }

    let _axis = parse_attr_int(attrs, "axis", 1)?;

    // Flatten reshapes to 2D
    // For now, use unary reshape operation
    let result = builder.unary(hologram_ir::NodeOp::Reshape { new_shape: vec![] }, inputs[0])?;

    Ok(vec![result])
}

/// Translate ONNX Expand to IR.
pub fn translate_expand(
    inputs: &[NodeIndex],
    _attrs: &[AttributeProto],
    _builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.len() < 2 {
        return Err(OnnxError::InvalidModel("Expand requires 2 inputs".into()));
    }

    // Expand broadcasts to a new shape
    // For now, return identity (broadcasting handled automatically)
    Ok(vec![inputs[0]])
}

/// Translate ONNX Split to IR.
pub fn translate_split(
    inputs: &[NodeIndex],
    attrs: &[AttributeProto],
    _builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel("Split requires 1 input".into()));
    }

    let _axis = parse_attr_int(attrs, "axis", 0)?;
    let _splits = parse_attr_ints(attrs, "split", vec![])?;

    // Split divides tensor along axis
    // For now, return single output (multi-output support needed)
    Ok(vec![inputs[0]])
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_ir::{DType, Shape};

    #[test]
    fn test_translate_transpose() {
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[2, 3, 4]), DType::F32);

        let result = translate_transpose(&[x], &[], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_concat() {
        let mut builder = GraphBuilder::new();
        let a = builder.input("a", Shape::static_shape(&[2, 3]), DType::F32);
        let b = builder.input("b", Shape::static_shape(&[2, 3]), DType::F32);

        let result = translate_concat(&[a, b], &[], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }
}
