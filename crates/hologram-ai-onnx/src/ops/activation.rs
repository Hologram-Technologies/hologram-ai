//! ONNX activation functions.
//!
//! Implements translators for neural network activation functions including
//! Softmax, Clip, LeakyReLU, ELU, SELU, PReLU, and Swish.

use hologram::ir::{GraphBuilder, NodeIndex};
use crate::core::{OnnxError, Result};
use crate::proto::AttributeProto;
use crate::ops::utils::{parse_attr_int, parse_attr_float};

/// Translate ONNX Softmax to IR.
///
/// Softmax(X, axis) = exp(X) / sum(exp(X), axis)
pub fn translate_softmax(
    inputs: &[NodeIndex],
    attrs: &[AttributeProto],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel("Softmax requires 1 input".into()));
    }

    let axis = parse_attr_int(attrs, "axis", -1)? as i32;
    let result = builder.softmax(inputs[0], axis)?;

    Ok(vec![result])
}

/// Translate ONNX Clip to IR.
///
/// Clip(X, min, max) = min(max(X, min), max)
pub fn translate_clip(
    inputs: &[NodeIndex],
    _attrs: &[AttributeProto],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel("Clip requires at least 1 input".into()));
    }

    // ONNX Clip can have min/max as inputs (opset 11+) or attributes
    // For now, handle opset 11+ with optional inputs
    let min_val = if inputs.len() > 1 {
        // Extract constant value from min input
        // For now, use None (will be fixed with constant folding)
        None
    } else {
        None
    };

    let max_val = if inputs.len() > 2 {
        // Extract constant value from max input
        None
    } else {
        None
    };

    let result = builder.clip(inputs[0], min_val, max_val)?;

    Ok(vec![result])
}

/// Translate ONNX LeakyReLU to IR.
///
/// LeakyReLU(X) = X if X >= 0 else alpha * X
pub fn translate_leaky_relu(
    inputs: &[NodeIndex],
    attrs: &[AttributeProto],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel("LeakyReLU requires 1 input".into()));
    }

    let alpha = parse_attr_float(attrs, "alpha", 0.01)?;

    // LeakyReLU(x) = max(0, x) + alpha * min(0, x)
    // Decompose into: max(x, alpha * x)
    let alpha_const = builder.constant(
        hologram::ir::ConstantData::F32(vec![alpha]),
        hologram::ir::Shape::static_shape(&[1]),
    );

    let scaled = builder.mul(inputs[0], alpha_const)?;
    let result = builder.binary(hologram::ir::NodeOp::Max, inputs[0], scaled)?;

    Ok(vec![result])
}

/// Translate ONNX ELU to IR.
///
/// ELU(x) = x if x >= 0 else alpha * (exp(x) - 1)
pub fn translate_elu(
    inputs: &[NodeIndex],
    attrs: &[AttributeProto],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel("ELU requires 1 input".into()));
    }

    let alpha = parse_attr_float(attrs, "alpha", 1.0)?;

    // ELU(x) = max(0, x) + alpha * (exp(min(0, x)) - 1)
    // Decompose: where(x >= 0, x, alpha * (exp(x) - 1))

    // Create constants
    let zero = builder.constant(
        hologram::ir::ConstantData::F32(vec![0.0]),
        hologram::ir::Shape::static_shape(&[1]),
    );
    let alpha_const = builder.constant(
        hologram::ir::ConstantData::F32(vec![alpha]),
        hologram::ir::Shape::static_shape(&[1]),
    );
    let one = builder.constant(
        hologram::ir::ConstantData::F32(vec![1.0]),
        hologram::ir::Shape::static_shape(&[1]),
    );

    // Positive part: max(0, x) = x where x >= 0
    let positive_part = builder.binary(hologram::ir::NodeOp::Max, inputs[0], zero)?;

    // Negative part: alpha * (exp(min(0, x)) - 1)
    let negative_part_input = builder.binary(hologram::ir::NodeOp::Min, inputs[0], zero)?;
    let exp_x = builder.unary(hologram::ir::NodeOp::Exp, negative_part_input)?;
    let exp_minus_one = builder.sub(exp_x, one)?;
    let negative_part = builder.mul(alpha_const, exp_minus_one)?;

    // Combine: positive_part + negative_part
    let result = builder.add(positive_part, negative_part)?;

    Ok(vec![result])
}

/// Translate ONNX SELU to IR.
///
/// SELU(x) = gamma * (x if x >= 0 else alpha * (exp(x) - 1))
pub fn translate_selu(
    inputs: &[NodeIndex],
    attrs: &[AttributeProto],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel("SELU requires 1 input".into()));
    }

    let alpha = parse_attr_float(attrs, "alpha", 1.67326)?;
    let gamma = parse_attr_float(attrs, "gamma", 1.0507)?;

    // SELU = gamma * ELU(x, alpha)
    // First compute ELU using decomposition
    let zero = builder.constant(
        hologram::ir::ConstantData::F32(vec![0.0]),
        hologram::ir::Shape::static_shape(&[1]),
    );
    let alpha_const = builder.constant(
        hologram::ir::ConstantData::F32(vec![alpha]),
        hologram::ir::Shape::static_shape(&[1]),
    );
    let one = builder.constant(
        hologram::ir::ConstantData::F32(vec![1.0]),
        hologram::ir::Shape::static_shape(&[1]),
    );

    // ELU decomposition
    let positive_part = builder.binary(hologram::ir::NodeOp::Max, inputs[0], zero)?;
    let negative_part_input = builder.binary(hologram::ir::NodeOp::Min, inputs[0], zero)?;
    let exp_x = builder.unary(hologram::ir::NodeOp::Exp, negative_part_input)?;
    let exp_minus_one = builder.sub(exp_x, one)?;
    let negative_part = builder.mul(alpha_const, exp_minus_one)?;
    let elu = builder.add(positive_part, negative_part)?;

    // Scale by gamma
    let gamma_const = builder.constant(
        hologram::ir::ConstantData::F32(vec![gamma]),
        hologram::ir::Shape::static_shape(&[1]),
    );
    let result = builder.mul(elu, gamma_const)?;

    Ok(vec![result])
}

/// Translate ONNX PReLU to IR.
///
/// PReLU(x) = x if x >= 0 else slope * x
pub fn translate_prelu(
    inputs: &[NodeIndex],
    _attrs: &[AttributeProto],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.len() < 2 {
        return Err(OnnxError::InvalidModel("PReLU requires 2 inputs (x, slope)".into()));
    }

    // PReLU(x, slope) = max(0, x) + slope * min(0, x)
    let zero = builder.constant(
        hologram::ir::ConstantData::F32(vec![0.0]),
        hologram::ir::Shape::static_shape(&[1]),
    );

    let pos_part = builder.binary(hologram::ir::NodeOp::Max, inputs[0], zero)?;
    let neg_part = builder.binary(hologram::ir::NodeOp::Min, inputs[0], zero)?;
    let scaled_neg = builder.mul(inputs[1], neg_part)?;
    let result = builder.add(pos_part, scaled_neg)?;

    Ok(vec![result])
}

/// Translate ONNX Swish to IR.
///
/// Swish(x) = x * sigmoid(x)
pub fn translate_swish(
    inputs: &[NodeIndex],
    _attrs: &[AttributeProto],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel("Swish requires 1 input".into()));
    }

    let sig = builder.sigmoid(inputs[0])?;
    let result = builder.mul(inputs[0], sig)?;

    Ok(vec![result])
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram::ir::{DType, Shape};

    #[test]
    fn test_translate_softmax() {
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);

        let result = translate_softmax(&[x], &[], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_clip() {
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);

        let result = translate_clip(&[x], &[], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_leaky_relu() {
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);

        let result = translate_leaky_relu(&[x], &[], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_elu() {
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);

        let result = translate_elu(&[x], &[], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_selu() {
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);

        let result = translate_selu(&[x], &[], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_prelu() {
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);
        let slope = builder.input("slope", Shape::static_shape(&[3]), DType::F32);

        let result = translate_prelu(&[x, slope], &[], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_swish() {
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);

        let result = translate_swish(&[x], &[], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }
}
