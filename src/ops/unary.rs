//! ONNX unary operations.
//!
//! Implements translators for unary mathematical operations.

use hologram_ir::{GraphBuilder, NodeIndex};
use crate::core::{OnnxError, Result};

/// Translate ONNX Abs to IR.
pub fn translate_abs(
    inputs: &[NodeIndex],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel("Abs requires 1 input".into()));
    }

    let result = builder.unary(hologram_ir::NodeOp::Abs, inputs[0])?;
    Ok(vec![result])
}

/// Translate ONNX Cos to IR.
pub fn translate_cos(
    inputs: &[NodeIndex],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel("Cos requires 1 input".into()));
    }

    let result = builder.unary(hologram_ir::NodeOp::Cos, inputs[0])?;
    Ok(vec![result])
}

/// Translate ONNX Erf to IR.
pub fn translate_erf(
    inputs: &[NodeIndex],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel("Erf requires 1 input".into()));
    }

    let result = builder.erf(inputs[0])?;
    Ok(vec![result])
}

/// Translate ONNX Exp to IR.
pub fn translate_exp(
    inputs: &[NodeIndex],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel("Exp requires 1 input".into()));
    }

    let result = builder.unary(hologram_ir::NodeOp::Exp, inputs[0])?;
    Ok(vec![result])
}

/// Translate ONNX Log to IR.
pub fn translate_log(
    inputs: &[NodeIndex],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel("Log requires 1 input".into()));
    }

    let result = builder.unary(hologram_ir::NodeOp::Log, inputs[0])?;
    Ok(vec![result])
}

/// Translate ONNX Neg to IR.
pub fn translate_neg(
    inputs: &[NodeIndex],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel("Neg requires 1 input".into()));
    }

    let result = builder.neg(inputs[0])?;
    Ok(vec![result])
}

/// Translate ONNX Reciprocal to IR.
///
/// Reciprocal(x) = 1 / x
pub fn translate_reciprocal(
    inputs: &[NodeIndex],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel("Reciprocal requires 1 input".into()));
    }

    // Decompose: 1 / x
    let one = builder.constant(
        hologram_ir::ConstantData::F32(vec![1.0]),
        hologram_ir::Shape::static_shape(&[1]),
    );

    let result = builder.div(one, inputs[0])?;
    Ok(vec![result])
}

/// Translate ONNX Sin to IR.
pub fn translate_sin(
    inputs: &[NodeIndex],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel("Sin requires 1 input".into()));
    }

    let result = builder.unary(hologram_ir::NodeOp::Sin, inputs[0])?;
    Ok(vec![result])
}

/// Translate ONNX Sqrt to IR.
pub fn translate_sqrt(
    inputs: &[NodeIndex],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel("Sqrt requires 1 input".into()));
    }

    let result = builder.unary(hologram_ir::NodeOp::Sqrt, inputs[0])?;
    Ok(vec![result])
}

/// Translate ONNX Tan to IR.
///
/// Tan(x) = sin(x) / cos(x)
pub fn translate_tan(
    inputs: &[NodeIndex],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel("Tan requires 1 input".into()));
    }

    // Decompose: tan(x) = sin(x) / cos(x)
    let sin_x = builder.unary(hologram_ir::NodeOp::Sin, inputs[0])?;
    let cos_x = builder.unary(hologram_ir::NodeOp::Cos, inputs[0])?;

    let result = builder.div(sin_x, cos_x)?;
    Ok(vec![result])
}
