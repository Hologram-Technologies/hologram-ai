//! ONNX unary operations.
//!
//! Implements translators for unary mathematical operations.

use hologram::ir::{GraphBuilder, NodeIndex};
use crate::core::{OnnxError, Result};

/// Translate ONNX Abs to IR.
pub fn translate_abs(
    inputs: &[NodeIndex],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel("Abs requires 1 input".into()));
    }

    let result = builder.unary(hologram::ir::NodeOp::Abs, inputs[0])?;
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

    let result = builder.unary(hologram::ir::NodeOp::Cos, inputs[0])?;
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

    let result = builder.unary(hologram::ir::NodeOp::Exp, inputs[0])?;
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

    let result = builder.unary(hologram::ir::NodeOp::Log, inputs[0])?;
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
        hologram::ir::ConstantData::F32(vec![1.0]),
        hologram::ir::Shape::static_shape(&[1]),
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

    let result = builder.unary(hologram::ir::NodeOp::Sin, inputs[0])?;
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

    let result = builder.unary(hologram::ir::NodeOp::Sqrt, inputs[0])?;
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
    let sin_x = builder.unary(hologram::ir::NodeOp::Sin, inputs[0])?;
    let cos_x = builder.unary(hologram::ir::NodeOp::Cos, inputs[0])?;

    let result = builder.div(sin_x, cos_x)?;
    Ok(vec![result])
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram::ir::{DType, Shape};

    #[test]
    fn test_translate_abs() {
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[3, 4]), DType::F32);

        let result = translate_abs(&[input], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_abs_no_inputs() {
        let mut builder = GraphBuilder::new();
        let result = translate_abs(&[], &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_translate_cos() {
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[2, 2]), DType::F32);

        let result = translate_cos(&[input], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_sin() {
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[5]), DType::F32);

        let result = translate_sin(&[input], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_tan() {
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[10, 10]), DType::F32);

        let result = translate_tan(&[input], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_exp() {
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[8]), DType::F32);

        let result = translate_exp(&[input], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_log() {
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[16]), DType::F32);

        let result = translate_log(&[input], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_sqrt() {
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[3, 3, 3]), DType::F32);

        let result = translate_sqrt(&[input], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_neg() {
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[7]), DType::F32);

        let result = translate_neg(&[input], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_erf() {
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[12, 12]), DType::F32);

        let result = translate_erf(&[input], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_reciprocal() {
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[4, 4]), DType::F32);

        let result = translate_reciprocal(&[input], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_reciprocal_no_inputs() {
        let mut builder = GraphBuilder::new();
        let result = translate_reciprocal(&[], &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_translate_sin_no_inputs() {
        let mut builder = GraphBuilder::new();
        let result = translate_sin(&[], &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_translate_exp_no_inputs() {
        let mut builder = GraphBuilder::new();
        let result = translate_exp(&[], &mut builder);
        assert!(result.is_err());
    }
}
