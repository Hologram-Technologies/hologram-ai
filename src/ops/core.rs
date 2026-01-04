//! Core ONNX arithmetic operations.
//!
//! This module provides translators for basic arithmetic operations like Add, Sub, Mul,
//! Div, MatMul, Gemm, and Pow.

use hologram_ir::{GraphBuilder, NodeIndex};
use crate::core::{OnnxError, Result};
use crate::proto::AttributeProto;
use crate::ops::utils::{parse_attr_float, parse_attr_int};

/// Translate ONNX Gemm operation to IR.
///
/// GEMM: General Matrix Multiplication
/// Y = alpha * A' * B' + beta * C
///
/// where A', B' are optionally transposed versions of A, B.
///
/// # Arguments
///
/// * `inputs` - [A, B, C (optional)]
/// * `attrs` - Attributes including alpha, beta, transA, transB
/// * `builder` - IR graph builder
///
/// # Returns
///
/// Vector with single output node
pub fn translate_gemm(
    inputs: &[NodeIndex],
    attrs: &[AttributeProto],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.len() < 2 {
        return Err(OnnxError::InvalidModel(format!(
            "Gemm requires at least 2 inputs, got {}",
            inputs.len()
        )));
    }

    let a = inputs[0];
    let b = inputs[1];
    let c = inputs.get(2).copied();

    // Parse attributes
    let alpha = parse_attr_float(attrs, "alpha", 1.0)?;
    let beta = parse_attr_float(attrs, "beta", 1.0)?;
    let trans_a = parse_attr_int(attrs, "transA", 0)? != 0;
    let trans_b = parse_attr_int(attrs, "transB", 0)? != 0;

    // Use builder's gemm function
    let result = builder.gemm(a, b, c, alpha, beta, trans_a, trans_b)?;

    Ok(vec![result])
}

/// Translate ONNX Pow operation to IR.
///
/// Y = X ^ Exponent (element-wise power)
///
/// # Arguments
///
/// * `inputs` - [X, Exponent]
/// * `attrs` - (none)
/// * `builder` - IR graph builder
///
/// # Returns
///
/// Vector with single output node
pub fn translate_pow(
    inputs: &[NodeIndex],
    _attrs: &[AttributeProto],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.len() != 2 {
        return Err(OnnxError::InvalidModel(format!(
            "Pow requires 2 inputs, got {}",
            inputs.len()
        )));
    }

    // Pow is a binary operation with broadcasting
    let result = builder.binary(hologram_ir::NodeOp::Pow, inputs[0], inputs[1])?;

    Ok(vec![result])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::attribute_proto::AttributeType;
    use hologram_ir::{DType, Shape};

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

    #[test]
    fn test_translate_gemm_basic() {
        let mut builder = GraphBuilder::new();

        let a = builder.input("a", Shape::static_shape(&[2, 3]), DType::F32);
        let b = builder.input("b", Shape::static_shape(&[3, 4]), DType::F32);

        let attrs = vec![];
        let result = translate_gemm(&[a, b], &attrs, &mut builder);

        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_gemm_with_bias() {
        let mut builder = GraphBuilder::new();

        let a = builder.input("a", Shape::static_shape(&[2, 3]), DType::F32);
        let b = builder.input("b", Shape::static_shape(&[3, 4]), DType::F32);
        let c = builder.input("c", Shape::static_shape(&[2, 4]), DType::F32);

        let attrs = vec![];
        let result = translate_gemm(&[a, b, c], &attrs, &mut builder);

        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_gemm_with_transpose() {
        let mut builder = GraphBuilder::new();

        let a = builder.input("a", Shape::static_shape(&[3, 2]), DType::F32);
        let b = builder.input("b", Shape::static_shape(&[4, 3]), DType::F32);

        let attrs = vec![
            make_int_attr("transA", 1),
            make_int_attr("transB", 1),
        ];
        let result = translate_gemm(&[a, b], &attrs, &mut builder);

        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_gemm_with_alpha_beta() {
        let mut builder = GraphBuilder::new();

        let a = builder.input("a", Shape::static_shape(&[2, 3]), DType::F32);
        let b = builder.input("b", Shape::static_shape(&[3, 4]), DType::F32);
        let c = builder.input("c", Shape::static_shape(&[2, 4]), DType::F32);

        let attrs = vec![
            make_float_attr("alpha", 2.0),
            make_float_attr("beta", 0.5),
        ];
        let result = translate_gemm(&[a, b, c], &attrs, &mut builder);

        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_gemm_missing_inputs() {
        let mut builder = GraphBuilder::new();

        let a = builder.input("a", Shape::static_shape(&[2, 3]), DType::F32);

        let attrs = vec![];
        let result = translate_gemm(&[a], &attrs, &mut builder);

        assert!(result.is_err());
    }

    #[test]
    fn test_translate_pow() {
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);
        let exp = builder.input("exp", Shape::static_shape(&[2, 3]), DType::F32);

        let result = translate_pow(&[x, exp], &[], &mut builder);

        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_pow_broadcast() {
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);
        let exp = builder.input("exp", Shape::static_shape(&[1]), DType::F32);

        let result = translate_pow(&[x, exp], &[], &mut builder);

        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_pow_missing_inputs() {
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);

        let result = translate_pow(&[x], &[], &mut builder);

        assert!(result.is_err());
    }
}
