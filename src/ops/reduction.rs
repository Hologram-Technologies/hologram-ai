//! ONNX reduction operations.

use hologram::ir::{GraphBuilder, NodeIndex};
use crate::core::{OnnxError, Result};
use crate::proto::AttributeProto;
use crate::ops::utils::{parse_attr_int, parse_attr_ints};

/// Translate ONNX ReduceMean to IR.
pub fn translate_reduce_mean(
    inputs: &[NodeIndex],
    attrs: &[AttributeProto],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel("ReduceMean requires 1 input".into()));
    }

    let axes = parse_attr_ints(attrs, "axes", vec![])?;
    let keepdims = parse_attr_int(attrs, "keepdims", 1)? != 0;

    let axes_i32: Vec<i32> = axes.iter().map(|&x| x as i32).collect();
    let result = builder.unary(hologram::ir::NodeOp::ReduceMean { axes: axes_i32, keepdims }, inputs[0])?;

    Ok(vec![result])
}

/// Translate ONNX ReduceSum to IR.
pub fn translate_reduce_sum(
    inputs: &[NodeIndex],
    attrs: &[AttributeProto],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel("ReduceSum requires 1 input".into()));
    }

    let axes = parse_attr_ints(attrs, "axes", vec![])?;
    let keepdims = parse_attr_int(attrs, "keepdims", 1)? != 0;

    let axes_i32: Vec<i32> = axes.iter().map(|&x| x as i32).collect();
    let result = builder.unary(hologram::ir::NodeOp::ReduceSum { axes: axes_i32, keepdims }, inputs[0])?;

    Ok(vec![result])
}

/// Translate ONNX ReduceMax to IR.
pub fn translate_reduce_max(
    inputs: &[NodeIndex],
    attrs: &[AttributeProto],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel("ReduceMax requires 1 input".into()));
    }

    let axes = parse_attr_ints(attrs, "axes", vec![])?;
    let keepdims = parse_attr_int(attrs, "keepdims", 1)? != 0;

    let axes_i32: Vec<i32> = axes.iter().map(|&x| x as i32).collect();
    let result = builder.unary(hologram::ir::NodeOp::ReduceMax { axes: axes_i32, keepdims }, inputs[0])?;

    Ok(vec![result])
}

/// Translate ONNX ReduceMin to IR.
pub fn translate_reduce_min(
    inputs: &[NodeIndex],
    attrs: &[AttributeProto],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel("ReduceMin requires 1 input".into()));
    }

    let axes = parse_attr_ints(attrs, "axes", vec![])?;
    let keepdims = parse_attr_int(attrs, "keepdims", 1)? != 0;

    let axes_i32: Vec<i32> = axes.iter().map(|&x| x as i32).collect();
    let result = builder.unary(hologram::ir::NodeOp::ReduceMin { axes: axes_i32, keepdims }, inputs[0])?;

    Ok(vec![result])
}

/// Translate ONNX ReduceProd to IR.
/// Note: ReduceProd is not supported in hologram-ir.
pub fn translate_reduce_prod(
    _inputs: &[NodeIndex],
    _attrs: &[AttributeProto],
    _builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    Err(OnnxError::UnsupportedOp {
        op_type: "ReduceProd".into(),
        opset_version: 13,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::attribute_proto::AttributeType;
    use hologram::ir::{DType, Shape};

    fn make_int_attr(name: &str, value: i64) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            i: value,
            r#type: AttributeType::Int as i32,
            ..Default::default()
        }
    }

    fn make_ints_attr(name: &str, values: Vec<i64>) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            ints: values,
            r#type: AttributeType::Ints as i32,
            ..Default::default()
        }
    }

    #[test]
    fn test_translate_reduce_sum_single_axis() {
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[2, 3, 4]), DType::F32);

        let attrs = vec![
            make_ints_attr("axes", vec![1]),
            make_int_attr("keepdims", 1),
        ];

        let result = translate_reduce_sum(&[input], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_reduce_sum_multiple_axes() {
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[2, 3, 4, 5]), DType::F32);

        let attrs = vec![
            make_ints_attr("axes", vec![1, 3]),
            make_int_attr("keepdims", 0),
        ];

        let result = translate_reduce_sum(&[input], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_reduce_sum_no_keepdims() {
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[2, 3, 4]), DType::F32);

        let attrs = vec![
            make_ints_attr("axes", vec![2]),
            make_int_attr("keepdims", 0),
        ];

        let result = translate_reduce_sum(&[input], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_reduce_mean_basic() {
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[4, 5, 6]), DType::F32);

        let attrs = vec![
            make_ints_attr("axes", vec![0, 2]),
            make_int_attr("keepdims", 1),
        ];

        let result = translate_reduce_mean(&[input], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_reduce_max_basic() {
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[10, 20]), DType::F32);

        let attrs = vec![
            make_ints_attr("axes", vec![1]),
            make_int_attr("keepdims", 1),
        ];

        let result = translate_reduce_max(&[input], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_reduce_min_basic() {
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[3, 3, 3]), DType::F32);

        let attrs = vec![
            make_ints_attr("axes", vec![0]),
            make_int_attr("keepdims", 0),
        ];

        let result = translate_reduce_min(&[input], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_reduce_sum_no_inputs() {
        let mut builder = GraphBuilder::new();

        let attrs = vec![make_ints_attr("axes", vec![0])];
        let result = translate_reduce_sum(&[], &attrs, &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_translate_reduce_mean_no_inputs() {
        let mut builder = GraphBuilder::new();

        let attrs = vec![make_ints_attr("axes", vec![0])];
        let result = translate_reduce_mean(&[], &attrs, &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_translate_reduce_prod_unsupported() {
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[2, 3]), DType::F32);

        let attrs = vec![make_ints_attr("axes", vec![1])];
        let result = translate_reduce_prod(&[input], &attrs, &mut builder);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), OnnxError::UnsupportedOp { .. }));
    }

    #[test]
    fn test_translate_reduce_sum_default_keepdims() {
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[5, 5]), DType::F32);

        // Default keepdims should be 1 (true)
        let attrs = vec![make_ints_attr("axes", vec![0])];
        let result = translate_reduce_sum(&[input], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }
}
