//! ONNX reduction operations.

use hologram_ir::{GraphBuilder, NodeIndex};
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
    let result = builder.unary(hologram_ir::NodeOp::ReduceMean { axes: axes_i32, keepdims }, inputs[0])?;

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
    let result = builder.unary(hologram_ir::NodeOp::ReduceSum { axes: axes_i32, keepdims }, inputs[0])?;

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
    let result = builder.unary(hologram_ir::NodeOp::ReduceMax { axes: axes_i32, keepdims }, inputs[0])?;

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
    let result = builder.unary(hologram_ir::NodeOp::ReduceMin { axes: axes_i32, keepdims }, inputs[0])?;

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
