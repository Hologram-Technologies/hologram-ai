//! ONNX normalization operations.

use hologram_ir::{GraphBuilder, NodeIndex};
use crate::core::{OnnxError, Result};
use crate::proto::AttributeProto;
use crate::ops::utils::{parse_attr_float, parse_attr_int};

/// Translate ONNX LayerNormalization to IR.
pub fn translate_layer_norm(
    inputs: &[NodeIndex],
    attrs: &[AttributeProto],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel("LayerNorm requires at least 1 input".into()));
    }

    let epsilon = parse_attr_float(attrs, "epsilon", 1e-5)?;
    let axis = parse_attr_int(attrs, "axis", -1)? as i32;

    // Get input node to determine rank
    let input_node = builder.graph().node(inputs[0])
        .ok_or_else(|| OnnxError::InvalidModel("Invalid input node".into()))?;
    let rank = input_node.shape.rank() as i32;

    // Normalize over last dimensions from axis onwards
    let axes: Vec<i32> = if axis < 0 {
        (axis..0).map(|i| rank + i).collect()
    } else {
        (axis..rank).collect()
    };

    let result = builder.unary(
        hologram_ir::NodeOp::LayerNorm { epsilon, axes },
        inputs[0]
    )?;

    Ok(vec![result])
}

/// Translate ONNX BatchNormalization to IR.
pub fn translate_batch_norm(
    inputs: &[NodeIndex],
    attrs: &[AttributeProto],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.len() < 5 {
        return Err(OnnxError::InvalidModel("BatchNorm requires 5 inputs".into()));
    }

    let epsilon = parse_attr_float(attrs, "epsilon", 1e-5)?;
    let momentum = parse_attr_float(attrs, "momentum", 0.9)?;

    // BatchNorm: (x - mean) / sqrt(var + eps) * scale + bias
    let result = builder.unary(
        hologram_ir::NodeOp::BatchNorm { epsilon, momentum },
        inputs[0]
    )?;

    Ok(vec![result])
}

/// Translate ONNX GroupNormalization to IR.
pub fn translate_group_norm(
    inputs: &[NodeIndex],
    attrs: &[AttributeProto],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.len() < 3 {
        return Err(OnnxError::InvalidModel("GroupNorm requires 3 inputs".into()));
    }

    let epsilon = parse_attr_float(attrs, "epsilon", 1e-5)?;
    let _num_groups = parse_attr_int(attrs, "num_groups", 1)?;

    // GroupNorm: normalize within groups
    // Approximate with LayerNorm over spatial dimensions
    let axes = vec![-2, -1]; // Normalize over last 2 dims

    let result = builder.unary(
        hologram_ir::NodeOp::LayerNorm { epsilon, axes },
        inputs[0]
    )?;

    Ok(vec![result])
}

/// Translate ONNX InstanceNormalization to IR.
pub fn translate_instance_norm(
    inputs: &[NodeIndex],
    attrs: &[AttributeProto],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.len() < 3 {
        return Err(OnnxError::InvalidModel("InstanceNorm requires 3 inputs".into()));
    }

    let epsilon = parse_attr_float(attrs, "epsilon", 1e-5)?;

    // InstanceNorm: normalize per instance (spatial dimensions)
    // Normalize over last 2 dimensions (H, W)
    let axes = vec![-2, -1];

    let result = builder.unary(
        hologram_ir::NodeOp::LayerNorm { epsilon, axes },
        inputs[0]
    )?;

    Ok(vec![result])
}
