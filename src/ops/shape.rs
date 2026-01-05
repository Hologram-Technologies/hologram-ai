//! ONNX shape manipulation operations.
//!
//! Implements translators for Reshape, Transpose, Concat, Squeeze, Unsqueeze, etc.

use hologram_ir::{GraphBuilder, NodeIndex};
use crate::core::{OnnxError, Result};
use crate::proto::AttributeProto;
use crate::ops::utils::{parse_attr_int, parse_attr_ints};

/// Translate ONNX Reshape to IR.
///
/// Supports both static reshape (constant shape) and dynamic reshape (runtime shape).
/// ONNX Reshape with allowzero=1 is supported via the dynamic reshape path.
pub fn translate_reshape(
    inputs: &[NodeIndex],
    attrs: &[AttributeProto],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.len() < 2 {
        return Err(OnnxError::InvalidModel("Reshape requires 2 inputs (data, shape)".into()));
    }

    // Check for allowzero attribute (ONNX opset 14+)
    let allow_zero = attrs
        .iter()
        .find(|a| a.name == "allowzero")
        .map(|a| a.i != 0)
        .unwrap_or(false);

    // Get shape from second input
    use hologram_ir::NodeOp;

    let shape_node = builder.graph().node(inputs[1])
        .ok_or_else(|| OnnxError::InvalidModel("Reshape: shape input not found".to_string()))?;

    // Check if shape is constant - if so, use static reshape for optimization
    let new_shape = match &shape_node.op {
        NodeOp::Constant { data } => {
            use hologram_ir::ConstantData;
            match data {
                ConstantData::I64(values) => Some(values.clone()),
                ConstantData::I32(values) => Some(values.iter().map(|&v| v as i64).collect()),
                _ => None,
            }
        }
        _ => None,
    };

    if let Some(shape_values) = new_shape {
        // Static reshape path (optimization when shape is constant)
        // Only use this if there's no -1 or special handling needed
        let has_infer = shape_values.iter().any(|&v| v == -1);
        let has_zero = allow_zero && shape_values.iter().any(|&v| v == 0);

        if !has_infer && !has_zero {
            // Simple static reshape
            tracing::debug!("Reshape: static path, new_shape = {:?}", shape_values);
            let result = builder.reshape(inputs[0], shape_values)?;
            return Ok(vec![result]);
        }
    }

    // Dynamic reshape path (supports runtime shapes, -1 inference, and allowzero)
    tracing::debug!("Reshape: dynamic path, allow_zero = {}", allow_zero);
    let result = builder.reshape_dynamic(inputs[0], inputs[1], allow_zero)?;
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

    let axis_raw = parse_attr_int(attrs, "axis", 0)? as i32;

    // Normalize negative axis (ONNX allows negative axes)
    // Validate all inputs have the same rank first
    let first_node = builder.graph().node(inputs[0])
        .ok_or_else(|| OnnxError::InvalidModel("Concat: first input not found".to_string()))?;
    let rank = first_node.shape.rank() as i32;

    // Check that all inputs have the same rank (ONNX requirement)
    let mut rank_mismatches = Vec::new();
    for (i, &input) in inputs.iter().enumerate() {
        if let Some(node) = builder.graph().node(input) {
            let input_rank = node.shape.rank() as i32;
            if input_rank != rank {
                rank_mismatches.push((i, input_rank));
                tracing::warn!("Concat: Input {} has rank {} but first input has rank {}", i, input_rank, rank);
            }
        }
    }

    // If there are rank mismatches, return an error
    if !rank_mismatches.is_empty() {
        let mismatch_details: Vec<String> = rank_mismatches.iter()
            .map(|(idx, r)| format!("input {} (rank {})", idx, r))
            .collect();
        return Err(OnnxError::InvalidModel(format!(
            "Concat: All inputs must have the same rank. First input has rank {}, but found mismatches: {}",
            rank,
            mismatch_details.join(", ")
        )));
    }

    let axis = if axis_raw < 0 {
        rank + axis_raw
    } else {
        axis_raw
    };

    // Validate axis is in bounds [0, rank)
    if axis < 0 || axis >= rank {
        return Err(OnnxError::InvalidModel(format!(
            "Concat: axis {} (raw: {}) is out of bounds for rank {} tensor (valid range: [0, {}))",
            axis, axis_raw, rank, rank
        )));
    }

    // Debug: check input shapes
    tracing::debug!("Concat: {} inputs, axis_raw = {}, normalized axis = {}, rank = {}", inputs.len(), axis_raw, axis, rank);
    for (i, &input) in inputs.iter().enumerate() {
        if let Some(node) = builder.graph().node(input) {
            tracing::debug!("  Input {}: op = {:?}, shape = {:?}, rank = {}",
                           i, node.op.name(), node.shape, node.shape.rank());
        }
    }

    // Constant folding: if all inputs are constants with same type, concatenate at compile time
    use hologram_ir::{NodeOp, ConstantData, Shape};

    let all_constants = inputs.iter().all(|&idx| {
        if let Some(node) = builder.graph().node(idx) {
            matches!(node.op, NodeOp::Constant { .. })
        } else {
            false
        }
    });

    if all_constants && axis == 0 {
        // Try to fold for 1D tensors concatenated along axis 0
        let first_node = builder.graph().node(inputs[0]).unwrap();
        if first_node.shape.rank() == 1 {
            // Check if all have the same data type
            if let NodeOp::Constant { data: first_data } = &first_node.op {
                match first_data {
                    ConstantData::I64(_) => {
                        let mut result_values = Vec::new();
                        let mut all_i64 = true;

                        for &idx in inputs.iter() {
                            let node = builder.graph().node(idx).unwrap();
                            if let NodeOp::Constant { data } = &node.op {
                                if let ConstantData::I64(values) = data {
                                    result_values.extend_from_slice(values);
                                } else {
                                    all_i64 = false;
                                    break;
                                }
                            } else {
                                all_i64 = false;
                                break;
                            }
                        }

                        if all_i64 {
                            let output_shape = Shape::static_shape(&[result_values.len()]);
                            let result = builder.constant(ConstantData::I64(result_values), output_shape);
                            tracing::debug!("Concat: constant folding succeeded");
                            return Ok(vec![result]);
                        }
                    }
                    ConstantData::I32(_) => {
                        let mut result_values = Vec::new();
                        let mut all_i32 = true;

                        for &idx in inputs.iter() {
                            let node = builder.graph().node(idx).unwrap();
                            if let NodeOp::Constant { data } = &node.op {
                                if let ConstantData::I32(values) = data {
                                    result_values.extend_from_slice(values);
                                } else {
                                    all_i32 = false;
                                    break;
                                }
                            } else {
                                all_i32 = false;
                                break;
                            }
                        }

                        if all_i32 {
                            let output_shape = Shape::static_shape(&[result_values.len()]);
                            let result = builder.constant(ConstantData::I32(result_values), output_shape);
                            tracing::debug!("Concat: constant folding succeeded");
                            return Ok(vec![result]);
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // No constant folding, create regular concat node
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
///
/// Unsqueeze adds dimensions of size 1 to the input tensor at specified axes.
///
/// # ONNX Opset Versions
/// - Opset 1-12: axes specified as attribute
/// - Opset 13+: axes specified as second input tensor
pub fn translate_unsqueeze(
    inputs: &[NodeIndex],
    attrs: &[AttributeProto],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel("Unsqueeze requires at least 1 input".into()));
    }

    let data = inputs[0];

    // Get the data node for potential constant folding
    use hologram_ir::NodeOp;
    let data_node = builder.graph().node(data)
        .ok_or_else(|| OnnxError::InvalidModel("Unsqueeze: input node not found".to_string()))?;

    // Get axes - either from attributes (opset < 13) or from second input (opset >= 13)
    let axes = if inputs.len() >= 2 {
        // Opset 13+: axes is a second input (constant tensor)
        let axes_node = builder.graph().node(inputs[1])
            .ok_or_else(|| OnnxError::InvalidModel("Unsqueeze: axes input not found".to_string()))?;

        // Extract axes from constant
        use hologram_ir::NodeOp;
        if let NodeOp::Constant { data: constant_data } = &axes_node.op {
            use hologram_ir::ConstantData;
            match constant_data {
                ConstantData::I64(values) => values.iter().map(|&v| v as i32).collect(),
                ConstantData::I32(values) => values.clone(),
                _ => return Err(OnnxError::InvalidModel(
                    "Unsqueeze: axes must be int32 or int64".to_string()
                )),
            }
        } else {
            return Err(OnnxError::InvalidModel(
                "Unsqueeze: axes input must be a constant".to_string()
            ));
        }
    } else {
        // Opset < 13: axes is an attribute
        attrs
            .iter()
            .find(|a| a.name == "axes")
            .map(|a| a.ints.iter().map(|&v| v as i32).collect())
            .ok_or_else(|| OnnxError::InvalidModel("Unsqueeze: missing axes attribute".to_string()))?
    };

    // Compute output shape for Unsqueeze
    let input_node = builder.graph().node(data)
        .ok_or_else(|| OnnxError::InvalidModel("Unsqueeze: input node not found".to_string()))?;
    let input_shape = &input_node.shape;
    let input_dtype = input_node.dtype;

    // Build output shape by inserting dimensions of size 1 at specified axes
    use hologram_ir::Dim;
    let input_dims = &input_shape.dims;
    let output_rank = input_dims.len() + axes.len();
    let mut output_dims = Vec::with_capacity(output_rank);

    // Normalize axes to positive values
    let mut normalized_axes: Vec<i32> = axes.iter().map(|&axis| {
        if axis < 0 {
            output_rank as i32 + axis
        } else {
            axis
        }
    }).collect();
    normalized_axes.sort_unstable();

    let mut input_idx = 0;
    let mut axis_idx = 0;

    for out_idx in 0..output_rank {
        if axis_idx < normalized_axes.len() && normalized_axes[axis_idx] == out_idx as i32 {
            // Insert dimension of size 1
            output_dims.push(Dim::Static(1));
            axis_idx += 1;
        } else {
            // Copy from input
            output_dims.push(input_dims[input_idx].clone());
            input_idx += 1;
        }
    }

    let output_shape = hologram_ir::Shape::new(output_dims);
    tracing::debug!("Unsqueeze: axes = {:?}, normalized_axes = {:?}, input shape = {:?} (rank {}), output shape = {:?} (rank {})",
                   axes, normalized_axes, input_shape, input_dims.len(), output_shape, output_rank);

    // Constant folding: if input is a constant, create unsqueezed constant
    if let NodeOp::Constant { data: const_data } = &data_node.op {
        use hologram_ir::ConstantData;

        // For constants, we just need to update the shape - the data stays the same
        // because unsqueeze adds dimensions of size 1
        let folded_data = match const_data {
            ConstantData::I64(values) => Some(ConstantData::I64(values.clone())),
            ConstantData::I32(values) => Some(ConstantData::I32(values.clone())),
            ConstantData::F32(values) => Some(ConstantData::F32(values.clone())),
            ConstantData::F64(values) => Some(ConstantData::F64(values.clone())),
            _ => None,
        };

        if let Some(data) = folded_data {
            tracing::debug!("Unsqueeze: constant folding succeeded");
            let result = builder.constant(data, output_shape);
            return Ok(vec![result]);
        }
    }

    // No constant folding, add unsqueeze node with computed shape
    let result = builder.graph_mut().add_op(
        hologram_ir::NodeOp::Unsqueeze { axes: axes.clone() },
        output_shape,
        input_dtype
    );
    builder.graph_mut().connect(data, result);

    Ok(vec![result])
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
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.len() < 2 {
        return Err(OnnxError::InvalidModel("Expand requires 2 inputs".into()));
    }

    let data = inputs[0];
    let shape = inputs[1];

    // Use hologram-ir's expand operation which properly handles shape inference
    let result = builder.expand(data, shape)?;

    Ok(vec![result])
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

    #[test]
    fn test_translate_unsqueeze_with_attribute() {
        use crate::proto::attribute_proto::AttributeType;
        use crate::proto::AttributeProto;

        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);

        // Unsqueeze at axes [0, 3] - should give shape [1, 2, 3, 1]
        let attrs = vec![AttributeProto {
            name: "axes".to_string(),
            ints: vec![0, 3],
            r#type: AttributeType::Ints as i32,
            ..Default::default()
        }];

        let result = translate_unsqueeze(&[x], &attrs, &mut builder);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), 1);

        // Verify output shape is [1, 2, 3, 1]
        let node = builder.graph().node(output[0]).unwrap();
        assert_eq!(node.shape.rank(), 4);
        assert_eq!(node.shape.dims[0], hologram_ir::Dim::Static(1));
        assert_eq!(node.shape.dims[1], hologram_ir::Dim::Static(2));
        assert_eq!(node.shape.dims[2], hologram_ir::Dim::Static(3));
        assert_eq!(node.shape.dims[3], hologram_ir::Dim::Static(1));
    }

    #[test]
    fn test_translate_unsqueeze_with_input_tensor() {
        use hologram_ir::ConstantData;

        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);

        // Create axes as a constant input (opset 13+)
        let axes = builder.constant(
            ConstantData::I64(vec![1]),
            Shape::static_shape(&[1])
        );

        let result = translate_unsqueeze(&[x, axes], &[], &mut builder);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), 1);

        // Verify output shape is [2, 1, 3]
        let node = builder.graph().node(output[0]).unwrap();
        assert_eq!(node.shape.rank(), 3);
        assert_eq!(node.shape.dims[0], hologram_ir::Dim::Static(2));
        assert_eq!(node.shape.dims[1], hologram_ir::Dim::Static(1));
        assert_eq!(node.shape.dims[2], hologram_ir::Dim::Static(3));
    }

    #[test]
    fn test_translate_unsqueeze_negative_axes() {
        use crate::proto::attribute_proto::AttributeType;
        use crate::proto::AttributeProto;

        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);

        // Unsqueeze at axis -1 (last position) - should give shape [2, 3, 1]
        let attrs = vec![AttributeProto {
            name: "axes".to_string(),
            ints: vec![-1],
            r#type: AttributeType::Ints as i32,
            ..Default::default()
        }];

        let result = translate_unsqueeze(&[x], &attrs, &mut builder);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), 1);

        // Verify output shape is [2, 3, 1]
        let node = builder.graph().node(output[0]).unwrap();
        assert_eq!(node.shape.rank(), 3);
        assert_eq!(node.shape.dims[0], hologram_ir::Dim::Static(2));
        assert_eq!(node.shape.dims[1], hologram_ir::Dim::Static(3));
        assert_eq!(node.shape.dims[2], hologram_ir::Dim::Static(1));
    }

    #[test]
    fn test_translate_unsqueeze_scalar_to_1d() {
        use hologram_ir::ConstantData;

        let mut builder = GraphBuilder::new();
        // Create a scalar (rank-0 tensor)
        let scalar = builder.constant(ConstantData::F32(vec![42.0]), Shape::static_shape(&[]));

        // Unsqueeze at axis 0
        let axes = builder.constant(
            ConstantData::I64(vec![0]),
            Shape::static_shape(&[1])
        );

        let result = translate_unsqueeze(&[scalar, axes], &[], &mut builder);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), 1);

        // Verify output shape is [1]
        let node = builder.graph().node(output[0]).unwrap();
        assert_eq!(node.shape.rank(), 1);
        assert_eq!(node.shape.dims[0], hologram_ir::Dim::Static(1));
    }

    #[test]
    fn test_translate_unsqueeze_missing_axes() {
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);

        // No axes attribute and no second input
        let result = translate_unsqueeze(&[x], &[], &mut builder);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("missing axes"));
    }
}
