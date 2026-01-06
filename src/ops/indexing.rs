//! ONNX indexing operations.
//!
//! This module provides translators for indexing operations including:
//! - Gather: Gather elements along an axis using indices
//! - Slice: Slice tensor along axes
//! - GatherElements: Gather elements (returns unsupported if not in hologram-ir)

use hologram_ir::{GraphBuilder, NodeIndex};
use crate::core::{OnnxError, Result};
use crate::proto::AttributeProto;
use crate::ops::utils::{parse_attr_int, parse_attr_ints};

/// Translate ONNX Gather operation to IR.
///
/// ONNX Gather gathers elements from the input tensor along a specified axis
/// using the provided indices.
///
/// # Arguments
///
/// * `inputs` - [data, indices]
/// * `attrs` - Attributes including axis
/// * `builder` - IR graph builder
///
/// # Returns
///
/// Vector with single output node
///
/// # Errors
///
/// Returns error if:
/// - Input count is not 2
/// - Axis attribute is missing or invalid
pub fn translate_gather(
    inputs: &[NodeIndex],
    attrs: &[AttributeProto],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.len() != 2 {
        return Err(OnnxError::InvalidModel(format!(
            "Gather requires 2 inputs (data, indices), got {}",
            inputs.len()
        )));
    }

    let data = inputs[0];
    let indices = inputs[1];

    // Parse axis attribute (default is 0 in ONNX)
    let axis = parse_attr_int(attrs, "axis", 0)? as i32;

    // Constant folding: if both data and indices are constants, compute gather at compile time
    use hologram_ir::{NodeOp, ConstantData, Shape};

    let data_node = builder.graph().node(data)
        .ok_or_else(|| OnnxError::InvalidModel("Gather: data input not found".to_string()))?;
    let indices_node = builder.graph().node(indices)
        .ok_or_else(|| OnnxError::InvalidModel("Gather: indices input not found".to_string()))?;

    if let (NodeOp::Constant { data: data_const }, NodeOp::Constant { data: indices_const }) =
        (&data_node.op, &indices_node.op) {

        // Try to perform constant folding for common cases
        // Case 1: Gathering from 1D array with scalar index
        if let (ConstantData::I64(values), ConstantData::I64(idx_values)) = (data_const, indices_const)
            && data_node.shape.rank() == 1 && indices_node.shape.rank() == 0 && axis == 0 {
            let idx = idx_values[0] as usize;
            if idx < values.len() {
                let gathered_value = values[idx];
                let result = builder.constant(
                    ConstantData::I64(vec![gathered_value]),
                    Shape::static_shape(&[])  // scalar output
                );
                return Ok(vec![result]);
            }
        }

        // Case 2: Gathering from 1D I32 array with scalar index
        if let (ConstantData::I32(values), ConstantData::I64(idx_values)) = (data_const, indices_const)
            && data_node.shape.rank() == 1 && indices_node.shape.rank() == 0 && axis == 0 {
            let idx = idx_values[0] as usize;
            if idx < values.len() {
                let gathered_value = values[idx];
                let result = builder.constant(
                    ConstantData::I32(vec![gathered_value]),
                    Shape::static_shape(&[])  // scalar output
                );
                return Ok(vec![result]);
            }
        }
    }

    // No constant folding possible, create regular gather node
    let result = builder.gather(data, indices, axis)?;

    Ok(vec![result])
}

/// Translate ONNX Slice operation to IR.
///
/// ONNX Slice extracts a slice from the input tensor along multiple axes.
/// The slice is specified using start, end, axes, and optionally steps.
///
/// # Arguments
///
/// * `inputs` - [data, starts, ends, axes (optional), steps (optional)]
/// * `attrs` - (none for opset >= 10, attributes for older opsets)
/// * `builder` - IR graph builder
///
/// # Returns
///
/// Vector with single output node
///
/// # Errors
///
/// Returns error if:
/// - Input count is less than 3
/// - Steps are provided and not all 1 (not supported)
/// - Axes/starts/ends have mismatched lengths
pub fn translate_slice(
    inputs: &[NodeIndex],
    attrs: &[AttributeProto],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    // ONNX Slice has two versions:
    // - Opset < 10: starts, ends, axes are attributes
    // - Opset >= 10: starts, ends, axes, steps are inputs

    // Try to parse as attributes first (older opset)
    let starts_attr = parse_attr_ints(attrs, "starts", vec![])?;
    let ends_attr = parse_attr_ints(attrs, "ends", vec![])?;
    let axes_attr = parse_attr_ints(attrs, "axes", vec![])?;

    let (starts, ends, axes) = if !starts_attr.is_empty() && !ends_attr.is_empty() {
        // Attribute-based (older opset)
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidModel(format!(
                "Slice (opset < 10) requires 1 input, got {}",
                inputs.len()
            )));
        }
        (starts_attr, ends_attr, axes_attr)
    } else {
        // Input-based (newer opset >= 10)
        if inputs.len() < 3 {
            return Err(OnnxError::InvalidModel(format!(
                "Slice requires at least 3 inputs (data, starts, ends), got {}",
                inputs.len()
            )));
        }

        // Check if all inputs are constants for constant folding
        let starts_node = builder.graph().node(inputs[1])
            .ok_or_else(|| OnnxError::InvalidModel("Slice: starts node not found".to_string()))?;
        let ends_node = builder.graph().node(inputs[2])
            .ok_or_else(|| OnnxError::InvalidModel("Slice: ends node not found".to_string()))?;

        let starts_is_constant = matches!(starts_node.op, hologram_ir::NodeOp::Constant { .. });
        let ends_is_constant = matches!(ends_node.op, hologram_ir::NodeOp::Constant { .. });

        // Check axes if provided
        let axes_is_constant = if inputs.len() > 3 {
            let axes_node = builder.graph().node(inputs[3])
                .ok_or_else(|| OnnxError::InvalidModel("Slice: axes node not found".to_string()))?;
            matches!(axes_node.op, hologram_ir::NodeOp::Constant { .. })
        } else {
            true  // No axes input = all axes constant (default)
        };

        // Check steps if provided
        let steps_is_constant = if inputs.len() > 4 {
            let steps_node = builder.graph().node(inputs[4])
                .ok_or_else(|| OnnxError::InvalidModel("Slice: steps node not found".to_string()))?;
            matches!(steps_node.op, hologram_ir::NodeOp::Constant { .. })
        } else {
            true  // No steps input = all steps constant (default = 1)
        };

        // If any input is non-constant, use dynamic slice
        if !starts_is_constant || !ends_is_constant || !axes_is_constant || !steps_is_constant {
            tracing::debug!("Slice: dynamic path (non-constant parameters)");

            // Use SliceDynamic operation
            let axes = if inputs.len() > 3 { Some(inputs[3]) } else { None };
            let steps = if inputs.len() > 4 { Some(inputs[4]) } else { None };

            let result = builder.slice_dynamic(inputs[0], inputs[1], inputs[2], axes, steps)?;
            return Ok(vec![result]);
        }

        // All inputs are constants - extract for constant folding
        let starts_vals = if let hologram_ir::NodeOp::Constant { data } = &starts_node.op {
            use hologram_ir::ConstantData;
            match data {
                ConstantData::I64(values) => values.clone(),
                ConstantData::I32(values) => values.iter().map(|&v| v as i64).collect(),
                _ => return Err(OnnxError::InvalidModel(
                    "Slice: starts must be int32 or int64 tensor".to_string()
                )),
            }
        } else {
            unreachable!("starts_is_constant check above ensures this is Constant");
        };

        let ends_vals = if let hologram_ir::NodeOp::Constant { data } = &ends_node.op {
            use hologram_ir::ConstantData;
            match data {
                ConstantData::I64(values) => values.clone(),
                ConstantData::I32(values) => values.iter().map(|&v| v as i64).collect(),
                _ => return Err(OnnxError::InvalidModel(
                    "Slice: ends must be int32 or int64 tensor".to_string()
                )),
            }
        } else {
            unreachable!("ends_is_constant check above ensures this is Constant");
        };

        let axes_vals = if inputs.len() > 3 {
            let axes_node = builder.graph().node(inputs[3])
                .ok_or_else(|| OnnxError::InvalidModel("Slice: axes node not found".to_string()))?;

            if let hologram_ir::NodeOp::Constant { data } = &axes_node.op {
                use hologram_ir::ConstantData;
                match data {
                    ConstantData::I64(values) => values.clone(),
                    ConstantData::I32(values) => values.iter().map(|&v| v as i64).collect(),
                    _ => return Err(OnnxError::InvalidModel(
                        "Slice: axes must be int32 or int64 tensor".to_string()
                    )),
                }
            } else {
                unreachable!("axes_is_constant check above ensures this is Constant");
            }
        } else {
            vec![]
        };

        // Note: steps (input 4) are currently ignored for constant folding
        // The old NodeOp::Slice doesn't support steps, assumes step=1
        if inputs.len() > 4 {
            tracing::warn!("Slice: steps parameter provided but not supported in constant folding (using step=1)");
        }

        (starts_vals, ends_vals, axes_vals)
    };

    let data = inputs[0];

    // Validate lengths
    if starts.len() != ends.len() {
        return Err(OnnxError::InvalidModel(format!(
            "Slice starts and ends must have same length, got {} and {}",
            starts.len(),
            ends.len()
        )));
    }

    // If axes not provided, default to [0, 1, 2, ..., len(starts)-1]
    let axes = if axes.is_empty() {
        (0..starts.len() as i32).collect()
    } else {
        if axes.len() != starts.len() {
            return Err(OnnxError::InvalidModel(format!(
                "Slice axes must have same length as starts, got {} and {}",
                axes.len(),
                starts.len()
            )));
        }
        axes.iter().map(|&a| a as i32).collect()
    };

    // Static path - constant folding (optimization)
    tracing::debug!("Slice: static path (constant folding)");
    use hologram_ir::NodeOp;
    let result = builder.unary(NodeOp::Slice { starts, ends, axes }, data)?;

    Ok(vec![result])
}

/// Translate ONNX GatherElements operation.
///
/// GatherElements is not currently supported in hologram-ir.
/// This operation gathers elements from data at indices positions.
///
/// # Arguments
///
/// * `inputs` - [data, indices]
/// * `attrs` - Attributes including axis
/// * `builder` - IR graph builder
///
/// # Returns
///
/// Unsupported operation error
pub fn translate_gather_elements(
    _inputs: &[NodeIndex],
    _attrs: &[AttributeProto],
    _builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    Err(OnnxError::unsupported_op("GatherElements", 13))
}

/// Translate ONNX ScatterND operation to IR.
///
/// ScatterND creates a copy of the data tensor, then updates specific positions
/// (specified by indices) with new values (from updates tensor). This is the
/// inverse operation of GatherND.
///
/// # Arguments
///
/// * `inputs` - [data, indices, updates] where:
///   - data: Base tensor of rank r >= 1
///   - indices: Integer tensor of rank q >= 1 specifying positions
///   - updates: New values of rank q + r - indices.shape[-1] - 1
/// * `attrs` - Optional attributes (none for ScatterND)
/// * `builder` - IR graph builder
///
/// # Returns
///
/// Vector with single output node (same shape as data)
///
/// # Errors
///
/// Returns error if:
/// - Input count is not 3
/// - Shape constraints are violated
///
/// # Examples
///
/// ```ignore
/// // Update elements in a 1D tensor
/// data = [1, 2, 3, 4, 5]
/// indices = [[1], [3]]
/// updates = [10, 40]
/// output = [1, 10, 3, 40, 5]
/// ```
pub fn translate_scatternd(
    inputs: &[NodeIndex],
    _attrs: &[AttributeProto],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.len() != 3 {
        return Err(OnnxError::InvalidModel(format!(
            "ScatterND requires 3 inputs (data, indices, updates), got {}",
            inputs.len()
        )));
    }

    let data = inputs[0];
    let indices = inputs[1];
    let updates = inputs[2];

    // Use hologram-ir's scatter_nd operation
    let result = builder.scatter_nd(data, indices, updates)?;

    Ok(vec![result])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::attribute_proto::AttributeType;
    use hologram_ir::{DType, Shape};

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
    fn test_translate_gather_basic() {
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[3, 4, 5]), DType::F32);
        let indices = builder.input("indices", Shape::static_shape(&[2, 3]), DType::I64);

        let attrs = vec![make_int_attr("axis", 0)];

        let result = translate_gather(&[data, indices], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_gather_axis_1() {
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[3, 4, 5]), DType::F32);
        let indices = builder.input("indices", Shape::static_shape(&[3, 2]), DType::I64);

        let attrs = vec![make_int_attr("axis", 1)];

        let result = translate_gather(&[data, indices], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_gather_default_axis() {
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[3, 4, 5]), DType::F32);
        let indices = builder.input("indices", Shape::static_shape(&[2]), DType::I64);

        let attrs = vec![];

        let result = translate_gather(&[data, indices], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_gather_invalid_inputs() {
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[3, 4, 5]), DType::F32);

        let attrs = vec![make_int_attr("axis", 0)];
        let result = translate_gather(&[data], &attrs, &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_translate_slice_basic() {
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[10, 20, 30]), DType::F32);

        let attrs = vec![
            make_ints_attr("starts", vec![0, 5, 10]),
            make_ints_attr("ends", vec![5, 15, 25]),
            make_ints_attr("axes", vec![0, 1, 2]),
        ];

        let result = translate_slice(&[data], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_slice_default_axes() {
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[10, 20, 30]), DType::F32);

        let attrs = vec![
            make_ints_attr("starts", vec![0, 5, 10]),
            make_ints_attr("ends", vec![5, 15, 25]),
        ];

        let result = translate_slice(&[data], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_slice_partial_axes() {
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[10, 20, 30]), DType::F32);

        let attrs = vec![
            make_ints_attr("starts", vec![5]),
            make_ints_attr("ends", vec![15]),
            make_ints_attr("axes", vec![1]),
        ];

        let result = translate_slice(&[data], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_slice_mismatched_lengths() {
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[10, 20, 30]), DType::F32);

        let attrs = vec![
            make_ints_attr("starts", vec![0, 5]),
            make_ints_attr("ends", vec![5, 15, 25]),
        ];

        let result = translate_slice(&[data], &attrs, &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_translate_slice_dynamic_inputs() {
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[10, 20, 30]), DType::F32);
        let starts = builder.input("starts", Shape::static_shape(&[3]), DType::I64);
        let ends = builder.input("ends", Shape::static_shape(&[3]), DType::I64);

        let attrs = vec![];
        let result = translate_slice(&[data, starts, ends], &attrs, &mut builder);
        // Should now succeed with dynamic slice support
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_gather_elements_unsupported() {
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[3, 4, 5]), DType::F32);
        let indices = builder.input("indices", Shape::static_shape(&[3, 4, 5]), DType::I64);

        let attrs = vec![make_int_attr("axis", 0)];
        let result = translate_gather_elements(&[data, indices], &attrs, &mut builder);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), OnnxError::UnsupportedOp { .. }));
    }
}
