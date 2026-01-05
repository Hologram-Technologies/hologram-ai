//! ONNX pooling operations.
//!
//! This module provides translators for pooling operations including:
//! - MaxPool: 2D max pooling
//! - AveragePool: 2D average pooling
//! - GlobalAveragePool: Global average pooling
//! - GlobalMaxPool: Global max pooling (returns unsupported error)

use hologram_ir::{GraphBuilder, NodeIndex, Padding};
use crate::core::{OnnxError, Result};
use crate::proto::AttributeProto;
use crate::ops::utils::{parse_attr_ints};

/// Translate ONNX MaxPool operation to IR.
///
/// ONNX MaxPool performs 2D max pooling with support for:
/// - Kernel size
/// - Strides
/// - Padding modes
/// - Dilations (must be [1, 1])
///
/// # Arguments
///
/// * `inputs` - [input]
/// * `attrs` - Attributes including kernel_shape, strides, pads
/// * `builder` - IR graph builder
///
/// # Returns
///
/// Vector with single output node
///
/// # Errors
///
/// Returns error if:
/// - Input count is not 1
/// - Kernel shape is not 2D
/// - Dilations are not [1, 1]
/// - Invalid padding format
pub fn translate_max_pool(
    inputs: &[NodeIndex],
    attrs: &[AttributeProto],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.len() != 1 {
        return Err(OnnxError::InvalidModel(format!(
            "MaxPool requires 1 input, got {}",
            inputs.len()
        )));
    }

    let input = inputs[0];

    // Parse attributes
    let kernel_shape = parse_attr_ints(attrs, "kernel_shape", vec![])?;
    if kernel_shape.len() != 2 {
        return Err(OnnxError::InvalidModel(format!(
            "MaxPool kernel_shape must be 2D, got {:?}",
            kernel_shape
        )));
    }
    let kernel = (kernel_shape[0] as usize, kernel_shape[1] as usize);

    let strides = parse_attr_ints(attrs, "strides", vec![1, 1])?;
    if strides.len() != 2 {
        return Err(OnnxError::InvalidModel(format!(
            "MaxPool strides must be 2D, got {:?}",
            strides
        )));
    }
    let stride = (strides[0] as usize, strides[1] as usize);

    // Parse dilations - hologram currently only supports dilation=1
    let dilations = parse_attr_ints(attrs, "dilations", vec![1, 1])?;
    if dilations.iter().any(|&d| d != 1) {
        return Err(OnnxError::unsupported_op("MaxPool", 13));
    }

    // Parse padding
    let pads = parse_attr_ints(attrs, "pads", vec![])?;
    let padding = parse_padding(&pads)?;

    // Create max pool node
    use hologram_ir::NodeOp;
    let result = builder.unary(NodeOp::MaxPool2d { kernel, stride, padding }, input)?;

    Ok(vec![result])
}

/// Translate ONNX AveragePool operation to IR.
///
/// ONNX AveragePool performs 2D average pooling with support for:
/// - Kernel size
/// - Strides
/// - Padding modes
/// - count_include_pad (not yet supported, must be 0)
///
/// # Arguments
///
/// * `inputs` - [input]
/// * `attrs` - Attributes including kernel_shape, strides, pads
/// * `builder` - IR graph builder
///
/// # Returns
///
/// Vector with single output node
///
/// # Errors
///
/// Returns error if:
/// - Input count is not 1
/// - Kernel shape is not 2D
/// - count_include_pad is not 0
/// - Invalid padding format
pub fn translate_average_pool(
    inputs: &[NodeIndex],
    attrs: &[AttributeProto],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.len() != 1 {
        return Err(OnnxError::InvalidModel(format!(
            "AveragePool requires 1 input, got {}",
            inputs.len()
        )));
    }

    let input = inputs[0];

    // Parse attributes
    let kernel_shape = parse_attr_ints(attrs, "kernel_shape", vec![])?;
    if kernel_shape.len() != 2 {
        return Err(OnnxError::InvalidModel(format!(
            "AveragePool kernel_shape must be 2D, got {:?}",
            kernel_shape
        )));
    }
    let kernel = (kernel_shape[0] as usize, kernel_shape[1] as usize);

    let strides = parse_attr_ints(attrs, "strides", vec![1, 1])?;
    if strides.len() != 2 {
        return Err(OnnxError::InvalidModel(format!(
            "AveragePool strides must be 2D, got {:?}",
            strides
        )));
    }
    let stride = (strides[0] as usize, strides[1] as usize);

    // Parse padding
    let pads = parse_attr_ints(attrs, "pads", vec![])?;
    let padding = parse_padding(&pads)?;

    // Create average pool node
    use hologram_ir::NodeOp;
    let result = builder.unary(NodeOp::AvgPool2d { kernel, stride, padding }, input)?;

    Ok(vec![result])
}

/// Translate ONNX GlobalAveragePool operation to IR.
///
/// ONNX GlobalAveragePool computes the average of all spatial dimensions,
/// producing a tensor with spatial dimensions of size 1.
///
/// # Arguments
///
/// * `inputs` - [input]
/// * `attrs` - (none)
/// * `builder` - IR graph builder
///
/// # Returns
///
/// Vector with single output node
///
/// # Errors
///
/// Returns error if input count is not 1
pub fn translate_global_average_pool(
    inputs: &[NodeIndex],
    _attrs: &[AttributeProto],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.len() != 1 {
        return Err(OnnxError::InvalidModel(format!(
            "GlobalAveragePool requires 1 input, got {}",
            inputs.len()
        )));
    }

    let input = inputs[0];

    // Create global average pool node
    use hologram_ir::NodeOp;
    let result = builder.unary(NodeOp::GlobalAvgPool, input)?;

    Ok(vec![result])
}

/// Translate ONNX GlobalMaxPool operation.
///
/// GlobalMaxPool is not currently supported in hologram-ir.
///
/// # Arguments
///
/// * `inputs` - [input]
/// * `attrs` - (none)
/// * `builder` - IR graph builder
///
/// # Returns
///
/// Unsupported operation error
pub fn translate_global_max_pool(
    _inputs: &[NodeIndex],
    _attrs: &[AttributeProto],
    _builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    Err(OnnxError::unsupported_op("GlobalMaxPool", 13))
}

/// Parse ONNX padding to hologram-ir Padding.
///
/// ONNX padding format: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
/// For 2D: [top, left, bottom, right]
///
/// # Arguments
///
/// * `pads` - ONNX padding array
///
/// # Returns
///
/// Hologram IR Padding enum
///
/// # Errors
///
/// Returns error if padding format is invalid
fn parse_padding(pads: &[i64]) -> Result<Padding> {
    if pads.is_empty() {
        return Ok(Padding::Valid);
    }

    // ONNX format: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
    // For 2D: [top, left, bottom, right]
    if pads.len() == 4 {
        let top = pads[0] as usize;
        let left = pads[1] as usize;
        let bottom = pads[2] as usize;
        let right = pads[3] as usize;

        if top == 0 && left == 0 && bottom == 0 && right == 0 {
            Ok(Padding::Valid)
        } else {
            Ok(Padding::Explicit { top, bottom, left, right })
        }
    } else {
        Err(OnnxError::InvalidModel(format!(
            "Invalid padding array length: expected 4, got {}",
            pads.len()
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::attribute_proto::AttributeType;
    use hologram_ir::{DType, Shape};

    fn make_ints_attr(name: &str, values: Vec<i64>) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            ints: values,
            r#type: AttributeType::Ints as i32,
            ..Default::default()
        }
    }

    #[test]
    fn test_translate_max_pool_basic() {
        let mut builder = GraphBuilder::new();

        let input = builder.input("input", Shape::static_shape(&[1, 64, 32, 32]), DType::F32);

        let attrs = vec![
            make_ints_attr("kernel_shape", vec![2, 2]),
            make_ints_attr("strides", vec![2, 2]),
        ];

        let result = translate_max_pool(&[input], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_max_pool_with_padding() {
        let mut builder = GraphBuilder::new();

        let input = builder.input("input", Shape::static_shape(&[1, 64, 32, 32]), DType::F32);

        let attrs = vec![
            make_ints_attr("kernel_shape", vec![3, 3]),
            make_ints_attr("strides", vec![1, 1]),
            make_ints_attr("pads", vec![1, 1, 1, 1]),
        ];

        let result = translate_max_pool(&[input], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_max_pool_invalid_inputs() {
        let mut builder = GraphBuilder::new();

        let input1 = builder.input("input1", Shape::static_shape(&[1, 64, 32, 32]), DType::F32);
        let input2 = builder.input("input2", Shape::static_shape(&[1, 64, 32, 32]), DType::F32);

        let attrs = vec![make_ints_attr("kernel_shape", vec![2, 2])];
        let result = translate_max_pool(&[input1, input2], &attrs, &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_translate_max_pool_dilations_unsupported() {
        let mut builder = GraphBuilder::new();

        let input = builder.input("input", Shape::static_shape(&[1, 64, 32, 32]), DType::F32);

        let attrs = vec![
            make_ints_attr("kernel_shape", vec![2, 2]),
            make_ints_attr("dilations", vec![2, 2]),
        ];

        let result = translate_max_pool(&[input], &attrs, &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_translate_average_pool_basic() {
        let mut builder = GraphBuilder::new();

        let input = builder.input("input", Shape::static_shape(&[1, 64, 32, 32]), DType::F32);

        let attrs = vec![
            make_ints_attr("kernel_shape", vec![2, 2]),
            make_ints_attr("strides", vec![2, 2]),
        ];

        let result = translate_average_pool(&[input], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_average_pool_with_padding() {
        let mut builder = GraphBuilder::new();

        let input = builder.input("input", Shape::static_shape(&[1, 64, 32, 32]), DType::F32);

        let attrs = vec![
            make_ints_attr("kernel_shape", vec![3, 3]),
            make_ints_attr("strides", vec![1, 1]),
            make_ints_attr("pads", vec![1, 1, 1, 1]),
        ];

        let result = translate_average_pool(&[input], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_global_average_pool() {
        let mut builder = GraphBuilder::new();

        let input = builder.input("input", Shape::static_shape(&[1, 64, 32, 32]), DType::F32);

        let result = translate_global_average_pool(&[input], &[], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_global_average_pool_invalid_inputs() {
        let mut builder = GraphBuilder::new();

        let input1 = builder.input("input1", Shape::static_shape(&[1, 64, 32, 32]), DType::F32);
        let input2 = builder.input("input2", Shape::static_shape(&[1, 64, 32, 32]), DType::F32);

        let result = translate_global_average_pool(&[input1, input2], &[], &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_translate_global_max_pool_unsupported() {
        let mut builder = GraphBuilder::new();

        let input = builder.input("input", Shape::static_shape(&[1, 64, 32, 32]), DType::F32);

        let result = translate_global_max_pool(&[input], &[], &mut builder);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), OnnxError::UnsupportedOp { .. }));
    }

    #[test]
    fn test_parse_padding_valid() {
        let padding = parse_padding(&[]).unwrap();
        assert!(matches!(padding, Padding::Valid));

        let padding = parse_padding(&[0, 0, 0, 0]).unwrap();
        assert!(matches!(padding, Padding::Valid));
    }

    #[test]
    fn test_parse_padding_explicit() {
        let padding = parse_padding(&[1, 2, 3, 4]).unwrap();
        assert!(matches!(padding, Padding::Explicit { top: 1, bottom: 3, left: 2, right: 4 }));
    }

    #[test]
    fn test_parse_padding_invalid() {
        let result = parse_padding(&[1, 2, 3]);
        assert!(result.is_err());
    }
}
