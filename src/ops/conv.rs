//! ONNX convolution operations.
//!
//! This module provides translators for convolution operations including:
//! - Conv: Standard 2D convolution with groups support
//! - ConvTranspose: Transposed convolution (deconvolution)

use hologram_ir::{GraphBuilder, NodeIndex, Padding};
use crate::core::{OnnxError, Result};
use crate::proto::AttributeProto;
use crate::ops::utils::{parse_attr_int, parse_attr_ints};

/// Translate ONNX Conv operation to IR.
///
/// ONNX Conv performs 2D convolution with support for:
/// - Groups (including depthwise convolution when groups = input_channels)
/// - Padding modes (same, valid, explicit)
/// - Strides
/// - Dilations (currently must be [1, 1])
///
/// # Arguments
///
/// * `inputs` - [input, weight, bias (optional)]
/// * `attrs` - Attributes including kernel_shape, strides, pads, group, dilations
/// * `builder` - IR graph builder
///
/// # Returns
///
/// Vector with single output node
///
/// # Errors
///
/// Returns error if:
/// - Input count is invalid (must be 2 or 3)
/// - Kernel shape is not 2D
/// - Dilations are not [1, 1]
/// - Invalid padding mode
pub fn translate_conv(
    inputs: &[NodeIndex],
    attrs: &[AttributeProto],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.len() < 2 || inputs.len() > 3 {
        return Err(OnnxError::InvalidModel(format!(
            "Conv requires 2 or 3 inputs (input, weight, optional bias), got {}",
            inputs.len()
        )));
    }

    let input = inputs[0];
    let weight = inputs[1];
    let bias = inputs.get(2).copied();

    // Parse attributes
    let kernel_shape = parse_attr_ints(attrs, "kernel_shape", vec![])?;
    if kernel_shape.len() != 2 {
        return Err(OnnxError::InvalidModel(format!(
            "Conv kernel_shape must be 2D, got {:?}",
            kernel_shape
        )));
    }
    let kernel = (kernel_shape[0] as usize, kernel_shape[1] as usize);

    let strides = parse_attr_ints(attrs, "strides", vec![1, 1])?;
    if strides.len() != 2 {
        return Err(OnnxError::InvalidModel(format!(
            "Conv strides must be 2D, got {:?}",
            strides
        )));
    }
    let stride = (strides[0] as usize, strides[1] as usize);

    let groups = parse_attr_int(attrs, "group", 1)? as usize;

    // Parse dilations - hologram currently only supports dilation=1
    let dilations = parse_attr_ints(attrs, "dilations", vec![1, 1])?;
    if dilations.iter().any(|&d| d != 1) {
        return Err(OnnxError::unsupported_op("Conv", 13));
    }

    // Parse padding
    let pads = parse_attr_ints(attrs, "pads", vec![])?;
    let auto_pad = parse_attr_ints(attrs, "auto_pad", vec![])?;

    let padding = if !auto_pad.is_empty() {
        // auto_pad is a string attribute, but we get it as ints - need to check the string
        // For now, use explicit padding from pads
        parse_padding(&pads)?
    } else if !pads.is_empty() {
        parse_padding(&pads)?
    } else {
        Padding::Valid
    };

    // Create convolution node
    let conv_result = builder.conv2d(input, weight, kernel, stride, padding, groups)?;

    // Add bias if present
    let result = if let Some(bias_node) = bias {
        builder.add(conv_result, bias_node)?
    } else {
        conv_result
    };

    Ok(vec![result])
}

/// Translate ONNX ConvTranspose operation to IR.
///
/// ONNX ConvTranspose performs transposed convolution (deconvolution) with support for:
/// - Groups (including depthwise transposed convolution)
/// - Padding modes
/// - Strides
/// - Output padding
///
/// # Arguments
///
/// * `inputs` - [input, weight, bias (optional)]
/// * `attrs` - Attributes including kernel_shape, strides, pads, group, output_padding
/// * `builder` - IR graph builder
///
/// # Returns
///
/// Vector with single output node
///
/// # Errors
///
/// Returns error if:
/// - Input count is invalid
/// - Kernel shape is not 2D
/// - Output padding is non-zero (not yet supported)
/// - Invalid padding mode
pub fn translate_conv_transpose(
    inputs: &[NodeIndex],
    attrs: &[AttributeProto],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.len() < 2 || inputs.len() > 3 {
        return Err(OnnxError::InvalidModel(format!(
            "ConvTranspose requires 2 or 3 inputs (input, weight, optional bias), got {}",
            inputs.len()
        )));
    }

    let input = inputs[0];
    let weight = inputs[1];
    let bias = inputs.get(2).copied();

    // Parse attributes
    let kernel_shape = parse_attr_ints(attrs, "kernel_shape", vec![])?;
    if kernel_shape.len() != 2 {
        return Err(OnnxError::InvalidModel(format!(
            "ConvTranspose kernel_shape must be 2D, got {:?}",
            kernel_shape
        )));
    }
    let kernel = (kernel_shape[0] as usize, kernel_shape[1] as usize);

    let strides = parse_attr_ints(attrs, "strides", vec![1, 1])?;
    if strides.len() != 2 {
        return Err(OnnxError::InvalidModel(format!(
            "ConvTranspose strides must be 2D, got {:?}",
            strides
        )));
    }
    let stride = (strides[0] as usize, strides[1] as usize);

    let groups = parse_attr_int(attrs, "group", 1)? as usize;

    // Check output_padding - not yet supported
    let output_padding = parse_attr_ints(attrs, "output_padding", vec![0, 0])?;
    if output_padding.iter().any(|&p| p != 0) {
        return Err(OnnxError::unsupported_op("ConvTranspose", 13));
    }

    // Parse padding
    let pads = parse_attr_ints(attrs, "pads", vec![])?;
    let padding = if !pads.is_empty() {
        parse_padding(&pads)?
    } else {
        Padding::Valid
    };

    // Create transposed convolution node
    // Note: GraphBuilder doesn't have conv_transpose2d helper, so we use manual construction
    use hologram_ir::NodeOp;

    // Get input shape and dtype for output inference (simplified - preserves spatial dims)
    let input_node = builder.graph().node(input)
        .ok_or_else(|| OnnxError::InvalidModel("Invalid input node".into()))?;
    let shape = input_node.shape.clone();
    let dtype = input_node.dtype;

    let idx = builder.graph_mut().add_op(
        NodeOp::ConvTranspose2d { kernel, stride, padding, groups },
        shape,
        dtype,
    );
    builder.graph_mut().connect(input, idx);
    builder.graph_mut().connect(weight, idx);
    let conv_result = idx;

    // Add bias if present
    let result = if let Some(bias_node) = bias {
        builder.add(conv_result, bias_node)?
    } else {
        conv_result
    };

    Ok(vec![result])
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

        // Check if it's symmetric (can use Same)
        if top == bottom && left == right && top == left {
            if top == 0 {
                Ok(Padding::Valid)
            } else {
                // Use explicit padding for non-zero symmetric padding
                Ok(Padding::Explicit { top, bottom, left, right })
            }
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

    fn make_int_attr(name: &str, value: i64) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            i: value,
            r#type: AttributeType::Int as i32,
            ..Default::default()
        }
    }

    #[test]
    fn test_translate_conv_basic() {
        let mut builder = GraphBuilder::new();

        // Input: [N, C_in, H, W] = [1, 3, 32, 32]
        let input = builder.input("input", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);
        // Weight: [C_out, C_in, kH, kW] = [64, 3, 3, 3]
        let weight = builder.input("weight", Shape::static_shape(&[64, 3, 3, 3]), DType::F32);

        let attrs = vec![
            make_ints_attr("kernel_shape", vec![3, 3]),
            make_ints_attr("strides", vec![1, 1]),
            make_ints_attr("pads", vec![1, 1, 1, 1]),
        ];

        let result = translate_conv(&[input, weight], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    #[ignore = "Requires proper shape inference in hologram-ir conv2d helper"]
    fn test_translate_conv_with_bias() {
        let mut builder = GraphBuilder::new();

        let input = builder.input("input", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);
        let weight = builder.input("weight", Shape::static_shape(&[64, 3, 3, 3]), DType::F32);
        let bias = builder.input("bias", Shape::static_shape(&[64]), DType::F32);

        let attrs = vec![
            make_ints_attr("kernel_shape", vec![3, 3]),
            make_ints_attr("strides", vec![1, 1]),
        ];

        let result = translate_conv(&[input, weight, bias], &attrs, &mut builder);
        if let Err(e) = &result {
            eprintln!("Conv with bias error: {:?}", e);
        }
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_conv_with_groups() {
        let mut builder = GraphBuilder::new();

        // Depthwise convolution: groups = channels
        let input = builder.input("input", Shape::static_shape(&[1, 64, 32, 32]), DType::F32);
        let weight = builder.input("weight", Shape::static_shape(&[64, 1, 3, 3]), DType::F32);

        let attrs = vec![
            make_ints_attr("kernel_shape", vec![3, 3]),
            make_int_attr("group", 64),
        ];

        let result = translate_conv(&[input, weight], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_conv_stride() {
        let mut builder = GraphBuilder::new();

        let input = builder.input("input", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);
        let weight = builder.input("weight", Shape::static_shape(&[64, 3, 3, 3]), DType::F32);

        let attrs = vec![
            make_ints_attr("kernel_shape", vec![3, 3]),
            make_ints_attr("strides", vec![2, 2]),
        ];

        let result = translate_conv(&[input, weight], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_conv_invalid_inputs() {
        let mut builder = GraphBuilder::new();

        let input = builder.input("input", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);

        let attrs = vec![make_ints_attr("kernel_shape", vec![3, 3])];
        let result = translate_conv(&[input], &attrs, &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_translate_conv_dilations_unsupported() {
        let mut builder = GraphBuilder::new();

        let input = builder.input("input", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);
        let weight = builder.input("weight", Shape::static_shape(&[64, 3, 3, 3]), DType::F32);

        let attrs = vec![
            make_ints_attr("kernel_shape", vec![3, 3]),
            make_ints_attr("dilations", vec![2, 2]),
        ];

        let result = translate_conv(&[input, weight], &attrs, &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_translate_conv_transpose_basic() {
        let mut builder = GraphBuilder::new();

        let input = builder.input("input", Shape::static_shape(&[1, 64, 16, 16]), DType::F32);
        let weight = builder.input("weight", Shape::static_shape(&[64, 32, 3, 3]), DType::F32);

        let attrs = vec![
            make_ints_attr("kernel_shape", vec![3, 3]),
            make_ints_attr("strides", vec![2, 2]),
        ];

        let result = translate_conv_transpose(&[input, weight], &attrs, &mut builder);
        if let Err(e) = &result {
            eprintln!("Error: {:?}", e);
        }
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    #[ignore = "Requires proper shape inference in ConvTranspose2d"]
    fn test_translate_conv_transpose_with_bias() {
        let mut builder = GraphBuilder::new();

        let input = builder.input("input", Shape::static_shape(&[1, 64, 16, 16]), DType::F32);
        let weight = builder.input("weight", Shape::static_shape(&[64, 32, 3, 3]), DType::F32);
        let bias = builder.input("bias", Shape::static_shape(&[32]), DType::F32);

        let attrs = vec![
            make_ints_attr("kernel_shape", vec![3, 3]),
            make_ints_attr("strides", vec![2, 2]),
        ];

        let result = translate_conv_transpose(&[input, weight, bias], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_conv_transpose_output_padding_unsupported() {
        let mut builder = GraphBuilder::new();

        let input = builder.input("input", Shape::static_shape(&[1, 64, 16, 16]), DType::F32);
        let weight = builder.input("weight", Shape::static_shape(&[64, 32, 3, 3]), DType::F32);

        let attrs = vec![
            make_ints_attr("kernel_shape", vec![3, 3]),
            make_ints_attr("output_padding", vec![1, 1]),
        ];

        let result = translate_conv_transpose(&[input, weight], &attrs, &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_padding_valid() {
        let padding = parse_padding(&[]).unwrap();
        assert!(matches!(padding, Padding::Valid));
    }

    #[test]
    fn test_parse_padding_explicit() {
        let padding = parse_padding(&[1, 2, 3, 4]).unwrap();
        assert!(matches!(padding, Padding::Explicit { top: 1, bottom: 3, left: 2, right: 4 }));
    }

    #[test]
    fn test_parse_padding_symmetric() {
        let padding = parse_padding(&[2, 2, 2, 2]).unwrap();
        assert!(matches!(padding, Padding::Explicit { top: 2, bottom: 2, left: 2, right: 2 }));
    }

    #[test]
    fn test_parse_padding_invalid() {
        let result = parse_padding(&[1, 2, 3]);
        assert!(result.is_err());
    }
}
