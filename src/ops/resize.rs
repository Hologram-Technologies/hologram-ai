//! ONNX resize and spatial transformation operations.
//!
//! This module provides translators for resize and spatial operations including:
//! - Resize: Resize/interpolate operation
//! - Upsample: Legacy upsample operation (deprecated, redirects to Resize)
//! - DepthToSpace: Rearrange depth to spatial dimensions (not supported)
//! - SpaceToDepth: Rearrange spatial to depth dimensions (not supported)

use hologram_ir::{GraphBuilder, NodeIndex, ResizeMode, CoordinateTransform};
use crate::core::{OnnxError, Result};
use crate::proto::AttributeProto;
use crate::ops::utils::{parse_attr_string_or, parse_attr_floats, parse_attr_ints};

/// Translate ONNX Resize operation to IR.
///
/// ONNX Resize performs resizing/interpolation with support for:
/// - Multiple interpolation modes (nearest, linear, cubic)
/// - Coordinate transformation modes (half_pixel, asymmetric, align_corners)
/// - Scales or sizes specification
///
/// # Arguments
///
/// * `inputs` - [X, roi (optional), scales (optional), sizes (optional)]
/// * `attrs` - Attributes including mode, coordinate_transformation_mode
/// * `builder` - IR graph builder
///
/// # Returns
///
/// Vector with single output node
///
/// # Errors
///
/// Returns error if:
/// - Input count is less than 1
/// - Both scales and sizes are provided
/// - Neither scales nor sizes are provided
/// - Unsupported mode or coordinate transformation
pub fn translate_resize(
    inputs: &[NodeIndex],
    attrs: &[AttributeProto],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel(
            "Resize requires at least 1 input".to_string()
        ));
    }

    let data = inputs[0];

    // Parse mode (default is "nearest")
    let mode_str = parse_attr_string_or(attrs, "mode", "nearest")?;
    let mode = match mode_str.as_str() {
        "nearest" => ResizeMode::Nearest,
        "linear" | "bilinear" => ResizeMode::Linear,
        "cubic" => ResizeMode::Cubic,
        _ => {
            return Err(OnnxError::InvalidModel(format!(
                "Unknown resize mode: {}",
                mode_str
            )));
        }
    };

    // Parse coordinate transformation mode (default is "half_pixel")
    let coord_mode_str = parse_attr_string_or(attrs, "coordinate_transformation_mode", "half_pixel")?;
    let coordinate_transform = match coord_mode_str.as_str() {
        "half_pixel" => CoordinateTransform::HalfPixel,
        "asymmetric" => CoordinateTransform::Asymmetric,
        "align_corners" => CoordinateTransform::AlignCorners,
        "pytorch_half_pixel" | "tf_half_pixel_for_nn" => {
            // These are variations that we'll approximate with HalfPixel
            CoordinateTransform::HalfPixel
        }
        _ => {
            return Err(OnnxError::InvalidModel(format!(
                "Unknown coordinate transformation mode: {}",
                coord_mode_str
            )));
        }
    };

    // Parse scales and sizes
    // For now, we only support attribute-based specification
    // A full implementation would extract from input tensors
    let scales_attr = parse_attr_floats(attrs, "scales", vec![])?;
    let sizes_attr = parse_attr_ints(attrs, "sizes", vec![])?;

    // Check for dynamic inputs (opset >= 11)
    if inputs.len() >= 3 {
        // Dynamic scales/sizes from inputs - not yet fully supported
        // We'll try to use attributes as fallback
        if scales_attr.is_empty() && sizes_attr.is_empty() {
            return Err(OnnxError::unsupported_op("Resize", 13));
        }
    }

    // Convert scales to f64
    let scales = if !scales_attr.is_empty() {
        Some(scales_attr.into_iter().map(|s| s as f64).collect())
    } else {
        None
    };

    // Convert sizes to usize
    let sizes = if !sizes_attr.is_empty() {
        Some(sizes_attr.into_iter().map(|s| s as usize).collect())
    } else {
        None
    };

    // Validate that exactly one of scales or sizes is provided
    if scales.is_some() && sizes.is_some() {
        return Err(OnnxError::InvalidModel(
            "Resize cannot have both scales and sizes specified".to_string()
        ));
    }

    if scales.is_none() && sizes.is_none() {
        return Err(OnnxError::InvalidModel(
            "Resize requires either scales or sizes to be specified".to_string()
        ));
    }

    // Create resize node
    let result = builder.resize(data, scales, sizes, mode, coordinate_transform)?;

    Ok(vec![result])
}

/// Translate ONNX Upsample operation to IR.
///
/// Upsample is a deprecated operation that has been replaced by Resize.
/// This function redirects to Resize with appropriate parameters.
///
/// # Arguments
///
/// * `inputs` - [X, scales]
/// * `attrs` - Attributes including mode
/// * `builder` - IR graph builder
///
/// # Returns
///
/// Vector with single output node
///
/// # Errors
///
/// Returns error if parameters are invalid
pub fn translate_upsample(
    inputs: &[NodeIndex],
    attrs: &[AttributeProto],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel(
            "Upsample requires at least 1 input".to_string()
        ));
    }

    let data = inputs[0];

    // Parse mode (default is "nearest")
    let mode_str = parse_attr_string_or(attrs, "mode", "nearest")?;
    let mode = match mode_str.as_str() {
        "nearest" => ResizeMode::Nearest,
        "linear" | "bilinear" => ResizeMode::Linear,
        _ => {
            return Err(OnnxError::InvalidModel(format!(
                "Unknown upsample mode: {}",
                mode_str
            )));
        }
    };

    // Parse scales
    let scales_attr = parse_attr_floats(attrs, "scales", vec![])?;

    if inputs.len() >= 2 {
        // Dynamic scales from input - not yet fully supported
        if scales_attr.is_empty() {
            return Err(OnnxError::unsupported_op("Upsample", 13));
        }
    }

    if scales_attr.is_empty() {
        return Err(OnnxError::InvalidModel(
            "Upsample requires scales to be specified".to_string()
        ));
    }

    let scales = scales_attr.into_iter().map(|s| s as f64).collect();

    // Upsample uses asymmetric coordinate transformation by default
    let coordinate_transform = CoordinateTransform::Asymmetric;

    // Create resize node (Upsample is just Resize with scales)
    let result = builder.resize(data, Some(scales), None, mode, coordinate_transform)?;

    Ok(vec![result])
}

/// Translate ONNX DepthToSpace operation.
///
/// DepthToSpace is not currently supported in hologram-ir.
///
/// # Arguments
///
/// * `inputs` - [input]
/// * `attrs` - Attributes including blocksize
/// * `builder` - IR graph builder
///
/// # Returns
///
/// Unsupported operation error
pub fn translate_depth_to_space(
    _inputs: &[NodeIndex],
    _attrs: &[AttributeProto],
    _builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    Err(OnnxError::unsupported_op("DepthToSpace", 13))
}

/// Translate ONNX SpaceToDepth operation.
///
/// SpaceToDepth is not currently supported in hologram-ir.
///
/// # Arguments
///
/// * `inputs` - [input]
/// * `attrs` - Attributes including blocksize
/// * `builder` - IR graph builder
///
/// # Returns
///
/// Unsupported operation error
pub fn translate_space_to_depth(
    _inputs: &[NodeIndex],
    _attrs: &[AttributeProto],
    _builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    Err(OnnxError::unsupported_op("SpaceToDepth", 13))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::attribute_proto::AttributeType;
    use hologram_ir::{DType, Shape};

    fn make_string_attr(name: &str, value: &str) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            s: value.as_bytes().to_vec(),
            r#type: AttributeType::String as i32,
            ..Default::default()
        }
    }

    fn make_floats_attr(name: &str, values: Vec<f32>) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            floats: values,
            r#type: AttributeType::Floats as i32,
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
    fn test_translate_resize_with_scales() {
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);

        let attrs = vec![
            make_string_attr("mode", "nearest"),
            make_floats_attr("scales", vec![1.0, 1.0, 2.0, 2.0]),
        ];

        let result = translate_resize(&[data], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_resize_with_sizes() {
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);

        let attrs = vec![
            make_string_attr("mode", "linear"),
            make_ints_attr("sizes", vec![1, 3, 64, 64]),
        ];

        let result = translate_resize(&[data], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_resize_linear_mode() {
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);

        let attrs = vec![
            make_string_attr("mode", "linear"),
            make_floats_attr("scales", vec![1.0, 1.0, 2.0, 2.0]),
        ];

        let result = translate_resize(&[data], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_resize_cubic_mode() {
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);

        let attrs = vec![
            make_string_attr("mode", "cubic"),
            make_floats_attr("scales", vec![1.0, 1.0, 0.5, 0.5]),
        ];

        let result = translate_resize(&[data], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_resize_coordinate_transform_modes() {
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);

        // Test half_pixel
        let attrs = vec![
            make_string_attr("mode", "linear"),
            make_string_attr("coordinate_transformation_mode", "half_pixel"),
            make_floats_attr("scales", vec![1.0, 1.0, 2.0, 2.0]),
        ];
        let result = translate_resize(&[data], &attrs, &mut builder);
        assert!(result.is_ok());

        // Test asymmetric
        let attrs = vec![
            make_string_attr("mode", "linear"),
            make_string_attr("coordinate_transformation_mode", "asymmetric"),
            make_floats_attr("scales", vec![1.0, 1.0, 2.0, 2.0]),
        ];
        let result = translate_resize(&[data], &attrs, &mut builder);
        assert!(result.is_ok());

        // Test align_corners
        let attrs = vec![
            make_string_attr("mode", "linear"),
            make_string_attr("coordinate_transformation_mode", "align_corners"),
            make_floats_attr("scales", vec![1.0, 1.0, 2.0, 2.0]),
        ];
        let result = translate_resize(&[data], &attrs, &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_resize_both_scales_and_sizes_error() {
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);

        let attrs = vec![
            make_string_attr("mode", "nearest"),
            make_floats_attr("scales", vec![1.0, 1.0, 2.0, 2.0]),
            make_ints_attr("sizes", vec![1, 3, 64, 64]),
        ];

        let result = translate_resize(&[data], &attrs, &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_translate_resize_no_scales_or_sizes_error() {
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);

        let attrs = vec![make_string_attr("mode", "nearest")];

        let result = translate_resize(&[data], &attrs, &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_translate_upsample_nearest() {
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);

        let attrs = vec![
            make_string_attr("mode", "nearest"),
            make_floats_attr("scales", vec![1.0, 1.0, 2.0, 2.0]),
        ];

        let result = translate_upsample(&[data], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_upsample_linear() {
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);

        let attrs = vec![
            make_string_attr("mode", "linear"),
            make_floats_attr("scales", vec![1.0, 1.0, 2.0, 2.0]),
        ];

        let result = translate_upsample(&[data], &attrs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_translate_upsample_missing_scales() {
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);

        let attrs = vec![make_string_attr("mode", "nearest")];

        let result = translate_upsample(&[data], &attrs, &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_translate_depth_to_space_unsupported() {
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[1, 12, 16, 16]), DType::F32);

        let result = translate_depth_to_space(&[data], &[], &mut builder);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), OnnxError::UnsupportedOp { .. }));
    }

    #[test]
    fn test_translate_space_to_depth_unsupported() {
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);

        let result = translate_space_to_depth(&[data], &[], &mut builder);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), OnnxError::UnsupportedOp { .. }));
    }
}
