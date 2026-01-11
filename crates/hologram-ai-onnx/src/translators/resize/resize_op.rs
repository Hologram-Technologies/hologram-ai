//! Resize operation translator.

use crate::proto::NodeProto;
use crate::translators::{InputRequirement, OnnxAttributes, OnnxTranslator, TranslationError};
use hologram::ir::{CoordinateTransform, GraphBuilder, NodeIndex, ResizeMode};

/// Translator for ONNX Resize operation.
///
/// Resize performs resizing/interpolation with support for multiple modes.
///
/// # ONNX Specification
///
/// - Inputs: X, [roi], [scales], [sizes]
/// - Attributes:
///   - mode (default: "nearest"): "nearest", "linear", "cubic"
///   - coordinate_transformation_mode (default: "half_pixel")
/// - Output: Y
///
/// # Supported Modes
///
/// - nearest: Nearest neighbor interpolation
/// - linear/bilinear: Linear interpolation
/// - cubic: Cubic interpolation
#[derive(Debug, Default)]
pub struct ResizeTranslator;

impl OnnxTranslator for ResizeTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "Resize"
    }

    fn input_requirement(&self) -> InputRequirement {
        InputRequirement::Range(1, 4)
    }

    fn translate(
        &self,
        node: &NodeProto,
        inputs: &[NodeIndex],
        builder: &mut GraphBuilder,
    ) -> Result<Vec<NodeIndex>, TranslationError> {
        let data = inputs[0];

        // Parse mode
        let mode_bytes = node.get_string("mode");
        let mode_str = mode_bytes
            .map(|b| String::from_utf8_lossy(b).to_string())
            .unwrap_or_else(|| "nearest".to_string());

        let mode = match mode_str.as_str() {
            "nearest" => ResizeMode::Nearest,
            "linear" | "bilinear" => ResizeMode::Linear,
            "cubic" => ResizeMode::Cubic,
            _ => {
                return Err(TranslationError::invalid_attribute(
                    "mode",
                    format!("unknown resize mode: {}", mode_str),
                ));
            }
        };

        // Parse coordinate transformation mode
        let coord_mode_bytes = node.get_string("coordinate_transformation_mode");
        let coord_mode_str = coord_mode_bytes
            .map(|b| String::from_utf8_lossy(b).to_string())
            .unwrap_or_else(|| "half_pixel".to_string());

        let coordinate_transform = match coord_mode_str.as_str() {
            "half_pixel" => CoordinateTransform::HalfPixel,
            "asymmetric" => CoordinateTransform::Asymmetric,
            "align_corners" => CoordinateTransform::AlignCorners,
            "pytorch_half_pixel" | "tf_half_pixel_for_nn" => CoordinateTransform::HalfPixel,
            _ => {
                return Err(TranslationError::invalid_attribute(
                    "coordinate_transformation_mode",
                    format!("unknown mode: {}", coord_mode_str),
                ));
            }
        };

        // Parse scales and sizes from attributes
        let scales_attr = node.get_floats("scales");
        let sizes_attr = node.get_ints("sizes");

        // Check for dynamic inputs (opset >= 11)
        if inputs.len() >= 3 && scales_attr.is_none() && sizes_attr.is_none() {
            return Err(TranslationError::unsupported_op("Resize", 13));
        }

        // Convert to Option<Vec>
        let scales: Option<Vec<f64>> = scales_attr
            .filter(|s| !s.is_empty())
            .map(|s| s.iter().map(|&v| v as f64).collect());

        let sizes: Option<Vec<usize>> = sizes_attr
            .filter(|s| !s.is_empty())
            .map(|s| s.iter().map(|&v| v as usize).collect());

        // Validate
        if scales.is_some() && sizes.is_some() {
            return Err(TranslationError::IrBuilder(
                "Resize cannot have both scales and sizes specified".to_string(),
            ));
        }

        if scales.is_none() && sizes.is_none() {
            return Err(TranslationError::IrBuilder(
                "Resize requires either scales or sizes to be specified".to_string(),
            ));
        }

        let result = builder
            .resize(data, scales, sizes, mode, coordinate_transform)
            .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;

        Ok(vec![result])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::AttributeProto;
    use hologram::ir::{DType, Shape};

    fn make_node_with_attrs(
        mode: &str,
        scales: Option<Vec<f32>>,
        sizes: Option<Vec<i64>>,
    ) -> NodeProto {
        let mut attrs = vec![AttributeProto {
            name: "mode".to_string(),
            s: mode.as_bytes().to_vec(),
            ..Default::default()
        }];

        if let Some(s) = scales {
            attrs.push(AttributeProto {
                name: "scales".to_string(),
                floats: s,
                ..Default::default()
            });
        }

        if let Some(s) = sizes {
            attrs.push(AttributeProto {
                name: "sizes".to_string(),
                ints: s,
                ..Default::default()
            });
        }

        NodeProto {
            name: "resize_test".to_string(),
            op_type: "Resize".to_string(),
            attribute: attrs,
            ..Default::default()
        }
    }

    #[test]
    fn test_resize_with_scales() {
        let translator = ResizeTranslator;
        let mut builder = GraphBuilder::new();
        let data = builder.input("data", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);

        let node = make_node_with_attrs("nearest", Some(vec![1.0, 1.0, 2.0, 2.0]), None);
        let result = translator.translate(&node, &[data], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_resize_with_sizes() {
        let translator = ResizeTranslator;
        let mut builder = GraphBuilder::new();
        let data = builder.input("data", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);

        let node = make_node_with_attrs("linear", None, Some(vec![1, 3, 64, 64]));
        let result = translator.translate(&node, &[data], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_resize_cubic() {
        let translator = ResizeTranslator;
        let mut builder = GraphBuilder::new();
        let data = builder.input("data", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);

        let node = make_node_with_attrs("cubic", Some(vec![1.0, 1.0, 0.5, 0.5]), None);
        let result = translator.translate(&node, &[data], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_resize_with_coord_transform() {
        let translator = ResizeTranslator;
        let mut builder = GraphBuilder::new();
        let data = builder.input("data", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);

        let mut node = make_node_with_attrs("linear", Some(vec![1.0, 1.0, 2.0, 2.0]), None);
        node.attribute.push(AttributeProto {
            name: "coordinate_transformation_mode".to_string(),
            s: b"align_corners".to_vec(),
            ..Default::default()
        });

        let result = translator.translate(&node, &[data], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_resize_both_scales_and_sizes_error() {
        let translator = ResizeTranslator;
        let mut builder = GraphBuilder::new();
        let data = builder.input("data", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);

        let node = make_node_with_attrs(
            "nearest",
            Some(vec![1.0, 1.0, 2.0, 2.0]),
            Some(vec![1, 3, 64, 64]),
        );
        let result = translator.translate(&node, &[data], &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_resize_neither_scales_nor_sizes_error() {
        let translator = ResizeTranslator;
        let mut builder = GraphBuilder::new();
        let data = builder.input("data", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);

        let node = make_node_with_attrs("nearest", None, None);
        let result = translator.translate(&node, &[data], &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_resize_invalid_mode() {
        let translator = ResizeTranslator;
        let mut builder = GraphBuilder::new();
        let data = builder.input("data", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);

        let node = make_node_with_attrs("invalid_mode", Some(vec![1.0, 1.0, 2.0, 2.0]), None);
        let result = translator.translate(&node, &[data], &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_resize_input_validation() {
        let translator = ResizeTranslator;

        // 0 inputs should fail
        let err = translator.input_requirement().validate(0, "Resize");
        assert!(err.is_err());

        // 1-4 inputs should pass
        assert!(translator.input_requirement().validate(1, "Resize").is_ok());
        assert!(translator.input_requirement().validate(4, "Resize").is_ok());

        // 5 inputs should fail
        let err = translator.input_requirement().validate(5, "Resize");
        assert!(err.is_err());
    }
}
