//! Upsample operation translator.

use hologram::ir::{GraphBuilder, NodeIndex, ResizeMode, CoordinateTransform};
use crate::proto::NodeProto;
use crate::translators::{OnnxTranslator, OnnxAttributes, InputRequirement, TranslationError};

/// Translator for ONNX Upsample operation.
///
/// Upsample is a deprecated operation that has been replaced by Resize.
/// This translator redirects to Resize with appropriate parameters.
///
/// # ONNX Specification
///
/// - Inputs: X, [scales]
/// - Attributes:
///   - mode (default: "nearest"): "nearest" or "linear"
///   - scales: scale factors (for older opsets)
/// - Output: Y
#[derive(Debug, Default)]
pub struct UpsampleTranslator;

impl OnnxTranslator for UpsampleTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "Upsample"
    }

    fn input_requirement(&self) -> InputRequirement {
        InputRequirement::Range(1, 2)
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
            _ => {
                return Err(TranslationError::invalid_attribute(
                    "mode",
                    format!("unknown upsample mode: {}", mode_str)
                ));
            }
        };

        // Parse scales from attribute
        let scales_attr = node.get_floats("scales");

        // Check for dynamic scales input (opset >= 9)
        if inputs.len() >= 2 && scales_attr.is_none() {
            return Err(TranslationError::unsupported_op("Upsample", 13));
        }

        let scales: Option<Vec<f64>> = scales_attr
            .filter(|s| !s.is_empty())
            .map(|s| s.iter().map(|&v| v as f64).collect());

        if scales.is_none() {
            return Err(TranslationError::missing_attribute("Upsample", "scales"));
        }

        // Upsample uses asymmetric coordinate transformation by default
        let coordinate_transform = CoordinateTransform::Asymmetric;

        let result = builder
            .resize(data, scales, None, mode, coordinate_transform)
            .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;

        Ok(vec![result])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram::ir::{DType, Shape};
    use crate::proto::AttributeProto;

    fn make_node_with_attrs(mode: &str, scales: Option<Vec<f32>>) -> NodeProto {
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

        NodeProto {
            name: "upsample_test".to_string(),
            op_type: "Upsample".to_string(),
            attribute: attrs,
            ..Default::default()
        }
    }

    #[test]
    fn test_upsample_nearest() {
        let translator = UpsampleTranslator;
        let mut builder = GraphBuilder::new();
        let data = builder.input("data", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);

        let node = make_node_with_attrs("nearest", Some(vec![1.0, 1.0, 2.0, 2.0]));
        let result = translator.translate(&node, &[data], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_upsample_linear() {
        let translator = UpsampleTranslator;
        let mut builder = GraphBuilder::new();
        let data = builder.input("data", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);

        let node = make_node_with_attrs("linear", Some(vec![1.0, 1.0, 2.0, 2.0]));
        let result = translator.translate(&node, &[data], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_upsample_bilinear() {
        let translator = UpsampleTranslator;
        let mut builder = GraphBuilder::new();
        let data = builder.input("data", Shape::static_shape(&[1, 3, 16, 16]), DType::F32);

        let node = make_node_with_attrs("bilinear", Some(vec![1.0, 1.0, 4.0, 4.0]));
        let result = translator.translate(&node, &[data], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_upsample_missing_scales() {
        let translator = UpsampleTranslator;
        let mut builder = GraphBuilder::new();
        let data = builder.input("data", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);

        let node = make_node_with_attrs("nearest", None);
        let result = translator.translate(&node, &[data], &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_upsample_invalid_mode() {
        let translator = UpsampleTranslator;
        let mut builder = GraphBuilder::new();
        let data = builder.input("data", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);

        let node = make_node_with_attrs("cubic", Some(vec![1.0, 1.0, 2.0, 2.0]));
        let result = translator.translate(&node, &[data], &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_upsample_input_validation() {
        let translator = UpsampleTranslator;

        // 0 inputs should fail
        let err = translator.input_requirement().validate(0, "Upsample");
        assert!(err.is_err());

        // 1-2 inputs should pass
        assert!(translator.input_requirement().validate(1, "Upsample").is_ok());
        assert!(translator.input_requirement().validate(2, "Upsample").is_ok());

        // 3 inputs should fail
        let err = translator.input_requirement().validate(3, "Upsample");
        assert!(err.is_err());
    }
}
