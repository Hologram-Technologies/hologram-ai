//! Conv operation translator.

use crate::proto::NodeProto;
use crate::translators::{InputRequirement, OnnxAttributes, OnnxTranslator, TranslationError};
use hologram::ir::{GraphBuilder, NodeIndex, Padding};

/// Translator for ONNX Conv operation.
///
/// Conv performs 2D convolution with support for groups, padding, strides, and dilations.
///
/// # ONNX Specification
///
/// - Inputs: X, W, [B]
///   - X: Input tensor [N, C, H, W]
///   - W: Weight tensor [M, C/group, kH, kW]
///   - B: Optional bias [M]
/// - Attributes:
///   - kernel_shape (required): [kH, kW]
///   - strides (default: [1, 1])
///   - pads (default: [0, 0, 0, 0]): [top, left, bottom, right]
///   - dilations (default: [1, 1])
///   - group (default: 1): Number of groups for grouped convolution
#[derive(Debug, Default)]
pub struct ConvTranslator;

impl OnnxTranslator for ConvTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "Conv"
    }

    fn input_requirement(&self) -> InputRequirement {
        InputRequirement::Range(2, 3)
    }

    fn translate(
        &self,
        node: &NodeProto,
        inputs: &[NodeIndex],
        builder: &mut GraphBuilder,
    ) -> Result<Vec<NodeIndex>, TranslationError> {
        let input = inputs[0];
        let weight = inputs[1];
        let bias = inputs.get(2).copied();

        // Parse kernel_shape
        let kernel_shape = node
            .get_ints("kernel_shape")
            .ok_or_else(|| TranslationError::missing_attribute("Conv", "kernel_shape"))?;
        if kernel_shape.len() != 2 {
            return Err(TranslationError::invalid_attribute(
                "kernel_shape",
                format!("must be 2D, got {:?}", kernel_shape),
            ));
        }
        let kernel = (kernel_shape[0] as usize, kernel_shape[1] as usize);

        // Parse strides (default [1, 1])
        let strides = node.get_ints_or("strides", &[1, 1]);
        if strides.len() != 2 {
            return Err(TranslationError::invalid_attribute(
                "strides",
                format!("must be 2D, got {:?}", strides),
            ));
        }
        let stride = (strides[0] as usize, strides[1] as usize);

        // Parse group
        let groups = node.get_int_or("group", 1) as usize;

        // Parse dilations - hologram currently only supports dilation=1
        let dilations = node.get_ints_or("dilations", &[1, 1]);
        if dilations.iter().any(|&d| d != 1) {
            return Err(TranslationError::unsupported_op("Conv", 13));
        }

        // Parse padding
        let pads = node.get_ints_or("pads", &[]);
        let padding = Self::parse_padding(pads)?;

        // Create convolution node
        let conv_result = builder
            .conv2d(input, weight, kernel, stride, padding, groups)
            .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;

        // Add bias if present
        let result = if let Some(bias_node) = bias {
            builder
                .add(conv_result, bias_node)
                .map_err(|e| TranslationError::IrBuilder(e.to_string()))?
        } else {
            conv_result
        };

        Ok(vec![result])
    }
}

impl ConvTranslator {
    /// Parse ONNX padding to hologram-ir Padding.
    ///
    /// ONNX format: [x1_begin, x2_begin, x1_end, x2_end] for 2D
    /// For 4-element: [top, left, bottom, right]
    fn parse_padding(pads: &[i64]) -> Result<Padding, TranslationError> {
        if pads.is_empty() {
            return Ok(Padding::Valid);
        }

        if pads.len() == 4 {
            let top = pads[0] as usize;
            let left = pads[1] as usize;
            let bottom = pads[2] as usize;
            let right = pads[3] as usize;

            if top == 0 && left == 0 && bottom == 0 && right == 0 {
                Ok(Padding::Valid)
            } else {
                Ok(Padding::Explicit {
                    top,
                    bottom,
                    left,
                    right,
                })
            }
        } else {
            Err(TranslationError::invalid_attribute(
                "pads",
                format!("expected 4 elements, got {}", pads.len()),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::AttributeProto;
    use hologram::ir::{DType, Shape};

    fn make_node_with_attrs(
        kernel: Vec<i64>,
        strides: Option<Vec<i64>>,
        pads: Option<Vec<i64>>,
        group: Option<i64>,
    ) -> NodeProto {
        let mut attrs = vec![AttributeProto {
            name: "kernel_shape".to_string(),
            ints: kernel,
            ..Default::default()
        }];

        if let Some(s) = strides {
            attrs.push(AttributeProto {
                name: "strides".to_string(),
                ints: s,
                ..Default::default()
            });
        }

        if let Some(p) = pads {
            attrs.push(AttributeProto {
                name: "pads".to_string(),
                ints: p,
                ..Default::default()
            });
        }

        if let Some(g) = group {
            attrs.push(AttributeProto {
                name: "group".to_string(),
                i: g,
                ..Default::default()
            });
        }

        NodeProto {
            name: "conv_test".to_string(),
            op_type: "Conv".to_string(),
            attribute: attrs,
            ..Default::default()
        }
    }

    #[test]
    fn test_conv_basic() {
        let translator = ConvTranslator;
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);
        let weight = builder.input("weight", Shape::static_shape(&[64, 3, 3, 3]), DType::F32);

        let node = make_node_with_attrs(vec![3, 3], Some(vec![1, 1]), Some(vec![1, 1, 1, 1]), None);
        let result = translator.translate(&node, &[input, weight], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_conv_with_stride() {
        let translator = ConvTranslator;
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);
        let weight = builder.input("weight", Shape::static_shape(&[64, 3, 3, 3]), DType::F32);

        let node = make_node_with_attrs(vec![3, 3], Some(vec![2, 2]), None, None);
        let result = translator.translate(&node, &[input, weight], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_conv_depthwise() {
        let translator = ConvTranslator;
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[1, 64, 32, 32]), DType::F32);
        let weight = builder.input("weight", Shape::static_shape(&[64, 1, 3, 3]), DType::F32);

        let node = make_node_with_attrs(vec![3, 3], None, None, Some(64));
        let result = translator.translate(&node, &[input, weight], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_conv_no_padding() {
        let translator = ConvTranslator;
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);
        let weight = builder.input("weight", Shape::static_shape(&[64, 3, 3, 3]), DType::F32);

        let node = make_node_with_attrs(vec![3, 3], None, None, None);
        let result = translator.translate(&node, &[input, weight], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_conv_missing_kernel_shape() {
        let translator = ConvTranslator;
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);
        let weight = builder.input("weight", Shape::static_shape(&[64, 3, 3, 3]), DType::F32);

        let node = NodeProto {
            name: "conv_test".to_string(),
            op_type: "Conv".to_string(),
            ..Default::default()
        };
        let result = translator.translate(&node, &[input, weight], &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_conv_dilations_unsupported() {
        let translator = ConvTranslator;
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);
        let weight = builder.input("weight", Shape::static_shape(&[64, 3, 3, 3]), DType::F32);

        let mut node = make_node_with_attrs(vec![3, 3], None, None, None);
        node.attribute.push(AttributeProto {
            name: "dilations".to_string(),
            ints: vec![2, 2],
            ..Default::default()
        });

        let result = translator.translate(&node, &[input, weight], &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_conv_input_validation() {
        let translator = ConvTranslator;

        // 0 or 1 input should fail
        let err = translator.input_requirement().validate(0, "Conv");
        assert!(err.is_err());
        let err = translator.input_requirement().validate(1, "Conv");
        assert!(err.is_err());

        // 2-3 inputs should pass
        assert!(translator.input_requirement().validate(2, "Conv").is_ok());
        assert!(translator.input_requirement().validate(3, "Conv").is_ok());

        // 4 inputs should fail
        let err = translator.input_requirement().validate(4, "Conv");
        assert!(err.is_err());
    }

    #[test]
    fn test_parse_padding_valid() {
        let padding = ConvTranslator::parse_padding(&[]).unwrap();
        assert!(matches!(padding, Padding::Valid));

        let padding = ConvTranslator::parse_padding(&[0, 0, 0, 0]).unwrap();
        assert!(matches!(padding, Padding::Valid));
    }

    #[test]
    fn test_parse_padding_explicit() {
        let padding = ConvTranslator::parse_padding(&[1, 2, 3, 4]).unwrap();
        assert!(matches!(
            padding,
            Padding::Explicit {
                top: 1,
                bottom: 3,
                left: 2,
                right: 4
            }
        ));
    }

    #[test]
    fn test_parse_padding_invalid() {
        let result = ConvTranslator::parse_padding(&[1, 2, 3]);
        assert!(result.is_err());
    }
}
