//! Pad operation translator.

use crate::proto::NodeProto;
use crate::translators::{InputRequirement, OnnxAttributes, OnnxTranslator, TranslationError};
use hologram::ir::{GraphBuilder, NodeIndex, PadMode};

/// Translator for ONNX Pad operation.
///
/// Pad adds padding to the input tensor with various modes.
///
/// # ONNX Specification
///
/// - Opset < 11: pads is an attribute
/// - Opset >= 11: pads is an input
///
/// - Inputs (opset >= 11): data, pads, [constant_value]
/// - Inputs (opset < 11): data
/// - Attributes:
///   - mode (default: "constant"): "constant", "reflect", "edge", "wrap"
///   - pads (opset < 11): padding amounts
///   - value (opset < 11): constant value for constant mode
/// - Output: padded tensor
///
/// # Padding Format
///
/// ONNX format: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
/// For 4D tensor: [batch_begin, channel_begin, height_begin, width_begin,
///                 batch_end, channel_end, height_end, width_end]
#[derive(Debug, Default)]
pub struct PadTranslator;

impl OnnxTranslator for PadTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "Pad"
    }

    fn input_requirement(&self) -> InputRequirement {
        InputRequirement::Range(1, 3)
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
            .unwrap_or_else(|| "constant".to_string());

        let mode = match mode_str.as_str() {
            "constant" => PadMode::Constant,
            "reflect" => PadMode::Reflect,
            "edge" => PadMode::Edge,
            "wrap" => PadMode::Wrap,
            _ => {
                return Err(TranslationError::invalid_attribute(
                    "mode",
                    format!("unknown padding mode: {}", mode_str),
                ));
            }
        };

        // Parse constant value (default 0.0)
        let constant_value = node.get_float_or("value", 0.0) as f64;

        // Parse pads
        let pads = if inputs.len() >= 2 {
            // Opset >= 11: pads is an input - not yet supported for dynamic pads
            return Err(TranslationError::unsupported_op("Pad", 13));
        } else {
            // Opset < 11: pads is an attribute
            node.get_ints("pads")
                .ok_or_else(|| TranslationError::missing_attribute("Pad", "pads"))?
        };

        if pads.is_empty() {
            return Err(TranslationError::missing_attribute("Pad", "pads"));
        }

        // Convert ONNX pads format to hologram-ir format
        // ONNX: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
        // Hologram: Vec<(before, after)> for each dimension
        let num_dims = pads.len() / 2;
        if pads.len() != num_dims * 2 {
            return Err(TranslationError::invalid_attribute(
                "pads",
                format!("length must be even, got {}", pads.len()),
            ));
        }

        let mut pad_pairs = Vec::with_capacity(num_dims);
        for i in 0..num_dims {
            let before = pads[i] as usize;
            let after = pads[num_dims + i] as usize;
            pad_pairs.push((before, after));
        }

        let result = builder
            .pad(data, pad_pairs, mode, constant_value)
            .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;

        Ok(vec![result])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::AttributeProto;
    use hologram::ir::{DType, Shape};

    fn make_node_with_attrs(mode: &str, pads: Vec<i64>, value: Option<f32>) -> NodeProto {
        let mut attrs = vec![
            AttributeProto {
                name: "mode".to_string(),
                s: mode.as_bytes().to_vec(),
                ..Default::default()
            },
            AttributeProto {
                name: "pads".to_string(),
                ints: pads,
                ..Default::default()
            },
        ];

        if let Some(v) = value {
            attrs.push(AttributeProto {
                name: "value".to_string(),
                f: v,
                ..Default::default()
            });
        }

        NodeProto {
            name: "pad_test".to_string(),
            op_type: "Pad".to_string(),
            attribute: attrs,
            ..Default::default()
        }
    }

    #[test]
    fn test_pad_constant_mode() {
        let translator = PadTranslator;
        let mut builder = GraphBuilder::new();
        let data = builder.input("data", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);

        // Pad: [N_begin, C_begin, H_begin, W_begin, N_end, C_end, H_end, W_end]
        let node = make_node_with_attrs("constant", vec![0, 0, 1, 1, 0, 0, 1, 1], Some(0.0));
        let result = translator.translate(&node, &[data], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_pad_reflect_mode() {
        let translator = PadTranslator;
        let mut builder = GraphBuilder::new();
        let data = builder.input("data", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);

        let node = make_node_with_attrs("reflect", vec![0, 0, 1, 1, 0, 0, 1, 1], None);
        let result = translator.translate(&node, &[data], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_pad_edge_mode() {
        let translator = PadTranslator;
        let mut builder = GraphBuilder::new();
        let data = builder.input("data", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);

        let node = make_node_with_attrs("edge", vec![0, 0, 2, 2, 0, 0, 2, 2], None);
        let result = translator.translate(&node, &[data], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_pad_wrap_mode() {
        let translator = PadTranslator;
        let mut builder = GraphBuilder::new();
        let data = builder.input("data", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);

        let node = make_node_with_attrs("wrap", vec![0, 0, 1, 1, 0, 0, 1, 1], None);
        let result = translator.translate(&node, &[data], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_pad_default_mode() {
        let translator = PadTranslator;
        let mut builder = GraphBuilder::new();
        let data = builder.input("data", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);

        // No mode specified - should default to constant
        let node = NodeProto {
            name: "pad_test".to_string(),
            op_type: "Pad".to_string(),
            attribute: vec![AttributeProto {
                name: "pads".to_string(),
                ints: vec![0, 0, 1, 1, 0, 0, 1, 1],
                ..Default::default()
            }],
            ..Default::default()
        };

        let result = translator.translate(&node, &[data], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_pad_custom_value() {
        let translator = PadTranslator;
        let mut builder = GraphBuilder::new();
        let data = builder.input("data", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);

        let node = make_node_with_attrs("constant", vec![0, 0, 1, 1, 0, 0, 1, 1], Some(1.5));
        let result = translator.translate(&node, &[data], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_pad_1d() {
        let translator = PadTranslator;
        let mut builder = GraphBuilder::new();
        let data = builder.input("data", Shape::static_shape(&[10]), DType::F32);

        let node = make_node_with_attrs("constant", vec![2, 3], Some(0.0));
        let result = translator.translate(&node, &[data], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_pad_asymmetric() {
        let translator = PadTranslator;
        let mut builder = GraphBuilder::new();
        let data = builder.input("data", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);

        // Asymmetric padding: different amounts for begin and end
        let node = make_node_with_attrs("constant", vec![0, 0, 1, 2, 0, 0, 3, 4], None);
        let result = translator.translate(&node, &[data], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_pad_missing_pads() {
        let translator = PadTranslator;
        let mut builder = GraphBuilder::new();
        let data = builder.input("data", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);

        let node = NodeProto {
            name: "pad_test".to_string(),
            op_type: "Pad".to_string(),
            attribute: vec![AttributeProto {
                name: "mode".to_string(),
                s: b"constant".to_vec(),
                ..Default::default()
            }],
            ..Default::default()
        };

        let result = translator.translate(&node, &[data], &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_pad_invalid_pads_length() {
        let translator = PadTranslator;
        let mut builder = GraphBuilder::new();
        let data = builder.input("data", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);

        // Odd length pads array
        let node = make_node_with_attrs("constant", vec![0, 0, 1], None);
        let result = translator.translate(&node, &[data], &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_pad_dynamic_pads_unsupported() {
        let translator = PadTranslator;
        let mut builder = GraphBuilder::new();
        let data = builder.input("data", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);
        let pads = builder.input("pads", Shape::static_shape(&[8]), DType::I64);

        // No pads attribute - uses input
        let node = NodeProto {
            name: "pad_test".to_string(),
            op_type: "Pad".to_string(),
            attribute: vec![AttributeProto {
                name: "mode".to_string(),
                s: b"constant".to_vec(),
                ..Default::default()
            }],
            ..Default::default()
        };

        let result = translator.translate(&node, &[data, pads], &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_pad_invalid_mode() {
        let translator = PadTranslator;
        let mut builder = GraphBuilder::new();
        let data = builder.input("data", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);

        let node = make_node_with_attrs("invalid_mode", vec![0, 0, 1, 1, 0, 0, 1, 1], None);
        let result = translator.translate(&node, &[data], &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_pad_input_validation() {
        let translator = PadTranslator;

        // 0 inputs should fail
        let err = translator.input_requirement().validate(0, "Pad");
        assert!(err.is_err());

        // 1-3 inputs should pass
        assert!(translator.input_requirement().validate(1, "Pad").is_ok());
        assert!(translator.input_requirement().validate(2, "Pad").is_ok());
        assert!(translator.input_requirement().validate(3, "Pad").is_ok());

        // 4 inputs should fail
        let err = translator.input_requirement().validate(4, "Pad");
        assert!(err.is_err());
    }
}
