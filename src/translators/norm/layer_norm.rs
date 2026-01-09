//! LayerNormalization operation translator.

use hologram::ir::{GraphBuilder, NodeIndex, NodeOp};
use crate::proto::NodeProto;
use crate::translators::{OnnxTranslator, OnnxAttributes, InputRequirement, TranslationError};

/// Translator for ONNX LayerNormalization operation.
///
/// LayerNormalization normalizes the input tensor over the last dimensions
/// starting from the specified axis.
///
/// # ONNX Specification
///
/// - Inputs: X, [Scale], [Bias]
/// - Attributes:
///   - axis (default: -1): First normalization dimension
///   - epsilon (default: 1e-5): Small constant for numerical stability
/// - Outputs: Y, [Mean], [InvStdDev]
#[derive(Debug, Default)]
pub struct LayerNormTranslator;

impl OnnxTranslator for LayerNormTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "LayerNormalization"
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
        let epsilon = node.get_float_or("epsilon", 1e-5);
        let axis = node.get_int_or("axis", -1) as i32;

        // Get input node to determine rank
        let input_node = builder.graph().node(inputs[0])
            .ok_or_else(|| TranslationError::IrBuilder("Invalid input node".to_string()))?;
        let rank = input_node.shape.rank() as i32;

        // Normalize over last dimensions from axis onwards
        let axes: Vec<i32> = if axis < 0 {
            (axis..0).map(|i| rank + i).collect()
        } else {
            (axis..rank).collect()
        };

        let result = builder
            .unary(NodeOp::LayerNorm { epsilon, axes }, inputs[0])
            .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;

        Ok(vec![result])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram::ir::{DType, Shape};
    use crate::proto::AttributeProto;

    fn make_node() -> NodeProto {
        NodeProto {
            name: "layer_norm_test".to_string(),
            op_type: "LayerNormalization".to_string(),
            ..Default::default()
        }
    }

    fn make_node_with_attrs(axis: i64, epsilon: f32) -> NodeProto {
        NodeProto {
            name: "layer_norm_test".to_string(),
            op_type: "LayerNormalization".to_string(),
            attribute: vec![
                AttributeProto {
                    name: "axis".to_string(),
                    i: axis,
                    ..Default::default()
                },
                AttributeProto {
                    name: "epsilon".to_string(),
                    f: epsilon,
                    ..Default::default()
                },
            ],
            ..Default::default()
        }
    }

    #[test]
    fn test_layer_norm_default_axis() {
        let translator = LayerNormTranslator;
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[2, 3, 4]), DType::F32);

        let result = translator.translate(&make_node(), &[input], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_layer_norm_custom_axis() {
        let translator = LayerNormTranslator;
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[4, 5, 6]), DType::F32);

        let node = make_node_with_attrs(1, 1e-6);
        let result = translator.translate(&node, &[input], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_layer_norm_negative_axis() {
        let translator = LayerNormTranslator;
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[2, 3, 4]), DType::F32);

        let node = make_node_with_attrs(-2, 1e-5);
        let result = translator.translate(&node, &[input], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_layer_norm_with_scale_bias() {
        let translator = LayerNormTranslator;
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[2, 3, 4]), DType::F32);
        let scale = builder.input("scale", Shape::static_shape(&[4]), DType::F32);
        let bias = builder.input("bias", Shape::static_shape(&[4]), DType::F32);

        let result = translator.translate(&make_node(), &[input, scale, bias], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_layer_norm_4d() {
        let translator = LayerNormTranslator;
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[1, 64, 32, 32]), DType::F32);

        let result = translator.translate(&make_node(), &[input], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_layer_norm_input_validation() {
        let translator = LayerNormTranslator;

        // 0 inputs should fail
        let err = translator.input_requirement().validate(0, "LayerNormalization");
        assert!(err.is_err());

        // 1-3 inputs should pass
        assert!(translator.input_requirement().validate(1, "LayerNormalization").is_ok());
        assert!(translator.input_requirement().validate(2, "LayerNormalization").is_ok());
        assert!(translator.input_requirement().validate(3, "LayerNormalization").is_ok());

        // 4 inputs should fail
        let err = translator.input_requirement().validate(4, "LayerNormalization");
        assert!(err.is_err());
    }
}
