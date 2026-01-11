//! GroupNormalization operation translator.

use crate::proto::NodeProto;
use crate::translators::{InputRequirement, OnnxAttributes, OnnxTranslator, TranslationError};
use hologram::ir::{GraphBuilder, NodeIndex, NodeOp};

/// Translator for ONNX GroupNormalization operation.
///
/// GroupNormalization divides channels into groups and normalizes
/// within each group independently.
///
/// # ONNX Specification
///
/// - Inputs: X, scale, bias
/// - Attributes:
///   - num_groups (required): Number of groups to divide channels into
///   - epsilon (default: 1e-5): Small constant for numerical stability
/// - Output: Y
///
/// # Note
///
/// Currently approximated using LayerNorm over spatial dimensions.
#[derive(Debug, Default)]
pub struct GroupNormTranslator;

impl OnnxTranslator for GroupNormTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "GroupNormalization"
    }

    fn input_requirement(&self) -> InputRequirement {
        InputRequirement::Exact(3)
    }

    fn translate(
        &self,
        node: &NodeProto,
        inputs: &[NodeIndex],
        builder: &mut GraphBuilder,
    ) -> Result<Vec<NodeIndex>, TranslationError> {
        let epsilon = node.get_float_or("epsilon", 1e-5);

        // GroupNorm: normalize within groups
        // Approximate with LayerNorm over spatial dimensions (last 2 dims)
        let axes = vec![-2, -1];

        let result = builder
            .unary(NodeOp::LayerNorm { epsilon, axes }, inputs[0])
            .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;

        Ok(vec![result])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::AttributeProto;
    use hologram::ir::{DType, Shape};

    fn make_node() -> NodeProto {
        NodeProto {
            name: "group_norm_test".to_string(),
            op_type: "GroupNormalization".to_string(),
            attribute: vec![AttributeProto {
                name: "num_groups".to_string(),
                i: 2,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn make_node_with_attrs(num_groups: i64, epsilon: f32) -> NodeProto {
        NodeProto {
            name: "group_norm_test".to_string(),
            op_type: "GroupNormalization".to_string(),
            attribute: vec![
                AttributeProto {
                    name: "num_groups".to_string(),
                    i: num_groups,
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

    fn create_group_norm_inputs(builder: &mut GraphBuilder, channels: usize) -> Vec<NodeIndex> {
        let input = builder.input(
            "input",
            Shape::static_shape(&[1, channels, 8, 8]),
            DType::F32,
        );
        let scale = builder.input("scale", Shape::static_shape(&[channels]), DType::F32);
        let bias = builder.input("bias", Shape::static_shape(&[channels]), DType::F32);
        vec![input, scale, bias]
    }

    #[test]
    fn test_group_norm_basic() {
        let translator = GroupNormTranslator;
        let mut builder = GraphBuilder::new();
        let inputs = create_group_norm_inputs(&mut builder, 6);

        let result = translator.translate(&make_node(), &inputs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_group_norm_custom_params() {
        let translator = GroupNormTranslator;
        let mut builder = GraphBuilder::new();
        let inputs = create_group_norm_inputs(&mut builder, 32);

        let node = make_node_with_attrs(8, 1e-6);
        let result = translator.translate(&node, &inputs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_group_norm_large_channels() {
        let translator = GroupNormTranslator;
        let mut builder = GraphBuilder::new();
        let inputs = create_group_norm_inputs(&mut builder, 256);

        let node = make_node_with_attrs(32, 1e-5);
        let result = translator.translate(&node, &inputs, &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_group_norm_input_validation_insufficient() {
        let translator = GroupNormTranslator;

        let err = translator
            .input_requirement()
            .validate(2, "GroupNormalization");
        assert!(err.is_err());

        let err = translator
            .input_requirement()
            .validate(1, "GroupNormalization");
        assert!(err.is_err());
    }

    #[test]
    fn test_group_norm_input_validation_correct() {
        let translator = GroupNormTranslator;
        assert!(
            translator
                .input_requirement()
                .validate(3, "GroupNormalization")
                .is_ok()
        );
    }

    #[test]
    fn test_group_norm_input_validation_too_many() {
        let translator = GroupNormTranslator;
        let err = translator
            .input_requirement()
            .validate(4, "GroupNormalization");
        assert!(err.is_err());
    }
}
