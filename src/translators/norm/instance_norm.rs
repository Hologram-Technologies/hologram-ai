//! InstanceNormalization operation translator.

use hologram::ir::{GraphBuilder, NodeIndex, NodeOp};
use crate::proto::NodeProto;
use crate::translators::{OnnxTranslator, OnnxAttributes, InputRequirement, TranslationError};

/// Translator for ONNX InstanceNormalization operation.
///
/// InstanceNormalization normalizes each instance independently across
/// spatial dimensions (height and width for 4D tensors).
///
/// # ONNX Specification
///
/// - Inputs: input, scale, B
/// - Attributes:
///   - epsilon (default: 1e-5): Small constant for numerical stability
/// - Output: output
///
/// This is commonly used in style transfer and generative models.
#[derive(Debug, Default)]
pub struct InstanceNormTranslator;

impl OnnxTranslator for InstanceNormTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "InstanceNormalization"
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

        // InstanceNorm: normalize per instance (spatial dimensions)
        // Normalize over last 2 dimensions (H, W for NCHW format)
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
    use hologram::ir::{DType, Shape};
    use crate::proto::AttributeProto;

    fn make_node() -> NodeProto {
        NodeProto {
            name: "instance_norm_test".to_string(),
            op_type: "InstanceNormalization".to_string(),
            ..Default::default()
        }
    }

    fn make_node_with_epsilon(epsilon: f32) -> NodeProto {
        NodeProto {
            name: "instance_norm_test".to_string(),
            op_type: "InstanceNormalization".to_string(),
            attribute: vec![AttributeProto {
                name: "epsilon".to_string(),
                f: epsilon,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn create_instance_norm_inputs(builder: &mut GraphBuilder, channels: usize) -> Vec<NodeIndex> {
        let input = builder.input("input", Shape::static_shape(&[1, channels, 32, 32]), DType::F32);
        let scale = builder.input("scale", Shape::static_shape(&[channels]), DType::F32);
        let bias = builder.input("bias", Shape::static_shape(&[channels]), DType::F32);
        vec![input, scale, bias]
    }

    #[test]
    fn test_instance_norm_basic() {
        let translator = InstanceNormTranslator;
        let mut builder = GraphBuilder::new();
        let inputs = create_instance_norm_inputs(&mut builder, 64);

        let result = translator.translate(&make_node(), &inputs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_instance_norm_custom_epsilon() {
        let translator = InstanceNormTranslator;
        let mut builder = GraphBuilder::new();
        let inputs = create_instance_norm_inputs(&mut builder, 32);

        let node = make_node_with_epsilon(1e-3);
        let result = translator.translate(&node, &inputs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_instance_norm_large_channels() {
        let translator = InstanceNormTranslator;
        let mut builder = GraphBuilder::new();
        let inputs = create_instance_norm_inputs(&mut builder, 512);

        let result = translator.translate(&make_node(), &inputs, &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_instance_norm_small_spatial() {
        let translator = InstanceNormTranslator;
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[2, 64, 4, 4]), DType::F32);
        let scale = builder.input("scale", Shape::static_shape(&[64]), DType::F32);
        let bias = builder.input("bias", Shape::static_shape(&[64]), DType::F32);

        let result = translator.translate(&make_node(), &[input, scale, bias], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_instance_norm_input_validation_insufficient() {
        let translator = InstanceNormTranslator;

        let err = translator.input_requirement().validate(2, "InstanceNormalization");
        assert!(err.is_err());

        let err = translator.input_requirement().validate(1, "InstanceNormalization");
        assert!(err.is_err());
    }

    #[test]
    fn test_instance_norm_input_validation_correct() {
        let translator = InstanceNormTranslator;
        assert!(translator.input_requirement().validate(3, "InstanceNormalization").is_ok());
    }

    #[test]
    fn test_instance_norm_input_validation_too_many() {
        let translator = InstanceNormTranslator;
        let err = translator.input_requirement().validate(4, "InstanceNormalization");
        assert!(err.is_err());
    }
}
