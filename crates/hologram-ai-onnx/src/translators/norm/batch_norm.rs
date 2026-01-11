//! BatchNormalization operation translator.

use crate::proto::NodeProto;
use crate::translators::{InputRequirement, OnnxAttributes, OnnxTranslator, TranslationError};
use hologram::ir::{GraphBuilder, NodeIndex, NodeOp};

/// Translator for ONNX BatchNormalization operation.
///
/// BatchNormalization normalizes the input along the channel dimension using
/// running statistics during inference.
///
/// # ONNX Specification
///
/// - Inputs: X, scale, B, input_mean, input_var
/// - Attributes:
///   - epsilon (default: 1e-5): Small constant for numerical stability
///   - momentum (default: 0.9): Factor for computing running stats (training only)
/// - Outputs: Y, [running_mean], [running_var] (extra outputs for training only)
///
/// Formula: Y = scale * (X - mean) / sqrt(var + epsilon) + B
#[derive(Debug, Default)]
pub struct BatchNormTranslator;

impl OnnxTranslator for BatchNormTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "BatchNormalization"
    }

    fn input_requirement(&self) -> InputRequirement {
        InputRequirement::Exact(5)
    }

    fn translate(
        &self,
        node: &NodeProto,
        inputs: &[NodeIndex],
        builder: &mut GraphBuilder,
    ) -> Result<Vec<NodeIndex>, TranslationError> {
        let epsilon = node.get_float_or("epsilon", 1e-5);
        let momentum = node.get_float_or("momentum", 0.9);

        // BatchNorm: (x - mean) / sqrt(var + eps) * scale + bias
        let result = builder
            .unary(NodeOp::BatchNorm { epsilon, momentum }, inputs[0])
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
            name: "batch_norm_test".to_string(),
            op_type: "BatchNormalization".to_string(),
            ..Default::default()
        }
    }

    fn make_node_with_attrs(epsilon: f32, momentum: f32) -> NodeProto {
        NodeProto {
            name: "batch_norm_test".to_string(),
            op_type: "BatchNormalization".to_string(),
            attribute: vec![
                AttributeProto {
                    name: "epsilon".to_string(),
                    f: epsilon,
                    ..Default::default()
                },
                AttributeProto {
                    name: "momentum".to_string(),
                    f: momentum,
                    ..Default::default()
                },
            ],
            ..Default::default()
        }
    }

    fn create_batch_norm_inputs(builder: &mut GraphBuilder, channels: usize) -> Vec<NodeIndex> {
        let input = builder.input(
            "input",
            Shape::static_shape(&[1, channels, 32, 32]),
            DType::F32,
        );
        let scale = builder.input("scale", Shape::static_shape(&[channels]), DType::F32);
        let bias = builder.input("bias", Shape::static_shape(&[channels]), DType::F32);
        let mean = builder.input("mean", Shape::static_shape(&[channels]), DType::F32);
        let var = builder.input("var", Shape::static_shape(&[channels]), DType::F32);
        vec![input, scale, bias, mean, var]
    }

    #[test]
    fn test_batch_norm_basic() {
        let translator = BatchNormTranslator;
        let mut builder = GraphBuilder::new();
        let inputs = create_batch_norm_inputs(&mut builder, 3);

        let result = translator.translate(&make_node(), &inputs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_batch_norm_custom_params() {
        let translator = BatchNormTranslator;
        let mut builder = GraphBuilder::new();
        let inputs = create_batch_norm_inputs(&mut builder, 64);

        let node = make_node_with_attrs(1e-3, 0.99);
        let result = translator.translate(&node, &inputs, &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_batch_norm_large_channels() {
        let translator = BatchNormTranslator;
        let mut builder = GraphBuilder::new();
        let inputs = create_batch_norm_inputs(&mut builder, 512);

        let result = translator.translate(&make_node(), &inputs, &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_batch_norm_input_validation_insufficient() {
        let translator = BatchNormTranslator;

        // Less than 5 inputs should fail
        let err = translator
            .input_requirement()
            .validate(4, "BatchNormalization");
        assert!(err.is_err());

        let err = translator
            .input_requirement()
            .validate(2, "BatchNormalization");
        assert!(err.is_err());
    }

    #[test]
    fn test_batch_norm_input_validation_correct() {
        let translator = BatchNormTranslator;
        assert!(
            translator
                .input_requirement()
                .validate(5, "BatchNormalization")
                .is_ok()
        );
    }

    #[test]
    fn test_batch_norm_input_validation_too_many() {
        let translator = BatchNormTranslator;
        let err = translator
            .input_requirement()
            .validate(6, "BatchNormalization");
        assert!(err.is_err());
    }
}
