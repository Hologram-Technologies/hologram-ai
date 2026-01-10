//! Dropout operation translator.

use hologram::ir::{GraphBuilder, NodeIndex};
use crate::proto::NodeProto;
use crate::translators::{OnnxTranslator, InputRequirement, TranslationError};

/// Translator for ONNX Dropout operation.
///
/// During inference, Dropout acts as an identity operation (no dropout applied).
/// The optional mask output is not supported.
///
/// # ONNX Specification
///
/// - Inputs: data, [ratio], [training_mode]
/// - Attributes: seed
/// - Outputs: output, [mask]
///
/// # Note
///
/// This translator implements inference-mode Dropout only (identity).
#[derive(Debug, Default)]
pub struct DropoutTranslator;

impl OnnxTranslator for DropoutTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "Dropout"
    }

    fn input_requirement(&self) -> InputRequirement {
        InputRequirement::Range(1, 3)
    }

    fn translate(
        &self,
        _node: &NodeProto,
        inputs: &[NodeIndex],
        _builder: &mut GraphBuilder,
    ) -> Result<Vec<NodeIndex>, TranslationError> {
        // During inference, Dropout is an identity operation
        // Simply pass through the input unchanged
        Ok(vec![inputs[0]])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram::ir::{DType, Shape};
    use crate::proto::AttributeProto;

    fn make_node() -> NodeProto {
        NodeProto {
            name: "dropout_test".to_string(),
            op_type: "Dropout".to_string(),
            ..Default::default()
        }
    }

    fn make_node_with_seed(seed: i64) -> NodeProto {
        NodeProto {
            name: "dropout_test".to_string(),
            op_type: "Dropout".to_string(),
            attribute: vec![AttributeProto {
                name: "seed".to_string(),
                i: seed,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn test_dropout_basic() {
        let translator = DropoutTranslator;
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[2, 3, 4]), DType::F32);

        let result = translator.translate(&make_node(), &[input], &mut builder);
        assert!(result.is_ok());

        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);
        // Dropout should return identity (same as input)
        assert_eq!(outputs[0], input);
    }

    #[test]
    fn test_dropout_with_ratio() {
        let translator = DropoutTranslator;
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[10, 20]), DType::F32);
        let ratio = builder.input("ratio", Shape::static_shape(&[]), DType::F32);

        let result = translator.translate(&make_node(), &[input, ratio], &mut builder);
        assert!(result.is_ok());

        // Should still be identity during inference
        let outputs = result.unwrap();
        assert_eq!(outputs[0], input);
    }

    #[test]
    fn test_dropout_with_training_mode() {
        let translator = DropoutTranslator;
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[5, 5]), DType::F32);
        let ratio = builder.input("ratio", Shape::static_shape(&[]), DType::F32);
        let training = builder.input("training", Shape::static_shape(&[]), DType::Bool);

        let result = translator.translate(&make_node(), &[input, ratio, training], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_dropout_with_seed() {
        let translator = DropoutTranslator;
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[4, 4]), DType::F32);

        let node = make_node_with_seed(12345);
        let result = translator.translate(&node, &[input], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_dropout_4d() {
        let translator = DropoutTranslator;
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[1, 64, 32, 32]), DType::F32);

        let result = translator.translate(&make_node(), &[input], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_dropout_input_validation() {
        let translator = DropoutTranslator;

        // 0 inputs should fail
        let err = translator.input_requirement().validate(0, "Dropout");
        assert!(err.is_err());

        // 1-3 inputs should pass
        assert!(translator.input_requirement().validate(1, "Dropout").is_ok());
        assert!(translator.input_requirement().validate(2, "Dropout").is_ok());
        assert!(translator.input_requirement().validate(3, "Dropout").is_ok());

        // 4 inputs should fail
        let err = translator.input_requirement().validate(4, "Dropout");
        assert!(err.is_err());
    }
}
