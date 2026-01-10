//! Trilu operation translator.

use hologram::ir::{GraphBuilder, NodeIndex};
use crate::proto::NodeProto;
use crate::translators::{OnnxTranslator, OnnxAttributes, InputRequirement, TranslationError};

/// Translator for ONNX Trilu operation.
///
/// Trilu returns the upper or lower triangular part of 2-D matrices or
/// batches of 2-D matrices. Used for attention masking in transformers.
///
/// # ONNX Specification
///
/// - Inputs: input, [k]
///   - input: Tensor of shape [*, N, M]
///   - k: Optional scalar diagonal offset (default: 0)
/// - Attributes:
///   - upper (default: 1): If 1, return upper triangle; else lower triangle
/// - Output: Same shape as input with elements zeroed out
#[derive(Debug, Default)]
pub struct TriluTranslator;

impl OnnxTranslator for TriluTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "Trilu"
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
        // Parse upper attribute (default: 1 = true)
        let upper = node.get_int_or("upper", 1) != 0;

        // Get k parameter if provided
        let k = if inputs.len() > 1 {
            Some(inputs[1])
        } else {
            None
        };

        let result = builder
            .trilu(inputs[0], k, upper)
            .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;

        Ok(vec![result])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram::ir::{DType, Shape, ConstantData};
    use crate::proto::AttributeProto;

    fn make_node() -> NodeProto {
        NodeProto {
            name: "trilu_test".to_string(),
            op_type: "Trilu".to_string(),
            ..Default::default()
        }
    }

    fn make_node_with_upper(upper: i64) -> NodeProto {
        NodeProto {
            name: "trilu_test".to_string(),
            op_type: "Trilu".to_string(),
            attribute: vec![AttributeProto {
                name: "upper".to_string(),
                i: upper,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn test_trilu_upper_default() {
        let translator = TriluTranslator;
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[4, 4]), DType::F32);

        let result = translator.translate(&make_node(), &[input], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_trilu_upper_explicit() {
        let translator = TriluTranslator;
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[5, 5]), DType::F32);

        let node = make_node_with_upper(1);
        let result = translator.translate(&node, &[input], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_trilu_lower() {
        let translator = TriluTranslator;
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[3, 3]), DType::F32);

        let node = make_node_with_upper(0);
        let result = translator.translate(&node, &[input], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_trilu_with_k() {
        let translator = TriluTranslator;
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[4, 4]), DType::F32);
        let k = builder.constant(ConstantData::I64(vec![1]), Shape::static_shape(&[]));

        let result = translator.translate(&make_node(), &[input, k], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_trilu_batched() {
        let translator = TriluTranslator;
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[2, 4, 4]), DType::F32);

        let result = translator.translate(&make_node(), &[input], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_trilu_4d() {
        let translator = TriluTranslator;
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[1, 8, 64, 64]), DType::F32);

        let result = translator.translate(&make_node(), &[input], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_trilu_input_validation() {
        let translator = TriluTranslator;

        // 0 inputs should fail
        let err = translator.input_requirement().validate(0, "Trilu");
        assert!(err.is_err());

        // 1-2 inputs should pass
        assert!(translator.input_requirement().validate(1, "Trilu").is_ok());
        assert!(translator.input_requirement().validate(2, "Trilu").is_ok());

        // 3 inputs should fail
        let err = translator.input_requirement().validate(3, "Trilu");
        assert!(err.is_err());
    }
}
