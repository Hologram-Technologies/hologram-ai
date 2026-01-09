//! Where operation translator.

use hologram::ir::{GraphBuilder, NodeIndex};
use crate::proto::NodeProto;
use crate::translators::{OnnxTranslator, InputRequirement, TranslationError};

/// Translator for ONNX Where operation.
///
/// Where(condition, X, Y) returns elements from X where condition is true,
/// and elements from Y where condition is false.
///
/// # ONNX Specification
///
/// - Inputs: condition, X, Y
/// - Output: output with same shape as broadcasted inputs
///
/// Formula: output[i] = condition[i] ? X[i] : Y[i]
#[derive(Debug, Default)]
pub struct WhereTranslator;

impl OnnxTranslator for WhereTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "Where"
    }

    fn input_requirement(&self) -> InputRequirement {
        InputRequirement::Exact(3)
    }

    fn translate(
        &self,
        _node: &NodeProto,
        inputs: &[NodeIndex],
        builder: &mut GraphBuilder,
    ) -> Result<Vec<NodeIndex>, TranslationError> {
        let condition = inputs[0];
        let x = inputs[1];
        let y = inputs[2];

        let result = builder
            .where_select(condition, x, y)
            .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;

        Ok(vec![result])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram::ir::{DType, Shape};

    fn make_node() -> NodeProto {
        NodeProto {
            name: "where_test".to_string(),
            op_type: "Where".to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn test_where_basic() {
        let translator = WhereTranslator;
        let mut builder = GraphBuilder::new();
        let condition = builder.input("condition", Shape::static_shape(&[2, 3]), DType::Bool);
        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);
        let y = builder.input("y", Shape::static_shape(&[2, 3]), DType::F32);

        let result = translator.translate(&make_node(), &[condition, x, y], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_where_large_shape() {
        let translator = WhereTranslator;
        let mut builder = GraphBuilder::new();
        let condition = builder.input("condition", Shape::static_shape(&[1, 128, 64, 64]), DType::Bool);
        let x = builder.input("x", Shape::static_shape(&[1, 128, 64, 64]), DType::F32);
        let y = builder.input("y", Shape::static_shape(&[1, 128, 64, 64]), DType::F32);

        let result = translator.translate(&make_node(), &[condition, x, y], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_where_1d() {
        let translator = WhereTranslator;
        let mut builder = GraphBuilder::new();
        let condition = builder.input("condition", Shape::static_shape(&[10]), DType::Bool);
        let x = builder.input("x", Shape::static_shape(&[10]), DType::F32);
        let y = builder.input("y", Shape::static_shape(&[10]), DType::F32);

        let result = translator.translate(&make_node(), &[condition, x, y], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_where_input_validation_insufficient() {
        let translator = WhereTranslator;

        let err = translator.input_requirement().validate(0, "Where");
        assert!(err.is_err());

        let err = translator.input_requirement().validate(1, "Where");
        assert!(err.is_err());

        let err = translator.input_requirement().validate(2, "Where");
        assert!(err.is_err());
    }

    #[test]
    fn test_where_input_validation_correct() {
        let translator = WhereTranslator;
        assert!(translator.input_requirement().validate(3, "Where").is_ok());
    }

    #[test]
    fn test_where_input_validation_too_many() {
        let translator = WhereTranslator;
        let err = translator.input_requirement().validate(4, "Where");
        assert!(err.is_err());
    }
}
