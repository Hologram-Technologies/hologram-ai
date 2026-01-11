//! Softmax activation translator.

use crate::proto::NodeProto;
use crate::translators::{InputRequirement, OnnxAttributes, OnnxTranslator, TranslationError};
use hologram::ir::{GraphBuilder, NodeIndex};

/// Translator for ONNX Softmax operation.
///
/// Softmax(x, axis) = exp(x) / sum(exp(x), axis)
#[derive(Debug, Default)]
pub struct SoftmaxTranslator;

impl OnnxTranslator for SoftmaxTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "Softmax"
    }

    fn input_requirement(&self) -> InputRequirement {
        InputRequirement::Exact(1)
    }

    fn translate(
        &self,
        node: &NodeProto,
        inputs: &[NodeIndex],
        builder: &mut GraphBuilder,
    ) -> Result<Vec<NodeIndex>, TranslationError> {
        let axis = node.get_int_or("axis", -1) as i32;
        let result = builder
            .softmax(inputs[0], axis)
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
            name: "softmax_test".to_string(),
            op_type: "Softmax".to_string(),
            ..Default::default()
        }
    }

    fn make_node_with_axis(axis: i64) -> NodeProto {
        NodeProto {
            name: "softmax_test".to_string(),
            op_type: "Softmax".to_string(),
            attribute: vec![AttributeProto {
                name: "axis".to_string(),
                i: axis,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn test_softmax_default_axis() {
        let translator = SoftmaxTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[2, 10]), DType::F32);

        let result = translator.translate(&make_node(), &[x], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_softmax_explicit_axis() {
        let translator = SoftmaxTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[2, 10]), DType::F32);

        let result = translator.translate(&make_node_with_axis(1), &[x], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_softmax_axis_zero() {
        let translator = SoftmaxTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[10, 5]), DType::F32);

        let result = translator.translate(&make_node_with_axis(0), &[x], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_softmax_negative_axis() {
        let translator = SoftmaxTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[2, 3, 10]), DType::F32);

        let result = translator.translate(&make_node_with_axis(-2), &[x], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_softmax_3d() {
        let translator = SoftmaxTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[1, 128, 32128]), DType::F32);

        let result = translator.translate(&make_node_with_axis(-1), &[x], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_softmax_no_inputs() {
        let translator = SoftmaxTranslator;
        let err = translator.input_requirement().validate(0, "Softmax");
        assert!(err.is_err());
    }

    #[test]
    fn test_softmax_too_many_inputs() {
        let translator = SoftmaxTranslator;
        let err = translator.input_requirement().validate(2, "Softmax");
        assert!(err.is_err());
    }
}
