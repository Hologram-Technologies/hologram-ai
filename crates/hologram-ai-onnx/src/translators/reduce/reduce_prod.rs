//! ReduceProd operation translator.

use crate::proto::NodeProto;
use crate::translators::{InputRequirement, OnnxTranslator, TranslationError};
use hologram::ir::{GraphBuilder, NodeIndex};

/// Translator for ONNX ReduceProd operation.
///
/// ReduceProd computes the product of elements along the specified axes.
///
/// # Note
///
/// ReduceProd is not currently supported in hologram-ir. This translator
/// returns an unsupported operation error.
///
/// # Attributes
///
/// - `axes`: int array (default: reduce all axes) - Axes along which to reduce.
/// - `keepdims`: int (default: 1) - If 1, reduced dimensions are retained with size 1.
/// - `noop_with_empty_axes`: int (default: 0) - If 1 and axes is empty, return input unchanged.
///
/// # Inputs
///
/// - `data` (required): Input tensor to reduce
/// - `axes` (optional): Axes along which to reduce
///
/// # Outputs
///
/// - `reduced`: Reduced tensor containing product values
#[derive(Debug, Default)]
pub struct ReduceProdTranslator;

impl OnnxTranslator for ReduceProdTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "ReduceProd"
    }

    fn input_requirement(&self) -> InputRequirement {
        // data is required, axes is optional as second input
        InputRequirement::Range(1, 2)
    }

    fn translate(
        &self,
        _node: &NodeProto,
        _inputs: &[NodeIndex],
        _builder: &mut GraphBuilder,
    ) -> Result<Vec<NodeIndex>, TranslationError> {
        // ReduceProd is not currently supported in hologram-ir
        Err(TranslationError::unsupported_op("ReduceProd", 13))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::AttributeProto;
    use hologram::ir::{DType, Shape};

    fn make_node() -> NodeProto {
        NodeProto {
            name: "reduce_prod_test".to_string(),
            op_type: "ReduceProd".to_string(),
            ..Default::default()
        }
    }

    fn make_node_with_axes(axes: Vec<i64>) -> NodeProto {
        NodeProto {
            name: "reduce_prod_test".to_string(),
            op_type: "ReduceProd".to_string(),
            attribute: vec![AttributeProto {
                name: "axes".to_string(),
                ints: axes,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    // ===== Unsupported Operation Tests =====

    #[test]
    fn test_reduce_prod_unsupported() {
        let translator = ReduceProdTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);

        let result = translator.translate(&make_node_with_axes(vec![1]), &[x], &mut builder);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            TranslationError::UnsupportedOp { op, opset: 13 } if op == "ReduceProd"
        ));
    }

    #[test]
    fn test_reduce_prod_unsupported_no_axes() {
        let translator = ReduceProdTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[4, 5, 6]), DType::F32);

        let result = translator.translate(&make_node(), &[x], &mut builder);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            TranslationError::UnsupportedOp { .. }
        ));
    }

    #[test]
    fn test_reduce_prod_unsupported_multiple_axes() {
        let translator = ReduceProdTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[2, 3, 4]), DType::F32);

        let result = translator.translate(&make_node_with_axes(vec![0, 2]), &[x], &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_reduce_prod_unsupported_1d() {
        let translator = ReduceProdTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[100]), DType::F32);

        let result = translator.translate(&make_node_with_axes(vec![0]), &[x], &mut builder);
        assert!(result.is_err());
    }

    // ===== Input Requirement Tests =====

    #[test]
    fn test_reduce_prod_no_inputs() {
        let translator = ReduceProdTranslator;
        let err = translator.input_requirement().validate(0, "ReduceProd");
        assert!(err.is_err());
        assert!(matches!(
            err.unwrap_err(),
            TranslationError::InputCountOutOfRange {
                min: 1,
                max: 2,
                got: 0,
                ..
            }
        ));
    }

    #[test]
    fn test_reduce_prod_too_many_inputs() {
        let translator = ReduceProdTranslator;
        let err = translator.input_requirement().validate(3, "ReduceProd");
        assert!(err.is_err());
    }

    #[test]
    fn test_reduce_prod_valid_input_count_single() {
        let translator = ReduceProdTranslator;
        let result = translator.input_requirement().validate(1, "ReduceProd");
        assert!(result.is_ok());
    }

    #[test]
    fn test_reduce_prod_valid_input_count_with_axes() {
        let translator = ReduceProdTranslator;
        let result = translator.input_requirement().validate(2, "ReduceProd");
        assert!(result.is_ok());
    }

    #[test]
    fn test_reduce_prod_input_requirement() {
        let translator = ReduceProdTranslator;
        assert_eq!(
            translator.input_requirement(),
            InputRequirement::Range(1, 2)
        );
    }

    #[test]
    fn test_reduce_prod_op_type() {
        let translator = ReduceProdTranslator;
        assert_eq!(translator.onnx_op_type(), "ReduceProd");
    }
}
