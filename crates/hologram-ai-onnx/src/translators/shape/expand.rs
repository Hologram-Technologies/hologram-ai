//! Expand operation translator.

use crate::proto::NodeProto;
use crate::translators::{InputRequirement, OnnxTranslator, TranslationError};
use hologram::ir::{GraphBuilder, NodeIndex};

/// Translator for ONNX Expand operation.
///
/// Expand broadcasts the input tensor to a target shape following numpy broadcasting rules.
///
/// # Inputs
/// - data: Input tensor
/// - shape: 1D tensor specifying the target shape
///
/// # Broadcasting Rules
/// - Dimensions are compared from right to left
/// - Two dimensions are compatible if:
///   - They are equal
///   - One of them is 1
/// - If input has fewer dimensions, leading dimensions are prepended with 1
///
/// # Example
/// - Input shape [3, 1] + target [2, 1, 6] -> Output [2, 3, 6]
#[derive(Debug, Default)]
pub struct ExpandTranslator;

impl OnnxTranslator for ExpandTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "Expand"
    }

    fn input_requirement(&self) -> InputRequirement {
        InputRequirement::Exact(2)
    }

    fn translate(
        &self,
        _node: &NodeProto,
        inputs: &[NodeIndex],
        builder: &mut GraphBuilder,
    ) -> Result<Vec<NodeIndex>, TranslationError> {
        let data = inputs[0];
        let shape = inputs[1];

        tracing::debug!("Expand: data={:?}, shape={:?}", data, shape);

        // Use hologram-ir's expand operation which properly handles shape inference
        let result = builder
            .expand(data, shape)
            .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;

        Ok(vec![result])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram::ir::{ConstantData, DType, Shape};

    fn make_node() -> NodeProto {
        NodeProto {
            name: "expand_test".to_string(),
            op_type: "Expand".to_string(),
            ..Default::default()
        }
    }

    // ===== Valid Input Tests =====

    #[test]
    fn test_expand_broadcast_2d() {
        let translator = ExpandTranslator;
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[3, 1]), DType::F32);
        let shape = builder.constant(ConstantData::I64(vec![3, 4]), Shape::static_shape(&[2]));

        let result = translator.translate(&make_node(), &[data, shape], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn test_expand_add_dimension() {
        let translator = ExpandTranslator;
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[3, 4]), DType::F32);
        let shape = builder.constant(ConstantData::I64(vec![2, 3, 4]), Shape::static_shape(&[3]));

        let result = translator.translate(&make_node(), &[data, shape], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_expand_broadcast_all_dims() {
        let translator = ExpandTranslator;
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[1, 1, 1]), DType::F32);
        let shape = builder.constant(ConstantData::I64(vec![2, 3, 4]), Shape::static_shape(&[3]));

        let result = translator.translate(&make_node(), &[data, shape], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_expand_identity() {
        let translator = ExpandTranslator;
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[2, 3]), DType::F32);
        let shape = builder.constant(ConstantData::I64(vec![2, 3]), Shape::static_shape(&[2]));

        let result = translator.translate(&make_node(), &[data, shape], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_expand_scalar() {
        let translator = ExpandTranslator;
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[]), DType::F32);
        let shape = builder.constant(ConstantData::I64(vec![2, 3, 4]), Shape::static_shape(&[3]));

        let result = translator.translate(&make_node(), &[data, shape], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_expand_1d_to_3d() {
        let translator = ExpandTranslator;
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[4]), DType::F32);
        let shape = builder.constant(ConstantData::I64(vec![2, 3, 4]), Shape::static_shape(&[3]));

        let result = translator.translate(&make_node(), &[data, shape], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_expand_dynamic_shape() {
        let translator = ExpandTranslator;
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[1, 3, 1]), DType::F32);
        // Shape as input (not constant)
        let shape = builder.input("shape", Shape::static_shape(&[3]), DType::I64);

        let result = translator.translate(&make_node(), &[data, shape], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_expand_broadcast_middle_dim() {
        let translator = ExpandTranslator;
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[2, 1, 4]), DType::F32);
        let shape = builder.constant(ConstantData::I64(vec![2, 3, 4]), Shape::static_shape(&[3]));

        let result = translator.translate(&make_node(), &[data, shape], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_expand_high_rank() {
        let translator = ExpandTranslator;
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[1, 1, 1, 1]), DType::F32);
        let shape = builder.constant(
            ConstantData::I64(vec![2, 3, 4, 5]),
            Shape::static_shape(&[4]),
        );

        let result = translator.translate(&make_node(), &[data, shape], &mut builder);
        assert!(result.is_ok());
    }

    // ===== Invalid Input Tests =====

    #[test]
    fn test_expand_no_inputs() {
        let translator = ExpandTranslator;
        let err = translator.input_requirement().validate(0, "Expand");
        assert!(err.is_err());
        assert!(matches!(
            err.unwrap_err(),
            TranslationError::WrongInputCount {
                expected: 2,
                got: 0,
                ..
            }
        ));
    }

    #[test]
    fn test_expand_one_input() {
        let translator = ExpandTranslator;
        let err = translator.input_requirement().validate(1, "Expand");
        assert!(err.is_err());
    }

    #[test]
    fn test_expand_too_many_inputs() {
        let translator = ExpandTranslator;
        let err = translator.input_requirement().validate(3, "Expand");
        assert!(err.is_err());
    }

    // ===== Trait Method Tests =====

    #[test]
    fn test_op_type() {
        let translator = ExpandTranslator;
        assert_eq!(translator.onnx_op_type(), "Expand");
    }

    #[test]
    fn test_input_requirement() {
        let translator = ExpandTranslator;
        let req = translator.input_requirement();
        assert!(matches!(req, InputRequirement::Exact(2)));
        assert!(!req.accepts_zero());
    }
}
