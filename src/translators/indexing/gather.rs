//! Gather operation translator.

use hologram::ir::{GraphBuilder, NodeIndex};
use crate::proto::NodeProto;
use crate::translators::{OnnxTranslator, OnnxAttributes, InputRequirement, TranslationError};

/// Translator for ONNX Gather operation.
///
/// Gather(data, indices, axis) gathers elements from data along the specified axis
/// using the provided indices tensor.
///
/// # ONNX Specification
///
/// - Inputs: data, indices
/// - Attributes: axis (default: 0)
/// - Output: gathered tensor with shape data.shape[:axis] + indices.shape + data.shape[axis+1:]
#[derive(Debug, Default)]
pub struct GatherTranslator;

impl OnnxTranslator for GatherTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "Gather"
    }

    fn input_requirement(&self) -> InputRequirement {
        InputRequirement::Exact(2)
    }

    fn translate(
        &self,
        node: &NodeProto,
        inputs: &[NodeIndex],
        builder: &mut GraphBuilder,
    ) -> Result<Vec<NodeIndex>, TranslationError> {
        let data = inputs[0];
        let indices = inputs[1];
        let axis = node.get_int_or("axis", 0) as i32;

        let result = builder
            .gather(data, indices, axis)
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
            name: "gather_test".to_string(),
            op_type: "Gather".to_string(),
            ..Default::default()
        }
    }

    fn make_node_with_axis(axis: i64) -> NodeProto {
        NodeProto {
            name: "gather_test".to_string(),
            op_type: "Gather".to_string(),
            attribute: vec![AttributeProto {
                name: "axis".to_string(),
                i: axis,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn test_gather_default_axis() {
        let translator = GatherTranslator;
        let mut builder = GraphBuilder::new();
        let data = builder.input("data", Shape::static_shape(&[3, 4, 5]), DType::F32);
        let indices = builder.input("indices", Shape::static_shape(&[2, 3]), DType::I64);

        let result = translator.translate(&make_node(), &[data, indices], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_gather_axis_1() {
        let translator = GatherTranslator;
        let mut builder = GraphBuilder::new();
        let data = builder.input("data", Shape::static_shape(&[3, 4, 5]), DType::F32);
        let indices = builder.input("indices", Shape::static_shape(&[3, 2]), DType::I64);

        let result = translator.translate(&make_node_with_axis(1), &[data, indices], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_gather_negative_axis() {
        let translator = GatherTranslator;
        let mut builder = GraphBuilder::new();
        let data = builder.input("data", Shape::static_shape(&[3, 4, 5]), DType::F32);
        let indices = builder.input("indices", Shape::static_shape(&[2]), DType::I64);

        let result = translator.translate(&make_node_with_axis(-1), &[data, indices], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_gather_scalar_indices() {
        let translator = GatherTranslator;
        let mut builder = GraphBuilder::new();
        let data = builder.input("data", Shape::static_shape(&[10, 5]), DType::F32);
        let indices = builder.input("indices", Shape::static_shape(&[]), DType::I64);

        let result = translator.translate(&make_node_with_axis(0), &[data, indices], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_gather_input_validation_no_inputs() {
        let translator = GatherTranslator;
        let err = translator.input_requirement().validate(0, "Gather");
        assert!(err.is_err());
    }

    #[test]
    fn test_gather_input_validation_one_input() {
        let translator = GatherTranslator;
        let err = translator.input_requirement().validate(1, "Gather");
        assert!(err.is_err());
    }

    #[test]
    fn test_gather_input_validation_too_many() {
        let translator = GatherTranslator;
        let err = translator.input_requirement().validate(3, "Gather");
        assert!(err.is_err());
    }
}
