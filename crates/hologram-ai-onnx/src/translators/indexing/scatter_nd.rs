//! ScatterND operation translator.

use crate::proto::NodeProto;
use crate::translators::{InputRequirement, OnnxTranslator, TranslationError};
use hologram::ir::{GraphBuilder, NodeIndex};

/// Translator for ONNX ScatterND operation.
///
/// ScatterND creates a copy of the data tensor, then updates specific positions
/// (specified by indices) with new values (from updates tensor).
/// This is the inverse operation of GatherND.
///
/// # ONNX Specification
///
/// - Inputs: data, indices, updates
///   - data: Base tensor of rank r >= 1
///   - indices: Integer tensor of rank q >= 1 specifying positions
///   - updates: New values of rank q + r - indices.shape[-1] - 1
/// - Output: same shape as data
///
/// # Example
///
/// ```text
/// data = [1, 2, 3, 4, 5]
/// indices = [[1], [3]]
/// updates = [10, 40]
/// output = [1, 10, 3, 40, 5]
/// ```
#[derive(Debug, Default)]
pub struct ScatterNDTranslator;

impl OnnxTranslator for ScatterNDTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "ScatterND"
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
        let data = inputs[0];
        let indices = inputs[1];
        let updates = inputs[2];

        let result = builder
            .scatter_nd(data, indices, updates)
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
            name: "scatter_nd_test".to_string(),
            op_type: "ScatterND".to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn test_scatter_nd_1d() {
        let translator = ScatterNDTranslator;
        let mut builder = GraphBuilder::new();
        let data = builder.input("data", Shape::static_shape(&[8]), DType::F32);
        let indices = builder.input("indices", Shape::static_shape(&[4, 1]), DType::I64);
        let updates = builder.input("updates", Shape::static_shape(&[4]), DType::F32);

        let result = translator.translate(&make_node(), &[data, indices, updates], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_scatter_nd_2d() {
        let translator = ScatterNDTranslator;
        let mut builder = GraphBuilder::new();
        let data = builder.input("data", Shape::static_shape(&[4, 4]), DType::F32);
        let indices = builder.input("indices", Shape::static_shape(&[2, 2]), DType::I64);
        let updates = builder.input("updates", Shape::static_shape(&[2]), DType::F32);

        let result = translator.translate(&make_node(), &[data, indices, updates], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_scatter_nd_3d() {
        let translator = ScatterNDTranslator;
        let mut builder = GraphBuilder::new();
        let data = builder.input("data", Shape::static_shape(&[2, 3, 4]), DType::F32);
        let indices = builder.input("indices", Shape::static_shape(&[2, 3]), DType::I64);
        let updates = builder.input("updates", Shape::static_shape(&[2]), DType::F32);

        let result = translator.translate(&make_node(), &[data, indices, updates], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_scatter_nd_input_validation_too_few() {
        let translator = ScatterNDTranslator;
        let err = translator.input_requirement().validate(2, "ScatterND");
        assert!(err.is_err());
    }

    #[test]
    fn test_scatter_nd_input_validation_too_many() {
        let translator = ScatterNDTranslator;
        let err = translator.input_requirement().validate(4, "ScatterND");
        assert!(err.is_err());
    }

    #[test]
    fn test_scatter_nd_input_validation_correct() {
        let translator = ScatterNDTranslator;
        assert!(
            translator
                .input_requirement()
                .validate(3, "ScatterND")
                .is_ok()
        );
    }
}
