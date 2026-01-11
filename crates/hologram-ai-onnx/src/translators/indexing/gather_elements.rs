//! GatherElements operation translator.

use crate::proto::NodeProto;
use crate::translators::{InputRequirement, OnnxTranslator, TranslationError};
use hologram::ir::{GraphBuilder, NodeIndex};

/// Translator for ONNX GatherElements operation.
///
/// GatherElements gathers elements from data at positions specified by indices.
/// Unlike Gather which selects slices, GatherElements selects individual elements.
///
/// # ONNX Specification
///
/// - Inputs: data, indices
/// - Attributes: axis (default: 0)
/// - Output: same shape as indices
///
/// # Note
///
/// This operation is not currently supported in hologram-ir.
#[derive(Debug, Default)]
pub struct GatherElementsTranslator;

impl OnnxTranslator for GatherElementsTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "GatherElements"
    }

    fn input_requirement(&self) -> InputRequirement {
        InputRequirement::Exact(2)
    }

    fn translate(
        &self,
        _node: &NodeProto,
        _inputs: &[NodeIndex],
        _builder: &mut GraphBuilder,
    ) -> Result<Vec<NodeIndex>, TranslationError> {
        // GatherElements is not currently supported in hologram-ir
        Err(TranslationError::unsupported_op("GatherElements", 13))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram::ir::{DType, Shape};

    fn make_node() -> NodeProto {
        NodeProto {
            name: "gather_elements_test".to_string(),
            op_type: "GatherElements".to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn test_gather_elements_unsupported() {
        let translator = GatherElementsTranslator;
        let mut builder = GraphBuilder::new();
        let data = builder.input("data", Shape::static_shape(&[3, 4, 5]), DType::F32);
        let indices = builder.input("indices", Shape::static_shape(&[3, 4, 5]), DType::I64);

        let result = translator.translate(&make_node(), &[data, indices], &mut builder);
        assert!(result.is_err());
        assert!(result.unwrap_err().is_unsupported_op());
    }

    #[test]
    fn test_gather_elements_input_validation() {
        let translator = GatherElementsTranslator;

        // 0 inputs should fail
        let err = translator.input_requirement().validate(0, "GatherElements");
        assert!(err.is_err());

        // 1 input should fail
        let err = translator.input_requirement().validate(1, "GatherElements");
        assert!(err.is_err());

        // 2 inputs should pass
        assert!(
            translator
                .input_requirement()
                .validate(2, "GatherElements")
                .is_ok()
        );

        // 3 inputs should fail
        let err = translator.input_requirement().validate(3, "GatherElements");
        assert!(err.is_err());
    }
}
