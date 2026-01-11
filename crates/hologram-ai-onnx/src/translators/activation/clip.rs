//! Clip activation translator.

use crate::proto::NodeProto;
use crate::translators::{InputRequirement, OnnxTranslator, TranslationError};
use hologram::ir::{GraphBuilder, NodeIndex};

/// Translator for ONNX Clip operation.
///
/// Clip(x, min, max) = min(max(x, min_val), max_val)
///
/// In opset 11+, min and max are optional tensor inputs.
/// In older opsets, they were attributes.
#[derive(Debug, Default)]
pub struct ClipTranslator;

impl OnnxTranslator for ClipTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "Clip"
    }

    fn input_requirement(&self) -> InputRequirement {
        // Opset 11+: 1-3 inputs (x, optional min, optional max)
        InputRequirement::Range(1, 3)
    }

    fn translate(
        &self,
        _node: &NodeProto,
        inputs: &[NodeIndex],
        builder: &mut GraphBuilder,
    ) -> Result<Vec<NodeIndex>, TranslationError> {
        // Currently, we pass None for min/max since extracting constant values
        // from input nodes requires constant folding infrastructure
        let result = builder
            .clip(inputs[0], None, None)
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
            name: "clip_test".to_string(),
            op_type: "Clip".to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn test_clip_one_input() {
        let translator = ClipTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);

        let result = translator.translate(&make_node(), &[x], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_clip_with_min_max() {
        let translator = ClipTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);
        let min_val = builder.input("min", Shape::static_shape(&[]), DType::F32);
        let max_val = builder.input("max", Shape::static_shape(&[]), DType::F32);

        let result = translator.translate(&make_node(), &[x, min_val, max_val], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_clip_with_min_only() {
        let translator = ClipTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[10]), DType::F32);
        let min_val = builder.input("min", Shape::static_shape(&[]), DType::F32);

        let result = translator.translate(&make_node(), &[x, min_val], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_clip_no_inputs() {
        let translator = ClipTranslator;
        let err = translator.input_requirement().validate(0, "Clip");
        assert!(err.is_err());
    }

    #[test]
    fn test_clip_too_many_inputs() {
        let translator = ClipTranslator;
        let err = translator.input_requirement().validate(4, "Clip");
        assert!(err.is_err());
    }
}
