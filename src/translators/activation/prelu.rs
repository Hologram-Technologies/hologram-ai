//! PReLU activation translator.

use hologram::ir::{GraphBuilder, NodeIndex, NodeOp, ConstantData, Shape};
use crate::proto::NodeProto;
use crate::translators::{OnnxTranslator, InputRequirement, TranslationError};

/// Translator for ONNX PRelu operation.
///
/// PRelu(x, slope) = x if x >= 0 else slope * x
///
/// The slope is a learned parameter tensor (second input).
#[derive(Debug, Default)]
pub struct PReluTranslator;

impl OnnxTranslator for PReluTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "PRelu"
    }

    fn input_requirement(&self) -> InputRequirement {
        // Requires x and slope
        InputRequirement::Exact(2)
    }

    fn translate(
        &self,
        _node: &NodeProto,
        inputs: &[NodeIndex],
        builder: &mut GraphBuilder,
    ) -> Result<Vec<NodeIndex>, TranslationError> {
        // PReLU(x, slope) = max(0, x) + slope * min(0, x)
        let zero = builder.constant(
            ConstantData::F32(vec![0.0]),
            Shape::static_shape(&[1]),
        );

        let pos_part = builder
            .binary(NodeOp::Max, inputs[0], zero)
            .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;
        let neg_part = builder
            .binary(NodeOp::Min, inputs[0], zero)
            .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;
        let scaled_neg = builder
            .mul(inputs[1], neg_part)
            .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;
        let result = builder
            .add(pos_part, scaled_neg)
            .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;

        Ok(vec![result])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram::ir::DType;

    fn make_node() -> NodeProto {
        NodeProto {
            name: "prelu_test".to_string(),
            op_type: "PRelu".to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn test_prelu_translation() {
        let translator = PReluTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);
        let slope = builder.input("slope", Shape::static_shape(&[3]), DType::F32);

        let result = translator.translate(&make_node(), &[x, slope], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_prelu_with_scalar_slope() {
        let translator = PReluTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[4, 4]), DType::F32);
        let slope = builder.input("slope", Shape::static_shape(&[1]), DType::F32);

        let result = translator.translate(&make_node(), &[x, slope], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_prelu_no_slope() {
        let translator = PReluTranslator;
        let err = translator.input_requirement().validate(1, "PRelu");
        assert!(err.is_err());
    }

    #[test]
    fn test_prelu_no_inputs() {
        let translator = PReluTranslator;
        let err = translator.input_requirement().validate(0, "PRelu");
        assert!(err.is_err());
    }

    #[test]
    fn test_prelu_too_many_inputs() {
        let translator = PReluTranslator;
        let err = translator.input_requirement().validate(3, "PRelu");
        assert!(err.is_err());
    }
}
