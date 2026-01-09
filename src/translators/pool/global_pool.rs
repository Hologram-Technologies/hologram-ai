//! Global pooling operation translators.

use hologram::ir::{GraphBuilder, NodeIndex, NodeOp};
use crate::proto::NodeProto;
use crate::translators::{OnnxTranslator, InputRequirement, TranslationError};

/// Translator for ONNX GlobalAveragePool operation.
///
/// GlobalAveragePool computes the average of all spatial dimensions,
/// producing a tensor with spatial dimensions of size 1.
///
/// # ONNX Specification
///
/// - Inputs: X [N, C, H, W]
/// - Outputs: Y [N, C, 1, 1]
#[derive(Debug, Default)]
pub struct GlobalAveragePoolTranslator;

impl OnnxTranslator for GlobalAveragePoolTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "GlobalAveragePool"
    }

    fn input_requirement(&self) -> InputRequirement {
        InputRequirement::Exact(1)
    }

    fn translate(
        &self,
        _node: &NodeProto,
        inputs: &[NodeIndex],
        builder: &mut GraphBuilder,
    ) -> Result<Vec<NodeIndex>, TranslationError> {
        let input = inputs[0];

        let result = builder
            .unary(NodeOp::GlobalAvgPool, input)
            .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;

        Ok(vec![result])
    }
}

/// Translator for ONNX GlobalMaxPool operation.
///
/// GlobalMaxPool computes the max of all spatial dimensions,
/// producing a tensor with spatial dimensions of size 1.
///
/// # ONNX Specification
///
/// - Inputs: X [N, C, H, W]
/// - Outputs: Y [N, C, 1, 1]
///
/// # Note
///
/// This operation is not currently supported in hologram-ir.
#[derive(Debug, Default)]
pub struct GlobalMaxPoolTranslator;

impl OnnxTranslator for GlobalMaxPoolTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "GlobalMaxPool"
    }

    fn input_requirement(&self) -> InputRequirement {
        InputRequirement::Exact(1)
    }

    fn translate(
        &self,
        _node: &NodeProto,
        _inputs: &[NodeIndex],
        _builder: &mut GraphBuilder,
    ) -> Result<Vec<NodeIndex>, TranslationError> {
        // GlobalMaxPool is not currently supported in hologram-ir
        Err(TranslationError::unsupported_op("GlobalMaxPool", 13))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram::ir::{DType, Shape};

    fn make_node(op_type: &str) -> NodeProto {
        NodeProto {
            name: format!("{}_test", op_type.to_lowercase()),
            op_type: op_type.to_string(),
            ..Default::default()
        }
    }

    // GlobalAveragePool tests

    #[test]
    fn test_global_avg_pool_basic() {
        let translator = GlobalAveragePoolTranslator;
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[1, 64, 32, 32]), DType::F32);

        let result = translator.translate(&make_node("GlobalAveragePool"), &[input], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_global_avg_pool_large_channels() {
        let translator = GlobalAveragePoolTranslator;
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[2, 512, 7, 7]), DType::F32);

        let result = translator.translate(&make_node("GlobalAveragePool"), &[input], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_global_avg_pool_small_spatial() {
        let translator = GlobalAveragePoolTranslator;
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[1, 64, 1, 1]), DType::F32);

        let result = translator.translate(&make_node("GlobalAveragePool"), &[input], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_global_avg_pool_input_validation() {
        let translator = GlobalAveragePoolTranslator;

        // 0 inputs should fail
        let err = translator.input_requirement().validate(0, "GlobalAveragePool");
        assert!(err.is_err());

        // 1 input should pass
        assert!(translator.input_requirement().validate(1, "GlobalAveragePool").is_ok());

        // 2 inputs should fail
        let err = translator.input_requirement().validate(2, "GlobalAveragePool");
        assert!(err.is_err());
    }

    // GlobalMaxPool tests

    #[test]
    fn test_global_max_pool_unsupported() {
        let translator = GlobalMaxPoolTranslator;
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[1, 64, 32, 32]), DType::F32);

        let result = translator.translate(&make_node("GlobalMaxPool"), &[input], &mut builder);
        assert!(result.is_err());
        assert!(result.unwrap_err().is_unsupported_op());
    }

    #[test]
    fn test_global_max_pool_input_validation() {
        let translator = GlobalMaxPoolTranslator;

        // 0 inputs should fail
        let err = translator.input_requirement().validate(0, "GlobalMaxPool");
        assert!(err.is_err());

        // 1 input should pass
        assert!(translator.input_requirement().validate(1, "GlobalMaxPool").is_ok());

        // 2 inputs should fail
        let err = translator.input_requirement().validate(2, "GlobalMaxPool");
        assert!(err.is_err());
    }
}
