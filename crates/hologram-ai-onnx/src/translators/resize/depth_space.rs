//! DepthToSpace and SpaceToDepth operation translators.

use hologram::ir::{GraphBuilder, NodeIndex};
use crate::proto::NodeProto;
use crate::translators::{OnnxTranslator, InputRequirement, TranslationError};

/// Translator for ONNX DepthToSpace operation.
///
/// DepthToSpace rearranges data from the depth (channel) dimension into
/// spatial dimensions (height and width).
///
/// # ONNX Specification
///
/// - Inputs: input [N, C, H, W]
/// - Attributes:
///   - blocksize (required): The size of the blocks to rearrange
///   - mode (default: "DCR"): "DCR" or "CRD"
/// - Output: [N, C/(blocksize^2), H*blocksize, W*blocksize]
///
/// # Note
///
/// This operation is not currently supported in hologram-ir.
#[derive(Debug, Default)]
pub struct DepthToSpaceTranslator;

impl OnnxTranslator for DepthToSpaceTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "DepthToSpace"
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
        Err(TranslationError::unsupported_op("DepthToSpace", 13))
    }
}

/// Translator for ONNX SpaceToDepth operation.
///
/// SpaceToDepth rearranges data from spatial dimensions (height and width)
/// into the depth (channel) dimension.
///
/// # ONNX Specification
///
/// - Inputs: input [N, C, H, W]
/// - Attributes:
///   - blocksize (required): The size of the blocks to rearrange
/// - Output: [N, C*blocksize^2, H/blocksize, W/blocksize]
///
/// # Note
///
/// This operation is not currently supported in hologram-ir.
#[derive(Debug, Default)]
pub struct SpaceToDepthTranslator;

impl OnnxTranslator for SpaceToDepthTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "SpaceToDepth"
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
        Err(TranslationError::unsupported_op("SpaceToDepth", 13))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram::ir::{DType, Shape};
    use crate::proto::AttributeProto;

    fn make_node(op_type: &str, blocksize: i64) -> NodeProto {
        NodeProto {
            name: format!("{}_test", op_type.to_lowercase()),
            op_type: op_type.to_string(),
            attribute: vec![AttributeProto {
                name: "blocksize".to_string(),
                i: blocksize,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    // DepthToSpace tests

    #[test]
    fn test_depth_to_space_unsupported() {
        let translator = DepthToSpaceTranslator;
        let mut builder = GraphBuilder::new();
        let data = builder.input("data", Shape::static_shape(&[1, 12, 16, 16]), DType::F32);

        let node = make_node("DepthToSpace", 2);
        let result = translator.translate(&node, &[data], &mut builder);
        assert!(result.is_err());
        assert!(result.unwrap_err().is_unsupported_op());
    }

    #[test]
    fn test_depth_to_space_input_validation() {
        let translator = DepthToSpaceTranslator;

        // 0 inputs should fail
        let err = translator.input_requirement().validate(0, "DepthToSpace");
        assert!(err.is_err());

        // 1 input should pass
        assert!(translator.input_requirement().validate(1, "DepthToSpace").is_ok());

        // 2 inputs should fail
        let err = translator.input_requirement().validate(2, "DepthToSpace");
        assert!(err.is_err());
    }

    // SpaceToDepth tests

    #[test]
    fn test_space_to_depth_unsupported() {
        let translator = SpaceToDepthTranslator;
        let mut builder = GraphBuilder::new();
        let data = builder.input("data", Shape::static_shape(&[1, 3, 32, 32]), DType::F32);

        let node = make_node("SpaceToDepth", 2);
        let result = translator.translate(&node, &[data], &mut builder);
        assert!(result.is_err());
        assert!(result.unwrap_err().is_unsupported_op());
    }

    #[test]
    fn test_space_to_depth_input_validation() {
        let translator = SpaceToDepthTranslator;

        // 0 inputs should fail
        let err = translator.input_requirement().validate(0, "SpaceToDepth");
        assert!(err.is_err());

        // 1 input should pass
        assert!(translator.input_requirement().validate(1, "SpaceToDepth").is_ok());

        // 2 inputs should fail
        let err = translator.input_requirement().validate(2, "SpaceToDepth");
        assert!(err.is_err());
    }
}
