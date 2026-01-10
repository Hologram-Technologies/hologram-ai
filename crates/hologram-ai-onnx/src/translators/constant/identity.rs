//! Identity operation translator.

use hologram::ir::{GraphBuilder, NodeIndex};
use crate::proto::NodeProto;
use crate::translators::{OnnxTranslator, InputRequirement, TranslationError};

/// Translator for ONNX Identity operation.
///
/// Identity is a no-op that passes the input through unchanged.
/// It's commonly used in ONNX for:
/// - Connecting graph outputs to intermediate tensors
/// - Creating explicit copies of tensors
/// - Serving as placeholders in conditional graphs
///
/// # Inputs
/// - input: The tensor to pass through
///
/// # Outputs
/// - output: Same as input (no copy is made)
#[derive(Debug, Default)]
pub struct IdentityTranslator;

impl OnnxTranslator for IdentityTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "Identity"
    }

    fn input_requirement(&self) -> InputRequirement {
        InputRequirement::Exact(1)
    }

    fn translate(
        &self,
        _node: &NodeProto,
        inputs: &[NodeIndex],
        _builder: &mut GraphBuilder,
    ) -> Result<Vec<NodeIndex>, TranslationError> {
        // Identity is a no-op, just return the input unchanged
        Ok(vec![inputs[0]])
    }

    fn supports_constant_folding(&self) -> bool {
        true
    }

    fn constant_fold(
        &self,
        _node: &NodeProto,
        constant_inputs: &[&[u8]],
    ) -> Option<Vec<u8>> {
        // Identity just returns the input as-is
        constant_inputs.first().map(|data| data.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram::ir::{DType, Shape};

    fn make_node() -> NodeProto {
        NodeProto {
            name: "identity_test".to_string(),
            op_type: "Identity".to_string(),
            ..Default::default()
        }
    }

    // ===== Valid Input Tests =====

    #[test]
    fn test_identity_single_input() {
        let translator = IdentityTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);

        let result = translator.translate(&make_node(), &[x], &mut builder);
        assert!(result.is_ok());

        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], x); // Should return same node
    }

    #[test]
    fn test_identity_1d_tensor() {
        let translator = IdentityTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[100]), DType::F32);

        let result = translator.translate(&make_node(), &[x], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap()[0], x);
    }

    #[test]
    fn test_identity_4d_tensor() {
        let translator = IdentityTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[1, 3, 224, 224]), DType::F32);

        let result = translator.translate(&make_node(), &[x], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap()[0], x);
    }

    #[test]
    fn test_identity_scalar() {
        let translator = IdentityTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[]), DType::F32);

        let result = translator.translate(&make_node(), &[x], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap()[0], x);
    }

    #[test]
    fn test_identity_different_dtypes() {
        let translator = IdentityTranslator;

        for dtype in [DType::F32, DType::F64, DType::I32, DType::I64, DType::Bool] {
            let mut builder = GraphBuilder::new();
            let x = builder.input("x", Shape::static_shape(&[2, 3]), dtype);

            let result = translator.translate(&make_node(), &[x], &mut builder);
            assert!(result.is_ok());
            assert_eq!(result.unwrap()[0], x);
        }
    }

    #[test]
    fn test_identity_preserves_shape() {
        let translator = IdentityTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[5, 10, 15]), DType::F32);

        let result = translator.translate(&make_node(), &[x], &mut builder);
        assert!(result.is_ok());

        let output = result.unwrap()[0];
        let output_node = builder.graph().node(output).unwrap();
        assert_eq!(output_node.shape.rank(), 3);
    }

    // ===== Constant Folding Tests =====

    #[test]
    fn test_identity_constant_fold() {
        let translator = IdentityTranslator;
        let input_bytes: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8];

        let result = translator.constant_fold(&make_node(), &[input_bytes.as_slice()]);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), input_bytes);
    }

    #[test]
    fn test_identity_constant_fold_f32() {
        let translator = IdentityTranslator;
        let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let input_bytes: &[u8] = bytemuck::cast_slice(&input);

        let result = translator.constant_fold(&make_node(), &[input_bytes]);
        assert!(result.is_some());

        let output_bytes = result.unwrap();
        let output: &[f32] = bytemuck::cast_slice(&output_bytes);
        assert_eq!(output, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_identity_constant_fold_empty() {
        let translator = IdentityTranslator;
        let result = translator.constant_fold(&make_node(), &[]);
        assert!(result.is_none());
    }

    // ===== Invalid Input Tests =====

    #[test]
    fn test_identity_no_inputs() {
        let translator = IdentityTranslator;
        let err = translator.input_requirement().validate(0, "Identity");
        assert!(err.is_err());
        assert!(matches!(
            err.unwrap_err(),
            TranslationError::WrongInputCount { expected: 1, got: 0, .. }
        ));
    }

    #[test]
    fn test_identity_too_many_inputs() {
        let translator = IdentityTranslator;
        let err = translator.input_requirement().validate(2, "Identity");
        assert!(err.is_err());
        assert!(matches!(
            err.unwrap_err(),
            TranslationError::WrongInputCount { expected: 1, got: 2, .. }
        ));
    }

    #[test]
    fn test_identity_three_inputs() {
        let translator = IdentityTranslator;
        let err = translator.input_requirement().validate(3, "Identity");
        assert!(err.is_err());
    }

    // ===== Trait Method Tests =====

    #[test]
    fn test_identity_op_type() {
        let translator = IdentityTranslator;
        assert_eq!(translator.onnx_op_type(), "Identity");
    }

    #[test]
    fn test_identity_supports_folding() {
        let translator = IdentityTranslator;
        assert!(translator.supports_constant_folding());
    }

    #[test]
    fn test_input_requirement() {
        let translator = IdentityTranslator;
        let req = translator.input_requirement();
        assert!(matches!(req, InputRequirement::Exact(1)));
        assert!(!req.accepts_zero());
    }

    #[test]
    #[allow(clippy::default_constructed_unit_structs)]
    fn test_default_trait() {
        // Intentionally testing Default trait implementation
        let translator = IdentityTranslator::default();
        assert_eq!(translator.onnx_op_type(), "Identity");
    }
}
