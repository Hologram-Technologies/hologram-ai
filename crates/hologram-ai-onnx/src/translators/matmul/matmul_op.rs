//! MatMul operation translator.

use crate::proto::NodeProto;
use crate::translators::{InputRequirement, OnnxTranslator, TranslationError};
use hologram::ir::{GraphBuilder, NodeIndex};

/// Translator for ONNX MatMul operation.
///
/// MatMul(A, B) performs matrix multiplication of A and B.
///
/// Supports:
/// - 2D matrices: (M, K) @ (K, N) -> (M, N)
/// - Batched: (B, M, K) @ (B, K, N) -> (B, M, N)
/// - Broadcasting: batch dimensions follow numpy broadcasting rules
#[derive(Debug, Default)]
pub struct MatMulTranslator;

impl OnnxTranslator for MatMulTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "MatMul"
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
        let result = builder
            .matmul(inputs[0], inputs[1])
            .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;
        Ok(vec![result])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram::ir::{DType, Dim, Shape};

    fn make_node() -> NodeProto {
        NodeProto {
            name: "matmul_test".to_string(),
            op_type: "MatMul".to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn test_matmul_2d_matrices() {
        let translator = MatMulTranslator;
        let mut builder = GraphBuilder::new();
        let a = builder.input("a", Shape::static_shape(&[2, 3]), DType::F32);
        let b = builder.input("b", Shape::static_shape(&[3, 4]), DType::F32);

        let result = translator.translate(&make_node(), &[a, b], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn test_matmul_batched() {
        let translator = MatMulTranslator;
        let mut builder = GraphBuilder::new();
        // Batched matmul: (batch, M, K) @ (batch, K, N) -> (batch, M, N)
        let a = builder.input("a", Shape::static_shape(&[8, 16, 32]), DType::F32);
        let b = builder.input("b", Shape::static_shape(&[8, 32, 64]), DType::F32);

        let result = translator.translate(&make_node(), &[a, b], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn test_matmul_symbolic_batch() {
        let translator = MatMulTranslator;
        let mut builder = GraphBuilder::new();
        // Symbolic batch dimension using Shape::new with Dim::Symbolic
        let a = builder.input(
            "a",
            Shape::new(vec![
                Dim::Symbolic("batch".into()),
                Dim::Static(16),
                Dim::Static(32),
            ]),
            DType::F32,
        );
        let b = builder.input(
            "b",
            Shape::new(vec![
                Dim::Symbolic("batch".into()),
                Dim::Static(32),
                Dim::Static(64),
            ]),
            DType::F32,
        );

        let result = translator.translate(&make_node(), &[a, b], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn test_matmul_vector_matrix() {
        let translator = MatMulTranslator;
        let mut builder = GraphBuilder::new();
        // (1, K) @ (K, N) -> (1, N)
        let a = builder.input("a", Shape::static_shape(&[1, 32]), DType::F32);
        let b = builder.input("b", Shape::static_shape(&[32, 64]), DType::F32);

        let result = translator.translate(&make_node(), &[a, b], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn test_matmul_matrix_vector() {
        let translator = MatMulTranslator;
        let mut builder = GraphBuilder::new();
        // (M, K) @ (K, 1) -> (M, 1)
        let a = builder.input("a", Shape::static_shape(&[32, 64]), DType::F32);
        let b = builder.input("b", Shape::static_shape(&[64, 1]), DType::F32);

        let result = translator.translate(&make_node(), &[a, b], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn test_matmul_f16_dtype() {
        let translator = MatMulTranslator;
        let mut builder = GraphBuilder::new();
        let a = builder.input("a", Shape::static_shape(&[2, 3]), DType::F16);
        let b = builder.input("b", Shape::static_shape(&[3, 4]), DType::F16);

        let result = translator.translate(&make_node(), &[a, b], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn test_matmul_op_type() {
        let translator = MatMulTranslator;
        assert_eq!(translator.onnx_op_type(), "MatMul");
    }

    #[test]
    fn test_matmul_input_requirement() {
        let translator = MatMulTranslator;
        let req = translator.input_requirement();
        assert_eq!(req, InputRequirement::Exact(2));
    }

    #[test]
    fn test_matmul_no_inputs_error() {
        let translator = MatMulTranslator;
        let err = translator.input_requirement().validate(0, "MatMul");
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
    fn test_matmul_one_input_error() {
        let translator = MatMulTranslator;
        let err = translator.input_requirement().validate(1, "MatMul");
        assert!(err.is_err());
        assert!(matches!(
            err.unwrap_err(),
            TranslationError::WrongInputCount {
                expected: 2,
                got: 1,
                ..
            }
        ));
    }

    #[test]
    fn test_matmul_three_inputs_error() {
        let translator = MatMulTranslator;
        let err = translator.input_requirement().validate(3, "MatMul");
        assert!(err.is_err());
        assert!(matches!(
            err.unwrap_err(),
            TranslationError::WrongInputCount {
                expected: 2,
                got: 3,
                ..
            }
        ));
    }

    #[test]
    fn test_matmul_large_matrices() {
        let translator = MatMulTranslator;
        let mut builder = GraphBuilder::new();
        // Large matrices
        let a = builder.input("a", Shape::static_shape(&[1024, 2048]), DType::F32);
        let b = builder.input("b", Shape::static_shape(&[2048, 4096]), DType::F32);

        let result = translator.translate(&make_node(), &[a, b], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn test_matmul_square_matrices() {
        let translator = MatMulTranslator;
        let mut builder = GraphBuilder::new();
        let a = builder.input("a", Shape::static_shape(&[256, 256]), DType::F32);
        let b = builder.input("b", Shape::static_shape(&[256, 256]), DType::F32);

        let result = translator.translate(&make_node(), &[a, b], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);
    }
}
