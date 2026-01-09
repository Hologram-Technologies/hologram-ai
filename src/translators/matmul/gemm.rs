//! Gemm operation translator.

use hologram::ir::{GraphBuilder, NodeIndex};
use crate::proto::NodeProto;
use crate::translators::{OnnxTranslator, OnnxAttributes, InputRequirement, TranslationError};

/// Translator for ONNX Gemm operation.
///
/// General Matrix Multiplication:
/// Y = alpha * A' @ B' + beta * C
///
/// Where A' and B' are optionally transposed versions of A and B.
///
/// # Inputs
///
/// - A: First input matrix (required)
/// - B: Second input matrix (required)
/// - C: Bias matrix (optional, broadcasts to output shape)
///
/// # Attributes
///
/// - `alpha` (float, default 1.0): Scalar multiplier for A @ B
/// - `beta` (float, default 1.0): Scalar multiplier for C
/// - `transA` (int, default 0): If 1, transpose A before multiplication
/// - `transB` (int, default 0): If 1, transpose B before multiplication
#[derive(Debug, Default)]
pub struct GemmTranslator;

impl OnnxTranslator for GemmTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "Gemm"
    }

    fn input_requirement(&self) -> InputRequirement {
        InputRequirement::Range(2, 3)
    }

    fn translate(
        &self,
        node: &NodeProto,
        inputs: &[NodeIndex],
        builder: &mut GraphBuilder,
    ) -> Result<Vec<NodeIndex>, TranslationError> {
        let a = inputs[0];
        let b = inputs[1];
        let c = inputs.get(2).copied();

        // Parse attributes with defaults
        let alpha = node.get_float_or("alpha", 1.0);
        let beta = node.get_float_or("beta", 1.0);
        let trans_a = node.get_int_or("transA", 0) != 0;
        let trans_b = node.get_int_or("transB", 0) != 0;

        // Use builder's gemm function which handles all the logic:
        // - Transpose A if transA is set
        // - Transpose B if transB is set
        // - Multiply by alpha
        // - Add beta * C if C is provided
        let result = builder
            .gemm(a, b, c, alpha, beta, trans_a, trans_b)
            .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;

        Ok(vec![result])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::AttributeProto;
    use crate::proto::attribute_proto::AttributeType;
    use hologram::ir::{Dim, DType, Shape};

    fn make_node() -> NodeProto {
        NodeProto {
            name: "gemm_test".to_string(),
            op_type: "Gemm".to_string(),
            ..Default::default()
        }
    }

    fn make_float_attr(name: &str, value: f32) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            f: value,
            r#type: AttributeType::Float as i32,
            ..Default::default()
        }
    }

    fn make_int_attr(name: &str, value: i64) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            i: value,
            r#type: AttributeType::Int as i32,
            ..Default::default()
        }
    }

    fn make_node_with_attrs(attrs: Vec<AttributeProto>) -> NodeProto {
        NodeProto {
            name: "gemm_test".to_string(),
            op_type: "Gemm".to_string(),
            attribute: attrs,
            ..Default::default()
        }
    }

    #[test]
    fn test_gemm_basic() {
        let translator = GemmTranslator;
        let mut builder = GraphBuilder::new();
        let a = builder.input("a", Shape::static_shape(&[2, 3]), DType::F32);
        let b = builder.input("b", Shape::static_shape(&[3, 4]), DType::F32);

        let result = translator.translate(&make_node(), &[a, b], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn test_gemm_with_bias() {
        let translator = GemmTranslator;
        let mut builder = GraphBuilder::new();
        let a = builder.input("a", Shape::static_shape(&[2, 3]), DType::F32);
        let b = builder.input("b", Shape::static_shape(&[3, 4]), DType::F32);
        let c = builder.input("c", Shape::static_shape(&[2, 4]), DType::F32);

        let result = translator.translate(&make_node(), &[a, b, c], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn test_gemm_with_1d_bias() {
        let translator = GemmTranslator;
        let mut builder = GraphBuilder::new();
        let a = builder.input("a", Shape::static_shape(&[2, 3]), DType::F32);
        let b = builder.input("b", Shape::static_shape(&[3, 4]), DType::F32);
        // 1D bias broadcasts across rows
        let c = builder.input("c", Shape::static_shape(&[4]), DType::F32);

        let result = translator.translate(&make_node(), &[a, b, c], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn test_gemm_with_transpose_a() {
        let translator = GemmTranslator;
        let mut builder = GraphBuilder::new();
        // A is (3, 2) but will be transposed to (2, 3)
        let a = builder.input("a", Shape::static_shape(&[3, 2]), DType::F32);
        let b = builder.input("b", Shape::static_shape(&[3, 4]), DType::F32);

        let node = make_node_with_attrs(vec![make_int_attr("transA", 1)]);
        let result = translator.translate(&node, &[a, b], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn test_gemm_with_transpose_b() {
        let translator = GemmTranslator;
        let mut builder = GraphBuilder::new();
        let a = builder.input("a", Shape::static_shape(&[2, 3]), DType::F32);
        // B is (4, 3) but will be transposed to (3, 4)
        let b = builder.input("b", Shape::static_shape(&[4, 3]), DType::F32);

        let node = make_node_with_attrs(vec![make_int_attr("transB", 1)]);
        let result = translator.translate(&node, &[a, b], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn test_gemm_with_both_transposes() {
        let translator = GemmTranslator;
        let mut builder = GraphBuilder::new();
        // A: (3, 2) -> transposed to (2, 3)
        // B: (4, 3) -> transposed to (3, 4)
        // Result: (2, 3) @ (3, 4) = (2, 4)
        let a = builder.input("a", Shape::static_shape(&[3, 2]), DType::F32);
        let b = builder.input("b", Shape::static_shape(&[4, 3]), DType::F32);

        let node = make_node_with_attrs(vec![
            make_int_attr("transA", 1),
            make_int_attr("transB", 1),
        ]);
        let result = translator.translate(&node, &[a, b], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn test_gemm_with_alpha() {
        let translator = GemmTranslator;
        let mut builder = GraphBuilder::new();
        let a = builder.input("a", Shape::static_shape(&[2, 3]), DType::F32);
        let b = builder.input("b", Shape::static_shape(&[3, 4]), DType::F32);

        let node = make_node_with_attrs(vec![make_float_attr("alpha", 2.0)]);
        let result = translator.translate(&node, &[a, b], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn test_gemm_with_beta() {
        let translator = GemmTranslator;
        let mut builder = GraphBuilder::new();
        let a = builder.input("a", Shape::static_shape(&[2, 3]), DType::F32);
        let b = builder.input("b", Shape::static_shape(&[3, 4]), DType::F32);
        let c = builder.input("c", Shape::static_shape(&[2, 4]), DType::F32);

        let node = make_node_with_attrs(vec![make_float_attr("beta", 0.5)]);
        let result = translator.translate(&node, &[a, b, c], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn test_gemm_with_all_attrs() {
        let translator = GemmTranslator;
        let mut builder = GraphBuilder::new();
        // Full configuration: transA, transB, alpha, beta
        let a = builder.input("a", Shape::static_shape(&[3, 2]), DType::F32);
        let b = builder.input("b", Shape::static_shape(&[4, 3]), DType::F32);
        let c = builder.input("c", Shape::static_shape(&[2, 4]), DType::F32);

        let node = make_node_with_attrs(vec![
            make_float_attr("alpha", 2.0),
            make_float_attr("beta", 0.5),
            make_int_attr("transA", 1),
            make_int_attr("transB", 1),
        ]);
        let result = translator.translate(&node, &[a, b, c], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn test_gemm_zero_alpha() {
        let translator = GemmTranslator;
        let mut builder = GraphBuilder::new();
        let a = builder.input("a", Shape::static_shape(&[2, 3]), DType::F32);
        let b = builder.input("b", Shape::static_shape(&[3, 4]), DType::F32);
        let c = builder.input("c", Shape::static_shape(&[2, 4]), DType::F32);

        // alpha=0 means matmul result is zeroed, only bias matters
        let node = make_node_with_attrs(vec![make_float_attr("alpha", 0.0)]);
        let result = translator.translate(&node, &[a, b, c], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn test_gemm_zero_beta() {
        let translator = GemmTranslator;
        let mut builder = GraphBuilder::new();
        let a = builder.input("a", Shape::static_shape(&[2, 3]), DType::F32);
        let b = builder.input("b", Shape::static_shape(&[3, 4]), DType::F32);
        let c = builder.input("c", Shape::static_shape(&[2, 4]), DType::F32);

        // beta=0 means bias is zeroed out
        let node = make_node_with_attrs(vec![make_float_attr("beta", 0.0)]);
        let result = translator.translate(&node, &[a, b, c], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn test_gemm_symbolic_shapes() {
        let translator = GemmTranslator;
        let mut builder = GraphBuilder::new();
        // Symbolic shapes using Shape::new with Dim::Symbolic
        let a = builder.input(
            "a",
            Shape::new(vec![
                Dim::Symbolic("batch".into()),
                Dim::Symbolic("hidden".into()),
            ]),
            DType::F32,
        );
        let b = builder.input(
            "b",
            Shape::new(vec![
                Dim::Symbolic("hidden".into()),
                Dim::Symbolic("output".into()),
            ]),
            DType::F32,
        );

        let result = translator.translate(&make_node(), &[a, b], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn test_gemm_op_type() {
        let translator = GemmTranslator;
        assert_eq!(translator.onnx_op_type(), "Gemm");
    }

    #[test]
    fn test_gemm_input_requirement() {
        let translator = GemmTranslator;
        let req = translator.input_requirement();
        assert_eq!(req, InputRequirement::Range(2, 3));
    }

    #[test]
    fn test_gemm_no_inputs_error() {
        let translator = GemmTranslator;
        let err = translator.input_requirement().validate(0, "Gemm");
        assert!(err.is_err());
        assert!(matches!(
            err.unwrap_err(),
            TranslationError::InputCountOutOfRange { min: 2, max: 3, got: 0, .. }
        ));
    }

    #[test]
    fn test_gemm_one_input_error() {
        let translator = GemmTranslator;
        let err = translator.input_requirement().validate(1, "Gemm");
        assert!(err.is_err());
        assert!(matches!(
            err.unwrap_err(),
            TranslationError::InputCountOutOfRange { min: 2, max: 3, got: 1, .. }
        ));
    }

    #[test]
    fn test_gemm_four_inputs_error() {
        let translator = GemmTranslator;
        let err = translator.input_requirement().validate(4, "Gemm");
        assert!(err.is_err());
        assert!(matches!(
            err.unwrap_err(),
            TranslationError::InputCountOutOfRange { min: 2, max: 3, got: 4, .. }
        ));
    }

    #[test]
    fn test_gemm_two_inputs_valid() {
        let translator = GemmTranslator;
        let result = translator.input_requirement().validate(2, "Gemm");
        assert!(result.is_ok());
    }

    #[test]
    fn test_gemm_three_inputs_valid() {
        let translator = GemmTranslator;
        let result = translator.input_requirement().validate(3, "Gemm");
        assert!(result.is_ok());
    }

    #[test]
    fn test_gemm_large_matrices() {
        let translator = GemmTranslator;
        let mut builder = GraphBuilder::new();
        let a = builder.input("a", Shape::static_shape(&[512, 1024]), DType::F32);
        let b = builder.input("b", Shape::static_shape(&[1024, 2048]), DType::F32);
        let c = builder.input("c", Shape::static_shape(&[512, 2048]), DType::F32);

        let result = translator.translate(&make_node(), &[a, b, c], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn test_gemm_negative_alpha() {
        let translator = GemmTranslator;
        let mut builder = GraphBuilder::new();
        let a = builder.input("a", Shape::static_shape(&[2, 3]), DType::F32);
        let b = builder.input("b", Shape::static_shape(&[3, 4]), DType::F32);

        let node = make_node_with_attrs(vec![make_float_attr("alpha", -1.0)]);
        let result = translator.translate(&node, &[a, b], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn test_gemm_f16_dtype() {
        let translator = GemmTranslator;
        let mut builder = GraphBuilder::new();
        let a = builder.input("a", Shape::static_shape(&[2, 3]), DType::F16);
        let b = builder.input("b", Shape::static_shape(&[3, 4]), DType::F16);

        let result = translator.translate(&make_node(), &[a, b], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);
    }
}
