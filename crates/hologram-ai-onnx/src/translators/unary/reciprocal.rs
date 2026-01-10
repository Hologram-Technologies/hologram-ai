//! Reciprocal operation translator.

use crate::proto::NodeProto;
use crate::translators::{InputRequirement, OnnxTranslator, TranslationError};
use hologram::ir::{ConstantData, GraphBuilder, NodeIndex, Shape};

/// Translator for ONNX Reciprocal operation.
///
/// Reciprocal(x) = 1/x (element-wise reciprocal)
///
/// This operation is decomposed into division since hologram IR
/// does not have a native Reciprocal op.
#[derive(Debug, Default)]
pub struct ReciprocalTranslator;

impl OnnxTranslator for ReciprocalTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "Reciprocal"
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
        // Decompose: 1 / x
        let one = builder.constant(ConstantData::F32(vec![1.0]), Shape::static_shape(&[1]));

        let result = builder
            .div(one, inputs[0])
            .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;
        Ok(vec![result])
    }

    fn supports_constant_folding(&self) -> bool {
        true
    }

    fn constant_fold(&self, _node: &NodeProto, constant_inputs: &[&[u8]]) -> Option<Vec<u8>> {
        let input = constant_inputs.first()?;
        let floats: &[f32] = bytemuck::cast_slice(input);
        let result: Vec<f32> = floats.iter().map(|x| 1.0 / x).collect();
        Some(bytemuck::cast_slice(&result).to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram::ir::DType;

    fn make_node() -> NodeProto {
        NodeProto {
            name: "reciprocal_test".to_string(),
            op_type: "Reciprocal".to_string(),
            ..Default::default()
        }
    }

    // ===== Valid Input Tests =====

    #[test]
    fn test_reciprocal_single_input() {
        let translator = ReciprocalTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);

        let result = translator.translate(&make_node(), &[x], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_reciprocal_1d_tensor() {
        let translator = ReciprocalTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[10]), DType::F32);

        let result = translator.translate(&make_node(), &[x], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_reciprocal_4d_tensor() {
        let translator = ReciprocalTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[1, 64, 32, 32]), DType::F32);

        let result = translator.translate(&make_node(), &[x], &mut builder);
        assert!(result.is_ok());
    }

    // ===== Constant Folding Tests =====

    #[test]
    fn test_reciprocal_constant_fold_basic() {
        let translator = ReciprocalTranslator;
        let input: Vec<f32> = vec![1.0, 2.0, 4.0, 5.0, 10.0];
        let input_bytes: &[u8] = bytemuck::cast_slice(&input);

        let result = translator.constant_fold(&make_node(), &[input_bytes]);
        assert!(result.is_some());

        let output_bytes = result.unwrap();
        let output: &[f32] = bytemuck::cast_slice(&output_bytes);
        assert_eq!(output, &[1.0, 0.5, 0.25, 0.2, 0.1]);
    }

    #[test]
    fn test_reciprocal_constant_fold_negative() {
        let translator = ReciprocalTranslator;
        let input: Vec<f32> = vec![-1.0, -2.0, -4.0];
        let input_bytes: &[u8] = bytemuck::cast_slice(&input);

        let result = translator.constant_fold(&make_node(), &[input_bytes]);
        assert!(result.is_some());

        let output_bytes = result.unwrap();
        let output: &[f32] = bytemuck::cast_slice(&output_bytes);
        assert_eq!(output, &[-1.0, -0.5, -0.25]);
    }

    #[test]
    fn test_reciprocal_constant_fold_fractional() {
        let translator = ReciprocalTranslator;
        let input: Vec<f32> = vec![0.5, 0.25, 0.1];
        let input_bytes: &[u8] = bytemuck::cast_slice(&input);

        let result = translator.constant_fold(&make_node(), &[input_bytes]);
        assert!(result.is_some());

        let output_bytes = result.unwrap();
        let output: &[f32] = bytemuck::cast_slice(&output_bytes);
        assert!((output[0] - 2.0).abs() < 1e-6);
        assert!((output[1] - 4.0).abs() < 1e-6);
        assert!((output[2] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_reciprocal_constant_fold_empty() {
        let translator = ReciprocalTranslator;
        let result = translator.constant_fold(&make_node(), &[]);
        assert!(result.is_none());
    }

    // ===== Invalid Input Tests =====

    #[test]
    fn test_reciprocal_no_inputs() {
        let translator = ReciprocalTranslator;
        let err = translator.input_requirement().validate(0, "Reciprocal");
        assert!(err.is_err());
        assert!(matches!(
            err.unwrap_err(),
            TranslationError::WrongInputCount {
                expected: 1,
                got: 0,
                ..
            }
        ));
    }

    #[test]
    fn test_reciprocal_too_many_inputs() {
        let translator = ReciprocalTranslator;
        let err = translator.input_requirement().validate(2, "Reciprocal");
        assert!(err.is_err());
    }
}
