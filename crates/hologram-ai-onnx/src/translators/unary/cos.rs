//! Cos operation translator.

use crate::proto::NodeProto;
use crate::translators::{InputRequirement, OnnxTranslator, TranslationError};
use hologram::ir::{GraphBuilder, NodeIndex, NodeOp};

/// Translator for ONNX Cos operation.
///
/// Cos(x) = cos(x) (element-wise cosine)
#[derive(Debug, Default)]
pub struct CosTranslator;

impl OnnxTranslator for CosTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "Cos"
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
        let result = builder
            .unary(NodeOp::Cos, inputs[0])
            .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;
        Ok(vec![result])
    }

    fn supports_constant_folding(&self) -> bool {
        true
    }

    fn constant_fold(&self, _node: &NodeProto, constant_inputs: &[&[u8]]) -> Option<Vec<u8>> {
        let input = constant_inputs.first()?;
        let floats: &[f32] = bytemuck::cast_slice(input);
        let result: Vec<f32> = floats.iter().map(|x| x.cos()).collect();
        Some(bytemuck::cast_slice(&result).to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram::ir::{DType, Shape};
    use std::f32::consts::PI;

    fn make_node() -> NodeProto {
        NodeProto {
            name: "cos_test".to_string(),
            op_type: "Cos".to_string(),
            ..Default::default()
        }
    }

    // ===== Valid Input Tests =====

    #[test]
    fn test_cos_single_input() {
        let translator = CosTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);

        let result = translator.translate(&make_node(), &[x], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_cos_1d_tensor() {
        let translator = CosTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[10]), DType::F32);

        let result = translator.translate(&make_node(), &[x], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cos_4d_tensor() {
        let translator = CosTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[1, 64, 32, 32]), DType::F32);

        let result = translator.translate(&make_node(), &[x], &mut builder);
        assert!(result.is_ok());
    }

    // ===== Constant Folding Tests =====

    #[test]
    fn test_cos_constant_fold_basic() {
        let translator = CosTranslator;
        let input: Vec<f32> = vec![0.0, PI / 3.0, PI / 4.0, PI / 2.0, PI];
        let input_bytes: &[u8] = bytemuck::cast_slice(&input);

        let result = translator.constant_fold(&make_node(), &[input_bytes]);
        assert!(result.is_some());

        let output_bytes = result.unwrap();
        let output: &[f32] = bytemuck::cast_slice(&output_bytes);
        assert!((output[0] - 1.0).abs() < 1e-6); // cos(0) = 1
        assert!((output[1] - 0.5).abs() < 1e-5); // cos(pi/3) = 0.5
        assert!((output[2] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-5); // cos(pi/4) = sqrt(2)/2
        assert!(output[3].abs() < 1e-5); // cos(pi/2) ~ 0
        assert!((output[4] - (-1.0)).abs() < 1e-5); // cos(pi) = -1
    }

    #[test]
    fn test_cos_constant_fold_negative() {
        let translator = CosTranslator;
        let input: Vec<f32> = vec![-PI / 2.0, -PI / 4.0, -PI];
        let input_bytes: &[u8] = bytemuck::cast_slice(&input);

        let result = translator.constant_fold(&make_node(), &[input_bytes]);
        assert!(result.is_some());

        let output_bytes = result.unwrap();
        let output: &[f32] = bytemuck::cast_slice(&output_bytes);
        assert!(output[0].abs() < 1e-5); // cos(-pi/2) ~ 0
        assert!((output[1] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-5); // cos(-pi/4) = sqrt(2)/2
        assert!((output[2] - (-1.0)).abs() < 1e-5); // cos(-pi) = -1
    }

    #[test]
    fn test_cos_constant_fold_zero() {
        let translator = CosTranslator;
        let input: Vec<f32> = vec![0.0, 0.0, 0.0];
        let input_bytes: &[u8] = bytemuck::cast_slice(&input);

        let result = translator.constant_fold(&make_node(), &[input_bytes]);
        assert!(result.is_some());

        let output_bytes = result.unwrap();
        let output: &[f32] = bytemuck::cast_slice(&output_bytes);
        assert!(output.iter().all(|&x| (x - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_cos_constant_fold_empty() {
        let translator = CosTranslator;
        let result = translator.constant_fold(&make_node(), &[]);
        assert!(result.is_none());
    }

    // ===== Invalid Input Tests =====

    #[test]
    fn test_cos_no_inputs() {
        let translator = CosTranslator;
        let err = translator.input_requirement().validate(0, "Cos");
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
    fn test_cos_too_many_inputs() {
        let translator = CosTranslator;
        let err = translator.input_requirement().validate(2, "Cos");
        assert!(err.is_err());
    }
}
