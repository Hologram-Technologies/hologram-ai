//! Sin operation translator.

use crate::proto::NodeProto;
use crate::translators::{InputRequirement, OnnxTranslator, TranslationError};
use hologram::ir::{GraphBuilder, NodeIndex, NodeOp};

/// Translator for ONNX Sin operation.
///
/// Sin(x) = sin(x) (element-wise sine)
#[derive(Debug, Default)]
pub struct SinTranslator;

impl OnnxTranslator for SinTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "Sin"
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
            .unary(NodeOp::Sin, inputs[0])
            .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;
        Ok(vec![result])
    }

    fn supports_constant_folding(&self) -> bool {
        true
    }

    fn constant_fold(&self, _node: &NodeProto, constant_inputs: &[&[u8]]) -> Option<Vec<u8>> {
        let input = constant_inputs.first()?;
        let floats: &[f32] = bytemuck::cast_slice(input);
        let result: Vec<f32> = floats.iter().map(|x| x.sin()).collect();
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
            name: "sin_test".to_string(),
            op_type: "Sin".to_string(),
            ..Default::default()
        }
    }

    // ===== Valid Input Tests =====

    #[test]
    fn test_sin_single_input() {
        let translator = SinTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);

        let result = translator.translate(&make_node(), &[x], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_sin_1d_tensor() {
        let translator = SinTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[10]), DType::F32);

        let result = translator.translate(&make_node(), &[x], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sin_4d_tensor() {
        let translator = SinTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[1, 64, 32, 32]), DType::F32);

        let result = translator.translate(&make_node(), &[x], &mut builder);
        assert!(result.is_ok());
    }

    // ===== Constant Folding Tests =====

    #[test]
    fn test_sin_constant_fold_basic() {
        let translator = SinTranslator;
        let input: Vec<f32> = vec![0.0, PI / 6.0, PI / 4.0, PI / 2.0, PI];
        let input_bytes: &[u8] = bytemuck::cast_slice(&input);

        let result = translator.constant_fold(&make_node(), &[input_bytes]);
        assert!(result.is_some());

        let output_bytes = result.unwrap();
        let output: &[f32] = bytemuck::cast_slice(&output_bytes);
        assert!((output[0] - 0.0).abs() < 1e-6); // sin(0) = 0
        assert!((output[1] - 0.5).abs() < 1e-5); // sin(pi/6) = 0.5
        assert!((output[2] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-5); // sin(pi/4) = sqrt(2)/2
        assert!((output[3] - 1.0).abs() < 1e-5); // sin(pi/2) = 1
        assert!(output[4].abs() < 1e-5); // sin(pi) ~ 0
    }

    #[test]
    fn test_sin_constant_fold_negative() {
        let translator = SinTranslator;
        let input: Vec<f32> = vec![-PI / 2.0, -PI / 4.0];
        let input_bytes: &[u8] = bytemuck::cast_slice(&input);

        let result = translator.constant_fold(&make_node(), &[input_bytes]);
        assert!(result.is_some());

        let output_bytes = result.unwrap();
        let output: &[f32] = bytemuck::cast_slice(&output_bytes);
        assert!((output[0] - (-1.0)).abs() < 1e-5); // sin(-pi/2) = -1
        assert!((output[1] - (-std::f32::consts::FRAC_1_SQRT_2)).abs() < 1e-5); // sin(-pi/4) = -sqrt(2)/2
    }

    #[test]
    fn test_sin_constant_fold_zero() {
        let translator = SinTranslator;
        let input: Vec<f32> = vec![0.0, 0.0, 0.0];
        let input_bytes: &[u8] = bytemuck::cast_slice(&input);

        let result = translator.constant_fold(&make_node(), &[input_bytes]);
        assert!(result.is_some());

        let output_bytes = result.unwrap();
        let output: &[f32] = bytemuck::cast_slice(&output_bytes);
        assert!(output.iter().all(|&x| x.abs() < 1e-6));
    }

    #[test]
    fn test_sin_constant_fold_empty() {
        let translator = SinTranslator;
        let result = translator.constant_fold(&make_node(), &[]);
        assert!(result.is_none());
    }

    // ===== Invalid Input Tests =====

    #[test]
    fn test_sin_no_inputs() {
        let translator = SinTranslator;
        let err = translator.input_requirement().validate(0, "Sin");
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
    fn test_sin_too_many_inputs() {
        let translator = SinTranslator;
        let err = translator.input_requirement().validate(2, "Sin");
        assert!(err.is_err());
    }
}
