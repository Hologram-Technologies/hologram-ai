//! Exp operation translator.

use crate::proto::NodeProto;
use crate::translators::{InputRequirement, OnnxTranslator, TranslationError};
use hologram::ir::{GraphBuilder, NodeIndex, NodeOp};

/// Translator for ONNX Exp operation.
///
/// Exp(x) = e^x (element-wise exponential)
#[derive(Debug, Default)]
pub struct ExpTranslator;

impl OnnxTranslator for ExpTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "Exp"
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
            .unary(NodeOp::Exp, inputs[0])
            .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;
        Ok(vec![result])
    }

    fn supports_constant_folding(&self) -> bool {
        true
    }

    fn constant_fold(&self, _node: &NodeProto, constant_inputs: &[&[u8]]) -> Option<Vec<u8>> {
        let input = constant_inputs.first()?;
        let floats: &[f32] = bytemuck::cast_slice(input);
        let result: Vec<f32> = floats.iter().map(|x| x.exp()).collect();
        Some(bytemuck::cast_slice(&result).to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram::ir::{DType, Shape};

    fn make_node() -> NodeProto {
        NodeProto {
            name: "exp_test".to_string(),
            op_type: "Exp".to_string(),
            ..Default::default()
        }
    }

    // ===== Valid Input Tests =====

    #[test]
    fn test_exp_single_input() {
        let translator = ExpTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);

        let result = translator.translate(&make_node(), &[x], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_exp_1d_tensor() {
        let translator = ExpTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[10]), DType::F32);

        let result = translator.translate(&make_node(), &[x], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_exp_4d_tensor() {
        let translator = ExpTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[1, 64, 32, 32]), DType::F32);

        let result = translator.translate(&make_node(), &[x], &mut builder);
        assert!(result.is_ok());
    }

    // ===== Constant Folding Tests =====

    #[test]
    fn test_exp_constant_fold_basic() {
        let translator = ExpTranslator;
        let input: Vec<f32> = vec![0.0, 1.0, 2.0];
        let input_bytes: &[u8] = bytemuck::cast_slice(&input);

        let result = translator.constant_fold(&make_node(), &[input_bytes]);
        assert!(result.is_some());

        let output_bytes = result.unwrap();
        let output: &[f32] = bytemuck::cast_slice(&output_bytes);
        assert!((output[0] - 1.0).abs() < 1e-6); // e^0 = 1
        assert!((output[1] - std::f32::consts::E).abs() < 1e-5); // e^1 = e
        assert!((output[2] - std::f32::consts::E.powi(2)).abs() < 1e-4); // e^2
    }

    #[test]
    fn test_exp_constant_fold_negative() {
        let translator = ExpTranslator;
        let input: Vec<f32> = vec![-1.0, -2.0];
        let input_bytes: &[u8] = bytemuck::cast_slice(&input);

        let result = translator.constant_fold(&make_node(), &[input_bytes]);
        assert!(result.is_some());

        let output_bytes = result.unwrap();
        let output: &[f32] = bytemuck::cast_slice(&output_bytes);
        assert!((output[0] - 1.0 / std::f32::consts::E).abs() < 1e-6);
        assert!((output[1] - 1.0 / std::f32::consts::E.powi(2)).abs() < 1e-5);
    }

    #[test]
    fn test_exp_constant_fold_zero() {
        let translator = ExpTranslator;
        let input: Vec<f32> = vec![0.0, 0.0, 0.0];
        let input_bytes: &[u8] = bytemuck::cast_slice(&input);

        let result = translator.constant_fold(&make_node(), &[input_bytes]);
        assert!(result.is_some());

        let output_bytes = result.unwrap();
        let output: &[f32] = bytemuck::cast_slice(&output_bytes);
        assert!(output.iter().all(|&x| (x - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_exp_constant_fold_empty() {
        let translator = ExpTranslator;
        let result = translator.constant_fold(&make_node(), &[]);
        assert!(result.is_none());
    }

    // ===== Invalid Input Tests =====

    #[test]
    fn test_exp_no_inputs() {
        let translator = ExpTranslator;
        let err = translator.input_requirement().validate(0, "Exp");
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
    fn test_exp_too_many_inputs() {
        let translator = ExpTranslator;
        let err = translator.input_requirement().validate(2, "Exp");
        assert!(err.is_err());
    }
}
