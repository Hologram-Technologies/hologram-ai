//! Log operation translator.

use crate::proto::NodeProto;
use crate::translators::{InputRequirement, OnnxTranslator, TranslationError};
use hologram::ir::{GraphBuilder, NodeIndex, NodeOp};

/// Translator for ONNX Log operation.
///
/// Log(x) = ln(x) (element-wise natural logarithm)
#[derive(Debug, Default)]
pub struct LogTranslator;

impl OnnxTranslator for LogTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "Log"
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
            .unary(NodeOp::Log, inputs[0])
            .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;
        Ok(vec![result])
    }

    fn supports_constant_folding(&self) -> bool {
        true
    }

    fn constant_fold(&self, _node: &NodeProto, constant_inputs: &[&[u8]]) -> Option<Vec<u8>> {
        let input = constant_inputs.first()?;
        let floats: &[f32] = bytemuck::cast_slice(input);
        let result: Vec<f32> = floats.iter().map(|x| x.ln()).collect();
        Some(bytemuck::cast_slice(&result).to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram::ir::{DType, Shape};

    fn make_node() -> NodeProto {
        NodeProto {
            name: "log_test".to_string(),
            op_type: "Log".to_string(),
            ..Default::default()
        }
    }

    // ===== Valid Input Tests =====

    #[test]
    fn test_log_single_input() {
        let translator = LogTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);

        let result = translator.translate(&make_node(), &[x], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_log_1d_tensor() {
        let translator = LogTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[10]), DType::F32);

        let result = translator.translate(&make_node(), &[x], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_log_4d_tensor() {
        let translator = LogTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[1, 64, 32, 32]), DType::F32);

        let result = translator.translate(&make_node(), &[x], &mut builder);
        assert!(result.is_ok());
    }

    // ===== Constant Folding Tests =====

    #[test]
    fn test_log_constant_fold_basic() {
        let translator = LogTranslator;
        let e = std::f32::consts::E;
        let input: Vec<f32> = vec![1.0, e, e * e];
        let input_bytes: &[u8] = bytemuck::cast_slice(&input);

        let result = translator.constant_fold(&make_node(), &[input_bytes]);
        assert!(result.is_some());

        let output_bytes = result.unwrap();
        let output: &[f32] = bytemuck::cast_slice(&output_bytes);
        assert!((output[0] - 0.0).abs() < 1e-6); // ln(1) = 0
        assert!((output[1] - 1.0).abs() < 1e-5); // ln(e) = 1
        assert!((output[2] - 2.0).abs() < 1e-4); // ln(e^2) = 2
    }

    #[test]
    fn test_log_constant_fold_positive() {
        let translator = LogTranslator;
        let input: Vec<f32> = vec![2.0, 10.0, 100.0];
        let input_bytes: &[u8] = bytemuck::cast_slice(&input);

        let result = translator.constant_fold(&make_node(), &[input_bytes]);
        assert!(result.is_some());

        let output_bytes = result.unwrap();
        let output: &[f32] = bytemuck::cast_slice(&output_bytes);
        assert!((output[0] - 2.0_f32.ln()).abs() < 1e-6);
        assert!((output[1] - 10.0_f32.ln()).abs() < 1e-5);
        assert!((output[2] - 100.0_f32.ln()).abs() < 1e-5);
    }

    #[test]
    fn test_log_constant_fold_one() {
        let translator = LogTranslator;
        let input: Vec<f32> = vec![1.0, 1.0, 1.0];
        let input_bytes: &[u8] = bytemuck::cast_slice(&input);

        let result = translator.constant_fold(&make_node(), &[input_bytes]);
        assert!(result.is_some());

        let output_bytes = result.unwrap();
        let output: &[f32] = bytemuck::cast_slice(&output_bytes);
        assert!(output.iter().all(|&x| x.abs() < 1e-6));
    }

    #[test]
    fn test_log_constant_fold_empty() {
        let translator = LogTranslator;
        let result = translator.constant_fold(&make_node(), &[]);
        assert!(result.is_none());
    }

    // ===== Invalid Input Tests =====

    #[test]
    fn test_log_no_inputs() {
        let translator = LogTranslator;
        let err = translator.input_requirement().validate(0, "Log");
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
    fn test_log_too_many_inputs() {
        let translator = LogTranslator;
        let err = translator.input_requirement().validate(2, "Log");
        assert!(err.is_err());
    }
}
