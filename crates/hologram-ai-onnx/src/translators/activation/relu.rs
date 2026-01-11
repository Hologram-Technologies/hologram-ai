//! ReLU activation translator.

use crate::proto::NodeProto;
use crate::translators::{InputRequirement, OnnxTranslator, TranslationError};
use hologram::ir::{GraphBuilder, NodeIndex};

/// Translator for ONNX Relu operation.
///
/// ReLU(x) = max(0, x)
#[derive(Debug, Default)]
pub struct ReluTranslator;

impl OnnxTranslator for ReluTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "Relu"
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
            .relu(inputs[0])
            .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;
        Ok(vec![result])
    }

    fn supports_constant_folding(&self) -> bool {
        true
    }

    fn constant_fold(&self, _node: &NodeProto, constant_inputs: &[&[u8]]) -> Option<Vec<u8>> {
        let input = constant_inputs.first()?;
        let floats: &[f32] = bytemuck::cast_slice(input);
        let result: Vec<f32> = floats.iter().map(|x| x.max(0.0)).collect();
        Some(bytemuck::cast_slice(&result).to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram::ir::{DType, Shape};

    fn make_node() -> NodeProto {
        NodeProto {
            name: "relu_test".to_string(),
            op_type: "Relu".to_string(),
            ..Default::default()
        }
    }

    // ===== Valid Input Tests =====

    #[test]
    fn test_relu_single_input() {
        let translator = ReluTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);

        let result = translator.translate(&make_node(), &[x], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_relu_1d_tensor() {
        let translator = ReluTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[10]), DType::F32);

        let result = translator.translate(&make_node(), &[x], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_relu_4d_tensor() {
        let translator = ReluTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[1, 64, 32, 32]), DType::F32);

        let result = translator.translate(&make_node(), &[x], &mut builder);
        assert!(result.is_ok());
    }

    // ===== Constant Folding Tests =====

    #[test]
    fn test_relu_constant_fold_basic() {
        let translator = ReluTranslator;
        let input: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let input_bytes: &[u8] = bytemuck::cast_slice(&input);

        let result = translator.constant_fold(&make_node(), &[input_bytes]);
        assert!(result.is_some());

        let output_bytes = result.unwrap();
        let output: &[f32] = bytemuck::cast_slice(&output_bytes);
        assert_eq!(output, &[0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_relu_constant_fold_all_negative() {
        let translator = ReluTranslator;
        let input: Vec<f32> = vec![-5.0, -3.0, -1.0, -0.1];
        let input_bytes: &[u8] = bytemuck::cast_slice(&input);

        let result = translator.constant_fold(&make_node(), &[input_bytes]);
        assert!(result.is_some());

        let output_bytes = result.unwrap();
        let output: &[f32] = bytemuck::cast_slice(&output_bytes);
        assert!(output.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_relu_constant_fold_all_positive() {
        let translator = ReluTranslator;
        let input: Vec<f32> = vec![0.1, 1.0, 5.0, 10.0];
        let input_bytes: &[u8] = bytemuck::cast_slice(&input);

        let result = translator.constant_fold(&make_node(), &[input_bytes]);
        assert!(result.is_some());

        let output_bytes = result.unwrap();
        let output: &[f32] = bytemuck::cast_slice(&output_bytes);
        assert_eq!(output, &[0.1, 1.0, 5.0, 10.0]);
    }

    #[test]
    fn test_relu_constant_fold_empty() {
        let translator = ReluTranslator;
        let result = translator.constant_fold(&make_node(), &[]);
        assert!(result.is_none());
    }

    // ===== Invalid Input Tests =====

    #[test]
    fn test_relu_no_inputs() {
        let translator = ReluTranslator;
        let err = translator.input_requirement().validate(0, "Relu");
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
    fn test_relu_too_many_inputs() {
        let translator = ReluTranslator;
        let err = translator.input_requirement().validate(2, "Relu");
        assert!(err.is_err());
    }
}
