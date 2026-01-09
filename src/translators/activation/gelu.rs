//! GELU activation translator.

use hologram::ir::{GraphBuilder, NodeIndex};
use crate::proto::NodeProto;
use crate::translators::{OnnxTranslator, InputRequirement, TranslationError};

/// Translator for ONNX Gelu operation.
///
/// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
#[derive(Debug, Default)]
pub struct GeluTranslator;

impl OnnxTranslator for GeluTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "Gelu"
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
            .gelu(inputs[0])
            .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;
        Ok(vec![result])
    }

    fn supports_constant_folding(&self) -> bool {
        true
    }

    fn constant_fold(
        &self,
        _node: &NodeProto,
        constant_inputs: &[&[u8]],
    ) -> Option<Vec<u8>> {
        let input = constant_inputs.first()?;
        let floats: &[f32] = bytemuck::cast_slice(input);
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        let sqrt_2_over_pi: f32 = (2.0_f32 / std::f32::consts::PI).sqrt();
        let result: Vec<f32> = floats
            .iter()
            .map(|&x| {
                let inner = sqrt_2_over_pi * (x + 0.044715 * x.powi(3));
                0.5 * x * (1.0 + inner.tanh())
            })
            .collect();
        Some(bytemuck::cast_slice(&result).to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram::ir::{DType, Shape};

    fn make_node() -> NodeProto {
        NodeProto {
            name: "gelu_test".to_string(),
            op_type: "Gelu".to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn test_gelu_translation() {
        let translator = GeluTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);

        let result = translator.translate(&make_node(), &[x], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_gelu_constant_fold_zero() {
        let translator = GeluTranslator;
        let input: Vec<f32> = vec![0.0];
        let input_bytes: &[u8] = bytemuck::cast_slice(&input);

        let result = translator.constant_fold(&make_node(), &[input_bytes]);
        assert!(result.is_some());

        let output_bytes = result.unwrap();
        let output: &[f32] = bytemuck::cast_slice(&output_bytes);
        assert!(output[0].abs() < 1e-6);
    }

    #[test]
    fn test_gelu_constant_fold_positive() {
        let translator = GeluTranslator;
        let input: Vec<f32> = vec![1.0];
        let input_bytes: &[u8] = bytemuck::cast_slice(&input);

        let result = translator.constant_fold(&make_node(), &[input_bytes]);
        assert!(result.is_some());

        let output_bytes = result.unwrap();
        let output: &[f32] = bytemuck::cast_slice(&output_bytes);
        // GELU(1) ≈ 0.841
        assert!((output[0] - 0.841).abs() < 0.01);
    }

    #[test]
    fn test_gelu_constant_fold_negative() {
        let translator = GeluTranslator;
        let input: Vec<f32> = vec![-1.0];
        let input_bytes: &[u8] = bytemuck::cast_slice(&input);

        let result = translator.constant_fold(&make_node(), &[input_bytes]);
        assert!(result.is_some());

        let output_bytes = result.unwrap();
        let output: &[f32] = bytemuck::cast_slice(&output_bytes);
        // GELU(-1) ≈ -0.159
        assert!((output[0] + 0.159).abs() < 0.01);
    }

    #[test]
    fn test_gelu_no_inputs() {
        let translator = GeluTranslator;
        let err = translator.input_requirement().validate(0, "Gelu");
        assert!(err.is_err());
    }
}
