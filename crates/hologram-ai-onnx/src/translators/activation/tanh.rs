//! Tanh activation translator.

use crate::core::op_hints::{ActivationType, add_simd_hint};
use crate::proto::NodeProto;
use crate::translators::{InputRequirement, OnnxTranslator, TranslationError};
use hologram::ir::{GraphBuilder, NodeIndex};

/// Translator for ONNX Tanh operation.
///
/// Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
#[derive(Debug, Default)]
pub struct TanhTranslator;

impl OnnxTranslator for TanhTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "Tanh"
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
            .tanh(inputs[0])
            .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;

        // Add SIMD lookup hint for backend optimization
        add_simd_hint(builder.graph_mut(), result, ActivationType::Tanh);

        Ok(vec![result])
    }

    fn supports_constant_folding(&self) -> bool {
        true
    }

    fn constant_fold(&self, _node: &NodeProto, constant_inputs: &[&[u8]]) -> Option<Vec<u8>> {
        let input = constant_inputs.first()?;
        let floats: &[f32] = bytemuck::cast_slice(input);
        let result: Vec<f32> = floats.iter().map(|x| x.tanh()).collect();
        Some(bytemuck::cast_slice(&result).to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram::ir::{DType, Shape};

    fn make_node() -> NodeProto {
        NodeProto {
            name: "tanh_test".to_string(),
            op_type: "Tanh".to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn test_tanh_translation() {
        let translator = TanhTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);

        let result = translator.translate(&make_node(), &[x], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_tanh_constant_fold_zero() {
        let translator = TanhTranslator;
        let input: Vec<f32> = vec![0.0];
        let input_bytes: &[u8] = bytemuck::cast_slice(&input);

        let result = translator.constant_fold(&make_node(), &[input_bytes]);
        assert!(result.is_some());

        let output_bytes = result.unwrap();
        let output: &[f32] = bytemuck::cast_slice(&output_bytes);
        assert!(output[0].abs() < 1e-6);
    }

    #[test]
    fn test_tanh_constant_fold_range() {
        let translator = TanhTranslator;
        let input: Vec<f32> = vec![-10.0, 0.0, 10.0];
        let input_bytes: &[u8] = bytemuck::cast_slice(&input);

        let result = translator.constant_fold(&make_node(), &[input_bytes]);
        assert!(result.is_some());

        let output_bytes = result.unwrap();
        let output: &[f32] = bytemuck::cast_slice(&output_bytes);
        assert!(output[0] < -0.99 && output[0] >= -1.0);
        assert!(output[1].abs() < 1e-6);
        assert!(output[2] > 0.99 && output[2] <= 1.0);
    }

    #[test]
    fn test_tanh_no_inputs() {
        let translator = TanhTranslator;
        let err = translator.input_requirement().validate(0, "Tanh");
        assert!(err.is_err());
    }
}
