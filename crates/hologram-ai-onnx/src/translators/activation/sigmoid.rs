//! Sigmoid activation translator.

use crate::core::op_hints::{ActivationType, add_simd_hint};
use crate::proto::NodeProto;
use crate::translators::{InputRequirement, OnnxTranslator, TranslationError};
use hologram::ir::{GraphBuilder, NodeIndex};

/// Translator for ONNX Sigmoid operation.
///
/// Sigmoid(x) = 1 / (1 + exp(-x))
#[derive(Debug, Default)]
pub struct SigmoidTranslator;

impl OnnxTranslator for SigmoidTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "Sigmoid"
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
            .sigmoid(inputs[0])
            .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;

        // Add SIMD lookup hint for backend optimization
        add_simd_hint(builder.graph_mut(), result, ActivationType::Sigmoid);

        Ok(vec![result])
    }

    fn supports_constant_folding(&self) -> bool {
        true
    }

    fn constant_fold(&self, _node: &NodeProto, constant_inputs: &[&[u8]]) -> Option<Vec<u8>> {
        let input = constant_inputs.first()?;
        let floats: &[f32] = bytemuck::cast_slice(input);
        let result: Vec<f32> = floats.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
        Some(bytemuck::cast_slice(&result).to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram::ir::{DType, Shape};

    fn make_node() -> NodeProto {
        NodeProto {
            name: "sigmoid_test".to_string(),
            op_type: "Sigmoid".to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn test_sigmoid_translation() {
        let translator = SigmoidTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);

        let result = translator.translate(&make_node(), &[x], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_sigmoid_constant_fold() {
        let translator = SigmoidTranslator;
        let input: Vec<f32> = vec![0.0];
        let input_bytes: &[u8] = bytemuck::cast_slice(&input);

        let result = translator.constant_fold(&make_node(), &[input_bytes]);
        assert!(result.is_some());

        let output_bytes = result.unwrap();
        let output: &[f32] = bytemuck::cast_slice(&output_bytes);
        assert!((output[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid_constant_fold_large_positive() {
        let translator = SigmoidTranslator;
        let input: Vec<f32> = vec![10.0];
        let input_bytes: &[u8] = bytemuck::cast_slice(&input);

        let result = translator.constant_fold(&make_node(), &[input_bytes]);
        assert!(result.is_some());

        let output_bytes = result.unwrap();
        let output: &[f32] = bytemuck::cast_slice(&output_bytes);
        assert!(output[0] > 0.99);
    }

    #[test]
    fn test_sigmoid_constant_fold_large_negative() {
        let translator = SigmoidTranslator;
        let input: Vec<f32> = vec![-10.0];
        let input_bytes: &[u8] = bytemuck::cast_slice(&input);

        let result = translator.constant_fold(&make_node(), &[input_bytes]);
        assert!(result.is_some());

        let output_bytes = result.unwrap();
        let output: &[f32] = bytemuck::cast_slice(&output_bytes);
        assert!(output[0] < 0.01);
    }

    #[test]
    fn test_sigmoid_no_inputs() {
        let translator = SigmoidTranslator;
        let err = translator.input_requirement().validate(0, "Sigmoid");
        assert!(err.is_err());
    }

    #[test]
    fn test_sigmoid_simd_hint_added() {
        use crate::core::op_hints::{get_simd_table_id, has_simd_hint};

        let translator = SigmoidTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);

        let result = translator.translate(&make_node(), &[x], &mut builder);
        assert!(result.is_ok());

        let output_nodes = result.unwrap();
        assert_eq!(output_nodes.len(), 1);
        let sigmoid_node = output_nodes[0];

        // Verify SIMD hint was added
        assert!(
            has_simd_hint(builder.graph(), sigmoid_node),
            "SIMD hint should be present on sigmoid node"
        );

        // Verify correct table_id (0 for sigmoid)
        let table_id = get_simd_table_id(builder.graph(), sigmoid_node);
        assert_eq!(table_id, Some(0), "Sigmoid should have table_id=0");
    }
}
