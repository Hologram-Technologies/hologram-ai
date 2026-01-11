//! LeakyReLU activation translator.

use crate::proto::NodeProto;
use crate::translators::{InputRequirement, OnnxAttributes, OnnxTranslator, TranslationError};
use hologram::ir::{ConstantData, GraphBuilder, NodeIndex, NodeOp, Shape};

/// Translator for ONNX LeakyRelu operation.
///
/// LeakyRelu(x, alpha) = x if x >= 0 else alpha * x
#[derive(Debug, Default)]
pub struct LeakyReluTranslator;

impl OnnxTranslator for LeakyReluTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "LeakyRelu"
    }

    fn input_requirement(&self) -> InputRequirement {
        InputRequirement::Exact(1)
    }

    fn translate(
        &self,
        node: &NodeProto,
        inputs: &[NodeIndex],
        builder: &mut GraphBuilder,
    ) -> Result<Vec<NodeIndex>, TranslationError> {
        let alpha = node.get_float_or("alpha", 0.01);

        // LeakyReLU(x) = max(x, alpha * x)
        let alpha_const =
            builder.constant(ConstantData::F32(vec![alpha]), Shape::static_shape(&[1]));

        let scaled = builder
            .mul(inputs[0], alpha_const)
            .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;
        let result = builder
            .binary(NodeOp::Max, inputs[0], scaled)
            .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;

        Ok(vec![result])
    }

    fn supports_constant_folding(&self) -> bool {
        true
    }

    fn constant_fold(&self, node: &NodeProto, constant_inputs: &[&[u8]]) -> Option<Vec<u8>> {
        let input = constant_inputs.first()?;
        let floats: &[f32] = bytemuck::cast_slice(input);
        let alpha = node.get_float_or("alpha", 0.01);
        let result: Vec<f32> = floats
            .iter()
            .map(|&x| if x >= 0.0 { x } else { alpha * x })
            .collect();
        Some(bytemuck::cast_slice(&result).to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::AttributeProto;
    use hologram::ir::DType;

    fn make_node() -> NodeProto {
        NodeProto {
            name: "leaky_relu_test".to_string(),
            op_type: "LeakyRelu".to_string(),
            ..Default::default()
        }
    }

    fn make_node_with_alpha(alpha: f32) -> NodeProto {
        NodeProto {
            name: "leaky_relu_test".to_string(),
            op_type: "LeakyRelu".to_string(),
            attribute: vec![AttributeProto {
                name: "alpha".to_string(),
                f: alpha,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn test_leaky_relu_translation() {
        let translator = LeakyReluTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);

        let result = translator.translate(&make_node(), &[x], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_leaky_relu_custom_alpha() {
        let translator = LeakyReluTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[10]), DType::F32);

        let result = translator.translate(&make_node_with_alpha(0.2), &[x], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_leaky_relu_constant_fold_default_alpha() {
        let translator = LeakyReluTranslator;
        let input: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let input_bytes: &[u8] = bytemuck::cast_slice(&input);

        let result = translator.constant_fold(&make_node(), &[input_bytes]);
        assert!(result.is_some());

        let output_bytes = result.unwrap();
        let output: &[f32] = bytemuck::cast_slice(&output_bytes);
        // Default alpha = 0.01
        assert!((output[0] - (-0.02)).abs() < 1e-6);
        assert!((output[1] - (-0.01)).abs() < 1e-6);
        assert!((output[2] - 0.0).abs() < 1e-6);
        assert!((output[3] - 1.0).abs() < 1e-6);
        assert!((output[4] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_leaky_relu_constant_fold_custom_alpha() {
        let translator = LeakyReluTranslator;
        let input: Vec<f32> = vec![-10.0, 10.0];
        let input_bytes: &[u8] = bytemuck::cast_slice(&input);

        let result = translator.constant_fold(&make_node_with_alpha(0.1), &[input_bytes]);
        assert!(result.is_some());

        let output_bytes = result.unwrap();
        let output: &[f32] = bytemuck::cast_slice(&output_bytes);
        assert!((output[0] - (-1.0)).abs() < 1e-6); // -10 * 0.1 = -1
        assert!((output[1] - 10.0).abs() < 1e-6); // positive unchanged
    }

    #[test]
    fn test_leaky_relu_no_inputs() {
        let translator = LeakyReluTranslator;
        let err = translator.input_requirement().validate(0, "LeakyRelu");
        assert!(err.is_err());
    }
}
