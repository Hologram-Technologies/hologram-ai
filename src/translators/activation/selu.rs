//! SELU activation translator.

use hologram::ir::{GraphBuilder, NodeIndex, NodeOp, ConstantData, Shape};
use crate::proto::NodeProto;
use crate::translators::{OnnxTranslator, OnnxAttributes, InputRequirement, TranslationError};

/// Translator for ONNX Selu operation.
///
/// SELU(x) = gamma * (x if x >= 0 else alpha * (exp(x) - 1))
///
/// Default values:
/// - alpha ≈ 1.67326
/// - gamma ≈ 1.0507
#[derive(Debug, Default)]
pub struct SeluTranslator;

impl OnnxTranslator for SeluTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "Selu"
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
        let alpha = node.get_float_or("alpha", 1.67326);
        let gamma = node.get_float_or("gamma", 1.0507);

        // Create constants
        let zero = builder.constant(
            ConstantData::F32(vec![0.0]),
            Shape::static_shape(&[1]),
        );
        let alpha_const = builder.constant(
            ConstantData::F32(vec![alpha]),
            Shape::static_shape(&[1]),
        );
        let one = builder.constant(
            ConstantData::F32(vec![1.0]),
            Shape::static_shape(&[1]),
        );
        let gamma_const = builder.constant(
            ConstantData::F32(vec![gamma]),
            Shape::static_shape(&[1]),
        );

        // SELU = gamma * ELU(x, alpha)
        // ELU(x, alpha) = max(0, x) + alpha * (exp(min(0, x)) - 1)
        let positive_part = builder
            .binary(NodeOp::Max, inputs[0], zero)
            .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;

        let negative_input = builder
            .binary(NodeOp::Min, inputs[0], zero)
            .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;
        let exp_x = builder
            .unary(NodeOp::Exp, negative_input)
            .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;
        let exp_minus_one = builder
            .sub(exp_x, one)
            .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;
        let negative_part = builder
            .mul(alpha_const, exp_minus_one)
            .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;

        let elu = builder
            .add(positive_part, negative_part)
            .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;

        let result = builder
            .mul(elu, gamma_const)
            .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;

        Ok(vec![result])
    }

    fn supports_constant_folding(&self) -> bool {
        true
    }

    fn constant_fold(
        &self,
        node: &NodeProto,
        constant_inputs: &[&[u8]],
    ) -> Option<Vec<u8>> {
        let input = constant_inputs.first()?;
        let floats: &[f32] = bytemuck::cast_slice(input);
        let alpha = node.get_float_or("alpha", 1.67326);
        let gamma = node.get_float_or("gamma", 1.0507);

        let result: Vec<f32> = floats
            .iter()
            .map(|&x| {
                let elu = if x >= 0.0 {
                    x
                } else {
                    alpha * (x.exp() - 1.0)
                };
                gamma * elu
            })
            .collect();
        Some(bytemuck::cast_slice(&result).to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram::ir::DType;
    use crate::proto::AttributeProto;

    fn make_node() -> NodeProto {
        NodeProto {
            name: "selu_test".to_string(),
            op_type: "Selu".to_string(),
            ..Default::default()
        }
    }

    fn make_node_with_params(alpha: f32, gamma: f32) -> NodeProto {
        NodeProto {
            name: "selu_test".to_string(),
            op_type: "Selu".to_string(),
            attribute: vec![
                AttributeProto {
                    name: "alpha".to_string(),
                    f: alpha,
                    ..Default::default()
                },
                AttributeProto {
                    name: "gamma".to_string(),
                    f: gamma,
                    ..Default::default()
                },
            ],
            ..Default::default()
        }
    }

    #[test]
    fn test_selu_translation() {
        let translator = SeluTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);

        let result = translator.translate(&make_node(), &[x], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_selu_custom_params() {
        let translator = SeluTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[10]), DType::F32);

        let result = translator.translate(&make_node_with_params(2.0, 1.5), &[x], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_selu_constant_fold_positive() {
        let translator = SeluTranslator;
        let input: Vec<f32> = vec![0.0, 1.0];
        let input_bytes: &[u8] = bytemuck::cast_slice(&input);

        let result = translator.constant_fold(&make_node(), &[input_bytes]);
        assert!(result.is_some());

        let output_bytes = result.unwrap();
        let output: &[f32] = bytemuck::cast_slice(&output_bytes);
        // SELU(0) = gamma * 0 = 0
        // SELU(1) = gamma * 1 ≈ 1.0507
        assert!(output[0].abs() < 1e-6);
        assert!((output[1] - 1.0507).abs() < 0.01);
    }

    #[test]
    fn test_selu_constant_fold_negative() {
        let translator = SeluTranslator;
        let input: Vec<f32> = vec![-1.0];
        let input_bytes: &[u8] = bytemuck::cast_slice(&input);

        let result = translator.constant_fold(&make_node(), &[input_bytes]);
        assert!(result.is_some());

        let output_bytes = result.unwrap();
        let output: &[f32] = bytemuck::cast_slice(&output_bytes);
        // SELU(-1) = gamma * alpha * (exp(-1) - 1) ≈ -1.1113
        assert!((output[0] + 1.1113).abs() < 0.1);
    }

    #[test]
    fn test_selu_no_inputs() {
        let translator = SeluTranslator;
        let err = translator.input_requirement().validate(0, "Selu");
        assert!(err.is_err());
    }
}
