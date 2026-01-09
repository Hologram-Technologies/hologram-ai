//! Erf (error function) translator.

use hologram::ir::{GraphBuilder, NodeIndex};
use crate::proto::NodeProto;
use crate::translators::{OnnxTranslator, InputRequirement, TranslationError};

/// Translator for ONNX Erf (error function) operation.
///
/// Erf(x) = 2/sqrt(pi) * integral(exp(-t^2), t=0..x)
///
/// This is the standard Gaussian error function used in GELU and other activations.
#[derive(Debug, Default)]
pub struct ErfTranslator;

impl OnnxTranslator for ErfTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "Erf"
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
            .erf(inputs[0])
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
        let result: Vec<f32> = floats.iter().map(|&x| erf_approx(x)).collect();
        Some(bytemuck::cast_slice(&result).to_vec())
    }
}

/// Approximation of the error function using Abramowitz and Stegun formula.
fn erf_approx(x: f32) -> f32 {
    // Constants for the approximation (truncated to f32 precision)
    const A1: f32 = 0.254_829_6;
    const A2: f32 = -0.284_496_72;
    const A3: f32 = 1.421_413_8;
    const A4: f32 = -1.453_152_1;
    const A5: f32 = 1.061_405_4;
    const P: f32 = 0.327_591_1;

    // Handle sign
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    // A&S formula 7.1.26
    let t = 1.0 / (1.0 + P * x);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;

    let y = 1.0 - (A1 * t + A2 * t2 + A3 * t3 + A4 * t4 + A5 * t5) * (-x * x).exp();

    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram::ir::{DType, Shape};

    fn make_node() -> NodeProto {
        NodeProto {
            name: "erf_test".to_string(),
            op_type: "Erf".to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn test_erf_translation() {
        let translator = ErfTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);

        let result = translator.translate(&make_node(), &[x], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_erf_constant_fold_zero() {
        let translator = ErfTranslator;
        let input: Vec<f32> = vec![0.0];
        let input_bytes: &[u8] = bytemuck::cast_slice(&input);

        let result = translator.constant_fold(&make_node(), &[input_bytes]);
        assert!(result.is_some());

        let output_bytes = result.unwrap();
        let output: &[f32] = bytemuck::cast_slice(&output_bytes);
        // erf(0) = 0
        assert!(output[0].abs() < 1e-6);
    }

    #[test]
    fn test_erf_constant_fold_positive() {
        let translator = ErfTranslator;
        let input: Vec<f32> = vec![1.0];
        let input_bytes: &[u8] = bytemuck::cast_slice(&input);

        let result = translator.constant_fold(&make_node(), &[input_bytes]);
        assert!(result.is_some());

        let output_bytes = result.unwrap();
        let output: &[f32] = bytemuck::cast_slice(&output_bytes);
        // erf(1) ≈ 0.8427
        assert!((output[0] - 0.8427).abs() < 0.01);
    }

    #[test]
    fn test_erf_constant_fold_negative() {
        let translator = ErfTranslator;
        let input: Vec<f32> = vec![-1.0];
        let input_bytes: &[u8] = bytemuck::cast_slice(&input);

        let result = translator.constant_fold(&make_node(), &[input_bytes]);
        assert!(result.is_some());

        let output_bytes = result.unwrap();
        let output: &[f32] = bytemuck::cast_slice(&output_bytes);
        // erf(-1) ≈ -0.8427
        assert!((output[0] + 0.8427).abs() < 0.01);
    }

    #[test]
    fn test_erf_constant_fold_large() {
        let translator = ErfTranslator;
        let input: Vec<f32> = vec![3.0, -3.0];
        let input_bytes: &[u8] = bytemuck::cast_slice(&input);

        let result = translator.constant_fold(&make_node(), &[input_bytes]);
        assert!(result.is_some());

        let output_bytes = result.unwrap();
        let output: &[f32] = bytemuck::cast_slice(&output_bytes);
        // erf(3) ≈ 0.9999779
        assert!((output[0] - 1.0).abs() < 0.001);
        // erf(-3) ≈ -0.9999779
        assert!((output[1] + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_erf_no_inputs() {
        let translator = ErfTranslator;
        let err = translator.input_requirement().validate(0, "Erf");
        assert!(err.is_err());
    }
}
