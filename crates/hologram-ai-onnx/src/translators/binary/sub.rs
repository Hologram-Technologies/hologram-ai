//! Sub operation translator.

use crate::proto::NodeProto;
use crate::translators::{InputRequirement, OnnxTranslator, TranslationError};
use hologram::ir::{GraphBuilder, NodeIndex};

/// Translator for ONNX Sub operation.
///
/// Sub(A, B) = A - B (with broadcasting)
#[derive(Debug, Default)]
pub struct SubTranslator;

impl OnnxTranslator for SubTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "Sub"
    }

    fn input_requirement(&self) -> InputRequirement {
        InputRequirement::Exact(2)
    }

    fn translate(
        &self,
        _node: &NodeProto,
        inputs: &[NodeIndex],
        builder: &mut GraphBuilder,
    ) -> Result<Vec<NodeIndex>, TranslationError> {
        let result = builder
            .sub(inputs[0], inputs[1])
            .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;
        Ok(vec![result])
    }

    fn supports_constant_folding(&self) -> bool {
        true
    }

    fn constant_fold(&self, _node: &NodeProto, constant_inputs: &[&[u8]]) -> Option<Vec<u8>> {
        if constant_inputs.len() != 2 {
            return None;
        }
        let a: &[f32] = bytemuck::cast_slice(constant_inputs[0]);
        let b: &[f32] = bytemuck::cast_slice(constant_inputs[1]);
        if a.len() != b.len() {
            return None;
        }
        let result: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();
        Some(bytemuck::cast_slice(&result).to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram::ir::{DType, Shape};

    fn make_node() -> NodeProto {
        NodeProto {
            name: "sub_test".to_string(),
            op_type: "Sub".to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn test_sub_translation() {
        let translator = SubTranslator;
        let mut builder = GraphBuilder::new();
        let a = builder.input("a", Shape::static_shape(&[2, 3]), DType::F32);
        let b = builder.input("b", Shape::static_shape(&[2, 3]), DType::F32);

        let result = translator.translate(&make_node(), &[a, b], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_sub_constant_fold() {
        let translator = SubTranslator;
        let a: Vec<f32> = vec![5.0, 7.0, 9.0];
        let b: Vec<f32> = vec![1.0, 2.0, 3.0];
        let a_bytes: &[u8] = bytemuck::cast_slice(&a);
        let b_bytes: &[u8] = bytemuck::cast_slice(&b);

        let result = translator.constant_fold(&make_node(), &[a_bytes, b_bytes]);
        assert!(result.is_some());

        let output_bytes = result.unwrap();
        let output: &[f32] = bytemuck::cast_slice(&output_bytes);
        assert_eq!(output, &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_sub_invalid_inputs() {
        let translator = SubTranslator;
        assert!(translator.input_requirement().validate(0, "Sub").is_err());
        assert!(translator.input_requirement().validate(1, "Sub").is_err());
        assert!(translator.input_requirement().validate(3, "Sub").is_err());
    }
}
