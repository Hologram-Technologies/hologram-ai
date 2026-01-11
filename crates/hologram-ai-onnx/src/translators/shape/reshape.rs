//! Reshape operation translator.

use crate::proto::NodeProto;
use crate::translators::{InputRequirement, OnnxAttributes, OnnxTranslator, TranslationError};
use hologram::ir::{ConstantData, GraphBuilder, NodeIndex, NodeOp};

/// Translator for ONNX Reshape operation.
///
/// Reshape changes the dimensions of a tensor without changing its data.
///
/// # Inputs
/// - data: Input tensor to reshape
/// - shape: 1D tensor specifying the target shape
///
/// # Attributes
/// - allowzero (opset 14+): If 1, allows 0 in shape to mean "copy from input"
///
/// # Shape Semantics
/// - Positive values: Use as dimension size
/// - -1: Infer dimension from input size
/// - 0 (with allowzero=0): Copy dimension from input (default)
/// - 0 (with allowzero=1): Use 0 as actual dimension size
#[derive(Debug, Default)]
pub struct ReshapeTranslator;

impl OnnxTranslator for ReshapeTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "Reshape"
    }

    fn input_requirement(&self) -> InputRequirement {
        InputRequirement::Exact(2)
    }

    fn translate(
        &self,
        node: &NodeProto,
        inputs: &[NodeIndex],
        builder: &mut GraphBuilder,
    ) -> Result<Vec<NodeIndex>, TranslationError> {
        let data = inputs[0];
        let shape_input = inputs[1];

        // Check for allowzero attribute (ONNX opset 14+)
        let allow_zero = node.get_int_or("allowzero", 0) != 0;

        // Get shape node to check if it's constant
        let shape_node = builder.graph().node(shape_input).ok_or_else(|| {
            TranslationError::IrBuilder("Reshape: shape input not found".to_string())
        })?;

        // Check if shape is constant for optimization
        let new_shape = match &shape_node.op.op {
            NodeOp::Constant { data } => match data {
                ConstantData::I64(values) => Some(values.clone()),
                ConstantData::I32(values) => Some(values.iter().map(|&v| v as i64).collect()),
                _ => None,
            },
            _ => None,
        };

        if let Some(shape_values) = new_shape {
            // Check for special values that require dynamic handling
            let has_infer = shape_values.contains(&-1);
            let has_zero = allow_zero && shape_values.contains(&0);

            if !has_infer && !has_zero {
                // Simple static reshape (no inference needed)
                tracing::debug!("Reshape: static path, new_shape = {:?}", shape_values);
                let result = builder
                    .reshape(data, shape_values)
                    .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;
                return Ok(vec![result]);
            }
        }

        // Dynamic reshape path (supports runtime shapes, -1 inference, and allowzero)
        tracing::debug!("Reshape: dynamic path, allow_zero = {}", allow_zero);
        let result = builder
            .reshape_dynamic(data, shape_input, allow_zero)
            .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;
        Ok(vec![result])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::AttributeProto;
    use hologram::ir::{DType, Shape};

    fn make_node() -> NodeProto {
        NodeProto {
            name: "reshape_test".to_string(),
            op_type: "Reshape".to_string(),
            ..Default::default()
        }
    }

    fn make_node_with_allowzero(allowzero: i64) -> NodeProto {
        NodeProto {
            name: "reshape_test".to_string(),
            op_type: "Reshape".to_string(),
            attribute: vec![AttributeProto {
                name: "allowzero".to_string(),
                i: allowzero,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    // ===== Valid Input Tests =====

    #[test]
    fn test_reshape_static_shape() {
        let translator = ReshapeTranslator;
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[2, 3, 4]), DType::F32);
        let shape = builder.constant(ConstantData::I64(vec![6, 4]), Shape::static_shape(&[2]));

        let result = translator.translate(&make_node(), &[data, shape], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn test_reshape_flatten_to_2d() {
        let translator = ReshapeTranslator;
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[2, 3, 4, 5]), DType::F32);
        let shape = builder.constant(ConstantData::I64(vec![2, 60]), Shape::static_shape(&[2]));

        let result = translator.translate(&make_node(), &[data, shape], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_reshape_with_inference() {
        let translator = ReshapeTranslator;
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[2, 3, 4]), DType::F32);
        // Shape with -1 to infer dimension
        let shape = builder.constant(ConstantData::I64(vec![-1, 4]), Shape::static_shape(&[2]));

        let result = translator.translate(&make_node(), &[data, shape], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_reshape_with_allowzero() {
        let translator = ReshapeTranslator;
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[2, 3, 4]), DType::F32);
        let shape = builder.constant(
            ConstantData::I64(vec![0, 3, 4]), // 0 should copy from input
            Shape::static_shape(&[3]),
        );

        let result =
            translator.translate(&make_node_with_allowzero(1), &[data, shape], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_reshape_to_scalar() {
        let translator = ReshapeTranslator;
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[1]), DType::F32);
        let shape = builder.constant(ConstantData::I64(vec![]), Shape::static_shape(&[0]));

        let result = translator.translate(&make_node(), &[data, shape], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_reshape_from_scalar() {
        let translator = ReshapeTranslator;
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[]), DType::F32);
        let shape = builder.constant(ConstantData::I64(vec![1, 1]), Shape::static_shape(&[2]));

        let result = translator.translate(&make_node(), &[data, shape], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_reshape_dynamic_shape() {
        let translator = ReshapeTranslator;
        let mut builder = GraphBuilder::new();

        let data = builder.input("data", Shape::static_shape(&[2, 3, 4]), DType::F32);
        // Non-constant shape input (for dynamic reshape)
        let shape = builder.input("shape", Shape::static_shape(&[2]), DType::I64);

        let result = translator.translate(&make_node(), &[data, shape], &mut builder);
        assert!(result.is_ok());
    }

    // ===== Invalid Input Tests =====

    #[test]
    fn test_reshape_no_inputs() {
        let translator = ReshapeTranslator;
        let err = translator.input_requirement().validate(0, "Reshape");
        assert!(err.is_err());
        assert!(matches!(
            err.unwrap_err(),
            TranslationError::WrongInputCount {
                expected: 2,
                got: 0,
                ..
            }
        ));
    }

    #[test]
    fn test_reshape_one_input() {
        let translator = ReshapeTranslator;
        let err = translator.input_requirement().validate(1, "Reshape");
        assert!(err.is_err());
    }

    #[test]
    fn test_reshape_too_many_inputs() {
        let translator = ReshapeTranslator;
        let err = translator.input_requirement().validate(3, "Reshape");
        assert!(err.is_err());
    }

    // ===== Trait Method Tests =====

    #[test]
    fn test_op_type() {
        let translator = ReshapeTranslator;
        assert_eq!(translator.onnx_op_type(), "Reshape");
    }

    #[test]
    fn test_input_requirement() {
        let translator = ReshapeTranslator;
        let req = translator.input_requirement();
        assert!(matches!(req, InputRequirement::Exact(2)));
        assert!(!req.accepts_zero());
    }
}
