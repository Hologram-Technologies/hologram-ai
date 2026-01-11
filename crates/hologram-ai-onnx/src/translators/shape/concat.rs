//! Concat operation translator.

use crate::proto::NodeProto;
use crate::translators::{InputRequirement, OnnxAttributes, OnnxTranslator, TranslationError};
use hologram::ir::{ConstantData, GraphBuilder, NodeIndex, NodeOp, Shape};

/// Translator for ONNX Concat operation.
///
/// Concat concatenates tensors along a specified axis.
///
/// # Inputs
/// - inputs: One or more tensors to concatenate (variadic)
///
/// # Attributes
/// - axis (required): The axis along which to concatenate
///
/// # Constant Folding
/// If all inputs are 1D constants of the same integer type, constant folding is performed.
#[derive(Debug, Default)]
pub struct ConcatTranslator;

impl OnnxTranslator for ConcatTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "Concat"
    }

    fn input_requirement(&self) -> InputRequirement {
        InputRequirement::AtLeast(1)
    }

    fn translate(
        &self,
        node: &NodeProto,
        inputs: &[NodeIndex],
        builder: &mut GraphBuilder,
    ) -> Result<Vec<NodeIndex>, TranslationError> {
        // Get axis attribute (required)
        let axis_raw = node
            .get_int("axis")
            .ok_or_else(|| TranslationError::missing_attribute("Concat", "axis"))?
            as i32;

        // Get first input to determine rank
        let first_node = builder.graph().node(inputs[0]).ok_or_else(|| {
            TranslationError::IrBuilder("Concat: first input not found".to_string())
        })?;
        let rank = first_node.op.shape.rank() as i32;

        // Check that all inputs have the same rank (ONNX requirement)
        for (i, &input) in inputs.iter().enumerate().skip(1) {
            if let Some(node) = builder.graph().node(input) {
                let input_rank = node.op.shape.rank() as i32;
                if input_rank != rank {
                    return Err(TranslationError::ShapeInference(format!(
                        "Concat: All inputs must have the same rank. First input has rank {}, but input {} has rank {}",
                        rank, i, input_rank
                    )));
                }
            }
        }

        // Normalize negative axis
        let axis = if axis_raw < 0 {
            rank + axis_raw
        } else {
            axis_raw
        };

        // Validate axis is in bounds
        if axis < 0 || axis >= rank {
            return Err(TranslationError::invalid_attribute(
                "axis",
                format!(
                    "axis {} (raw: {}) is out of bounds for rank {} tensor (valid range: [0, {}))",
                    axis, axis_raw, rank, rank
                ),
            ));
        }

        tracing::debug!(
            "Concat: {} inputs, axis_raw = {}, normalized axis = {}, rank = {}",
            inputs.len(),
            axis_raw,
            axis,
            rank
        );

        // Try constant folding for 1D integer tensors concatenated along axis 0
        if axis == 0
            && rank == 1
            && let Some(result) = try_constant_fold_concat(inputs, builder)
        {
            tracing::debug!("Concat: constant folding succeeded");
            return Ok(vec![result]);
        }

        // No constant folding, create regular concat node
        let result = builder
            .concat(inputs, axis)
            .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;

        Ok(vec![result])
    }

    fn supports_constant_folding(&self) -> bool {
        true
    }
}

/// Try to constant-fold a concat operation.
/// Returns Some(result) if all inputs are constants of the same integer type.
fn try_constant_fold_concat(inputs: &[NodeIndex], builder: &mut GraphBuilder) -> Option<NodeIndex> {
    // Check if all inputs are constants
    let all_constants = inputs.iter().all(|&idx| {
        builder
            .graph()
            .node(idx)
            .is_some_and(|node| matches!(node.op.op, NodeOp::Constant { .. }))
    });

    if !all_constants {
        return None;
    }

    // Get first node to determine type
    let first_node = builder.graph().node(inputs[0])?;
    if let NodeOp::Constant { data: first_data } = &first_node.op.op {
        match first_data {
            ConstantData::I64(_) => {
                let mut result_values = Vec::new();
                for &idx in inputs {
                    let node = builder.graph().node(idx)?;
                    if let NodeOp::Constant {
                        data: ConstantData::I64(values),
                    } = &node.op.op
                    {
                        result_values.extend_from_slice(values);
                    } else {
                        return None;
                    }
                }
                let output_shape = Shape::static_shape(&[result_values.len()]);
                Some(builder.constant(ConstantData::I64(result_values), output_shape))
            }
            ConstantData::I32(_) => {
                let mut result_values = Vec::new();
                for &idx in inputs {
                    let node = builder.graph().node(idx)?;
                    if let NodeOp::Constant {
                        data: ConstantData::I32(values),
                    } = &node.op.op
                    {
                        result_values.extend_from_slice(values);
                    } else {
                        return None;
                    }
                }
                let output_shape = Shape::static_shape(&[result_values.len()]);
                Some(builder.constant(ConstantData::I32(result_values), output_shape))
            }
            ConstantData::F32(_) => {
                let mut result_values = Vec::new();
                for &idx in inputs {
                    let node = builder.graph().node(idx)?;
                    if let NodeOp::Constant {
                        data: ConstantData::F32(values),
                    } = &node.op.op
                    {
                        result_values.extend_from_slice(values);
                    } else {
                        return None;
                    }
                }
                let output_shape = Shape::static_shape(&[result_values.len()]);
                Some(builder.constant(ConstantData::F32(result_values), output_shape))
            }
            _ => None,
        }
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::AttributeProto;
    use hologram::ir::DType;

    fn make_node(axis: i64) -> NodeProto {
        NodeProto {
            name: "concat_test".to_string(),
            op_type: "Concat".to_string(),
            attribute: vec![AttributeProto {
                name: "axis".to_string(),
                i: axis,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn make_node_no_axis() -> NodeProto {
        NodeProto {
            name: "concat_test".to_string(),
            op_type: "Concat".to_string(),
            ..Default::default()
        }
    }

    // ===== Valid Input Tests =====

    #[test]
    fn test_concat_axis_0() {
        let translator = ConcatTranslator;
        let mut builder = GraphBuilder::new();

        let a = builder.input("a", Shape::static_shape(&[2, 3]), DType::F32);
        let b = builder.input("b", Shape::static_shape(&[2, 3]), DType::F32);

        let result = translator.translate(&make_node(0), &[a, b], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn test_concat_axis_1() {
        let translator = ConcatTranslator;
        let mut builder = GraphBuilder::new();

        let a = builder.input("a", Shape::static_shape(&[2, 3]), DType::F32);
        let b = builder.input("b", Shape::static_shape(&[2, 4]), DType::F32);

        let result = translator.translate(&make_node(1), &[a, b], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_concat_negative_axis() {
        let translator = ConcatTranslator;
        let mut builder = GraphBuilder::new();

        let a = builder.input("a", Shape::static_shape(&[2, 3, 4]), DType::F32);
        let b = builder.input("b", Shape::static_shape(&[2, 3, 5]), DType::F32);

        // axis=-1 should be equivalent to axis=2
        let result = translator.translate(&make_node(-1), &[a, b], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_concat_single_input() {
        let translator = ConcatTranslator;
        let mut builder = GraphBuilder::new();

        let a = builder.input("a", Shape::static_shape(&[2, 3]), DType::F32);

        let result = translator.translate(&make_node(0), &[a], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_concat_multiple_inputs() {
        let translator = ConcatTranslator;
        let mut builder = GraphBuilder::new();

        let a = builder.input("a", Shape::static_shape(&[2, 3]), DType::F32);
        let b = builder.input("b", Shape::static_shape(&[2, 3]), DType::F32);
        let c = builder.input("c", Shape::static_shape(&[2, 3]), DType::F32);
        let d = builder.input("d", Shape::static_shape(&[2, 3]), DType::F32);

        let result = translator.translate(&make_node(0), &[a, b, c, d], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_concat_1d() {
        let translator = ConcatTranslator;
        let mut builder = GraphBuilder::new();

        let a = builder.input("a", Shape::static_shape(&[3]), DType::F32);
        let b = builder.input("b", Shape::static_shape(&[4]), DType::F32);

        let result = translator.translate(&make_node(0), &[a, b], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_concat_constant_folding_i64() {
        let translator = ConcatTranslator;
        let mut builder = GraphBuilder::new();

        let a = builder.constant(ConstantData::I64(vec![1, 2, 3]), Shape::static_shape(&[3]));
        let b = builder.constant(ConstantData::I64(vec![4, 5]), Shape::static_shape(&[2]));

        let result = translator.translate(&make_node(0), &[a, b], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();

        let node = builder.graph().node(outputs[0]).unwrap();
        if let NodeOp::Constant { data } = &node.op.op {
            if let ConstantData::I64(values) = data {
                assert_eq!(values.as_slice(), &[1i64, 2, 3, 4, 5]);
            } else {
                panic!("Expected I64 data");
            }
        } else {
            panic!("Expected Constant node");
        }
    }

    #[test]
    fn test_concat_constant_folding_i32() {
        let translator = ConcatTranslator;
        let mut builder = GraphBuilder::new();

        let a = builder.constant(ConstantData::I32(vec![10, 20]), Shape::static_shape(&[2]));
        let b = builder.constant(ConstantData::I32(vec![30]), Shape::static_shape(&[1]));

        let result = translator.translate(&make_node(0), &[a, b], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();

        let node = builder.graph().node(outputs[0]).unwrap();
        if let NodeOp::Constant { data } = &node.op.op {
            if let ConstantData::I32(values) = data {
                assert_eq!(values.as_slice(), &[10i32, 20, 30]);
            } else {
                panic!("Expected I32 data");
            }
        } else {
            panic!("Expected Constant node");
        }
    }

    #[test]
    fn test_concat_4d() {
        let translator = ConcatTranslator;
        let mut builder = GraphBuilder::new();

        let a = builder.input("a", Shape::static_shape(&[1, 3, 224, 224]), DType::F32);
        let b = builder.input("b", Shape::static_shape(&[1, 3, 224, 224]), DType::F32);

        let result = translator.translate(&make_node(0), &[a, b], &mut builder);
        assert!(result.is_ok());
    }

    // ===== Invalid Input Tests =====

    #[test]
    fn test_concat_no_inputs() {
        let translator = ConcatTranslator;
        let err = translator.input_requirement().validate(0, "Concat");
        assert!(err.is_err());
        assert!(matches!(
            err.unwrap_err(),
            TranslationError::NotEnoughInputs { min: 1, got: 0, .. }
        ));
    }

    #[test]
    fn test_concat_missing_axis() {
        let translator = ConcatTranslator;
        let mut builder = GraphBuilder::new();

        let a = builder.input("a", Shape::static_shape(&[2, 3]), DType::F32);
        let b = builder.input("b", Shape::static_shape(&[2, 3]), DType::F32);

        let result = translator.translate(&make_node_no_axis(), &[a, b], &mut builder);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("axis"));
    }

    #[test]
    fn test_concat_axis_out_of_bounds() {
        let translator = ConcatTranslator;
        let mut builder = GraphBuilder::new();

        let a = builder.input("a", Shape::static_shape(&[2, 3]), DType::F32);
        let b = builder.input("b", Shape::static_shape(&[2, 3]), DType::F32);

        let result = translator.translate(&make_node(5), &[a, b], &mut builder);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("out of bounds"));
    }

    #[test]
    fn test_concat_rank_mismatch() {
        let translator = ConcatTranslator;
        let mut builder = GraphBuilder::new();

        let a = builder.input("a", Shape::static_shape(&[2, 3]), DType::F32);
        let b = builder.input("b", Shape::static_shape(&[2, 3, 4]), DType::F32);

        let result = translator.translate(&make_node(0), &[a, b], &mut builder);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("rank"));
    }

    // ===== Trait Method Tests =====

    #[test]
    fn test_op_type() {
        let translator = ConcatTranslator;
        assert_eq!(translator.onnx_op_type(), "Concat");
    }

    #[test]
    fn test_input_requirement() {
        let translator = ConcatTranslator;
        let req = translator.input_requirement();
        assert!(matches!(req, InputRequirement::AtLeast(1)));
        assert!(!req.accepts_zero());
    }

    #[test]
    fn test_supports_constant_folding() {
        let translator = ConcatTranslator;
        assert!(translator.supports_constant_folding());
    }
}
