//! Split operation translator.

use crate::proto::NodeProto;
use crate::translators::{InputRequirement, OnnxAttributes, OnnxTranslator, TranslationError};
use hologram::ir::{ConstantData, Dim, GraphBuilder, NodeIndex, NodeOp};

/// Translator for ONNX Split operation.
///
/// Split divides a tensor into a list of tensors along a specified axis.
///
/// # Inputs
/// - data: Input tensor to split
/// - split (opset 13+, optional): 1D tensor specifying sizes of each output
///
/// # Attributes
/// - axis (default: 0): Axis along which to split
/// - split (opset < 13): Sizes of each output. If not specified, splits equally.
/// - num_outputs (opset 18+): Number of outputs when split is not specified
///
/// # Output
/// Returns multiple tensors from splitting the input.
#[derive(Debug, Default)]
pub struct SplitTranslator;

impl OnnxTranslator for SplitTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "Split"
    }

    fn input_requirement(&self) -> InputRequirement {
        // 1 input (data) or 2 inputs (data + split sizes for opset 13+)
        InputRequirement::Range(1, 2)
    }

    fn translate(
        &self,
        node: &NodeProto,
        inputs: &[NodeIndex],
        builder: &mut GraphBuilder,
    ) -> Result<Vec<NodeIndex>, TranslationError> {
        let data = inputs[0];

        // Get input node
        let input_node = builder.graph().node(data).ok_or_else(|| {
            TranslationError::IrBuilder("Split: input node not found".to_string())
        })?;
        let input_shape = input_node.op.shape.clone();
        let rank = input_shape.rank() as i64;

        // Get axis attribute (default: 0)
        let axis_raw = node.get_int_or("axis", 0);
        let axis = if axis_raw < 0 {
            rank + axis_raw
        } else {
            axis_raw
        };

        // Validate axis
        if axis < 0 || axis >= rank {
            return Err(TranslationError::invalid_attribute(
                "axis",
                format!("axis {} is out of bounds for rank {}", axis_raw, rank),
            ));
        }
        let axis_usize = axis as usize;

        // Get the dimension size at the split axis
        // Symbolic/Dynamic dimensions are allowed - hologram will resolve at runtime
        let axis_dim = &input_shape.dims[axis_usize];
        let axis_size: Option<usize> = match axis_dim {
            Dim::Static(s) => Some(*s),
            Dim::Symbolic(_) | Dim::Dynamic => None,
        };

        // Get split sizes - from second input (opset 13+) or attribute
        let split_sizes: Vec<usize> = if inputs.len() >= 2 {
            // Opset 13+: split sizes from second input
            let split_node = builder.graph().node(inputs[1]).ok_or_else(|| {
                TranslationError::IrBuilder("Split: split input not found".to_string())
            })?;

            if let NodeOp::Constant { data: const_data } = &split_node.op.op {
                match const_data {
                    ConstantData::I64(values) => values.iter().map(|&v| v as usize).collect(),
                    ConstantData::I32(values) => values.iter().map(|&v| v as usize).collect(),
                    _ => {
                        return Err(TranslationError::ShapeInference(
                            "Split: split sizes must be int32 or int64".to_string(),
                        ));
                    }
                }
            } else {
                return Err(TranslationError::ShapeInference(
                    "Split: split input must be a constant".to_string(),
                ));
            }
        } else if let Some(split_attr) = node.get_ints("split") {
            // Opset < 13: split sizes from attribute
            split_attr.iter().map(|&v| v as usize).collect()
        } else {
            // No split sizes specified - check for num_outputs or split equally
            let num_outputs = node.get_int("num_outputs").unwrap_or(0) as usize;
            if num_outputs > 0 {
                // Split into equal parts - requires known axis size
                match axis_size {
                    Some(size) => {
                        if size % num_outputs != 0 {
                            return Err(TranslationError::ShapeInference(format!(
                                "Split: cannot split dimension {} into {} equal parts",
                                size, num_outputs
                            )));
                        }
                        let part_size = size / num_outputs;
                        vec![part_size; num_outputs]
                    }
                    None => {
                        // For symbolic dimensions with num_outputs, we cannot compute
                        // exact split sizes at compile time. This requires explicit split sizes.
                        return Err(TranslationError::ShapeInference(
                            "Split: symbolic axis dimension requires explicit split sizes"
                                .to_string(),
                        ));
                    }
                }
            } else {
                // Default: return input as single output (identity)
                match axis_size {
                    Some(size) => vec![size],
                    None => {
                        // For symbolic dimension, return single output preserving the symbolic shape
                        // The slice operation will preserve symbolic dimensions
                        vec![0] // Marker for "full dimension" - handled below
                    }
                }
            }
        };

        // Validate split sizes sum to axis dimension (only when axis_size is known)
        // For symbolic dimensions, skip validation - hologram will resolve at runtime
        let sum: usize = split_sizes.iter().sum();
        if let Some(size) = axis_size {
            // Check for special case: single output with marker 0 for full symbolic dimension
            if !(split_sizes.len() == 1 && split_sizes[0] == 0) && sum != size {
                return Err(TranslationError::ShapeInference(format!(
                    "Split: split sizes sum ({}) does not match axis dimension ({})",
                    sum, size
                )));
            }
        }

        tracing::debug!(
            "Split: axis = {}, axis_size = {:?}, split_sizes = {:?}",
            axis,
            axis_size,
            split_sizes
        );

        // Create output shapes
        let mut outputs = Vec::with_capacity(split_sizes.len());
        let mut start = 0usize;

        for &size in &split_sizes {
            // Handle special case: size=0 means full dimension (symbolic identity split)
            if size == 0 {
                // For symbolic identity split, just pass through the input
                outputs.push(data);
                continue;
            }

            // Use Slice to extract this portion
            // starts: [0, ..., start, ..., 0]
            // ends: [dim0, ..., start+size, ..., dimN]
            let starts: Vec<i64> = input_shape
                .dims
                .iter()
                .enumerate()
                .map(|(i, _)| if i == axis_usize { start as i64 } else { 0 })
                .collect();

            let ends: Vec<i64> = input_shape
                .dims
                .iter()
                .enumerate()
                .map(|(i, d)| {
                    if i == axis_usize {
                        (start + size) as i64
                    } else {
                        match d {
                            Dim::Static(s) => *s as i64,
                            _ => i64::MAX, // Use max to indicate "all"
                        }
                    }
                })
                .collect();

            // Create axes vector (all axes in order)
            let axes: Vec<i32> = (0..input_shape.rank() as i32).collect();

            // Create slice operation
            let output = builder
                .slice(data, starts, ends, axes)
                .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;

            outputs.push(output);
            start += size;
        }

        Ok(outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::AttributeProto;
    use hologram::ir::{DType, Shape};

    fn make_node() -> NodeProto {
        NodeProto {
            name: "split_test".to_string(),
            op_type: "Split".to_string(),
            ..Default::default()
        }
    }

    fn make_node_with_axis(axis: i64) -> NodeProto {
        NodeProto {
            name: "split_test".to_string(),
            op_type: "Split".to_string(),
            attribute: vec![AttributeProto {
                name: "axis".to_string(),
                i: axis,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn make_node_with_split(axis: i64, split: Vec<i64>) -> NodeProto {
        NodeProto {
            name: "split_test".to_string(),
            op_type: "Split".to_string(),
            attribute: vec![
                AttributeProto {
                    name: "axis".to_string(),
                    i: axis,
                    ..Default::default()
                },
                AttributeProto {
                    name: "split".to_string(),
                    ints: split,
                    ..Default::default()
                },
            ],
            ..Default::default()
        }
    }

    fn make_node_with_num_outputs(axis: i64, num_outputs: i64) -> NodeProto {
        NodeProto {
            name: "split_test".to_string(),
            op_type: "Split".to_string(),
            attribute: vec![
                AttributeProto {
                    name: "axis".to_string(),
                    i: axis,
                    ..Default::default()
                },
                AttributeProto {
                    name: "num_outputs".to_string(),
                    i: num_outputs,
                    ..Default::default()
                },
            ],
            ..Default::default()
        }
    }

    // ===== Valid Input Tests =====

    #[test]
    fn test_split_equal_parts() {
        let translator = SplitTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[6, 4]), DType::F32);

        let result =
            translator.translate(&make_node_with_split(0, vec![2, 2, 2]), &[x], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 3);
    }

    #[test]
    fn test_split_unequal_parts() {
        let translator = SplitTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[6, 4]), DType::F32);

        let result =
            translator.translate(&make_node_with_split(0, vec![1, 2, 3]), &[x], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 3);
    }

    #[test]
    fn test_split_axis_1() {
        let translator = SplitTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[2, 6]), DType::F32);

        let result = translator.translate(&make_node_with_split(1, vec![3, 3]), &[x], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 2);
    }

    #[test]
    fn test_split_negative_axis() {
        let translator = SplitTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[2, 4]), DType::F32);

        // axis=-1 is equivalent to axis=1
        let result =
            translator.translate(&make_node_with_split(-1, vec![2, 2]), &[x], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 2);
    }

    #[test]
    fn test_split_with_num_outputs() {
        let translator = SplitTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[9, 4]), DType::F32);

        let result = translator.translate(&make_node_with_num_outputs(0, 3), &[x], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 3);
    }

    #[test]
    fn test_split_opset13_input() {
        let translator = SplitTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[6, 4]), DType::F32);
        let split = builder.constant(ConstantData::I64(vec![2, 4]), Shape::static_shape(&[2]));

        let result = translator.translate(&make_node_with_axis(0), &[x, split], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 2);
    }

    #[test]
    fn test_split_single_output() {
        let translator = SplitTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[4, 4]), DType::F32);

        // No split specified, single output
        let result = translator.translate(&make_node(), &[x], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn test_split_3d() {
        let translator = SplitTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[2, 6, 4]), DType::F32);

        let result =
            translator.translate(&make_node_with_split(1, vec![2, 2, 2]), &[x], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 3);
    }

    // ===== Invalid Input Tests =====

    #[test]
    fn test_split_no_inputs() {
        let translator = SplitTranslator;
        let err = translator.input_requirement().validate(0, "Split");
        assert!(err.is_err());
    }

    #[test]
    fn test_split_too_many_inputs() {
        let translator = SplitTranslator;
        let err = translator.input_requirement().validate(3, "Split");
        assert!(err.is_err());
    }

    #[test]
    fn test_split_axis_out_of_bounds() {
        let translator = SplitTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[4, 4]), DType::F32);

        let result = translator.translate(&make_node_with_axis(5), &[x], &mut builder);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("out of bounds"));
    }

    #[test]
    fn test_split_sum_mismatch() {
        let translator = SplitTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[6, 4]), DType::F32);

        // Split sizes don't sum to axis dimension
        let result = translator.translate(&make_node_with_split(0, vec![2, 2]), &[x], &mut builder);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("does not match"));
    }

    #[test]
    fn test_split_uneven_division() {
        let translator = SplitTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[5, 4]), DType::F32);

        // Cannot split 5 into 2 equal parts
        let result = translator.translate(&make_node_with_num_outputs(0, 2), &[x], &mut builder);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("equal parts"));
    }

    // ===== Trait Method Tests =====

    #[test]
    fn test_op_type() {
        let translator = SplitTranslator;
        assert_eq!(translator.onnx_op_type(), "Split");
    }

    #[test]
    fn test_input_requirement() {
        let translator = SplitTranslator;
        let req = translator.input_requirement();
        assert!(matches!(req, InputRequirement::Range(1, 2)));
        assert!(!req.accepts_zero());
    }

    // ===== Symbolic Dimension Tests =====

    #[test]
    fn test_split_symbolic_axis_with_explicit_sizes() {
        let translator = SplitTranslator;
        let mut builder = GraphBuilder::new();

        // Input with symbolic first dimension
        let shape = Shape::new(vec![Dim::Symbolic("seq_len".to_string()), Dim::Static(4)]);
        let x = builder.input("x", shape, DType::F32);

        // Split on static dimension (axis=1) with explicit sizes - should work
        let result = translator.translate(&make_node_with_split(1, vec![2, 2]), &[x], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 2);
    }

    #[test]
    fn test_split_symbolic_axis_identity() {
        let translator = SplitTranslator;
        let mut builder = GraphBuilder::new();

        // Input with symbolic first dimension
        let shape = Shape::new(vec![Dim::Symbolic("seq_len".to_string()), Dim::Static(4)]);
        let x = builder.input("x", shape, DType::F32);

        // Split on symbolic dimension without split sizes or num_outputs - identity
        let result = translator.translate(&make_node_with_axis(0), &[x], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        // Should return single output (identity pass-through)
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn test_split_symbolic_axis_with_num_outputs_fails() {
        let translator = SplitTranslator;
        let mut builder = GraphBuilder::new();

        // Input with symbolic first dimension
        let shape = Shape::new(vec![Dim::Symbolic("seq_len".to_string()), Dim::Static(4)]);
        let x = builder.input("x", shape, DType::F32);

        // Split on symbolic dimension with num_outputs - should fail
        // (can't compute equal split sizes without knowing axis dimension)
        let result = translator.translate(&make_node_with_num_outputs(0, 2), &[x], &mut builder);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("symbolic"));
    }

    #[test]
    fn test_split_dynamic_dim_identity() {
        let translator = SplitTranslator;
        let mut builder = GraphBuilder::new();

        // Input with dynamic first dimension
        let shape = Shape::new(vec![Dim::Dynamic, Dim::Static(4)]);
        let x = builder.input("x", shape, DType::F32);

        // Split on dynamic dimension without split sizes - identity
        let result = translator.translate(&make_node_with_axis(0), &[x], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);
    }
}
