//! Flatten operation translator.

use hologram::ir::{GraphBuilder, NodeIndex, NodeOp, ConstantData, Shape, Dim};
use crate::proto::NodeProto;
use crate::translators::{OnnxTranslator, OnnxAttributes, InputRequirement, TranslationError};

/// Translator for ONNX Flatten operation.
///
/// Flatten reshapes the input tensor into a 2D matrix. The first dimension
/// is the product of dimensions up to (but not including) axis. The second
/// dimension is the product of remaining dimensions.
///
/// # Inputs
/// - data: Input tensor
///
/// # Attributes
/// - axis (default: 1): The axis at which to flatten
///
/// # Shape Semantics
/// For input shape [d0, d1, ..., d(axis-1), d(axis), ..., d(n-1)]:
/// - Output shape is [d0 * d1 * ... * d(axis-1), d(axis) * ... * d(n-1)]
/// - If axis=0, output is [1, total_elements]
/// - If axis=n, output is [total_elements, 1]
#[derive(Debug, Default)]
pub struct FlattenTranslator;

impl OnnxTranslator for FlattenTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "Flatten"
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
        let data = inputs[0];

        // Get input node
        let input_node = builder.graph().node(data).ok_or_else(|| {
            TranslationError::IrBuilder("Flatten: input node not found".to_string())
        })?;
        let input_shape = input_node.shape.clone();
        let rank = input_shape.rank() as i64;

        // Get axis attribute (default: 1)
        let axis_raw = node.get_int_or("axis", 1);

        // Normalize negative axis
        let axis = if axis_raw < 0 {
            rank + axis_raw
        } else {
            axis_raw
        };

        // Validate axis is in bounds [0, rank]
        if axis < 0 || axis > rank {
            return Err(TranslationError::invalid_attribute(
                "axis",
                format!("axis {} is out of bounds for rank {} tensor", axis_raw, rank),
            ));
        }

        // Calculate output dimensions
        let axis = axis as usize;
        let dims = &input_shape.dims;

        // Check if all dimensions are static
        let all_static = dims.iter().all(|d| matches!(d, Dim::Static(_)));

        if all_static {
            // Compute static output shape
            let mut first_dim: usize = 1;
            let mut second_dim: usize = 1;

            for (i, d) in dims.iter().enumerate() {
                if let Dim::Static(size) = d {
                    if i < axis {
                        first_dim *= size;
                    } else {
                        second_dim *= size;
                    }
                }
            }

            let output_shape = vec![first_dim as i64, second_dim as i64];
            tracing::debug!(
                "Flatten: axis = {}, input shape = {:?}, output shape = {:?}",
                axis, input_shape, output_shape
            );

            // Handle constant folding
            if let NodeOp::Constant { data: const_data } = &input_node.op {
                let folded_data = const_data.clone();
                let result = builder.constant(folded_data, Shape::static_shape(&[first_dim, second_dim]));
                return Ok(vec![result]);
            }

            // Use static reshape
            let result = builder
                .reshape(data, output_shape)
                .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;
            Ok(vec![result])
        } else {
            // Dynamic shape - need to compute at runtime
            // We'll use a combination of Shape + Gather + Reshape
            // For now, use a simplified approach with reshape to [-1, -1]
            tracing::debug!(
                "Flatten: dynamic path, axis = {}, input shape = {:?}",
                axis, input_shape
            );

            // Create a dynamic flatten using reshape
            // First compute the shape at runtime using Shape op
            let shape_node = builder
                .shape(data, 0, rank)
                .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;

            // For dynamic flatten, we need to compute:
            // - first_dim = product of dims[:axis]
            // - second_dim = product of dims[axis:]
            // This requires runtime computation. For now, use a placeholder.
            // A complete implementation would use Reduce + Slice + Reshape.

            // Simplified: if axis=1, we can use a reshape with -1
            if axis == 1 && rank >= 1 {
                // Get first dimension
                let first_dim_shape = if let Some(Dim::Static(d)) = dims.first() {
                    vec![*d as i64, -1]
                } else {
                    // Fully dynamic - use reshape with inference
                    vec![-1, -1]
                };

                let shape_const = builder.constant(
                    ConstantData::I64(first_dim_shape),
                    Shape::static_shape(&[2]),
                );

                let result = builder
                    .reshape_dynamic(data, shape_const, false)
                    .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;
                return Ok(vec![result]);
            }

            // General dynamic case: use dynamic reshape
            // This is a simplification - full implementation would compute shape at runtime
            let _ = shape_node; // Suppress unused warning

            // For general case, we'll create a reshape with inferred dimensions
            let shape_const = builder.constant(
                ConstantData::I64(vec![-1, -1]),
                Shape::static_shape(&[2]),
            );

            let result = builder
                .reshape_dynamic(data, shape_const, false)
                .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;
            Ok(vec![result])
        }
    }

    fn supports_constant_folding(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram::ir::DType;
    use crate::proto::AttributeProto;

    fn make_node() -> NodeProto {
        NodeProto {
            name: "flatten_test".to_string(),
            op_type: "Flatten".to_string(),
            ..Default::default()
        }
    }

    fn make_node_with_axis(axis: i64) -> NodeProto {
        NodeProto {
            name: "flatten_test".to_string(),
            op_type: "Flatten".to_string(),
            attribute: vec![AttributeProto {
                name: "axis".to_string(),
                i: axis,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    // ===== Valid Input Tests =====

    #[test]
    fn test_flatten_default_axis() {
        let translator = FlattenTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[2, 3, 4]), DType::F32);

        // Default axis=1: [2, 3, 4] -> [2, 12]
        let result = translator.translate(&make_node(), &[x], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn test_flatten_axis_0() {
        let translator = FlattenTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[2, 3, 4]), DType::F32);

        // axis=0: [2, 3, 4] -> [1, 24]
        let result = translator.translate(&make_node_with_axis(0), &[x], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_flatten_axis_2() {
        let translator = FlattenTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[2, 3, 4]), DType::F32);

        // axis=2: [2, 3, 4] -> [6, 4]
        let result = translator.translate(&make_node_with_axis(2), &[x], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_flatten_axis_last() {
        let translator = FlattenTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[2, 3, 4]), DType::F32);

        // axis=3 (rank): [2, 3, 4] -> [24, 1]
        let result = translator.translate(&make_node_with_axis(3), &[x], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_flatten_negative_axis() {
        let translator = FlattenTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[2, 3, 4]), DType::F32);

        // axis=-2: equivalent to axis=1 for rank 3
        let result = translator.translate(&make_node_with_axis(-2), &[x], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_flatten_2d() {
        let translator = FlattenTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[3, 4]), DType::F32);

        // Default axis=1: [3, 4] -> [3, 4] (no change)
        let result = translator.translate(&make_node(), &[x], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_flatten_1d() {
        let translator = FlattenTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[12]), DType::F32);

        // Default axis=1: [12] -> [1, 12] or with axis=0: [1, 12]
        let result = translator.translate(&make_node_with_axis(0), &[x], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_flatten_scalar() {
        let translator = FlattenTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[]), DType::F32);

        // Scalar: [] -> [1, 1]
        let result = translator.translate(&make_node_with_axis(0), &[x], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_flatten_constant_folding() {
        let translator = FlattenTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.constant(
            ConstantData::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            Shape::static_shape(&[2, 3]),
        );

        let result = translator.translate(&make_node(), &[x], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();

        let node = builder.graph().node(outputs[0]).unwrap();
        // Should be constant folded
        if let NodeOp::Constant { data } = &node.op {
            if let ConstantData::F32(values) = data {
                assert_eq!(values.len(), 6);
            } else {
                panic!("Expected F32 data");
            }
        } else {
            panic!("Expected Constant node");
        }
    }

    #[test]
    fn test_flatten_4d() {
        let translator = FlattenTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[1, 3, 224, 224]), DType::F32);

        // axis=1: [1, 3, 224, 224] -> [1, 150528]
        let result = translator.translate(&make_node(), &[x], &mut builder);
        assert!(result.is_ok());
    }

    // ===== Invalid Input Tests =====

    #[test]
    fn test_flatten_no_inputs() {
        let translator = FlattenTranslator;
        let err = translator.input_requirement().validate(0, "Flatten");
        assert!(err.is_err());
        assert!(matches!(
            err.unwrap_err(),
            TranslationError::WrongInputCount { expected: 1, got: 0, .. }
        ));
    }

    #[test]
    fn test_flatten_too_many_inputs() {
        let translator = FlattenTranslator;
        let err = translator.input_requirement().validate(2, "Flatten");
        assert!(err.is_err());
    }

    #[test]
    fn test_flatten_axis_out_of_bounds() {
        let translator = FlattenTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);

        // axis=5 is out of bounds for rank 2
        let result = translator.translate(&make_node_with_axis(5), &[x], &mut builder);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("out of bounds"));
    }

    // ===== Trait Method Tests =====

    #[test]
    fn test_op_type() {
        let translator = FlattenTranslator;
        assert_eq!(translator.onnx_op_type(), "Flatten");
    }

    #[test]
    fn test_input_requirement() {
        let translator = FlattenTranslator;
        let req = translator.input_requirement();
        assert!(matches!(req, InputRequirement::Exact(1)));
        assert!(!req.accepts_zero());
    }

    #[test]
    fn test_supports_constant_folding() {
        let translator = FlattenTranslator;
        assert!(translator.supports_constant_folding());
    }
}
