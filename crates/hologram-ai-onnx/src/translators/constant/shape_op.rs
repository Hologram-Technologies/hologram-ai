//! Shape operation translator.

use crate::proto::NodeProto;
use crate::translators::{InputRequirement, OnnxAttributes, OnnxTranslator, TranslationError};
use hologram::ir::{ConstantData, Dim, GraphBuilder, NodeIndex, Shape};

/// Translator for ONNX Shape operation.
///
/// Extracts the shape of the input tensor as a 1D int64 tensor.
/// Supports optional start/end attributes for partial shape extraction (opset 15+).
///
/// # Inputs
/// - data: Input tensor whose shape to extract
///
/// # Attributes
/// - start: Starting index for shape slice (default: 0)
/// - end: Ending index for shape slice (default: rank)
///
/// # Outputs
/// - shape: 1D int64 tensor containing shape dimensions
#[derive(Debug, Default)]
pub struct ShapeOpTranslator;

impl OnnxTranslator for ShapeOpTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "Shape"
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
        let data_input = inputs[0];

        // Get input node to determine rank
        let input_node = builder.graph().node(data_input).ok_or_else(|| {
            TranslationError::IrBuilder("Shape: input node not found".to_string())
        })?;
        let rank = input_node.op.shape.rank() as i64;

        // Extract start/end indices if specified (ONNX opset 15+)
        let start = node.get_int_or("start", 0);
        let end = node.get_int_or("end", rank);

        // Normalize negative indices
        let start_idx = if start < 0 { rank + start } else { start };
        let end_idx = if end < 0 { rank + end } else { end };

        // Validate range
        if start_idx < 0 || end_idx > rank || start_idx > end_idx {
            return Err(TranslationError::invalid_attribute(
                "start/end",
                format!("invalid range [{}:{}] for rank {}", start, end, rank),
            ));
        }

        // Get dimensions from input shape
        let dims = &input_node.op.shape.dims;

        // Check if all dimensions in the range are static
        let mut all_static = true;
        let mut shape_values = Vec::with_capacity((end_idx - start_idx) as usize);

        for i in start_idx..end_idx {
            match &dims[i as usize] {
                Dim::Static(size) => shape_values.push(*size as i64),
                Dim::Symbolic(_) | Dim::Dynamic => {
                    all_static = false;
                    break;
                }
            }
        }

        if all_static {
            // All dimensions are static, use constant folding
            let output_shape = Shape::static_shape(&[shape_values.len()]);
            let constant_data = ConstantData::I64(shape_values);
            let result = builder.constant(constant_data, output_shape);
            Ok(vec![result])
        } else {
            // Has symbolic/dynamic dimensions, use runtime Shape operation
            let result = builder
                .shape(data_input, start_idx, end_idx)
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
    use crate::proto::AttributeProto;
    use hologram::ir::{DType, NodeOp};

    fn make_node() -> NodeProto {
        NodeProto {
            name: "shape_test".to_string(),
            op_type: "Shape".to_string(),
            ..Default::default()
        }
    }

    fn make_node_with_range(start: i64, end: i64) -> NodeProto {
        NodeProto {
            name: "shape_test".to_string(),
            op_type: "Shape".to_string(),
            attribute: vec![
                AttributeProto {
                    name: "start".to_string(),
                    i: start,
                    ..Default::default()
                },
                AttributeProto {
                    name: "end".to_string(),
                    i: end,
                    ..Default::default()
                },
            ],
            ..Default::default()
        }
    }

    // ===== Valid Input Tests =====

    #[test]
    fn test_shape_static_2d() {
        let translator = ShapeOpTranslator;
        let mut builder = GraphBuilder::new();

        let input = builder.input("input", Shape::static_shape(&[2, 3]), DType::F32);
        let node = make_node();

        let result = translator.translate(&node, &[input], &mut builder);
        assert!(result.is_ok());

        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);

        // Output should be constant [2, 3]
        let output_node = builder.graph().node(outputs[0]).unwrap();
        assert_eq!(output_node.op.shape.rank(), 1);
        assert_eq!(output_node.op.dtype, DType::I64);

        if let NodeOp::Constant { data } = &output_node.op.op {
            if let ConstantData::I64(values) = data {
                assert_eq!(values.as_slice(), &[2i64, 3]);
            } else {
                panic!("Expected I64 data");
            }
        } else {
            panic!("Expected Constant node");
        }
    }

    #[test]
    fn test_shape_static_4d() {
        let translator = ShapeOpTranslator;
        let mut builder = GraphBuilder::new();

        let input = builder.input("input", Shape::static_shape(&[1, 3, 224, 224]), DType::F32);
        let node = make_node();

        let result = translator.translate(&node, &[input], &mut builder);
        assert!(result.is_ok());

        let outputs = result.unwrap();
        let output_node = builder.graph().node(outputs[0]).unwrap();

        if let NodeOp::Constant { data } = &output_node.op.op {
            if let ConstantData::I64(values) = data {
                assert_eq!(values.as_slice(), &[1i64, 3, 224, 224]);
            } else {
                panic!("Expected I64 data");
            }
        } else {
            panic!("Expected Constant node");
        }
    }

    #[test]
    fn test_shape_with_start_end() {
        let translator = ShapeOpTranslator;
        let mut builder = GraphBuilder::new();

        let input = builder.input("input", Shape::static_shape(&[2, 3, 4, 5]), DType::F32);
        let node = make_node_with_range(1, 3); // Extract [3, 4]

        let result = translator.translate(&node, &[input], &mut builder);
        assert!(result.is_ok());

        let outputs = result.unwrap();
        let output_node = builder.graph().node(outputs[0]).unwrap();

        if let NodeOp::Constant { data } = &output_node.op.op {
            if let ConstantData::I64(values) = data {
                assert_eq!(values.as_slice(), &[3i64, 4]);
            } else {
                panic!("Expected I64 data");
            }
        } else {
            panic!("Expected Constant node");
        }
    }

    #[test]
    fn test_shape_negative_indices() {
        let translator = ShapeOpTranslator;
        let mut builder = GraphBuilder::new();

        let input = builder.input("input", Shape::static_shape(&[2, 3, 4, 5]), DType::F32);
        // -2 to -1 means [4] (second to last element)
        let node = make_node_with_range(-2, -1);

        let result = translator.translate(&node, &[input], &mut builder);
        assert!(result.is_ok());

        let outputs = result.unwrap();
        let output_node = builder.graph().node(outputs[0]).unwrap();

        if let NodeOp::Constant { data } = &output_node.op.op {
            if let ConstantData::I64(values) = data {
                assert_eq!(values.as_slice(), &[4i64]);
            } else {
                panic!("Expected I64 data");
            }
        } else {
            panic!("Expected Constant node");
        }
    }

    #[test]
    fn test_shape_symbolic_dimension() {
        let translator = ShapeOpTranslator;
        let mut builder = GraphBuilder::new();

        // Create input with symbolic dimension
        let shape = Shape::new(vec![
            Dim::Symbolic("batch".to_string()),
            Dim::Static(3),
            Dim::Static(224),
            Dim::Static(224),
        ]);
        let input = builder.input("input", shape, DType::F32);
        let node = make_node();

        // Should succeed and create a runtime Shape operation
        let result = translator.translate(&node, &[input], &mut builder);
        assert!(result.is_ok());

        let outputs = result.unwrap();
        let output_node = builder.graph().node(outputs[0]).unwrap();
        assert_eq!(output_node.op.shape.rank(), 1);
        assert_eq!(output_node.op.dtype, DType::I64);
    }

    #[test]
    fn test_shape_partial_static() {
        let translator = ShapeOpTranslator;
        let mut builder = GraphBuilder::new();

        // Create input with symbolic batch dimension
        let shape = Shape::new(vec![Dim::Symbolic("batch".to_string()), Dim::Static(768)]);
        let input = builder.input("input", shape, DType::F32);

        // Extract only the static part [1:2]
        let node = make_node_with_range(1, 2);

        let result = translator.translate(&node, &[input], &mut builder);
        assert!(result.is_ok());

        let outputs = result.unwrap();
        let output_node = builder.graph().node(outputs[0]).unwrap();

        // Should be constant since we extracted only static dims
        if let NodeOp::Constant { data } = &output_node.op.op {
            if let ConstantData::I64(values) = data {
                assert_eq!(values.as_slice(), &[768i64]);
            } else {
                panic!("Expected I64 data");
            }
        } else {
            panic!("Expected Constant node");
        }
    }

    #[test]
    fn test_shape_scalar_input() {
        let translator = ShapeOpTranslator;
        let mut builder = GraphBuilder::new();

        let input = builder.input("input", Shape::static_shape(&[]), DType::F32);
        let node = make_node();

        let result = translator.translate(&node, &[input], &mut builder);
        assert!(result.is_ok());

        let outputs = result.unwrap();
        let output_node = builder.graph().node(outputs[0]).unwrap();

        // Scalar has rank 0, so output is empty 1D tensor
        if let NodeOp::Constant { data } = &output_node.op.op {
            if let ConstantData::I64(values) = data {
                assert!(values.is_empty());
            } else {
                panic!("Expected I64 data");
            }
        } else {
            panic!("Expected Constant node");
        }
    }

    // ===== Invalid Input Tests =====

    #[test]
    fn test_shape_no_inputs() {
        let translator = ShapeOpTranslator;
        let err = translator.input_requirement().validate(0, "Shape");
        assert!(err.is_err());
        assert!(matches!(
            err.unwrap_err(),
            TranslationError::WrongInputCount {
                expected: 1,
                got: 0,
                ..
            }
        ));
    }

    #[test]
    fn test_shape_too_many_inputs() {
        let translator = ShapeOpTranslator;
        let err = translator.input_requirement().validate(2, "Shape");
        assert!(err.is_err());
    }

    #[test]
    fn test_shape_invalid_range() {
        let translator = ShapeOpTranslator;
        let mut builder = GraphBuilder::new();

        let input = builder.input("input", Shape::static_shape(&[2, 3]), DType::F32);
        let node = make_node_with_range(3, 5); // Out of bounds

        let result = translator.translate(&node, &[input], &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_shape_start_greater_than_end() {
        let translator = ShapeOpTranslator;
        let mut builder = GraphBuilder::new();

        let input = builder.input("input", Shape::static_shape(&[2, 3, 4]), DType::F32);
        let node = make_node_with_range(2, 1); // start > end

        let result = translator.translate(&node, &[input], &mut builder);
        assert!(result.is_err());
    }

    // ===== Trait Method Tests =====

    #[test]
    fn test_shape_op_type() {
        let translator = ShapeOpTranslator;
        assert_eq!(translator.onnx_op_type(), "Shape");
    }

    #[test]
    fn test_shape_supports_folding() {
        let translator = ShapeOpTranslator;
        assert!(translator.supports_constant_folding());
    }

    #[test]
    fn test_input_requirement() {
        let translator = ShapeOpTranslator;
        let req = translator.input_requirement();
        assert!(matches!(req, InputRequirement::Exact(1)));
        assert!(!req.accepts_zero());
    }
}
