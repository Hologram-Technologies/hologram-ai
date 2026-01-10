//! Squeeze operation translator.

use hologram::ir::{GraphBuilder, NodeIndex, NodeOp, ConstantData, Shape, Dim};
use crate::proto::NodeProto;
use crate::translators::{OnnxTranslator, OnnxAttributes, InputRequirement, TranslationError};

/// Translator for ONNX Squeeze operation.
///
/// Squeeze removes dimensions of size 1 from the input tensor.
///
/// # Inputs
/// - data: Input tensor
/// - axes (opset 13+, optional): 1D tensor specifying which axes to squeeze
///
/// # Attributes
/// - axes (opset < 13): List of axes to squeeze. If not specified, all axes with size 1 are removed.
///
/// # Constant Folding
/// If the input is a constant, the squeeze is performed at compile time.
#[derive(Debug, Default)]
pub struct SqueezeTranslator;

impl OnnxTranslator for SqueezeTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "Squeeze"
    }

    fn input_requirement(&self) -> InputRequirement {
        // 1 input (data) or 2 inputs (data + axes for opset 13+)
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
            TranslationError::IrBuilder("Squeeze: input node not found".to_string())
        })?;
        let input_shape = input_node.shape.clone();
        let input_dtype = input_node.dtype;
        let rank = input_shape.rank();

        // Get axes to squeeze
        let axes: Vec<i32> = if inputs.len() >= 2 {
            // Opset 13+: axes from second input (must be constant)
            let axes_node = builder.graph().node(inputs[1]).ok_or_else(|| {
                TranslationError::IrBuilder("Squeeze: axes input not found".to_string())
            })?;

            if let NodeOp::Constant { data: constant_data } = &axes_node.op {
                match constant_data {
                    ConstantData::I64(values) => values.iter().map(|&v| v as i32).collect(),
                    ConstantData::I32(values) => values.clone(),
                    _ => {
                        return Err(TranslationError::ShapeInference(
                            "Squeeze: axes must be int32 or int64".to_string(),
                        ))
                    }
                }
            } else {
                return Err(TranslationError::ShapeInference(
                    "Squeeze: axes input must be a constant".to_string(),
                ));
            }
        } else if let Some(axes_attr) = node.get_ints("axes") {
            // Opset < 13: axes from attribute
            axes_attr.iter().map(|&v| v as i32).collect()
        } else {
            // No axes specified: squeeze all dimensions of size 1
            input_shape.dims.iter()
                .enumerate()
                .filter_map(|(i, d)| {
                    if d.static_value() == Some(1) {
                        Some(i as i32)
                    } else {
                        None
                    }
                })
                .collect()
        };

        // Normalize negative axes
        let normalized_axes: Vec<i32> = axes.iter()
            .map(|&a| if a < 0 { rank as i32 + a } else { a })
            .collect();

        // Validate axes
        for &axis in &normalized_axes {
            if axis < 0 || axis >= rank as i32 {
                return Err(TranslationError::invalid_attribute(
                    "axes",
                    format!("axis {} is out of bounds for rank {}", axis, rank),
                ));
            }
            // Validate that dimension at axis is 1
            let dim = &input_shape.dims[axis as usize];
            if let Some(size) = dim.static_value()
                && size != 1
            {
                return Err(TranslationError::invalid_attribute(
                    "axes",
                    format!("cannot squeeze axis {} with size {} (must be 1)", axis, size),
                ));
            }
        }

        // Build output shape by removing squeezed dimensions
        let output_dims: Vec<Dim> = input_shape.dims.iter()
            .enumerate()
            .filter_map(|(i, d)| {
                if normalized_axes.contains(&(i as i32)) {
                    None
                } else {
                    Some(d.clone())
                }
            })
            .collect();
        let output_shape = Shape::new(output_dims);

        tracing::debug!(
            "Squeeze: axes = {:?}, input shape = {:?}, output shape = {:?}",
            normalized_axes, input_shape, output_shape
        );

        // Constant folding: if input is a constant, create squeezed constant
        if let NodeOp::Constant { data: const_data } = &input_node.op {
            // For constants, data stays the same - only shape changes
            let folded_data = const_data.clone();
            tracing::debug!("Squeeze: constant folding succeeded");
            let result = builder.constant(folded_data, output_shape);
            return Ok(vec![result]);
        }

        // Non-constant path: add squeeze node
        let result = builder.graph_mut().add_op(
            NodeOp::Squeeze { axes: axes.clone() },
            output_shape,
            input_dtype,
        );
        builder.graph_mut().connect(data, result);

        Ok(vec![result])
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
            name: "squeeze_test".to_string(),
            op_type: "Squeeze".to_string(),
            ..Default::default()
        }
    }

    fn make_node_with_axes(axes: Vec<i64>) -> NodeProto {
        NodeProto {
            name: "squeeze_test".to_string(),
            op_type: "Squeeze".to_string(),
            attribute: vec![AttributeProto {
                name: "axes".to_string(),
                ints: axes,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    // ===== Valid Input Tests =====

    #[test]
    fn test_squeeze_all_ones() {
        let translator = SqueezeTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[1, 3, 1, 4]), DType::F32);

        // No axes specified: squeeze all 1-sized dims
        let result = translator.translate(&make_node(), &[x], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);

        let node = builder.graph().node(outputs[0]).unwrap();
        // [1, 3, 1, 4] -> [3, 4]
        assert_eq!(node.shape.rank(), 2);
    }

    #[test]
    fn test_squeeze_specific_axis() {
        let translator = SqueezeTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[1, 3, 1, 4]), DType::F32);

        // Only squeeze axis 0
        let result = translator.translate(&make_node_with_axes(vec![0]), &[x], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();

        let node = builder.graph().node(outputs[0]).unwrap();
        // [1, 3, 1, 4] -> [3, 1, 4]
        assert_eq!(node.shape.rank(), 3);
        assert_eq!(node.shape.dims[0], Dim::Static(3));
    }

    #[test]
    fn test_squeeze_multiple_axes() {
        let translator = SqueezeTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[1, 3, 1, 4]), DType::F32);

        // Squeeze axes 0 and 2
        let result = translator.translate(&make_node_with_axes(vec![0, 2]), &[x], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();

        let node = builder.graph().node(outputs[0]).unwrap();
        // [1, 3, 1, 4] -> [3, 4]
        assert_eq!(node.shape.rank(), 2);
    }

    #[test]
    fn test_squeeze_negative_axis() {
        let translator = SqueezeTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[1, 3, 1]), DType::F32);

        // axis=-1 is the last axis
        let result = translator.translate(&make_node_with_axes(vec![-1]), &[x], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();

        let node = builder.graph().node(outputs[0]).unwrap();
        // [1, 3, 1] -> [1, 3]
        assert_eq!(node.shape.rank(), 2);
    }

    #[test]
    fn test_squeeze_opset13_axes_input() {
        let translator = SqueezeTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[1, 3, 1, 4]), DType::F32);
        let axes = builder.constant(
            ConstantData::I64(vec![0, 2]),
            Shape::static_shape(&[2]),
        );

        let result = translator.translate(&make_node(), &[x, axes], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();

        let node = builder.graph().node(outputs[0]).unwrap();
        // [1, 3, 1, 4] -> [3, 4]
        assert_eq!(node.shape.rank(), 2);
    }

    #[test]
    fn test_squeeze_constant_folding() {
        let translator = SqueezeTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.constant(
            ConstantData::F32(vec![1.0, 2.0, 3.0]),
            Shape::static_shape(&[1, 3]),
        );

        let result = translator.translate(&make_node_with_axes(vec![0]), &[x], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();

        let node = builder.graph().node(outputs[0]).unwrap();
        // Should be constant folded
        if let NodeOp::Constant { data } = &node.op {
            if let ConstantData::F32(values) = data {
                assert_eq!(values.as_slice(), &[1.0, 2.0, 3.0]);
            } else {
                panic!("Expected F32 data");
            }
        } else {
            panic!("Expected Constant node");
        }
        // Shape should be [3]
        assert_eq!(node.shape.rank(), 1);
        assert_eq!(node.shape.dims[0], Dim::Static(3));
    }

    #[test]
    fn test_squeeze_no_change() {
        let translator = SqueezeTranslator;
        let mut builder = GraphBuilder::new();

        // No dimensions of size 1
        let x = builder.input("x", Shape::static_shape(&[2, 3, 4]), DType::F32);

        let result = translator.translate(&make_node(), &[x], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();

        let node = builder.graph().node(outputs[0]).unwrap();
        // Shape unchanged: [2, 3, 4]
        assert_eq!(node.shape.rank(), 3);
    }

    // ===== Invalid Input Tests =====

    #[test]
    fn test_squeeze_no_inputs() {
        let translator = SqueezeTranslator;
        let err = translator.input_requirement().validate(0, "Squeeze");
        assert!(err.is_err());
    }

    #[test]
    fn test_squeeze_too_many_inputs() {
        let translator = SqueezeTranslator;
        let err = translator.input_requirement().validate(3, "Squeeze");
        assert!(err.is_err());
    }

    #[test]
    fn test_squeeze_axis_not_size_1() {
        let translator = SqueezeTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[2, 3, 4]), DType::F32);

        // Try to squeeze axis with size > 1
        let result = translator.translate(&make_node_with_axes(vec![1]), &[x], &mut builder);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("must be 1"));
    }

    #[test]
    fn test_squeeze_axis_out_of_bounds() {
        let translator = SqueezeTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[1, 3, 1]), DType::F32);

        let result = translator.translate(&make_node_with_axes(vec![5]), &[x], &mut builder);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("out of bounds"));
    }

    // ===== Trait Method Tests =====

    #[test]
    fn test_op_type() {
        let translator = SqueezeTranslator;
        assert_eq!(translator.onnx_op_type(), "Squeeze");
    }

    #[test]
    fn test_input_requirement() {
        let translator = SqueezeTranslator;
        let req = translator.input_requirement();
        assert!(matches!(req, InputRequirement::Range(1, 2)));
        assert!(!req.accepts_zero());
    }

    #[test]
    fn test_supports_constant_folding() {
        let translator = SqueezeTranslator;
        assert!(translator.supports_constant_folding());
    }
}
