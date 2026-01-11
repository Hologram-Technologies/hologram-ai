//! Unsqueeze operation translator.

use crate::proto::NodeProto;
use crate::translators::{InputRequirement, OnnxAttributes, OnnxTranslator, TranslationError};
use hologram::ir::{ConstantData, Dim, GraphBuilder, NodeIndex, NodeOp, Shape};

/// Translator for ONNX Unsqueeze operation.
///
/// Unsqueeze inserts dimensions of size 1 at specified axes.
///
/// # Inputs
/// - data: Input tensor
/// - axes (opset 13+): 1D tensor specifying where to insert new dimensions
///
/// # Attributes
/// - axes (opset < 13): List of axes where to insert new dimensions
///
/// # Constant Folding
/// If the input is a constant, the unsqueeze is performed at compile time.
#[derive(Debug, Default)]
pub struct UnsqueezeTranslator;

impl OnnxTranslator for UnsqueezeTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "Unsqueeze"
    }

    fn input_requirement(&self) -> InputRequirement {
        // 1 input with axes attribute (opset < 13) or 2 inputs (opset 13+)
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
            TranslationError::IrBuilder("Unsqueeze: input node not found".to_string())
        })?;
        let input_shape = input_node.shape.clone();
        let input_dtype = input_node.dtype;

        // Get axes - either from second input (opset 13+) or from attribute
        let axes: Vec<i32> = if inputs.len() >= 2 {
            // Opset 13+: axes from second input (must be constant)
            let axes_node = builder.graph().node(inputs[1]).ok_or_else(|| {
                TranslationError::IrBuilder("Unsqueeze: axes input not found".to_string())
            })?;

            if let NodeOp::Constant {
                data: constant_data,
            } = &axes_node.op
            {
                match constant_data {
                    ConstantData::I64(values) => values.iter().map(|&v| v as i32).collect(),
                    ConstantData::I32(values) => values.clone(),
                    _ => {
                        return Err(TranslationError::ShapeInference(
                            "Unsqueeze: axes must be int32 or int64".to_string(),
                        ));
                    }
                }
            } else {
                return Err(TranslationError::ShapeInference(
                    "Unsqueeze: axes input must be a constant".to_string(),
                ));
            }
        } else if let Some(axes_attr) = node.get_ints("axes") {
            // Opset < 13: axes from attribute
            axes_attr.iter().map(|&v| v as i32).collect()
        } else {
            return Err(TranslationError::missing_attribute("Unsqueeze", "axes"));
        };

        // Calculate output rank
        let input_rank = input_shape.rank();
        let output_rank = input_rank + axes.len();

        // Normalize negative axes to positive
        let mut normalized_axes: Vec<i32> = axes
            .iter()
            .map(|&axis| {
                if axis < 0 {
                    output_rank as i32 + axis
                } else {
                    axis
                }
            })
            .collect();
        normalized_axes.sort_unstable();

        // Validate axes
        for &axis in &normalized_axes {
            if axis < 0 || axis >= output_rank as i32 {
                return Err(TranslationError::invalid_attribute(
                    "axes",
                    format!(
                        "axis {} is out of bounds for output rank {}",
                        axis, output_rank
                    ),
                ));
            }
        }

        // Check for duplicates
        for i in 1..normalized_axes.len() {
            if normalized_axes[i] == normalized_axes[i - 1] {
                return Err(TranslationError::invalid_attribute(
                    "axes",
                    format!("duplicate axis {} in axes", normalized_axes[i]),
                ));
            }
        }

        // Build output shape by inserting dimensions of size 1
        let input_dims = &input_shape.dims;
        let mut output_dims = Vec::with_capacity(output_rank);
        let mut input_idx = 0;
        let mut axis_idx = 0;

        for out_idx in 0..output_rank {
            if axis_idx < normalized_axes.len() && normalized_axes[axis_idx] == out_idx as i32 {
                // Insert dimension of size 1
                output_dims.push(Dim::Static(1));
                axis_idx += 1;
            } else {
                // Copy from input
                output_dims.push(input_dims[input_idx].clone());
                input_idx += 1;
            }
        }

        let output_shape = Shape::new(output_dims);

        tracing::debug!(
            "Unsqueeze: axes = {:?}, normalized_axes = {:?}, input shape = {:?}, output shape = {:?}",
            axes,
            normalized_axes,
            input_shape,
            output_shape
        );

        // Constant folding: if input is a constant, create unsqueezed constant
        if let NodeOp::Constant { data: const_data } = &input_node.op {
            // For constants, data stays the same - only shape changes
            let folded_data = const_data.clone();
            tracing::debug!("Unsqueeze: constant folding succeeded");
            let result = builder.constant(folded_data, output_shape);
            return Ok(vec![result]);
        }

        // Non-constant path: add unsqueeze node
        let result = builder.graph_mut().add_op(
            NodeOp::Unsqueeze { axes: axes.clone() },
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
    use crate::proto::AttributeProto;
    use hologram::ir::DType;

    fn make_node() -> NodeProto {
        NodeProto {
            name: "unsqueeze_test".to_string(),
            op_type: "Unsqueeze".to_string(),
            ..Default::default()
        }
    }

    fn make_node_with_axes(axes: Vec<i64>) -> NodeProto {
        NodeProto {
            name: "unsqueeze_test".to_string(),
            op_type: "Unsqueeze".to_string(),
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
    fn test_unsqueeze_single_axis() {
        let translator = UnsqueezeTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);

        let result = translator.translate(&make_node_with_axes(vec![0]), &[x], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();

        let node = builder.graph().node(outputs[0]).unwrap();
        // [2, 3] -> [1, 2, 3]
        assert_eq!(node.shape.rank(), 3);
        assert_eq!(node.shape.dims[0], Dim::Static(1));
        assert_eq!(node.shape.dims[1], Dim::Static(2));
        assert_eq!(node.shape.dims[2], Dim::Static(3));
    }

    #[test]
    fn test_unsqueeze_multiple_axes() {
        let translator = UnsqueezeTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);

        let result = translator.translate(&make_node_with_axes(vec![0, 3]), &[x], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();

        let node = builder.graph().node(outputs[0]).unwrap();
        // [2, 3] -> [1, 2, 3, 1]
        assert_eq!(node.shape.rank(), 4);
        assert_eq!(node.shape.dims[0], Dim::Static(1));
        assert_eq!(node.shape.dims[1], Dim::Static(2));
        assert_eq!(node.shape.dims[2], Dim::Static(3));
        assert_eq!(node.shape.dims[3], Dim::Static(1));
    }

    #[test]
    fn test_unsqueeze_negative_axis() {
        let translator = UnsqueezeTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);

        // axis=-1 in output of rank 3 means axis 2
        let result = translator.translate(&make_node_with_axes(vec![-1]), &[x], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();

        let node = builder.graph().node(outputs[0]).unwrap();
        // [2, 3] -> [2, 3, 1]
        assert_eq!(node.shape.rank(), 3);
        assert_eq!(node.shape.dims[0], Dim::Static(2));
        assert_eq!(node.shape.dims[1], Dim::Static(3));
        assert_eq!(node.shape.dims[2], Dim::Static(1));
    }

    #[test]
    fn test_unsqueeze_opset13_axes_input() {
        let translator = UnsqueezeTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);
        let axes = builder.constant(ConstantData::I64(vec![1]), Shape::static_shape(&[1]));

        let result = translator.translate(&make_node(), &[x, axes], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();

        let node = builder.graph().node(outputs[0]).unwrap();
        // [2, 3] -> [2, 1, 3]
        assert_eq!(node.shape.rank(), 3);
        assert_eq!(node.shape.dims[0], Dim::Static(2));
        assert_eq!(node.shape.dims[1], Dim::Static(1));
        assert_eq!(node.shape.dims[2], Dim::Static(3));
    }

    #[test]
    fn test_unsqueeze_constant_folding() {
        let translator = UnsqueezeTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.constant(
            ConstantData::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            Shape::static_shape(&[2, 3]),
        );

        let result = translator.translate(&make_node_with_axes(vec![0]), &[x], &mut builder);
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
        // Shape should be [1, 2, 3]
        assert_eq!(node.shape.rank(), 3);
    }

    #[test]
    fn test_unsqueeze_scalar_to_1d() {
        let translator = UnsqueezeTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.constant(ConstantData::F32(vec![42.0]), Shape::static_shape(&[]));

        let result = translator.translate(&make_node_with_axes(vec![0]), &[x], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();

        let node = builder.graph().node(outputs[0]).unwrap();
        // [] -> [1]
        assert_eq!(node.shape.rank(), 1);
        assert_eq!(node.shape.dims[0], Dim::Static(1));
    }

    #[test]
    fn test_unsqueeze_middle_axis() {
        let translator = UnsqueezeTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[2, 3, 4]), DType::F32);

        let result = translator.translate(&make_node_with_axes(vec![2]), &[x], &mut builder);
        assert!(result.is_ok());
        let outputs = result.unwrap();

        let node = builder.graph().node(outputs[0]).unwrap();
        // [2, 3, 4] -> [2, 3, 1, 4]
        assert_eq!(node.shape.rank(), 4);
        assert_eq!(node.shape.dims[0], Dim::Static(2));
        assert_eq!(node.shape.dims[1], Dim::Static(3));
        assert_eq!(node.shape.dims[2], Dim::Static(1));
        assert_eq!(node.shape.dims[3], Dim::Static(4));
    }

    // ===== Invalid Input Tests =====

    #[test]
    fn test_unsqueeze_no_inputs() {
        let translator = UnsqueezeTranslator;
        let err = translator.input_requirement().validate(0, "Unsqueeze");
        assert!(err.is_err());
    }

    #[test]
    fn test_unsqueeze_too_many_inputs() {
        let translator = UnsqueezeTranslator;
        let err = translator.input_requirement().validate(3, "Unsqueeze");
        assert!(err.is_err());
    }

    #[test]
    fn test_unsqueeze_missing_axes() {
        let translator = UnsqueezeTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);

        // No axes attribute and no second input
        let result = translator.translate(&make_node(), &[x], &mut builder);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("axes"));
    }

    #[test]
    fn test_unsqueeze_axis_out_of_bounds() {
        let translator = UnsqueezeTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);

        // Output would have rank 3, so axis 5 is out of bounds
        let result = translator.translate(&make_node_with_axes(vec![5]), &[x], &mut builder);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("out of bounds"));
    }

    #[test]
    fn test_unsqueeze_duplicate_axis() {
        let translator = UnsqueezeTranslator;
        let mut builder = GraphBuilder::new();

        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);

        let result = translator.translate(&make_node_with_axes(vec![0, 0]), &[x], &mut builder);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("duplicate"));
    }

    // ===== Trait Method Tests =====

    #[test]
    fn test_op_type() {
        let translator = UnsqueezeTranslator;
        assert_eq!(translator.onnx_op_type(), "Unsqueeze");
    }

    #[test]
    fn test_input_requirement() {
        let translator = UnsqueezeTranslator;
        let req = translator.input_requirement();
        assert!(matches!(req, InputRequirement::Range(1, 2)));
        assert!(!req.accepts_zero());
    }

    #[test]
    fn test_supports_constant_folding() {
        let translator = UnsqueezeTranslator;
        assert!(translator.supports_constant_folding());
    }
}
