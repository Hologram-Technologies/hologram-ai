//! Slice operation translator.

use crate::proto::NodeProto;
use crate::translators::{InputRequirement, OnnxAttributes, OnnxTranslator, TranslationError};
use hologram::ir::{ConstantData, GraphBuilder, NodeIndex, NodeOp};

/// Translator for ONNX Slice operation.
///
/// Slice extracts a slice from the input tensor along multiple axes.
///
/// # ONNX Specification
///
/// - Opset < 10: starts, ends, axes are attributes
/// - Opset >= 10: starts, ends, axes, steps are inputs
///
/// - Inputs (opset >= 10): data, starts, ends, [axes], [steps]
/// - Attributes (opset < 10): starts, ends, [axes]
#[derive(Debug, Default)]
pub struct SliceTranslator;

impl OnnxTranslator for SliceTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "Slice"
    }

    fn input_requirement(&self) -> InputRequirement {
        // 1 input for opset < 10 (attributes), 3-5 inputs for opset >= 10
        InputRequirement::Range(1, 5)
    }

    fn translate(
        &self,
        node: &NodeProto,
        inputs: &[NodeIndex],
        builder: &mut GraphBuilder,
    ) -> Result<Vec<NodeIndex>, TranslationError> {
        // Try attribute-based first (older opset)
        let starts_attr = node.get_ints("starts");
        let ends_attr = node.get_ints("ends");
        let axes_attr = node.get_ints("axes");

        let (starts, ends, axes) = if let (Some(starts_vals), Some(ends_vals)) =
            (starts_attr, ends_attr)
        {
            // Attribute-based (older opset < 10)
            if inputs.len() != 1 {
                return Err(TranslationError::IrBuilder(format!(
                    "Slice (opset < 10) requires 1 input, got {}",
                    inputs.len()
                )));
            }

            let starts: Vec<i64> = starts_vals.to_vec();
            let ends: Vec<i64> = ends_vals.to_vec();
            let axes: Vec<i64> = axes_attr.map(|a| a.to_vec()).unwrap_or_default();

            (starts, ends, axes)
        } else {
            // Input-based (newer opset >= 10)
            if inputs.len() < 3 {
                return Err(TranslationError::IrBuilder(format!(
                    "Slice requires at least 3 inputs (data, starts, ends), got {}",
                    inputs.len()
                )));
            }

            // Check if all slice parameters are constants
            let starts_node = builder.graph().node(inputs[1]).ok_or_else(|| {
                TranslationError::IrBuilder("Slice: starts node not found".to_string())
            })?;
            let ends_node = builder.graph().node(inputs[2]).ok_or_else(|| {
                TranslationError::IrBuilder("Slice: ends node not found".to_string())
            })?;

            let starts_is_constant = matches!(starts_node.op, NodeOp::Constant { .. });
            let ends_is_constant = matches!(ends_node.op, NodeOp::Constant { .. });

            // Check axes if provided
            let axes_is_constant = if inputs.len() > 3 {
                let axes_node = builder.graph().node(inputs[3]).ok_or_else(|| {
                    TranslationError::IrBuilder("Slice: axes node not found".to_string())
                })?;
                matches!(axes_node.op, NodeOp::Constant { .. })
            } else {
                true
            };

            // Check steps if provided
            let steps_is_constant = if inputs.len() > 4 {
                let steps_node = builder.graph().node(inputs[4]).ok_or_else(|| {
                    TranslationError::IrBuilder("Slice: steps node not found".to_string())
                })?;
                matches!(steps_node.op, NodeOp::Constant { .. })
            } else {
                true
            };

            // If any input is non-constant, use dynamic slice
            if !starts_is_constant || !ends_is_constant || !axes_is_constant || !steps_is_constant {
                let axes = if inputs.len() > 3 {
                    Some(inputs[3])
                } else {
                    None
                };
                let steps = if inputs.len() > 4 {
                    Some(inputs[4])
                } else {
                    None
                };

                let result = builder
                    .slice_dynamic(inputs[0], inputs[1], inputs[2], axes, steps)
                    .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;
                return Ok(vec![result]);
            }

            // Extract constant values for static slice path
            let starts_vals = Self::extract_i64_constant(starts_node)?;
            let ends_vals = Self::extract_i64_constant(ends_node)?;

            let axes_vals = if inputs.len() > 3 {
                let axes_node = builder.graph().node(inputs[3]).ok_or_else(|| {
                    TranslationError::IrBuilder("Slice: axes node not found".to_string())
                })?;
                Self::extract_i64_constant(axes_node)?
            } else {
                vec![]
            };

            (starts_vals, ends_vals, axes_vals)
        };

        let data = inputs[0];

        // Validate lengths
        if starts.len() != ends.len() {
            return Err(TranslationError::IrBuilder(format!(
                "Slice starts and ends must have same length, got {} and {}",
                starts.len(),
                ends.len()
            )));
        }

        // If axes not provided, default to [0, 1, 2, ..., len(starts)-1]
        let axes_i32: Vec<i32> = if axes.is_empty() {
            (0..starts.len() as i32).collect()
        } else {
            if axes.len() != starts.len() {
                return Err(TranslationError::IrBuilder(format!(
                    "Slice axes must have same length as starts, got {} and {}",
                    axes.len(),
                    starts.len()
                )));
            }
            axes.iter().map(|&a| a as i32).collect()
        };

        // Static slice path
        let result = builder
            .unary(
                NodeOp::Slice {
                    starts,
                    ends,
                    axes: axes_i32,
                },
                data,
            )
            .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;

        Ok(vec![result])
    }
}

impl SliceTranslator {
    /// Extract i64 values from a constant node.
    fn extract_i64_constant(node: &hologram::ir::Node) -> Result<Vec<i64>, TranslationError> {
        if let NodeOp::Constant { data } = &node.op {
            match data {
                ConstantData::I64(values) => Ok(values.clone()),
                ConstantData::I32(values) => Ok(values.iter().map(|&v| v as i64).collect()),
                _ => Err(TranslationError::IrBuilder(
                    "Slice: expected int32 or int64 tensor".to_string(),
                )),
            }
        } else {
            Err(TranslationError::IrBuilder(
                "Slice: expected constant input".to_string(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::AttributeProto;
    use hologram::ir::{DType, Shape};

    fn make_node() -> NodeProto {
        NodeProto {
            name: "slice_test".to_string(),
            op_type: "Slice".to_string(),
            ..Default::default()
        }
    }

    fn make_node_with_attrs(starts: Vec<i64>, ends: Vec<i64>, axes: Option<Vec<i64>>) -> NodeProto {
        let mut attrs = vec![
            AttributeProto {
                name: "starts".to_string(),
                ints: starts,
                ..Default::default()
            },
            AttributeProto {
                name: "ends".to_string(),
                ints: ends,
                ..Default::default()
            },
        ];
        if let Some(ax) = axes {
            attrs.push(AttributeProto {
                name: "axes".to_string(),
                ints: ax,
                ..Default::default()
            });
        }
        NodeProto {
            name: "slice_test".to_string(),
            op_type: "Slice".to_string(),
            attribute: attrs,
            ..Default::default()
        }
    }

    #[test]
    fn test_slice_with_attributes() {
        let translator = SliceTranslator;
        let mut builder = GraphBuilder::new();
        let data = builder.input("data", Shape::static_shape(&[10, 20, 30]), DType::F32);

        let node = make_node_with_attrs(vec![0, 5, 10], vec![5, 15, 25], Some(vec![0, 1, 2]));
        let result = translator.translate(&node, &[data], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_slice_default_axes() {
        let translator = SliceTranslator;
        let mut builder = GraphBuilder::new();
        let data = builder.input("data", Shape::static_shape(&[10, 20, 30]), DType::F32);

        let node = make_node_with_attrs(vec![0, 5, 10], vec![5, 15, 25], None);
        let result = translator.translate(&node, &[data], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_slice_partial_axes() {
        let translator = SliceTranslator;
        let mut builder = GraphBuilder::new();
        let data = builder.input("data", Shape::static_shape(&[10, 20, 30]), DType::F32);

        let node = make_node_with_attrs(vec![5], vec![15], Some(vec![1]));
        let result = translator.translate(&node, &[data], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_slice_constant_inputs() {
        let translator = SliceTranslator;
        let mut builder = GraphBuilder::new();
        let data = builder.input("data", Shape::static_shape(&[10, 20, 30]), DType::F32);
        let starts = builder.constant(ConstantData::I64(vec![0, 5]), Shape::static_shape(&[2]));
        let ends = builder.constant(ConstantData::I64(vec![5, 15]), Shape::static_shape(&[2]));

        let result = translator.translate(&make_node(), &[data, starts, ends], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_slice_dynamic_inputs() {
        let translator = SliceTranslator;
        let mut builder = GraphBuilder::new();
        let data = builder.input("data", Shape::static_shape(&[10, 20, 30]), DType::F32);
        let starts = builder.input("starts", Shape::static_shape(&[3]), DType::I64);
        let ends = builder.input("ends", Shape::static_shape(&[3]), DType::I64);

        let result = translator.translate(&make_node(), &[data, starts, ends], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_slice_mismatched_lengths() {
        let translator = SliceTranslator;
        let mut builder = GraphBuilder::new();
        let data = builder.input("data", Shape::static_shape(&[10, 20, 30]), DType::F32);

        let node = make_node_with_attrs(vec![0, 5], vec![5, 15, 25], None);
        let result = translator.translate(&node, &[data], &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_slice_input_validation() {
        let translator = SliceTranslator;

        // 0 inputs should fail
        let err = translator.input_requirement().validate(0, "Slice");
        assert!(err.is_err());

        // 6 inputs should fail
        let err = translator.input_requirement().validate(6, "Slice");
        assert!(err.is_err());

        // 1-5 inputs should pass
        assert!(translator.input_requirement().validate(1, "Slice").is_ok());
        assert!(translator.input_requirement().validate(3, "Slice").is_ok());
        assert!(translator.input_requirement().validate(5, "Slice").is_ok());
    }
}
