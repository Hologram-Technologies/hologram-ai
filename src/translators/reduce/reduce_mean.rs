//! ReduceMean operation translator.

use hologram::ir::{GraphBuilder, NodeIndex, NodeOp};
use crate::proto::NodeProto;
use crate::translators::{OnnxTranslator, OnnxAttributes, InputRequirement, TranslationError};

/// Translator for ONNX ReduceMean operation.
///
/// ReduceMean computes the mean of elements along the specified axes.
///
/// # Attributes
///
/// - `axes`: int array (default: reduce all axes) - Axes along which to reduce.
///   In ONNX opset 18+, axes can also be provided as a second input.
/// - `keepdims`: int (default: 1) - If 1, reduced dimensions are retained with size 1.
/// - `noop_with_empty_axes`: int (default: 0) - If 1 and axes is empty, return input unchanged.
///
/// # Inputs
///
/// - `data` (required): Input tensor to reduce
/// - `axes` (optional, opset 18+): Axes along which to reduce
///
/// # Outputs
///
/// - `reduced`: Reduced tensor containing mean values
#[derive(Debug, Default)]
pub struct ReduceMeanTranslator;

impl OnnxTranslator for ReduceMeanTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "ReduceMean"
    }

    fn input_requirement(&self) -> InputRequirement {
        // data is required, axes is optional as second input (opset 18+)
        InputRequirement::Range(1, 2)
    }

    fn translate(
        &self,
        node: &NodeProto,
        inputs: &[NodeIndex],
        builder: &mut GraphBuilder,
    ) -> Result<Vec<NodeIndex>, TranslationError> {
        let data = inputs[0];

        // Get axes from attribute (fallback to empty for reduce-all behavior)
        let axes_attr = node.get_ints("axes");
        let axes: Vec<i32> = match axes_attr {
            Some(ints) => ints.iter().map(|&x| x as i32).collect(),
            None => vec![], // Empty means reduce all axes
        };

        // Get keepdims attribute (default: 1 = true)
        let keepdims = node.get_int_or("keepdims", 1) != 0;

        // Get noop_with_empty_axes attribute (default: 0 = false)
        let noop_with_empty_axes = node.get_int_or("noop_with_empty_axes", 0) != 0;

        // If axes is empty and noop_with_empty_axes is true, return input unchanged
        if axes.is_empty() && noop_with_empty_axes {
            return Ok(vec![data]);
        }

        let result = builder
            .unary(NodeOp::ReduceMean { axes, keepdims }, data)
            .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;

        Ok(vec![result])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram::ir::{DType, Shape};
    use crate::proto::AttributeProto;

    fn make_node() -> NodeProto {
        NodeProto {
            name: "reduce_mean_test".to_string(),
            op_type: "ReduceMean".to_string(),
            ..Default::default()
        }
    }

    fn make_node_with_axes(axes: Vec<i64>) -> NodeProto {
        NodeProto {
            name: "reduce_mean_test".to_string(),
            op_type: "ReduceMean".to_string(),
            attribute: vec![AttributeProto {
                name: "axes".to_string(),
                ints: axes,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn make_node_with_attrs(axes: Vec<i64>, keepdims: i64) -> NodeProto {
        NodeProto {
            name: "reduce_mean_test".to_string(),
            op_type: "ReduceMean".to_string(),
            attribute: vec![
                AttributeProto {
                    name: "axes".to_string(),
                    ints: axes,
                    ..Default::default()
                },
                AttributeProto {
                    name: "keepdims".to_string(),
                    i: keepdims,
                    ..Default::default()
                },
            ],
            ..Default::default()
        }
    }

    fn make_node_with_noop(axes: Vec<i64>, noop_with_empty_axes: i64) -> NodeProto {
        NodeProto {
            name: "reduce_mean_test".to_string(),
            op_type: "ReduceMean".to_string(),
            attribute: vec![
                AttributeProto {
                    name: "axes".to_string(),
                    ints: axes,
                    ..Default::default()
                },
                AttributeProto {
                    name: "noop_with_empty_axes".to_string(),
                    i: noop_with_empty_axes,
                    ..Default::default()
                },
            ],
            ..Default::default()
        }
    }

    // ===== Single Axis Reduction Tests =====

    #[test]
    fn test_reduce_mean_single_axis() {
        let translator = ReduceMeanTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[4, 5, 6]), DType::F32);

        let result = translator.translate(&make_node_with_axes(vec![1]), &[x], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_reduce_mean_negative_axis() {
        let translator = ReduceMeanTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[2, 3, 4]), DType::F32);

        let result = translator.translate(&make_node_with_axes(vec![-1]), &[x], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_reduce_mean_first_axis() {
        let translator = ReduceMeanTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[10, 20]), DType::F32);

        let result = translator.translate(&make_node_with_axes(vec![0]), &[x], &mut builder);
        assert!(result.is_ok());
    }

    // ===== Multiple Axes Reduction Tests =====

    #[test]
    fn test_reduce_mean_multiple_axes() {
        let translator = ReduceMeanTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[4, 5, 6]), DType::F32);

        let result = translator.translate(&make_node_with_axes(vec![0, 2]), &[x], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_reduce_mean_all_axes_explicit() {
        let translator = ReduceMeanTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);

        let result = translator.translate(&make_node_with_axes(vec![0, 1]), &[x], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_reduce_mean_non_contiguous_axes() {
        let translator = ReduceMeanTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[2, 3, 4, 5]), DType::F32);

        let result = translator.translate(&make_node_with_axes(vec![0, 2]), &[x], &mut builder);
        assert!(result.is_ok());
    }

    // ===== keepdims Tests =====

    #[test]
    fn test_reduce_mean_keepdims_true() {
        let translator = ReduceMeanTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[2, 3, 4]), DType::F32);

        let result = translator.translate(&make_node_with_attrs(vec![1], 1), &[x], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_reduce_mean_keepdims_false() {
        let translator = ReduceMeanTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[2, 3, 4]), DType::F32);

        let result = translator.translate(&make_node_with_attrs(vec![1], 0), &[x], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_reduce_mean_default_keepdims() {
        let translator = ReduceMeanTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[5, 5]), DType::F32);

        // Default keepdims should be 1 (true)
        let result = translator.translate(&make_node_with_axes(vec![0]), &[x], &mut builder);
        assert!(result.is_ok());
    }

    // ===== All Axes Reduction Tests =====

    #[test]
    fn test_reduce_mean_no_axes_reduce_all() {
        let translator = ReduceMeanTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[2, 3, 4]), DType::F32);

        // Empty axes means reduce all dimensions
        let result = translator.translate(&make_node(), &[x], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_reduce_mean_noop_with_empty_axes() {
        let translator = ReduceMeanTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[2, 3]), DType::F32);

        // With noop_with_empty_axes=1, should return input unchanged
        let result = translator.translate(&make_node_with_noop(vec![], 1), &[x], &mut builder);
        assert!(result.is_ok());
        // Result should be the same node as input
        assert_eq!(result.unwrap()[0], x);
    }

    // ===== Different Tensor Shapes =====

    #[test]
    fn test_reduce_mean_1d() {
        let translator = ReduceMeanTranslator;
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::static_shape(&[50]), DType::F32);

        let result = translator.translate(&make_node_with_axes(vec![0]), &[x], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_reduce_mean_3d_batch_norm_like() {
        let translator = ReduceMeanTranslator;
        let mut builder = GraphBuilder::new();
        // BatchNorm-style: reduce over batch and spatial dimensions
        let x = builder.input("x", Shape::static_shape(&[32, 64, 224]), DType::F32);

        let result = translator.translate(&make_node_with_axes(vec![0, 2]), &[x], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_reduce_mean_4d_global_avg_pool() {
        let translator = ReduceMeanTranslator;
        let mut builder = GraphBuilder::new();
        // Global average pooling: reduce over spatial dimensions
        let x = builder.input("x", Shape::static_shape(&[1, 512, 7, 7]), DType::F32);

        let result = translator.translate(&make_node_with_axes(vec![2, 3]), &[x], &mut builder);
        assert!(result.is_ok());
    }

    // ===== Invalid Input Tests =====

    #[test]
    fn test_reduce_mean_no_inputs() {
        let translator = ReduceMeanTranslator;
        let err = translator.input_requirement().validate(0, "ReduceMean");
        assert!(err.is_err());
        assert!(matches!(
            err.unwrap_err(),
            TranslationError::InputCountOutOfRange {
                min: 1,
                max: 2,
                got: 0,
                ..
            }
        ));
    }

    #[test]
    fn test_reduce_mean_too_many_inputs() {
        let translator = ReduceMeanTranslator;
        let err = translator.input_requirement().validate(3, "ReduceMean");
        assert!(err.is_err());
    }

    #[test]
    fn test_reduce_mean_input_requirement() {
        let translator = ReduceMeanTranslator;
        assert_eq!(
            translator.input_requirement(),
            InputRequirement::Range(1, 2)
        );
    }
}
