//! AveragePool operation translator.

use crate::proto::NodeProto;
use crate::translators::{InputRequirement, OnnxAttributes, OnnxTranslator, TranslationError};
use hologram::ir::{GraphBuilder, NodeIndex, NodeOp, Padding};

/// Translator for ONNX AveragePool operation.
///
/// AveragePool performs 2D average pooling over the input tensor.
///
/// # ONNX Specification
///
/// - Inputs: X
/// - Attributes:
///   - kernel_shape (required): [kH, kW]
///   - strides (default: [1, 1])
///   - pads (default: [0, 0, 0, 0])
///   - count_include_pad (default: 0)
/// - Outputs: Y
#[derive(Debug, Default)]
pub struct AveragePoolTranslator;

impl OnnxTranslator for AveragePoolTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "AveragePool"
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
        let input = inputs[0];

        // Parse kernel_shape
        let kernel_shape = node
            .get_ints("kernel_shape")
            .ok_or_else(|| TranslationError::missing_attribute("AveragePool", "kernel_shape"))?;
        if kernel_shape.len() != 2 {
            return Err(TranslationError::invalid_attribute(
                "kernel_shape",
                format!("must be 2D, got {:?}", kernel_shape),
            ));
        }
        let kernel = (kernel_shape[0] as usize, kernel_shape[1] as usize);

        // Parse strides (default [1, 1])
        let strides = node.get_ints_or("strides", &[1, 1]);
        if strides.len() != 2 {
            return Err(TranslationError::invalid_attribute(
                "strides",
                format!("must be 2D, got {:?}", strides),
            ));
        }
        let stride = (strides[0] as usize, strides[1] as usize);

        // Parse padding
        let pads = node.get_ints_or("pads", &[]);
        let padding = Self::parse_padding(pads)?;

        // Create average pool node
        let result = builder
            .unary(
                NodeOp::AvgPool2d {
                    kernel,
                    stride,
                    padding,
                },
                input,
            )
            .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;

        Ok(vec![result])
    }
}

impl AveragePoolTranslator {
    fn parse_padding(pads: &[i64]) -> Result<Padding, TranslationError> {
        if pads.is_empty() {
            return Ok(Padding::Valid);
        }

        if pads.len() == 4 {
            let top = pads[0] as usize;
            let left = pads[1] as usize;
            let bottom = pads[2] as usize;
            let right = pads[3] as usize;

            if top == 0 && left == 0 && bottom == 0 && right == 0 {
                Ok(Padding::Valid)
            } else {
                Ok(Padding::Explicit {
                    top,
                    bottom,
                    left,
                    right,
                })
            }
        } else {
            Err(TranslationError::invalid_attribute(
                "pads",
                format!("expected 4 elements, got {}", pads.len()),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::AttributeProto;
    use hologram::ir::{DType, Shape};

    fn make_node_with_attrs(
        kernel: Vec<i64>,
        strides: Option<Vec<i64>>,
        pads: Option<Vec<i64>>,
    ) -> NodeProto {
        let mut attrs = vec![AttributeProto {
            name: "kernel_shape".to_string(),
            ints: kernel,
            ..Default::default()
        }];

        if let Some(s) = strides {
            attrs.push(AttributeProto {
                name: "strides".to_string(),
                ints: s,
                ..Default::default()
            });
        }

        if let Some(p) = pads {
            attrs.push(AttributeProto {
                name: "pads".to_string(),
                ints: p,
                ..Default::default()
            });
        }

        NodeProto {
            name: "avg_pool_test".to_string(),
            op_type: "AveragePool".to_string(),
            attribute: attrs,
            ..Default::default()
        }
    }

    #[test]
    fn test_avg_pool_basic() {
        let translator = AveragePoolTranslator;
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[1, 64, 32, 32]), DType::F32);

        let node = make_node_with_attrs(vec![2, 2], Some(vec![2, 2]), None);
        let result = translator.translate(&node, &[input], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_avg_pool_with_padding() {
        let translator = AveragePoolTranslator;
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[1, 64, 32, 32]), DType::F32);

        let node = make_node_with_attrs(vec![3, 3], Some(vec![1, 1]), Some(vec![1, 1, 1, 1]));
        let result = translator.translate(&node, &[input], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_avg_pool_default_stride() {
        let translator = AveragePoolTranslator;
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[1, 64, 32, 32]), DType::F32);

        let node = make_node_with_attrs(vec![3, 3], None, None);
        let result = translator.translate(&node, &[input], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_avg_pool_missing_kernel() {
        let translator = AveragePoolTranslator;
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[1, 64, 32, 32]), DType::F32);

        let node = NodeProto {
            name: "avg_pool_test".to_string(),
            op_type: "AveragePool".to_string(),
            ..Default::default()
        };

        let result = translator.translate(&node, &[input], &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_avg_pool_input_validation() {
        let translator = AveragePoolTranslator;

        // 0 inputs should fail
        let err = translator.input_requirement().validate(0, "AveragePool");
        assert!(err.is_err());

        // 1 input should pass
        assert!(
            translator
                .input_requirement()
                .validate(1, "AveragePool")
                .is_ok()
        );

        // 2 inputs should fail
        let err = translator.input_requirement().validate(2, "AveragePool");
        assert!(err.is_err());
    }
}
