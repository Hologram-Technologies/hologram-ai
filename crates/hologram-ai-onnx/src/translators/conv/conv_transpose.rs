//! ConvTranspose operation translator.

use crate::proto::NodeProto;
use crate::translators::{InputRequirement, OnnxAttributes, OnnxTranslator, TranslationError};
use hologram::ir::{GraphBuilder, NodeIndex, NodeOp, Padding};

/// Translator for ONNX ConvTranspose operation.
///
/// ConvTranspose performs transposed convolution (deconvolution), commonly used
/// in generative models for upsampling.
///
/// # ONNX Specification
///
/// - Inputs: X, W, [B]
///   - X: Input tensor [N, C, H, W]
///   - W: Weight tensor [C, M/group, kH, kW]
///   - B: Optional bias [M]
/// - Attributes:
///   - kernel_shape (required): [kH, kW]
///   - strides (default: [1, 1])
///   - pads (default: [0, 0, 0, 0])
///   - output_padding (default: [0, 0]): currently must be [0, 0]
///   - group (default: 1)
#[derive(Debug, Default)]
pub struct ConvTransposeTranslator;

impl OnnxTranslator for ConvTransposeTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "ConvTranspose"
    }

    fn input_requirement(&self) -> InputRequirement {
        InputRequirement::Range(2, 3)
    }

    fn translate(
        &self,
        node: &NodeProto,
        inputs: &[NodeIndex],
        builder: &mut GraphBuilder,
    ) -> Result<Vec<NodeIndex>, TranslationError> {
        let input = inputs[0];
        let weight = inputs[1];
        let bias = inputs.get(2).copied();

        // Parse kernel_shape
        let kernel_shape = node
            .get_ints("kernel_shape")
            .ok_or_else(|| TranslationError::missing_attribute("ConvTranspose", "kernel_shape"))?;
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

        // Parse group
        let groups = node.get_int_or("group", 1) as usize;

        // Check output_padding - not yet supported
        let output_padding = node.get_ints_or("output_padding", &[0, 0]);
        if output_padding.iter().any(|&p| p != 0) {
            return Err(TranslationError::unsupported_op("ConvTranspose", 13));
        }

        // Parse padding
        let pads = node.get_ints_or("pads", &[]);
        let padding = Self::parse_padding(pads)?;

        // Get input shape and dtype for output inference
        let input_node = builder
            .graph()
            .node(input)
            .ok_or_else(|| TranslationError::IrBuilder("Invalid input node".to_string()))?;
        let shape = input_node.op.shape.clone();
        let dtype = input_node.op.dtype;

        // Create transposed convolution node
        let idx = builder.graph_mut().add_op(
            NodeOp::ConvTranspose2d {
                kernel,
                stride,
                padding,
                groups,
            },
            shape,
            dtype,
        );
        builder.graph_mut().connect(input, idx);
        builder.graph_mut().connect(weight, idx);
        let conv_result = idx;

        // Add bias if present
        let result = if let Some(bias_node) = bias {
            builder
                .add(conv_result, bias_node)
                .map_err(|e| TranslationError::IrBuilder(e.to_string()))?
        } else {
            conv_result
        };

        Ok(vec![result])
    }
}

impl ConvTransposeTranslator {
    /// Parse ONNX padding to hologram-ir Padding.
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
            name: "conv_transpose_test".to_string(),
            op_type: "ConvTranspose".to_string(),
            attribute: attrs,
            ..Default::default()
        }
    }

    #[test]
    fn test_conv_transpose_basic() {
        let translator = ConvTransposeTranslator;
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[1, 64, 16, 16]), DType::F32);
        let weight = builder.input("weight", Shape::static_shape(&[64, 32, 3, 3]), DType::F32);

        let node = make_node_with_attrs(vec![3, 3], Some(vec![2, 2]), None);
        let result = translator.translate(&node, &[input, weight], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_conv_transpose_with_padding() {
        let translator = ConvTransposeTranslator;
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[1, 64, 16, 16]), DType::F32);
        let weight = builder.input("weight", Shape::static_shape(&[64, 32, 3, 3]), DType::F32);

        let node = make_node_with_attrs(vec![3, 3], Some(vec![2, 2]), Some(vec![1, 1, 1, 1]));
        let result = translator.translate(&node, &[input, weight], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_conv_transpose_no_stride() {
        let translator = ConvTransposeTranslator;
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[1, 64, 16, 16]), DType::F32);
        let weight = builder.input("weight", Shape::static_shape(&[64, 32, 3, 3]), DType::F32);

        let node = make_node_with_attrs(vec![3, 3], None, None);
        let result = translator.translate(&node, &[input, weight], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_conv_transpose_output_padding_unsupported() {
        let translator = ConvTransposeTranslator;
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[1, 64, 16, 16]), DType::F32);
        let weight = builder.input("weight", Shape::static_shape(&[64, 32, 3, 3]), DType::F32);

        let mut node = make_node_with_attrs(vec![3, 3], None, None);
        node.attribute.push(AttributeProto {
            name: "output_padding".to_string(),
            ints: vec![1, 1],
            ..Default::default()
        });

        let result = translator.translate(&node, &[input, weight], &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_conv_transpose_missing_kernel() {
        let translator = ConvTransposeTranslator;
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[1, 64, 16, 16]), DType::F32);
        let weight = builder.input("weight", Shape::static_shape(&[64, 32, 3, 3]), DType::F32);

        let node = NodeProto {
            name: "conv_transpose_test".to_string(),
            op_type: "ConvTranspose".to_string(),
            ..Default::default()
        };

        let result = translator.translate(&node, &[input, weight], &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_conv_transpose_input_validation() {
        let translator = ConvTransposeTranslator;

        // 0 or 1 input should fail
        let err = translator.input_requirement().validate(0, "ConvTranspose");
        assert!(err.is_err());
        let err = translator.input_requirement().validate(1, "ConvTranspose");
        assert!(err.is_err());

        // 2-3 inputs should pass
        assert!(
            translator
                .input_requirement()
                .validate(2, "ConvTranspose")
                .is_ok()
        );
        assert!(
            translator
                .input_requirement()
                .validate(3, "ConvTranspose")
                .is_ok()
        );

        // 4 inputs should fail
        let err = translator.input_requirement().validate(4, "ConvTranspose");
        assert!(err.is_err());
    }
}
