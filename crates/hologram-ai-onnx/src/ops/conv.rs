//! Convolutional operations - Conv, ConvTranspose, etc.

use anyhow::{Context, Result};
use hologram::compiler::OpKind;

use super::{OpTranslator, TranslateContext, TranslateResult};
use crate::proto;

/// Helper to get integer attribute.
fn get_int_attr(node: &proto::NodeProto, name: &str) -> Option<i64> {
    node.attribute.iter().find(|a| a.name == name).map(|a| a.i)
}

/// Helper to get integer array attribute.
fn get_ints_attr(node: &proto::NodeProto, name: &str) -> Option<Vec<i64>> {
    node.attribute
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.ints.clone())
}

/// ONNX Conv operation.
pub struct ConvOp;

impl OpTranslator for ConvOp {
    fn op_type(&self) -> &'static str {
        "Conv"
    }

    fn translate(
        &self,
        node: &proto::NodeProto,
        ctx: &TranslateContext,
    ) -> Result<TranslateResult> {
        let input_name = node.input.first().context("Conv has no input")?;
        let weight_name = node.input.get(1).context("Conv has no weight")?;

        let input_node = ctx.get_node(input_name).context("Conv input not found")?;
        let weight_node = ctx.get_node(weight_name).context("Conv weight not found")?;

        // Extract Conv attributes
        let strides = get_ints_attr(node, "strides").unwrap_or_else(|| vec![1, 1]);
        let pads = get_ints_attr(node, "pads").unwrap_or_else(|| vec![0, 0, 0, 0]);
        let dilations = get_ints_attr(node, "dilations").unwrap_or_else(|| vec![1, 1]);
        let groups = get_int_attr(node, "group").unwrap_or(1) as usize;

        // Input shape: [N, C_in, H_in, W_in]
        // Weight shape: [C_out, C_in/group, KH, KW]
        if input_node.shape.len() != 4 {
            anyhow::bail!("Conv expects 4D input, got {:?}", input_node.shape);
        }
        if weight_node.shape.len() != 4 {
            anyhow::bail!("Conv expects 4D weight, got {:?}", weight_node.shape);
        }

        let batch = input_node.shape[0];
        let out_channels = weight_node.shape[0];
        let h_in = input_node.shape[2];
        let w_in = input_node.shape[3];

        let kernel_h = weight_node.shape[2];
        let kernel_w = weight_node.shape[3];

        // Calculate output dimensions using ONNX Conv formula
        // H_out = (H_in + pad_top + pad_bottom - dilation_h * (kernel_h - 1) - 1) / stride_h + 1
        let stride_h = strides[0] as usize;
        let stride_w = strides.get(1).copied().unwrap_or(strides[0]) as usize;

        let pad_top = pads[0] as usize;
        let pad_bottom = pads.get(2).copied().unwrap_or(pads[0]) as usize;
        let pad_left = pads.get(1).copied().unwrap_or(pads[0]) as usize;
        let pad_right = pads
            .get(3)
            .copied()
            .unwrap_or(pads.get(1).copied().unwrap_or(pads[0])) as usize;

        let dilation_h = dilations[0] as usize;
        let dilation_w = dilations.get(1).copied().unwrap_or(dilations[0]) as usize;

        let h_out = (h_in + pad_top + pad_bottom - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
        let w_out = (w_in + pad_left + pad_right - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

        let output_shape = vec![batch, out_channels, h_out, w_out];

        // Map to hologram Conv2d operation
        Ok(TranslateResult::runtime(
            OpKind::Conv2d {
                kernel: (kernel_h, kernel_w),
                stride: (stride_h, stride_w),
                padding: (pad_top, pad_left),
                dilation: (dilation_h, dilation_w),
                groups,
            },
            output_shape,
            input_node.dtype,
        ))
    }
}

/// ONNX MaxPool operation.
pub struct MaxPoolOp;

impl OpTranslator for MaxPoolOp {
    fn op_type(&self) -> &'static str {
        "MaxPool"
    }

    fn translate(
        &self,
        node: &proto::NodeProto,
        ctx: &TranslateContext,
    ) -> Result<TranslateResult> {
        let input_name = node.input.first().context("MaxPool has no input")?;
        let input_node = ctx
            .get_node(input_name)
            .context("MaxPool input not found")?;

        // Extract attributes
        let kernel_shape =
            get_ints_attr(node, "kernel_shape").context("MaxPool missing kernel_shape")?;
        let strides = get_ints_attr(node, "strides").unwrap_or_else(|| vec![1, 1]);
        let pads = get_ints_attr(node, "pads").unwrap_or_else(|| vec![0, 0, 0, 0]);

        // Input shape: [N, C, H_in, W_in]
        if input_node.shape.len() != 4 {
            anyhow::bail!("MaxPool expects 4D input, got {:?}", input_node.shape);
        }

        let batch = input_node.shape[0];
        let channels = input_node.shape[1];
        let h_in = input_node.shape[2];
        let w_in = input_node.shape[3];

        let kernel_h = kernel_shape[0] as usize;
        let kernel_w = kernel_shape.get(1).copied().unwrap_or(kernel_shape[0]) as usize;
        let stride_h = strides[0] as usize;
        let stride_w = strides.get(1).copied().unwrap_or(strides[0]) as usize;

        let pad_top = pads[0] as usize;
        let pad_bottom = pads.get(2).copied().unwrap_or(pads[0]) as usize;
        let pad_left = pads.get(1).copied().unwrap_or(pads[0]) as usize;
        let pad_right = pads
            .get(3)
            .copied()
            .unwrap_or(pads.get(1).copied().unwrap_or(pads[0])) as usize;

        let h_out = (h_in + pad_top + pad_bottom - kernel_h) / stride_h + 1;
        let w_out = (w_in + pad_left + pad_right - kernel_w) / stride_w + 1;

        let output_shape = vec![batch, channels, h_out, w_out];

        // Map to hologram MaxPool operation
        Ok(TranslateResult::runtime(
            OpKind::MaxPool {
                kernel: (kernel_h, kernel_w),
                stride: (stride_h, stride_w),
                padding: (pad_top, pad_left),
            },
            output_shape,
            input_node.dtype,
        ))
    }
}

/// ONNX GlobalAveragePool operation.
pub struct GlobalAveragePoolOp;

impl OpTranslator for GlobalAveragePoolOp {
    fn op_type(&self) -> &'static str {
        "GlobalAveragePool"
    }

    fn translate(
        &self,
        node: &proto::NodeProto,
        ctx: &TranslateContext,
    ) -> Result<TranslateResult> {
        let input_name = node
            .input
            .first()
            .context("GlobalAveragePool has no input")?;
        let input_node = ctx
            .get_node(input_name)
            .context("GlobalAveragePool input not found")?;

        // Input shape: [N, C, H, W]
        // Output shape: [N, C, 1, 1]
        if input_node.shape.len() != 4 {
            anyhow::bail!(
                "GlobalAveragePool expects 4D input, got {:?}",
                input_node.shape
            );
        }

        let output_shape = vec![input_node.shape[0], input_node.shape[1], 1, 1];

        // Map to hologram GlobalAveragePool operation
        Ok(TranslateResult::runtime(
            OpKind::GlobalAveragePool,
            output_shape,
            input_node.dtype,
        ))
    }
}

/// ONNX AveragePool operation.
pub struct AveragePoolOp;

impl OpTranslator for AveragePoolOp {
    fn op_type(&self) -> &'static str {
        "AveragePool"
    }

    fn translate(
        &self,
        node: &proto::NodeProto,
        ctx: &TranslateContext,
    ) -> Result<TranslateResult> {
        let input_name = node.input.first().context("AveragePool has no input")?;
        let input_node = ctx
            .get_node(input_name)
            .context("AveragePool input not found")?;

        // Similar to MaxPool
        let kernel_shape =
            get_ints_attr(node, "kernel_shape").context("AveragePool missing kernel_shape")?;
        let strides = get_ints_attr(node, "strides").unwrap_or_else(|| vec![1, 1]);
        let pads = get_ints_attr(node, "pads").unwrap_or_else(|| vec![0, 0, 0, 0]);

        if input_node.shape.len() != 4 {
            anyhow::bail!("AveragePool expects 4D input, got {:?}", input_node.shape);
        }

        let batch = input_node.shape[0];
        let channels = input_node.shape[1];
        let h_in = input_node.shape[2];
        let w_in = input_node.shape[3];

        let kernel_h = kernel_shape[0] as usize;
        let kernel_w = kernel_shape.get(1).copied().unwrap_or(kernel_shape[0]) as usize;
        let stride_h = strides[0] as usize;
        let stride_w = strides.get(1).copied().unwrap_or(strides[0]) as usize;

        let pad_top = pads[0] as usize;
        let pad_bottom = pads.get(2).copied().unwrap_or(pads[0]) as usize;
        let pad_left = pads.get(1).copied().unwrap_or(pads[0]) as usize;
        let pad_right = pads
            .get(3)
            .copied()
            .unwrap_or(pads.get(1).copied().unwrap_or(pads[0])) as usize;

        let h_out = (h_in + pad_top + pad_bottom - kernel_h) / stride_h + 1;
        let w_out = (w_in + pad_left + pad_right - kernel_w) / stride_w + 1;

        let output_shape = vec![batch, channels, h_out, w_out];

        // Map to hologram AvgPool operation
        Ok(TranslateResult::runtime(
            OpKind::AvgPool {
                kernel: (kernel_h, kernel_w),
                stride: (stride_h, stride_w),
                padding: (pad_top, pad_left),
            },
            output_shape,
            input_node.dtype,
        ))
    }
}

/// ONNX BatchNormalization operation.
pub struct BatchNormalizationOp;

impl OpTranslator for BatchNormalizationOp {
    fn op_type(&self) -> &'static str {
        "BatchNormalization"
    }

    fn translate(
        &self,
        node: &proto::NodeProto,
        ctx: &TranslateContext,
    ) -> Result<TranslateResult> {
        let input_name = node
            .input
            .first()
            .context("BatchNormalization has no input")?;
        let input_node = ctx
            .get_node(input_name)
            .context("BatchNormalization input not found")?;

        // Extract epsilon attribute (default 1e-5)
        let epsilon = node
            .attribute
            .iter()
            .find(|a| a.name == "epsilon")
            .map(|a| a.f)
            .unwrap_or(1e-5);

        // BatchNorm doesn't change shape
        Ok(TranslateResult::runtime(
            OpKind::BatchNormalization { epsilon },
            input_node.shape.clone(),
            input_node.dtype,
        ))
    }
}

/// ONNX Flatten operation.
pub struct FlattenOp;

impl OpTranslator for FlattenOp {
    fn op_type(&self) -> &'static str {
        "Flatten"
    }

    fn translate(
        &self,
        node: &proto::NodeProto,
        ctx: &TranslateContext,
    ) -> Result<TranslateResult> {
        let input_name = node.input.first().context("Flatten has no input")?;
        let input_node = ctx
            .get_node(input_name)
            .context("Flatten input not found")?;

        let axis = get_int_attr(node, "axis").unwrap_or(1) as usize;

        if axis >= input_node.shape.len() {
            anyhow::bail!(
                "Flatten axis {} >= input dims {}",
                axis,
                input_node.shape.len()
            );
        }

        let mut output_shape = input_node.shape[..axis].to_vec();
        let suffix: usize = input_node.shape[axis..].iter().product();
        output_shape.push(suffix);

        Ok(TranslateResult::runtime(
            OpKind::Flatten { start_dim: axis },
            output_shape,
            input_node.dtype,
        ))
    }
}
