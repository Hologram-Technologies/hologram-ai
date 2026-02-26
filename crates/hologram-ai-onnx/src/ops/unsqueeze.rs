//! Unsqueeze and Squeeze operations - add/remove dimensions.

use anyhow::{Context, Result};
use hologram::compiler::OpKind;

use super::{OpTranslator, TranslateContext, TranslateResult};
use crate::proto;

/// ONNX Unsqueeze operation.
///
/// Inserts singleton dimensions at specified axis positions.
pub struct UnsqueezeOp;

impl OpTranslator for UnsqueezeOp {
    fn op_type(&self) -> &'static str {
        "Unsqueeze"
    }

    fn try_fold(&self, node: &proto::NodeProto, ctx: &TranslateContext) -> Option<TranslateResult> {
        let data_name = node.input.first()?;

        if !ctx.is_constant(data_name) {
            return None;
        }

        let axes = get_unsqueeze_axes(node, ctx)?;
        let data_node = ctx.get_node(data_name)?;
        let data_const = ctx.get_constant_data(data_name)?.clone();

        let new_shape = compute_unsqueeze_shape(&data_node.shape, &axes);

        tracing::debug!(
            "Unsqueeze '{}': {:?} + axes {:?} -> {:?}",
            node.output.first().unwrap_or(&String::new()),
            data_node.shape,
            axes,
            new_shape
        );

        Some(TranslateResult::constant(
            new_shape,
            data_node.dtype,
            data_const,
        ))
    }

    fn translate(
        &self,
        node: &proto::NodeProto,
        ctx: &TranslateContext,
    ) -> Result<TranslateResult> {
        let data_name = node.input.first().context("Unsqueeze has no input")?;
        let data_node = ctx
            .get_node(data_name)
            .context("Unsqueeze input not found")?;

        let axes = get_unsqueeze_axes(node, ctx).context("Could not determine axes")?;
        let new_shape = compute_unsqueeze_shape(&data_node.shape, &axes);

        // Only the first input (data) creates an edge, axes is metadata
        Ok(TranslateResult::runtime_with_inputs(
            OpKind::Reshape {
                shape: new_shape.clone(),
            },
            new_shape,
            data_node.dtype,
            1, // Only 1 data input
        ))
    }
}

/// ONNX Squeeze operation.
///
/// Removes singleton dimensions at specified axis positions.
pub struct SqueezeOp;

impl OpTranslator for SqueezeOp {
    fn op_type(&self) -> &'static str {
        "Squeeze"
    }

    fn try_fold(&self, node: &proto::NodeProto, ctx: &TranslateContext) -> Option<TranslateResult> {
        let data_name = node.input.first()?;

        if !ctx.is_constant(data_name) {
            return None;
        }

        let data_node = ctx.get_node(data_name)?;
        let data_const = ctx.get_constant_data(data_name)?.clone();

        let axes = get_squeeze_axes(node, ctx, &data_node.shape)?;
        let new_shape = compute_squeeze_shape(&data_node.shape, &axes);

        Some(TranslateResult::constant(
            new_shape,
            data_node.dtype,
            data_const,
        ))
    }

    fn translate(
        &self,
        node: &proto::NodeProto,
        ctx: &TranslateContext,
    ) -> Result<TranslateResult> {
        let data_name = node.input.first().context("Squeeze has no input")?;
        let data_node = ctx.get_node(data_name).context("Squeeze input not found")?;

        let axes =
            get_squeeze_axes(node, ctx, &data_node.shape).context("Could not determine axes")?;
        let new_shape = compute_squeeze_shape(&data_node.shape, &axes);

        // Only the first input (data) creates an edge, axes is metadata
        Ok(TranslateResult::runtime_with_inputs(
            OpKind::Reshape {
                shape: new_shape.clone(),
            },
            new_shape,
            data_node.dtype,
            1, // Only 1 data input
        ))
    }
}

fn get_unsqueeze_axes(node: &proto::NodeProto, ctx: &TranslateContext) -> Option<Vec<i64>> {
    // ONNX opset 13+: axes from second input
    if node.input.len() > 1 {
        let axes_name = &node.input[1];
        return ctx.get_constant_i64(axes_name);
    }

    // ONNX opset < 13: axes from attribute
    node.attribute
        .iter()
        .find(|a| a.name == "axes")
        .map(|a| a.ints.clone())
}

fn get_squeeze_axes(
    node: &proto::NodeProto,
    ctx: &TranslateContext,
    shape: &[usize],
) -> Option<Vec<i64>> {
    // ONNX opset 13+: axes from second input
    if node.input.len() > 1 {
        let axes_name = &node.input[1];
        return ctx.get_constant_i64(axes_name);
    }

    // ONNX opset < 13: axes from attribute, or squeeze all 1s
    if let Some(attr) = node.attribute.iter().find(|a| a.name == "axes") {
        return Some(attr.ints.clone());
    }

    // Default: squeeze all singleton dimensions
    Some(
        shape
            .iter()
            .enumerate()
            .filter(|&(_, dim)| *dim == 1)
            .map(|(i, _)| i as i64)
            .collect(),
    )
}

fn compute_unsqueeze_shape(shape: &[usize], axes: &[i64]) -> Vec<usize> {
    let final_rank = shape.len() + axes.len();

    // Normalize negative axes to positive indices in the output shape
    let normalized_axes: Vec<usize> = axes
        .iter()
        .map(|&a| {
            if a < 0 {
                (final_rank as i64 + a) as usize
            } else {
                a as usize
            }
        })
        .collect();

    // Build output shape: place 1s at specified axes, fill rest with input dims
    let mut new_shape = vec![0usize; final_rank];

    // Mark axis positions with 1
    for &axis in &normalized_axes {
        new_shape[axis] = 1;
    }

    // Fill remaining positions with input dimensions in order
    let mut input_idx = 0;
    for dim in &mut new_shape {
        if *dim == 0 {
            *dim = shape[input_idx];
            input_idx += 1;
        }
    }

    new_shape
}

fn compute_squeeze_shape(shape: &[usize], axes: &[i64]) -> Vec<usize> {
    let rank = shape.len();

    // Normalize negative axes
    let normalized_axes: Vec<usize> = axes
        .iter()
        .map(|&a| {
            if a < 0 {
                (rank as i64 + a) as usize
            } else {
                a as usize
            }
        })
        .collect();

    shape
        .iter()
        .enumerate()
        .filter(|(i, _)| !normalized_axes.contains(i))
        .map(|(_, &dim)| dim)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unsqueeze_shape() {
        assert_eq!(compute_unsqueeze_shape(&[3, 4], &[0]), vec![1, 3, 4]);
        assert_eq!(compute_unsqueeze_shape(&[3, 4], &[1]), vec![3, 1, 4]);
        assert_eq!(compute_unsqueeze_shape(&[3, 4], &[2]), vec![3, 4, 1]);
        assert_eq!(compute_unsqueeze_shape(&[3, 4], &[-1]), vec![3, 4, 1]);
        assert_eq!(compute_unsqueeze_shape(&[3, 4], &[0, 2]), vec![1, 3, 1, 4]);
    }

    #[test]
    fn test_squeeze_shape() {
        assert_eq!(compute_squeeze_shape(&[1, 3, 4], &[0]), vec![3, 4]);
        assert_eq!(compute_squeeze_shape(&[3, 1, 4], &[1]), vec![3, 4]);
        assert_eq!(compute_squeeze_shape(&[3, 4, 1], &[2]), vec![3, 4]);
        assert_eq!(compute_squeeze_shape(&[1, 3, 1, 4], &[0, 2]), vec![3, 4]);
    }
}
