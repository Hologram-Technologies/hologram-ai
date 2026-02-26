//! Expand operation - broadcasts a tensor to a larger shape.

use anyhow::{Context, Result};
use hologram::compiler::{ConstantData, OpKind};

use super::{OpTranslator, TranslateContext, TranslateResult};
use crate::proto;

/// ONNX Expand operation.
///
/// Broadcasts input tensor to a larger shape based on numpy broadcasting rules.
/// The `shape` input specifies the target shape.
pub struct ExpandOp;

impl OpTranslator for ExpandOp {
    fn op_type(&self) -> &'static str {
        "Expand"
    }

    fn try_fold(&self, node: &proto::NodeProto, ctx: &TranslateContext) -> Option<TranslateResult> {
        let input_name = node.input.first()?;
        let shape_name = node.input.get(1)?;

        // Only fold if both inputs are constant
        if !ctx.is_constant(input_name) || !ctx.is_constant(shape_name) {
            return None;
        }

        let input_data = ctx.get_constant_data(input_name)?;
        let input_node = ctx.get_node(input_name)?;
        let shape_values = ctx.get_constant_i64(shape_name)?;

        let output_shape: Vec<usize> = shape_values.iter().map(|&d| d as usize).collect();
        let total_size: usize = output_shape.iter().product();

        // Expand the constant data
        let expanded_data = expand_constant(input_data, &input_node.shape, &output_shape)?;

        tracing::debug!(
            "Expand '{}': {:?} -> {:?}",
            node.output.first().unwrap_or(&String::new()),
            input_node.shape,
            output_shape
        );

        // Verify size matches
        if expanded_data.len() != total_size {
            return None;
        }

        Some(TranslateResult::constant(
            output_shape,
            input_node.dtype,
            expanded_data,
        ))
    }

    fn translate(
        &self,
        node: &proto::NodeProto,
        ctx: &TranslateContext,
    ) -> Result<TranslateResult> {
        let input_name = node.input.first().context("Expand has no input")?;
        let shape_name = node.input.get(1).context("Expand has no shape input")?;

        let input_node = ctx.get_node(input_name).context("Expand input not found")?;

        // Get target shape from constant (required for Expand)
        let shape_values = ctx
            .get_constant_i64(shape_name)
            .context("Expand requires constant shape input")?;

        let output_shape: Vec<usize> = shape_values.iter().map(|&d| d as usize).collect();

        // Use native hologram Expand operation
        Ok(TranslateResult::runtime_with_inputs(
            OpKind::Expand {
                shape: output_shape.clone(),
            },
            output_shape,
            input_node.dtype,
            1, // Only the data input creates an edge, not the shape
        ))
    }
}

/// Expand constant data by broadcasting.
fn expand_constant(
    data: &ConstantData,
    src_shape: &[usize],
    tgt_shape: &[usize],
) -> Option<ConstantData> {
    let total_size: usize = tgt_shape.iter().product();

    macro_rules! expand_impl {
        ($variant:ident, $values:expr) => {{
            let mut result = Vec::with_capacity(total_size);
            for i in 0..total_size {
                let src_idx = broadcast_index(i, tgt_shape, src_shape);
                result.push($values[src_idx].clone());
            }
            Some(ConstantData::$variant(result))
        }};
    }

    match data {
        ConstantData::F32(v) => expand_impl!(F32, v),
        ConstantData::F64(v) => expand_impl!(F64, v),
        ConstantData::I32(v) => expand_impl!(I32, v),
        ConstantData::I64(v) => expand_impl!(I64, v),
        ConstantData::U8(v) => expand_impl!(U8, v),
        ConstantData::U16(v) => expand_impl!(U16, v),
        ConstantData::U32(v) => expand_impl!(U32, v),
        ConstantData::Bool(v) => expand_impl!(Bool, v),
    }
}

/// Compute the source index for a broadcast operation.
fn broadcast_index(flat_idx: usize, output_shape: &[usize], input_shape: &[usize]) -> usize {
    if input_shape.is_empty() || input_shape.iter().product::<usize>() == 1 {
        return 0; // Scalar broadcast
    }

    let rank_diff = output_shape.len().saturating_sub(input_shape.len());
    let mut result = 0;
    let mut stride = 1;

    for i in (0..input_shape.len()).rev() {
        let out_idx = i + rank_diff;
        let coord = (flat_idx / stride_at(output_shape, out_idx)) % output_shape[out_idx];
        let input_coord = if input_shape[i] == 1 { 0 } else { coord };
        result += input_coord * stride;
        stride *= input_shape[i];
    }

    result
}

fn stride_at(shape: &[usize], idx: usize) -> usize {
    shape[idx + 1..].iter().product()
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram::compiler::{DType, OperationGraph};
    use std::collections::HashMap;

    #[test]
    fn test_expand_scalar() {
        let mut graph = OperationGraph::default();
        let mut value_to_node = HashMap::new();

        // Scalar input
        let input = hologram::compiler::OpNode::new(0, OpKind::Constant, vec![1], DType::F32)
            .with_name("input".to_string());
        graph.nodes.push(input);
        value_to_node.insert("input".to_string(), 0);
        graph.constants.push(ConstantData::F32(vec![5.0]));

        // Target shape [2, 3]
        let shape = hologram::compiler::OpNode::new(1, OpKind::Constant, vec![2], DType::I64)
            .with_name("shape".to_string());
        graph.nodes.push(shape);
        value_to_node.insert("shape".to_string(), 1);
        graph.constants.push(ConstantData::I64(vec![2, 3]));

        let ctx = TranslateContext::new(&graph, &value_to_node, &graph.constants);

        let node = proto::NodeProto {
            input: vec!["input".to_string(), "shape".to_string()],
            output: vec!["out".to_string()],
            op_type: "Expand".to_string(),
            ..Default::default()
        };

        let result = ExpandOp.try_fold(&node, &ctx).expect("Should fold");
        assert_eq!(result.shape, vec![2, 3]);

        if let Some(ConstantData::F32(data)) = result.constant_data {
            assert_eq!(data, vec![5.0, 5.0, 5.0, 5.0, 5.0, 5.0]);
        } else {
            panic!("Expected F32 constant data");
        }
    }
}
