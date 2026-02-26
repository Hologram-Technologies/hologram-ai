//! Reduction operations - ReduceMean, ReduceSum, etc.

use anyhow::{Context, Result};
use hologram::compiler::OpKind;

use super::{OpTranslator, TranslateContext, TranslateResult};
use crate::proto;

/// ONNX ReduceMean operation.
pub struct ReduceMeanOp;

impl OpTranslator for ReduceMeanOp {
    fn op_type(&self) -> &'static str {
        "ReduceMean"
    }

    fn translate(
        &self,
        node: &proto::NodeProto,
        ctx: &TranslateContext,
    ) -> Result<TranslateResult> {
        reduce_translate(node, ctx, OpKind::Mean)
    }
}

/// ONNX ReduceSum operation.
pub struct ReduceSumOp;

impl OpTranslator for ReduceSumOp {
    fn op_type(&self) -> &'static str {
        "ReduceSum"
    }

    fn translate(
        &self,
        node: &proto::NodeProto,
        ctx: &TranslateContext,
    ) -> Result<TranslateResult> {
        reduce_translate(node, ctx, OpKind::Sum)
    }
}

/// ONNX ReduceMax operation.
pub struct ReduceMaxOp;

impl OpTranslator for ReduceMaxOp {
    fn op_type(&self) -> &'static str {
        "ReduceMax"
    }

    fn translate(
        &self,
        node: &proto::NodeProto,
        ctx: &TranslateContext,
    ) -> Result<TranslateResult> {
        reduce_translate(node, ctx, OpKind::Max)
    }
}

fn reduce_translate(
    node: &proto::NodeProto,
    ctx: &TranslateContext,
    op_kind: OpKind,
) -> Result<TranslateResult> {
    let input_name = node.input.first().context("Reduce op has no input")?;
    let input_node = ctx.get_node(input_name).context("Reduce input not found")?;

    let keepdims = node
        .attribute
        .iter()
        .find(|a| a.name == "keepdims")
        .map(|a| a.i != 0)
        .unwrap_or(true);

    // Get axes - from second input (opset 18+) or attribute
    let axes = if node.input.len() > 1 {
        let axes_name = &node.input[1];
        ctx.get_constant_i64(axes_name).map(|v| {
            v.iter()
                .map(|&a| normalize_axis(a, input_node.shape.len()))
                .collect()
        })
    } else {
        node.attribute.iter().find(|a| a.name == "axes").map(|a| {
            a.ints
                .iter()
                .map(|&a| normalize_axis(a, input_node.shape.len()))
                .collect()
        })
    };

    // Default: reduce all dimensions
    let axes: Vec<usize> = axes.unwrap_or_else(|| (0..input_node.shape.len()).collect());

    // Compute output shape
    let output_shape: Vec<usize> = if keepdims {
        input_node
            .shape
            .iter()
            .enumerate()
            .map(|(i, &dim)| if axes.contains(&i) { 1 } else { dim })
            .collect()
    } else {
        input_node
            .shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !axes.contains(i))
            .map(|(_, &dim)| dim)
            .collect()
    };

    // Note: hologram's reduction ops (Sum, Mean, Max, Min) reduce over all dimensions.
    // For partial reductions, we may need to use Reshape + full reduction + Reshape.
    // For now, emit the basic op and let the backend handle it.

    Ok(TranslateResult::runtime(
        op_kind,
        output_shape,
        input_node.dtype,
    ))
}

fn normalize_axis(axis: i64, rank: usize) -> usize {
    if axis < 0 {
        (rank as i64 + axis) as usize
    } else {
        axis as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram::compiler::{DType, OperationGraph};
    use std::collections::HashMap;

    #[test]
    fn test_reduce_mean() {
        let mut graph = OperationGraph::default();
        let mut value_to_node = HashMap::new();

        let input = hologram::compiler::OpNode::new(0, OpKind::Input, vec![2, 3, 4], DType::F32)
            .with_name("input".to_string());
        graph.nodes.push(input);
        value_to_node.insert("input".to_string(), 0);

        let ctx = TranslateContext::new(&graph, &value_to_node, &[]);

        // Reduce over axis 1, keepdims=true
        let node = proto::NodeProto {
            input: vec!["input".to_string()],
            output: vec!["out".to_string()],
            op_type: "ReduceMean".to_string(),
            attribute: vec![
                proto::AttributeProto {
                    name: "axes".to_string(),
                    ints: vec![1],
                    ..Default::default()
                },
                proto::AttributeProto {
                    name: "keepdims".to_string(),
                    i: 1,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        let result = ReduceMeanOp.translate(&node, &ctx).unwrap();
        assert_eq!(result.shape, vec![2, 1, 4]);
    }
}
