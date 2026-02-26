//! Range operation - generate a sequence of numbers.

use anyhow::{Context, Result};
use hologram::compiler::{ConstantData, DType};

use super::{ConstantScalar, OpTranslator, TranslateContext, TranslateResult};
use crate::proto;

/// ONNX Range operation.
///
/// Generates a 1D tensor containing a sequence of numbers from start to limit
/// with the given step (delta). When all inputs are constants, this is
/// evaluated at compile time.
pub struct RangeOp;

impl OpTranslator for RangeOp {
    fn op_type(&self) -> &'static str {
        "Range"
    }

    fn try_fold(&self, node: &proto::NodeProto, ctx: &TranslateContext) -> Option<TranslateResult> {
        let start_name = node.input.first()?;
        let limit_name = node.input.get(1)?;
        let delta_name = node.input.get(2)?;

        // All inputs must be constants
        if !ctx.is_constant(start_name)
            || !ctx.is_constant(limit_name)
            || !ctx.is_constant(delta_name)
        {
            return None;
        }

        let start_const = ctx.get_constant_data(start_name)?;
        let limit_const = ctx.get_constant_data(limit_name)?;
        let delta_const = ctx.get_constant_data(delta_name)?;

        // Try integer range first
        if let (Some(start), Some(limit), Some(delta)) = (
            start_const.as_i64_scalar(),
            limit_const.as_i64_scalar(),
            delta_const.as_i64_scalar(),
        ) {
            if delta == 0 {
                return None; // Prevent infinite loop
            }

            let range_len = ((limit - start) / delta).max(0) as usize;
            let range_data: Vec<i64> = (0..range_len).map(|i| start + (i as i64) * delta).collect();

            tracing::debug!(
                "Range '{}': start={} limit={} delta={} -> [{}]",
                node.output.first().unwrap_or(&String::new()),
                start,
                limit,
                delta,
                range_len
            );

            return Some(TranslateResult::constant(
                vec![range_len],
                DType::I64,
                ConstantData::I64(range_data),
            ));
        }

        // Try float range
        if let (Some(start), Some(limit), Some(delta)) = (
            start_const.as_f32_scalar(),
            limit_const.as_f32_scalar(),
            delta_const.as_f32_scalar(),
        ) {
            if delta.abs() < f32::EPSILON {
                return None;
            }

            let range_len = ((limit - start) / delta).max(0.0).ceil() as usize;
            let range_data: Vec<f32> = (0..range_len).map(|i| start + (i as f32) * delta).collect();

            return Some(TranslateResult::constant(
                vec![range_len],
                DType::F32,
                ConstantData::F32(range_data),
            ));
        }

        None
    }

    fn translate(
        &self,
        node: &proto::NodeProto,
        ctx: &TranslateContext,
    ) -> Result<TranslateResult> {
        // Range must be constant-folded - all inputs should be constants
        let start_name = node.input.first().context("Range has no start")?;
        let limit_name = node.input.get(1).context("Range has no limit")?;
        let delta_name = node.input.get(2).context("Range has no delta")?;

        // Try to get constant values
        let start_node = ctx.get_node(start_name).context("Range start not found")?;

        // Try I64 first, then F32
        if let (Some(start_vals), Some(limit_vals), Some(delta_vals)) = (
            ctx.get_constant_i64(start_name),
            ctx.get_constant_i64(limit_name),
            ctx.get_constant_i64(delta_name),
        ) {
            let start = start_vals.first().copied().unwrap_or(0);
            let limit = limit_vals.first().copied().unwrap_or(0);
            let delta = delta_vals.first().copied().unwrap_or(1);

            let range_len = ((limit - start) / delta.max(1)) as usize;
            let range_data: Vec<i64> = (0..range_len as i64).map(|i| start + i * delta).collect();

            return Ok(TranslateResult::constant(
                vec![range_len],
                DType::I64,
                ConstantData::I64(range_data),
            ));
        }

        if let (Some(start_vals), Some(limit_vals), Some(delta_vals)) = (
            ctx.get_constant_f32(start_name),
            ctx.get_constant_f32(limit_name),
            ctx.get_constant_f32(delta_name),
        ) {
            let start = start_vals.first().copied().unwrap_or(0.0);
            let limit = limit_vals.first().copied().unwrap_or(0.0);
            let delta = delta_vals.first().copied().unwrap_or(1.0);

            let range_len = ((limit - start) / delta.max(f32::EPSILON)).ceil() as usize;
            let range_data: Vec<f32> = (0..range_len).map(|i| start + (i as f32) * delta).collect();

            return Ok(TranslateResult::constant(
                vec![range_len],
                DType::F32,
                ConstantData::F32(range_data),
            ));
        }

        anyhow::bail!(
            "Range operation requires all inputs (start, limit, delta) to be constants. \
             Input types: start={:?}",
            start_node.dtype
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram::compiler::{OpKind, OperationGraph};
    use std::collections::HashMap;

    fn setup_range_test(
        start: i64,
        limit: i64,
        delta: i64,
    ) -> (OperationGraph, HashMap<String, u32>, Vec<ConstantData>) {
        let mut graph = OperationGraph::default();
        let mut value_to_node = HashMap::new();
        let mut constants = Vec::new();

        // Add start constant
        let start_node = hologram::compiler::OpNode::new(0, OpKind::Constant, vec![], DType::I64)
            .with_name("start".to_string());
        graph.nodes.push(start_node);
        value_to_node.insert("start".to_string(), 0);
        constants.push(ConstantData::I64(vec![start]));

        // Add limit constant
        let limit_node = hologram::compiler::OpNode::new(1, OpKind::Constant, vec![], DType::I64)
            .with_name("limit".to_string());
        graph.nodes.push(limit_node);
        value_to_node.insert("limit".to_string(), 1);
        constants.push(ConstantData::I64(vec![limit]));

        // Add delta constant
        let delta_node = hologram::compiler::OpNode::new(2, OpKind::Constant, vec![], DType::I64)
            .with_name("delta".to_string());
        graph.nodes.push(delta_node);
        value_to_node.insert("delta".to_string(), 2);
        constants.push(ConstantData::I64(vec![delta]));

        (graph, value_to_node, constants)
    }

    #[test]
    fn test_range_constant_fold() {
        let (graph, value_to_node, constants) = setup_range_test(0, 5, 1);
        let ctx = TranslateContext::new(&graph, &value_to_node, &constants);

        let node = proto::NodeProto {
            input: vec![
                "start".to_string(),
                "limit".to_string(),
                "delta".to_string(),
            ],
            output: vec!["range_out".to_string()],
            op_type: "Range".to_string(),
            ..Default::default()
        };

        let op = RangeOp;
        let result = op.try_fold(&node, &ctx).expect("Should fold");

        assert_eq!(result.shape, vec![5]);
        if let Some(ConstantData::I64(data)) = result.constant_data {
            assert_eq!(data, vec![0, 1, 2, 3, 4]);
        } else {
            panic!("Expected I64 constant");
        }
    }

    #[test]
    fn test_range_with_step() {
        let (graph, value_to_node, constants) = setup_range_test(0, 10, 2);
        let ctx = TranslateContext::new(&graph, &value_to_node, &constants);

        let node = proto::NodeProto {
            input: vec![
                "start".to_string(),
                "limit".to_string(),
                "delta".to_string(),
            ],
            output: vec!["range_out".to_string()],
            op_type: "Range".to_string(),
            ..Default::default()
        };

        let op = RangeOp;
        let result = op.try_fold(&node, &ctx).expect("Should fold");

        assert_eq!(result.shape, vec![5]);
        if let Some(ConstantData::I64(data)) = result.constant_data {
            assert_eq!(data, vec![0, 2, 4, 6, 8]);
        } else {
            panic!("Expected I64 constant");
        }
    }
}
