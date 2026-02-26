//! Gemm operation - General Matrix Multiply.
//!
//! ONNX Gemm computes Y = alpha * A @ B' + beta * C
//! We expand this into separate ops: optional Transpose + MatMul + Add

use anyhow::{Context, Result, bail};
use hologram::compiler::OpKind;

use super::{OpTranslator, TranslateContext, TranslateResult};
use crate::proto;

/// ONNX Gemm operation.
///
/// This operation is special because it expands into multiple hologram ops.
/// The builder handles this expansion, but we still implement the trait
/// for consistency and to provide shape inference.
pub struct GemmOp;

impl OpTranslator for GemmOp {
    fn op_type(&self) -> &'static str {
        "Gemm"
    }

    fn requires_expansion(&self) -> bool {
        true // Gemm expands into MatMul + Add + optional Transpose
    }

    fn translate(
        &self,
        node: &proto::NodeProto,
        ctx: &TranslateContext,
    ) -> Result<TranslateResult> {
        // Gemm has 3 inputs: A (input), B (weight), C (bias - optional)
        let input_a = node.input.first().context("Gemm missing input A")?;
        let input_b = node.input.get(1).context("Gemm missing input B")?;

        let a_node = ctx.get_node(input_a).context("Gemm input A not found")?;
        let b_node = ctx.get_node(input_b).context("Gemm input B not found")?;

        // Check transB attribute
        let trans_b = node
            .attribute
            .iter()
            .any(|attr| attr.name == "transB" && attr.i == 1);

        // Compute output shape based on A and B (with optional transpose)
        let a_shape = &a_node.shape;
        let b_shape = &b_node.shape;

        if a_shape.len() != 2 || b_shape.len() != 2 {
            bail!(
                "Gemm requires 2D inputs, got A: {:?}, B: {:?}",
                a_shape,
                b_shape
            );
        }

        let m = a_shape[0];
        let k = a_shape[1];
        let n = if trans_b { b_shape[0] } else { b_shape[1] };

        // Verify dimension compatibility
        let b_k = if trans_b { b_shape[1] } else { b_shape[0] };
        if k != b_k {
            bail!(
                "Gemm dimension mismatch: A[{}, {}] x B{} with k={}",
                m,
                k,
                if trans_b { "[n, k]'" } else { "[k, n]" },
                b_k
            );
        }

        // Return shape info - the builder will handle the actual expansion
        // We use a placeholder OpKind since the builder creates multiple nodes
        Ok(TranslateResult::runtime(
            OpKind::MatMul { m, k, n },
            vec![m, n],
            a_node.dtype,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram::compiler::{DType, OperationGraph};
    use std::collections::HashMap;

    #[test]
    fn test_gemm_basic() {
        let mut graph = OperationGraph::default();
        let mut value_to_node = HashMap::new();

        let a = hologram::compiler::OpNode::new(0, OpKind::Input, vec![2, 3], DType::F32)
            .with_name("a".to_string());
        let b = hologram::compiler::OpNode::new(1, OpKind::Input, vec![3, 4], DType::F32)
            .with_name("b".to_string());

        graph.nodes.push(a);
        graph.nodes.push(b);
        value_to_node.insert("a".to_string(), 0);
        value_to_node.insert("b".to_string(), 1);

        let ctx = TranslateContext::new(&graph, &value_to_node, &[]);

        let node = proto::NodeProto {
            input: vec!["a".to_string(), "b".to_string()],
            output: vec!["out".to_string()],
            op_type: "Gemm".to_string(),
            ..Default::default()
        };

        let result = GemmOp.translate(&node, &ctx).unwrap();
        assert_eq!(result.shape, vec![2, 4]);
    }

    #[test]
    fn test_gemm_trans_b() {
        let mut graph = OperationGraph::default();
        let mut value_to_node = HashMap::new();

        let a = hologram::compiler::OpNode::new(0, OpKind::Input, vec![2, 3], DType::F32)
            .with_name("a".to_string());
        // B is [4, 3] but transB=1 means we treat it as [3, 4]
        let b = hologram::compiler::OpNode::new(1, OpKind::Input, vec![4, 3], DType::F32)
            .with_name("b".to_string());

        graph.nodes.push(a);
        graph.nodes.push(b);
        value_to_node.insert("a".to_string(), 0);
        value_to_node.insert("b".to_string(), 1);

        let ctx = TranslateContext::new(&graph, &value_to_node, &[]);

        let node = proto::NodeProto {
            input: vec!["a".to_string(), "b".to_string()],
            output: vec!["out".to_string()],
            op_type: "Gemm".to_string(),
            attribute: vec![proto::AttributeProto {
                name: "transB".to_string(),
                i: 1,
                ..Default::default()
            }],
            ..Default::default()
        };

        let result = GemmOp.translate(&node, &ctx).unwrap();
        assert_eq!(result.shape, vec![2, 4]);
    }
}
