//! MatMul operation - matrix multiplication.

use anyhow::{Context, Result, bail};
use hologram::compiler::OpKind;

use super::{OpTranslator, TranslateContext, TranslateResult};
use crate::proto;

/// ONNX MatMul operation.
///
/// Matrix multiplication with optional batching for higher-rank tensors.
pub struct MatMulOp;

impl OpTranslator for MatMulOp {
    fn op_type(&self) -> &'static str {
        "MatMul"
    }

    fn translate(
        &self,
        node: &proto::NodeProto,
        ctx: &TranslateContext,
    ) -> Result<TranslateResult> {
        let a_name = node.input.first().context("MatMul has no first input")?;
        let b_name = node.input.get(1).context("MatMul has no second input")?;

        let a_node = ctx
            .get_node(a_name)
            .context("MatMul first input not found")?;
        let b_node = ctx
            .get_node(b_name)
            .context("MatMul second input not found")?;

        let a_shape = &a_node.shape;
        let b_shape = &b_node.shape;

        // Handle different MatMul cases
        match (a_shape.len(), b_shape.len()) {
            // 2D x 2D: Standard matrix multiplication
            (2, 2) => {
                let m = a_shape[0];
                let k = a_shape[1];
                let n = b_shape[1];

                if b_shape[0] != k {
                    bail!(
                        "MatMul dimension mismatch: A[{}, {}] x B[{}, {}]",
                        m,
                        k,
                        b_shape[0],
                        n
                    );
                }

                Ok(TranslateResult::runtime(
                    OpKind::MatMul { m, k, n },
                    vec![m, n],
                    a_node.dtype,
                ))
            }

            // Batched MatMul: A[...batch, M, K] x B[...batch, K, N]
            (a_rank, b_rank) if a_rank >= 3 && b_rank >= 3 && a_rank == b_rank => {
                let rank = a_rank;
                let batch: Vec<usize> = a_shape[..rank - 2].to_vec();

                // Validate batch dimensions match
                let b_batch = &b_shape[..rank - 2];
                if batch != b_batch {
                    bail!("BatchMatMul batch mismatch: {:?} vs {:?}", batch, b_batch);
                }

                let m = a_shape[rank - 2];
                let k = a_shape[rank - 1];
                let n = b_shape[rank - 1];

                if b_shape[rank - 2] != k {
                    bail!(
                        "BatchMatMul inner dimension mismatch: A has k={}, B has k={}",
                        k,
                        b_shape[rank - 2]
                    );
                }

                let mut output_shape = batch.clone();
                output_shape.push(m);
                output_shape.push(n);

                Ok(TranslateResult::runtime(
                    OpKind::BatchMatMul { batch, m, k, n },
                    output_shape,
                    a_node.dtype,
                ))
            }

            // 1D x 2D or 2D x 1D: Vector-matrix multiplication
            (1, 2) => {
                let k = a_shape[0];
                let n = b_shape[1];

                if b_shape[0] != k {
                    bail!(
                        "MatMul: vector length {} doesn't match matrix rows {}",
                        k,
                        b_shape[0]
                    );
                }

                Ok(TranslateResult::runtime(
                    OpKind::MatMul { m: 1, k, n },
                    vec![n],
                    a_node.dtype,
                ))
            }

            (2, 1) => {
                let m = a_shape[0];
                let k = a_shape[1];

                if b_shape[0] != k {
                    bail!(
                        "MatMul: matrix cols {} doesn't match vector length {}",
                        k,
                        b_shape[0]
                    );
                }

                Ok(TranslateResult::runtime(
                    OpKind::MatMul { m, k, n: 1 },
                    vec![m],
                    a_node.dtype,
                ))
            }

            // ND x 2D: Broadcast matmul - A[...batch, M, K] x B[K, N] -> [...batch, M, N]
            // This is the common pattern for linear layers in transformers
            (a_rank, 2) if a_rank >= 3 => {
                let batch: Vec<usize> = a_shape[..a_rank - 2].to_vec();
                let m = a_shape[a_rank - 2];
                let k = a_shape[a_rank - 1];
                let n = b_shape[1];

                if b_shape[0] != k {
                    bail!(
                        "MatMul broadcast dimension mismatch: A[..., {}, {}] x B[{}, {}]",
                        m,
                        k,
                        b_shape[0],
                        n
                    );
                }

                let mut output_shape = batch.clone();
                output_shape.push(m);
                output_shape.push(n);

                Ok(TranslateResult::runtime(
                    OpKind::BatchMatMul { batch, m, k, n },
                    output_shape,
                    a_node.dtype,
                ))
            }

            // 2D x ND: A[M, K] x B[...batch, K, N] -> [...batch, M, N]
            (2, b_rank) if b_rank >= 3 => {
                let batch: Vec<usize> = b_shape[..b_rank - 2].to_vec();
                let m = a_shape[0];
                let k = a_shape[1];
                let n = b_shape[b_rank - 1];

                if b_shape[b_rank - 2] != k {
                    bail!(
                        "MatMul broadcast dimension mismatch: A[{}, {}] x B[..., {}, {}]",
                        m,
                        k,
                        b_shape[b_rank - 2],
                        n
                    );
                }

                let mut output_shape = batch.clone();
                output_shape.push(m);
                output_shape.push(n);

                Ok(TranslateResult::runtime(
                    OpKind::BatchMatMul { batch, m, k, n },
                    output_shape,
                    a_node.dtype,
                ))
            }

            // Broadcast batched matmul with different ranks (neither is 2D)
            _ => {
                bail!("Unsupported MatMul shapes: {:?} x {:?}", a_shape, b_shape);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram::compiler::{DType, OperationGraph};
    use std::collections::HashMap;

    #[test]
    fn test_matmul_2d() {
        let mut graph = OperationGraph::default();
        let mut value_to_node = HashMap::new();

        let a = hologram::compiler::OpNode::new(0, OpKind::Input, vec![3, 4], DType::F32)
            .with_name("a".to_string());
        let b = hologram::compiler::OpNode::new(1, OpKind::Input, vec![4, 5], DType::F32)
            .with_name("b".to_string());

        graph.nodes.push(a);
        graph.nodes.push(b);
        value_to_node.insert("a".to_string(), 0);
        value_to_node.insert("b".to_string(), 1);

        let ctx = TranslateContext::new(&graph, &value_to_node, &[]);

        let node = proto::NodeProto {
            input: vec!["a".to_string(), "b".to_string()],
            output: vec!["out".to_string()],
            op_type: "MatMul".to_string(),
            ..Default::default()
        };

        let result = MatMulOp.translate(&node, &ctx).unwrap();
        assert_eq!(result.shape, vec![3, 5]);
        assert!(matches!(
            result.op_kind,
            OpKind::MatMul { m: 3, k: 4, n: 5 }
        ));
    }

    #[test]
    fn test_batched_matmul() {
        let mut graph = OperationGraph::default();
        let mut value_to_node = HashMap::new();

        let a = hologram::compiler::OpNode::new(0, OpKind::Input, vec![2, 3, 4, 5], DType::F32)
            .with_name("a".to_string());
        let b = hologram::compiler::OpNode::new(1, OpKind::Input, vec![2, 3, 5, 6], DType::F32)
            .with_name("b".to_string());

        graph.nodes.push(a);
        graph.nodes.push(b);
        value_to_node.insert("a".to_string(), 0);
        value_to_node.insert("b".to_string(), 1);

        let ctx = TranslateContext::new(&graph, &value_to_node, &[]);

        let node = proto::NodeProto {
            input: vec!["a".to_string(), "b".to_string()],
            output: vec!["out".to_string()],
            op_type: "MatMul".to_string(),
            ..Default::default()
        };

        let result = MatMulOp.translate(&node, &ctx).unwrap();
        assert_eq!(result.shape, vec![2, 3, 4, 6]);
    }

    #[test]
    fn test_broadcast_3d_2d() {
        // Common transformer pattern: [batch, seq, hidden] x [hidden, out]
        let mut graph = OperationGraph::default();
        let mut value_to_node = HashMap::new();

        let a = hologram::compiler::OpNode::new(0, OpKind::Input, vec![1, 5, 512], DType::F32)
            .with_name("a".to_string());
        let b = hologram::compiler::OpNode::new(1, OpKind::Input, vec![512, 512], DType::F32)
            .with_name("b".to_string());

        graph.nodes.push(a);
        graph.nodes.push(b);
        value_to_node.insert("a".to_string(), 0);
        value_to_node.insert("b".to_string(), 1);

        let ctx = TranslateContext::new(&graph, &value_to_node, &[]);

        let node = proto::NodeProto {
            input: vec!["a".to_string(), "b".to_string()],
            output: vec!["out".to_string()],
            op_type: "MatMul".to_string(),
            ..Default::default()
        };

        let result = MatMulOp.translate(&node, &ctx).unwrap();
        assert_eq!(result.shape, vec![1, 5, 512]);
    }

    #[test]
    fn test_broadcast_4d_2d() {
        // [batch, heads, seq, head_dim] x [head_dim, out]
        let mut graph = OperationGraph::default();
        let mut value_to_node = HashMap::new();

        let a = hologram::compiler::OpNode::new(0, OpKind::Input, vec![2, 8, 16, 64], DType::F32)
            .with_name("a".to_string());
        let b = hologram::compiler::OpNode::new(1, OpKind::Input, vec![64, 128], DType::F32)
            .with_name("b".to_string());

        graph.nodes.push(a);
        graph.nodes.push(b);
        value_to_node.insert("a".to_string(), 0);
        value_to_node.insert("b".to_string(), 1);

        let ctx = TranslateContext::new(&graph, &value_to_node, &[]);

        let node = proto::NodeProto {
            input: vec!["a".to_string(), "b".to_string()],
            output: vec!["out".to_string()],
            op_type: "MatMul".to_string(),
            ..Default::default()
        };

        let result = MatMulOp.translate(&node, &ctx).unwrap();
        assert_eq!(result.shape, vec![2, 8, 16, 128]);
    }
}
