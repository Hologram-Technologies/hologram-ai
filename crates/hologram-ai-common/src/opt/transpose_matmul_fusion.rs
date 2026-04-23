//! Transpose + MatMul absorption (Plan 054 Wave 2).
//!
//! Detects `Transpose(swap-last-2-dims) → MatMul` and converts to
//! `Gemm { trans_a/trans_b }`, eliminating the Transpose buffer.
//!
//! # Patterns
//!
//! ## Pattern A: Transpose on first input (trans_a)
//! ```text
//! transposed = Transpose(A, perm=[..., -1, -2])
//! out = MatMul(transposed, B)
//! →  out = Gemm(A, B, alpha=1, beta=0, trans_a=true, trans_b=false)
//! ```
//!
//! ## Pattern B: Transpose on second input (trans_b)
//! ```text
//! transposed = Transpose(B, perm=[..., -1, -2])
//! out = MatMul(A, transposed)
//! →  out = Gemm(A, B, alpha=1, beta=0, trans_a=false, trans_b=true)
//! ```
//!
//! This avoids materializing the transposed intermediate buffer.
//! Constraint: Transpose output must have exactly one consumer (the MatMul).

use super::graph_utils::{apply_node_mutations, build_producer_map};
use super::pipeline::Pass;
use crate::ir::{AiGraph, AiNode, AiOp, TensorId};
use std::collections::{HashMap, HashSet};

pub struct TransposeMatMulFusion;

impl Pass for TransposeMatMulFusion {
    fn name(&self) -> &str {
        "TransposeMatMulFusion"
    }

    fn run(&self, mut graph: AiGraph) -> anyhow::Result<AiGraph> {
        let tid_to_node = build_producer_map(&graph);

        let mut consumer_count: HashMap<TensorId, usize> = HashMap::new();
        for n in &graph.nodes {
            for &tid in &n.inputs {
                *consumer_count.entry(tid).or_default() += 1;
            }
        }

        let mut to_remove: HashSet<usize> = HashSet::new();
        let mut replacements: HashMap<usize, AiNode> = HashMap::new();
        let mut fused_count: u32 = 0;

        for (mm_idx, mm_node) in graph.nodes.iter().enumerate() {
            if !matches!(mm_node.op, AiOp::MatMul) || mm_node.inputs.len() < 2 {
                continue;
            }
            if to_remove.contains(&mm_idx) {
                continue;
            }

            let mm_out = match mm_node.outputs.first() {
                Some(&tid) => tid,
                None => continue,
            };

            // Check input A for Transpose.
            let (trans_a, a_tid) =
                check_transpose_input(&graph, &tid_to_node, &consumer_count, mm_node.inputs[0]);
            let (trans_b, b_tid) =
                check_transpose_input(&graph, &tid_to_node, &consumer_count, mm_node.inputs[1]);

            if !trans_a && !trans_b {
                continue;
            }

            // Build Gemm node replacing the MatMul.
            let fused = AiNode::new(
                mm_node.id,
                AiOp::Gemm {
                    alpha: 1.0,
                    beta: 0.0,
                    trans_a,
                    trans_b,
                },
                vec![a_tid, b_tid],
                vec![mm_out],
            );

            // Mark Transpose node(s) for removal.
            if trans_a {
                if let Some(&t_idx) = tid_to_node.get(&mm_node.inputs[0]) {
                    to_remove.insert(t_idx);
                }
            }
            if trans_b {
                if let Some(&t_idx) = tid_to_node.get(&mm_node.inputs[1]) {
                    to_remove.insert(t_idx);
                }
            }
            replacements.insert(mm_idx, fused);
            fused_count += 1;

            tracing::debug!(
                mm_idx,
                trans_a,
                trans_b,
                "TransposeMatMulFusion: absorbed Transpose into Gemm"
            );
        }

        if fused_count > 0 {
            tracing::info!(
                fused_count,
                "TransposeMatMulFusion: absorbed Transpose→MatMul pairs"
            );
        }

        apply_node_mutations(&mut graph, &to_remove, &mut replacements);

        Ok(graph)
    }
}

/// Check if `input_tid` is produced by a Transpose that swaps the last two dims.
/// Returns `(is_transposed, actual_input_tid)`.
/// If transposed, `actual_input_tid` is the pre-transpose tensor.
/// If not, `actual_input_tid` is the original `input_tid`.
fn check_transpose_input(
    graph: &AiGraph,
    tid_to_node: &HashMap<TensorId, usize>,
    consumer_count: &HashMap<TensorId, usize>,
    input_tid: TensorId,
) -> (bool, TensorId) {
    let t_idx = match tid_to_node.get(&input_tid) {
        Some(&idx) => idx,
        None => return (false, input_tid),
    };

    let t_node = &graph.nodes[t_idx];
    let perm: Vec<i64> = match &t_node.op {
        AiOp::Transpose { perm } => perm.iter().map(|&p| p as i64).collect(),
        _ => return (false, input_tid),
    };

    // Transpose output must have exactly one consumer (this MatMul).
    if consumer_count.get(&input_tid).copied().unwrap_or(0) != 1 {
        return (false, input_tid);
    }

    // Check if perm swaps the last two dims and leaves all others in place.
    // E.g., [0, 1, 3, 2] for 4D or [1, 0] for 2D.
    if !is_last_two_swap(&perm) {
        return (false, input_tid);
    }

    let pre_transpose_tid = match t_node.inputs.first() {
        Some(&tid) => tid,
        None => return (false, input_tid),
    };

    (true, pre_transpose_tid)
}

/// Check if a permutation swaps only the last two dimensions.
fn is_last_two_swap(perm: &[i64]) -> bool {
    if perm.len() < 2 {
        return false;
    }
    let n = perm.len();
    // Last two must be swapped.
    if perm[n - 1] != (n as i64 - 2) || perm[n - 2] != (n as i64 - 1) {
        return false;
    }
    // All others must be identity.
    for (i, &p) in perm.iter().enumerate().take(n - 2) {
        if p != i as i64 {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{AiGraph, AiNode, AiOp};

    fn empty_graph() -> AiGraph {
        AiGraph {
            name: "test".to_string(),
            nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            input_names: Vec::new(),
            output_names: Vec::new(),
            params: Default::default(),
            tensor_info: Default::default(),
            metadata: Default::default(),
            warnings: Vec::new(),
            dim_vars: Default::default(),
            shape_constraints: Default::default(),
            subgraphs: Default::default(),
            tensor_names: Default::default(),
            topo_cache: Default::default(),
        }
    }

    #[test]
    fn fuses_transpose_b_matmul() {
        let mut g = empty_graph();
        // A=10, B=11
        // Transpose(B, [1,0]) → B_t=20
        // MatMul(A, B_t) → out=30
        g.inputs = vec![10, 11];
        g.outputs = vec![30];
        g.nodes = vec![
            AiNode::new(
                0,
                AiOp::Transpose {
                    perm: vec![1u32, 0],
                },
                vec![11],
                vec![20],
            ),
            AiNode::new(1, AiOp::MatMul, vec![10, 20], vec![30]),
        ];

        let result = TransposeMatMulFusion.run(g).expect("pass should succeed");
        assert_eq!(result.nodes.len(), 1, "Transpose removed, 1 Gemm remains");
        match &result.nodes[0].op {
            AiOp::Gemm {
                trans_a,
                trans_b,
                alpha,
                beta,
            } => {
                assert!(!trans_a);
                assert!(trans_b);
                assert!((*alpha - 1.0).abs() < f32::EPSILON);
                assert!((*beta - 0.0).abs() < f32::EPSILON);
            }
            other => panic!("expected Gemm, got {other:?}"),
        }
        assert_eq!(result.nodes[0].inputs, vec![10, 11]);
        assert_eq!(result.nodes[0].outputs, vec![30]);
    }

    #[test]
    fn fuses_transpose_a_matmul() {
        let mut g = empty_graph();
        // Transpose(A, [1,0]) → A_t=20
        // MatMul(A_t, B) → out=30
        g.inputs = vec![10, 11];
        g.outputs = vec![30];
        g.nodes = vec![
            AiNode::new(
                0,
                AiOp::Transpose {
                    perm: vec![1u32, 0],
                },
                vec![10],
                vec![20],
            ),
            AiNode::new(1, AiOp::MatMul, vec![20, 11], vec![30]),
        ];

        let result = TransposeMatMulFusion.run(g).expect("pass should succeed");
        assert_eq!(result.nodes.len(), 1);
        match &result.nodes[0].op {
            AiOp::Gemm {
                trans_a, trans_b, ..
            } => {
                assert!(trans_a);
                assert!(!trans_b);
            }
            other => panic!("expected Gemm, got {other:?}"),
        }
        assert_eq!(result.nodes[0].inputs, vec![10, 11]);
    }

    #[test]
    fn fuses_4d_transpose_last_two() {
        let mut g = empty_graph();
        // 4D transpose: [0,1,3,2] swaps last two dims
        g.inputs = vec![10, 11];
        g.outputs = vec![30];
        g.nodes = vec![
            AiNode::new(
                0,
                AiOp::Transpose {
                    perm: vec![0u32, 1, 3, 2],
                },
                vec![11],
                vec![20],
            ),
            AiNode::new(1, AiOp::MatMul, vec![10, 20], vec![30]),
        ];

        let result = TransposeMatMulFusion.run(g).expect("pass should succeed");
        assert_eq!(result.nodes.len(), 1);
        assert!(matches!(
            result.nodes[0].op,
            AiOp::Gemm { trans_b: true, .. }
        ));
    }

    #[test]
    fn skips_non_swap_transpose() {
        let mut g = empty_graph();
        // [2, 0, 1] is NOT a last-two-dim swap
        g.inputs = vec![10, 11];
        g.outputs = vec![30];
        g.nodes = vec![
            AiNode::new(
                0,
                AiOp::Transpose {
                    perm: vec![2u32, 0, 1],
                },
                vec![11],
                vec![20],
            ),
            AiNode::new(1, AiOp::MatMul, vec![10, 20], vec![30]),
        ];

        let result = TransposeMatMulFusion.run(g).expect("pass should succeed");
        assert_eq!(result.nodes.len(), 2, "should not fuse non-swap transpose");
    }

    #[test]
    fn skips_transpose_with_multiple_consumers() {
        let mut g = empty_graph();
        g.inputs = vec![10, 11];
        g.outputs = vec![30, 40];
        g.nodes = vec![
            AiNode::new(
                0,
                AiOp::Transpose {
                    perm: vec![1u32, 0],
                },
                vec![11],
                vec![20],
            ),
            AiNode::new(1, AiOp::MatMul, vec![10, 20], vec![30]),
            AiNode::new(2, AiOp::Add, vec![20, 10], vec![40]),
        ];

        let result = TransposeMatMulFusion.run(g).expect("pass should succeed");
        assert_eq!(
            result.nodes.len(),
            3,
            "should not fuse multi-consumer transpose"
        );
    }
}
