//! Scalar broadcast absorption (Plan 054 Wave 2).
//!
//! Detects `MatMul → Mul(scalar)` and folds the scalar into `Gemm { alpha }`,
//! eliminating a full-tensor scalar multiply.
//!
//! # Pattern
//! ```text
//! mm_out = MatMul(A, B)
//! scaled = Mul(mm_out, scalar_constant)
//! →  scaled = Gemm(A, B, alpha=scalar, beta=0, trans_a=false, trans_b=false)
//! ```
//!
//! Also handles `Mul(scalar_constant, mm_out)` (commutative).
//!
//! Constraint: MatMul output must have exactly one consumer (the Mul).

use super::pipeline::Pass;
use crate::ir::{AiGraph, AiNode, AiOp, AiParam, TensorId};
use std::collections::{HashMap, HashSet};

pub struct ScalarAbsorption;

impl Pass for ScalarAbsorption {
    fn name(&self) -> &str {
        "ScalarAbsorption"
    }

    fn run(&self, mut graph: AiGraph) -> anyhow::Result<AiGraph> {
        let tid_to_node: HashMap<TensorId, usize> = graph
            .nodes
            .iter()
            .enumerate()
            .flat_map(|(i, n)| n.outputs.iter().map(move |&tid| (tid, i)))
            .collect();

        let mut consumer_count: HashMap<TensorId, usize> = HashMap::new();
        for n in &graph.nodes {
            for &tid in &n.inputs {
                *consumer_count.entry(tid).or_default() += 1;
            }
        }

        let mut to_remove: HashSet<usize> = HashSet::new();
        let mut replacements: HashMap<usize, AiNode> = HashMap::new();
        let mut fused_count: u32 = 0;

        for (mul_idx, mul_node) in graph.nodes.iter().enumerate() {
            if !matches!(mul_node.op, AiOp::Mul) || mul_node.inputs.len() < 2 {
                continue;
            }
            if to_remove.contains(&mul_idx) {
                continue;
            }

            let mul_out = match mul_node.outputs.first() {
                Some(&tid) => tid,
                None => continue,
            };

            // Try both operand orderings: Mul(mm_out, scalar) or Mul(scalar, mm_out).
            let result = try_match_matmul_scalar(
                &graph,
                &tid_to_node,
                &consumer_count,
                mul_node.inputs[0],
                mul_node.inputs[1],
            )
            .or_else(|| {
                try_match_matmul_scalar(
                    &graph,
                    &tid_to_node,
                    &consumer_count,
                    mul_node.inputs[1],
                    mul_node.inputs[0],
                )
            });

            let (mm_idx, a_tid, b_tid, scalar) = match result {
                Some(r) => r,
                None => continue,
            };

            // Also absorb if the MatMul is already a Gemm.
            let (base_alpha, base_beta, trans_a, trans_b) =
                match &graph.nodes[mm_idx].op {
                    AiOp::MatMul => (1.0_f32, 0.0_f32, false, false),
                    AiOp::Gemm {
                        alpha,
                        beta,
                        trans_a,
                        trans_b,
                    } => (*alpha, *beta, *trans_a, *trans_b),
                    _ => continue,
                };

            let fused = AiNode::new(
                mul_node.id,
                AiOp::Gemm {
                    alpha: base_alpha * scalar,
                    beta: base_beta,
                    trans_a,
                    trans_b,
                },
                vec![a_tid, b_tid],
                vec![mul_out],
            );

            to_remove.insert(mm_idx);
            replacements.insert(mul_idx, fused);
            fused_count += 1;

            tracing::debug!(
                mm_idx,
                mul_idx,
                scalar,
                "ScalarAbsorption: folded Mul(scalar) into Gemm alpha"
            );
        }

        if fused_count > 0 {
            tracing::info!(
                fused_count,
                "ScalarAbsorption: absorbed scalar Mul into Gemm alpha"
            );
        }

        if !to_remove.is_empty() || !replacements.is_empty() {
            let mut new_nodes = Vec::with_capacity(graph.nodes.len() - to_remove.len());
            for (i, node) in graph.nodes.into_iter().enumerate() {
                if to_remove.contains(&i) {
                    continue;
                }
                if let Some(fused) = replacements.remove(&i) {
                    new_nodes.push(fused);
                } else {
                    new_nodes.push(node);
                }
            }
            graph.nodes = new_nodes;
            graph.invalidate_topo_cache();
        }

        Ok(graph)
    }
}

/// Try to match `mm_tid` as a MatMul output and `scalar_tid` as a scalar constant.
/// Returns `(mm_node_idx, a_tid, b_tid, scalar_value)` if matched.
fn try_match_matmul_scalar(
    graph: &AiGraph,
    tid_to_node: &HashMap<TensorId, usize>,
    consumer_count: &HashMap<TensorId, usize>,
    mm_tid: TensorId,
    scalar_tid: TensorId,
) -> Option<(usize, TensorId, TensorId, f32)> {
    // Check if mm_tid comes from a MatMul or Gemm.
    let mm_idx = *tid_to_node.get(&mm_tid)?;
    let mm_node = &graph.nodes[mm_idx];
    if !matches!(mm_node.op, AiOp::MatMul | AiOp::Gemm { .. }) || mm_node.inputs.len() < 2 {
        return None;
    }

    // MatMul output must have exactly one consumer (this Mul).
    if consumer_count.get(&mm_tid).copied().unwrap_or(0) != 1 {
        return None;
    }

    // Check if scalar_tid is a scalar constant (1 element).
    let scalar = scalar_f32_value(graph, scalar_tid)?;

    Some((mm_idx, mm_node.inputs[0], mm_node.inputs[1], scalar))
}

/// Extract a scalar f32 from a constant parameter or tensor info.
fn scalar_f32_value(graph: &AiGraph, tid: TensorId) -> Option<f32> {
    // Check inline parameter.
    if let Some(AiParam::Inline { data, info }) = graph.params.get(&tid) {
        let n_elems: u64 = info
            .shape
            .iter()
            .filter_map(|d| d.as_concrete())
            .product();
        if n_elems == 1 && data.len() == 4 {
            let bytes: [u8; 4] = [data[0], data[1], data[2], data[3]];
            return Some(f32::from_le_bytes(bytes));
        }
    }

    // Check known_i64_values (sometimes scalars are stored as i64 via Shape/Gather).
    // Not common for f32 scalars, skip for now.
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{AiGraph, AiNode, AiOp, AiParam, DType, SemanticHint, TensorInfo};

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

    fn scalar_param(val: f32) -> (AiParam, TensorInfo) {
        let data = val.to_le_bytes().to_vec();
        let info = TensorInfo {
            shape: crate::ir::shape_from_concrete(&[1]),
            logical_dtype: DType::F32,
            storage_dtype: DType::F32,
            quant: hologram_ai_quant::QuantDescriptor::none(),
            known_i64_values: None,
            semantic: SemanticHint::Unknown,
        };
        (AiParam::Inline { data, info: info.clone() }, info)
    }

    #[test]
    fn absorbs_matmul_mul_scalar() {
        let mut g = empty_graph();
        // A=10, B=11, scale=12 (scalar 0.125)
        // MatMul(A, B) → mm_out=20
        // Mul(mm_out, scale) → out=30
        g.inputs = vec![10, 11];
        g.outputs = vec![30];

        let (param, info) = scalar_param(0.125);
        g.params.insert(12, param);
        g.tensor_info.insert(12, info);

        g.nodes = vec![
            AiNode::new(0, AiOp::MatMul, vec![10, 11], vec![20]),
            AiNode::new(1, AiOp::Mul, vec![20, 12], vec![30]),
        ];

        let result = ScalarAbsorption.run(g).expect("pass should succeed");
        assert_eq!(result.nodes.len(), 1, "MatMul+Mul collapsed to 1 Gemm");
        match &result.nodes[0].op {
            AiOp::Gemm { alpha, beta, trans_a, trans_b } => {
                assert!((*alpha - 0.125).abs() < f32::EPSILON, "alpha should be 0.125");
                assert!((*beta - 0.0).abs() < f32::EPSILON);
                assert!(!trans_a);
                assert!(!trans_b);
            }
            other => panic!("expected Gemm, got {other:?}"),
        }
        assert_eq!(result.nodes[0].inputs, vec![10, 11]);
    }

    #[test]
    fn absorbs_swapped_mul_operands() {
        let mut g = empty_graph();
        // Mul(scale, mm_out) — scalar on the left
        g.inputs = vec![10, 11];
        g.outputs = vec![30];

        let (param, info) = scalar_param(2.0);
        g.params.insert(12, param);
        g.tensor_info.insert(12, info);

        g.nodes = vec![
            AiNode::new(0, AiOp::MatMul, vec![10, 11], vec![20]),
            AiNode::new(1, AiOp::Mul, vec![12, 20], vec![30]),
        ];

        let result = ScalarAbsorption.run(g).expect("pass should succeed");
        assert_eq!(result.nodes.len(), 1);
        match &result.nodes[0].op {
            AiOp::Gemm { alpha, .. } => {
                assert!((*alpha - 2.0).abs() < f32::EPSILON);
            }
            other => panic!("expected Gemm, got {other:?}"),
        }
    }

    #[test]
    fn skips_non_scalar_mul() {
        let mut g = empty_graph();
        // Mul with non-scalar (vector) — should not absorb.
        g.inputs = vec![10, 11, 12];
        g.outputs = vec![30];
        g.nodes = vec![
            AiNode::new(0, AiOp::MatMul, vec![10, 11], vec![20]),
            AiNode::new(1, AiOp::Mul, vec![20, 12], vec![30]),
        ];
        // No param for tid 12 → not a scalar constant.

        let result = ScalarAbsorption.run(g).expect("pass should succeed");
        assert_eq!(result.nodes.len(), 2, "should not fuse non-scalar Mul");
    }
}
