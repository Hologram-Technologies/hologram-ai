//! SwiGLU + Down Projection fusion (Plan 054).
//!
//! Detects `FusedSwiGLU → MatMul` and fuses into `FusedSwiGluProjection`.
//!
//! # Pattern
//!
//! ```text
//! activated = FusedSwiGLU(gate, up)
//! down_out  = MatMul(activated, W_down)
//! ```
//!
//! Fused into:
//!
//! ```text
//! down_out = FusedSwiGluProjection(gate, up, W_down)
//! ```
//!
//! The fused kernel computes `silu(gate) * up` in-register during the GEMV
//! inner loop, never materializing the intermediate activation buffer.
//! Constraint: FusedSwiGLU output must have exactly one consumer (the MatMul).

use super::pipeline::Pass;
use crate::ir::{AiGraph, AiNode, AiOp, TensorId};
use std::collections::{HashMap, HashSet};

pub struct SwiGluProjectionFusion;

impl Pass for SwiGluProjectionFusion {
    fn name(&self) -> &str {
        "SwiGluProjectionFusion"
    }

    fn run(&self, mut graph: AiGraph) -> anyhow::Result<AiGraph> {
        // Map: tensor_id → node index that produces it.
        let tid_to_node: HashMap<TensorId, usize> = graph
            .nodes
            .iter()
            .enumerate()
            .flat_map(|(i, n)| n.outputs.iter().map(move |&tid| (tid, i)))
            .collect();

        // Map: tensor_id → number of consumers.
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
            // Look for MatMul nodes.
            if !matches!(mm_node.op, AiOp::MatMul) || mm_node.inputs.len() < 2 {
                continue;
            }
            if to_remove.contains(&mm_idx) {
                continue;
            }

            let mm_input = mm_node.inputs[0]; // activation input
            let mm_weight = mm_node.inputs[1]; // W_down

            // Check if the activation input comes from FusedSwiGLU.
            let swiglu_idx = match tid_to_node.get(&mm_input) {
                Some(&idx) => idx,
                None => continue,
            };
            if !matches!(graph.nodes[swiglu_idx].op, AiOp::FusedSwiGLU) {
                continue;
            }

            // FusedSwiGLU output must have exactly one consumer (this MatMul).
            let swiglu_out = graph.nodes[swiglu_idx].outputs[0];
            if consumer_count.get(&swiglu_out).copied().unwrap_or(0) != 1 {
                continue;
            }

            let gate_tid = graph.nodes[swiglu_idx].inputs[0];
            let up_tid = graph.nodes[swiglu_idx].inputs[1];
            let mm_out_tid = match mm_node.outputs.first() {
                Some(&tid) => tid,
                None => continue,
            };

            // Create fused node reusing the MatMul node's id and output.
            let fused = AiNode::new(
                mm_node.id,
                AiOp::FusedSwiGluProjection,
                vec![gate_tid, up_tid, mm_weight],
                vec![mm_out_tid],
            );

            to_remove.insert(swiglu_idx);
            replacements.insert(mm_idx, fused);
            fused_count += 1;

            tracing::debug!(
                swiglu_idx,
                mm_idx,
                gate_tid,
                up_tid,
                mm_out_tid,
                "SwiGluProjectionFusion: fused FusedSwiGLU + MatMul → FusedSwiGluProjection"
            );
        }

        if fused_count > 0 {
            tracing::info!(
                fused_count,
                "SwiGluProjectionFusion: fused SwiGLU+MatMul pairs"
            );
        }

        // Apply replacements and removals.
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

        Ok(graph)
    }
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
    fn fuses_swiglu_matmul() {
        let mut g = empty_graph();
        // gate=10, up=11, W_down=12
        // FusedSwiGLU(gate, up) → activated=20
        // MatMul(activated, W_down) → out=30
        g.inputs = vec![10, 11, 12];
        g.outputs = vec![30];
        g.nodes = vec![
            AiNode::new(0, AiOp::FusedSwiGLU, vec![10, 11], vec![20]),
            AiNode::new(1, AiOp::MatMul, vec![20, 12], vec![30]),
        ];

        let result = SwiGluProjectionFusion.run(g).expect("pass should succeed");
        assert_eq!(result.nodes.len(), 1, "should have 1 fused node");
        assert!(
            matches!(result.nodes[0].op, AiOp::FusedSwiGluProjection),
            "should be FusedSwiGluProjection"
        );
        assert_eq!(
            result.nodes[0].inputs,
            vec![10, 11, 12],
            "inputs: gate, up, W_down"
        );
        assert_eq!(result.nodes[0].outputs, vec![30], "output preserved");
    }

    #[test]
    fn skips_when_swiglu_has_multiple_consumers() {
        let mut g = empty_graph();
        // FusedSwiGLU output used by both MatMul and Add
        g.inputs = vec![10, 11, 12];
        g.outputs = vec![30, 40];
        g.nodes = vec![
            AiNode::new(0, AiOp::FusedSwiGLU, vec![10, 11], vec![20]),
            AiNode::new(1, AiOp::MatMul, vec![20, 12], vec![30]),
            AiNode::new(2, AiOp::Add, vec![20, 12], vec![40]),
        ];

        let result = SwiGluProjectionFusion.run(g).expect("pass should succeed");
        assert_eq!(result.nodes.len(), 3, "nothing should be fused");
        assert!(matches!(result.nodes[0].op, AiOp::FusedSwiGLU));
    }

    #[test]
    fn skips_matmul_without_swiglu_input() {
        let mut g = empty_graph();
        g.inputs = vec![10, 11];
        g.outputs = vec![30];
        g.nodes = vec![AiNode::new(0, AiOp::MatMul, vec![10, 11], vec![30])];

        let result = SwiGluProjectionFusion.run(g).expect("pass should succeed");
        assert_eq!(result.nodes.len(), 1);
        assert!(matches!(result.nodes[0].op, AiOp::MatMul));
    }
}
