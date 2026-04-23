//! Add + RMSNorm residual fusion.
//!
//! Detects the residual connection pattern and fuses into a single
//! `AiOp::FusedLayerNormResidual` node.
//!
//! # Pattern
//!
//! ```text
//! residual = Add(x, attn_output)
//! normed   = RmsNorm(residual, weight, epsilon)
//! ```
//!
//! Fused into:
//!
//! ```text
//! normed = FusedLayerNormResidual(x, attn_output, weight, epsilon)
//! ```
//!
//! This pattern appears twice per transformer layer (post-attention and
//! post-FFN). Fusing eliminates the intermediate residual buffer.
//!
//! # Prerequisites
//!
//! Requires `RmsNormFusion` to have already run (this pass matches fused
//! `AiOp::RmsNorm` nodes, not the decomposed chain).
//!
//! **Not yet registered in the pipeline** — requires a corresponding
//! `FloatOp::AddRmsNorm` + kernel in hologram base crate. See Plan 019.

use super::graph_utils::{apply_node_mutations, build_consumer_map, build_producer_map};
use super::pipeline::Pass;
use crate::ir::{AiGraph, AiNode, AiOp};
use std::collections::{HashMap, HashSet};

/// Fuse `Add + RmsNorm` into `AiOp::FusedLayerNormResidual`.
pub struct AddRmsNormFusion;

impl Pass for AddRmsNormFusion {
    fn name(&self) -> &str {
        "AddRmsNormFusion"
    }

    fn should_run(&self, graph: &AiGraph) -> bool {
        graph.nodes.iter().any(|n| matches!(n.op, AiOp::RmsNorm { .. }))
    }

    fn run(&self, mut graph: AiGraph) -> anyhow::Result<AiGraph> {
        let tid_to_node = build_producer_map(&graph);
        let consumers = build_consumer_map(&graph);

        let mut to_remove: HashSet<usize> = HashSet::new();
        let mut replacements: HashMap<usize, AiNode> = HashMap::new();
        let mut fused_count: u32 = 0;

        for (norm_idx, norm_node) in graph.nodes.iter().enumerate() {
            // Look for RmsNorm nodes (produced by RmsNormFusion pass).
            let epsilon = match &norm_node.op {
                AiOp::RmsNorm { epsilon } => *epsilon,
                _ => continue,
            };

            if norm_node.inputs.len() < 2 || norm_node.outputs.is_empty() {
                continue;
            }
            if to_remove.contains(&norm_idx) {
                continue;
            }

            let norm_input_tid = norm_node.inputs[0]; // x going into RmsNorm
            let weight_tid = norm_node.inputs[1];
            let norm_out_tid = norm_node.outputs[0];

            // Check if norm_input_tid is produced by an Add node.
            let add_idx = match tid_to_node.get(&norm_input_tid) {
                Some(&idx) => idx,
                None => continue,
            };
            let add_node = &graph.nodes[add_idx];
            if !matches!(add_node.op, AiOp::Add) || add_node.inputs.len() < 2 {
                continue;
            }

            // The Add output (residual) must have exactly one consumer (this RmsNorm).
            // If it's used elsewhere (e.g., as a skip connection output), we can't
            // remove the Add node.
            let add_out_tid = add_node.outputs[0];
            let add_consumers = consumers.get(&add_out_tid).map_or(0, |c| c.len());
            if add_consumers != 1 {
                tracing::trace!(
                    add_idx,
                    add_consumers,
                    "AddRmsNormFusion: Add output has multiple consumers, skipping"
                );
                continue;
            }

            let x_tid = add_node.inputs[0];
            let residual_tid = add_node.inputs[1];

            // Create fused node: inputs = [x, residual, weight], output = normed
            let fused = AiNode::new(
                norm_node.id,
                AiOp::FusedLayerNormResidual { epsilon },
                vec![x_tid, residual_tid, weight_tid],
                vec![norm_out_tid],
            );

            to_remove.insert(add_idx);
            replacements.insert(norm_idx, fused);
            fused_count += 1;

            tracing::debug!(
                add_idx,
                norm_idx,
                x_tid,
                residual_tid,
                weight_tid,
                epsilon,
                "AddRmsNormFusion: fused Add + RmsNorm → FusedLayerNormResidual"
            );
        }

        if fused_count > 0 {
            tracing::info!(fused_count, "AddRmsNormFusion: fused Add+RmsNorm pairs");
        }

        apply_node_mutations(&mut graph, &to_remove, &mut replacements);
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
    fn fuses_add_rmsnorm() {
        let mut g = empty_graph();
        // x=10, attn_out=11, weight=12
        // Add(x, attn_out) → residual=20
        // RmsNorm(residual, weight) → normed=30
        g.inputs = vec![10, 11, 12];
        g.outputs = vec![30];
        g.nodes = vec![
            AiNode::new(0, AiOp::Add, vec![10, 11], vec![20]),
            AiNode::new(1, AiOp::RmsNorm { epsilon: 1e-5 }, vec![20, 12], vec![30]),
        ];

        let result = AddRmsNormFusion.run(g).expect("pass should succeed");
        assert_eq!(result.nodes.len(), 1, "should have 1 fused node");
        assert!(
            matches!(result.nodes[0].op, AiOp::FusedLayerNormResidual { .. }),
            "should be FusedLayerNormResidual"
        );
        assert_eq!(
            result.nodes[0].inputs,
            vec![10, 11, 12],
            "inputs: x, residual, weight"
        );
        assert_eq!(result.nodes[0].outputs, vec![30]);
    }

    #[test]
    fn skips_when_add_has_multiple_consumers() {
        let mut g = empty_graph();
        // Add output used by both RmsNorm and another op
        g.inputs = vec![10, 11, 12];
        g.outputs = vec![30, 40];
        g.nodes = vec![
            AiNode::new(0, AiOp::Add, vec![10, 11], vec![20]),
            AiNode::new(1, AiOp::RmsNorm { epsilon: 1e-5 }, vec![20, 12], vec![30]),
            AiNode::new(2, AiOp::Relu, vec![20], vec![40]),
        ];

        let result = AddRmsNormFusion.run(g).expect("pass should succeed");
        assert_eq!(
            result.nodes.len(),
            3,
            "no fusion when Add has multiple consumers"
        );
    }

    #[test]
    fn skips_when_input_is_not_add() {
        let mut g = empty_graph();
        // RmsNorm input is Mul, not Add
        g.inputs = vec![10, 11, 12];
        g.outputs = vec![30];
        g.nodes = vec![
            AiNode::new(0, AiOp::Mul, vec![10, 11], vec![20]),
            AiNode::new(1, AiOp::RmsNorm { epsilon: 1e-5 }, vec![20, 12], vec![30]),
        ];

        let result = AddRmsNormFusion.run(g).expect("pass should succeed");
        assert_eq!(result.nodes.len(), 2, "no fusion when input is Mul");
    }

    #[test]
    fn fuses_multiple_layers() {
        let mut g = empty_graph();
        // Two transformer layers, each with Add+RmsNorm
        g.inputs = vec![1, 2, 3, 4, 5];
        g.outputs = vec![60];
        g.nodes = vec![
            // Layer 1: Add(x=1, attn=2) → res=10, RmsNorm(res, w=3) → norm=20
            AiNode::new(0, AiOp::Add, vec![1, 2], vec![10]),
            AiNode::new(1, AiOp::RmsNorm { epsilon: 1e-5 }, vec![10, 3], vec![20]),
            // Layer 2: Add(norm=20, ffn=4) → res=30, RmsNorm(res, w=5) → out=60
            AiNode::new(2, AiOp::Add, vec![20, 4], vec![30]),
            AiNode::new(3, AiOp::RmsNorm { epsilon: 1e-6 }, vec![30, 5], vec![60]),
        ];

        let result = AddRmsNormFusion.run(g).expect("pass should succeed");
        assert_eq!(result.nodes.len(), 2, "both layers should be fused");
        assert!(matches!(
            result.nodes[0].op,
            AiOp::FusedLayerNormResidual { .. }
        ));
        assert!(matches!(
            result.nodes[1].op,
            AiOp::FusedLayerNormResidual { .. }
        ));
    }
}
