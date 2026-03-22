//! SwiGLU activation fusion.
//!
//! Detects the decomposed SwiGLU pattern and fuses into a single
//! `AiOp::FusedSwiGLU` node.
//!
//! # Pattern
//!
//! ```text
//! silu_out = SiLU(gate)
//! out      = Mul(silu_out, up)     — or Mul(up, silu_out)
//! ```
//!
//! Fused into:
//!
//! ```text
//! out = FusedSwiGLU(gate, up)      — silu(gate) * up
//! ```
//!
//! This pattern appears in every transformer layer using the SwiGLU FFN
//! (LLaMA, Qwen, Mistral, Gemma, etc.). Fusing eliminates the intermediate
//! SiLU buffer and one dispatch.

use super::pipeline::Pass;
use crate::ir::{AiGraph, AiNode, AiOp, TensorId};
use std::collections::{HashMap, HashSet};

/// Fuse `SiLU + Mul` into `AiOp::FusedSwiGLU`.
pub struct SwiGluFusion;

impl Pass for SwiGluFusion {
    fn name(&self) -> &str {
        "SwiGluFusion"
    }

    fn run(&self, mut graph: AiGraph) -> anyhow::Result<AiGraph> {
        // Map: tensor_id → node index that produces it.
        let tid_to_node: HashMap<TensorId, usize> = graph
            .nodes
            .iter()
            .enumerate()
            .flat_map(|(i, n)| n.outputs.iter().map(move |&tid| (tid, i)))
            .collect();

        // Map: tensor_id → list of (consuming_node_idx, input_position).
        let mut consumers: HashMap<TensorId, Vec<(usize, usize)>> = HashMap::new();
        for (i, n) in graph.nodes.iter().enumerate() {
            for (pos, &tid) in n.inputs.iter().enumerate() {
                consumers.entry(tid).or_default().push((i, pos));
            }
        }

        let mut to_remove: HashSet<usize> = HashSet::new();
        let mut replacements: HashMap<usize, AiNode> = HashMap::new();
        let mut fused_count: u32 = 0;

        for (mul_idx, mul_node) in graph.nodes.iter().enumerate() {
            // Look for Mul nodes.
            if !matches!(mul_node.op, AiOp::Mul) || mul_node.inputs.len() < 2 {
                continue;
            }
            if to_remove.contains(&mul_idx) {
                continue;
            }

            let mul_in_a = mul_node.inputs[0];
            let mul_in_b = mul_node.inputs[1];

            // Check if either input comes from a SiLU node.
            let (silu_idx, gate_tid, up_tid) =
                match try_find_silu(&tid_to_node, &graph, mul_in_a, mul_in_b) {
                    Some(result) => result,
                    None => continue,
                };

            // SiLU output must have exactly one consumer (this Mul).
            // If SiLU output is used elsewhere, we can't remove the SiLU node.
            let silu_out_tid = graph.nodes[silu_idx].outputs[0];
            let silu_consumers = consumers.get(&silu_out_tid).map_or(0, |c| c.len());
            if silu_consumers != 1 {
                tracing::trace!(
                    silu_idx,
                    silu_consumers,
                    "SwiGluFusion: SiLU output has multiple consumers, skipping"
                );
                continue;
            }

            let mul_out_tid = match mul_node.outputs.first() {
                Some(&tid) => tid,
                None => continue,
            };

            // Create fused node reusing the Mul node's id and output tensor.
            let fused = AiNode::new(
                mul_node.id,
                AiOp::FusedSwiGLU,
                vec![gate_tid, up_tid],
                vec![mul_out_tid],
            );

            // Propagate tensor info from the Mul output to the fused node's output.
            // (The Mul output TensorId is reused, so existing info carries over.)

            to_remove.insert(silu_idx);
            replacements.insert(mul_idx, fused);
            fused_count += 1;

            tracing::debug!(
                silu_idx,
                mul_idx,
                gate_tid,
                up_tid,
                mul_out_tid,
                "SwiGluFusion: fused SiLU + Mul → FusedSwiGLU"
            );
        }

        if fused_count > 0 {
            tracing::info!(fused_count, "SwiGluFusion: fused SiLU+Mul pairs");
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

/// Check if either `a` or `b` is produced by a `SiLU` node.
/// Returns `(silu_node_idx, gate_tid, up_tid)` where `gate_tid` is the
/// input to SiLU and `up_tid` is the other Mul operand.
fn try_find_silu(
    tid_to_node: &HashMap<TensorId, usize>,
    graph: &AiGraph,
    a: TensorId,
    b: TensorId,
) -> Option<(usize, TensorId, TensorId)> {
    // Try a = SiLU output, b = up
    if let Some(&silu_idx) = tid_to_node.get(&a) {
        let silu_node = &graph.nodes[silu_idx];
        if matches!(silu_node.op, AiOp::Silu) && !silu_node.inputs.is_empty() {
            return Some((silu_idx, silu_node.inputs[0], b));
        }
    }
    // Try b = SiLU output, a = up
    if let Some(&silu_idx) = tid_to_node.get(&b) {
        let silu_node = &graph.nodes[silu_idx];
        if matches!(silu_node.op, AiOp::Silu) && !silu_node.inputs.is_empty() {
            return Some((silu_idx, silu_node.inputs[0], a));
        }
    }
    None
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
    fn fuses_silu_mul_into_swiglu() {
        let mut g = empty_graph();
        // gate_tid=10, up_tid=11
        // SiLU(gate) → silu_out=20
        // Mul(silu_out, up) → out=30
        g.inputs = vec![10, 11];
        g.outputs = vec![30];
        g.nodes = vec![
            AiNode::new(0, AiOp::Silu, vec![10], vec![20]),
            AiNode::new(1, AiOp::Mul, vec![20, 11], vec![30]),
        ];

        let result = SwiGluFusion.run(g).expect("pass should succeed");
        assert_eq!(result.nodes.len(), 1, "should have 1 fused node");
        assert!(
            matches!(result.nodes[0].op, AiOp::FusedSwiGLU),
            "should be FusedSwiGLU"
        );
        assert_eq!(result.nodes[0].inputs, vec![10, 11], "inputs: gate, up");
        assert_eq!(result.nodes[0].outputs, vec![30], "output preserved");
    }

    #[test]
    fn fuses_with_swapped_mul_operands() {
        let mut g = empty_graph();
        // Mul(up, silu_out) — operands swapped
        g.inputs = vec![10, 11];
        g.outputs = vec![30];
        g.nodes = vec![
            AiNode::new(0, AiOp::Silu, vec![10], vec![20]),
            AiNode::new(1, AiOp::Mul, vec![11, 20], vec![30]),
        ];

        let result = SwiGluFusion.run(g).expect("pass should succeed");
        assert_eq!(result.nodes.len(), 1);
        assert!(matches!(result.nodes[0].op, AiOp::FusedSwiGLU));
        assert_eq!(result.nodes[0].inputs, vec![10, 11]);
    }

    #[test]
    fn skips_when_silu_has_multiple_consumers() {
        let mut g = empty_graph();
        // SiLU output used by both Mul and Add — can't remove SiLU
        g.inputs = vec![10, 11];
        g.outputs = vec![30, 40];
        g.nodes = vec![
            AiNode::new(0, AiOp::Silu, vec![10], vec![20]),
            AiNode::new(1, AiOp::Mul, vec![20, 11], vec![30]),
            AiNode::new(2, AiOp::Add, vec![20, 11], vec![40]),
        ];

        let result = SwiGluFusion.run(g).expect("pass should succeed");
        // Nothing should be fused
        assert_eq!(result.nodes.len(), 3);
        assert!(matches!(result.nodes[0].op, AiOp::Silu));
        assert!(matches!(result.nodes[1].op, AiOp::Mul));
    }

    #[test]
    fn no_fusion_without_silu() {
        let mut g = empty_graph();
        g.inputs = vec![10, 11];
        g.outputs = vec![30];
        g.nodes = vec![AiNode::new(0, AiOp::Mul, vec![10, 11], vec![30])];

        let result = SwiGluFusion.run(g).expect("pass should succeed");
        assert_eq!(result.nodes.len(), 1);
        assert!(matches!(result.nodes[0].op, AiOp::Mul));
    }
}
