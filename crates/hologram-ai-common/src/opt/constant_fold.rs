//! Constant folding: eliminate identity nodes and remove dead constants.

use super::pipeline::Pass;
use crate::ir::{AiGraph, AiOp, TensorId};
use std::collections::{HashMap, HashSet};

/// Fold identity nodes and remove dead constants.
///
/// 1. Identity elimination: remap output tensors to input tensors.
/// 2. Reshape of constant: remap output to input (shape metadata already set).
/// 3. Dead constant removal: drop params not referenced by any remaining node.
pub struct ConstantFolding;

impl Pass for ConstantFolding {
    fn name(&self) -> &str {
        "ConstantFolding"
    }

    fn run(&self, mut graph: AiGraph) -> anyhow::Result<AiGraph> {
        // Build tensor remap table from identity/reshape-of-constant nodes.
        let mut remap: HashMap<TensorId, TensorId> = HashMap::new();
        let mut removed_node_ids: HashSet<u32> = HashSet::new();

        let param_tids: HashSet<TensorId> = graph.params.keys().copied().collect();

        for node in &graph.nodes {
            match &node.op {
                // Identity: output → input (always foldable).
                AiOp::Identity => {
                    if let (Some(&in_tid), Some(&out_tid)) =
                        (node.inputs.first(), node.outputs.first())
                    {
                        remap.insert(out_tid, resolve(&remap, in_tid));
                        removed_node_ids.insert(node.id);
                    }
                }
                // Reshape of a constant: output → input (shape is metadata-only).
                AiOp::Reshape { .. } => {
                    if let Some(&in_tid) = node.inputs.first() {
                        let resolved = resolve(&remap, in_tid);
                        if param_tids.contains(&resolved) {
                            if let Some(&out_tid) = node.outputs.first() {
                                remap.insert(out_tid, resolved);
                                removed_node_ids.insert(node.id);
                            }
                        }
                    }
                }
                // Materialized constant output: DataPropagation has already
                // computed this node's output and stored it as an AiParam.
                // The node is redundant — remove it so it doesn't get lowered
                // to a runtime op with potentially wrong semantics (e.g.,
                // hologram's FloatOp::Shape returns element count, not dims).
                _ => {
                    if !node.outputs.is_empty()
                        && node.outputs.iter().all(|tid| param_tids.contains(tid))
                    {
                        removed_node_ids.insert(node.id);
                    }
                }
            }
        }

        if removed_node_ids.is_empty() {
            return Ok(graph);
        }

        // Remove folded nodes.
        graph.nodes.retain(|n| !removed_node_ids.contains(&n.id));
        graph.invalidate_topo_cache();

        // Apply remaps to remaining node inputs/outputs.
        for node in &mut graph.nodes {
            for tid in &mut node.inputs {
                *tid = resolve(&remap, *tid);
            }
            for tid in &mut node.outputs {
                *tid = resolve(&remap, *tid);
            }
        }

        // Apply remaps to graph-level inputs/outputs.
        for tid in &mut graph.inputs {
            *tid = resolve(&remap, *tid);
        }
        for tid in &mut graph.outputs {
            *tid = resolve(&remap, *tid);
        }

        // Remove dead constants (params not referenced by any remaining node).
        let live_tids: HashSet<TensorId> = graph
            .nodes
            .iter()
            .flat_map(|n| n.inputs.iter().chain(n.outputs.iter()))
            .chain(graph.inputs.iter())
            .chain(graph.outputs.iter())
            .copied()
            .collect();
        graph.params.retain(|tid, _| live_tids.contains(tid));

        Ok(graph)
    }
}

/// Resolve a tensor ID through the remap chain.
fn resolve(remap: &HashMap<TensorId, TensorId>, mut tid: TensorId) -> TensorId {
    while let Some(&target) = remap.get(&tid) {
        if target == tid {
            break;
        }
        tid = target;
    }
    tid
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{shape_from_concrete, AiGraph, AiNode, AiOp, AiParam, DType, TensorInfo};
    use std::collections::HashMap;

    fn make_graph(
        nodes: Vec<AiNode>,
        inputs: Vec<TensorId>,
        outputs: Vec<TensorId>,
        params: HashMap<TensorId, AiParam>,
        tensor_info: HashMap<TensorId, TensorInfo>,
    ) -> AiGraph {
        AiGraph {
            name: "test".into(),
            nodes,
            inputs,
            outputs,
            input_names: vec![],
            output_names: vec![],
            params,
            tensor_info,
            metadata: HashMap::new(),
            warnings: vec![],
            dim_vars: Default::default(),
            shape_constraints: Default::default(),
            subgraphs: HashMap::new(),
            tensor_names: HashMap::new(),
            topo_cache: Default::default(),
        }
    }

    #[test]
    fn fold_identity_passthrough() {
        // input(0) → Identity → output(1) → Relu → output(2)
        let mut ti = HashMap::new();
        ti.insert(0u32, TensorInfo::new(DType::F32, shape_from_concrete(&[4])));
        ti.insert(1u32, TensorInfo::new(DType::F32, shape_from_concrete(&[4])));
        ti.insert(2u32, TensorInfo::new(DType::F32, shape_from_concrete(&[4])));

        let g = make_graph(
            vec![
                AiNode::new(0, AiOp::Identity, vec![0], vec![1]),
                AiNode::new(1, AiOp::Relu, vec![1], vec![2]),
            ],
            vec![0],
            vec![2],
            HashMap::new(),
            ti,
        );

        let pass = ConstantFolding;
        let g2 = pass.run(g).unwrap();
        // Identity removed, Relu remains. Relu's input remapped 1 → 0.
        assert_eq!(g2.nodes.len(), 1);
        assert!(matches!(g2.nodes[0].op, AiOp::Relu));
        assert_eq!(g2.nodes[0].inputs, vec![0]);
    }

    #[test]
    fn fold_chained_identities() {
        // input(0) → Identity(0→1) → Identity(1→2) → Relu(2→3)
        let mut ti = HashMap::new();
        for i in 0..=3 {
            ti.insert(i, TensorInfo::new(DType::F32, shape_from_concrete(&[4])));
        }

        let g = make_graph(
            vec![
                AiNode::new(0, AiOp::Identity, vec![0], vec![1]),
                AiNode::new(1, AiOp::Identity, vec![1], vec![2]),
                AiNode::new(2, AiOp::Relu, vec![2], vec![3]),
            ],
            vec![0],
            vec![3],
            HashMap::new(),
            ti,
        );

        let g2 = ConstantFolding.run(g).unwrap();
        assert_eq!(g2.nodes.len(), 1);
        assert_eq!(g2.nodes[0].inputs, vec![0]);
    }

    #[test]
    fn fold_reshape_of_constant() {
        // const(10) → Reshape → Relu(output)
        let mut ti = HashMap::new();
        ti.insert(
            10u32,
            TensorInfo::new(DType::F32, shape_from_concrete(&[4])),
        );
        ti.insert(
            11u32,
            TensorInfo::new(DType::F32, shape_from_concrete(&[2, 2])),
        );
        ti.insert(
            12u32,
            TensorInfo::new(DType::F32, shape_from_concrete(&[2, 2])),
        );

        let mut params = HashMap::new();
        params.insert(
            10u32,
            AiParam::inline(
                vec![0u8; 16],
                TensorInfo::new(DType::F32, shape_from_concrete(&[4])),
            ),
        );

        let g = make_graph(
            vec![
                AiNode::new(0, AiOp::Reshape { allow_zero: false }, vec![10], vec![11]),
                AiNode::new(1, AiOp::Relu, vec![11], vec![12]),
            ],
            vec![],
            vec![12],
            params,
            ti,
        );

        let g2 = ConstantFolding.run(g).unwrap();
        // Reshape removed, Relu remains with input remapped 11 → 10.
        assert_eq!(g2.nodes.len(), 1);
        assert_eq!(g2.nodes[0].inputs, vec![10]);
        // Constant 10 is still alive.
        assert!(g2.params.contains_key(&10));
    }

    #[test]
    fn dead_constant_removal() {
        // const(10) is unused — should be removed.
        let mut ti = HashMap::new();
        ti.insert(0u32, TensorInfo::new(DType::F32, shape_from_concrete(&[4])));
        ti.insert(1u32, TensorInfo::new(DType::F32, shape_from_concrete(&[4])));
        ti.insert(
            10u32,
            TensorInfo::new(DType::F32, shape_from_concrete(&[4])),
        );

        let mut params = HashMap::new();
        params.insert(
            10u32,
            AiParam::inline(
                vec![0u8; 16],
                TensorInfo::new(DType::F32, shape_from_concrete(&[4])),
            ),
        );

        // Identity(0→1) eliminates itself, const(10) becomes dead.
        let g = make_graph(
            vec![AiNode::new(0, AiOp::Identity, vec![0], vec![1])],
            vec![0],
            vec![1],
            params,
            ti,
        );

        let g2 = ConstantFolding.run(g).unwrap();
        assert!(g2.params.is_empty());
    }

    #[test]
    fn no_fold_non_constant_reshape() {
        // input(0) → Reshape → output(1) — input is NOT a constant.
        let mut ti = HashMap::new();
        ti.insert(0u32, TensorInfo::new(DType::F32, shape_from_concrete(&[4])));
        ti.insert(
            1u32,
            TensorInfo::new(DType::F32, shape_from_concrete(&[2, 2])),
        );

        let g = make_graph(
            vec![AiNode::new(
                0,
                AiOp::Reshape { allow_zero: false },
                vec![0],
                vec![1],
            )],
            vec![0],
            vec![1],
            HashMap::new(),
            ti,
        );

        let g2 = ConstantFolding.run(g).unwrap();
        // Reshape of non-constant should NOT be folded.
        assert_eq!(g2.nodes.len(), 1);
    }

    #[test]
    fn pass_is_idempotent() {
        let pass = ConstantFolding;

        // First run: identity gets folded.
        let g1 = make_graph(
            vec![AiNode::new(0, AiOp::Identity, vec![0], vec![1])],
            vec![0],
            vec![1],
            HashMap::new(),
            {
                let mut ti = HashMap::new();
                ti.insert(0u32, TensorInfo::new(DType::F32, shape_from_concrete(&[1])));
                ti.insert(1u32, TensorInfo::new(DType::F32, shape_from_concrete(&[1])));
                ti
            },
        );
        let g2 = pass.run(g1).unwrap();
        assert_eq!(g2.nodes.len(), 0);

        // Second run on the already-folded graph: no changes.
        let g3 = pass.run(g2).unwrap();
        assert_eq!(g3.nodes.len(), 0);
    }
}
