use super::pipeline::Pass;
use crate::ir::{AiGraph, TensorId};
use std::collections::HashSet;

/// Remove nodes whose outputs are not reachable from any graph output.
pub struct DeadNodeElimination;

impl Pass for DeadNodeElimination {
    fn name(&self) -> &str {
        "DeadNodeElimination"
    }

    fn run(&self, graph: AiGraph) -> anyhow::Result<AiGraph> {
        let live = live_tensors(&graph);

        let nodes: Vec<_> = graph
            .nodes
            .into_iter()
            .filter(|n| n.outputs.iter().any(|tid| live.contains(tid)))
            .collect();

        Ok(AiGraph {
            name: graph.name,
            nodes,
            inputs: graph.inputs,
            outputs: graph.outputs,
            input_names: graph.input_names,
            output_names: graph.output_names,
            params: graph.params,
            tensor_info: graph.tensor_info,
            metadata: graph.metadata,
            warnings: graph.warnings,
            dim_vars: graph.dim_vars,
            shape_constraints: graph.shape_constraints,
            subgraphs: graph.subgraphs,
            tensor_names: graph.tensor_names,
        })
    }
}

/// Backward reachability from graph outputs.
fn live_tensors(graph: &AiGraph) -> HashSet<TensorId> {
    // Build a map from each tensor to the node that produces it.
    let mut produced_by = std::collections::HashMap::<TensorId, u32>::new();
    for node in &graph.nodes {
        for &tid in &node.outputs {
            produced_by.insert(tid, node.id);
        }
    }

    // Build a map from NodeId → node so we can look up inputs.
    let node_map: std::collections::HashMap<u32, &crate::ir::AiNode> =
        graph.nodes.iter().map(|n| (n.id, n)).collect();

    let mut live: HashSet<TensorId> = graph.outputs.iter().copied().collect();
    let mut worklist: Vec<TensorId> = graph.outputs.clone();

    while let Some(tid) = worklist.pop() {
        if let Some(&nid) = produced_by.get(&tid) {
            if let Some(node) = node_map.get(&nid) {
                for &inp in &node.inputs {
                    if live.insert(inp) {
                        worklist.push(inp);
                    }
                }
            }
        }
    }

    live
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{shape_from_concrete, AiGraph, AiNode, AiOp, DType, TensorInfo};
    use std::collections::HashMap;

    fn two_node_graph() -> AiGraph {
        let mut ti = HashMap::new();
        for i in 0u32..=3 {
            ti.insert(i, TensorInfo::new(DType::F32, shape_from_concrete(&[1])));
        }
        AiGraph {
            name: "test".into(),
            nodes: vec![
                AiNode::new(0, AiOp::Identity, vec![0], vec![1]), // live
                AiNode::new(1, AiOp::Identity, vec![0], vec![2]), // dead — output is 2, not used
            ],
            inputs: vec![0],
            outputs: vec![1],
            input_names: vec![],
            output_names: vec![],
            params: HashMap::new(),
            tensor_info: ti,
            metadata: HashMap::new(),
            warnings: vec![],
            dim_vars: Default::default(),
            shape_constraints: Default::default(),
            subgraphs: HashMap::new(),
            tensor_names: HashMap::new(),
        }
    }

    #[test]
    fn removes_dead_node() {
        let g = two_node_graph();
        let pass = DeadNodeElimination;
        let g2 = pass.run(g).unwrap();
        assert_eq!(g2.nodes.len(), 1);
        assert_eq!(g2.nodes[0].id, 0);
    }

    #[test]
    fn idempotent_on_clean_graph() {
        let g = two_node_graph();
        let pass = DeadNodeElimination;
        let g2 = pass.run(g).unwrap();
        let g3 = pass.run(g2).unwrap();
        assert_eq!(g3.nodes.len(), 1);
    }
}
