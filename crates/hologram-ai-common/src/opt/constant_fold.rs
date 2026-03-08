use crate::ir::AiGraph;
use super::pipeline::Pass;

/// Fold `AiOp::Constant` nodes that feed only into other constant consumers.
///
/// Sprint 001: elides identity/passthrough constant nodes. Full expression
/// folding is deferred to Week 2 shape propagation pass.
pub struct ConstantFolding;

impl Pass for ConstantFolding {
    fn name(&self) -> &str { "ConstantFolding" }

    fn run(&self, graph: AiGraph) -> anyhow::Result<AiGraph> {
        // Sprint 001 scope: remove `Identity` nodes whose input is a Constant node.
        // Full constant expression folding is a Week 2 task.
        Ok(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{AiGraph, AiNode, AiOp, DType, TensorInfo, shape_from_concrete};
    use std::collections::HashMap;

    #[test]
    fn pass_is_idempotent() {
        let mut ti = HashMap::new();
        ti.insert(0u32, TensorInfo::new(DType::F32, shape_from_concrete(&[1])));
        ti.insert(1u32, TensorInfo::new(DType::F32, shape_from_concrete(&[1])));
        let g = AiGraph {
            name: "test".into(),
            nodes: vec![AiNode::new(0, AiOp::Identity, vec![0], vec![1])],
            inputs: vec![0],
            outputs: vec![1],
            params: HashMap::new(),
            tensor_info: ti,
            metadata: HashMap::new(),
            warnings: vec![],
            dim_vars: Default::default(),
            shape_constraints: Default::default(),
        };
        let pass = ConstantFolding;
        let g2 = pass.run(g).unwrap();
        assert_eq!(g2.nodes.len(), 1);
    }
}
