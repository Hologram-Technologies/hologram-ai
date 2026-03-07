use super::op::AiOp;

/// Stable identifier for a tensor flowing through the graph.
pub type TensorId = u32;

/// Stable identifier for a node in the graph.
pub type NodeId = u32;

/// A single operation node in `AiGraph`.
#[derive(Debug, Clone)]
pub struct AiNode {
    pub id: NodeId,
    pub op: AiOp,
    pub inputs: Vec<TensorId>,
    pub outputs: Vec<TensorId>,
}

impl AiNode {
    pub fn new(id: NodeId, op: AiOp, inputs: Vec<TensorId>, outputs: Vec<TensorId>) -> Self {
        Self { id, op, inputs, outputs }
    }
}
