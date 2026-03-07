use std::collections::{HashMap, HashSet, VecDeque};
use hologram_ai_quant::QuantDescriptor;
use super::{dtype::DType, shape::Shape, node::{AiNode, TensorId, NodeId}, param::AiParam};

/// Full type + quantization information for a tensor.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Arithmetic dtype (what math "sees").
    pub logical_dtype: DType,
    /// Storage dtype (how the bytes are packed on disk or in memory).
    pub storage_dtype: DType,
    pub shape: Shape,
    pub quant: QuantDescriptor,
}

impl TensorInfo {
    pub fn new(dtype: DType, shape: Shape) -> Self {
        Self {
            logical_dtype: dtype,
            storage_dtype: dtype,
            shape,
            quant: QuantDescriptor::none(),
        }
    }
}

/// Arbitrary key-value metadata attached to an `AiGraph`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum MetaValue {
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Ints(Vec<i64>),
}

/// A non-fatal warning produced during import or optimization.
#[derive(Debug, Clone)]
pub struct ImportWarning {
    pub message: String,
    pub node_name: Option<String>,
}

/// A fatal invariant violation found by `AiGraph::validate()`.
#[derive(Debug, Clone)]
pub struct ValidationError {
    pub message: String,
}

/// The canonical semantic IR graph used by all importers and optimization passes.
///
/// Nodes are stored in topological order (maintained by the importer and opt passes).
pub struct AiGraph {
    pub name: String,
    pub nodes: Vec<AiNode>,
    pub inputs: Vec<TensorId>,
    pub outputs: Vec<TensorId>,
    pub params: HashMap<TensorId, AiParam>,
    pub tensor_info: HashMap<TensorId, TensorInfo>,
    pub metadata: HashMap<String, MetaValue>,
    pub warnings: Vec<ImportWarning>,
}

impl AiGraph {
    /// Validate graph invariants. Returns a (possibly empty) list of errors.
    pub fn validate(&self) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        // All input/output TensorIds must be registered.
        for &tid in &self.inputs {
            if !self.tensor_info.contains_key(&tid) {
                errors.push(ValidationError {
                    message: format!("graph input TensorId {tid} missing from tensor_info"),
                });
            }
        }
        for &tid in &self.outputs {
            if !self.tensor_info.contains_key(&tid) {
                errors.push(ValidationError {
                    message: format!("graph output TensorId {tid} missing from tensor_info"),
                });
            }
        }

        // All node inputs/outputs must be registered.
        for node in &self.nodes {
            for &tid in &node.inputs {
                if !self.tensor_info.contains_key(&tid) {
                    errors.push(ValidationError {
                        message: format!(
                            "node {} input TensorId {tid} missing from tensor_info",
                            node.id
                        ),
                    });
                }
            }
            for &tid in &node.outputs {
                if !self.tensor_info.contains_key(&tid) {
                    errors.push(ValidationError {
                        message: format!(
                            "node {} output TensorId {tid} missing from tensor_info",
                            node.id
                        ),
                    });
                }
            }
        }

        // All param TensorIds must be registered.
        for &tid in self.params.keys() {
            if !self.tensor_info.contains_key(&tid) {
                errors.push(ValidationError {
                    message: format!("param TensorId {tid} missing from tensor_info"),
                });
            }
        }

        // No inline params with empty data.
        for (&tid, param) in &self.params {
            if param.is_empty() {
                errors.push(ValidationError {
                    message: format!("param TensorId {tid} has empty data"),
                });
            }
        }

        // DAG check — Kahn's algorithm.
        if self.has_cycle() {
            errors.push(ValidationError { message: "graph contains a cycle".to_string() });
        }

        errors
    }

    /// Returns nodes in topological order (Kahn's algorithm on NodeIds).
    pub fn topo_order(&self) -> Vec<NodeId> {
        // Build producer map: TensorId → NodeId that produces it.
        let mut producer: HashMap<TensorId, NodeId> = HashMap::new();
        for node in &self.nodes {
            for &tid in &node.outputs {
                producer.insert(tid, node.id);
            }
        }

        // Build adjacency: NodeId → set of NodeIds that depend on it.
        let mut adj: HashMap<NodeId, HashSet<NodeId>> = HashMap::new();
        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
        for node in &self.nodes {
            in_degree.entry(node.id).or_insert(0);
            for &tid in &node.inputs {
                if let Some(&prod) = producer.get(&tid) {
                    adj.entry(prod).or_default().insert(node.id);
                    *in_degree.entry(node.id).or_insert(0) += 1;
                }
            }
        }

        let mut queue: VecDeque<NodeId> = in_degree
            .iter()
            .filter_map(|(&id, &deg)| if deg == 0 { Some(id) } else { None })
            .collect();
        let mut order = Vec::with_capacity(self.nodes.len());

        while let Some(nid) = queue.pop_front() {
            order.push(nid);
            if let Some(succs) = adj.get(&nid) {
                for &succ in succs {
                    let deg = in_degree.get_mut(&succ).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(succ);
                    }
                }
            }
        }

        order
    }

    fn has_cycle(&self) -> bool {
        let order = self.topo_order();
        order.len() != self.nodes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{node::AiNode, op::AiOp, shape::shape_from_concrete};

    fn minimal_graph() -> AiGraph {
        let mut ti = HashMap::new();
        ti.insert(0u32, TensorInfo::new(DType::F32, shape_from_concrete(&[1, 64])));
        ti.insert(1u32, TensorInfo::new(DType::F32, shape_from_concrete(&[1, 64])));

        AiGraph {
            name: "test".into(),
            nodes: vec![AiNode::new(0, AiOp::Identity, vec![0], vec![1])],
            inputs: vec![0],
            outputs: vec![1],
            params: HashMap::new(),
            tensor_info: ti,
            metadata: HashMap::new(),
            warnings: vec![],
        }
    }

    #[test]
    fn validate_clean_graph() {
        let g = minimal_graph();
        assert!(g.validate().is_empty());
    }

    #[test]
    fn topo_order_single_node() {
        let g = minimal_graph();
        assert_eq!(g.topo_order(), vec![0u32]);
    }

    #[test]
    fn validate_missing_tensor_info() {
        let mut g = minimal_graph();
        g.inputs.push(99); // no tensor_info for 99
        let errs = g.validate();
        assert!(!errs.is_empty());
        assert!(errs[0].message.contains("99"));
    }
}
