use super::{
    dtype::DType,
    node::{AiNode, NodeId, TensorId},
    param::AiParam,
    shape::Shape,
    shape::{ConstraintStore, DimVarTable},
};
use hologram_ai_quant::QuantDescriptor;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::{Arc, Mutex};

/// Semantic classification of tensor content.
///
/// Tracks what kind of information a tensor represents, enabling
/// precision analysis and validation. The `natural_bits` method returns
/// the estimated information content per the thermodynamic framework
/// (Landauer's principle applied to neural network precision).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SemanticHint {
    #[default]
    Unknown,
    /// Raw pixel data (RGB/RGBA). ~24 bits natural precision.
    Pixel,
    /// Latent space representation (VAE output). ~4 bits.
    Latent,
    /// Token IDs. log2(vocab_size) bits.
    Token,
    /// Dense embedding vectors. ~12 bits.
    Embedding,
    /// Attention scores (post-softmax). ~8 bits.
    AttentionWeight,
    /// Pre-softmax logits. ~12 bits.
    Logit,
    /// Residual stream (accumulates across layers). High precision needed.
    Residual,
    /// Normalization output (RmsNorm/LayerNorm). ~12 bits.
    NormOutput,
    /// Positional encoding (RoPE, sinusoidal). ~8 bits.
    Position,
}

impl SemanticHint {
    /// Estimated natural precision in bits.
    /// Returns `None` for `Unknown`.
    pub fn natural_bits(&self) -> Option<u8> {
        match self {
            Self::Unknown => None,
            Self::Pixel => Some(24),
            Self::Latent => Some(4),
            Self::Token => Some(16),
            Self::Embedding => Some(12),
            Self::AttentionWeight => Some(8),
            Self::Logit => Some(12),
            Self::Residual => Some(16),
            Self::NormOutput => Some(12),
            Self::Position => Some(8),
        }
    }
}

/// Full type + quantization information for a tensor.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Arithmetic dtype (what math "sees").
    pub logical_dtype: DType,
    /// Storage dtype (how the bytes are packed on disk or in memory).
    pub storage_dtype: DType,
    pub shape: Shape,
    pub quant: QuantDescriptor,
    /// Known constant values for small integer tensors (shape computation).
    /// `Some(v)` = concrete value, `None` = dynamic/symbolic.
    /// Populated by the `DataPropagation` pass.
    pub known_i64_values: Option<Vec<Option<i64>>>,
    /// Semantic classification of this tensor's content.
    /// Populated by importers and optimization passes.
    pub semantic: SemanticHint,
}

impl TensorInfo {
    pub fn new(dtype: DType, shape: Shape) -> Self {
        Self {
            logical_dtype: dtype,
            storage_dtype: dtype,
            shape,
            quant: QuantDescriptor::none(),
            known_i64_values: None,
            semantic: SemanticHint::Unknown,
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
///
/// `Clone` is cheap: most weight data uses `AiParam::Mmap` (copies a path +
/// offset, not the bytes). Only small inline constants are deep-copied.
/// Manually implemented because `Mutex` (topo_cache) doesn't derive `Clone`.
pub struct AiGraph {
    pub name: String,
    pub nodes: Vec<AiNode>,
    pub inputs: Vec<TensorId>,
    pub outputs: Vec<TensorId>,
    /// Human-readable names for graph inputs (parallel to `inputs`).
    /// Falls back to `input_0`, `input_1`, … when empty or shorter.
    pub input_names: Vec<String>,
    /// Human-readable names for graph outputs (parallel to `outputs`).
    /// Falls back to `output_0`, `output_1`, … when empty or shorter.
    pub output_names: Vec<String>,
    pub params: HashMap<TensorId, AiParam>,
    pub tensor_info: HashMap<TensorId, TensorInfo>,
    pub metadata: HashMap<String, MetaValue>,
    pub warnings: Vec<ImportWarning>,
    /// Registry of symbolic dimension variables (batch, seq_len, etc.).
    pub dim_vars: DimVarTable,
    /// Shape constraints collected during import and shape propagation.
    pub shape_constraints: ConstraintStore,
    /// Named subgraphs for control flow ops (If branches, Loop/Scan bodies).
    /// Empty for models without control flow — zero cost.
    pub subgraphs: HashMap<String, AiGraph>,
    /// Reverse mapping from TensorId to original source name (e.g. ONNX tensor name).
    /// Populated by importers, used by `compile_with_debug_info()` for conformance testing.
    /// Empty for models imported without name tracking — zero cost.
    pub tensor_names: HashMap<TensorId, String>,

    /// Cached topological order. Invalidated by any structural graph mutation.
    /// Uses `Mutex<Option<Arc<...>>>` so callers share the allocation (zero clones).
    /// `Mutex` (not `RefCell`) makes `AiGraph: Sync`, enabling parallel compilation.
    /// **Do not read or write directly** — use `topo_order()` and `invalidate_topo_cache()`.
    #[doc(hidden)]
    pub topo_cache: Mutex<Option<Arc<Vec<NodeId>>>>,
}

impl Clone for AiGraph {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            nodes: self.nodes.clone(),
            inputs: self.inputs.clone(),
            outputs: self.outputs.clone(),
            input_names: self.input_names.clone(),
            output_names: self.output_names.clone(),
            params: self.params.clone(),
            tensor_info: self.tensor_info.clone(),
            metadata: self.metadata.clone(),
            warnings: self.warnings.clone(),
            dim_vars: self.dim_vars.clone(),
            shape_constraints: self.shape_constraints.clone(),
            subgraphs: self.subgraphs.clone(),
            tensor_names: self.tensor_names.clone(),
            // Don't carry over the topo cache — the clone starts fresh.
            topo_cache: Mutex::new(None),
        }
    }
}

impl AiGraph {
    /// Get the human-readable name for graph input at `index`.
    pub fn input_name(&self, index: usize) -> String {
        self.input_names
            .get(index)
            .cloned()
            .unwrap_or_else(|| format!("input_{index}"))
    }

    /// Get the human-readable name for graph output at `index`.
    pub fn output_name(&self, index: usize) -> String {
        self.output_names
            .get(index)
            .cloned()
            .unwrap_or_else(|| format!("output_{index}"))
    }

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
            errors.push(ValidationError {
                message: "graph contains a cycle".to_string(),
            });
        }

        // Recurse into subgraphs.
        for (key, sub) in &self.subgraphs {
            for e in sub.validate() {
                errors.push(ValidationError {
                    message: format!("subgraph '{key}': {}", e.message),
                });
            }
        }

        errors
    }

    /// Returns nodes in topological order (Kahn's algorithm on NodeIds).
    ///
    /// The result is cached internally. Call [`Self::invalidate_topo_cache`] after
    /// any structural graph mutation (add/remove nodes, rewire edges).
    pub fn topo_order(&self) -> Arc<Vec<NodeId>> {
        // Return cached order if available (Arc::clone is O(1), no data copy).
        let guard = self.topo_cache.lock().expect("topo_cache mutex poisoned");
        if let Some(cached) = guard.as_ref() {
            return Arc::clone(cached);
        }
        drop(guard);

        let order = Arc::new(self.compute_topo_order());

        // Cache the result.
        let mut guard = self.topo_cache.lock().expect("topo_cache mutex poisoned");
        *guard = Some(Arc::clone(&order));
        order
    }

    /// Invalidate the cached topological order.
    ///
    /// Must be called after any structural mutation: adding/removing nodes,
    /// changing node inputs/outputs, etc. Optimization passes that rewrite
    /// the `nodes` vec should call this.
    pub fn invalidate_topo_cache(&self) {
        let mut guard = self.topo_cache.lock().expect("topo_cache mutex poisoned");
        *guard = None;
    }

    /// Compute topological order from scratch (Kahn's algorithm).
    ///
    /// The result is a **pure function of the `nodes` vector** — never of a
    /// `HashMap`/`HashSet` iteration seed. A DAG admits many valid topological
    /// orders, and the one chosen here fixes the compiled archive's node and
    /// buffer layout, hence its content address κ. Seeding the ready set and
    /// draining successors in ascending `nodes`-vector position makes that
    /// choice deterministic across processes and platforms: content addressing
    /// (a model must have a stable κ) and κ-dedup both require it, and the
    /// per-map seed can no longer select a different — previously
    /// executor-hostile — ordering across a rebuild. When `nodes` is already a
    /// valid topological order (the invariant importers and opt passes
    /// maintain), this reproduces the vector order exactly.
    fn compute_topo_order(&self) -> Vec<NodeId> {
        let n = self.nodes.len();

        // Producer of each TensorId: the position (in the stable `nodes`
        // vector) of the node that emits it. A keyed lookup whose own
        // iteration order is never observed.
        let mut producer: HashMap<TensorId, usize> = HashMap::with_capacity(n);
        for (pos, node) in self.nodes.iter().enumerate() {
            for &tid in &node.outputs {
                producer.insert(tid, pos);
            }
        }

        // Dependency edges as node-vector positions, deduplicated per
        // (producer, consumer) exactly as the original adjacency `HashSet` did.
        // `successors[p]` holds the positions of nodes consuming an output of
        // `self.nodes[p]`; `in_degree[c]` counts c's distinct producers. A
        // self-referencing edge keeps `in_degree` above zero, so a cycle is
        // still surfaced by a short order (see `has_cycle`).
        let mut successors: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut in_degree: Vec<usize> = vec![0; n];
        let mut seen_edges: HashSet<(usize, usize)> = HashSet::new();
        for (consumer, node) in self.nodes.iter().enumerate() {
            for &tid in &node.inputs {
                if let Some(&prod) = producer.get(&tid) {
                    if seen_edges.insert((prod, consumer)) {
                        successors[prod].push(consumer);
                        in_degree[consumer] += 1;
                    }
                }
            }
        }

        // Kahn's algorithm with the ready set ordered by ascending position (a
        // min-heap over positions), the deterministic tiebreak.
        let mut ready: BinaryHeap<Reverse<usize>> = (0..n)
            .filter(|&pos| in_degree[pos] == 0)
            .map(Reverse)
            .collect();
        let mut order = Vec::with_capacity(n);
        while let Some(Reverse(pos)) = ready.pop() {
            order.push(self.nodes[pos].id);
            for &succ in &successors[pos] {
                in_degree[succ] -= 1;
                if in_degree[succ] == 0 {
                    ready.push(Reverse(succ));
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
        ti.insert(
            0u32,
            TensorInfo::new(DType::F32, shape_from_concrete(&[1, 64])),
        );
        ti.insert(
            1u32,
            TensorInfo::new(DType::F32, shape_from_concrete(&[1, 64])),
        );

        AiGraph {
            name: "test".into(),
            nodes: vec![AiNode::new(0, AiOp::Identity, vec![0], vec![1])],
            inputs: vec![0],
            outputs: vec![1],
            input_names: vec![],
            output_names: vec![],
            params: HashMap::new(),
            tensor_info: ti,
            metadata: HashMap::new(),
            warnings: vec![],
            dim_vars: DimVarTable::default(),
            shape_constraints: ConstraintStore::default(),
            subgraphs: HashMap::new(),
            tensor_names: HashMap::new(),
            topo_cache: Default::default(),
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
        assert_eq!(*g.topo_order(), vec![0u32]);
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
