//! Attention pattern detection for execution group creation.
//!
//! This module detects self-attention patterns in ONNX graphs to enable
//! parallel execution of Q, K, V projections.
//!
//! # Detection Strategy
//!
//! Attention patterns typically follow this structure:
//! ```text
//! hidden_states
//!   ├── Q = MatMul(hidden, Wq) + Bq
//!   ├── K = MatMul(hidden, Wk) + Bk  (parallel with Q, V)
//!   └── V = MatMul(hidden, Wv) + Bv  (parallel with Q, K)
//!         │
//!         └── Attention = Softmax(Q @ K.T / sqrt(d)) @ V
//! ```
//!
//! Q, K, V projections share the same input and can execute in parallel.
//!
//! # Supported Patterns
//!
//! - BERT/RoBERTa: `encoder.layer.N.attention.self.query/key/value`
//! - GPT-2/OPT: `transformer.h.N.attn.c_attn` (combined QKV or separate)
//! - LLaMA: `model.layers.N.self_attn.q_proj/k_proj/v_proj`
//! - T5: `encoder.block.N.layer.0.SelfAttention.q/k/v`

use crate::proto::{GraphProto, NodeProto};
use ahash::AHashMap;
use tracing::{debug, trace};

/// Information about a detected attention pattern.
#[derive(Debug, Clone)]
pub struct AttentionPattern {
    /// Name prefix for this attention block (e.g., "encoder.layer.0.attention")
    pub name_prefix: String,

    /// Node indices for Q projection operations
    pub q_nodes: Vec<usize>,

    /// Node indices for K projection operations
    pub k_nodes: Vec<usize>,

    /// Node indices for V projection operations
    pub v_nodes: Vec<usize>,

    /// Node indices for attention computation (softmax, matmul, etc.)
    pub attention_nodes: Vec<usize>,

    /// The common input tensor name (hidden states)
    pub input_tensor: String,
}

impl AttentionPattern {
    /// Get all projection node indices (Q, K, V).
    pub fn projection_nodes(&self) -> Vec<usize> {
        let mut nodes =
            Vec::with_capacity(self.q_nodes.len() + self.k_nodes.len() + self.v_nodes.len());
        nodes.extend(&self.q_nodes);
        nodes.extend(&self.k_nodes);
        nodes.extend(&self.v_nodes);
        nodes
    }
}

/// Known Q/K/V projection name patterns.
const QKV_PATTERNS: &[(&str, &str, &str)] = &[
    // BERT-style
    ("query", "key", "value"),
    // LLaMA-style
    ("q_proj", "k_proj", "v_proj"),
    // GPT-style
    ("c_attn_q", "c_attn_k", "c_attn_v"),
    // Generic
    (".q", ".k", ".v"),
    ("_q", "_k", "_v"),
    // T5-style
    ("SelfAttention.q", "SelfAttention.k", "SelfAttention.v"),
];

/// Detect attention patterns in an ONNX graph.
///
/// This function analyzes node names and dependencies to identify
/// attention blocks where Q, K, V projections can run in parallel.
///
/// # Arguments
///
/// * `graph` - The ONNX graph to analyze
///
/// # Returns
///
/// A vector of detected attention patterns, sorted by name prefix.
pub fn detect_attention_patterns(graph: &GraphProto) -> Vec<AttentionPattern> {
    debug!(
        "Detecting attention patterns in graph with {} nodes",
        graph.node.len()
    );

    let mut patterns = Vec::new();

    // Build name-to-index map (for future use in more complex pattern detection)
    let _name_to_idx: AHashMap<&str, usize> = graph
        .node
        .iter()
        .enumerate()
        .map(|(i, n)| (n.name.as_str(), i))
        .collect();

    // Build tensor-to-producer map
    let tensor_producer: AHashMap<&str, usize> = graph
        .node
        .iter()
        .enumerate()
        .flat_map(|(i, n)| n.output.iter().map(move |out| (out.as_str(), i)))
        .collect();

    // Group nodes by attention block prefix
    let attention_blocks = group_by_attention_block(&graph.node);

    for (prefix, nodes) in attention_blocks {
        if let Some(pattern) = analyze_attention_block(graph, &prefix, &nodes, &tensor_producer) {
            debug!(
                "Found attention pattern: {} (Q={}, K={}, V={} nodes)",
                pattern.name_prefix,
                pattern.q_nodes.len(),
                pattern.k_nodes.len(),
                pattern.v_nodes.len()
            );
            patterns.push(pattern);
        }
    }

    // Sort by name prefix for deterministic output
    patterns.sort_by(|a, b| a.name_prefix.cmp(&b.name_prefix));

    debug!("Detected {} attention patterns total", patterns.len());
    patterns
}

/// Group nodes by their attention block prefix.
fn group_by_attention_block(nodes: &[NodeProto]) -> AHashMap<String, Vec<usize>> {
    let mut blocks: AHashMap<String, Vec<usize>> = AHashMap::new();

    for (idx, node) in nodes.iter().enumerate() {
        if let Some(prefix) = extract_attention_prefix(&node.name) {
            blocks.entry(prefix).or_default().push(idx);
        }
    }

    blocks
}

/// Extract the attention block prefix from a node name.
///
/// For example:
/// - "encoder.layer.0.attention.self.query" → "encoder.layer.0.attention"
/// - "model.layers.0.self_attn.q_proj" → "model.layers.0.self_attn"
fn extract_attention_prefix(name: &str) -> Option<String> {
    // Look for attention-related substrings
    let attention_markers = ["attention", "attn", "self_attn"];

    for marker in attention_markers {
        if let Some(pos) = name.find(marker) {
            // Find the end of the attention block name
            let prefix_end = pos + marker.len();

            // Include any trailing ".self" if present
            let full_prefix = if name[prefix_end..].starts_with(".self") {
                &name[..prefix_end + 5]
            } else {
                &name[..prefix_end]
            };

            return Some(full_prefix.to_string());
        }
    }

    None
}

/// Analyze a group of nodes to identify Q, K, V patterns.
fn analyze_attention_block(
    graph: &GraphProto,
    prefix: &str,
    node_indices: &[usize],
    tensor_producer: &AHashMap<&str, usize>,
) -> Option<AttentionPattern> {
    let mut q_nodes = Vec::new();
    let mut k_nodes = Vec::new();
    let mut v_nodes = Vec::new();
    let mut attention_nodes = Vec::new();
    let mut common_input: Option<String> = None;

    // Categorize nodes by Q/K/V patterns
    for &idx in node_indices {
        let node = &graph.node[idx];
        let name_lower = node.name.to_lowercase();

        // Check if this is a Q, K, or V projection
        let is_q = QKV_PATTERNS
            .iter()
            .any(|(q, _, _)| name_lower.contains(&q.to_lowercase()));
        let is_k = QKV_PATTERNS
            .iter()
            .any(|(_, k, _)| name_lower.contains(&k.to_lowercase()));
        let is_v = QKV_PATTERNS
            .iter()
            .any(|(_, _, v)| name_lower.contains(&v.to_lowercase()));

        if is_q {
            q_nodes.push(idx);
            if let Some(input) = find_hidden_state_input(graph, node, tensor_producer) {
                common_input = Some(input);
            }
        } else if is_k {
            k_nodes.push(idx);
        } else if is_v {
            v_nodes.push(idx);
        } else {
            // Non-projection node in attention block
            attention_nodes.push(idx);
        }
    }

    // Need at least Q, K, and V projections
    if q_nodes.is_empty() || k_nodes.is_empty() || v_nodes.is_empty() {
        trace!("Attention block '{}' missing Q/K/V projections", prefix);
        return None;
    }

    Some(AttentionPattern {
        name_prefix: prefix.to_string(),
        q_nodes,
        k_nodes,
        v_nodes,
        attention_nodes,
        input_tensor: common_input.unwrap_or_default(),
    })
}

/// Find the hidden state input tensor for a projection node.
fn find_hidden_state_input(
    _graph: &GraphProto,
    node: &NodeProto,
    _tensor_producer: &AHashMap<&str, usize>,
) -> Option<String> {
    // The first non-weight input is typically the hidden state
    // Weights are usually named with "weight", "kernel", etc.
    for input in &node.input {
        if !input.is_empty()
            && !input.to_lowercase().contains("weight")
            && !input.to_lowercase().contains("kernel")
            && !input.to_lowercase().contains("bias")
        {
            return Some(input.clone());
        }
    }
    None
}

/// Assign execution groups based on detected attention patterns.
///
/// Returns a mapping from node index to group ID.
/// - Group 0: Default (non-attention nodes)
/// - Group 1, 2, 3...: Per-attention-block groups (Q, K, V parallel within each)
pub fn assign_execution_groups(patterns: &[AttentionPattern], total_nodes: usize) -> Vec<u64> {
    let mut groups = vec![0u64; total_nodes];
    let mut next_group = 1u64;

    for pattern in patterns {
        let base_group = next_group;

        // Q nodes get one group
        for &idx in &pattern.q_nodes {
            if idx < total_nodes {
                groups[idx] = base_group;
            }
        }

        // K nodes get parallel group
        for &idx in &pattern.k_nodes {
            if idx < total_nodes {
                groups[idx] = base_group + 1;
            }
        }

        // V nodes get parallel group
        for &idx in &pattern.v_nodes {
            if idx < total_nodes {
                groups[idx] = base_group + 2;
            }
        }

        // Attention computation nodes get dependent group
        for &idx in &pattern.attention_nodes {
            if idx < total_nodes {
                groups[idx] = base_group + 3;
            }
        }

        next_group += 4;
    }

    groups
}

/// Get group dependencies for attention patterns.
///
/// Returns pairs of (dependent_group, dependency_group).
pub fn get_group_dependencies(patterns: &[AttentionPattern]) -> Vec<(u64, u64)> {
    let mut deps = Vec::new();
    let mut base_group = 1u64;

    for _pattern in patterns {
        let q_group = base_group;
        let k_group = base_group + 1;
        let v_group = base_group + 2;
        let attn_group = base_group + 3;

        // Q, K, V all depend on input (group 0)
        deps.push((q_group, 0));
        deps.push((k_group, 0));
        deps.push((v_group, 0));

        // Attention depends on Q, K, V
        deps.push((attn_group, q_group));
        deps.push((attn_group, k_group));
        deps.push((attn_group, v_group));

        base_group += 4;
    }

    deps
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_bert_attention_graph() -> GraphProto {
        let mut graph = GraphProto {
            name: "bert_attention".to_string(),
            ..Default::default()
        };

        // Simulated BERT attention block
        let nodes = vec![
            NodeProto {
                name: "encoder.layer.0.attention.self.query/MatMul".to_string(),
                op_type: "MatMul".to_string(),
                input: vec!["hidden_states".to_string(), "query_weight".to_string()],
                output: vec!["query_output".to_string()],
                ..Default::default()
            },
            NodeProto {
                name: "encoder.layer.0.attention.self.key/MatMul".to_string(),
                op_type: "MatMul".to_string(),
                input: vec!["hidden_states".to_string(), "key_weight".to_string()],
                output: vec!["key_output".to_string()],
                ..Default::default()
            },
            NodeProto {
                name: "encoder.layer.0.attention.self.value/MatMul".to_string(),
                op_type: "MatMul".to_string(),
                input: vec!["hidden_states".to_string(), "value_weight".to_string()],
                output: vec!["value_output".to_string()],
                ..Default::default()
            },
            NodeProto {
                name: "encoder.layer.0.attention.self/Softmax".to_string(),
                op_type: "Softmax".to_string(),
                input: vec!["attention_scores".to_string()],
                output: vec!["attention_probs".to_string()],
                ..Default::default()
            },
        ];

        graph.node = nodes;
        graph
    }

    fn create_llama_attention_graph() -> GraphProto {
        let mut graph = GraphProto {
            name: "llama_attention".to_string(),
            ..Default::default()
        };

        let nodes = vec![
            NodeProto {
                name: "model.layers.0.self_attn.q_proj".to_string(),
                op_type: "MatMul".to_string(),
                input: vec!["hidden".to_string(), "q_weight".to_string()],
                output: vec!["q".to_string()],
                ..Default::default()
            },
            NodeProto {
                name: "model.layers.0.self_attn.k_proj".to_string(),
                op_type: "MatMul".to_string(),
                input: vec!["hidden".to_string(), "k_weight".to_string()],
                output: vec!["k".to_string()],
                ..Default::default()
            },
            NodeProto {
                name: "model.layers.0.self_attn.v_proj".to_string(),
                op_type: "MatMul".to_string(),
                input: vec!["hidden".to_string(), "v_weight".to_string()],
                output: vec!["v".to_string()],
                ..Default::default()
            },
            NodeProto {
                name: "model.layers.0.self_attn/attn_output".to_string(),
                op_type: "MatMul".to_string(),
                input: vec!["attn".to_string(), "o_weight".to_string()],
                output: vec!["output".to_string()],
                ..Default::default()
            },
        ];

        graph.node = nodes;
        graph
    }

    #[test]
    fn test_detect_bert_attention() {
        let graph = create_bert_attention_graph();
        let patterns = detect_attention_patterns(&graph);

        assert_eq!(patterns.len(), 1);
        assert_eq!(patterns[0].q_nodes.len(), 1);
        assert_eq!(patterns[0].k_nodes.len(), 1);
        assert_eq!(patterns[0].v_nodes.len(), 1);
        assert!(patterns[0].name_prefix.contains("attention"));
    }

    #[test]
    fn test_detect_llama_attention() {
        let graph = create_llama_attention_graph();
        let patterns = detect_attention_patterns(&graph);

        assert_eq!(patterns.len(), 1);
        assert_eq!(patterns[0].q_nodes.len(), 1);
        assert_eq!(patterns[0].k_nodes.len(), 1);
        assert_eq!(patterns[0].v_nodes.len(), 1);
        assert!(patterns[0].name_prefix.contains("self_attn"));
    }

    #[test]
    fn test_no_attention_pattern() {
        let mut graph = GraphProto {
            name: "conv_net".to_string(),
            ..Default::default()
        };

        graph.node.push(NodeProto {
            name: "conv1".to_string(),
            op_type: "Conv".to_string(),
            ..Default::default()
        });
        graph.node.push(NodeProto {
            name: "conv2".to_string(),
            op_type: "Conv".to_string(),
            ..Default::default()
        });

        let patterns = detect_attention_patterns(&graph);
        assert!(patterns.is_empty());
    }

    #[test]
    fn test_extract_attention_prefix() {
        assert_eq!(
            extract_attention_prefix("encoder.layer.0.attention.self.query"),
            Some("encoder.layer.0.attention.self".to_string())
        );
        assert_eq!(
            extract_attention_prefix("model.layers.0.self_attn.q_proj"),
            Some("model.layers.0.self_attn".to_string())
        );
        assert_eq!(
            extract_attention_prefix("transformer.h.0.attn.c_attn"),
            Some("transformer.h.0.attn".to_string())
        );
        assert_eq!(extract_attention_prefix("conv.layer.0"), None);
    }

    #[test]
    fn test_assign_execution_groups() {
        let graph = create_bert_attention_graph();
        let patterns = detect_attention_patterns(&graph);
        let groups = assign_execution_groups(&patterns, graph.node.len());

        // Q, K, V should be in groups 1, 2, 3 respectively
        // They should be different groups (parallel)
        assert_ne!(groups[0], groups[1]); // Q != K
        assert_ne!(groups[0], groups[2]); // Q != V
        assert_ne!(groups[1], groups[2]); // K != V

        // Softmax (attention computation) should be in group 4
        assert_eq!(groups[3], 4);
    }

    #[test]
    fn test_group_dependencies() {
        let graph = create_bert_attention_graph();
        let patterns = detect_attention_patterns(&graph);
        let deps = get_group_dependencies(&patterns);

        // Q, K, V depend on input group (0)
        assert!(deps.contains(&(1, 0)));
        assert!(deps.contains(&(2, 0)));
        assert!(deps.contains(&(3, 0)));

        // Attention depends on Q, K, V
        assert!(deps.contains(&(4, 1)));
        assert!(deps.contains(&(4, 2)));
        assert!(deps.contains(&(4, 3)));
    }

    #[test]
    fn test_multiple_attention_blocks() {
        let mut graph = GraphProto::default();

        // Layer 0
        for name in &["query", "key", "value"] {
            graph.node.push(NodeProto {
                name: format!("encoder.layer.0.attention.self.{}", name),
                op_type: "MatMul".to_string(),
                input: vec!["input".to_string()],
                ..Default::default()
            });
        }

        // Layer 1
        for name in &["query", "key", "value"] {
            graph.node.push(NodeProto {
                name: format!("encoder.layer.1.attention.self.{}", name),
                op_type: "MatMul".to_string(),
                input: vec!["input".to_string()],
                ..Default::default()
            });
        }

        let patterns = detect_attention_patterns(&graph);
        assert_eq!(patterns.len(), 2);
    }
}
