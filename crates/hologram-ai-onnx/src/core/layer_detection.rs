//! Detect transformer layer structure in ONNX models.
//!
//! This module identifies repeating transformer layer patterns in ONNX graphs,
//! enabling layer-wise compilation and execution for memory-efficient inference.
//!
//! # Supported Patterns
//!
//! The detection logic recognizes common transformer layer naming conventions:
//! - `encoder.layer.N.*` (BERT, RoBERTa, DistilBERT)
//! - `decoder.layer.N.*` / `decoder.block.N.*` (GPT-2, T5 decoder)
//! - `transformer.h.N.*` (GPT, OPT)
//! - `model.layers.N.*` (LLaMA, Mistral)
//! - `layers.N.*` (generic)
//!
//! # Example
//!
//! ```rust,ignore
//! use hologram_ai_onnx::core::layer_detection::detect_transformer_layers;
//!
//! let layers = detect_transformer_layers(&graph)?;
//! if let Some(layers) = layers {
//!     println!("Found {} transformer layers", layers.len());
//!     for layer in &layers {
//!         println!("Layer {}: {} nodes", layer.index, layer.node_names.len());
//!     }
//! }
//! ```

use crate::proto::GraphProto;
use ahash::{AHashMap, AHashSet};
use tracing::{debug, trace};

/// Information about a detected transformer layer.
#[derive(Debug, Clone)]
pub struct LayerInfo {
    /// Layer prefix (e.g., "encoder.layer", "decoder.block", "transformer.h")
    pub prefix: String,

    /// Layer index (0-based)
    pub index: usize,

    /// Names of nodes belonging to this layer
    pub node_names: Vec<String>,

    /// Original indices of nodes in the graph
    pub node_indices: Vec<usize>,

    /// Input tensor names (from previous layer or embedding)
    pub inputs: Vec<String>,

    /// Output tensor names (to next layer or final output)
    pub outputs: Vec<String>,
}

impl LayerInfo {
    /// Get the number of nodes in this layer.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.node_names.len()
    }

    /// Get the full layer name (e.g., "encoder.layer.0").
    #[must_use]
    pub fn full_name(&self) -> String {
        format!("{}.{}", self.prefix, self.index)
    }
}

/// Known layer prefixes in transformer models.
///
/// Order matters - more specific patterns should come first.
const LAYER_PREFIXES: &[&str] = &[
    // BERT-style: encoder.layer.N, decoder.layer.N
    "encoder.layer.",
    "decoder.layer.",
    // T5/BART-style: encoder.block.N, decoder.block.N
    "encoder.block.",
    "decoder.block.",
    // GPT-2/OPT-style: transformer.h.N
    "transformer.h.",
    // LLaMA/Mistral-style: model.layers.N
    "model.layers.",
    // Generic: layers.N
    "layers.",
    // Bloom-style: h.N
    "h.",
];

/// Detect transformer layers in an ONNX graph.
///
/// Analyzes node names to identify repeating layer patterns typical of
/// transformer architectures. Returns `None` if no layer structure is found.
///
/// # Arguments
///
/// * `graph` - The ONNX graph to analyze
///
/// # Returns
///
/// - `Some(Vec<LayerInfo>)` - Detected layers sorted by index
/// - `None` - No transformer layer structure detected
///
/// # Example
///
/// ```rust,ignore
/// let model = parse_model(&onnx_bytes)?;
/// let graph = model.graph.as_ref().unwrap();
///
/// if let Some(layers) = detect_transformer_layers(graph) {
///     println!("Model has {} layers", layers.len());
/// } else {
///     println!("No layer structure detected");
/// }
/// ```
pub fn detect_transformer_layers(graph: &GraphProto) -> Option<Vec<LayerInfo>> {
    debug!(
        "Detecting transformer layers in graph with {} nodes",
        graph.node.len()
    );

    // Try each prefix in order of specificity
    for prefix in LAYER_PREFIXES {
        if let Some(layers) = try_detect_with_prefix(graph, prefix) {
            debug!("Detected {} layers using prefix '{}'", layers.len(), prefix);
            return Some(layers);
        }
    }

    debug!("No transformer layer pattern detected");
    None
}

/// Parse layer index from a node name given a prefix.
///
/// For example, with prefix "encoder.layer." and name "encoder.layer.5.attention.self":
/// Returns Some((5, "encoder.layer"))
fn parse_layer_index(name: &str, prefix: &str) -> Option<(usize, String)> {
    // Check if name starts with prefix
    if !name.starts_with(prefix) {
        return None;
    }

    // Extract the part after prefix
    let after_prefix = &name[prefix.len()..];

    // Find the layer index (digits until next '.' or end)
    let end_idx = after_prefix
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(after_prefix.len());

    if end_idx == 0 {
        return None;
    }

    let index_str = &after_prefix[..end_idx];
    let index: usize = index_str.parse().ok()?;

    // Return prefix without trailing dot for cleaner names
    let clean_prefix = prefix.trim_end_matches('.');
    Some((index, clean_prefix.to_string()))
}

/// Try to detect layers using a specific prefix.
fn try_detect_with_prefix(graph: &GraphProto, prefix: &str) -> Option<Vec<LayerInfo>> {
    // Map: layer_index -> Vec<(node_name, node_index)>
    let mut layer_nodes: AHashMap<usize, Vec<(String, usize)>> = AHashMap::new();
    let mut detected_prefix: Option<String> = None;

    for (node_idx, node) in graph.node.iter().enumerate() {
        if let Some((layer_idx, clean_prefix)) = parse_layer_index(&node.name, prefix) {
            detected_prefix.get_or_insert(clean_prefix);
            layer_nodes
                .entry(layer_idx)
                .or_default()
                .push((node.name.clone(), node_idx));
        }
    }

    // Need at least 2 layers for a meaningful pattern
    if layer_nodes.len() < 2 {
        return None;
    }

    let prefix_str = detected_prefix?;

    // Verify this is a consistent layer structure
    // All layers should have similar node counts (within 20% tolerance)
    let node_counts: Vec<usize> = layer_nodes.values().map(|v| v.len()).collect();
    let avg_count = node_counts.iter().sum::<usize>() / node_counts.len();
    let tolerance = avg_count / 5; // 20% tolerance

    for count in &node_counts {
        if (*count as isize - avg_count as isize).unsigned_abs() > tolerance {
            trace!(
                "Layer node count {} differs too much from average {}, skipping prefix",
                count, avg_count
            );
            return None;
        }
    }

    // Convert to sorted list
    let mut layers: Vec<_> = layer_nodes.into_iter().collect();
    layers.sort_by_key(|(idx, _)| *idx);

    // Verify consecutive layer indices starting from 0
    let indices: Vec<usize> = layers.iter().map(|(idx, _)| *idx).collect();
    if !are_consecutive(&indices) {
        trace!("Layer indices are not consecutive: {:?}", indices);
        return None;
    }

    // Build LayerInfo structs
    let layer_infos: Vec<LayerInfo> = layers
        .into_iter()
        .map(|(index, nodes)| {
            let node_names: Vec<String> = nodes.iter().map(|(name, _)| name.clone()).collect();
            let node_indices: Vec<usize> = nodes.iter().map(|(_, idx)| *idx).collect();

            LayerInfo {
                prefix: prefix_str.clone(),
                index,
                node_names,
                node_indices,
                inputs: Vec::new(),  // Filled by analyze_layer_boundaries
                outputs: Vec::new(), // Filled by analyze_layer_boundaries
            }
        })
        .collect();

    // Analyze input/output boundaries
    let layer_infos = analyze_layer_boundaries(graph, layer_infos);

    Some(layer_infos)
}

/// Check if indices are consecutive starting from 0.
fn are_consecutive(indices: &[usize]) -> bool {
    if indices.is_empty() {
        return true;
    }

    for (i, &idx) in indices.iter().enumerate() {
        if idx != i {
            return false;
        }
    }
    true
}

/// Pre-computed tensor dependency graph for efficient boundary analysis.
struct TensorGraph<'a> {
    /// tensor_name -> producing layer index (-1 for external inputs/initializers)
    producer: AHashMap<&'a str, isize>,
    /// tensor_name -> list of consuming layer indices
    consumers: AHashMap<&'a str, Vec<usize>>,
    /// Tensors that are graph outputs
    graph_outputs: AHashSet<&'a str>,
}

impl<'a> TensorGraph<'a> {
    /// Build tensor dependency graph from ONNX graph and detected layers.
    fn build(graph: &'a GraphProto, layers: &[LayerInfo]) -> Self {
        let mut producer: AHashMap<&str, isize> = AHashMap::new();
        let mut consumers: AHashMap<&str, Vec<usize>> = AHashMap::new();

        // Mark graph inputs and initializers as external (-1)
        for input in &graph.input {
            producer.insert(&input.name, -1);
        }
        for init in &graph.initializer {
            producer.insert(&init.name, -1);
        }

        // Map tensors to producing layers and build consumer list
        for (layer_idx, layer) in layers.iter().enumerate() {
            for node_idx in &layer.node_indices {
                let node = &graph.node[*node_idx];

                // Record this layer as producer of its outputs
                for output in &node.output {
                    producer.insert(output.as_str(), layer_idx as isize);
                }

                // Record this layer as consumer of its inputs
                for input in &node.input {
                    if !input.is_empty() {
                        consumers.entry(input.as_str()).or_default().push(layer_idx);
                    }
                }
            }
        }

        // Collect graph outputs
        let graph_outputs: AHashSet<&str> = graph.output.iter().map(|o| o.name.as_str()).collect();

        Self {
            producer,
            consumers,
            graph_outputs,
        }
    }

    /// Get layer inputs: tensors consumed by layer but produced elsewhere.
    fn layer_inputs(&self, layer_idx: usize, layer: &LayerInfo, graph: &GraphProto) -> Vec<String> {
        let mut inputs: AHashSet<String> = AHashSet::new();

        for node_idx in &layer.node_indices {
            let node = &graph.node[*node_idx];
            for input in &node.input {
                if input.is_empty() {
                    continue;
                }
                let is_external_input = self
                    .producer
                    .get(input.as_str())
                    .is_some_and(|&prod| prod != layer_idx as isize);

                if is_external_input {
                    inputs.insert(input.clone());
                }
            }
        }

        let mut result: Vec<_> = inputs.into_iter().collect();
        result.sort();
        result
    }

    /// Get layer outputs: tensors produced by layer and consumed by other layers or graph outputs.
    fn layer_outputs(
        &self,
        layer_idx: usize,
        layer: &LayerInfo,
        graph: &GraphProto,
    ) -> Vec<String> {
        let mut outputs: AHashSet<String> = AHashSet::new();

        for node_idx in &layer.node_indices {
            let node = &graph.node[*node_idx];
            for output in &node.output {
                // Check if consumed by other layers
                let consumed_by_other = self
                    .consumers
                    .get(output.as_str())
                    .is_some_and(|consumers| consumers.iter().any(|&idx| idx != layer_idx));

                if consumed_by_other {
                    outputs.insert(output.clone());
                    continue;
                }
                // Check if it's a graph output
                if self.graph_outputs.contains(output.as_str()) {
                    outputs.insert(output.clone());
                }
            }
        }

        let mut result: Vec<_> = outputs.into_iter().collect();
        result.sort();
        result
    }
}

/// Analyze and fill in input/output tensor boundaries for each layer.
///
/// Uses a pre-built tensor dependency graph for O(n) complexity instead of O(n²).
fn analyze_layer_boundaries(graph: &GraphProto, mut layers: Vec<LayerInfo>) -> Vec<LayerInfo> {
    // Build tensor graph once (O(n) where n = total nodes)
    let tensor_graph = TensorGraph::build(graph, &layers);

    // Collect all boundary data first (avoids borrow conflicts)
    let boundaries: Vec<_> = layers
        .iter()
        .enumerate()
        .map(|(idx, layer)| {
            let inputs = tensor_graph.layer_inputs(idx, layer, graph);
            let outputs = tensor_graph.layer_outputs(idx, layer, graph);
            (inputs, outputs)
        })
        .collect();

    // Apply boundaries to layers
    for (layer, (inputs, outputs)) in layers.iter_mut().zip(boundaries) {
        trace!(
            "Layer {}: {} inputs, {} outputs",
            layer.full_name(),
            inputs.len(),
            outputs.len()
        );
        layer.inputs = inputs;
        layer.outputs = outputs;
    }

    layers
}

/// Get layer statistics for a detected layer structure.
#[derive(Debug, Clone)]
pub struct LayerStats {
    /// Total number of layers
    pub layer_count: usize,

    /// Average nodes per layer
    pub avg_nodes_per_layer: usize,

    /// Minimum nodes in any layer
    pub min_nodes: usize,

    /// Maximum nodes in any layer
    pub max_nodes: usize,

    /// Layer prefix pattern
    pub prefix: String,
}

impl LayerStats {
    /// Compute statistics from detected layers.
    #[must_use]
    pub fn from_layers(layers: &[LayerInfo]) -> Option<Self> {
        if layers.is_empty() {
            return None;
        }

        let node_counts: Vec<usize> = layers.iter().map(|l| l.node_count()).collect();
        let total_nodes: usize = node_counts.iter().sum();

        Some(Self {
            layer_count: layers.len(),
            avg_nodes_per_layer: total_nodes / layers.len(),
            min_nodes: *node_counts.iter().min().unwrap_or(&0),
            max_nodes: *node_counts.iter().max().unwrap_or(&0),
            prefix: layers.first().map(|l| l.prefix.clone()).unwrap_or_default(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::NodeProto;

    fn create_test_graph(layer_count: usize, nodes_per_layer: usize) -> GraphProto {
        let mut graph = GraphProto {
            name: "test_model".to_string(),
            ..Default::default()
        };

        // Create encoder layer pattern
        for layer_idx in 0..layer_count {
            for node_idx in 0..nodes_per_layer {
                let node = NodeProto {
                    name: format!(
                        "encoder.layer.{}.attention.self.query_{}",
                        layer_idx, node_idx
                    ),
                    op_type: "MatMul".to_string(),
                    input: vec![format!("layer_{}_input_{}", layer_idx, node_idx)],
                    output: vec![format!("layer_{}_output_{}", layer_idx, node_idx)],
                    ..Default::default()
                };
                graph.node.push(node);
            }
        }

        graph
    }

    #[test]
    fn test_parse_layer_index() {
        // BERT-style
        assert_eq!(
            parse_layer_index("encoder.layer.5.attention.self", "encoder.layer."),
            Some((5, "encoder.layer".to_string()))
        );

        // GPT-style
        assert_eq!(
            parse_layer_index("transformer.h.12.attn", "transformer.h."),
            Some((12, "transformer.h".to_string()))
        );

        // LLaMA-style
        assert_eq!(
            parse_layer_index("model.layers.0.self_attn", "model.layers."),
            Some((0, "model.layers".to_string()))
        );

        // No match
        assert_eq!(parse_layer_index("some.other.node", "encoder.layer."), None);

        // Prefix matches but no digit
        assert_eq!(
            parse_layer_index("encoder.layer.attention", "encoder.layer."),
            None
        );
    }

    #[test]
    fn test_detect_encoder_layers() {
        let graph = create_test_graph(12, 10);

        let layers = detect_transformer_layers(&graph);
        assert!(layers.is_some());

        let layers = layers.unwrap();
        assert_eq!(layers.len(), 12);

        for (i, layer) in layers.iter().enumerate() {
            assert_eq!(layer.index, i);
            assert_eq!(layer.prefix, "encoder.layer");
            assert_eq!(layer.node_count(), 10);
        }
    }

    #[test]
    fn test_detect_gpt_layers() {
        let mut graph = GraphProto {
            name: "gpt_model".to_string(),
            ..Default::default()
        };

        // Create GPT-2 style layers
        for layer_idx in 0..6 {
            for node_idx in 0..5 {
                let node = NodeProto {
                    name: format!(
                        "transformer.h.{}.attn.c_attn.weight_{}",
                        layer_idx, node_idx
                    ),
                    op_type: "MatMul".to_string(),
                    ..Default::default()
                };
                graph.node.push(node);
            }
        }

        let layers = detect_transformer_layers(&graph);
        assert!(layers.is_some());

        let layers = layers.unwrap();
        assert_eq!(layers.len(), 6);
        assert_eq!(layers[0].prefix, "transformer.h");
    }

    #[test]
    fn test_detect_llama_layers() {
        let mut graph = GraphProto {
            name: "llama_model".to_string(),
            ..Default::default()
        };

        // Create LLaMA style layers
        for layer_idx in 0..32 {
            for node_idx in 0..8 {
                let node = NodeProto {
                    name: format!("model.layers.{}.self_attn.q_proj_{}", layer_idx, node_idx),
                    op_type: "MatMul".to_string(),
                    ..Default::default()
                };
                graph.node.push(node);
            }
        }

        let layers = detect_transformer_layers(&graph);
        assert!(layers.is_some());

        let layers = layers.unwrap();
        assert_eq!(layers.len(), 32);
        assert_eq!(layers[0].prefix, "model.layers");
    }

    #[test]
    fn test_no_layers_detected() {
        let mut graph = GraphProto {
            name: "non_transformer".to_string(),
            ..Default::default()
        };

        // Create non-transformer graph
        for i in 0..10 {
            let node = NodeProto {
                name: format!("conv_{}", i),
                op_type: "Conv".to_string(),
                ..Default::default()
            };
            graph.node.push(node);
        }

        let layers = detect_transformer_layers(&graph);
        assert!(layers.is_none());
    }

    #[test]
    fn test_single_layer_not_detected() {
        let mut graph = GraphProto {
            name: "single_layer".to_string(),
            ..Default::default()
        };

        // Only one layer - should not be detected as a layer pattern
        for i in 0..5 {
            let node = NodeProto {
                name: format!("encoder.layer.0.attention_{}", i),
                op_type: "MatMul".to_string(),
                ..Default::default()
            };
            graph.node.push(node);
        }

        let layers = detect_transformer_layers(&graph);
        assert!(layers.is_none());
    }

    #[test]
    fn test_non_consecutive_layers_rejected() {
        let mut graph = GraphProto {
            name: "gap_layers".to_string(),
            ..Default::default()
        };

        // Create layers 0, 2, 4 (gaps)
        for layer_idx in [0, 2, 4] {
            for i in 0..5 {
                let node = NodeProto {
                    name: format!("encoder.layer.{}.node_{}", layer_idx, i),
                    op_type: "MatMul".to_string(),
                    ..Default::default()
                };
                graph.node.push(node);
            }
        }

        let layers = detect_transformer_layers(&graph);
        assert!(layers.is_none());
    }

    #[test]
    fn test_layer_stats() {
        let graph = create_test_graph(6, 10);
        let layers = detect_transformer_layers(&graph).unwrap();

        let stats = LayerStats::from_layers(&layers).unwrap();
        assert_eq!(stats.layer_count, 6);
        assert_eq!(stats.avg_nodes_per_layer, 10);
        assert_eq!(stats.min_nodes, 10);
        assert_eq!(stats.max_nodes, 10);
        assert_eq!(stats.prefix, "encoder.layer");
    }

    #[test]
    fn test_layer_full_name() {
        let layer = LayerInfo {
            prefix: "encoder.layer".to_string(),
            index: 5,
            node_names: vec!["node1".to_string()],
            node_indices: vec![0],
            inputs: vec![],
            outputs: vec![],
        };

        assert_eq!(layer.full_name(), "encoder.layer.5");
    }

    #[test]
    fn test_are_consecutive() {
        assert!(are_consecutive(&[]));
        assert!(are_consecutive(&[0]));
        assert!(are_consecutive(&[0, 1, 2, 3]));
        assert!(!are_consecutive(&[1, 2, 3]));
        assert!(!are_consecutive(&[0, 2, 4]));
        assert!(!are_consecutive(&[0, 1, 3]));
    }

    #[test]
    fn test_layer_boundaries() {
        use crate::proto::ValueInfoProto;

        let mut graph = GraphProto {
            name: "boundary_test".to_string(),
            ..Default::default()
        };

        // Add graph input
        graph.input.push(ValueInfoProto {
            name: "input_ids".to_string(),
            ..Default::default()
        });

        // Layer 0: takes input_ids, produces layer_0_out
        graph.node.push(NodeProto {
            name: "encoder.layer.0.attention".to_string(),
            op_type: "MatMul".to_string(),
            input: vec!["input_ids".to_string()],
            output: vec!["layer_0_out".to_string()],
            ..Default::default()
        });
        graph.node.push(NodeProto {
            name: "encoder.layer.0.ffn".to_string(),
            op_type: "MatMul".to_string(),
            input: vec!["layer_0_out".to_string()],
            output: vec!["layer_0_final".to_string()],
            ..Default::default()
        });

        // Layer 1: takes layer_0_final, produces output
        graph.node.push(NodeProto {
            name: "encoder.layer.1.attention".to_string(),
            op_type: "MatMul".to_string(),
            input: vec!["layer_0_final".to_string()],
            output: vec!["layer_1_out".to_string()],
            ..Default::default()
        });
        graph.node.push(NodeProto {
            name: "encoder.layer.1.ffn".to_string(),
            op_type: "MatMul".to_string(),
            input: vec!["layer_1_out".to_string()],
            output: vec!["output".to_string()],
            ..Default::default()
        });

        // Add graph output
        graph.output.push(ValueInfoProto {
            name: "output".to_string(),
            ..Default::default()
        });

        let layers = detect_transformer_layers(&graph).unwrap();
        assert_eq!(layers.len(), 2);

        // Layer 0 should have input_ids as input, layer_0_final as output
        assert!(layers[0].inputs.contains(&"input_ids".to_string()));
        assert!(layers[0].outputs.contains(&"layer_0_final".to_string()));

        // Layer 1 should have layer_0_final as input, output as output
        assert!(layers[1].inputs.contains(&"layer_0_final".to_string()));
        assert!(layers[1].outputs.contains(&"output".to_string()));
    }
}
