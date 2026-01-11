//! Split ONNX models into per-layer subgraphs.
//!
//! This module takes a detected layer structure and extracts each layer
//! as an independent ONNX model that can be compiled and executed separately.
//!
//! # Usage
//!
//! ```rust,ignore
//! use hologram_ai_onnx::core::{layer_detection, layer_splitter};
//!
//! let model = parse_model(&onnx_bytes)?;
//! let graph = model.graph.as_ref().unwrap();
//!
//! // Detect layer structure
//! let layers = layer_detection::detect_transformer_layers(graph)?;
//!
//! // Split into per-layer models
//! let layer_models = layer_splitter::split_by_layers(&model, &layers)?;
//!
//! for (layer_name, layer_model) in layer_models {
//!     println!("Layer {} has {} nodes", layer_name, layer_model.graph.unwrap().node.len());
//! }
//! ```

use super::layer_detection::LayerInfo;
use crate::proto::{
    GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto, ValueInfoProto,
};
use ahash::AHashSet;
use tracing::{debug, trace};

/// Error type for layer splitting operations.
#[derive(Debug, Clone)]
pub enum SplitError {
    /// Model has no graph.
    NoGraph,
    /// Layer references invalid node indices.
    InvalidNodeIndex {
        /// Layer name.
        layer: String,
        /// Invalid node index.
        index: usize,
    },
    /// Required tensor not found.
    MissingTensor {
        /// Tensor name.
        tensor: String,
        /// Layer name.
        layer: String,
    },
}

impl std::fmt::Display for SplitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoGraph => write!(f, "Model has no graph"),
            Self::InvalidNodeIndex { layer, index } => {
                write!(
                    f,
                    "Layer '{}' references invalid node index {}",
                    layer, index
                )
            }
            Self::MissingTensor { tensor, layer } => {
                write!(f, "Tensor '{}' not found for layer '{}'", tensor, layer)
            }
        }
    }
}

impl std::error::Error for SplitError {}

/// Result type for layer splitting operations.
pub type SplitResult<T> = std::result::Result<T, SplitError>;

/// Split an ONNX model by detected layers.
///
/// Creates independent ONNX models for each transformer layer, suitable
/// for separate compilation and layer-by-layer execution.
///
/// # Arguments
///
/// * `model` - The original ONNX model
/// * `layers` - Detected layer information from `detect_transformer_layers`
///
/// # Returns
///
/// A vector of (layer_name, ModelProto) pairs, one for each layer.
///
/// # Example
///
/// ```rust,ignore
/// let layers = detect_transformer_layers(graph)?;
/// let layer_models = split_by_layers(&model, &layers)?;
///
/// for (name, layer_model) in layer_models {
///     let compiled = compiler.compile_to_bundle(&serialize(&layer_model))?;
///     fs::write(format!("{}.holb", name), compiled)?;
/// }
/// ```
pub fn split_by_layers(
    model: &ModelProto,
    layers: &[LayerInfo],
) -> SplitResult<Vec<(String, ModelProto)>> {
    let graph = model.graph.as_ref().ok_or(SplitError::NoGraph)?;

    debug!("Splitting model into {} layers", layers.len());

    let mut result = Vec::with_capacity(layers.len());

    for layer in layers {
        let layer_name = layer.full_name();
        trace!("Extracting layer: {}", layer_name);

        let subgraph = extract_layer_subgraph(graph, layer)?;

        let layer_model = ModelProto {
            ir_version: model.ir_version,
            opset_import: model.opset_import.clone(),
            producer_name: format!("{} (layer {})", model.producer_name, layer.index),
            producer_version: model.producer_version.clone(),
            domain: model.domain.clone(),
            model_version: model.model_version,
            doc_string: format!("Layer {} extracted from original model", layer.index),
            graph: Some(subgraph),
            metadata_props: Vec::new(),
            training_info: Vec::new(),
            functions: Vec::new(),
            configuration: Vec::new(),
        };

        result.push((layer_name, layer_model));
    }

    debug!("Split into {} layer models", result.len());
    Ok(result)
}

/// Extract a subgraph for a single layer.
fn extract_layer_subgraph(graph: &GraphProto, layer: &LayerInfo) -> SplitResult<GraphProto> {
    // Validate node indices
    for &idx in &layer.node_indices {
        if idx >= graph.node.len() {
            return Err(SplitError::InvalidNodeIndex {
                layer: layer.full_name(),
                index: idx,
            });
        }
    }

    // Extract nodes for this layer
    let nodes: Vec<NodeProto> = layer
        .node_indices
        .iter()
        .map(|&idx| graph.node[idx].clone())
        .collect();

    // Build set of tensors consumed by this layer
    let consumed_tensors: AHashSet<&str> = nodes
        .iter()
        .flat_map(|n| n.input.iter().filter(|s| !s.is_empty()).map(|s| s.as_str()))
        .collect();

    // Create input ValueInfoProto for external inputs
    let inputs: Vec<ValueInfoProto> = layer
        .inputs
        .iter()
        .map(|name| {
            // Try to find shape info from original graph
            find_value_info(graph, name).unwrap_or_else(|| ValueInfoProto {
                name: name.clone(),
                r#type: None,
                doc_string: String::new(),
                metadata_props: Vec::new(),
            })
        })
        .collect();

    // Create output ValueInfoProto for layer outputs
    let outputs: Vec<ValueInfoProto> = layer
        .outputs
        .iter()
        .map(|name| {
            find_value_info(graph, name).unwrap_or_else(|| ValueInfoProto {
                name: name.clone(),
                r#type: None,
                doc_string: String::new(),
                metadata_props: Vec::new(),
            })
        })
        .collect();

    // Extract initializers that are used by this layer
    let initializers: Vec<TensorProto> = graph
        .initializer
        .iter()
        .filter(|init| consumed_tensors.contains(init.name.as_str()))
        .cloned()
        .collect();

    // Create the subgraph
    Ok(GraphProto {
        node: nodes,
        name: layer.full_name(),
        initializer: initializers,
        sparse_initializer: Vec::new(),
        doc_string: format!("Subgraph for layer {}", layer.full_name()),
        input: inputs,
        output: outputs,
        value_info: Vec::new(),
        quantization_annotation: Vec::new(),
        metadata_props: Vec::new(),
    })
}

/// Find ValueInfoProto for a tensor in the graph.
fn find_value_info(graph: &GraphProto, name: &str) -> Option<ValueInfoProto> {
    // Check graph inputs
    if let Some(info) = graph.input.iter().find(|v| v.name == name) {
        return Some(info.clone());
    }

    // Check graph outputs
    if let Some(info) = graph.output.iter().find(|v| v.name == name) {
        return Some(info.clone());
    }

    // Check value_info
    if let Some(info) = graph.value_info.iter().find(|v| v.name == name) {
        return Some(info.clone());
    }

    None
}

/// Create a minimal model with just metadata (for embedding prefix).
///
/// This creates a stub model for non-layer components like embedding layers
/// or final output heads that aren't part of the repeating transformer structure.
pub fn create_stub_model(model: &ModelProto, name: &str) -> ModelProto {
    ModelProto {
        ir_version: model.ir_version,
        opset_import: model.opset_import.clone(),
        producer_name: model.producer_name.clone(),
        producer_version: model.producer_version.clone(),
        domain: model.domain.clone(),
        model_version: model.model_version,
        doc_string: format!("Stub model for {}", name),
        graph: Some(GraphProto {
            name: name.to_string(),
            ..Default::default()
        }),
        metadata_props: Vec::new(),
        training_info: Vec::new(),
        functions: Vec::new(),
        configuration: Vec::new(),
    }
}

/// Get opset version for a specific domain.
pub fn get_opset_version(opsets: &[OperatorSetIdProto], domain: &str) -> Option<i64> {
    opsets
        .iter()
        .find(|op| op.domain == domain || (domain.is_empty() && op.domain.is_empty()))
        .map(|op| op.version)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::layer_detection::detect_transformer_layers;
    use crate::proto::NodeProto;

    fn create_test_model(layer_count: usize, nodes_per_layer: usize) -> ModelProto {
        let mut graph = GraphProto {
            name: "test_model".to_string(),
            ..Default::default()
        };

        // Add graph input
        graph.input.push(ValueInfoProto {
            name: "input_ids".to_string(),
            ..Default::default()
        });

        // Create encoder layers with proper connections
        let mut prev_output = "input_ids".to_string();

        for layer_idx in 0..layer_count {
            let layer_output = format!("layer_{}_output", layer_idx);

            for node_idx in 0..nodes_per_layer {
                let node_input = if node_idx == 0 {
                    prev_output.clone()
                } else {
                    format!("layer_{}_internal_{}", layer_idx, node_idx - 1)
                };

                let node_output = if node_idx == nodes_per_layer - 1 {
                    layer_output.clone()
                } else {
                    format!("layer_{}_internal_{}", layer_idx, node_idx)
                };

                let node = NodeProto {
                    name: format!("encoder.layer.{}.attention.node_{}", layer_idx, node_idx),
                    op_type: "MatMul".to_string(),
                    input: vec![node_input],
                    output: vec![node_output],
                    ..Default::default()
                };
                graph.node.push(node);
            }

            prev_output = layer_output;
        }

        // Add graph output
        graph.output.push(ValueInfoProto {
            name: prev_output,
            ..Default::default()
        });

        ModelProto {
            ir_version: 8,
            opset_import: vec![OperatorSetIdProto {
                domain: String::new(),
                version: 17,
            }],
            producer_name: "test".to_string(),
            producer_version: "1.0".to_string(),
            graph: Some(graph),
            ..Default::default()
        }
    }

    #[test]
    fn test_split_by_layers() {
        let model = create_test_model(4, 3);
        let graph = model.graph.as_ref().unwrap();

        let layers = detect_transformer_layers(graph).unwrap();
        assert_eq!(layers.len(), 4);

        let layer_models = split_by_layers(&model, &layers).unwrap();
        assert_eq!(layer_models.len(), 4);

        for (i, (name, layer_model)) in layer_models.iter().enumerate() {
            assert_eq!(name, &format!("encoder.layer.{}", i));
            let layer_graph = layer_model.graph.as_ref().unwrap();
            assert_eq!(layer_graph.node.len(), 3);
        }
    }

    #[test]
    fn test_layer_inputs_outputs() {
        let model = create_test_model(3, 2);
        let graph = model.graph.as_ref().unwrap();

        let layers = detect_transformer_layers(graph).unwrap();
        let layer_models = split_by_layers(&model, &layers).unwrap();

        // First layer should have input_ids as input
        let (_, first_layer) = &layer_models[0];
        let first_graph = first_layer.graph.as_ref().unwrap();
        assert!(first_graph.input.iter().any(|v| v.name == "input_ids"));

        // Last layer should have the final output
        let (_, last_layer) = &layer_models[2];
        let last_graph = last_layer.graph.as_ref().unwrap();
        assert_eq!(last_graph.output.len(), 1);
    }

    #[test]
    fn test_layer_model_metadata() {
        let model = create_test_model(2, 2);
        let graph = model.graph.as_ref().unwrap();

        let layers = detect_transformer_layers(graph).unwrap();
        let layer_models = split_by_layers(&model, &layers).unwrap();

        let (name, layer_model) = &layer_models[0];
        assert_eq!(name, "encoder.layer.0");
        assert_eq!(layer_model.ir_version, 8);
        assert!(!layer_model.opset_import.is_empty());
        assert!(layer_model.producer_name.contains("layer 0"));
    }

    #[test]
    fn test_split_no_graph_error() {
        let model = ModelProto::default();
        let layers = vec![];

        let result = split_by_layers(&model, &layers);
        assert!(matches!(result, Err(SplitError::NoGraph)));
    }

    #[test]
    fn test_split_empty_layers() {
        let model = create_test_model(2, 2);

        let layer_models = split_by_layers(&model, &[]).unwrap();
        assert!(layer_models.is_empty());
    }

    #[test]
    fn test_create_stub_model() {
        let model = create_test_model(2, 2);
        let stub = create_stub_model(&model, "embedding");

        assert_eq!(stub.ir_version, 8);
        assert!(stub.doc_string.contains("embedding"));
        let graph = stub.graph.as_ref().unwrap();
        assert_eq!(graph.name, "embedding");
    }

    #[test]
    fn test_get_opset_version() {
        let opsets = vec![
            OperatorSetIdProto {
                domain: String::new(),
                version: 17,
            },
            OperatorSetIdProto {
                domain: "com.microsoft".to_string(),
                version: 1,
            },
        ];

        assert_eq!(get_opset_version(&opsets, ""), Some(17));
        assert_eq!(get_opset_version(&opsets, "com.microsoft"), Some(1));
        assert_eq!(get_opset_version(&opsets, "unknown"), None);
    }

    #[test]
    fn test_initializers_extracted() {
        let mut model = create_test_model(2, 2);
        let graph = model.graph.as_mut().unwrap();

        // Add an initializer
        graph.initializer.push(TensorProto {
            name: "weight".to_string(),
            data_type: 1, // FLOAT
            dims: vec![768, 768],
            ..Default::default()
        });

        // Make first node use the initializer
        graph.node[0].input.push("weight".to_string());

        let layers = detect_transformer_layers(graph).unwrap();
        let layer_models = split_by_layers(&model, &layers).unwrap();

        // First layer should have the initializer
        let (_, first_layer) = &layer_models[0];
        let first_graph = first_layer.graph.as_ref().unwrap();
        assert_eq!(first_graph.initializer.len(), 1);
        assert_eq!(first_graph.initializer[0].name, "weight");
    }
}
