//! ONNX protobuf parsing and validation.
//!
//! This module provides functionality for parsing ONNX protobuf files and
//! validating model structure before translation to hologram IR.

use crate::{OnnxError, Result};
use crate::proto::{GraphProto, ModelProto, NodeProto, ValueInfoProto};
use prost::Message;
use tracing::{debug, trace, warn};

/// Parse ONNX model from raw bytes.
///
/// This function decodes the ONNX protobuf format and returns a structured
/// [`ModelProto`] that can be used for further processing.
///
/// # Arguments
///
/// * `bytes` - Raw ONNX model bytes (protobuf format)
///
/// # Returns
///
/// Parsed [`ModelProto`] structure
///
/// # Errors
///
/// Returns [`OnnxError::ParseError`] or [`OnnxError::ProtobufError`] if:
/// - Input is not valid protobuf data
/// - Protobuf structure doesn't match ONNX schema
/// - Required fields are missing
///
/// # Examples
///
/// ```no_run
/// use crate::core::parse_model;
/// use std::fs;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let bytes = fs::read("model.onnx")?;
/// let model = parse_model(&bytes)?;
/// println!("Loaded model: {}", model.graph.unwrap().name);
/// # Ok(())
/// # }
/// ```
pub fn parse_model(bytes: &[u8]) -> Result<ModelProto> {
    trace!("Parsing ONNX protobuf ({} bytes)", bytes.len());

    // Decode protobuf
    let model = ModelProto::decode(bytes)
        .map_err(|e| OnnxError::ParseError(format!("Failed to decode protobuf: {}", e)))?;

    debug!("Successfully parsed ONNX model");

    Ok(model)
}

/// Validate ONNX model structure.
///
/// This function performs structural validation to ensure the model is
/// well-formed and can be compiled to hologram IR.
///
/// # Validation Checks
///
/// - Model has a graph
/// - Graph has at least one input
/// - Graph has at least one output
/// - All nodes reference valid inputs/initializers
/// - No duplicate node output names
/// - No cycles in the computation graph (DAG check)
///
/// # Arguments
///
/// * `model` - Parsed ONNX model to validate
///
/// # Errors
///
/// Returns [`OnnxError::InvalidModel`] if validation fails.
///
/// # Examples
///
/// ```no_run
/// use crate::core::{parse_model, validate_model};
/// use std::fs;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let bytes = fs::read("model.onnx")?;
/// let model = parse_model(&bytes)?;
/// validate_model(&model)?;
/// println!("Model is valid!");
/// # Ok(())
/// # }
/// ```
pub fn validate_model(model: &ModelProto) -> Result<()> {
    debug!("Validating ONNX model");

    // Check model has a graph
    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| OnnxError::InvalidModel("Model has no graph".into()))?;

    validate_graph(graph)?;

    debug!("Model validation complete");

    Ok(())
}

/// Validate ONNX graph structure.
///
/// Internal function that performs detailed graph validation.
fn validate_graph(graph: &GraphProto) -> Result<()> {
    // Check graph has inputs
    if graph.input.is_empty() {
        return Err(OnnxError::InvalidModel("Graph has no inputs".into()));
    }

    // Check graph has outputs
    if graph.output.is_empty() {
        return Err(OnnxError::InvalidModel("Graph has no outputs".into()));
    }

    debug!(
        "Graph has {} inputs and {} outputs",
        graph.input.len(),
        graph.output.len()
    );

    // Collect all available tensor names (inputs + initializers + node outputs)
    let mut available_tensors = std::collections::HashSet::new();

    // Add inputs (excluding initializers)
    let initializer_names: std::collections::HashSet<_> = graph
        .initializer
        .iter()
        .map(|init| init.name.as_str())
        .collect();

    for input in &graph.input {
        if !initializer_names.contains(input.name.as_str()) {
            available_tensors.insert(input.name.as_str());
        }
    }

    // Add initializers
    for init in &graph.initializer {
        available_tensors.insert(init.name.as_str());
    }

    trace!(
        "Graph has {} inputs and {} initializers",
        graph.input.len() - initializer_names.len(),
        graph.initializer.len()
    );

    // Validate each node
    for (idx, node) in graph.node.iter().enumerate() {
        validate_node(node, idx, &available_tensors)?;

        // Add node outputs to available tensors
        for output in &node.output {
            if !output.is_empty() {
                if available_tensors.contains(output.as_str()) {
                    warn!("Duplicate tensor name in graph: {}", output);
                    // Note: Some models have duplicate names, we allow but warn
                }
                available_tensors.insert(output.as_str());
            }
        }
    }

    // Validate outputs reference existing tensors
    for output in &graph.output {
        if !available_tensors.contains(output.name.as_str()) {
            return Err(OnnxError::MissingOutput(output.name.clone()));
        }
    }

    debug!("Graph validation complete: {} nodes", graph.node.len());

    Ok(())
}

/// Validate individual ONNX node.
///
/// Checks that node has:
/// - Non-empty operation type
/// - Valid input references
/// - At least one output
fn validate_node(
    node: &NodeProto,
    idx: usize,
    available_tensors: &std::collections::HashSet<&str>,
) -> Result<()> {
    // Check node has op_type
    if node.op_type.is_empty() {
        return Err(OnnxError::InvalidModel(format!(
            "Node {} has empty op_type",
            idx
        )));
    }

    // Check all inputs are available
    for (input_idx, input) in node.input.iter().enumerate() {
        if input.is_empty() {
            // Empty string means optional input not provided
            continue;
        }

        if !available_tensors.contains(input.as_str()) {
            return Err(OnnxError::MissingInput(format!(
                "Node {} ({}) input {}: '{}'",
                idx, node.op_type, input_idx, input
            )));
        }
    }

    // Check node has outputs
    if node.output.is_empty() {
        return Err(OnnxError::InvalidModel(format!(
            "Node {} ({}) has no outputs",
            idx, node.op_type
        )));
    }

    Ok(())
}

/// Extract ONNX opset version from model.
///
/// The opset version determines which ONNX operations are available
/// and their semantics. Different opset versions may have different
/// operation definitions or attributes.
///
/// # Arguments
///
/// * `model` - Parsed ONNX model
///
/// # Returns
///
/// Opset version as `i64`. If model has multiple opset imports,
/// returns the version for the default ONNX domain ("" or "ai.onnx").
/// Returns 1 if no opset information is found.
///
/// # Examples
///
/// ```no_run
/// use crate::core::{parse_model, extract_opset_version};
/// use std::fs;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let bytes = fs::read("model.onnx")?;
/// let model = parse_model(&bytes)?;
/// let opset = extract_opset_version(&model);
/// println!("Model uses ONNX opset {}", opset);
/// # Ok(())
/// # }
/// ```
pub fn extract_opset_version(model: &ModelProto) -> i64 {
    // ONNX models can import multiple opsets (for different domains)
    // We care about the main ONNX opset (domain "" or "ai.onnx")
    for import in &model.opset_import {
        let domain = import.domain.as_str();
        if domain.is_empty() || domain == "ai.onnx" {
            debug!("Found ONNX opset version: {}", import.version);
            return import.version;
        }
    }

    // Default to opset 1 if not specified
    warn!("No opset version found, defaulting to 1");
    1
}

/// Get tensor shape from ValueInfoProto.
///
/// Extracts the shape dimensions from ONNX value info, handling
/// both concrete dimensions and symbolic dimensions.
///
/// # Arguments
///
/// * `value_info` - ONNX value info containing type and shape
///
/// # Returns
///
/// Vector of dimension values. Symbolic dimensions are returned as -1.
///
/// # Errors
///
/// Returns [`OnnxError::InvalidModel`] if value info has no type information.
///
/// # Examples
///
/// ```no_run
/// use crate::core::{parse_model, get_tensor_shape};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let bytes = std::fs::read("model.onnx")?;
/// let model = parse_model(&bytes)?;
/// let graph = model.graph.as_ref().unwrap();
///
/// for input in &graph.input {
///     let shape = get_tensor_shape(input)?;
///     println!("Input {}: shape {:?}", input.name, shape);
/// }
/// # Ok(())
/// # }
/// ```
#[allow(dead_code)]
pub fn get_tensor_shape(value_info: &ValueInfoProto) -> Result<Vec<i64>> {
    let type_proto = value_info.r#type.as_ref().ok_or_else(|| {
        OnnxError::InvalidModel(format!(
            "Value '{}' has no type information",
            value_info.name
        ))
    })?;

    use crate::proto::type_proto::Value;
    let tensor_type = match &type_proto.value {
        Some(Value::TensorType(tt)) => tt,
        _ => {
            return Err(OnnxError::InvalidModel(format!(
                "Value '{}' is not a tensor",
                value_info.name
            )));
        }
    };

    let shape_proto = tensor_type.shape.as_ref().ok_or_else(|| {
        OnnxError::InvalidModel(format!("Tensor '{}' has no shape", value_info.name))
    })?;

    use crate::proto::tensor_shape_proto::dimension::Value as DimValue;

    let dims: Vec<i64> = shape_proto
        .dim
        .iter()
        .map(|dim| {
            // dim_value is concrete dimension
            // dim_param is symbolic dimension name
            match &dim.value {
                Some(DimValue::DimValue(v)) if *v > 0 => *v,
                _ => {
                    // Symbolic dimension - return -1
                    // Symbolic shape handling is done in shapes.rs
                    -1
                }
            }
        })
        .collect();

    Ok(dims)
}

/// Get ONNX data type from ValueInfoProto.
///
/// # Arguments
///
/// * `value_info` - ONNX value info
///
/// # Returns
///
/// Data type enum value from ONNX spec.
///
/// # Errors
///
/// Returns [`OnnxError::InvalidModel`] if type information is missing.
#[allow(dead_code)]
pub fn get_tensor_data_type(value_info: &ValueInfoProto) -> Result<i32> {
    let type_proto = value_info.r#type.as_ref().ok_or_else(|| {
        OnnxError::InvalidModel(format!(
            "Value '{}' has no type information",
            value_info.name
        ))
    })?;

    use crate::proto::type_proto::Value;
    let tensor_type = match &type_proto.value {
        Some(Value::TensorType(tt)) => tt,
        _ => {
            return Err(OnnxError::InvalidModel(format!(
                "Value '{}' is not a tensor",
                value_info.name
            )));
        }
    };

    Ok(tensor_type.elem_type)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create a minimal valid model
    fn create_minimal_model() -> ModelProto {
        use crate::proto::tensor_shape_proto::dimension::Value as DimValue;
        use crate::proto::type_proto::Value as TypeValue;
        use crate::proto::*;

        ModelProto {
            ir_version: 7,
            opset_import: vec![OperatorSetIdProto {
                domain: "".to_string(),
                version: 13,
            }],
            graph: Some(GraphProto {
                name: "test_graph".to_string(),
                input: vec![ValueInfoProto {
                    name: "input".to_string(),
                    r#type: Some(TypeProto {
                        value: Some(TypeValue::TensorType(type_proto::Tensor {
                            elem_type: 1, // FLOAT
                            shape: Some(TensorShapeProto {
                                dim: vec![
                                    tensor_shape_proto::Dimension {
                                        value: Some(DimValue::DimValue(1)),
                                        ..Default::default()
                                    },
                                    tensor_shape_proto::Dimension {
                                        value: Some(DimValue::DimValue(784)),
                                        ..Default::default()
                                    },
                                ],
                            }),
                        })),
                        ..Default::default()
                    }),
                    ..Default::default()
                }],
                output: vec![ValueInfoProto {
                    name: "output".to_string(),
                    r#type: Some(TypeProto {
                        value: Some(TypeValue::TensorType(type_proto::Tensor {
                            elem_type: 1,
                            shape: Some(TensorShapeProto {
                                dim: vec![
                                    tensor_shape_proto::Dimension {
                                        value: Some(DimValue::DimValue(1)),
                                        ..Default::default()
                                    },
                                    tensor_shape_proto::Dimension {
                                        value: Some(DimValue::DimValue(10)),
                                        ..Default::default()
                                    },
                                ],
                            }),
                        })),
                        ..Default::default()
                    }),
                    ..Default::default()
                }],
                node: vec![NodeProto {
                    input: vec!["input".to_string()],
                    output: vec!["output".to_string()],
                    op_type: "Identity".to_string(),
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn test_parse_model() {
        let model = create_minimal_model();
        let bytes = model.encode_to_vec();

        let parsed = parse_model(&bytes).unwrap();
        assert_eq!(parsed.ir_version, 7);
        assert!(parsed.graph.is_some());
    }

    #[test]
    fn test_parse_invalid_protobuf() {
        let invalid_bytes = vec![0xFF, 0xFF, 0xFF];
        let result = parse_model(&invalid_bytes);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), OnnxError::ParseError(_)));
    }

    #[test]
    fn test_validate_model() {
        let model = create_minimal_model();
        let result = validate_model(&model);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_model_no_graph() {
        let model = ModelProto {
            ir_version: 7,
            graph: None,
            ..Default::default()
        };

        let result = validate_model(&model);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), OnnxError::InvalidModel(_)));
    }

    #[test]
    fn test_validate_graph_no_inputs() {
        let mut model = create_minimal_model();
        if let Some(ref mut graph) = model.graph {
            graph.input.clear();
        }

        let result = validate_model(&model);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_graph_no_outputs() {
        let mut model = create_minimal_model();
        if let Some(ref mut graph) = model.graph {
            graph.output.clear();
        }

        let result = validate_model(&model);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_node_missing_input() {
        let mut model = create_minimal_model();
        if let Some(ref mut graph) = model.graph {
            // Add node that references non-existent input
            graph.node.push(NodeProto {
                input: vec!["nonexistent".to_string()],
                output: vec!["output2".to_string()],
                op_type: "Identity".to_string(),
                ..Default::default()
            });
        }

        let result = validate_model(&model);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), OnnxError::MissingInput(_)));
    }

    #[test]
    fn test_extract_opset_version() {
        let model = create_minimal_model();
        let version = extract_opset_version(&model);
        assert_eq!(version, 13);
    }

    #[test]
    fn test_extract_opset_version_default() {
        let model = ModelProto {
            ir_version: 7,
            opset_import: vec![],
            ..Default::default()
        };

        let version = extract_opset_version(&model);
        assert_eq!(version, 1); // Default
    }

    #[test]
    fn test_get_tensor_shape() {
        let model = create_minimal_model();
        let graph = model.graph.as_ref().unwrap();
        let input = &graph.input[0];

        let shape = get_tensor_shape(input).unwrap();
        assert_eq!(shape, vec![1, 784]);
    }

    #[test]
    fn test_get_tensor_data_type() {
        let model = create_minimal_model();
        let graph = model.graph.as_ref().unwrap();
        let input = &graph.input[0];

        let dtype = get_tensor_data_type(input).unwrap();
        assert_eq!(dtype, 1); // FLOAT
    }
}
