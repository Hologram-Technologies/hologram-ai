//! ONNX protobuf parsing utilities.

use anyhow::{Context, Result};

use crate::proto;

/// Parse ONNX protobuf bytes into a ModelProto.
pub fn parse_model(bytes: &[u8]) -> Result<proto::ModelProto> {
    use prost::Message;
    proto::ModelProto::decode(bytes).context("Failed to parse ONNX protobuf")
}

/// Extract concrete shape from ONNX ValueInfoProto.
///
/// Dynamic dimensions (DimParam) default to 1 for batch size.
pub fn extract_shape(value_info: &proto::ValueInfoProto) -> Result<Vec<usize>> {
    let type_proto = value_info
        .r#type
        .as_ref()
        .context("ValueInfo has no type")?;

    let tensor_type = type_proto
        .value
        .as_ref()
        .and_then(|v| {
            if let proto::type_proto::Value::TensorType(tt) = v {
                Some(tt)
            } else {
                None
            }
        })
        .context("Not a tensor type")?;

    let shape_proto = tensor_type.shape.as_ref().context("Tensor has no shape")?;

    let mut shape = Vec::new();
    for dim in &shape_proto.dim {
        let value = dim.value.as_ref().context("Dimension has no value")?;

        match value {
            proto::tensor_shape_proto::dimension::Value::DimValue(v) => {
                shape.push(*v as usize);
            }
            proto::tensor_shape_proto::dimension::Value::DimParam(param) => {
                // Dynamic dimension - use sensible defaults based on common patterns
                let default = if param.contains("batch") {
                    1 // batch_size defaults to 1
                } else if param.contains("seq") || param.contains("length") {
                    // sequence_length - use 512 for transformer models
                    512
                } else {
                    // Unknown dynamic dimension - use 1
                    1
                };
                tracing::debug!("Dynamic dimension '{}' defaulting to {}", param, default);
                shape.push(default);
            }
        }
    }

    Ok(shape)
}

/// Extract ONNX opset version from ModelProto.
pub fn extract_opset_version(model: &proto::ModelProto) -> i64 {
    model
        .opset_import
        .iter()
        .find(|o| o.domain.is_empty() || o.domain == "ai.onnx")
        .map(|o| o.version)
        .unwrap_or(1)
}

/// Validate an ONNX model.
///
/// Returns Ok if the model appears valid, Err with details otherwise.
pub fn validate_model(model: &proto::ModelProto) -> Result<()> {
    // Check model has a graph
    let graph = model.graph.as_ref().context("Model has no graph")?;

    // Check graph has at least one input and output
    if graph.input.is_empty() {
        anyhow::bail!("Graph has no inputs");
    }
    if graph.output.is_empty() {
        anyhow::bail!("Graph has no outputs");
    }

    // Check all nodes have op_type
    for node in &graph.node {
        if node.op_type.is_empty() {
            anyhow::bail!("Node '{}' has no op_type", node.name);
        }
    }

    Ok(())
}

/// Extract dtype from ONNX ValueInfoProto.
pub fn extract_dtype(value_info: &proto::ValueInfoProto) -> Result<hologram::compiler::DType> {
    let type_proto = value_info
        .r#type
        .as_ref()
        .context("ValueInfo has no type")?;

    let tensor_type = type_proto
        .value
        .as_ref()
        .and_then(|v| {
            if let proto::type_proto::Value::TensorType(tt) = v {
                Some(tt)
            } else {
                None
            }
        })
        .context("Not a tensor type")?;

    crate::dtypes::from_onnx(tensor_type.elem_type)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_shape_proto(dims: &[i64]) -> proto::TensorShapeProto {
        proto::TensorShapeProto {
            dim: dims
                .iter()
                .map(|&d| proto::tensor_shape_proto::Dimension {
                    value: Some(proto::tensor_shape_proto::dimension::Value::DimValue(d)),
                    ..Default::default()
                })
                .collect(),
        }
    }

    fn create_value_info(name: &str, dims: &[i64], dtype: i32) -> proto::ValueInfoProto {
        proto::ValueInfoProto {
            name: name.to_string(),
            r#type: Some(proto::TypeProto {
                value: Some(proto::type_proto::Value::TensorType(
                    proto::type_proto::Tensor {
                        elem_type: dtype,
                        shape: Some(create_shape_proto(dims)),
                    },
                )),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn test_extract_shape_concrete() {
        let value_info = create_value_info("test", &[2, 3, 4], 1);
        let shape = extract_shape(&value_info).unwrap();
        assert_eq!(shape, vec![2, 3, 4]);
    }

    #[test]
    fn test_extract_shape_dynamic() {
        let value_info = proto::ValueInfoProto {
            name: "test".to_string(),
            r#type: Some(proto::TypeProto {
                value: Some(proto::type_proto::Value::TensorType(
                    proto::type_proto::Tensor {
                        elem_type: 1,
                        shape: Some(proto::TensorShapeProto {
                            dim: vec![
                                proto::tensor_shape_proto::Dimension {
                                    value: Some(
                                        proto::tensor_shape_proto::dimension::Value::DimParam(
                                            "batch".to_string(),
                                        ),
                                    ),
                                    ..Default::default()
                                },
                                proto::tensor_shape_proto::Dimension {
                                    value: Some(
                                        proto::tensor_shape_proto::dimension::Value::DimValue(128),
                                    ),
                                    ..Default::default()
                                },
                            ],
                        }),
                    },
                )),
                ..Default::default()
            }),
            ..Default::default()
        };

        let shape = extract_shape(&value_info).unwrap();
        assert_eq!(shape, vec![1, 128]); // batch defaults to 1
    }
}
