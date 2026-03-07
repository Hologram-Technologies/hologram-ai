//! ONNX importer for hologram-ai.
//!
//! Parses an ONNX model (protobuf binary) and produces a canonical `AiGraph`
//! ready for optimization and lowering. Priority importer for Sprint 001.

use prost::Message;
use hologram_ai_common::AiGraph;
use error::OnnxError;

mod onnx_pb {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

mod dtype_map;
mod op_map;
mod tensor_map;
mod graph_builder;
pub mod error;

/// Import an ONNX model from a byte slice (protobuf binary format).
///
/// The returned `AiGraph` is not yet optimized — pass it through
/// `OptPipeline::mvp().run()` before lowering.
pub fn import_onnx(bytes: &[u8]) -> anyhow::Result<AiGraph> {
    let model = onnx_pb::ModelProto::decode(bytes)
        .map_err(OnnxError::Decode)?;

    let graph_proto = model.graph.ok_or(OnnxError::NoGraph)?;
    let graph_name  = model.domain.as_deref().unwrap_or("onnx_model");

    let ai_graph = graph_builder::build_ai_graph(&graph_proto, graph_name)?;

    // Surface warnings.
    for w in &ai_graph.warnings {
        if let Some(ref node) = w.node_name {
            tracing::warn!(node = %node, "{}", w.message);
        } else {
            tracing::warn!("{}", w.message);
        }
    }

    Ok(ai_graph)
}

/// Import an ONNX model from a file path.
pub fn import_onnx_path(path: &std::path::Path) -> anyhow::Result<AiGraph> {
    let bytes = std::fs::read(path)
        .map_err(|e| anyhow::anyhow!("reading ONNX file {path:?}: {e}"))?;
    import_onnx(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

    /// Build a minimal ONNX model with a single Identity op.
    fn minimal_identity_model() -> Vec<u8> {
        use onnx_pb::*;
        let model = ModelProto {
            ir_version: 8,
            graph: Some(GraphProto {
                name: "test".to_string(),
                node: vec![NodeProto {
                    op_type: "Identity".to_string(),
                    input:  vec!["x".to_string()],
                    output: vec!["y".to_string()],
                    ..Default::default()
                }],
                input: vec![ValueInfoProto {
                    name: "x".to_string(),
                    r#type: Some(TypeProto {
                        value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                            elem_type: 1, // FLOAT
                            shape: Some(TensorShapeProto {
                                dim: vec![
                                    tensor_shape_proto::Dimension {
                                        value: Some(tensor_shape_proto::dimension::Value::DimValue(1)),
                                    },
                                    tensor_shape_proto::Dimension {
                                        value: Some(tensor_shape_proto::dimension::Value::DimValue(64)),
                                    },
                                ],
                            }),
                        })),
                    }),
                }],
                output: vec![ValueInfoProto { name: "y".to_string(), ..Default::default() }],
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut buf = Vec::new();
        model.encode(&mut buf).unwrap();
        buf
    }

    #[test]
    fn import_identity_model() {
        let bytes = minimal_identity_model();
        let g = import_onnx(&bytes).expect("import failed");
        assert_eq!(g.nodes.len(), 1);
        assert!(g.validate().is_empty());
    }

    #[test]
    fn import_rejects_empty_bytes() {
        assert!(import_onnx(&[]).is_err());
    }
}
