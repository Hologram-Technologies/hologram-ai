//! ONNX importer for hologram-ai.
//!
//! Parses an ONNX model (protobuf binary) and produces a canonical `AiGraph`
//! ready for optimization and lowering. Priority importer for Sprint 001.

use error::OnnxError;
use hologram_ai_common::AiGraph;
use prost::Message;

mod onnx_pb {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

mod dtype_map;
pub mod error;
mod graph_builder;
mod op_map;
mod tensor_map;

/// Options controlling ONNX import behaviour.
#[derive(Debug, Clone, Default)]
pub struct OnnxImportOptions {
    /// Maximum ONNX opset version to accept (0 = no limit).
    pub max_opset: u32,
}

/// Import an ONNX model from a byte slice (protobuf binary format).
///
/// The returned `AiGraph` is not yet optimized — pass it through
/// `OptPipeline::mvp().run()` before lowering.
///
/// Note: models using external data (weights in separate files) must be loaded
/// via [`import_onnx_path`] so the model directory can be resolved.
pub fn import_onnx(bytes: &[u8], opts: OnnxImportOptions) -> anyhow::Result<AiGraph> {
    import_onnx_inner(bytes, opts, None)
}

/// Import an ONNX model from a file path.
///
/// Automatically resolves external data files relative to the model directory.
pub fn import_onnx_path(
    path: &std::path::Path,
    opts: OnnxImportOptions,
) -> anyhow::Result<AiGraph> {
    let bytes =
        std::fs::read(path).map_err(|e| anyhow::anyhow!("reading ONNX file {path:?}: {e}"))?;
    let model_dir = path.parent();
    import_onnx_inner(&bytes, opts, model_dir)
}

fn import_onnx_inner(
    bytes: &[u8],
    _opts: OnnxImportOptions,
    model_dir: Option<&std::path::Path>,
) -> anyhow::Result<AiGraph> {
    let model = onnx_pb::ModelProto::decode(bytes).map_err(OnnxError::Decode)?;

    let graph_proto = model.graph.ok_or(OnnxError::NoGraph)?;
    let graph_name = if model.domain.is_empty() {
        "onnx_model"
    } else {
        &model.domain
    };

    let ai_graph = graph_builder::build_ai_graph(&graph_proto, graph_name, model_dir)?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

    fn minimal_identity_model() -> Vec<u8> {
        use onnx_pb::*;
        let model = ModelProto {
            ir_version: 8,
            graph: Some(GraphProto {
                name: "test".to_string(),
                node: vec![NodeProto {
                    op_type: "Identity".to_string(),
                    input: vec!["x".to_string()],
                    output: vec!["y".to_string()],
                    ..Default::default()
                }],
                input: vec![ValueInfoProto {
                    name: "x".to_string(),
                    r#type: Some(TypeProto {
                        value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                            elem_type: 1,
                            shape: Some(TensorShapeProto {
                                dim: vec![
                                    tensor_shape_proto::Dimension {
                                        value: Some(
                                            tensor_shape_proto::dimension::Value::DimValue(1),
                                        ),
                                    },
                                    tensor_shape_proto::Dimension {
                                        value: Some(
                                            tensor_shape_proto::dimension::Value::DimValue(64),
                                        ),
                                    },
                                ],
                            }),
                        })),
                    }),
                }],
                output: vec![ValueInfoProto {
                    name: "y".to_string(),
                    ..Default::default()
                }],
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
        let g = import_onnx(&bytes, Default::default()).expect("import failed");
        assert_eq!(g.nodes.len(), 1);
        assert!(g.validate().is_empty());
    }

    #[test]
    fn import_rejects_empty_bytes() {
        assert!(import_onnx(&[], Default::default()).is_err());
    }
}
