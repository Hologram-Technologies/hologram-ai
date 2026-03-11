//! ONNX importer for hologram-ai.
//!
//! Parses an ONNX model (protobuf binary) and produces a canonical `AiGraph`
//! ready for optimization and lowering. Priority importer for Sprint 001.

use error::OnnxError;
use hologram_ai_common::{opt::pipeline::Pass, AiGraph, ShapeOraclePass};
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
    opts: OnnxImportOptions,
    model_dir: Option<&std::path::Path>,
) -> anyhow::Result<AiGraph> {
    let model = onnx_pb::ModelProto::decode(bytes).map_err(OnnxError::Decode)?;

    // Parse opset version from opset_import (empty domain = standard ONNX ops).
    let opset_version = model
        .opset_import
        .iter()
        .find(|osi| osi.domain.is_empty())
        .map(|osi| osi.version)
        .unwrap_or(0);

    // Enforce max_opset if configured.
    if opts.max_opset > 0 && opset_version > opts.max_opset as i64 {
        anyhow::bail!(
            "ONNX model requires opset {} but max_opset is {}",
            opset_version,
            opts.max_opset
        );
    }

    let graph_proto = model.graph.ok_or(OnnxError::NoGraph)?;
    let graph_name = if model.domain.is_empty() {
        "onnx_model"
    } else {
        &model.domain
    };

    let (mut ai_graph, oracle) =
        graph_builder::build_ai_graph(&graph_proto, graph_name, model_dir)?;

    // Store opset version in graph metadata for downstream passes.
    if opset_version > 0 {
        ai_graph.metadata.insert(
            "opset_version".to_string(),
            hologram_ai_common::ir::graph::MetaValue::Int(opset_version),
        );
    }

    // Surface warnings.
    for w in &ai_graph.warnings {
        if let Some(ref node) = w.node_name {
            tracing::warn!(node = %node, "{}", w.message);
        } else {
            tracing::warn!("{}", w.message);
        }
    }

    // Apply shape oracle: seed any empty tensor shapes from value_info.
    // This ensures oracle-provided shapes are present before the opt pipeline
    // runs, and the settled-shape protection in ShapePropagation will keep
    // them from being overwritten by AggressiveShapePropagation.
    let ai_graph = ShapeOraclePass::new(oracle).run(ai_graph)?;

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

    /// An intermediate tensor whose shape appears in value_info should have
    /// that shape filled by the oracle after import — even when the tensor's
    /// shape wasn't populated during node processing.
    #[test]
    fn oracle_fills_intermediate_tensor_shape() {
        use onnx_pb::*;
        // Build a two-node graph: Relu(x) -> y -> Relu(y) -> z
        // with shape annotations for both intermediate tensors.
        let dim_val = |v: i64| tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimValue(v)),
        };
        let tensor_type = |elem: i32, dims: Vec<i64>| TypeProto {
            value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                elem_type: elem,
                shape: Some(TensorShapeProto {
                    dim: dims.into_iter().map(dim_val).collect(),
                }),
            })),
        };

        let model = ModelProto {
            ir_version: 8,
            graph: Some(GraphProto {
                name: "test".to_string(),
                node: vec![
                    NodeProto {
                        op_type: "Relu".to_string(),
                        input: vec!["x".to_string()],
                        output: vec!["y".to_string()],
                        ..Default::default()
                    },
                    NodeProto {
                        op_type: "Relu".to_string(),
                        input: vec!["y".to_string()],
                        output: vec!["z".to_string()],
                        ..Default::default()
                    },
                ],
                input: vec![ValueInfoProto {
                    name: "x".to_string(),
                    r#type: Some(tensor_type(1, vec![2, 8])),
                }],
                output: vec![ValueInfoProto {
                    name: "z".to_string(),
                    r#type: Some(tensor_type(1, vec![2, 8])),
                }],
                // Intermediate tensor 'y' has its shape in value_info.
                value_info: vec![ValueInfoProto {
                    name: "y".to_string(),
                    r#type: Some(tensor_type(1, vec![2, 8])),
                }],
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut buf = Vec::new();
        model.encode(&mut buf).unwrap();

        let g = import_onnx(&buf, Default::default()).expect("import failed");

        // The intermediate tensor 'y' should have shape [2, 8] from the oracle.
        let y_tid = g.nodes[0].outputs[0];
        let y_shape = &g.tensor_info[&y_tid].shape;
        assert_eq!(y_shape.len(), 2);
        assert_eq!(y_shape[0].as_concrete(), Some(2));
        assert_eq!(y_shape[1].as_concrete(), Some(8));
    }
}
