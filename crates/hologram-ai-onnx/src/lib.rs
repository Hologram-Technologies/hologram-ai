//! ONNX importer for hologram-ai.
//!
//! Parses an ONNX model (protobuf binary) and produces a canonical `AiGraph`
//! ready for optimization and lowering. Priority importer for Sprint 001.

use error::OnnxError;
use hologram_ai_common::{
    opt::pipeline::Pass, AiGraph, AiNode, AiOp, DType, DimExpr, ShapeOraclePass, TensorInfo,
};
use prost::Message;

mod onnx_pb {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

mod dtype_map;
pub mod error;
mod graph_builder;
mod op_map;
mod resolve_op_params;
mod tensor_map;

/// Decode a standalone ONNX `TensorProto` (e.g. an `input_0.pb` / `output_0.pb`
/// from the official ONNX backend node-test corpus) into its little-endian
/// byte image, dims, and ONNX data-type tag. Used by the V&V harness to feed
/// and validate against the ONNX operator spec's authoritative test artifacts.
pub fn decode_tensor_proto_bytes(bytes: &[u8]) -> anyhow::Result<(Vec<u8>, Vec<i64>, i32)> {
    let t = onnx_pb::TensorProto::decode(bytes)?;
    let raw = if !t.raw_data.is_empty() {
        t.raw_data.clone()
    } else {
        t.float_data.iter().flat_map(|f| f.to_le_bytes()).collect()
    };
    Ok((raw, t.dims.clone(), t.data_type))
}

/// Decode a standalone `TensorProto` into its `f32` values (from `raw_data` or
/// `float_data`) — the authoritative expected output for numeric comparison.
pub fn decode_tensor_proto_f32(bytes: &[u8]) -> anyhow::Result<Vec<f32>> {
    let t = onnx_pb::TensorProto::decode(bytes)?;
    if !t.raw_data.is_empty() {
        Ok(t.raw_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    } else {
        Ok(t.float_data.clone())
    }
}

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

    // Resolve optional op parameters from weight/input shapes.
    // Must run before shape propagation so Conv kernel_shape etc. are available.
    resolve_op_params::resolve_op_params(&mut ai_graph);

    // Store opset version in graph metadata for downstream passes.
    if opset_version > 0 {
        ai_graph.metadata.insert(
            "opset_version".to_string(),
            hologram_ai_common::ir::graph::MetaValue::Int(opset_version),
        );
    }

    // Surface warnings: a single summary at warn (so the count is visible
    // without scrolling), and each individual warning at debug. Real transformer
    // imports legitimately produce dozens of per-node warnings (e.g. dynamic
    // Slice bounds resolved later by SliceToGather), so logging each at warn
    // floods the CLI; the structured list stays on `graph.warnings` for callers.
    if !ai_graph.warnings.is_empty() {
        tracing::warn!(
            count = ai_graph.warnings.len(),
            "import produced warnings (run with -v / RUST_LOG=debug for detail)"
        );
        for w in &ai_graph.warnings {
            if let Some(ref node) = w.node_name {
                tracing::debug!(node = %node, "{}", w.message);
            } else {
                tracing::debug!("{}", w.message);
            }
        }
    }

    // Apply shape oracle: seed any empty tensor shapes from value_info.
    // This ensures oracle-provided shapes are present before the opt pipeline
    // runs, and the settled-shape protection in ShapePropagation will keep
    // them from being overwritten by AggressiveShapePropagation.
    let ai_graph = ShapeOraclePass::new(oracle).run(ai_graph)?;

    // Inject lm_head if the ONNX export only has last_hidden_state output
    // (i.e., the language model head was not included in the export).
    let ai_graph = inject_lm_head_if_needed(ai_graph);

    Ok(ai_graph)
}

/// Inject a language model head Gemm node for ONNX models that export
/// `last_hidden_state` instead of `logits`.
///
/// Many ONNX exports of decoder-only transformers (e.g., TinyLlama from
/// HuggingFace without the `--task causal-lm` flag) omit the final
/// projection from hidden_size → vocab_size. TinyLlama uses tied weights
/// (lm_head = embed_tokens.weight), so if `embed_tokens.weight` is present
/// in the graph we can reconstruct the missing head:
///
///   logits = last_hidden_state @ embed_tokens.weight^T
///         = [batch, seq, hidden] @ [vocab, hidden]^T
///         = [batch, seq, vocab]
///
/// This is a lossless fix — no approximation or new weights are introduced.
fn inject_lm_head_if_needed(mut graph: AiGraph) -> AiGraph {
    // Only inject when output is last_hidden_state (not already logits).
    let lhs_idx = match graph
        .output_names
        .iter()
        .position(|n| n == "last_hidden_state")
    {
        Some(i) => i,
        None => return graph,
    };
    if graph.output_names.iter().any(|n| n == "logits") {
        return graph;
    }

    // Find the embedding weight — try several name variants across HF export versions.
    // Different exporters use different names for the same weight:
    //   "embed_tokens.weight"        — older HF optimum exports
    //   "model.embed_tokens.weight"  — recent transformers / optimum-onnx exports
    //   "token_embd.weight"          — llama.cpp / GGUF-derived ONNX exports
    const EMBED_WEIGHT_NAMES: &[&str] = &[
        "embed_tokens.weight",
        "model.embed_tokens.weight",
        "token_embd.weight",
    ];
    let embed_tid = match EMBED_WEIGHT_NAMES.iter().find_map(|candidate| {
        graph
            .tensor_names
            .iter()
            .find(|(tid, name)| name.as_str() == *candidate && graph.params.contains_key(tid))
            .map(|(tid, _)| *tid)
    }) {
        Some(t) => t,
        None => {
            tracing::debug!(
                "inject_lm_head: no embedding weight found (tried {:?}); skipping injection",
                EMBED_WEIGHT_NAMES
            );
            return graph;
        }
    };

    // Derive vocab_size from the embedding weight shape [vocab_size, emb_dim].
    let vocab_size = match graph
        .tensor_info
        .get(&embed_tid)
        .and_then(|info| info.shape.first())
        .and_then(|d| d.as_concrete())
    {
        Some(v) if v > 0 => v,
        _ => return graph,
    };

    let lhs_tid = graph.outputs[lhs_idx];

    // Build logits output shape: same as last_hidden_state but last dim = vocab_size.
    let logits_shape = match graph.tensor_info.get(&lhs_tid) {
        Some(info) if info.shape.len() >= 2 => {
            let mut shape = info.shape.clone();
            if let Some(last) = shape.last_mut() {
                *last = DimExpr::Concrete(vocab_size);
            }
            shape
        }
        _ => return graph,
    };

    // hologram_compiler's Gemm lowering reads `m=A.dim(0), k=A.dim(1),
    // n=B.dim(1)` (see hologram-compiler/src/lower.rs:99-101) and *does
    // not honour* `trans_a`/`trans_b` — those flags are silently dropped
    // at the kernel boundary (GemmCall has no trans fields). A Gemm
    // with `trans_b=true` against a rank-3 A and a [vocab, hidden]
    // B would therefore corrupt m/k/n. We emit the transpose
    // explicitly and use `AiOp::MatMul`, whose `desugar_matmul`
    // correctly flattens the leading batch dims (rank-3 A with rank-2
    // B) and produces canonical [m,k]·[k,n] operands for the
    // compiler. The resulting kernel call has m=batch*seq, k=hidden,
    // n=vocab — i.e. the LM-head shape we want.
    let embed_shape = match graph.tensor_info.get(&embed_tid) {
        Some(info) if info.shape.len() == 2 => info.shape.clone(),
        _ => return graph,
    };
    let hidden_dim = match embed_shape[1].as_concrete() {
        Some(v) if v > 0 => v,
        _ => return graph,
    };

    let mut alloc_tid = graph
        .tensor_info
        .keys()
        .chain(graph.params.keys())
        .max()
        .copied()
        .unwrap_or(0);
    let mut alloc_nid = graph.nodes.iter().map(|n| n.id).max().unwrap_or(0);
    let mut next_tid = || {
        alloc_tid += 1;
        alloc_tid
    };
    let mut next_nid = || {
        alloc_nid += 1;
        alloc_nid
    };

    // embed_tokens.weight is [vocab, hidden]; the LM-head needs
    // it as [hidden, vocab]. Use a swap-last-two Transpose so
    // TransposeMatMulFusion does not re-absorb it into a
    // Gemm{trans_b} (this rewrite is the inverse of that fusion).
    let weight_t_tid = next_tid();
    let mut weight_t_shape = hologram_ai_common::Shape::new();
    weight_t_shape.push(embed_shape[1].clone());
    weight_t_shape.push(embed_shape[0].clone());
    graph.tensor_info.insert(
        weight_t_tid,
        TensorInfo::new(DType::F32, weight_t_shape.clone()),
    );
    graph
        .tensor_names
        .insert(weight_t_tid, "lm_head.weight.T".to_string());
    graph.nodes.push(AiNode::new(
        next_nid(),
        AiOp::Transpose { perm: vec![1, 0] },
        vec![embed_tid],
        vec![weight_t_tid],
    ));

    // logits = last_hidden_state @ weight^T  →  MatMul handles
    // rank-3 A by flattening batch+seq into the matmul rows
    // (desugar_matmul, builder.rs:1060), producing the canonical
    // [batch*seq, hidden] · [hidden, vocab] kernel call.
    let _ = hidden_dim; // kept for the asserting comment above
    let logits_tid = next_tid();
    graph
        .tensor_info
        .insert(logits_tid, TensorInfo::new(DType::F32, logits_shape));
    graph
        .tensor_names
        .insert(logits_tid, "logits".to_string());
    graph.nodes.push(AiNode::new(
        next_nid(),
        AiOp::MatMul,
        vec![lhs_tid, weight_t_tid],
        vec![logits_tid],
    ));

    graph.outputs[lhs_idx] = logits_tid;
    graph.output_names[lhs_idx] = "logits".to_string();

    tracing::info!(
        vocab_size,
        "injected lm_head (embed_tokens.weight) for ONNX model missing logits output"
    );

    graph
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

    /// `inject_lm_head_if_needed` should inject a Gemm node and rename the output
    /// to "logits" when `embed_tokens.weight` is present as a param and the graph
    /// output is named "last_hidden_state".
    #[test]
    fn inject_lm_head_activates_when_weight_present() {
        use hologram_ai_common::{shape_from_concrete, AiParam};
        use std::collections::HashMap;

        let lhs_tid: u32 = 0;
        let embed_tid: u32 = 1;

        let lhs_info = TensorInfo::new(DType::F32, shape_from_concrete(&[1u64, 4, 2048]));
        let embed_info = TensorInfo::new(DType::F32, shape_from_concrete(&[32000u64, 2048]));
        // Minimal non-empty byte slice so AiParam::is_empty() returns false.
        let embed_param = AiParam::inline(vec![0u8; 4], embed_info.clone());

        let mut tensor_info = HashMap::new();
        tensor_info.insert(lhs_tid, lhs_info);
        tensor_info.insert(embed_tid, embed_info);

        let mut params = HashMap::new();
        params.insert(embed_tid, embed_param);

        let mut tensor_names = HashMap::new();
        tensor_names.insert(lhs_tid, "last_hidden_state".to_string());
        tensor_names.insert(embed_tid, "embed_tokens.weight".to_string());

        let graph = AiGraph {
            name: "test".to_string(),
            nodes: vec![],
            inputs: vec![lhs_tid],
            outputs: vec![lhs_tid],
            input_names: vec!["last_hidden_state".to_string()],
            output_names: vec!["last_hidden_state".to_string()],
            params,
            tensor_info,
            tensor_names,
            metadata: Default::default(),
            warnings: vec![],
            dim_vars: Default::default(),
            shape_constraints: Default::default(),
            subgraphs: Default::default(),
            topo_cache: Default::default(),
        };

        let result = inject_lm_head_if_needed(graph);

        assert_eq!(
            result.output_names[0], "logits",
            "output should be renamed to logits"
        );
        // The new output tensor should have vocab_size=32000 as its last dimension.
        let out_tid = result.outputs[0];
        let out_shape = &result.tensor_info[&out_tid].shape;
        let last_dim = out_shape
            .last()
            .and_then(|d| d.as_concrete())
            .expect("last dim should be concrete vocab_size");
        assert_eq!(last_dim, 32000);
        // Transpose(W) + MatMul nodes should have been appended.
        // (Gemm{trans_b: true} cannot be used here — hologram_compiler's
        // Gemm lowering silently drops trans_b; see compiler boundary
        // discussion in `inject_lm_head_if_needed`.)
        assert_eq!(result.nodes.len(), 2);
        assert!(matches!(
            result.nodes[0].op,
            AiOp::Transpose { ref perm } if perm.as_slice() == [1, 0]
        ));
        assert!(matches!(result.nodes[1].op, AiOp::MatMul));
    }

    /// `inject_lm_head_if_needed` must be a no-op when `embed_tokens.weight` is
    /// absent from params — the graph must be returned unchanged.
    #[test]
    fn inject_lm_head_no_op_when_weight_absent() {
        use hologram_ai_common::shape_from_concrete;
        use std::collections::HashMap;

        let lhs_tid: u32 = 0;
        let lhs_info = TensorInfo::new(DType::F32, shape_from_concrete(&[1u64, 4, 2048]));

        let mut tensor_info = HashMap::new();
        tensor_info.insert(lhs_tid, lhs_info);

        let mut tensor_names = HashMap::new();
        tensor_names.insert(lhs_tid, "last_hidden_state".to_string());

        let graph = AiGraph {
            name: "test".to_string(),
            nodes: vec![],
            inputs: vec![lhs_tid],
            outputs: vec![lhs_tid],
            input_names: vec!["last_hidden_state".to_string()],
            output_names: vec!["last_hidden_state".to_string()],
            params: Default::default(),
            tensor_info,
            tensor_names,
            metadata: Default::default(),
            warnings: vec![],
            dim_vars: Default::default(),
            shape_constraints: Default::default(),
            subgraphs: Default::default(),
            topo_cache: Default::default(),
        };

        let result = inject_lm_head_if_needed(graph);

        assert_eq!(
            result.output_names[0], "last_hidden_state",
            "output name must remain last_hidden_state when weight absent"
        );
        assert_eq!(
            result.outputs[0], lhs_tid,
            "output tensor id must be unchanged"
        );
        assert!(
            result.nodes.is_empty(),
            "no nodes should have been injected"
        );
    }
}
