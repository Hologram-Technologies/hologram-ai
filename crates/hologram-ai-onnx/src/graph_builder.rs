//! Converts a parsed `GraphProto` into an `AiGraph`.

use std::collections::HashMap;
use anyhow::Context;
use hologram_ai_common::{
    AiGraph, AiNode, AiOp, DType, TensorId, NodeId,
    TensorInfo, ImportWarning, shape_from_concrete, QuantDescriptor,
    Dim, Shape,
};
use crate::{
    onnx_pb::{GraphProto, ValueInfoProto, TensorShapeProto},
    tensor_map::tensor_to_param,
    dtype_map::onnx_dtype,
    op_map::{map_op, OpContext},
};

/// Build an `AiGraph` from an ONNX `GraphProto`.
pub fn build_ai_graph(g: &GraphProto, graph_name: &str) -> anyhow::Result<AiGraph> {
    let mut next_tid: TensorId = 0;
    // name → TensorId mapping.
    let mut name_to_tid: HashMap<String, TensorId> = HashMap::new();
    let mut tensor_info: HashMap<TensorId, TensorInfo> = HashMap::new();
    let mut params = HashMap::new();
    let mut warnings = Vec::new();

    let mut alloc_tid = |name: &str, name_to_tid: &mut HashMap<String, TensorId>| -> TensorId {
        if let Some(&tid) = name_to_tid.get(name) { return tid; }
        let tid = next_tid;
        next_tid += 1;
        name_to_tid.insert(name.to_owned(), tid);
        tid
    };

    // ── Initializers (weights) ────────────────────────────────────────────
    for init in &g.initializer {
        let tid = alloc_tid(&init.name, &mut name_to_tid);
        match tensor_to_param(init) {
            Ok((param, info)) => {
                tensor_info.insert(tid, info);
                params.insert(tid, param);
            }
            Err(e) => {
                warnings.push(ImportWarning {
                    message: format!("skipping initializer '{}': {e}", init.name),
                    node_name: Some(init.name.clone()),
                });
            }
        }
    }

    // ── Graph inputs ──────────────────────────────────────────────────────
    let graph_inputs: Vec<TensorId> = g.input.iter()
        .filter(|vi| !params.contains_key(name_to_tid.get(&vi.name).unwrap_or(&u32::MAX)))
        .map(|vi| {
            let tid = alloc_tid(&vi.name, &mut name_to_tid);
            let info = value_info_to_tensor_info(vi);
            tensor_info.insert(tid, info);
            tid
        })
        .collect();

    // ── Graph outputs ─────────────────────────────────────────────────────
    let graph_outputs: Vec<TensorId> = g.output.iter()
        .map(|vi| alloc_tid(&vi.name, &mut name_to_tid))
        .collect();

    // ── Nodes ─────────────────────────────────────────────────────────────
    let mut nodes: Vec<AiNode> = Vec::new();
    let mut next_nid: NodeId = 0;

    for n in &g.node {
        // Allocate output TensorIds.
        let output_tids: Vec<TensorId> = n.output.iter()
            .map(|name| alloc_tid(name, &mut name_to_tid))
            .collect();

        let input_tids: Vec<TensorId> = n.input.iter()
            .filter(|name| !name.is_empty())
            .map(|name| alloc_tid(name, &mut name_to_tid))
            .collect();

        // Add placeholder TensorInfo for any not-yet-seen outputs.
        for &tid in &output_tids {
            tensor_info.entry(tid).or_insert_with(|| TensorInfo {
                logical_dtype: DType::F32,
                storage_dtype: DType::F32,
                shape: Shape::new(),
                quant: QuantDescriptor::none(),
            });
        }
        for &tid in &input_tids {
            tensor_info.entry(tid).or_insert_with(|| TensorInfo {
                logical_dtype: DType::F32,
                storage_dtype: DType::F32,
                shape: Shape::new(),
                quant: QuantDescriptor::none(),
            });
        }

        let ctx = OpContext {
            op_type: &n.op_type,
            domain: &n.domain,
            attrs: &n.attribute,
        };

        match map_op(&ctx) {
            Ok(Some(op)) => {
                if matches!(op, AiOp::Opaque { ref op_type, .. } if !op_type.is_empty()) {
                    warnings.push(ImportWarning {
                        message: format!("unsupported op '{}' mapped to Opaque", n.op_type),
                        node_name: Some(n.name.clone()),
                    });
                }
                let nid = next_nid;
                next_nid += 1;
                nodes.push(AiNode::new(nid, op, input_tids, output_tids));
            }
            Ok(None) => {
                // Intentional no-op (e.g. Dropout at inference).
                tracing::debug!(op_type = %n.op_type, "skipping no-op");
            }
            Err(e) => {
                warnings.push(ImportWarning {
                    message: format!("error mapping op '{}': {e}", n.op_type),
                    node_name: Some(n.name.clone()),
                });
            }
        }
    }

    // Re-resolve graph outputs (they may not have been allocated yet before node pass).
    let graph_outputs: Vec<TensorId> = g.output.iter()
        .map(|vi| alloc_tid(&vi.name, &mut name_to_tid))
        .collect();

    Ok(AiGraph {
        name: graph_name.to_owned(),
        nodes,
        inputs: graph_inputs,
        outputs: graph_outputs,
        params,
        tensor_info,
        metadata: HashMap::new(),
        warnings,
    })
}

fn value_info_to_tensor_info(vi: &ValueInfoProto) -> TensorInfo {
    let (dtype, shape) = match &vi.r#type {
        Some(tp) => match &tp.value {
            Some(crate::onnx_pb::type_proto::Value::TensorType(t)) => {
                let dtype = onnx_dtype(t.elem_type).unwrap_or(DType::F32);
                let shape = t.shape.as_ref()
                    .map(|s| shape_from_shape_proto(s))
                    .unwrap_or_default();
                (dtype, shape)
            }
            _ => (DType::F32, Shape::new()),
        },
        None => (DType::F32, Shape::new()),
    };

    TensorInfo {
        logical_dtype: dtype,
        storage_dtype: dtype,
        shape,
        quant: QuantDescriptor::none(),
    }
}

fn shape_from_shape_proto(s: &TensorShapeProto) -> Shape {
    s.dim.iter().map(|d| {
        match &d.value {
            Some(crate::onnx_pb::tensor_shape_proto::dimension::Value::DimValue(v)) => {
                Dim::Concrete(*v as u64)
            }
            Some(crate::onnx_pb::tensor_shape_proto::dimension::Value::DimParam(p)) => {
                Dim::Symbolic(p.clone())
            }
            None => Dim::Dynamic,
        }
    }).collect()
}
