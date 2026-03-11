//! Converts a parsed `GraphProto` into an `AiGraph`.

use crate::{
    dtype_map::onnx_dtype,
    onnx_pb::{GraphProto, TensorShapeProto, ValueInfoProto},
    op_map::{map_op, OpContext},
    tensor_map::tensor_to_param,
};
use hologram_ai_common::{
    AiGraph, AiNode, AiOp, DType, Dim, DimVarSource, DimVarTable, ImportWarning, NodeId,
    QuantDescriptor, Shape, TensorId, TensorInfo,
};
use std::collections::HashMap;
use std::path::Path;

/// Build an `AiGraph` from an ONNX `GraphProto`.
///
/// Returns the graph and an oracle map (TensorId → TensorInfo) built from all
/// ONNX `value_info`, `input`, and `output` annotations. The oracle is used by
/// [`ShapeOraclePass`][hologram_ai_common::ShapeOraclePass] to seed any tensor
/// shapes that remain empty after construction.
///
/// `model_dir` is the directory containing the `.onnx` file, used to resolve
/// external data file paths for tensors with `data_location == EXTERNAL`.
pub fn build_ai_graph(
    g: &GraphProto,
    graph_name: &str,
    model_dir: Option<&Path>,
) -> anyhow::Result<(AiGraph, HashMap<TensorId, TensorInfo>)> {
    let mut next_tid: TensorId = 0;
    // name → TensorId mapping.
    let mut name_to_tid: HashMap<String, TensorId> = HashMap::new();
    let mut tensor_info: HashMap<TensorId, TensorInfo> = HashMap::new();
    let mut params = HashMap::new();
    let mut warnings = Vec::new();
    let mut dim_vars = DimVarTable::default();
    let mut subgraphs: HashMap<String, AiGraph> = HashMap::new();

    let mut alloc_tid = |name: &str, name_to_tid: &mut HashMap<String, TensorId>| -> TensorId {
        if let Some(&tid) = name_to_tid.get(name) {
            return tid;
        }
        let tid = next_tid;
        next_tid += 1;
        name_to_tid.insert(name.to_owned(), tid);
        tid
    };

    // ── Initializers (weights) ────────────────────────────────────────────
    for init in &g.initializer {
        let tid = alloc_tid(&init.name, &mut name_to_tid);
        match tensor_to_param(init, model_dir) {
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
    // Collect non-param inputs first (immutable borrow of name_to_tid),
    // then allocate tensor IDs (mutable borrow) in a separate pass.
    let input_vis: Vec<&ValueInfoProto> = g
        .input
        .iter()
        .filter(|vi| !params.contains_key(name_to_tid.get(&vi.name).unwrap_or(&u32::MAX)))
        .collect();
    let graph_inputs_with_names: Vec<(TensorId, String)> = input_vis
        .into_iter()
        .map(|vi| {
            let tid = alloc_tid(&vi.name, &mut name_to_tid);
            let info = value_info_to_tensor_info(vi, &mut dim_vars);
            tensor_info.insert(tid, info);
            (tid, vi.name.clone())
        })
        .collect();
    let graph_inputs: Vec<TensorId> = graph_inputs_with_names.iter().map(|(t, _)| *t).collect();
    let input_names: Vec<String> = graph_inputs_with_names
        .into_iter()
        .map(|(_, n)| n)
        .collect();

    // ── Intermediate tensor shapes (value_info) ──────────────────────────
    // ONNX stores type/shape info for intermediate tensors here.
    for vi in &g.value_info {
        if vi.name.is_empty() {
            continue;
        }
        let tid = alloc_tid(&vi.name, &mut name_to_tid);
        let info = value_info_to_tensor_info(vi, &mut dim_vars);
        // Only insert if not already populated (params/inputs take priority).
        tensor_info.entry(tid).or_insert(info);
    }

    // ── Nodes ─────────────────────────────────────────────────────────────
    let mut nodes: Vec<AiNode> = Vec::new();
    let mut next_nid: NodeId = 0;

    for n in &g.node {
        // Allocate output TensorIds.
        let output_tids: Vec<TensorId> = n
            .output
            .iter()
            .map(|name| alloc_tid(name, &mut name_to_tid))
            .collect();

        let input_tids: Vec<TensorId> = n
            .input
            .iter()
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
                known_i64_values: None,
            });
        }
        for &tid in &input_tids {
            tensor_info.entry(tid).or_insert_with(|| TensorInfo {
                logical_dtype: DType::F32,
                storage_dtype: DType::F32,
                shape: Shape::new(),
                quant: QuantDescriptor::none(),
                known_i64_values: None,
            });
        }

        // Handle Constant nodes: extract the `value` attribute as a param.
        if n.op_type == "Constant" {
            if let Some(tensor_attr) = n.attribute.iter().find(|a| a.name == "value") {
                if let Some(ref tensor_proto) = tensor_attr.t {
                    if let Some(&out_tid) = output_tids.first() {
                        match tensor_to_param(tensor_proto, model_dir) {
                            Ok((param, info)) => {
                                tensor_info.insert(out_tid, info);
                                params.insert(out_tid, param);
                            }
                            Err(e) => {
                                warnings.push(ImportWarning {
                                    message: format!(
                                        "error extracting Constant value '{}': {e}",
                                        n.name
                                    ),
                                    node_name: Some(n.name.clone()),
                                });
                            }
                        }
                    }
                }
            }
            continue;
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

        // ── Recursive subgraph import for control flow ops ─────────────
        if matches!(n.op_type.as_str(), "If" | "Loop" | "Scan") {
            let nid = next_nid.saturating_sub(1);
            import_subgraph_attrs(
                &ctx,
                &n.op_type,
                nid,
                &mut subgraphs,
                &mut warnings,
                model_dir,
            );
            // Rewrite placeholder branch names to actual subgraph keys.
            if let Some(node) = nodes.last_mut() {
                rewrite_subgraph_keys(&mut node.op, nid);
            }
        }
    }

    // Re-resolve graph outputs (tensors may have been allocated during node pass).
    let graph_outputs_with_names: Vec<(TensorId, String)> = g
        .output
        .iter()
        .map(|vi| {
            let tid = alloc_tid(&vi.name, &mut name_to_tid);
            // Populate shape/dtype from output's ValueInfoProto if available.
            let info = value_info_to_tensor_info(vi, &mut dim_vars);
            if !info.shape.is_empty() {
                tensor_info
                    .entry(tid)
                    .and_modify(|existing| {
                        if existing.shape.is_empty() {
                            *existing = info.clone();
                        }
                    })
                    .or_insert(info);
            }
            (tid, vi.name.clone())
        })
        .collect();
    let graph_outputs: Vec<TensorId> = graph_outputs_with_names.iter().map(|(t, _)| *t).collect();
    let output_names: Vec<String> = graph_outputs_with_names
        .into_iter()
        .map(|(_, n)| n)
        .collect();

    // Post-process: resolve op parameters that are inputs in ONNX opset 10+/13+.
    // Slice, Unsqueeze, Squeeze take their axis/start/end parameters as tensor
    // inputs rather than node attributes. We resolve them from constants here.
    resolve_dynamic_op_params(&mut nodes, &params, &tensor_info, &mut warnings);

    // Build the shape oracle from all ONNX-provided annotations.
    // Covers value_info (intermediate tensors), graph inputs, and graph outputs.
    // Only entries with non-empty shapes are included — empty annotations from
    // untyped outputs would just add noise.
    let mut oracle: HashMap<TensorId, TensorInfo> = HashMap::new();
    let all_vis = g
        .value_info
        .iter()
        .chain(g.input.iter())
        .chain(g.output.iter());
    for vi in all_vis {
        if let Some(&tid) = name_to_tid.get(&vi.name) {
            // Skip params: their shapes come from initializer data.
            if params.contains_key(&tid) {
                continue;
            }
            let info = value_info_to_tensor_info(vi, &mut dim_vars);
            if !info.shape.is_empty() {
                oracle.insert(tid, info);
            }
        }
    }

    // Resolve head_dim for ONNX MultiHeadAttention nodes that were given
    // head_dim: 0 in op_map (placeholder; real value comes from oracle shapes).
    resolve_attention_head_dims(&mut nodes, &oracle, &tensor_info);

    Ok((
        AiGraph {
            name: graph_name.to_owned(),
            nodes,
            inputs: graph_inputs,
            outputs: graph_outputs,
            input_names,
            output_names,
            params,
            tensor_info,
            metadata: HashMap::new(),
            warnings,
            dim_vars,
            shape_constraints: Default::default(),
            subgraphs,
        },
        oracle,
    ))
}

/// Resolve `head_dim: 0` for ONNX MultiHeadAttention / GroupedQueryAttention nodes.
///
/// ONNX MHA nodes are imported with `head_dim: 0` because the value is not
/// available as a node attribute — it must be derived from tensor shapes.
///
/// Priority order:
/// 1. Oracle output shape: if the output is `[_, _, hidden]` with concrete
///    `hidden`, then `head_dim = hidden / num_heads`.
/// 2. Query input shape (input[0]): same formula.
/// 3. Leave as 0 and emit a tracing warning (will produce a zero-dim shape
///    later, caught by the compiler's diagnostic pass).
fn resolve_attention_head_dims(
    nodes: &mut [AiNode],
    oracle: &HashMap<TensorId, TensorInfo>,
    tensor_info: &HashMap<TensorId, TensorInfo>,
) {
    for node in nodes.iter_mut() {
        let (num_heads, head_dim) = match &node.op {
            AiOp::MultiHeadAttention { num_heads, head_dim, .. } => (*num_heads, *head_dim),
            AiOp::GroupedQueryAttention { num_heads, head_dim, .. } => (*num_heads, *head_dim),
            _ => continue,
        };

        if head_dim != 0 || num_heads == 0 {
            continue;
        }

        // Try to derive head_dim from the output tensor's oracle shape.
        let resolved = node
            .outputs
            .first()
            .and_then(|tid| oracle.get(tid).or_else(|| tensor_info.get(tid)))
            .and_then(|info| info.shape.last())
            .and_then(|d| d.as_concrete())
            .map(|hidden| hidden / num_heads as u64)
            // Fallback: try the query input (input[0]) shape.
            .or_else(|| {
                node.inputs
                    .first()
                    .and_then(|tid| oracle.get(tid).or_else(|| tensor_info.get(tid)))
                    .and_then(|info| info.shape.last())
                    .and_then(|d| d.as_concrete())
                    .map(|hidden| hidden / num_heads as u64)
            });

        match resolved {
            Some(hd) if hd > 0 => {
                match &mut node.op {
                    AiOp::MultiHeadAttention { head_dim, .. } => *head_dim = hd as u32,
                    AiOp::GroupedQueryAttention { head_dim, .. } => *head_dim = hd as u32,
                    _ => {}
                }
                tracing::debug!(
                    node_id = node.id,
                    num_heads,
                    head_dim = hd,
                    "resolved attention head_dim from oracle"
                );
            }
            _ => {
                tracing::warn!(
                    node_id = node.id,
                    num_heads,
                    "could not resolve head_dim for attention node; left as 0"
                );
            }
        }
    }
}

fn value_info_to_tensor_info(vi: &ValueInfoProto, dim_vars: &mut DimVarTable) -> TensorInfo {
    let (dtype, shape) = match &vi.r#type {
        Some(tp) => match &tp.value {
            Some(crate::onnx_pb::type_proto::Value::TensorType(t)) => {
                let dtype = onnx_dtype(t.elem_type).unwrap_or(DType::F32);
                let shape = t
                    .shape
                    .as_ref()
                    .map(|s| shape_from_shape_proto(s, dim_vars))
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
        known_i64_values: None,
    }
}

/// Resolve op parameters that ONNX opset 10+/13+ provides as tensor inputs
/// rather than node attributes (Slice starts/ends/axes/steps, Unsqueeze axes, etc.).
fn resolve_dynamic_op_params(
    nodes: &mut [AiNode],
    params: &HashMap<TensorId, hologram_ai_common::AiParam>,
    tensor_info: &HashMap<TensorId, TensorInfo>,
    warnings: &mut Vec<ImportWarning>,
) {
    for node in nodes.iter_mut() {
        match &node.op {
            AiOp::Slice {
                axes,
                starts,
                ends,
                ..
            } if axes.is_empty() && starts.is_empty() && ends.is_empty() => {
                // ONNX opset 10+: Slice(data, starts, ends, [axes], [steps])
                // inputs[0] = data, inputs[1] = starts, inputs[2] = ends,
                // inputs[3] = axes (optional), inputs[4] = steps (optional)
                if node.inputs.len() < 3 {
                    continue;
                }
                let starts_vals = extract_i64_const(node.inputs[1], params, tensor_info);
                let ends_vals = extract_i64_const(node.inputs[2], params, tensor_info);
                let axes_vals = if node.inputs.len() > 3 {
                    extract_i64_const(node.inputs[3], params, tensor_info)
                } else {
                    None
                };
                let steps_vals = if node.inputs.len() > 4 {
                    extract_i64_const(node.inputs[4], params, tensor_info)
                } else {
                    None
                };

                match (starts_vals, ends_vals) {
                    (Some(s), Some(e)) => {
                        let axes = axes_vals.unwrap_or_else(|| (0..s.len() as i64).collect());
                        let steps = steps_vals.unwrap_or_else(|| vec![1; s.len()]);
                        node.op = AiOp::Slice {
                            axes,
                            starts: s,
                            ends: e,
                            steps,
                        };
                        // Keep only the data input; remove the constant inputs.
                        node.inputs.truncate(1);
                    }
                    _ => {
                        warnings.push(ImportWarning {
                            message: format!(
                                "Slice node {}: could not resolve starts/ends from constant inputs",
                                node.id
                            ),
                            node_name: None,
                        });
                    }
                }
            }

            AiOp::Unsqueeze { axes } if axes.is_empty() => {
                // ONNX opset 13+: Unsqueeze(data, axes)
                if node.inputs.len() >= 2 {
                    if let Some(axes_vals) = extract_i64_const(node.inputs[1], params, tensor_info)
                    {
                        node.op = AiOp::Unsqueeze { axes: axes_vals };
                        node.inputs.truncate(1);
                    }
                }
            }

            AiOp::Squeeze { axes } if axes.is_empty() => {
                // ONNX opset 13+: Squeeze(data, [axes])
                if node.inputs.len() >= 2 {
                    if let Some(axes_vals) = extract_i64_const(node.inputs[1], params, tensor_info)
                    {
                        node.op = AiOp::Squeeze { axes: axes_vals };
                        node.inputs.truncate(1);
                    }
                }
                // If only 1 input, Squeeze with empty axes = squeeze all size-1 dims (already correct).
            }

            // ONNX opset 18+: ReduceMean/Sum/Max/Min(data, axes) — axes as input tensor.
            AiOp::ReduceMean { axes, keepdims } if axes.is_empty() => {
                let kd = *keepdims;
                if node.inputs.len() >= 2 {
                    if let Some(axes_vals) = extract_i64_const(node.inputs[1], params, tensor_info)
                    {
                        node.op = AiOp::ReduceMean { axes: axes_vals, keepdims: kd };
                        node.inputs.truncate(1);
                    }
                }
            }
            AiOp::ReduceSum { axes, keepdims } if axes.is_empty() => {
                let kd = *keepdims;
                if node.inputs.len() >= 2 {
                    if let Some(axes_vals) = extract_i64_const(node.inputs[1], params, tensor_info)
                    {
                        node.op = AiOp::ReduceSum { axes: axes_vals, keepdims: kd };
                        node.inputs.truncate(1);
                    }
                }
            }
            AiOp::ReduceMax { axes, keepdims } if axes.is_empty() => {
                let kd = *keepdims;
                if node.inputs.len() >= 2 {
                    if let Some(axes_vals) = extract_i64_const(node.inputs[1], params, tensor_info)
                    {
                        node.op = AiOp::ReduceMax { axes: axes_vals, keepdims: kd };
                        node.inputs.truncate(1);
                    }
                }
            }
            AiOp::ReduceMin { axes, keepdims } if axes.is_empty() => {
                let kd = *keepdims;
                if node.inputs.len() >= 2 {
                    if let Some(axes_vals) = extract_i64_const(node.inputs[1], params, tensor_info)
                    {
                        node.op = AiOp::ReduceMin { axes: axes_vals, keepdims: kd };
                        node.inputs.truncate(1);
                    }
                }
            }

            _ => {}
        }
    }
}

/// Extract i64 values from a constant parameter tensor.
fn extract_i64_const(
    tid: TensorId,
    params: &HashMap<TensorId, hologram_ai_common::AiParam>,
    tensor_info: &HashMap<TensorId, TensorInfo>,
) -> Option<Vec<i64>> {
    let param = params.get(&tid)?;
    let info = tensor_info.get(&tid)?;
    let data = match param {
        hologram_ai_common::AiParam::Inline { data, .. } => data.as_slice(),
        _ => return None,
    };

    match info.logical_dtype {
        DType::INT64 => {
            if data.len() % 8 != 0 {
                return None;
            }
            Some(
                data.chunks_exact(8)
                    .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                    .collect(),
            )
        }
        DType::INT32 => {
            if data.len() % 4 != 0 {
                return None;
            }
            Some(
                data.chunks_exact(4)
                    .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as i64)
                    .collect(),
            )
        }
        _ => None,
    }
}

/// Rewrite placeholder branch names in control flow AiOps to the actual
/// subgraph keys (e.g., "then_branch" → "then_branch_42").
fn rewrite_subgraph_keys(op: &mut AiOp, node_id: u32) {
    match op {
        AiOp::If {
            then_branch,
            else_branch,
        } => {
            *then_branch = format!("then_branch_{node_id}");
            if let Some(eb) = else_branch {
                *eb = format!("else_branch_{node_id}");
            }
        }
        AiOp::Loop { body, .. } | AiOp::Scan { body, .. } => {
            *body = format!("body_{node_id}");
        }
        _ => {}
    }
}

/// Recursively import subgraph attributes from control flow ops (If/Loop/Scan).
///
/// Extracts `GraphProto` attributes from the ONNX node, builds child `AiGraph`s
/// via `build_ai_graph`, and stores them in the parent's `subgraphs` map.
/// The keys match the branch names used by the `AiOp::If`/`Loop`/`Scan` variants.
fn import_subgraph_attrs(
    ctx: &OpContext<'_>,
    op_type: &str,
    node_id: u32,
    subgraphs: &mut HashMap<String, AiGraph>,
    warnings: &mut Vec<ImportWarning>,
    model_dir: Option<&Path>,
) {
    let branch_attrs: Vec<(&str, &str)> = match op_type {
        "If" => vec![("then_branch", "then_branch"), ("else_branch", "else_branch")],
        "Loop" => vec![("body", "body")],
        "Scan" => vec![("body", "body")],
        _ => return,
    };

    for (attr_name, key_prefix) in branch_attrs {
        if let Some(graph_proto) = ctx.attr_g(attr_name) {
            let subgraph_key = format!("{key_prefix}_{node_id}");
            match build_ai_graph(graph_proto, &subgraph_key, model_dir) {
                Ok((child_graph, _oracle)) => {
                    tracing::debug!(
                        node_id,
                        op_type,
                        key = %subgraph_key,
                        nodes = child_graph.nodes.len(),
                        "imported subgraph"
                    );
                    subgraphs.insert(subgraph_key, child_graph);
                }
                Err(e) => {
                    warnings.push(ImportWarning {
                        message: format!(
                            "failed to import {attr_name} subgraph for {op_type} node {node_id}: {e}"
                        ),
                        node_name: None,
                    });
                }
            }
        }
    }
}

fn shape_from_shape_proto(s: &TensorShapeProto, dim_vars: &mut DimVarTable) -> Shape {
    s.dim
        .iter()
        .map(|d| {
            match &d.value {
                Some(crate::onnx_pb::tensor_shape_proto::dimension::Value::DimValue(v)) => {
                    Dim::Concrete(*v as u64)
                }
                Some(crate::onnx_pb::tensor_shape_proto::dimension::Value::DimParam(name)) => {
                    // Intern the ONNX dim_param as a named DimVar.
                    let var_id = dim_vars.intern_with_bounds(
                        name,
                        Some(1),
                        None, // upper bound unknown from ONNX alone
                        DimVarSource::Import,
                    );
                    Dim::Var(var_id)
                }
                None => Dim::Dynamic,
            }
        })
        .collect()
}
