//! Builds a `hologram::Graph` from a dispatched `AiGraph`.
//!
//! Uses `hologram::GraphBuilder` (fluent, index-based): each node-adding method
//! increments the builder's index counter; `tid_to_idx` maps `TensorId` → builder index.

use super::dispatch::{dispatch, DispatchTarget};
use super::strategy::{
    ai_dtype_to_float_dtype, input_float_dtype, ConcreteStrategy, DeferredStrategy,
    LoweringStrategy,
};
use super::LowerPhase;
use crate::exec_context::{ContextBundle, NodeShapeRecipe, ShapeRecipeSection};
use crate::ir::{AiGraph, AiOp, Dim, DimVarId, TensorId, TensorInfo};
use crate::mem::KvCacheLayout;
use anyhow::Context;
use hologram::{ConstantData, GraphBuilder, GraphOp};
use std::collections::HashMap;

// ── Public types ──────────────────────────────────────────────────────────────

/// Options controlling lowering behaviour.
pub struct LoweringOptions {
    pub quant_strategy: QuantStrategy,
}

impl Default for LoweringOptions {
    fn default() -> Self {
        Self {
            quant_strategy: QuantStrategy::Auto,
        }
    }
}

/// Quantized weight dequantization strategy.
pub enum QuantStrategy {
    /// Auto-detect from backend capabilities.
    Auto,
    /// Always dequantize eagerly at plan start.
    EagerDequant,
    /// Use fused quantized kernels where available.
    FusedKernels,
}

/// Output of the lowering pass.
pub struct LoweringOutput {
    pub graph: hologram::Graph,
    /// Layer name for archive metadata (e.g. "lm.prefill", "model.forward").
    pub layer_name: String,
    /// All context sections produced during lowering (shape recipes, etc.).
    pub context: ContextBundle,
}

// ── Entry point ───────────────────────────────────────────────────────────────

/// Lower an optimised `AiGraph` to `hologram::Graph`.
///
/// All ops emit native `GraphOp` variants — no `CustomOpRegistry` needed.
/// Does NOT call `hologram::compile()` — that is the caller's responsibility.
pub fn lower(
    ai_graph: &AiGraph,
    _kv_layout: &KvCacheLayout,
    _opts: &LoweringOptions,
    phase: &LowerPhase,
) -> anyhow::Result<LoweringOutput> {
    let mut builder = GraphBuilder::new();

    // Map AiGraph TensorId → builder node index.
    let mut tid_to_idx: HashMap<TensorId, usize> = HashMap::new();

    // Build dim_var_names mapping: DimVarId → index in recipe dim_vars list.
    let dim_var_names: HashMap<DimVarId, u32> = ai_graph
        .dim_vars
        .iter()
        .map(|(id, _entry)| (id, id.0))
        .collect();

    // Collect dim var names for the recipe section.
    let recipe_dim_vars: Vec<String> = ai_graph
        .dim_vars
        .iter()
        .map(|(_, entry)| entry.name.clone())
        .collect();

    // Strategy chain: try concrete first, then deferred.
    let strategies: Vec<Box<dyn LoweringStrategy>> =
        vec![Box::new(ConcreteStrategy), Box::new(DeferredStrategy)];

    // Collect shape recipes from deferred lowerings.
    let mut node_recipes: Vec<NodeShapeRecipe> = Vec::new();

    // Register named graph inputs and insert Input nodes.
    for (i, &tid) in ai_graph.inputs.iter().enumerate() {
        let name = ai_graph.input_name(i);
        builder = builder.input(name);
        builder = builder.node_from_graph_input(GraphOp::Input, i as u32);
        let idx = builder.len() - 1;
        if let Some(shape) = output_shape(Some(&tid), &ai_graph.tensor_info) {
            builder = builder.set_node_shape(idx, shape);
        }
        let dtype = input_float_dtype(Some(&tid), &ai_graph.tensor_info);
        builder = builder.set_node_dtype(idx, dtype);
        tid_to_idx.insert(tid, idx);
    }

    // Insert constant param nodes (weights, biases).
    let mut sorted_params: Vec<_> = ai_graph.params.iter().collect();
    sorted_params.sort_by_key(|(&tid, _)| tid);

    let mut mmap_offset: u64 = 0;
    for (&tid, param) in sorted_params.iter() {
        let constant = match param {
            crate::ir::AiParam::Mmap { len, .. } => {
                let d = ConstantData::Deferred {
                    byte_size: *len,
                    source_id: mmap_offset,
                };
                mmap_offset += *len;
                d
            }
            _ => {
                let data = param_bytes_owned(param)?;
                ConstantData::Bytes(data)
            }
        };
        let shape = param_shape(param, tid, &ai_graph.tensor_info);
        if let Some(shape) = shape {
            builder = builder.constant_with_shape(constant, shape);
        } else {
            // Always emit a shape even when param_shape fails.
            // Compute from byte size and dtype to avoid 1-D fallback at runtime.
            let byte_sz = match param {
                crate::ir::AiParam::Inline { data, .. } => data.len() as u64,
                crate::ir::AiParam::Mmap { len, .. } => *len,
            };
            let info = match param {
                crate::ir::AiParam::Inline { info, .. } => info,
                crate::ir::AiParam::Mmap { info, .. } => info,
            };
            // Use tensor_info shape with 0-sentinels for symbolic dims (same as output_shape).
            let shape_from_info = output_shape(Some(&tid), &ai_graph.tensor_info)
                .or_else(|| {
                    if !info.shape.is_empty() {
                        Some(info.shape.iter().map(|d| match d {
                            Dim::Concrete(n) => *n as usize,
                            _ => 0,
                        }).collect())
                    } else {
                        None
                    }
                });
            if let Some(shape) = shape_from_info {
                tracing::warn!("param_shape failed for T{tid}, using output_shape with sentinels: {shape:?}");
                builder = builder.constant_with_shape(constant, shape);
            } else {
                // Last resort: 1-D shape from byte size / dtype elem size.
                let elem_sz = info.logical_dtype.byte_size().unwrap_or(4).max(1);
                let elems = byte_sz as usize / elem_sz;
                tracing::warn!("no shape for T{tid}, inferring 1-D [{elems}]");
                builder = builder.constant_with_shape(constant, vec![elems]);
            }
        }
        let builder_idx = builder.len() - 1;
        // Emit dtype for constants using the param's own dtype (not tensor_info).
        // tensor_info may reflect a downstream Cast's output type, but the
        // stored data is in the param's original format.
        let param_info = match param {
            crate::ir::AiParam::Inline { info, .. } => info,
            crate::ir::AiParam::Mmap { info, .. } => info,
        };
        let dtype = ai_dtype_to_float_dtype(&param_info.logical_dtype);
        builder = builder.set_node_dtype(builder_idx, dtype);
        // Diagnostic: check for shape/size mismatch.
        let byte_sz = match param {
            crate::ir::AiParam::Inline { data, .. } => data.len(),
            crate::ir::AiParam::Mmap { len, .. } => *len as usize,
        };
        let shape_ref = param_shape(param, tid, &ai_graph.tensor_info);
        let shape_elems: usize = shape_ref.as_ref().map(|s| s.iter().product()).unwrap_or(0);
        let expected_bytes = shape_elems * dtype.byte_size();
        if shape_elems > 0 && byte_sz != expected_bytes {
            let param_dtype = match param {
                crate::ir::AiParam::Inline { info, .. } => format!("{:?}", info.logical_dtype),
                crate::ir::AiParam::Mmap { info, .. } => format!("{:?}", info.logical_dtype),
            };
            tracing::warn!("constant T{tid} idx={builder_idx} shape/size mismatch: shape={shape_ref:?} elems={shape_elems} expected_bytes={expected_bytes} actual={byte_sz} dtype={dtype:?} param_dtype={param_dtype}");
        }
        tid_to_idx.insert(tid, builder_idx);
    }

    // Emit each node in topological order.
    let topo = ai_graph.topo_order();
    let node_map: HashMap<u32, &_> = ai_graph.nodes.iter().map(|n| (n.id, n)).collect();

    for nid in topo {
        let node = node_map[&nid];

        let input_idxs: Vec<usize> = node
            .inputs
            .iter()
            .map(|tid| {
                tid_to_idx
                    .get(tid)
                    .copied()
                    .with_context(|| format!("missing builder index for tensor {tid}"))
            })
            .collect::<anyhow::Result<_>>()?;

        // ONNX Gather has (data, indices) but hologram executor expects
        // (indices, data). Swap inputs for Gather/GatherElements.
        let input_idxs = swap_gather_inputs(&node.op, input_idxs);

        match dispatch(&node.op) {
            DispatchTarget::GraphOp(graph_op) => {
                builder = builder.node_with_inputs(graph_op, &input_idxs);
                let idx = builder.len() - 1;
                if let Some(&tid) = node.outputs.first() {
                    let out_shape = output_shape(Some(&tid), &ai_graph.tensor_info);
                    let inferred = infer_reshape_shape(
                        &node.op,
                        &node.inputs,
                        node.outputs.first().copied(),
                        ai_graph,
                    );
                    let shape = match (&out_shape, &inferred) {
                        (Some(os), Some(inf)) if os.len() == inf.len() => {
                            let os_zeros = os.iter().filter(|&&d| d == 0).count();
                            let inf_zeros = inf.iter().filter(|&&d| d == 0).count();
                            if inf_zeros < os_zeros {
                                inferred
                            } else {
                                out_shape
                            }
                        }
                        _ => out_shape.or(inferred),
                    };
                    if let Some(ref s) = shape {
                        builder = builder.set_node_shape(idx, s.clone());
                    } else {
                        tracing::warn!("no shape for GraphOp node {} (AiOp {:?}, T{tid}, idx={idx})", node.id, node.op);
                    }
                    let dtype = input_float_dtype(Some(&tid), &ai_graph.tensor_info);
                    builder = builder.set_node_dtype(idx, dtype);
                    tid_to_idx.insert(tid, idx);
                }
            }
            DispatchTarget::FloatNeedsShape => {
                // Try each strategy in order until one succeeds.
                let mut lowered = None;
                for strategy in &strategies {
                    match strategy.lower(
                        &node.op,
                        &node.inputs,
                        &ai_graph.tensor_info,
                        &dim_var_names,
                    )? {
                        Some(result) => {
                            lowered = Some(result);
                            break;
                        }
                        None => continue,
                    }
                }

                let result = lowered.with_context(|| {
                    let input_shapes: Vec<_> = node.inputs.iter().map(|tid| {
                        ai_graph.tensor_info.get(tid).map(|info| format!("T{}:{:?}", tid, info.shape.as_slice()))
                            .unwrap_or_else(|| format!("T{}:<missing>", tid))
                    }).collect();
                    format!(
                        "no strategy could lower op {:?} with inputs [{}] (all strategies returned None)",
                        node.op,
                        input_shapes.join(", ")
                    )
                })?;

                builder = builder.node_with_inputs(result.graph_op, &input_idxs);
                let idx = builder.len() - 1;

                // Record recipe with the actual node index.
                if let Some(mut recipe) = result.recipe {
                    recipe.node_index = idx as u32;
                    node_recipes.push(recipe);
                }

                if let Some(&tid) = node.outputs.first() {
                    if let Some(shape) = output_shape(Some(&tid), &ai_graph.tensor_info) {
                        builder = builder.set_node_shape(idx, shape);
                    }
                    let dtype = input_float_dtype(Some(&tid), &ai_graph.tensor_info);
                    builder = builder.set_node_dtype(idx, dtype);
                    tid_to_idx.insert(tid, idx);
                }
            }
            DispatchTarget::Identity => {
                if let (Some(&in_tid), Some(&out_tid)) = (node.inputs.first(), node.outputs.first())
                {
                    if let Some(&idx) = tid_to_idx.get(&in_tid) {
                        tid_to_idx.insert(out_tid, idx);
                        let dtype = input_float_dtype(Some(&out_tid), &ai_graph.tensor_info);
                        builder = builder.set_node_dtype(idx, dtype);
                    }
                }
            }
            DispatchTarget::Unsupported { reason } => {
                anyhow::bail!("cannot lower op {:?}: {reason}", node.op);
            }
        }
    }

    // Add Output nodes and register named graph outputs.
    for (i, &tid) in ai_graph.outputs.iter().enumerate() {
        let src_idx = tid_to_idx
            .get(&tid)
            .copied()
            .with_context(|| format!("missing builder index for output tensor {tid}"))?;
        builder = builder.node_with_inputs(GraphOp::Output, &[src_idx]);
        let out_node_idx = builder.len() - 1;
        let name = ai_graph.output_name(i);
        builder = builder.output(name, out_node_idx);
    }

    let graph = builder.build();

    let mut context = ContextBundle::new();
    if !node_recipes.is_empty() {
        context.insert(&ShapeRecipeSection {
            dim_vars: recipe_dim_vars,
            node_recipes,
        });
    }

    Ok(LoweringOutput {
        graph,
        layer_name: phase.layer_name().to_string(),
        context,
    })
}

// ── Input reordering ─────────────────────────────────────────────────────────

/// ONNX Gather/GatherElements: `(data, indices)` → hologram executor: `(indices, data)`.
fn swap_gather_inputs(op: &AiOp, mut idxs: Vec<usize>) -> Vec<usize> {
    if matches!(op, AiOp::Gather { .. } | AiOp::GatherElements { .. }) && idxs.len() >= 2 {
        idxs.swap(0, 1);
    }
    idxs
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Extract the concrete value from a `Dim`, returning `None` for symbolic/dynamic dims.
fn concrete_dim(dim: &Dim) -> Option<u64> {
    match dim {
        Dim::Concrete(n) => Some(*n),
        _ => None,
    }
}

/// Extract the concrete N-D shape from a parameter's TensorInfo.
///
/// Returns `None` if any dimension is symbolic (not yet concretized).
fn param_shape(
    param: &crate::ir::AiParam,
    tid: TensorId,
    tensor_info: &HashMap<TensorId, TensorInfo>,
) -> Option<Vec<usize>> {
    let info = match param {
        crate::ir::AiParam::Inline { info, .. } => info,
        crate::ir::AiParam::Mmap { info, .. } => info,
    };
    // Try to extract concrete dims from the param's TensorInfo shape.
    let shape: Option<Vec<usize>> = info
        .shape
        .iter()
        .map(|dim| concrete_dim(dim).map(|v| v as usize))
        .collect();
    if shape.is_some() {
        return shape;
    }
    // Fallback: try tensor_info map (may have been concretized during opt).
    tensor_info.get(&tid).and_then(|ti| {
        ti.shape
            .iter()
            .map(|dim| concrete_dim(dim).map(|v| v as usize))
            .collect()
    })
}

/// Extract the N-D output shape for a tensor, using 0 for symbolic dims.
///
/// Concrete dims are preserved; symbolic dims (batch, seq_len) become 0,
/// which the executor resolves at runtime from actual buffer sizes.
fn output_shape(
    tid: Option<&TensorId>,
    tensor_info: &HashMap<TensorId, TensorInfo>,
) -> Option<Vec<usize>> {
    let info = tid.and_then(|t| tensor_info.get(t))?;
    if info.shape.is_empty() {
        return None;
    }
    Some(
        info.shape
            .iter()
            .map(|dim| match dim {
                Dim::Concrete(n) => *n as usize,
                _ => 0, // symbolic → 0 sentinel
            })
            .collect(),
    )
}

/// Infer the output shape for Reshape ops from the input shape and shape tensor.
///
/// For Reshape nodes, the second input is a shape tensor. If it's a constant
/// param, we can read the i64 values directly and compute the output shape.
/// If the shape tensor is computed at runtime (by a shape subgraph), we try
/// to infer from the first input's shape and any available tensor_info.
fn infer_reshape_shape(
    op: &AiOp,
    inputs: &[TensorId],
    output_tid: Option<TensorId>,
    ai_graph: &AiGraph,
) -> Option<Vec<usize>> {
    // Only for Reshape and Flatten (which dispatch to Reshape).
    if !matches!(op, AiOp::Reshape { .. } | AiOp::Flatten { .. }) {
        return None;
    }

    // Check the Reshape OUTPUT tensor's known_i64_values first.
    // data_prop resolves -1 per-consumer and stores the result there, which
    // is more accurate than the shared shape tensor's values.
    if let Some(out_tid) = output_tid {
        if let Some(info) = ai_graph.tensor_info.get(&out_tid) {
            if let Some(known) = &info.known_i64_values {
                let data_tid = inputs[0];
                let data_elems: Option<usize> =
                    ai_graph.tensor_info.get(&data_tid).and_then(|di| {
                        let mut product = 1usize;
                        for dim in di.shape.iter() {
                            match dim {
                                Dim::Concrete(n) => {
                                    product = product.saturating_mul(*n as usize);
                                }
                                _ => return None,
                            }
                        }
                        Some(product)
                    });

                let shape: Vec<usize> = known
                    .iter()
                    .map(|v| match v {
                        Some(-1) => 0,
                        Some(0) => 0,
                        Some(n) if *n > 0 => *n as usize,
                        _ => 0,
                    })
                    .collect();

                if let Some(total) = data_elems {
                    let zero_count = shape.iter().filter(|&&d| d == 0).count();
                    if zero_count == 1 {
                        let known_product: usize =
                            shape.iter().filter(|&&d| d > 0).product::<usize>().max(1);
                        let unknown = total / known_product;
                        return Some(
                            shape
                                .iter()
                                .map(|&d| if d == 0 { unknown } else { d })
                                .collect(),
                        );
                    }
                }
                if !shape.is_empty() {
                    return Some(shape);
                }
            }
        }
    }

    // Reshape has 2 inputs: (data, shape_tensor).
    // Try reading shape values from the shape tensor if it's a constant param.
    if inputs.len() >= 2 {
        let shape_tid = inputs[1];
        if let Some(param) = ai_graph.params.get(&shape_tid) {
            tracing::trace!(shape_tid, "infer_reshape: found shape in params");
            // Read i64 values from the constant shape tensor.
            let data = match param {
                crate::ir::AiParam::Inline { data, .. } => data.as_slice(),
                _ => return None, // Mmap shape tensors are unusual
            };
            if data.len() % 8 == 0 && !data.is_empty() {
                let i64_vals: Vec<i64> = data
                    .chunks_exact(8)
                    .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
                    .collect();

                // Get the data tensor's total element count for resolving -1 dims.
                let data_tid = inputs[0];
                let data_info = ai_graph.tensor_info.get(&data_tid);
                let data_elems: Option<usize> = data_info.and_then(|info| {
                    let mut product = 1usize;
                    for dim in info.shape.iter() {
                        match dim {
                            Dim::Concrete(n) => product = product.saturating_mul(*n as usize),
                            _ => return None, // Can't compute total if any dim is symbolic
                        }
                    }
                    Some(product)
                });

                let shape: Vec<usize> = i64_vals
                    .iter()
                    .map(|&v| {
                        if v == -1 || v == 0 {
                            0 // 0 sentinel — resolved at runtime (-1 = infer, 0 = keep)
                        } else if v < 0 {
                            1 // invalid negative
                        } else {
                            v as usize
                        }
                    })
                    .collect();

                // Try to resolve a single -1 dim if we know total elements.
                if let Some(total) = data_elems {
                    let zero_count = shape.iter().filter(|&&d| d == 0).count();
                    if zero_count == 1 {
                        let known_product: usize =
                            shape.iter().filter(|&&d| d > 0).product::<usize>().max(1);
                        let unknown = total / known_product;
                        return Some(
                            shape
                                .iter()
                                .map(|&d| if d == 0 { unknown } else { d })
                                .collect(),
                        );
                    }
                }
                return Some(shape);
            }
        }
    }

    // Try data-propagated known values from the shape tensor.
    if inputs.len() >= 2 {
        let shape_tid = inputs[1];
        if let Some(info) = ai_graph.tensor_info.get(&shape_tid) {
            tracing::trace!(shape_tid, known = ?info.known_i64_values, "infer_reshape: checking known_i64_values");
            if let Some(known) = &info.known_i64_values {
                let data_tid = inputs[0];
                let data_elems: Option<usize> =
                    ai_graph.tensor_info.get(&data_tid).and_then(|di| {
                        let mut product = 1usize;
                        for dim in di.shape.iter() {
                            match dim {
                                Dim::Concrete(n) => product = product.saturating_mul(*n as usize),
                                _ => return None,
                            }
                        }
                        Some(product)
                    });

                let shape: Vec<usize> = known
                    .iter()
                    .map(|v| match v {
                        Some(-1) => 0, // -1 sentinel → 0 (resolve at runtime)
                        Some(0) => 0,  // 0 "keep dim" → 0 sentinel
                        Some(n) if *n > 0 => *n as usize,
                        _ => 0, // None (dynamic) → 0 sentinel
                    })
                    .collect();

                // Try to resolve a single unknown dim from total elements.
                if let Some(total) = data_elems {
                    let zero_count = shape.iter().filter(|&&d| d == 0).count();
                    if zero_count == 1 {
                        let known_product: usize =
                            shape.iter().filter(|&&d| d > 0).product::<usize>().max(1);
                        let unknown = total / known_product;
                        return Some(
                            shape
                                .iter()
                                .map(|&d| if d == 0 { unknown } else { d })
                                .collect(),
                        );
                    }
                }
                return Some(shape);
            }
        }
    }

    None
}

/// Read parameter bytes into an owned `Vec<u8>`.
fn param_bytes_owned(param: &crate::ir::AiParam) -> anyhow::Result<Vec<u8>> {
    use crate::ir::AiParam;
    match param {
        AiParam::Inline { data, .. } => Ok(data.clone()),
        AiParam::Mmap {
            path, offset, len, ..
        } => {
            use std::io::{Read, Seek, SeekFrom};
            let mut f = std::fs::File::open(path)
                .with_context(|| format!("opening mmap param at {path:?}"))?;
            f.seek(SeekFrom::Start(*offset))?;
            let mut buf = vec![0u8; *len as usize];
            f.read_exact(&mut buf)?;
            Ok(buf)
        }
    }
}
