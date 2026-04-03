//! Builds a `hologram::Graph` from a dispatched `AiGraph`.
//!
//! Uses `hologram::GraphBuilder` (fluent, index-based): each node-adding method
//! increments the builder's index counter; `tid_to_idx` maps `TensorId` → builder index.

use super::dispatch::{dispatch, DispatchTarget};
use super::shape_spec_bridge::ShapeProjection;
use super::strategy::{
    ai_dtype_to_float_dtype, input_float_dtype, ConcreteStrategy, DeferredStrategy,
    LoweringStrategy,
};
use super::LowerPhase;
use crate::exec_context::{
    ContextBundle, NodeShapeRecipe, ShapeContextGraph, ShapeProjectionEntry, ShapeRecipeSection,
    ShapeSeed,
};
use crate::ir::{AiGraph, AiNode, AiOp, Dim, DimVarId, TensorId, TensorInfo};
use crate::mem::KvCacheLayout;
use anyhow::Context;
use hologram::{ConstantData, FloatOp, GraphBuilder, GraphOp, SubgraphDef};
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

/// Quantized weight handling strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantStrategy {
    /// No compile-time quantization — emit f32 MatMul as-is.
    None,
    /// Auto-detect from backend capabilities.
    Auto,
    /// Always dequantize eagerly at plan start.
    EagerDequant,
    /// Use fused quantized kernels where available.
    FusedKernels,
    /// Quantize f32 weights to Q4_0 at compile time → LUT-GEMM.
    Q4_0,
    /// Quantize f32 weights to Q8_0 at compile time → LUT-GEMM.
    Q8_0,
}

/// Output of the lowering pass.
pub struct LoweringOutput {
    pub graph: hologram::Graph,
    /// Layer name for archive metadata (e.g. "lm.prefill", "model.forward").
    pub layer_name: String,
    /// All context sections produced during lowering (shape recipes, etc.).
    pub context: ContextBundle,
    /// Mapping from AiGraph TensorId → builder node index.
    /// Preserved for `compile_with_debug_info()` conformance testing.
    pub tid_to_idx: HashMap<TensorId, usize>,
}

// ── Entry point ───────────────────────────────────────────────────────────────

/// Lower an optimised `AiGraph` to `hologram::Graph`.
///
/// All ops emit native `GraphOp` variants — no `CustomOpRegistry` needed.
/// Does NOT call `hologram::compile()` — that is the caller's responsibility.
pub fn lower(
    ai_graph: &AiGraph,
    _kv_layout: &KvCacheLayout,
    opts: &LoweringOptions,
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

    // Accumulate the compile-time shape context graph.
    let mut shape_context = ShapeContextGraph::new();

    // Register named graph inputs and insert Input nodes.
    for (i, &tid) in ai_graph.inputs.iter().enumerate() {
        let name = ai_graph.input_name(i);
        builder = builder.input(name);
        builder = builder.node_from_graph_input(GraphOp::Input, i as u32);
        let idx = builder.len() - 1;
        if let Some(shape) = output_shape(Some(&tid), &ai_graph.tensor_info) {
            builder = builder.set_node_shape(idx, shape.clone());
            // Emit a seed only if the shape is fully concrete (no 0-sentinels).
            if !shape.contains(&0) {
                shape_context.seeds.push(ShapeSeed {
                    node_id: idx as u32,
                    shape: shape.iter().map(|&d| d as u32).collect(),
                    known_i64_values: None,
                });
            }
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

        // Emit a seed for constant params whose shape is fully concrete.
        if let Some(shape) = param_shape(param, tid, &ai_graph.tensor_info) {
            if !shape.contains(&0) {
                // Try tensor_info first; fall back to extracting i64 values
                // directly from small INT64 constant bytes (tensor_info values
                // may have been cleared by post-concretization passes).
                let known_i64_values = ai_graph
                    .tensor_info
                    .get(&tid)
                    .and_then(|ti| ti.known_i64_values.clone())
                    .or_else(|| extract_i64_values_from_param(param, &shape));
                shape_context.seeds.push(ShapeSeed {
                    node_id: builder_idx as u32,
                    shape: shape.iter().map(|&d| d as u32).collect(),
                    known_i64_values,
                });
            }
        }
    }

    // Emit each node in topological order.
    let topo = ai_graph.topo_order();
    let node_map: HashMap<u32, &_> = ai_graph.nodes.iter().map(|n| (n.id, n)).collect();

    for &nid in topo.iter() {
        let node = node_map[&nid];

        let input_idxs: Vec<usize> = node
            .inputs
            .iter()
            .map(|tid| {
                tid_to_idx
                    .get(tid)
                    .copied()
                    .with_context(|| {
                        let tensor_name = ai_graph.tensor_names.get(tid)
                            .map(|s| s.as_str())
                            .unwrap_or("?");
                        format!(
                            "missing builder index for tensor {} '{}' (referenced by node id={})",
                            tid, tensor_name, node.id,
                        )
                    })
            })
            .collect::<anyhow::Result<_>>()?;

        // ONNX Gather has (data, indices) but hologram executor expects
        // (indices, data). Swap inputs for Gather/GatherElements.
        let input_idxs = swap_gather_inputs(&node.op, input_idxs);

        match dispatch(&node.op) {
            DispatchTarget::GraphOp(graph_op) => {
                // Capture the FloatOp for shape projection before move.
                let float_op_for_spec: Option<FloatOp> = match &graph_op {
                    GraphOp::Float(fop) => Some(*fop),
                    _ => None,
                };
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

                    // Emit shape projection entry via ShapeProjection trait.
                    if let Some((spec, shape_value_input)) =
                        float_op_for_spec.and_then(|fop| fop.shape_spec())
                    {
                        let input_node_ids: Vec<u32> =
                            input_idxs.iter().map(|&i| i as u32).collect();
                        shape_context.projections.push(ShapeProjectionEntry {
                            node_id: idx as u32,
                            input_node_ids,
                            spec,
                            shape_value_input,
                        });
                    }
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

                // ── LUT-GEMM interception ────────────────────────────────
                // If the strategy produced a Gemm with quant_b=1 (Q4_0),
                // convert to MatMulLut4 using the hologram LUT-GEMM kernel.
                if let GraphOp::Float(FloatOp::Gemm { quant_b: 1, .. }) = &result.graph_op {
                    if let Some(lut_result) = try_convert_q4_0_to_lut4(
                        node,
                        ai_graph,
                        &input_idxs,
                    )? {
                        builder = builder.matmul_lut_4bit(
                            ConstantData::Bytes(lut_result.serialized_weights),
                            &[input_idxs[0]], // activation input only
                        );
                        let idx = builder.len() - 1;
                        if let Some(&tid) = node.outputs.first() {
                            // Output shape: [m, n] (activation rows × weight cols).
                            let out_shape = output_shape(Some(&tid), &ai_graph.tensor_info);
                            if let Some(ref s) = out_shape {
                                builder = builder.set_node_shape(idx, s.clone());
                            }
                            let dtype = input_float_dtype(Some(&tid), &ai_graph.tensor_info);
                            builder = builder.set_node_dtype(idx, dtype);
                            tid_to_idx.insert(tid, idx);
                        }
                        tracing::info!(
                            node_id = node.id,
                            rows = lut_result.rows,
                            cols = lut_result.cols,
                            "LUT-GEMM: converted Q4_0 Gemm → MatMulLut4"
                        );
                        continue; // Skip normal FloatNeedsShape emission.
                    }
                }

                // ── LUT-GEMM interception for plain f32 MatMul ──
                // Fused variants (FusedMatMulActivation) keep their f32 fused kernel
                // which already eliminates the intermediate buffer. Only plain MatMul
                // benefits from LUT-GEMM quantization.
                // Skip MatMuls that produce Q/K/V/O projections for attention.
                // These have output N matching attention head dimensions AND input K
                // matching hidden_size. FFN down_proj has K=ffn_intermediate (≠hidden_size)
                // and is NOT skipped despite having N=hidden_size.
                let attn_dims: Vec<usize> = ai_graph.nodes.iter()
                    .filter_map(|n| match &n.op {
                        AiOp::GroupedQueryAttention { num_heads, num_kv_heads, head_dim, .. } => {
                            let q_dim = *num_heads as usize * *head_dim as usize;
                            let kv_dim = *num_kv_heads as usize * *head_dim as usize;
                            // Include individual dims AND fused QKV dim (from
                            // SharedInputProjectionFusion: N_q + N_k + N_v).
                            Some(vec![q_dim, kv_dim, q_dim + 2 * kv_dim])
                        }
                        _ => None,
                    })
                    .flatten()
                    .collect();
                let feeds_attention = if let GraphOp::Float(FloatOp::MatMul { n, .. }) = &result.graph_op {
                    attn_dims.contains(&(*n as usize))
                } else {
                    false
                };
                if matches!(opts.quant_strategy, QuantStrategy::Q4_0)
                    && matches!(result.graph_op, GraphOp::Float(FloatOp::MatMul { .. }))
                    && !feeds_attention
                {
                    if let Some(lut_result) = try_convert_f32_to_lut4(
                        node,
                        ai_graph,
                        &input_idxs,
                    )? {
                        builder = builder.matmul_lut_4bit(
                            ConstantData::Bytes(lut_result.serialized_weights),
                            &[input_idxs[0]],
                        );
                        let idx = builder.len() - 1;
                        if let Some(&tid) = node.outputs.first() {
                            let out_shape = output_shape(Some(&tid), &ai_graph.tensor_info);
                            if let Some(ref s) = out_shape {
                                builder = builder.set_node_shape(idx, s.clone());
                            }
                            let dtype = input_float_dtype(Some(&tid), &ai_graph.tensor_info);
                            builder = builder.set_node_dtype(idx, dtype);
                            tid_to_idx.insert(tid, idx);
                        }
                        tracing::info!(
                            node_id = node.id,
                            rows = lut_result.rows,
                            cols = lut_result.cols,
                            "LUT-GEMM: quantized f32 MatMul → MatMulLut4"
                        );
                        continue;
                    }
                }

                // ── LUT-GEMM Q8_0 interception for plain f32 MatMul ──
                if matches!(opts.quant_strategy, QuantStrategy::Q8_0)
                    && matches!(result.graph_op, GraphOp::Float(FloatOp::MatMul { .. }))
                    && !feeds_attention
                {
                    if let Some(lut_result) = try_convert_f32_to_lut8(
                        node,
                        ai_graph,
                        &input_idxs,
                    )? {
                        builder = builder.matmul_lut_8bit(
                            ConstantData::Bytes(lut_result.serialized_weights),
                            &[input_idxs[0]],
                        );
                        let idx = builder.len() - 1;
                        if let Some(&tid) = node.outputs.first() {
                            let out_shape = output_shape(Some(&tid), &ai_graph.tensor_info);
                            if let Some(ref s) = out_shape {
                                builder = builder.set_node_shape(idx, s.clone());
                            }
                            let dtype = input_float_dtype(Some(&tid), &ai_graph.tensor_info);
                            builder = builder.set_node_dtype(idx, dtype);
                            tid_to_idx.insert(tid, idx);
                        }
                        tracing::info!(
                            node_id = node.id,
                            rows = lut_result.rows,
                            cols = lut_result.cols,
                            "LUT-GEMM: quantized f32 MatMul → MatMulLut8"
                        );
                        continue;
                    }
                }

                // ── Conv2d LUT-GEMM interception ─────────────────────────────
                // Pre-quantize Conv2d weights at compile time for zero runtime overhead.
                if let GraphOp::Float(FloatOp::Conv2d {
                    kernel_h, kernel_w, stride_h, stride_w,
                    pad_h, pad_w, dilation_h, dilation_w,
                    group, input_h, input_w,
                }) = &result.graph_op
                {
                    if let Some(lut_result) = try_convert_conv2d_to_lut4(
                        node, ai_graph, *kernel_h, *kernel_w, *group,
                    )? {
                        builder = builder.conv2d_lut_4bit(
                            ConstantData::Bytes(lut_result.serialized_weights),
                            &input_idxs,
                            *kernel_h, *kernel_w,
                            *stride_h, *stride_w,
                            *pad_h, *pad_w,
                            *dilation_h, *dilation_w,
                            *group, *input_h, *input_w,
                        );
                        let idx = builder.len() - 1;
                        if let Some(&tid) = node.outputs.first() {
                            let out_shape = output_shape(Some(&tid), &ai_graph.tensor_info);
                            if let Some(ref s) = out_shape {
                                builder = builder.set_node_shape(idx, s.clone());
                            }
                            let dtype = input_float_dtype(Some(&tid), &ai_graph.tensor_info);
                            builder = builder.set_node_dtype(idx, dtype);
                            tid_to_idx.insert(tid, idx);
                        }
                        tracing::info!(
                            node_id = node.id,
                            rows = lut_result.rows,
                            cols = lut_result.cols,
                            "LUT-GEMM: pre-quantized Conv2d → Conv2dLut4"
                        );
                        continue; // Skip normal FloatNeedsShape emission.
                    }
                }

                // Capture FloatOp for shape projection before move.
                // Extract FloatOp for shape projection. For fused ops, use the
                // base op's shape semantics (e.g., MatMul for FusedMatMulActivation).
                let float_op_for_spec: Option<FloatOp> = match &result.graph_op {
                    GraphOp::Float(fop) => Some(*fop),
                    GraphOp::FusedMatMulActivation { m, k, n, .. } => {
                        Some(FloatOp::MatMul { m: *m, k: *k, n: *n })
                    }
                    _ => None,
                };

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

                    // Emit shape projection entry via ShapeProjection trait.
                    if let Some((spec, shape_value_input)) =
                        float_op_for_spec.and_then(|fop| fop.shape_spec())
                    {
                        let input_node_ids: Vec<u32> =
                            input_idxs.iter().map(|&i| i as u32).collect();
                        shape_context.projections.push(ShapeProjectionEntry {
                            node_id: idx as u32,
                            input_node_ids,
                            spec,
                            shape_value_input,
                        });
                    }
                }
            }
            DispatchTarget::Identity => {
                if let (Some(&in_tid), Some(&out_tid)) = (node.inputs.first(), node.outputs.first())
                {
                    if let Some(&in_idx) = tid_to_idx.get(&in_tid) {
                        let in_shape = output_shape(Some(&in_tid), &ai_graph.tensor_info);
                        let out_shape_val = output_shape(Some(&out_tid), &ai_graph.tensor_info);

                        let shapes_differ = match (&in_shape, &out_shape_val) {
                            (Some(a), Some(b)) => a.len() != b.len() || a != b,
                            _ => false,
                        };

                        if shapes_differ {
                            // Shape-changing identity op (Unsqueeze, Squeeze, etc.):
                            // emit a Reshape node so the ShapeContextGraph has full
                            // coverage and the walker can propagate shapes through.
                            builder = builder.node_with_inputs(
                                GraphOp::Float(FloatOp::Reshape),
                                &[in_idx],
                            );
                            let idx = builder.len() - 1;
                            if let Some(ref s) = out_shape_val {
                                builder = builder.set_node_shape(idx, s.clone());
                            }
                            let dtype = input_float_dtype(Some(&out_tid), &ai_graph.tensor_info);
                            builder = builder.set_node_dtype(idx, dtype);
                            tid_to_idx.insert(out_tid, idx);

                            // Use ShapeProjection trait on the AiOp to get the spec.
                            if let Some((spec, shape_value_input)) = node.op.shape_spec() {
                                shape_context.projections.push(ShapeProjectionEntry {
                                    node_id: idx as u32,
                                    input_node_ids: vec![in_idx as u32],
                                    spec,
                                    shape_value_input,
                                });
                            }
                        } else {
                            // Pure identity: alias as before.
                            tid_to_idx.insert(out_tid, in_idx);
                            let dtype = input_float_dtype(Some(&out_tid), &ai_graph.tensor_info);
                            builder = builder.set_node_dtype(in_idx, dtype);
                        }
                    }
                }
            }
            DispatchTarget::MultiOutput => {
                // FusedNormProjection → 1 norm node + N MatMul nodes.
                // Each projection weight is a separate input; the norm output
                // lives in the arena and is shared by all N projections.
                if let AiOp::FusedNormProjection { epsilon, split_sizes, has_residual_add } = &node.op {
                    let eps_bits = hologram::f32_to_bits(*epsilon as f32);

                    // Input layout:
                    //   has_residual_add=false: [x, norm_weight, W_0, W_1, ...]
                    //   has_residual_add=true:  [x, residual, norm_weight, W_0, W_1, ...]
                    let weight_start = if *has_residual_add { 3 } else { 2 };
                    let n_projections = split_sizes.len();

                    // Step 1: Emit the norm node.
                    let norm_size = node.inputs.first()
                        .and_then(|tid| ai_graph.tensor_info.get(tid))
                        .and_then(|info| info.shape.last())
                        .and_then(|d| d.evaluate())
                        .unwrap_or(0) as u32;

                    let norm_op = if *has_residual_add {
                        GraphOp::Float(FloatOp::AddRmsNorm { size: norm_size, epsilon: eps_bits })
                    } else {
                        GraphOp::Float(FloatOp::RmsNorm { size: norm_size, epsilon: eps_bits })
                    };
                    let norm_inputs: Vec<usize> = if *has_residual_add {
                        // [x, residual, norm_weight]
                        vec![input_idxs[0], input_idxs[1], input_idxs[2]]
                    } else {
                        // [x, norm_weight]
                        vec![input_idxs[0], input_idxs[1]]
                    };
                    builder = builder.node_with_inputs(norm_op, &norm_inputs);
                    let norm_idx = builder.len() - 1;

                    // Set norm output shape (same as x).
                    if let Some(x_shape) = output_shape(node.inputs.first(), &ai_graph.tensor_info) {
                        builder = builder.set_node_shape(norm_idx, x_shape);
                    }
                    let norm_dtype = input_float_dtype(node.inputs.first(), &ai_graph.tensor_info);
                    builder = builder.set_node_dtype(norm_idx, norm_dtype);

                    // Step 2: Emit N MatMul nodes, one per projection.
                    for (i, &out_tid) in node.outputs.iter().enumerate() {
                        if i >= n_projections { break; }
                        let weight_input_pos = weight_start + i;
                        if weight_input_pos >= input_idxs.len() { break; }

                        let n_val = split_sizes[i] as u32;
                        let matmul_op = GraphOp::Float(FloatOp::MatMul {
                            m: 0, // runtime-inferred
                            k: norm_size,
                            n: n_val,
                        });
                        builder = builder.node_with_inputs(
                            matmul_op,
                            &[norm_idx, input_idxs[weight_input_pos]],
                        );
                        let proj_idx = builder.len() - 1;

                        let out_shape = output_shape(Some(&out_tid), &ai_graph.tensor_info);
                        if let Some(ref s) = out_shape {
                            builder = builder.set_node_shape(proj_idx, s.clone());
                        }
                        let dtype = input_float_dtype(Some(&out_tid), &ai_graph.tensor_info);
                        builder = builder.set_node_dtype(proj_idx, dtype);
                        tid_to_idx.insert(out_tid, proj_idx);
                    }

                    tracing::info!(
                        node_id = node.id,
                        n_projections,
                        has_residual_add,
                        "lowered FusedNormProjection as 1 norm + {} MatMul nodes",
                        n_projections,
                    );
                }
                // Split → N Slice nodes, one per output tensor.
                else if let AiOp::Split { axis, sizes } = &node.op {
                    tracing::info!(
                        node_id = node.id,
                        axis,
                        sizes = ?sizes,
                        n_outputs = node.outputs.len(),
                        "lowering Split as multi-output Slice"
                    );
                    let in_idx = input_idxs[0];
                    // Normalize axis: get input ndim from tensor_info.
                    let ndim = node.inputs.first()
                        .and_then(|tid| ai_graph.tensor_info.get(tid))
                        .map(|info| info.shape.len())
                        .unwrap_or(4);
                    let norm_axis = if *axis < 0 { (ndim as i64 + *axis) as usize } else { *axis as usize };
                    // Get the axis size from tensor_info for axis_size param.
                    let full_axis_size = node.inputs.first()
                        .and_then(|tid| ai_graph.tensor_info.get(tid))
                        .and_then(|info| info.shape.get(norm_axis))
                        .and_then(|d| d.evaluate())
                        .unwrap_or(0) as u32;

                    // If sizes is empty, Split equally into N outputs.
                    let effective_sizes: Vec<u64> = if sizes.is_empty() {
                        let n = node.outputs.len() as u64;
                        if n > 0 && full_axis_size > 0 {
                            let chunk = full_axis_size as u64 / n;
                            vec![chunk; n as usize]
                        } else {
                            vec![]
                        }
                    } else {
                        sizes.clone()
                    };

                    let mut offset: u32 = 0;
                    for (i, &size) in effective_sizes.iter().enumerate() {
                        let start = offset;
                        let end = offset + size as u32;
                        offset = end;

                        let slice_op = FloatOp::Slice {
                            axis_from_end: (ndim - 1 - norm_axis) as u8,
                            start,
                            end,
                            axis_size: full_axis_size,
                        };
                        builder = builder.node_with_inputs(
                            GraphOp::Float(slice_op),
                            &[in_idx],
                        );
                        let idx = builder.len() - 1;
                        if let Some(&out_tid) = node.outputs.get(i) {
                            let out_shape = output_shape(Some(&out_tid), &ai_graph.tensor_info);
                            if let Some(ref s) = out_shape {
                                builder = builder.set_node_shape(idx, s.clone());
                            }
                            let dtype = input_float_dtype(Some(&out_tid), &ai_graph.tensor_info);
                            builder = builder.set_node_dtype(idx, dtype);
                            tid_to_idx.insert(out_tid, idx);
                        }
                    }
                }
            }
            DispatchTarget::Subgraph => {
                lower_subgraph_op(
                    node,
                    &input_idxs,
                    ai_graph,
                    &mut builder,
                    &mut tid_to_idx,
                    _kv_layout,
                    opts,
                    phase,
                )?;
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
    if !shape_context.is_empty() {
        context.insert(&shape_context);
    }

    Ok(LoweringOutput {
        graph,
        layer_name: phase.layer_name().to_string(),
        context,
        tid_to_idx,
    })
}

// ── Subgraph lowering ────────────────────────────────────────────────────────

/// Shared context for recursive subgraph lowering.
struct SubgraphCtx<'a> {
    ai_graph: &'a AiGraph,
    builder: &'a mut GraphBuilder,
    tid_to_idx: &'a mut HashMap<TensorId, usize>,
    kv_layout: &'a KvCacheLayout,
    opts: &'a LoweringOptions,
    phase: &'a LowerPhase,
}

/// Lower a control flow op (If/Loop/Scan) by recursively lowering its
/// child subgraphs and flattening them into the parent graph.
#[allow(clippy::too_many_arguments)]
fn lower_subgraph_op(
    node: &AiNode,
    input_idxs: &[usize],
    ai_graph: &AiGraph,
    builder: &mut GraphBuilder,
    tid_to_idx: &mut HashMap<TensorId, usize>,
    kv_layout: &KvCacheLayout,
    opts: &LoweringOptions,
    phase: &LowerPhase,
) -> anyhow::Result<()> {
    let mut ctx = SubgraphCtx {
        ai_graph,
        builder,
        tid_to_idx,
        kv_layout,
        opts,
        phase,
    };
    match &node.op {
        AiOp::If {
            then_branch,
            else_branch,
        } => lower_if_op(node, input_idxs, then_branch, else_branch.as_deref(), &mut ctx),
        AiOp::Loop {
            body,
            max_trip_count,
        } => lower_loop_op(node, input_idxs, body, *max_trip_count, &mut ctx),
        AiOp::Scan {
            body,
            num_scan_inputs,
        } => {
            let child = ctx
                .ai_graph
                .subgraphs
                .get(body)
                .with_context(|| format!("Scan body subgraph '{body}' not found"))?;
            let lowered = lower(child, ctx.kv_layout, ctx.opts, ctx.phase)?;
            let sub_id =
                ctx.builder
                    .subgraph_with_id(SubgraphDef::new(body.clone(), lowered.graph));
            *ctx.builder = std::mem::take(ctx.builder).node_with_inputs(
                GraphOp::CallSubgraph(sub_id),
                &input_idxs[..1.min(input_idxs.len())],
            );
            let idx = ctx.builder.len() - 1;
            for &tid in &node.outputs {
                let dtype = input_float_dtype(Some(&tid), &ctx.ai_graph.tensor_info);
                *ctx.builder = std::mem::take(ctx.builder).set_node_dtype(idx, dtype);
                ctx.tid_to_idx.insert(tid, idx);
            }
            tracing::warn!(
                "Scan op lowered to CallSubgraph — requires runtime dispatch (num_scan_inputs={num_scan_inputs})"
            );
            Ok(())
        }
        _ => anyhow::bail!("lower_subgraph_op called with non-subgraph op: {:?}", node.op),
    }
}

/// Lower an If op: flatten both branches, select outputs with Where.
fn lower_if_op(
    node: &AiNode,
    input_idxs: &[usize],
    then_branch: &str,
    else_branch: Option<&str>,
    ctx: &mut SubgraphCtx<'_>,
) -> anyhow::Result<()> {
    let cond_idx = input_idxs
        .first()
        .copied()
        .with_context(|| "If op has no condition input")?;
    let feed_idxs = &input_idxs[1..];

    // Lower and flatten then branch.
    let then_child = ctx
        .ai_graph
        .subgraphs
        .get(then_branch)
        .with_context(|| format!("If then_branch subgraph '{then_branch}' not found"))?;
    let then_lowered = lower(then_child, ctx.kv_layout, ctx.opts, ctx.phase)?;
    let then_sub_id = ctx
        .builder
        .subgraph_with_id(SubgraphDef::new(then_branch.to_string(), then_lowered.graph));
    let bindings: Vec<(u32, usize)> = feed_idxs
        .iter()
        .enumerate()
        .map(|(i, &idx)| (i as u32, idx))
        .collect();
    let then_outputs = ctx
        .builder
        .flatten_registered_subgraph(then_sub_id, &bindings)
        .map_err(|e| anyhow::anyhow!("failed to flatten If then_branch: {e}"))?;

    if let Some(else_name) = else_branch {
        // Lower and flatten else branch.
        let else_child = ctx
            .ai_graph
            .subgraphs
            .get(else_name)
            .with_context(|| format!("If else_branch subgraph '{else_name}' not found"))?;
        let else_lowered = lower(else_child, ctx.kv_layout, ctx.opts, ctx.phase)?;
        let else_sub_id = ctx
            .builder
            .subgraph_with_id(SubgraphDef::new(else_name.to_string(), else_lowered.graph));
        let else_outputs = ctx
            .builder
            .flatten_registered_subgraph(else_sub_id, &bindings)
            .map_err(|e| anyhow::anyhow!("failed to flatten If else_branch: {e}"))?;

        // For each output: Where(condition, then_out, else_out).
        for (i, (&then_out, &else_out)) in then_outputs
            .iter()
            .zip(else_outputs.iter())
            .take(node.outputs.len())
            .enumerate()
        {
            let where_inputs = [cond_idx, then_out, else_out];
            *ctx.builder = std::mem::take(ctx.builder)
                .node_with_inputs(GraphOp::Float(FloatOp::Where), &where_inputs);
            let idx = ctx.builder.len() - 1;
            let tid = node.outputs[i];
            if let Some(shape) = output_shape(Some(&tid), &ctx.ai_graph.tensor_info) {
                *ctx.builder = std::mem::take(ctx.builder).set_node_shape(idx, shape);
            }
            let dtype = input_float_dtype(Some(&tid), &ctx.ai_graph.tensor_info);
            *ctx.builder = std::mem::take(ctx.builder).set_node_dtype(idx, dtype);
            ctx.tid_to_idx.insert(tid, idx);
        }
    } else {
        // No else branch: outputs are just the then branch outputs.
        for (tid, &out_idx) in node.outputs.iter().zip(then_outputs.iter()) {
            ctx.tid_to_idx.insert(*tid, out_idx);
        }
    }

    Ok(())
}

/// Lower a Loop op. If the trip count is known at compile time, unroll.
/// Otherwise emit CallSubgraph for runtime dispatch.
fn lower_loop_op(
    node: &AiNode,
    input_idxs: &[usize],
    body: &str,
    max_trip_count: Option<i64>,
    ctx: &mut SubgraphCtx<'_>,
) -> anyhow::Result<()> {
    let child = ctx
        .ai_graph
        .subgraphs
        .get(body)
        .with_context(|| format!("Loop body subgraph '{body}' not found"))?;

    // Try to resolve trip count from the AiOp field or from constant input.
    let trip_count = max_trip_count.or_else(|| {
        node.inputs.first().and_then(|&tid| {
            ctx.ai_graph
                .tensor_info
                .get(&tid)
                .and_then(|ti| ti.known_i64_values.as_ref())
                .and_then(|vals| vals.first().copied().flatten())
        })
    });

    if let Some(n) = trip_count {
        if n <= 0 {
            // Zero iterations: outputs are the initial carry state (inputs[2..]).
            for (i, &tid) in node.outputs.iter().enumerate() {
                if let Some(&src_idx) = input_idxs.get(i + 2) {
                    ctx.tid_to_idx.insert(tid, src_idx);
                }
            }
            return Ok(());
        }

        let n = n.min(1024) as usize;
        if n > 64 {
            tracing::warn!("Loop unrolling {n} iterations — consider runtime dispatch");
        }

        let lowered_body = lower(child, ctx.kv_layout, ctx.opts, ctx.phase)?;
        let sub_id = ctx
            .builder
            .subgraph_with_id(SubgraphDef::new(body.to_string(), lowered_body.graph));

        // Initial carry state: input_idxs[2..] (skip trip_count and condition).
        let num_carry = child.inputs.len().saturating_sub(2);
        let mut carry_idxs: Vec<usize> = input_idxs
            .get(2..)
            .unwrap_or(&[])
            .iter()
            .take(num_carry)
            .copied()
            .collect();

        for _iter in 0..n {
            // Bind carry state to body inputs 2..
            let bindings: Vec<(u32, usize)> = carry_idxs
                .iter()
                .enumerate()
                .map(|(i, &idx)| ((i + 2) as u32, idx))
                .collect();

            let outputs = ctx
                .builder
                .flatten_registered_subgraph(sub_id, &bindings)
                .map_err(|e| anyhow::anyhow!("failed to flatten Loop body iteration: {e}"))?;

            // Body outputs: [condition, ...updated_carry, ...scan_outputs].
            carry_idxs = outputs.get(1..1 + num_carry).unwrap_or(&[]).to_vec();
        }

        // Map node outputs to final carry state.
        for (i, &tid) in node.outputs.iter().enumerate() {
            if let Some(&idx) = carry_idxs.get(i) {
                if let Some(shape) = output_shape(Some(&tid), &ctx.ai_graph.tensor_info) {
                    *ctx.builder = std::mem::take(ctx.builder).set_node_shape(idx, shape);
                }
                let dtype = input_float_dtype(Some(&tid), &ctx.ai_graph.tensor_info);
                *ctx.builder = std::mem::take(ctx.builder).set_node_dtype(idx, dtype);
                ctx.tid_to_idx.insert(tid, idx);
            }
        }
    } else {
        // Dynamic trip count: emit CallSubgraph for runtime dispatch.
        let lowered_body = lower(child, ctx.kv_layout, ctx.opts, ctx.phase)?;
        let sub_id = ctx
            .builder
            .subgraph_with_id(SubgraphDef::new(body.to_string(), lowered_body.graph));
        *ctx.builder = std::mem::take(ctx.builder).node_with_inputs(
            GraphOp::CallSubgraph(sub_id),
            &input_idxs[..1.min(input_idxs.len())],
        );
        let idx = ctx.builder.len() - 1;
        for &tid in &node.outputs {
            let dtype = input_float_dtype(Some(&tid), &ctx.ai_graph.tensor_info);
            *ctx.builder = std::mem::take(ctx.builder).set_node_dtype(idx, dtype);
            ctx.tid_to_idx.insert(tid, idx);
        }
        tracing::warn!("Loop with dynamic trip count lowered to CallSubgraph — requires runtime dispatch");
    }

    Ok(())
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

/// Extract i64 values from a small INT64 constant parameter.
///
/// Used to seed `ShapeSeed::known_i64_values` when `tensor_info.known_i64_values`
/// has been cleared by the post-concretization passes. Only extracts values for
/// small 1-D INT64 tensors (≤16 elements) — these are shape-computation constants
/// (Reshape targets, Unsqueeze axes, etc.).
fn extract_i64_values_from_param(
    param: &crate::ir::AiParam,
    shape: &[usize],
) -> Option<Vec<Option<i64>>> {
    use crate::ir::AiParam;

    // Only extract for small 1-D tensors (shape computation constants).
    let n_elems: usize = shape.iter().product();
    if n_elems == 0 || n_elems > 16 {
        return None;
    }

    let info = match param {
        AiParam::Inline { info, .. } => info,
        AiParam::Mmap { info, .. } => info,
    };

    // Only INT64 or INT32 dtype.
    let elem_size = match info.logical_dtype {
        crate::ir::DType::INT64 => 8usize,
        crate::ir::DType::INT32 => 4usize,
        _ => return None,
    };

    let data = match param {
        AiParam::Inline { data, .. } => data.as_slice(),
        AiParam::Mmap { .. } => return None, // Don't read from disk for seeds.
    };

    if data.len() < n_elems * elem_size {
        return None;
    }

    let values: Vec<Option<i64>> = if elem_size == 8 {
        data.chunks_exact(8)
            .take(n_elems)
            .map(|c| Some(i64::from_le_bytes(c.try_into().expect("8 bytes"))))
            .collect()
    } else {
        data.chunks_exact(4)
            .take(n_elems)
            .map(|c| Some(i32::from_le_bytes(c.try_into().expect("4 bytes")) as i64))
            .collect()
    };

    Some(values)
}

// ── LUT-GEMM Q4_0 conversion ─────────────────────────────────────────────────

struct Lut4ConversionResult {
    serialized_weights: Vec<u8>,
    rows: u32,
    cols: u32,
}

/// Try to convert a Q4_0 Gemm node to LUT-GEMM format.
///
/// Reads the Q4_0 weight bytes, dequantizes to f32, runs k-means
/// quantization (16 centroids), and serializes as `QuantizedWeights4`.
///
/// Returns `None` if the weight param can't be found or isn't Q4_0.
fn try_convert_q4_0_to_lut4(
    node: &AiNode,
    ai_graph: &AiGraph,
    input_idxs: &[usize],
) -> anyhow::Result<Option<Lut4ConversionResult>> {
    use hologram::hologram_exec::lut_gemm::quantize::quantize_4bit;
    use hologram_ai_quant::q4_0::dequant_q4_0;

    // Weight is input[1] of the Gemm node.
    let weight_tid = match node.inputs.get(1) {
        Some(&tid) => tid,
        None => return Ok(None),
    };

    // Check that it's actually Q4_0.
    let info = match ai_graph.tensor_info.get(&weight_tid) {
        Some(info) if info.quant.scheme == hologram_ai_quant::QuantScheme::Q4_0 => info,
        _ => return Ok(None),
    };

    // Get the weight param bytes.
    let param = match ai_graph.params.get(&weight_tid) {
        Some(p) => p,
        None => return Ok(None),
    };
    let raw_bytes = param_bytes_owned(param)?;

    // Extract weight shape from tensor_info.
    // For trans_b=true Gemm, weight shape is [n, k] (transposed).
    let trans_b = matches!(node.op, AiOp::Gemm { trans_b: true, .. });
    let (rows, cols) = {
        let dims: Vec<usize> = info.shape.iter().filter_map(|d| match d {
            Dim::Concrete(n) => Some(*n as usize),
            _ => None,
        }).collect();
        if dims.len() >= 2 {
            if trans_b {
                // Weight stored as [n, k]; we need [k, n] for LUT-GEMM.
                (dims[1], dims[0])
            } else {
                (dims[0], dims[1])
            }
        } else {
            tracing::warn!("Q4_0 weight shape has <2 concrete dims, skipping LUT conversion");
            return Ok(None);
        }
    };

    // Dequantize Q4_0 → f32.
    let f32_weights = dequant_q4_0(&raw_bytes);
    let expected = rows * cols;
    if f32_weights.len() != expected {
        tracing::warn!(
            got = f32_weights.len(),
            expected,
            "Q4_0 dequant size mismatch, skipping LUT conversion"
        );
        return Ok(None);
    }

    // Transpose if needed: [n, k] → [k, n] (row-major).
    let f32_for_kmeans = if trans_b {
        let n = cols; // after our swap: cols = original dim[0]
        let k = rows; // rows = original dim[1]
        // Original layout: [cols_orig, rows_orig] = [n, k] (since we swapped above)
        // Wait — let's be precise. Original dims[0]=n_orig, dims[1]=k_orig.
        // trans_b means weight is [n, k]. We set rows=k, cols=n.
        // f32_weights is in row-major [n, k] order (n_orig rows of k_orig cols).
        // quantize_4bit expects [rows, cols] = [k, n], so transpose.
        let mut transposed = vec![0.0f32; k * n];
        for i in 0..n {
            for j in 0..k {
                transposed[j * n + i] = f32_weights[i * k + j];
            }
        }
        transposed
    } else {
        f32_weights
    };

    // K-means quantization → QuantizedWeights4 (16 centroids).
    let qw4 = quantize_4bit(&f32_for_kmeans, rows as u32, cols as u32);

    // Serialize via rkyv.
    let serialized = rkyv::to_bytes::<rkyv::rancor::Error>(&qw4)
        .map_err(|e| anyhow::anyhow!("rkyv serialize QuantizedWeights4: {e}"))?
        .to_vec();

    let _ = input_idxs; // used by caller for activation input

    Ok(Some(Lut4ConversionResult {
        serialized_weights: serialized,
        rows: rows as u32,
        cols: cols as u32,
    }))
}

/// Try to convert an f32 MatMul weight to LUT-GEMM Q4_0 format.
///
/// Reads the f32 weight bytes, runs k-means quantization (16 centroids),
/// and serializes as `QuantizedWeights4`. Used when `--quantize q4_0` is set.
///
/// Returns `None` if the weight param can't be found or isn't a 2D f32 tensor.
fn try_convert_f32_to_lut4(
    node: &AiNode,
    ai_graph: &AiGraph,
    _input_idxs: &[usize],
) -> anyhow::Result<Option<Lut4ConversionResult>> {
    use hologram::hologram_exec::lut_gemm::quantize::quantize_4bit;

    // Weight is input[1] of the MatMul node (A × W).
    let weight_tid = match node.inputs.get(1) {
        Some(&tid) => tid,
        None => return Ok(None),
    };

    // Must be a parameter (not an intermediate activation).
    let param = match ai_graph.params.get(&weight_tid) {
        Some(p) => p,
        None => return Ok(None),
    };

    // Get weight shape — must be 2D with all concrete dims.
    let info = match ai_graph.tensor_info.get(&weight_tid) {
        Some(info) => info,
        None => return Ok(None),
    };
    let dims: Vec<usize> = info
        .shape
        .iter()
        .filter_map(|d| match d {
            Dim::Concrete(n) => Some(*n as usize),
            _ => None,
        })
        .collect();
    if dims.len() != 2 {
        return Ok(None);
    }
    let (rows, cols) = (dims[0], dims[1]);

    // Skip tiny weights (embedding lookups, biases, etc.)
    // Only quantize matrices with ≥ 256 elements per dimension.
    if rows < 256 || cols < 256 {
        return Ok(None);
    }

    // Read f32 weight data.
    let raw_bytes = param_bytes_owned(param)?;
    let expected_bytes = rows * cols * 4;
    if raw_bytes.len() != expected_bytes {
        tracing::warn!(
            got = raw_bytes.len(),
            expected = expected_bytes,
            "f32 weight size mismatch, skipping LUT quantization"
        );
        return Ok(None);
    }

    // Interpret as f32 slice.
    let f32_weights: Vec<f32> = raw_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))
        .collect();

    // K-means quantization → QuantizedWeights4 (16 centroids).
    let qw4 = quantize_4bit(&f32_weights, rows as u32, cols as u32);

    // Serialize via rkyv.
    let serialized = rkyv::to_bytes::<rkyv::rancor::Error>(&qw4)
        .map_err(|e| anyhow::anyhow!("rkyv serialize QuantizedWeights4: {e}"))?
        .to_vec();

    Ok(Some(Lut4ConversionResult {
        serialized_weights: serialized,
        rows: rows as u32,
        cols: cols as u32,
    }))
}

/// Try to convert an f32 MatMul weight to LUT-GEMM Q8_0 format.
///
/// Similar to `try_convert_f32_to_lut4` but uses 256-level uniform quantization
/// for higher quality at the cost of larger quantized weights.
fn try_convert_f32_to_lut8(
    node: &AiNode,
    ai_graph: &AiGraph,
    _input_idxs: &[usize],
) -> anyhow::Result<Option<Lut4ConversionResult>> {
    use hologram::hologram_exec::lut_gemm::quantize::quantize_8bit;

    let weight_tid = match node.inputs.get(1) {
        Some(&tid) => tid,
        None => return Ok(None),
    };
    let param = match ai_graph.params.get(&weight_tid) {
        Some(p) => p,
        None => return Ok(None),
    };
    let info = match ai_graph.tensor_info.get(&weight_tid) {
        Some(info) => info,
        None => return Ok(None),
    };
    let dims: Vec<usize> = info
        .shape
        .iter()
        .filter_map(|d| match d {
            Dim::Concrete(n) => Some(*n as usize),
            _ => None,
        })
        .collect();
    if dims.len() != 2 {
        return Ok(None);
    }
    let (rows, cols) = (dims[0], dims[1]);

    // Skip tiny weights — only quantize matrices with >= 256 elements per dimension.
    if rows < 256 || cols < 256 {
        return Ok(None);
    }

    let raw_bytes = param_bytes_owned(param)?;
    let expected_bytes = rows * cols * 4;
    if raw_bytes.len() != expected_bytes {
        tracing::warn!(
            got = raw_bytes.len(),
            expected = expected_bytes,
            "f32 weight size mismatch, skipping Q8 LUT quantization"
        );
        return Ok(None);
    }

    let f32_weights: Vec<f32> = raw_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))
        .collect();

    let qw8 = quantize_8bit(&f32_weights, rows as u32, cols as u32);

    let serialized = rkyv::to_bytes::<rkyv::rancor::Error>(&qw8)
        .map_err(|e| anyhow::anyhow!("rkyv serialize QuantizedWeights8: {e}"))?
        .to_vec();

    Ok(Some(Lut4ConversionResult {
        serialized_weights: serialized,
        rows: rows as u32,
        cols: cols as u32,
    }))
}

/// Try to convert a Conv2d weight to LUT-GEMM Q4 format for compile-time quantization.
///
/// Conv2d weights are [OC, IC/group, KH, KW]. Reshape to 2D [OC, kernel_size],
/// transpose to [kernel_size, OC] (LUT-GEMM layout: rows=K, cols=N), then quantize.
///
/// Returns `None` if weight isn't found or is too small for LUT-GEMM benefit.
fn try_convert_conv2d_to_lut4(
    node: &AiNode,
    ai_graph: &AiGraph,
    _kernel_h: u32,
    _kernel_w: u32,
    group: u32,
) -> anyhow::Result<Option<Lut4ConversionResult>> {
    use hologram::hologram_exec::lut_gemm::quantize::quantize_4bit;

    // Weight is input[1] of the Conv2d node.
    let weight_tid = match node.inputs.get(1) {
        Some(&tid) => tid,
        None => return Ok(None),
    };

    // Must be a parameter (not an intermediate activation).
    let param = match ai_graph.params.get(&weight_tid) {
        Some(p) => p,
        None => return Ok(None),
    };

    // Get weight shape: [OC, IC/group, KH, KW].
    let info = match ai_graph.tensor_info.get(&weight_tid) {
        Some(info) => info,
        None => return Ok(None),
    };
    let dims: Vec<usize> = info
        .shape
        .iter()
        .filter_map(|d| match d {
            Dim::Concrete(n) => Some(*n as usize),
            _ => None,
        })
        .collect();
    if dims.len() < 2 {
        return Ok(None);
    }

    let oc = dims[0];
    let group = group.max(1) as usize;
    let oc_per_group = oc / group;
    let kernel_size: usize = dims[1..].iter().product();

    // Skip small convolutions where LUT-GEMM overhead isn't worth it.
    if oc_per_group < 64 || kernel_size < 16 {
        return Ok(None);
    }

    // Read f32 weight data.
    let raw_bytes = param_bytes_owned(param)?;
    let expected_bytes = oc * kernel_size * 4;
    if raw_bytes.len() != expected_bytes {
        tracing::warn!(
            got = raw_bytes.len(),
            expected = expected_bytes,
            "Conv2d weight size mismatch, skipping LUT quantization"
        );
        return Ok(None);
    }

    let f32_weights: Vec<f32> = raw_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))
        .collect();

    // For grouped conv, quantize each group's weights separately.
    // For now, only handle group=1 (all SD Conv2d ops are group=1).
    if group > 1 {
        return Ok(None);
    }

    // Transpose from [OC, kernel_size] → [kernel_size, OC] for LUT-GEMM layout.
    let mut w_t = vec![0.0f32; kernel_size * oc_per_group];
    for oc_idx in 0..oc_per_group {
        for k in 0..kernel_size {
            w_t[k * oc_per_group + oc_idx] = f32_weights[oc_idx * kernel_size + k];
        }
    }

    // K-means quantization → QuantizedWeights4 (16 centroids).
    let qw4 = quantize_4bit(&w_t, kernel_size as u32, oc_per_group as u32);

    // Serialize via rkyv.
    let serialized = rkyv::to_bytes::<rkyv::rancor::Error>(&qw4)
        .map_err(|e| anyhow::anyhow!("rkyv serialize QuantizedWeights4: {e}"))?
        .to_vec();

    Ok(Some(Lut4ConversionResult {
        serialized_weights: serialized,
        rows: kernel_size as u32,
        cols: oc_per_group as u32,
    }))
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
