//! Builds a `hologram::Graph` from a dispatched `AiGraph`.
//!
//! Uses `hologram::GraphBuilder` (fluent, index-based): each node-adding method
//! increments the builder's index counter; `tid_to_idx` maps `TensorId` → builder index.

use std::collections::HashMap;
use anyhow::Context;
use hologram::{ConstantData, CustomOpId, GraphBuilder, GraphOp};
use crate::ir::{AiGraph, AiOp, Dim, TensorId, TensorInfo};
use crate::mem::KvCacheLayout;
use super::dispatch::{dispatch, DispatchTarget};
use super::custom_ops::{
    abs_handler, add_handler, and_handler, attention_handler, cast_handler,
    ceil_handler, clip_handler, concat_handler, cos_handler, dequant_handler,
    div_handler, embed_handler, equal_handler, erf_handler, exp_handler,
    flatten_handler, floor_handler, gather_handler, gather_nd_handler,
    gelu_handler, greater_handler, greater_or_equal_handler, isnan_handler,
    layer_norm_handler, less_handler, less_or_equal_handler, log_handler,
    log_softmax_handler, matmul_handler, max_handler, min_handler,
    mod_handler, mul_handler, neg_handler, not_handler, or_handler,
    pow_handler, range_handler, reciprocal_handler, reduce_max_handler,
    reduce_mean_handler, reduce_min_handler, reduce_sum_handler,
    relu_handler, reshape_handler, rms_norm_handler, rope_handler,
    round_handler, shape_handler, sigmoid_handler, sign_handler,
    silu_handler, sin_handler, softmax_handler, sqrt_handler, sub_handler,
    swiglu_handler, tanh_handler, where_handler, xor_handler,
};

// ── Public types ──────────────────────────────────────────────────────────────

/// Options controlling lowering behaviour.
pub struct LoweringOptions {
    pub quant_strategy: QuantStrategy,
}

impl Default for LoweringOptions {
    fn default() -> Self { Self { quant_strategy: QuantStrategy::Auto } }
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
    pub registry: hologram::CustomOpRegistry,
}

// ── Entry point ───────────────────────────────────────────────────────────────

/// Lower an optimised `AiGraph` to `hologram::Graph + CustomOpRegistry`.
///
/// Does NOT call `hologram::compile()` — that is the caller's responsibility.
pub fn lower(
    ai_graph: &AiGraph,
    _kv_layout: &KvCacheLayout,
    _opts: &LoweringOptions,
) -> anyhow::Result<LoweringOutput> {
    let mut registry = hologram::CustomOpRegistry::new();
    let mut builder  = GraphBuilder::new();

    // Map AiGraph TensorId → builder node index.
    let mut tid_to_idx: HashMap<TensorId, usize> = HashMap::new();

    // Register named graph inputs and insert Input nodes.
    for (i, &tid) in ai_graph.inputs.iter().enumerate() {
        builder = builder.input(format!("input_{i}"));
        builder = builder.node_from_graph_input(GraphOp::Input, i as u32);
        tid_to_idx.insert(tid, builder.len() - 1);
    }

    // Insert constant param nodes (weights, biases).
    // Sort by TensorId for deterministic iteration — Mmap params use cumulative
    // byte offset as source_id so the executor can resolve them from the weights
    // section of the .holo archive.
    let mut sorted_params: Vec<_> = ai_graph.params.iter().collect();
    sorted_params.sort_by_key(|(&tid, _)| tid);

    let mut mmap_offset: u64 = 0;
    for (&tid, param) in sorted_params {
        let constant = match param {
            crate::ir::AiParam::Mmap { len, .. } => {
                let d = ConstantData::Deferred { byte_size: *len, source_id: mmap_offset };
                mmap_offset += *len;
                d
            }
            _ => {
                let data = param_bytes_owned(param)?;
                ConstantData::Bytes(data)
            }
        };
        builder = builder.constant(constant);
        tid_to_idx.insert(tid, builder.len() - 1);
    }

    // Emit each node in topological order.
    let topo = ai_graph.topo_order();
    let node_map: HashMap<u32, &_> = ai_graph.nodes.iter().map(|n| (n.id, n)).collect();

    // Counter for allocating unique CustomOpIds for shape-dependent ops.
    // Shape-dependent ops (MatMul, Gather, Attention) bake tensor dimensions into
    // their handlers, so each node instance needs its own handler and ID.
    // Start above the fixed IDs in dispatch.rs (currently up to 46).
    let mut next_unique_id: u32 = 1000;

    for nid in topo {
        let node = node_map[&nid];

        let input_idxs: Vec<usize> = node.inputs.iter()
            .map(|tid| tid_to_idx.get(tid).copied()
                .with_context(|| format!("missing builder index for tensor {tid}")))
            .collect::<anyhow::Result<_>>()?;

        match dispatch(&node.op) {
            DispatchTarget::GraphOp(graph_op) => {
                builder = builder.node_with_inputs(graph_op, &input_idxs);
                if let Some(&tid) = node.outputs.first() {
                    tid_to_idx.insert(tid, builder.len() - 1);
                }
            }
            DispatchTarget::Custom { id, arity } => {
                // Shape-dependent ops need unique IDs per node instance,
                // since different nodes have different baked-in dimensions.
                let effective_id = if is_shape_dependent(&node.op) {
                    let uid = CustomOpId(next_unique_id);
                    next_unique_id += 1;
                    uid
                } else {
                    id
                };
                register_handler(&mut registry, effective_id, arity, &node.op,
                                 &node.inputs, &ai_graph.tensor_info)?;
                builder = builder.custom_op(effective_id, arity, &input_idxs);
                if let Some(&tid) = node.outputs.first() {
                    tid_to_idx.insert(tid, builder.len() - 1);
                }
            }
            DispatchTarget::Identity => {
                // Pass-through: output tensor maps to the same index as the input.
                if let (Some(&in_tid), Some(&out_tid)) =
                    (node.inputs.first(), node.outputs.first())
                {
                    if let Some(&idx) = tid_to_idx.get(&in_tid) {
                        tid_to_idx.insert(out_tid, idx);
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
        let src_idx = tid_to_idx.get(&tid).copied()
            .with_context(|| format!("missing builder index for output tensor {tid}"))?;
        builder = builder.node_with_inputs(GraphOp::Output, &[src_idx]);
        let out_node_idx = builder.len() - 1;
        builder = builder.output(format!("output_{i}"), out_node_idx);
    }

    let graph = builder.build();
    Ok(LoweringOutput { graph, registry })
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Register the appropriate `CustomHandler` in the registry for the given op.
///
/// `inputs` and `tensor_info` are needed by shape-sensitive ops (Gather, MatMul).
fn register_handler(
    registry: &mut hologram::CustomOpRegistry,
    id: CustomOpId,
    arity: u8,
    op: &AiOp,
    inputs: &[TensorId],
    tensor_info: &HashMap<TensorId, TensorInfo>,
) -> anyhow::Result<()> {
    let handler = match op {
        AiOp::RmsNorm { epsilon }              => rms_norm_handler(*epsilon),
        AiOp::LayerNorm { epsilon, .. }        => layer_norm_handler(*epsilon),
        AiOp::Softmax { axis }                 => softmax_handler(*axis),
        AiOp::Embed                            => embed_handler(),
        AiOp::Dequantize                       => dequant_handler(),
        AiOp::FusedSwiGLU                      => swiglu_handler(),
        AiOp::Reshape { .. } | AiOp::Transpose { .. }
        | AiOp::Squeeze { .. } | AiOp::Unsqueeze { .. }
        | AiOp::Expand | AiOp::Slice { .. }
        | AiOp::Split { .. } | AiOp::Tile { .. }     => reshape_handler(),
        AiOp::Cast { .. }                      => cast_handler(),
        AiOp::Concat { .. }                    => concat_handler(),
        AiOp::RotaryEmbedding { base, dim }    => rope_handler(*base, *dim),
        AiOp::MultiHeadAttention { head_dim, scale, causal, .. } => {
            let s = scale.unwrap_or((*head_dim as f32).sqrt().recip());
            attention_handler(*head_dim, s, *causal)
        }
        AiOp::GroupedQueryAttention { head_dim, scale, causal, .. } => {
            let s = scale.unwrap_or((*head_dim as f32).sqrt().recip());
            attention_handler(*head_dim, s, *causal)
        }
        AiOp::FlashAttentionHint => attention_handler(64, 0.125, true),
        AiOp::Gather { .. } | AiOp::GatherElements { .. } => {
            // data = inputs[0]; row_size = last concrete dim of the data tensor.
            let row_size = inputs.first()
                .and_then(|tid| tensor_info.get(tid))
                .and_then(|info| info.shape.last())
                .and_then(concrete_dim)
                .unwrap_or(1) as usize;
            gather_handler(row_size)
        }
        AiOp::MatMul | AiOp::BatchMatMul | AiOp::Gemm { .. } => {
            // A = inputs[0]; inner = last dim of A; B = inputs[1]; n_cols = last dim of B.
            let inner = inputs.first()
                .and_then(|tid| tensor_info.get(tid))
                .and_then(|info| info.shape.last())
                .and_then(concrete_dim)
                .unwrap_or(1) as usize;
            let n_cols = inputs.get(1)
                .and_then(|tid| tensor_info.get(tid))
                .and_then(|info| info.shape.last())
                .and_then(concrete_dim)
                .unwrap_or(1) as usize;
            matmul_handler(inner, n_cols)
        }
        AiOp::Shape                            => shape_handler(),
        AiOp::Where                            => where_handler(),
        AiOp::Range                            => range_handler(),
        AiOp::GatherND { .. }                  => gather_nd_handler(),
        AiOp::IsNaN                            => isnan_handler(),
        AiOp::Flatten { .. }                   => flatten_handler(),
        AiOp::Div                              => div_handler(),
        AiOp::Pow                              => pow_handler(),
        AiOp::Mod                              => mod_handler(),
        AiOp::Min                              => min_handler(),
        AiOp::Max                              => max_handler(),
        AiOp::And                              => and_handler(),
        AiOp::Or                               => or_handler(),
        AiOp::Xor                              => xor_handler(),
        AiOp::Not                              => not_handler(),
        AiOp::Equal                            => equal_handler(),
        AiOp::Less                             => less_handler(),
        AiOp::LessOrEqual                      => less_or_equal_handler(),
        AiOp::Greater                          => greater_handler(),
        AiOp::GreaterOrEqual                   => greater_or_equal_handler(),
        AiOp::Reciprocal                       => reciprocal_handler(),
        AiOp::Sign                             => sign_handler(),
        AiOp::Floor                            => floor_handler(),
        AiOp::Ceil                             => ceil_handler(),
        AiOp::Round                            => round_handler(),
        AiOp::Clip                             => clip_handler(),
        AiOp::Erf                              => erf_handler(),
        AiOp::ReduceSum { .. }                 => reduce_sum_handler(),
        AiOp::ReduceMean { .. }                => reduce_mean_handler(),
        AiOp::ReduceMax { .. }                 => reduce_max_handler(),
        AiOp::ReduceMin { .. }                 => reduce_min_handler(),
        AiOp::LogSoftmax { .. }                => log_softmax_handler(),
        AiOp::Add                              => add_handler(),
        AiOp::Sub                              => sub_handler(),
        AiOp::Mul                              => mul_handler(),
        AiOp::Neg                              => neg_handler(),
        AiOp::Relu                             => relu_handler(),
        AiOp::Gelu | AiOp::GeluApprox         => gelu_handler(),
        AiOp::Silu                             => silu_handler(),
        AiOp::Tanh                             => tanh_handler(),
        AiOp::Sigmoid                          => sigmoid_handler(),
        AiOp::Exp                              => exp_handler(),
        AiOp::Log                              => log_handler(),
        AiOp::Sqrt                             => sqrt_handler(),
        AiOp::Abs                              => abs_handler(),
        AiOp::Cos                              => cos_handler(),
        AiOp::Sin                              => sin_handler(),
        _ => anyhow::bail!("no custom handler registered for op {:?}", op),
    };
    registry.register(id, arity, handler);
    Ok(())
}

/// Whether this op bakes tensor dimensions into its custom handler.
///
/// Shape-dependent ops need unique `CustomOpId`s per node so each instance
/// gets its own handler with the correct dimensions.
fn is_shape_dependent(op: &AiOp) -> bool {
    matches!(op,
        AiOp::MatMul | AiOp::BatchMatMul | AiOp::Gemm { .. }
        | AiOp::Gather { .. } | AiOp::GatherElements { .. }
        | AiOp::MultiHeadAttention { .. }
        | AiOp::GroupedQueryAttention { .. }
        | AiOp::FlashAttentionHint
    )
}

/// Extract the concrete value from a `Dim`, returning `None` for symbolic/dynamic dims.
fn concrete_dim(dim: &Dim) -> Option<u64> {
    match dim { Dim::Concrete(n) => Some(*n), _ => None }
}

/// Read parameter bytes into an owned `Vec<u8>`.
fn param_bytes_owned(param: &crate::ir::AiParam) -> anyhow::Result<Vec<u8>> {
    use crate::ir::AiParam;
    match param {
        AiParam::Inline { data, .. } => Ok(data.clone()),
        AiParam::Mmap { path, offset, len, .. } => {
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
