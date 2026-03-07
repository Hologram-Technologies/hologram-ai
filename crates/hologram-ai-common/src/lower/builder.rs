//! Builds a `hologram::Graph` from a dispatched `AiGraph`.
//!
//! Uses `hologram::GraphBuilder` (fluent, index-based): each node-adding method
//! increments the builder's index counter; `tid_to_idx` maps `TensorId` → builder index.

use std::collections::HashMap;
use anyhow::Context;
use hologram::{ConstantData, CustomOpId, GraphBuilder, GraphOp};
use crate::ir::{AiGraph, AiOp, TensorId};
use crate::mem::KvCacheLayout;
use super::dispatch::{dispatch, DispatchTarget};
use super::custom_ops::{
    attention_handler, cast_handler, concat_handler, dequant_handler,
    embed_handler, layer_norm_handler, reshape_handler, rms_norm_handler,
    rope_handler, softmax_handler, swiglu_handler,
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
    _opts: LoweringOptions,
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
    for (&tid, param) in &ai_graph.params {
        let data = param_bytes_owned(param)?;
        builder = builder.constant(ConstantData::Bytes(data));
        tid_to_idx.insert(tid, builder.len() - 1);
    }

    // Emit each node in topological order.
    let topo = ai_graph.topo_order();
    let node_map: HashMap<u32, &_> = ai_graph.nodes.iter().map(|n| (n.id, n)).collect();

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
                register_handler(&mut registry, id, arity, &node.op)?;
                builder = builder.custom_op(id, arity, &input_idxs);
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
fn register_handler(
    registry: &mut hologram::CustomOpRegistry,
    id: CustomOpId,
    arity: u8,
    op: &AiOp,
) -> anyhow::Result<()> {
    let handler = match op {
        AiOp::RmsNorm { epsilon }              => rms_norm_handler(*epsilon),
        AiOp::LayerNorm { epsilon, .. }        => layer_norm_handler(*epsilon),
        AiOp::Softmax { axis }                 => softmax_handler(*axis),
        AiOp::Embed                            => embed_handler(),
        AiOp::Dequantize                       => dequant_handler(),
        AiOp::FusedSwiGLU                      => swiglu_handler(),
        AiOp::Reshape { .. } | AiOp::Transpose { .. } => reshape_handler(),
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
        _ => anyhow::bail!("no custom handler registered for op {:?}", op),
    };
    registry.register(id, arity, handler);
    Ok(())
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
