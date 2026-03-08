//! Builds a `hologram::Graph` from a dispatched `AiGraph`.
//!
//! Uses `hologram::GraphBuilder` (fluent, index-based): each node-adding method
//! increments the builder's index counter; `tid_to_idx` maps `TensorId` → builder index.

use super::dispatch::{dispatch, DispatchTarget};
use super::LowerPhase;
use crate::ir::{AiGraph, AiOp, Dim, TensorId, TensorInfo};
use crate::mem::KvCacheLayout;
use anyhow::Context;
use hologram::{f32_to_bits, ConstantData, FloatDType, FloatOp, GraphBuilder, GraphOp};
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

    // Register named graph inputs and insert Input nodes.
    for (i, &tid) in ai_graph.inputs.iter().enumerate() {
        let name = ai_graph.input_name(i);
        builder = builder.input(name);
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
        builder = builder.constant(constant);
        tid_to_idx.insert(tid, builder.len() - 1);
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
                if let Some(&tid) = node.outputs.first() {
                    tid_to_idx.insert(tid, builder.len() - 1);
                }
            }
            DispatchTarget::FloatNeedsShape => {
                let graph_op = resolve_float_op(&node.op, &node.inputs, &ai_graph.tensor_info)?;
                builder = builder.node_with_inputs(graph_op, &input_idxs);
                if let Some(&tid) = node.outputs.first() {
                    tid_to_idx.insert(tid, builder.len() - 1);
                }
            }
            DispatchTarget::Identity => {
                // Pass-through: output tensor maps to the same index as the input.
                if let (Some(&in_tid), Some(&out_tid)) = (node.inputs.first(), node.outputs.first())
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
    Ok(LoweringOutput {
        graph,
        layer_name: phase.layer_name().to_string(),
    })
}

// ── Float op shape resolution ─────────────────────────────────────────────────

/// Resolve an `AiOp` that needs tensor shape info into a `GraphOp::Float(FloatOp::...)`.
fn resolve_float_op(
    op: &AiOp,
    inputs: &[TensorId],
    tensor_info: &HashMap<TensorId, TensorInfo>,
) -> anyhow::Result<GraphOp> {
    let float_op = match op {
        AiOp::MatMul | AiOp::BatchMatMul => {
            let k = last_dim(inputs.first(), tensor_info).unwrap_or(1) as u32;
            let n = last_dim(inputs.get(1), tensor_info).unwrap_or(1) as u32;
            let m = second_last_dim(inputs.first(), tensor_info).unwrap_or(1) as u32;
            FloatOp::MatMul { m, k, n }
        }
        AiOp::Gemm {
            alpha,
            beta,
            trans_a,
            trans_b,
        } => {
            let k = last_dim(inputs.first(), tensor_info).unwrap_or(1) as u32;
            let n = last_dim(inputs.get(1), tensor_info).unwrap_or(1) as u32;
            let m = second_last_dim(inputs.first(), tensor_info).unwrap_or(1) as u32;
            FloatOp::Gemm {
                m,
                k,
                n,
                alpha: f32_to_bits(*alpha),
                beta: f32_to_bits(*beta),
                trans_a: *trans_a,
                trans_b: *trans_b,
            }
        }
        AiOp::Softmax { .. } => {
            let size = last_dim(inputs.first(), tensor_info).unwrap_or(1) as u32;
            FloatOp::Softmax { size }
        }
        AiOp::LogSoftmax { .. } => {
            let size = last_dim(inputs.first(), tensor_info).unwrap_or(1) as u32;
            FloatOp::LogSoftmax { size }
        }
        AiOp::RmsNorm { epsilon } => {
            let size = last_dim(inputs.first(), tensor_info).unwrap_or(1) as u32;
            FloatOp::RmsNorm {
                size,
                epsilon: f32_to_bits(*epsilon),
            }
        }
        AiOp::LayerNorm { epsilon, .. } => {
            let size = last_dim(inputs.first(), tensor_info).unwrap_or(1) as u32;
            FloatOp::LayerNorm {
                size,
                epsilon: f32_to_bits(*epsilon),
            }
        }
        AiOp::ReduceSum { .. } => {
            let size = last_dim(inputs.first(), tensor_info).unwrap_or(1) as u32;
            FloatOp::ReduceSum { size }
        }
        AiOp::ReduceMean { .. } => {
            let size = last_dim(inputs.first(), tensor_info).unwrap_or(1) as u32;
            FloatOp::ReduceMean { size }
        }
        AiOp::ReduceMax { .. } => {
            let size = last_dim(inputs.first(), tensor_info).unwrap_or(1) as u32;
            FloatOp::ReduceMax { size }
        }
        AiOp::ReduceMin { .. } => {
            let size = last_dim(inputs.first(), tensor_info).unwrap_or(1) as u32;
            FloatOp::ReduceMin { size }
        }
        AiOp::Gather { .. } | AiOp::GatherElements { .. } => {
            let dim = last_dim(inputs.first(), tensor_info).unwrap_or(1) as u32;
            FloatOp::Gather { dim }
        }
        AiOp::Concat { .. } => {
            let size_a = last_dim(inputs.first(), tensor_info).unwrap_or(1) as u32;
            let size_b = last_dim(inputs.get(1), tensor_info).unwrap_or(1) as u32;
            FloatOp::Concat { size_a, size_b }
        }
        AiOp::Embed => {
            let dim = last_dim(inputs.get(1), tensor_info).unwrap_or(1) as u32;
            FloatOp::Embed { dim }
        }
        AiOp::MultiHeadAttention {
            head_dim,
            scale,
            causal,
            ..
        } => {
            let s = scale.unwrap_or((*head_dim as f32).sqrt().recip());
            FloatOp::Attention {
                head_dim: *head_dim,
                scale: f32_to_bits(s),
                causal: *causal,
            }
        }
        AiOp::GroupedQueryAttention {
            head_dim,
            scale,
            causal,
            ..
        } => {
            let s = scale.unwrap_or((*head_dim as f32).sqrt().recip());
            FloatOp::Attention {
                head_dim: *head_dim,
                scale: f32_to_bits(s),
                causal: *causal,
            }
        }
        AiOp::FlashAttentionHint => FloatOp::Attention {
            head_dim: 64,
            scale: f32_to_bits(0.125),
            causal: true,
        },
        AiOp::Cast { to } => {
            let from = input_float_dtype(inputs.first(), tensor_info);
            FloatOp::Cast {
                from,
                to: ai_dtype_to_float_dtype(to),
            }
        }
        AiOp::Shape => {
            let dtype = input_float_dtype(inputs.first(), tensor_info);
            FloatOp::Shape { dtype }
        }
        _ => anyhow::bail!("resolve_float_op: unexpected op {:?}", op),
    };
    Ok(GraphOp::Float(float_op))
}

// ── Input reordering ─────────────────────────────────────────────────────────

/// ONNX Gather/GatherElements: `(data, indices)` → hologram executor: `(indices, data)`.
fn swap_gather_inputs(op: &AiOp, mut idxs: Vec<usize>) -> Vec<usize> {
    if matches!(op, AiOp::Gather { .. } | AiOp::GatherElements { .. }) && idxs.len() >= 2 {
        idxs.swap(0, 1);
    }
    idxs
}

// ── Dtype conversion ─────────────────────────────────────────────────────────

/// Convert hologram-ai `DType` to hologram base crate `FloatDType`.
fn ai_dtype_to_float_dtype(dtype: &crate::ir::DType) -> FloatDType {
    use crate::ir::DType;
    match dtype {
        DType::F32 => FloatDType::F32,
        DType::F16 => FloatDType::F16,
        DType::BF16 => FloatDType::BF16,
        DType::INT8 => FloatDType::I8,
        DType::INT4 => FloatDType::I8,
        DType::U8 => FloatDType::U8,
        DType::INT32 => FloatDType::I32,
        DType::INT64 => FloatDType::I64,
        DType::BOOL => FloatDType::Bool,
    }
}

/// Look up the logical dtype of the first input tensor, defaulting to F32.
fn input_float_dtype(
    tid: Option<&TensorId>,
    tensor_info: &HashMap<TensorId, TensorInfo>,
) -> FloatDType {
    tid.and_then(|t| tensor_info.get(t))
        .map(|info| ai_dtype_to_float_dtype(&info.logical_dtype))
        .unwrap_or(FloatDType::F32)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Extract last concrete dimension from a tensor.
fn last_dim(tid: Option<&TensorId>, tensor_info: &HashMap<TensorId, TensorInfo>) -> Option<u64> {
    tid.and_then(|t| tensor_info.get(t))
        .and_then(|info| info.shape.last())
        .and_then(concrete_dim)
}

/// Extract second-to-last concrete dimension from a tensor.
fn second_last_dim(
    tid: Option<&TensorId>,
    tensor_info: &HashMap<TensorId, TensorInfo>,
) -> Option<u64> {
    tid.and_then(|t| tensor_info.get(t))
        .and_then(|info| {
            let n = info.shape.len();
            if n >= 2 {
                info.shape.get(n - 2)
            } else {
                None
            }
        })
        .and_then(concrete_dim)
}

/// Extract the concrete value from a `Dim`, returning `None` for symbolic/dynamic dims.
fn concrete_dim(dim: &Dim) -> Option<u64> {
    match dim {
        Dim::Concrete(n) => Some(*n),
        _ => None,
    }
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
