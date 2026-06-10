//! Scaled dot-product attention fusion.
//!
//! Detects the decomposed ONNX SDPA pattern and fuses into a single
//! `AiOp::GroupedQueryAttention` node.
//!
//! # Why this is needed
//!
//! PyTorch exports `torch.nn.functional.scaled_dot_product_attention` as a
//! chain of standard ONNX ops:
//!
//! ```text
//! scores   = MatMul(Q, K^T)          — Q@K^T
//! masked   = Add(scores, causal_mask) — apply mask
//! weights  = Softmax(masked, axis=-1)
//! [guard]  = Where(IsNaN(weights), 0, weights)   — NaN guard (optional)
//! output   = MatMul(weights, V)       — weighted sum
//! ```
//!
//! Fusing into `GroupedQueryAttention` enables:
//! 1. The fused `dispatch_attention` kernel (with inline causal masking)
//! 2. Unified KV cache injection for both ONNX and GGUF models
//! 3. ~30% fewer graph nodes per transformer layer
//!
//! Important: the fused runtime kernel currently supports only the built-in
//! causal mask flag, not an arbitrary additive mask operand. This pass must
//! therefore preserve explicit `Add(scores, mask)` semantics exactly:
//! - maskless SDPA may fuse and rely on `causal=true` when the model is a decoder
//! - explicit masks may fuse only when the mask is provably a causal mask
//! - padding / bidirectional / otherwise dynamic masks must remain decomposed

use super::graph_utils::{
    apply_node_mutations, build_consumer_map, build_producer_map, next_node_id,
};
use super::pipeline::Pass;
use crate::ir::{AiGraph, AiNode, AiOp, TensorId};
use crate::MetaValue;
use std::collections::{HashMap, HashSet};

/// Fuse decomposed SDPA chains into `AiOp::GroupedQueryAttention`.
///
/// When `force_causal` is true, all fused attention ops are marked causal
/// regardless of whether an explicit mask Add was detected in the ONNX graph.
/// This is necessary for causal LM models (GPT, LLaMA, etc.) where PyTorch's
/// `scaled_dot_product_attention` applies the causal mask internally without
/// emitting it as an explicit Add node in the ONNX export.
///
/// When `force_causal` is `None`, the pass auto-detects from graph metadata
/// first (`arch` / `model_type`), then falls back to a `logits` output-name
/// signature. Encoder models (BERT, CLIP) and diffusion components (UNet, VAE)
/// do not match and correctly get bidirectional attention.
pub struct AttentionFusion {
    pub force_causal: Option<bool>,
}

impl Pass for AttentionFusion {
    fn name(&self) -> &str {
        "AttentionFusion"
    }

    fn run(&self, mut graph: AiGraph) -> anyhow::Result<AiGraph> {
        let force_causal = self
            .force_causal
            .unwrap_or_else(|| auto_detect_force_causal(&graph));
        if force_causal {
            tracing::info!("AttentionFusion: forcing causal=true (causal LM detected)");
        }

        let tid_to_node = build_producer_map(&graph);
        let consumers = build_consumer_map(&graph);

        let mut to_remove: HashSet<usize> = HashSet::new();
        let mut replacements: HashMap<usize, AiNode> = HashMap::new();
        let mut next_id = next_node_id(&graph);

        // Cache head params from the first successfully-fused layer so subsequent
        // layers with Dynamic shape dims can reuse them (all layers share architecture).
        let mut cached_heads: Option<(u32, u32, u32)> = None;

        for (node_idx, node) in graph.nodes.iter().enumerate() {
            // Look for the Q@K^T MatMul (first MatMul in the SDPA chain).
            if !matches!(node.op, AiOp::MatMul) || node.inputs.len() < 2 {
                continue;
            }
            if to_remove.contains(&node_idx) {
                continue;
            }

            let qkt_out = match node.outputs.first() {
                Some(&tid) => tid,
                None => continue,
            };

            // Try to match the SDPA chain starting from this MatMul.
            tracing::trace!(node_idx, qkt_out, "AttentionFusion: trying MatMul");
            if let Some(chain) = match_sdpa_chain(qkt_out, &tid_to_node, &consumers, &graph) {
                // Extract Q, K, V tensor IDs from the SDPA chain.
                let q_tid = node.inputs[0]; // Q: [batch, heads, seq, dim]
                let k_tid = node.inputs[1]; // K^T: [batch, heads, dim, seq] (pre-multiplied with scale)
                let v_tid = chain.v_tid; // V: [batch, heads, seq, dim] (expanded to num_q_heads)

                // Infer head dimensions from Q's and K's shapes.
                let (num_heads, num_kv_heads_inferred, head_dim) = {
                    let inferred = infer_all_head_params(q_tid, k_tid, chain.v_tid, &graph);
                    if inferred.0 > 0 && inferred.2 > 0 {
                        inferred
                    } else if let Some(cached) = cached_heads {
                        cached
                    } else {
                        inferred
                    }
                };

                if num_heads == 0 || head_dim == 0 {
                    tracing::trace!(
                        q_tid,
                        num_heads,
                        head_dim,
                        "SDPA: skipping — can't infer head params"
                    );
                    continue;
                }

                // Trace K back through Transpose and Scale to get un-transposed K.
                // The GQA kernel receives pre-Transpose K and handles layout
                // via the heads_first flag. KvSlotInjection then inserts
                // KvWrite on this same pre-Transpose K tensor.
                //
                // NOTE: PyTorch SDPA may split 1/√d across Q and K as separate
                // Mul(Q, 1/√8) + Mul(K, 1/√8). We absorb the K-side Mul here.
                // The Q-side Mul stays in the graph — Q arrives prescaled. The
                // combined effect: Q_prescaled @ K_raw^T * effective_scale
                // = (s_q * Q) @ K^T * s_k = s_q * s_k * Q@K^T = Q@K^T / √d. ✓
                let (k_pre_transpose, effective_scale) = {
                    let (k, k_scale) = find_pre_transpose_with_scale(k_tid, &tid_to_node, &graph);
                    let eff = k_scale.unwrap_or(1.0) * chain.scale;
                    (k, eff)
                };
                let q_input = q_tid;

                // Trace K and V back past GQA expansion ops (Expand/Reshape/
                // Unsqueeze) to find the un-expanded tensors. At runtime,
                // Expand is a no-op so the data stays at num_kv_heads.
                // The attention kernel handles GQA via group_size internally.
                let k_actual = trace_past_expand(k_pre_transpose, &tid_to_node, &graph);
                let v_actual = trace_past_expand(v_tid, &tid_to_node, &graph);

                // Infer true num_kv_heads from the un-expanded K's shape.
                // If the un-expanded K has a Dynamic heads dim, fall back to the
                // cached corrected value from a prior layer (all layers share
                // the same GQA config).
                let k_actual_heads = extract_heads_dim(k_actual, &graph);
                let num_kv_heads = k_actual_heads
                    .map(|h| h as u32)
                    .or_else(|| cached_heads.map(|(_, kv, _)| kv))
                    .unwrap_or(num_kv_heads_inferred);

                // Cache the corrected (num_heads, num_kv_heads, head_dim) so
                // subsequent layers with Dynamic shape dims reuse the right
                // GQA ratio instead of falling back to num_q_heads.
                cached_heads = Some((num_heads, num_kv_heads, head_dim));

                let explicit_causal_mask =
                    is_supported_causal_mask(chain.mask_tid, &tid_to_node, &graph);
                if chain.mask_tid.is_some() && !explicit_causal_mask {
                    tracing::trace!(
                        node_idx,
                        qkt_out,
                        mask_tid = ?chain.mask_tid,
                        "SDPA: skipping fusion — explicit mask is not representable by causal-only kernel"
                    );
                    continue;
                }

                let causal = explicit_causal_mask || (chain.mask_tid.is_none() && force_causal);

                // Mark all chain nodes for removal.
                for &idx in &chain.node_indices {
                    to_remove.insert(idx);
                }
                to_remove.insert(node_idx);

                let fused = AiNode::new(
                    next_id,
                    AiOp::GroupedQueryAttention {
                        num_heads,
                        num_kv_heads,
                        head_dim,
                        scale: Some(effective_scale),
                        causal,
                        heads_first: true,
                        qk_norm: false,
                        rope: false,
                        rope_base: 0.0,
                    },
                    vec![q_input, k_actual, v_actual],
                    vec![chain.output_tid],
                );
                next_id += 1;

                replacements.insert(chain.output_matmul_idx, fused);

                // Ensure the output tensor's shape AND dtype match Q's. The
                // SDPA chain's terminal `MatMul(weights, V)` output tensor was
                // originally registered in `tensor_info` from the ONNX
                // ValueInfoProto. Some exporters (e.g. the onnx-community
                // Qwen2 export) declare this tensor with a dtype that
                // disagrees with the actual SDPA semantics (BOOL was observed
                // — possibly an export-time artifact of an IsNaN-guarded
                // branch annotation). Attention output is **always** Q's
                // dtype semantically — propagate it here so downstream
                // shape/dtype propagation sees the correct type. The diag
                // harness `tests/diag_qwen2_gqa_shapes.rs` is the V&V witness;
                // before this fix, the AttentionCall's `dtype` field (= the
                // GQA node output's dtype after lowering) was set to BOOL,
                // and hologram's CPU kernel computed `kv_total*es` with
                // `es = 1` (BOOL) ⇒ a buffer-size mismatch with the f32-
                // sized BufferRef ⇒ `SlotOutOfRange(K)`.
                {
                    let q_info = graph.tensor_info.get(&q_tid).cloned();
                    if let (Some(qi), Some(out_info)) =
                        (q_info, graph.tensor_info.get_mut(&chain.output_tid))
                    {
                        out_info.shape = qi.shape;
                        out_info.logical_dtype = qi.logical_dtype;
                        out_info.storage_dtype = qi.storage_dtype;
                    }
                }

                tracing::debug!(
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    effective_scale,
                    causal,
                    mask_in_graph = chain.has_mask,
                    force_causal,
                    "AttentionFusion: fused SDPA chain"
                );
            }
        }

        if replacements.is_empty() {
            return Ok(graph);
        }

        let fused_count = replacements.len();
        apply_node_mutations(&mut graph, &to_remove, &mut replacements);

        tracing::info!("AttentionFusion: fused {fused_count} SDPA chain(s)");
        Ok(graph)
    }
}

fn auto_detect_force_causal(graph: &AiGraph) -> bool {
    match graph
        .metadata
        .get("arch")
        .or_else(|| graph.metadata.get("model_type"))
    {
        Some(MetaValue::Str(arch)) if is_causal_arch(arch) => return true,
        Some(MetaValue::Str(_)) => return false,
        _ => {}
    }

    graph.output_names.iter().any(|name| name == "logits")
}

fn is_causal_arch(arch: &str) -> bool {
    let arch = arch.to_ascii_lowercase();
    matches!(
        arch.as_str(),
        "llama"
            | "mistral"
            | "gemma"
            | "gemma2"
            | "qwen"
            | "qwen2"
            | "phi"
            | "phi3"
            | "gpt2"
            | "gpt_neo"
            | "gptj"
            | "gpt_bigcode"
            | "bloom"
            | "falcon"
            | "stablelm"
            | "smollm"
    )
}

// ── Pattern matching ────────────────────────────────────────────────────────

/// Matched SDPA chain info.
struct SdpaChain {
    /// V tensor input to the output MatMul.
    v_tid: TensorId,
    /// Scale factor (from Mul node or default 1.0).
    scale: f32,
    /// Whether a mask (Add) was detected.
    has_mask: bool,
    /// Mask tensor ID (the additive mask input to the Add node), if present.
    #[allow(dead_code)]
    mask_tid: Option<TensorId>,
    /// Output tensor of the final MatMul (attention @ V).
    output_tid: TensorId,
    /// Node index of the output MatMul (for replacement).
    output_matmul_idx: usize,
    /// All node indices to remove (Add, Softmax, IsNaN, Where, output MatMul).
    node_indices: Vec<usize>,
}

/// Try to match: scores → [Mul(scale)] → [Add(mask)] → Softmax → [IsNaN→Where] → MatMul(_, V).
fn match_sdpa_chain(
    scores_tid: TensorId,
    _tid_to_node: &HashMap<TensorId, usize>,
    consumers: &HashMap<TensorId, Vec<(usize, usize)>>,
    graph: &AiGraph,
) -> Option<SdpaChain> {
    let mut current_tid = scores_tid;
    let mut chain_indices: Vec<usize> = Vec::new();
    let mut scale = 1.0f32;
    let mut has_mask = false;

    // The Q@K^T MatMul output may have multiple consumers (e.g., Add + shape subgraph).
    let all_consumers: Vec<(usize, &AiOp)> = consumers
        .get(&scores_tid)
        .map(|c| {
            c.iter()
                .map(|&(idx, _)| (idx, &graph.nodes[idx].op))
                .collect()
        })
        .unwrap_or_default();
    tracing::trace!(
        scores_tid,
        consumers = ?all_consumers.iter().map(|(i, op)| (i, format!("{op:?}").chars().take(20).collect::<String>())).collect::<Vec<_>>(),
        "SDPA: chain consumers"
    );

    // Optional: Mul(scores, scale_constant) or Div(scores, scalar_constant) —
    // scale the attention scores before masking/softmax.
    if let Some((next, scale_val)) = find_score_scale_consumer(current_tid, consumers, graph) {
        scale = scale_val;
        chain_indices.push(next);
        current_tid = *graph.nodes[next].outputs.first()?;
    }

    // Optional: Add(scores, mask) — absorb into chain and capture mask tensor ID.
    // The mask tensor is passed as a 4th input to the fused Attention op,
    // which applies it as an additive mask before Softmax.
    let mut mask_tid: Option<TensorId> = None;
    if let Some(next) =
        find_consumer_by_op(current_tid, consumers, graph, |op| matches!(op, AiOp::Add))
    {
        let n = &graph.nodes[next];
        if n.inputs.len() >= 2 && (n.inputs[0] == current_tid || n.inputs[1] == current_tid) {
            has_mask = true;
            mask_tid = if n.inputs[0] == current_tid {
                Some(n.inputs[1])
            } else {
                Some(n.inputs[0])
            };
            chain_indices.push(next);
            current_tid = *n.outputs.first()?;
        }
    }

    // Required: Softmax(axis=-1).
    let softmax_consumer = find_consumer_by_op(current_tid, consumers, graph, |op| {
        matches!(op, AiOp::Softmax { .. })
    })?;
    let softmax_node = &graph.nodes[softmax_consumer];
    chain_indices.push(softmax_consumer);
    current_tid = *softmax_node.outputs.first()?;

    tracing::trace!(current_tid, "SDPA: after softmax");

    // Optional: IsNaN → Where (NaN guard, common in PyTorch SDPA export).
    if let Some(next) = find_consumer_by_op(current_tid, consumers, graph, |op| {
        matches!(op, AiOp::IsNaN)
    }) {
        let n = &graph.nodes[next];
        if matches!(n.op, AiOp::IsNaN) {
            // IsNaN feeds into Where.
            let isnan_out = *n.outputs.first()?;
            chain_indices.push(next);
            if let Some(where_idx) = single_consumer(isnan_out, consumers) {
                let where_node = &graph.nodes[where_idx];
                if matches!(where_node.op, AiOp::Where) {
                    chain_indices.push(where_idx);
                    current_tid = *where_node.outputs.first()?;
                }
            }
        }
    }

    tracing::trace!(current_tid, "SDPA: looking for output MatMul");

    // Required: MatMul(weights, V) — the output attention multiplication.
    let out_matmul = find_consumer_by_op(current_tid, consumers, graph, |op| {
        matches!(op, AiOp::MatMul)
    })?;
    let out_node = &graph.nodes[out_matmul];
    if !matches!(out_node.op, AiOp::MatMul) || out_node.inputs.len() < 2 {
        return None;
    }

    // V is the other input to this MatMul (not the softmax output).
    let v_tid = if out_node.inputs[0] == current_tid {
        out_node.inputs[1]
    } else if out_node.inputs[1] == current_tid {
        out_node.inputs[0]
    } else {
        return None;
    };

    let output_tid = *out_node.outputs.first()?;

    Some(SdpaChain {
        v_tid,
        scale,
        has_mask,
        mask_tid,
        output_tid,
        output_matmul_idx: out_matmul,
        node_indices: chain_indices,
    })
}

fn find_score_scale_consumer(
    tid: TensorId,
    consumers: &HashMap<TensorId, Vec<(usize, usize)>>,
    graph: &AiGraph,
) -> Option<(usize, f32)> {
    let consumer = find_consumer_by_op(tid, consumers, graph, |op| {
        matches!(op, AiOp::Mul | AiOp::Div)
    })?;
    let node = &graph.nodes[consumer];
    if node.inputs.len() < 2 {
        return None;
    }

    match node.op {
        AiOp::Mul => {
            let (other_idx, scale_val) = if let Some(s) = scalar_param(node.inputs[1], graph) {
                (0, s)
            } else if let Some(s) = scalar_param(node.inputs[0], graph) {
                (1, s)
            } else {
                return None;
            };
            (node.inputs[other_idx] == tid).then_some((consumer, scale_val))
        }
        AiOp::Div => {
            let denom = scalar_param(node.inputs[1], graph)?;
            if node.inputs[0] != tid || denom == 0.0 {
                return None;
            }
            Some((consumer, 1.0 / denom))
        }
        _ => None,
    }
}

/// Find a consumer of a tensor whose op matches a predicate.
/// Returns the first matching consumer's node index, or `None`.
fn find_consumer_by_op(
    tid: TensorId,
    consumers: &HashMap<TensorId, Vec<(usize, usize)>>,
    graph: &AiGraph,
    pred: impl Fn(&AiOp) -> bool,
) -> Option<usize> {
    let c = consumers.get(&tid)?;
    c.iter()
        .map(|&(node_idx, _)| node_idx)
        .find(|&node_idx| pred(&graph.nodes[node_idx].op))
}

/// Find the single consumer of a tensor. Returns `None` if 0 or 2+ consumers.
fn single_consumer(
    tid: TensorId,
    consumers: &HashMap<TensorId, Vec<(usize, usize)>>,
) -> Option<usize> {
    let c = consumers.get(&tid)?;
    if c.len() == 1 {
        Some(c[0].0)
    } else {
        None
    }
}

/// Extract a scalar f32 from a param tensor.
fn scalar_param(tid: TensorId, graph: &AiGraph) -> Option<f32> {
    use crate::ir::AiParam;
    match graph.params.get(&tid)? {
        AiParam::Inline { data, .. } if data.len() == 4 => {
            let arr: [u8; 4] = data.as_slice().try_into().ok()?;
            Some(f32::from_le_bytes(arr))
        }
        _ => None,
    }
}

fn is_supported_causal_mask(
    mask_tid: Option<TensorId>,
    tid_to_node: &HashMap<TensorId, usize>,
    graph: &AiGraph,
) -> bool {
    let Some(mask_tid) = mask_tid else {
        return false;
    };
    is_supported_causal_mask_tensor(mask_tid, tid_to_node, graph)
}

fn is_supported_causal_mask_tensor(
    tid: TensorId,
    tid_to_node: &HashMap<TensorId, usize>,
    graph: &AiGraph,
) -> bool {
    if inline_param_is_causal_mask(tid, graph) {
        return true;
    }

    let Some(&producer_idx) = tid_to_node.get(&tid) else {
        return false;
    };
    let producer = &graph.nodes[producer_idx];
    match &producer.op {
        AiOp::Identity | AiOp::Expand | AiOp::Reshape { .. } | AiOp::Unsqueeze { .. } => producer
            .inputs
            .first()
            .copied()
            .is_some_and(|input| is_supported_causal_mask_tensor(input, tid_to_node, graph)),
        AiOp::Transpose { perm } if is_last_two_dim_swap(perm) => producer
            .inputs
            .first()
            .copied()
            .is_some_and(|input| is_supported_causal_mask_tensor(input, tid_to_node, graph)),
        _ => false,
    }
}

fn inline_param_is_causal_mask(tid: TensorId, graph: &AiGraph) -> bool {
    use crate::ir::AiParam;

    let Some(AiParam::Inline { data, info }) = graph.params.get(&tid) else {
        return false;
    };
    if info.logical_dtype != crate::DType::F32 {
        return false;
    }

    let dims = info
        .shape
        .as_slice()
        .iter()
        .map(crate::ir::DimExpr::evaluate)
        .collect::<Option<Vec<_>>>();
    let Some(dims) = dims else {
        return false;
    };
    if dims.len() < 2 {
        return false;
    }

    let rows = dims[dims.len() - 2] as usize;
    let cols = dims[dims.len() - 1] as usize;
    if rows == 0 || rows != cols {
        return false;
    }

    let leading = dims[..dims.len() - 2]
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim as usize));
    let Some(leading) = leading else {
        return false;
    };
    let expected = leading
        .checked_mul(rows)
        .and_then(|n| n.checked_mul(cols))
        .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()));
    if expected != Some(data.len()) {
        return false;
    }

    let values: Vec<f32> = data
        .chunks_exact(4)
        .map(|chunk| {
            let bytes: [u8; 4] = chunk.try_into().expect("4-byte f32 chunk");
            f32::from_le_bytes(bytes)
        })
        .collect();

    (0..leading).all(|plane| plane_is_causal_mask(&values, plane, rows, cols))
}

fn plane_is_causal_mask(values: &[f32], plane: usize, rows: usize, cols: usize) -> bool {
    let plane_stride = rows * cols;
    let base = plane * plane_stride;

    (0..rows).all(|row| {
        (0..cols).all(|col| {
            let value = values[base + row * cols + col];
            let allowed = col <= row;
            causal_mask_entry_matches(value, allowed)
        })
    })
}

fn causal_mask_entry_matches(value: f32, allowed: bool) -> bool {
    if allowed {
        return value.abs() <= 1e-6;
    }
    value.is_infinite() && value.is_sign_negative() || value <= -1e4
}

fn is_last_two_dim_swap(perm: &[u32]) -> bool {
    if perm.len() < 2 {
        return false;
    }
    let tail = perm.len() - 2;
    let prefix_ok = perm
        .iter()
        .take(tail)
        .enumerate()
        .all(|(idx, &axis)| axis as usize == idx);
    if !prefix_ok {
        return false;
    }
    perm[tail] as usize == tail + 1 && perm[tail + 1] as usize == tail
}

/// Infer (num_heads, num_kv_heads, head_dim) from Q, K, V tensor shapes.
///
/// For 4-D shapes `[batch, heads, seq, dim]`, tries direct dim extraction first.
/// Falls back to inferring heads from the product of known dims when some are Dynamic.
fn infer_all_head_params(
    q_tid: TensorId,
    k_tid: TensorId,
    _v_tid: TensorId,
    graph: &AiGraph,
) -> (u32, u32, u32) {
    let head_dim = extract_last_dim(q_tid, graph).unwrap_or(0);
    if head_dim == 0 {
        return (0, 0, 0);
    }

    let q_heads = extract_heads_dim(q_tid, graph);
    let k_heads = extract_heads_dim(k_tid, graph);

    let num_heads = q_heads.unwrap_or(0) as u32;
    let num_kv_heads = k_heads.unwrap_or(num_heads as u64) as u32;

    (num_heads, num_kv_heads, head_dim as u32)
}

/// Extract the last dimension (head_dim) from a tensor shape.
fn extract_last_dim(tid: TensorId, graph: &AiGraph) -> Option<u64> {
    let info = graph.tensor_info.get(&tid)?;
    let shape = info.shape.as_slice();
    shape.last()?.evaluate()
}

/// Extract the heads dimension from a 4-D shape [batch, heads, seq, dim].
/// Returns `Some(heads)` if the heads dim (position 1 in 4-D, position 0 in 3-D) is concrete.
/// Also tries to infer from the total product when heads is Dynamic but other dims are known.
fn extract_heads_dim(tid: TensorId, graph: &AiGraph) -> Option<u64> {
    let info = graph.tensor_info.get(&tid)?;
    let shape = info.shape.as_slice();
    if shape.len() >= 4 {
        // Direct: shape[1] is heads.
        if let Some(h) = shape[1].evaluate() {
            return Some(h);
        }
        // Fallback: if batch and head_dim are concrete, infer heads from any
        // known element count on this tensor. This is the common case where
        // shape propagation resolves batch=1 and head_dim=64 but leaves
        // heads and seq as Dynamic.
        //
        // We can't infer without more info, so return None and let the caller
        // decide how to handle it.
        return None;
    }
    if shape.len() >= 3 {
        return shape[0].evaluate();
    }
    None
}

/// Trace a tensor back through Transpose/Reshape/View/Mul nodes to find the
/// un-transposed K input and any scale applied on the K path.
///
/// Returns `(k_tensor_id, optional_scale)`:
/// - `k_tensor_id`: the un-transposed K (input to the Transpose), or the
///   original `tid` if no Transpose is found.
/// - `optional_scale`: scalar from any Mul(K, scalar) node on the path.
fn find_pre_transpose_with_scale(
    tid: TensorId,
    tid_to_node: &HashMap<TensorId, usize>,
    graph: &AiGraph,
) -> (TensorId, Option<f32>) {
    let mut current = tid;
    let mut accumulated_scale: Option<f32> = None;

    // Walk back through at most 5 nodes looking for a Transpose.
    for _ in 0..5 {
        let node_idx = match tid_to_node.get(&current) {
            Some(&idx) => idx,
            None => return (tid, accumulated_scale),
        };
        let node = &graph.nodes[node_idx];
        match &node.op {
            AiOp::Transpose { .. } => {
                let pre = node.inputs.first().copied().unwrap_or(tid);
                return (pre, accumulated_scale);
            }
            // Reshape/View/Unsqueeze/Expand don't change data at runtime in
            // hologram's tape executor — keep tracing through them.
            AiOp::Reshape { .. }
            | AiOp::Flatten { .. }
            | AiOp::Squeeze { .. }
            | AiOp::Unsqueeze { .. }
            | AiOp::Expand
            | AiOp::Identity => {
                current = match node.inputs.first() {
                    Some(&inp) => inp,
                    None => return (tid, accumulated_scale),
                };
            }
            // Mul(tensor, scalar): extract scale and continue tracing the tensor.
            AiOp::Mul if node.inputs.len() >= 2 => {
                let s1 = scalar_param(node.inputs[1], graph);
                let s0 = scalar_param(node.inputs[0], graph);
                if let Some(sv) = s1 {
                    accumulated_scale = Some(accumulated_scale.unwrap_or(1.0) * sv);
                    current = node.inputs[0];
                } else if let Some(sv) = s0 {
                    accumulated_scale = Some(accumulated_scale.unwrap_or(1.0) * sv);
                    current = node.inputs[1];
                } else {
                    return (tid, accumulated_scale);
                }
            }
            _ => return (tid, accumulated_scale),
        }
    }
    (tid, accumulated_scale)
}

/// Trace a tensor back past GQA expansion ops (Expand, Reshape, Unsqueeze)
/// to find the un-expanded version with the true num_kv_heads.
///
/// In ONNX GQA models (e.g., TinyLlama), K/V are expanded from
/// `[batch, num_kv_heads, seq, head_dim]` to `[batch, num_q_heads, seq, head_dim]`
/// via Unsqueeze→Expand→Reshape before the SDPA MatMul. At runtime, Expand is
/// a no-op (lowered to Reshape/identity), so the data stays at num_kv_heads.
///
/// Returns the tensor ID of the un-expanded K or V with the true head count.
fn trace_past_expand(
    tid: TensorId,
    tid_to_node: &HashMap<TensorId, usize>,
    graph: &AiGraph,
) -> TensorId {
    let mut current = tid;
    for _ in 0..8 {
        let node_idx = match tid_to_node.get(&current) {
            Some(&idx) => idx,
            None => return current,
        };
        let node = &graph.nodes[node_idx];
        match &node.op {
            // These ops don't change data at runtime — trace through them.
            AiOp::Reshape { .. }
            | AiOp::Flatten { .. }
            | AiOp::Squeeze { .. }
            | AiOp::Unsqueeze { .. }
            | AiOp::Expand
            | AiOp::Identity => {
                current = match node.inputs.first() {
                    Some(&inp) => inp,
                    None => return current,
                };
            }
            // Stop at any "real" op (MatMul projection, Transpose, etc.)
            _ => return current,
        }
    }
    current
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{AiParam, DType, TensorInfo};
    use crate::opt::pipeline::Pass;

    fn shape_4d(b: u64, h: u64, s: u64, d: u64) -> crate::ir::Shape {
        crate::shape_from_concrete(&[b, h, s, d])
    }

    fn f32_info(shape: crate::ir::Shape) -> TensorInfo {
        TensorInfo::new(DType::F32, shape)
    }

    /// Build a minimal SDPA graph: Q@K^T → Add(mask) → Softmax → MatMul(V)
    fn build_sdpa_graph() -> AiGraph {
        let mut graph = AiGraph {
            name: "test_sdpa".into(),
            nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            input_names: Vec::new(),
            output_names: Vec::new(),
            params: HashMap::new(),
            tensor_info: HashMap::new(),
            metadata: HashMap::new(),
            warnings: Vec::new(),
            dim_vars: crate::ir::DimVarTable::default(),
            shape_constraints: crate::ir::ConstraintStore::default(),
            subgraphs: HashMap::new(),
            tensor_names: HashMap::new(),
            topo_cache: Default::default(),
        };
        let mut next_tid: TensorId = 0;
        let mut next_nid: u32 = 0;

        let alloc_tid = |next: &mut TensorId| -> TensorId {
            let t = *next;
            *next += 1;
            t
        };

        // Tensors: Q=0, K_T=1, V=2, mask=3, scale_val=4
        let q = alloc_tid(&mut next_tid); // 0
        let k_t = alloc_tid(&mut next_tid); // 1
        let v = alloc_tid(&mut next_tid); // 2
        let mask = alloc_tid(&mut next_tid); // 3
        let scale_val = alloc_tid(&mut next_tid); // 4

        graph.tensor_info.insert(q, f32_info(shape_4d(1, 4, 8, 16)));
        graph
            .tensor_info
            .insert(k_t, f32_info(shape_4d(1, 4, 16, 8)));
        graph.tensor_info.insert(v, f32_info(shape_4d(1, 4, 8, 16)));
        graph
            .tensor_info
            .insert(mask, f32_info(shape_4d(1, 1, 8, 8)));

        // Scale = 0.25 (1/sqrt(16))
        graph.params.insert(
            scale_val,
            AiParam::Inline {
                data: 0.25f32.to_le_bytes().to_vec().into(),
                info: f32_info(crate::shape_from_concrete(&[1])),
            },
        );

        // MatMul(Q, K^T) → scores
        let scores = alloc_tid(&mut next_tid);
        graph
            .tensor_info
            .insert(scores, f32_info(shape_4d(1, 4, 8, 8)));
        graph.nodes.push(AiNode::new(
            {
                let n = next_nid;
                next_nid += 1;
                n
            },
            AiOp::MatMul,
            vec![q, k_t],
            vec![scores],
        ));

        // Mul(scores, scale) → scaled
        let scaled = alloc_tid(&mut next_tid);
        graph
            .tensor_info
            .insert(scaled, f32_info(shape_4d(1, 4, 8, 8)));
        graph.nodes.push(AiNode::new(
            {
                let n = next_nid;
                next_nid += 1;
                n
            },
            AiOp::Mul,
            vec![scores, scale_val],
            vec![scaled],
        ));

        // Add(scaled, mask) → masked
        let masked = alloc_tid(&mut next_tid);
        graph
            .tensor_info
            .insert(masked, f32_info(shape_4d(1, 4, 8, 8)));
        graph.nodes.push(AiNode::new(
            {
                let n = next_nid;
                next_nid += 1;
                n
            },
            AiOp::Add,
            vec![scaled, mask],
            vec![masked],
        ));

        // Softmax(masked) → weights
        let weights = alloc_tid(&mut next_tid);
        graph
            .tensor_info
            .insert(weights, f32_info(shape_4d(1, 4, 8, 8)));
        graph.nodes.push(AiNode::new(
            {
                let n = next_nid;
                next_nid += 1;
                n
            },
            AiOp::Softmax { axis: -1 },
            vec![masked],
            vec![weights],
        ));

        // MatMul(weights, V) → output
        let output = alloc_tid(&mut next_tid);
        graph
            .tensor_info
            .insert(output, f32_info(shape_4d(1, 4, 8, 16)));
        graph.nodes.push(AiNode::new(
            {
                let _n = next_nid;
                next_nid += 1;
                _n
            },
            AiOp::MatMul,
            vec![weights, v],
            vec![output],
        ));

        graph.inputs = vec![q, k_t, v, mask];
        graph.outputs = vec![output];
        let _ = next_nid; // suppress warning
        graph
    }

    fn build_sdpa_div_graph() -> AiGraph {
        let mut graph = build_sdpa_graph();
        let scores = graph.nodes[0].outputs[0];
        let scaled = graph.nodes[1].outputs[0];
        let scale_val = graph.nodes[1].inputs[1];
        graph.params.insert(
            scale_val,
            AiParam::Inline {
                data: 4.0f32.to_le_bytes().to_vec().into(),
                info: f32_info(crate::shape_from_concrete(&[1])),
            },
        );
        graph.nodes[1] = AiNode::new(1, AiOp::Div, vec![scores, scale_val], vec![scaled]);
        graph
    }

    #[test]
    fn sdpa_chain_with_runtime_mask_does_not_fuse() {
        let graph = build_sdpa_graph();
        let original_nodes = graph.nodes.len();

        let fused = AttentionFusion {
            force_causal: Some(false),
        }
        .run(graph)
        .expect("fusion failed");

        assert_eq!(
            fused.nodes.len(),
            original_nodes,
            "runtime padding masks must remain decomposed, got {} nodes: {:?}",
            fused.nodes.len(),
            fused.nodes.iter().map(|n| &n.op).collect::<Vec<_>>()
        );
        assert!(matches!(fused.nodes[2].op, AiOp::Add));
    }

    fn build_sdpa_constant_causal_mask_graph() -> AiGraph {
        let mut graph = build_sdpa_graph();
        let mask = graph.inputs[3];
        graph.inputs.pop();
        graph
            .tensor_info
            .insert(mask, f32_info(shape_4d(1, 1, 8, 8)));

        let mask_values: Vec<f32> = (0..8usize)
            .flat_map(|row| {
                (0..8usize).map(move |col| if col <= row { 0.0 } else { f32::NEG_INFINITY })
            })
            .collect();
        let mask_bytes: Vec<u8> = mask_values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        graph.params.insert(
            mask,
            AiParam::Inline {
                data: mask_bytes.into(),
                info: f32_info(shape_4d(1, 1, 8, 8)),
            },
        );
        graph
    }

    #[test]
    fn sdpa_chain_with_constant_causal_mask_fuses_to_gqa() {
        let graph = build_sdpa_constant_causal_mask_graph();
        assert_eq!(graph.nodes.len(), 5);

        let fused = AttentionFusion {
            force_causal: Some(false),
        }
        .run(graph)
        .expect("fusion failed");

        assert_eq!(
            fused.nodes.len(),
            1,
            "expected 1 fused node, got {}: {:?}",
            fused.nodes.len(),
            fused.nodes.iter().map(|n| &n.op).collect::<Vec<_>>()
        );

        match &fused.nodes[0].op {
            AiOp::GroupedQueryAttention {
                num_heads,
                num_kv_heads,
                head_dim,
                scale,
                causal,
                ..
            } => {
                assert_eq!(*num_heads, 4);
                assert_eq!(*num_kv_heads, 4);
                assert_eq!(*head_dim, 16);
                assert!((scale.expect("scale") - 0.25).abs() < 1e-6);
                assert!(*causal, "should detect causal mask");
            }
            other => panic!("expected GroupedQueryAttention, got {other:?}"),
        }
    }

    #[test]
    fn sdpa_div_chain_fuses_to_gqa() {
        let graph = {
            let mut g = build_sdpa_div_graph();
            let mask = g.inputs[3];
            g.inputs.pop();
            let mask_values: Vec<f32> = (0..8usize)
                .flat_map(|row| {
                    (0..8usize).map(move |col| if col <= row { 0.0 } else { f32::NEG_INFINITY })
                })
                .collect();
            let mask_bytes: Vec<u8> = mask_values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            g.params.insert(
                mask,
                AiParam::Inline {
                    data: mask_bytes.into(),
                    info: f32_info(shape_4d(1, 1, 8, 8)),
                },
            );
            g
        };
        let fused = AttentionFusion {
            force_causal: Some(false),
        }
        .run(graph)
        .expect("fusion failed");

        assert_eq!(fused.nodes.len(), 1);
        match &fused.nodes[0].op {
            AiOp::GroupedQueryAttention { scale, .. } => {
                assert!((scale.expect("scale") - 0.25).abs() < 1e-6);
            }
            other => panic!("expected GroupedQueryAttention, got {other:?}"),
        }
    }

    #[test]
    fn runtime_mask_is_not_overridden_by_force_causal() {
        let graph = build_sdpa_graph();
        let fused = AttentionFusion {
            force_causal: Some(true),
        }
        .run(graph)
        .expect("fusion failed");

        assert_eq!(fused.nodes.len(), 5);
        assert!(matches!(fused.nodes[2].op, AiOp::Add));
    }

    /// Build a minimal SDPA graph WITHOUT the Add(mask) step:
    /// Q@K^T → Softmax → MatMul(V). This mimics PyTorch SDPA export
    /// where the causal mask is applied internally, not as an explicit op.
    fn build_maskless_sdpa_graph(output_name: &str) -> AiGraph {
        let mut graph = AiGraph {
            name: "test_maskless_sdpa".into(),
            nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            input_names: vec!["input_ids".into()],
            output_names: vec![output_name.into()],
            params: HashMap::new(),
            tensor_info: HashMap::new(),
            metadata: HashMap::new(),
            warnings: Vec::new(),
            dim_vars: crate::ir::DimVarTable::default(),
            shape_constraints: crate::ir::ConstraintStore::default(),
            subgraphs: HashMap::new(),
            tensor_names: HashMap::new(),
            topo_cache: Default::default(),
        };
        let mut next_tid: TensorId = 0;
        let mut next_nid: u32 = 0;
        let alloc_tid = |next: &mut TensorId| -> TensorId {
            let t = *next;
            *next += 1;
            t
        };

        let q = alloc_tid(&mut next_tid);
        let k_t = alloc_tid(&mut next_tid);
        let v = alloc_tid(&mut next_tid);

        graph.tensor_info.insert(q, f32_info(shape_4d(1, 4, 8, 16)));
        graph
            .tensor_info
            .insert(k_t, f32_info(shape_4d(1, 4, 16, 8)));
        graph.tensor_info.insert(v, f32_info(shape_4d(1, 4, 8, 16)));

        // MatMul(Q, K^T) → scores
        let scores = alloc_tid(&mut next_tid);
        graph
            .tensor_info
            .insert(scores, f32_info(shape_4d(1, 4, 8, 8)));
        graph.nodes.push(AiNode::new(
            {
                let n = next_nid;
                next_nid += 1;
                n
            },
            AiOp::MatMul,
            vec![q, k_t],
            vec![scores],
        ));

        // Softmax(scores) → weights (no Add/mask in between)
        let weights = alloc_tid(&mut next_tid);
        graph
            .tensor_info
            .insert(weights, f32_info(shape_4d(1, 4, 8, 8)));
        graph.nodes.push(AiNode::new(
            {
                let n = next_nid;
                next_nid += 1;
                n
            },
            AiOp::Softmax { axis: -1 },
            vec![scores],
            vec![weights],
        ));

        // MatMul(weights, V) → output
        let output = alloc_tid(&mut next_tid);
        graph
            .tensor_info
            .insert(output, f32_info(shape_4d(1, 4, 8, 16)));
        graph.nodes.push(AiNode::new(
            {
                let _n = next_nid;
                next_nid += 1;
                _n
            },
            AiOp::MatMul,
            vec![weights, v],
            vec![output],
        ));

        graph.inputs = vec![q, k_t, v];
        graph.outputs = vec![output];
        let _ = next_nid;
        graph
    }

    #[test]
    fn maskless_sdpa_auto_detects_causal_for_llm() {
        // Graph with output named "logits" → auto-detect should force causal.
        let graph = build_maskless_sdpa_graph("logits");
        let fused = AttentionFusion { force_causal: None }
            .run(graph)
            .expect("fusion failed");

        assert_eq!(fused.nodes.len(), 1);
        match &fused.nodes[0].op {
            AiOp::GroupedQueryAttention { causal, .. } => {
                assert!(*causal, "auto-detect should force causal for LLM output");
            }
            other => panic!("expected GroupedQueryAttention, got {other:?}"),
        }
    }

    #[test]
    fn maskless_sdpa_not_causal_for_encoder() {
        // Graph with output named "last_hidden_state" → should NOT be causal.
        let graph = build_maskless_sdpa_graph("last_hidden_state");
        let fused = AttentionFusion { force_causal: None }
            .run(graph)
            .expect("fusion failed");

        assert_eq!(fused.nodes.len(), 1);
        match &fused.nodes[0].op {
            AiOp::GroupedQueryAttention { causal, .. } => {
                assert!(!*causal, "encoder attention should not be causal");
            }
            other => panic!("expected GroupedQueryAttention, got {other:?}"),
        }
    }

    #[test]
    fn force_causal_override_true() {
        // Explicitly force causal even for encoder-like output.
        let graph = build_maskless_sdpa_graph("last_hidden_state");
        let fused = AttentionFusion {
            force_causal: Some(true),
        }
        .run(graph)
        .expect("fusion failed");

        assert_eq!(fused.nodes.len(), 1);
        match &fused.nodes[0].op {
            AiOp::GroupedQueryAttention { causal, .. } => {
                assert!(
                    *causal,
                    "force_causal=Some(true) should override auto-detect"
                );
            }
            other => panic!("expected GroupedQueryAttention, got {other:?}"),
        }
    }
}
