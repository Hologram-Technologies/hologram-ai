//! Decode-step attention rewrite (dictionary row `decode-plan`).
//!
//! Replaces each fused [`AiOp::GroupedQueryAttention`] node in a **seq-1,
//! batch-1** graph with the decomposed masked past-attention block over a
//! FIXED past bucket, turning the per-token forward pass from a window-sized
//! computation into a single-position one:
//!
//! * carried K/V enter as per-layer graph **inputs** `past_k_l`/`past_v_l`
//!   (`[kv, bucket, head_dim]`) — derived content through named ports, not a
//!   mutable cache. The engine splices each step's `k_new_l`/`v_new_l`
//!   **outputs** into row `pos` of its buffers between steps; no scatter op
//!   exists in the graph.
//! * bucket rows past the realized length hold arbitrary bytes; the additive
//!   `decode_mask` input (`[1, bucket+1]`, `0` / `-1e9`) erases them inside
//!   the softmax, so a fixed bucket never changes the numbers.
//! * RoPE becomes plain arithmetic on **runtime** `rope_cos`/`rope_sin`
//!   inputs (`[1, head_dim]`) the engine synthesizes at the absolute position
//!   — the canonical `rope_rotate` lowering bakes tables by relative index at
//!   compile time, so rotation must arrive as data, not as an op. Rotate-half
//!   pairs `j ± head_dim/2` (non-interleaved), matching the fused kernel.
//! * attention decomposes per kv-head group into strictly 2-D matmuls
//!   (`q_group · past_kᵀ`, softmax over `Concat(scores_past, score_self)`,
//!   `probs · Concat(past_v, v_new)`) — the substrate's MatMul kernel is 2-D,
//!   so no batched-matmul support is assumed.
//!
//! Causality is vacuous at q = 1 (the single query is the last position);
//! `causal` is therefore ignored. `qk_norm` attention is rejected loud — the
//! parametric safetensors recipe never emits it, and silently dropping the
//! norm would be a wrong number, not a missing feature.

use anyhow::{bail, ensure, Context, Result};

use crate::ir::{
    shape_from_concrete, AiGraph, AiNode, AiOp, AiParam, DType, NodeId, TensorId, TensorInfo,
};

/// Named input port for the runtime cosine RoPE table (`[1, head_dim]` f32).
pub const DECODE_ROPE_COS_PORT: &str = "rope_cos";
/// Named input port for the runtime sine RoPE table (`[1, head_dim]` f32).
pub const DECODE_ROPE_SIN_PORT: &str = "rope_sin";
/// Named input port for the additive decode mask (`[1, bucket+1]` f32).
pub const DECODE_MASK_PORT: &str = "decode_mask";

/// Named input port carrying layer `l`'s past keys (`[kv, bucket, head_dim]`).
pub fn past_key_port(layer: usize) -> String {
    format!("past_k_{layer}")
}
/// Named input port carrying layer `l`'s past values (`[kv, bucket, head_dim]`).
pub fn past_value_port(layer: usize) -> String {
    format!("past_v_{layer}")
}
/// Named output port carrying layer `l`'s post-RoPE new key row (`[kv, head_dim]`).
pub fn new_key_port(layer: usize) -> String {
    format!("k_new_{layer}")
}
/// Named output port carrying layer `l`'s new value row (`[kv, head_dim]`).
pub fn new_value_port(layer: usize) -> String {
    format!("v_new_{layer}")
}

/// What [`rewrite_decode_attention`] did to the graph.
#[derive(Debug, Clone, Copy)]
pub struct DecodeRewrite {
    /// Number of fused attention nodes decomposed (= decoder layers).
    pub layers: usize,
    /// The fixed past bucket size the per-layer K/V ports were shaped to.
    pub bucket: u64,
    /// Head dim shared by every rewritten node (the rope table width).
    pub head_dim: u64,
    /// KV heads shared by every rewritten node (the K/V buffer row count).
    pub kv_heads: u64,
}

/// Fresh-id allocator + append-only node buffer over a graph under rewrite.
struct Emitter<'g> {
    graph: &'g mut AiGraph,
    next_tid: TensorId,
    next_nid: NodeId,
    nodes: Vec<AiNode>,
}

impl<'g> Emitter<'g> {
    fn new(graph: &'g mut AiGraph) -> Self {
        let next_tid = graph
            .tensor_info
            .keys()
            .chain(graph.params.keys())
            .copied()
            .max()
            .unwrap_or(0)
            + 1;
        let next_nid = graph.nodes.iter().map(|n| n.id).max().unwrap_or(0) + 1;
        Self {
            graph,
            next_tid,
            next_nid,
            nodes: Vec::new(),
        }
    }

    fn tensor(&mut self, name: &str, dtype: DType, dims: &[u64]) -> TensorId {
        let tid = self.next_tid;
        self.next_tid += 1;
        self.graph
            .tensor_info
            .insert(tid, TensorInfo::new(dtype, shape_from_concrete(dims)));
        self.graph.tensor_names.insert(tid, name.to_string());
        tid
    }

    fn node(&mut self, op: AiOp, inputs: Vec<TensorId>, outputs: Vec<TensorId>) {
        let nid = self.next_nid;
        self.next_nid += 1;
        self.nodes.push(AiNode::new(nid, op, inputs, outputs));
    }

    fn input(&mut self, name: &str, dtype: DType, dims: &[u64]) -> TensorId {
        let tid = self.tensor(name, dtype, dims);
        self.graph.inputs.push(tid);
        self.graph.input_names.push(name.to_string());
        tid
    }

    fn output(&mut self, tid: TensorId, name: &str) {
        self.graph.outputs.push(tid);
        self.graph.output_names.push(name.to_string());
    }

    fn const_f32(&mut self, name: &str, values: &[f32], dims: &[u64]) -> TensorId {
        let tid = self.tensor(name, DType::F32, dims);
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let info = TensorInfo::new(DType::F32, shape_from_concrete(dims));
        self.graph.params.insert(tid, AiParam::inline(bytes, info));
        tid
    }

    /// `x · cos + rotate_half(x) · sin` — rotate-half pairs `j ± head_dim/2`.
    fn rope(
        &mut self,
        x: TensorId,
        rows: u64,
        head_dim: u64,
        cos: TensorId,
        sin: TensorId,
        name: &str,
    ) -> TensorId {
        let half = head_dim / 2;
        let x_lo = self.tensor(&format!("{name}_lo"), DType::F32, &[rows, half]);
        self.node(
            AiOp::Slice {
                axes: vec![1],
                starts: vec![0],
                ends: vec![half as i64],
                steps: vec![1],
            },
            vec![x],
            vec![x_lo],
        );
        let x_hi = self.tensor(&format!("{name}_hi"), DType::F32, &[rows, half]);
        self.node(
            AiOp::Slice {
                axes: vec![1],
                starts: vec![half as i64],
                ends: vec![head_dim as i64],
                steps: vec![1],
            },
            vec![x],
            vec![x_hi],
        );
        let neg_hi = self.tensor(&format!("{name}_neg_hi"), DType::F32, &[rows, half]);
        self.node(AiOp::Neg, vec![x_hi], vec![neg_hi]);
        let rot = self.tensor(&format!("{name}_rot"), DType::F32, &[rows, head_dim]);
        self.node(AiOp::Concat { axis: 1 }, vec![neg_hi, x_lo], vec![rot]);
        let x_cos = self.tensor(&format!("{name}_cos"), DType::F32, &[rows, head_dim]);
        self.node(AiOp::Mul, vec![x, cos], vec![x_cos]);
        let rot_sin = self.tensor(&format!("{name}_sin"), DType::F32, &[rows, head_dim]);
        self.node(AiOp::Mul, vec![rot, sin], vec![rot_sin]);
        let out = self.tensor(name, DType::F32, &[rows, head_dim]);
        self.node(AiOp::Add, vec![x_cos, rot_sin], vec![out]);
        out
    }

    /// Contiguous axis-0 slice `[start, start+len)`.
    fn slice0(
        &mut self,
        src: TensorId,
        name: &str,
        start: u64,
        len: u64,
        tail: &[u64],
    ) -> TensorId {
        let mut dims = vec![len];
        dims.extend_from_slice(tail);
        let out = self.tensor(name, DType::F32, &dims);
        self.node(
            AiOp::Slice {
                axes: vec![0],
                starts: vec![start as i64],
                ends: vec![(start + len) as i64],
                steps: vec![1],
            },
            vec![src],
            vec![out],
        );
        out
    }
}

/// The concrete `[u64]` shape of a tensor, or an error naming the port.
fn concrete_dims(graph: &AiGraph, tid: TensorId, what: &str) -> Result<Vec<u64>> {
    let info = graph
        .tensor_info
        .get(&tid)
        .with_context(|| format!("{what}: tensor T{tid} has no tensor_info"))?;
    info.shape
        .iter()
        .map(|d| {
            d.as_concrete().with_context(|| {
                format!("{what}: dimension {d:?} is symbolic — the decode-step rewrite requires a fully concretized seq-1 graph")
            })
        })
        .collect()
}

/// Decompose every fused GQA node into the masked past-attention block.
///
/// The graph must already be built at `batch = 1, seq = 1` with concrete
/// shapes at each attention node's ports. New ports are appended: shared
/// `rope_cos`/`rope_sin`/`decode_mask` inputs, per-layer `past_k_l`/`past_v_l`
/// inputs and `k_new_l`/`v_new_l` outputs (layer index = attention-node order,
/// offset by `layer_base` — 0 for a monolithic graph, the stage's first
/// absolute layer for a stage graph).
pub fn rewrite_decode_attention(
    graph: &mut AiGraph,
    bucket: u64,
    layer_base: usize,
) -> Result<DecodeRewrite> {
    ensure!(bucket > 0, "decode bucket must be non-empty");

    let gqa_positions: Vec<usize> = graph
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| matches!(n.op, AiOp::GroupedQueryAttention { .. }))
        .map(|(i, _)| i)
        .collect();
    if gqa_positions.is_empty() {
        bail!("decode-step rewrite found no GroupedQueryAttention node — not a fused-attention decoder graph");
    }

    let mut em = Emitter::new(graph);
    let mut shared: Option<(TensorId, TensorId, TensorId, u64)> = None; // cos, sin, mask, head_dim
    let mut report: Option<DecodeRewrite> = None;
    // node vec position → replacement nodes (spliced in place, preserving order).
    let mut replacements: Vec<(usize, Vec<AiNode>)> = Vec::new();

    for (order, &pos) in gqa_positions.iter().enumerate() {
        let layer = layer_base + order;
        let node = em.graph.nodes[pos].clone();
        let AiOp::GroupedQueryAttention {
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            heads_first,
            qk_norm,
            rope,
            ..
        } = node.op
        else {
            unreachable!("filtered to GQA nodes");
        };
        ensure!(
            !qk_norm,
            "decode-step rewrite does not support qk_norm attention (layer {layer}) — \
             decomposing it without the norm would be silently wrong"
        );
        let (h, kv, dh) = (num_heads as u64, num_kv_heads as u64, head_dim as u64);
        ensure!(
            kv > 0 && h % kv == 0,
            "attention head grouping {h}/{kv} is not integral (layer {layer})"
        );
        let g = h / kv;

        let q_tid = *node.inputs.first().context("GQA node missing Q input")?;
        let k_tid = *node.inputs.get(1).context("GQA node missing K input")?;
        let v_tid = *node.inputs.get(2).context("GQA node missing V input")?;
        let out_tid = *node.outputs.first().context("GQA node missing output")?;

        // Both layouts ([b,1,h,dh] and [b,h,1,dh]) flatten identically at
        // batch=1, seq=1 — verify exactly that, loud on anything else.
        let q_dims = concrete_dims(em.graph, q_tid, "decode attention Q")?;
        let expected = if heads_first {
            vec![1, h, 1, dh]
        } else {
            vec![1, 1, h, dh]
        };
        ensure!(
            q_dims == expected,
            "decode-step rewrite requires a batch-1, seq-1 graph: layer {layer} Q is \
             {q_dims:?}, expected {expected:?}"
        );

        // Shared runtime ports, created at the first attention node.
        let (cos, sin, mask, shared_dh) = *shared.get_or_insert_with(|| {
            let cos = em.input(DECODE_ROPE_COS_PORT, DType::F32, &[1, dh]);
            let sin = em.input(DECODE_ROPE_SIN_PORT, DType::F32, &[1, dh]);
            let mask = em.input(DECODE_MASK_PORT, DType::F32, &[1, bucket + 1]);
            (cos, sin, mask, dh)
        });
        ensure!(
            shared_dh == dh,
            "attention head_dim varies across layers ({shared_dh} vs {dh} at layer {layer}) — \
             the shared rope tables cannot serve both"
        );

        // Per-layer carried K/V ports.
        let past_k = em.input(&past_key_port(layer), DType::F32, &[kv, bucket, dh]);
        let past_v = em.input(&past_value_port(layer), DType::F32, &[kv, bucket, dh]);

        em.nodes.clear();

        // Flatten the seq-1 4-D projections to their 2-D row forms.
        let q2 = em.tensor(&format!("dq_{layer}"), DType::F32, &[h, dh]);
        em.node(AiOp::Reshape { allow_zero: false }, vec![q_tid], vec![q2]);
        let k2 = em.tensor(&format!("dk_{layer}"), DType::F32, &[kv, dh]);
        em.node(AiOp::Reshape { allow_zero: false }, vec![k_tid], vec![k2]);
        let v2 = em.tensor(&format!("dv_{layer}"), DType::F32, &[kv, dh]);
        em.node(AiOp::Reshape { allow_zero: false }, vec![v_tid], vec![v2]);

        // RoPE as data: rotate q and the new k at the absolute position.
        let (q_r, k_r) = if rope {
            (
                em.rope(q2, h, dh, cos, sin, &format!("dq_rope_{layer}")),
                em.rope(k2, kv, dh, cos, sin, &format!("dk_rope_{layer}")),
            )
        } else {
            (q2, k2)
        };

        // Attention scale folded into q once.
        let scale_val = scale.unwrap_or(1.0 / (dh as f32).sqrt());
        let scale_c = em.const_f32(&format!("dscale_{layer}"), &[scale_val], &[1, 1]);
        let q_s = em.tensor(&format!("dq_scaled_{layer}"), DType::F32, &[h, dh]);
        em.node(AiOp::Mul, vec![q_r, scale_c], vec![q_s]);

        // Per kv-head group: strictly 2-D masked attention over the bucket.
        let mut group_outs = Vec::with_capacity(kv as usize);
        for j in 0..kv {
            let n = format!("l{layer}_kv{j}");
            let q_j = em.slice0(q_s, &format!("dqj_{n}"), j * g, g, &[dh]);

            let pk3 = em.slice0(past_k, &format!("dpk3_{n}"), j, 1, &[bucket, dh]);
            let pk = em.tensor(&format!("dpk_{n}"), DType::F32, &[bucket, dh]);
            em.node(AiOp::Reshape { allow_zero: false }, vec![pk3], vec![pk]);
            let pk_t = em.tensor(&format!("dpkt_{n}"), DType::F32, &[dh, bucket]);
            em.node(AiOp::Transpose { perm: vec![1, 0] }, vec![pk], vec![pk_t]);

            let kn = em.slice0(k_r, &format!("dkn_{n}"), j, 1, &[dh]);
            let kn_t = em.tensor(&format!("dknt_{n}"), DType::F32, &[dh, 1]);
            em.node(AiOp::Transpose { perm: vec![1, 0] }, vec![kn], vec![kn_t]);

            let s_past = em.tensor(&format!("dspast_{n}"), DType::F32, &[g, bucket]);
            em.node(AiOp::MatMul, vec![q_j, pk_t], vec![s_past]);
            let s_self = em.tensor(&format!("dsself_{n}"), DType::F32, &[g, 1]);
            em.node(AiOp::MatMul, vec![q_j, kn_t], vec![s_self]);
            let scores = em.tensor(&format!("dscores_{n}"), DType::F32, &[g, bucket + 1]);
            em.node(AiOp::Concat { axis: 1 }, vec![s_past, s_self], vec![scores]);

            let masked = em.tensor(&format!("dmasked_{n}"), DType::F32, &[g, bucket + 1]);
            em.node(AiOp::Add, vec![scores, mask], vec![masked]);
            let probs = em.tensor(&format!("dprobs_{n}"), DType::F32, &[g, bucket + 1]);
            em.node(AiOp::Softmax { axis: 1 }, vec![masked], vec![probs]);

            let pv3 = em.slice0(past_v, &format!("dpv3_{n}"), j, 1, &[bucket, dh]);
            let pv = em.tensor(&format!("dpv_{n}"), DType::F32, &[bucket, dh]);
            em.node(AiOp::Reshape { allow_zero: false }, vec![pv3], vec![pv]);
            let vn = em.slice0(v2, &format!("dvn_{n}"), j, 1, &[dh]);
            let vals = em.tensor(&format!("dvals_{n}"), DType::F32, &[bucket + 1, dh]);
            em.node(AiOp::Concat { axis: 0 }, vec![pv, vn], vec![vals]);

            let out_j = em.tensor(&format!("dout_{n}"), DType::F32, &[g, dh]);
            em.node(AiOp::MatMul, vec![probs, vals], vec![out_j]);
            group_outs.push(out_j);
        }

        // Chain-concat the kv groups back to [h, dh], then restore the fused
        // node's own output tensor (its 4-D shape is already declared).
        let mut acc = group_outs[0];
        for (j, &t) in group_outs.iter().enumerate().skip(1) {
            let rows = (j as u64 + 1) * g;
            let cat = em.tensor(&format!("dcat_l{layer}_{j}"), DType::F32, &[rows, dh]);
            em.node(AiOp::Concat { axis: 0 }, vec![acc, t], vec![cat]);
            acc = cat;
        }
        em.node(
            AiOp::Reshape { allow_zero: false },
            vec![acc],
            vec![out_tid],
        );

        // The step's derived K/V rows leave through named ports.
        em.output(k_r, &new_key_port(layer));
        em.output(v2, &new_value_port(layer));

        replacements.push((pos, std::mem::take(&mut em.nodes)));
        report = Some(DecodeRewrite {
            layers: order + 1,
            bucket,
            head_dim: dh,
            kv_heads: kv,
        });
    }

    // Splice each replacement block in at its fused node's position.
    let mut by_pos: std::collections::HashMap<usize, Vec<AiNode>> =
        replacements.into_iter().collect();
    let old_nodes = std::mem::take(&mut graph.nodes);
    let mut nodes = Vec::with_capacity(old_nodes.len() + by_pos.len() * 32);
    for (i, node) in old_nodes.into_iter().enumerate() {
        match by_pos.remove(&i) {
            Some(block) => nodes.extend(block),
            None => nodes.push(node),
        }
    }
    graph.nodes = nodes;
    graph.invalidate_topo_cache();

    Ok(report.expect("at least one GQA node was rewritten"))
}
