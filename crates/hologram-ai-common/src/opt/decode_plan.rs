//! Decode-plan attention rewrite (dictionary rows `decode-plan`,
//! `chunked-prefill`).
//!
//! Replaces each fused [`AiOp::GroupedQueryAttention`] node in a **batch-1,
//! seq-C** graph with the decomposed masked past-attention block over a
//! FIXED past bucket. One plan family, parametric in the chunk `C`:
//!
//! * `C = 1` is the generation step — one position per pass, the window
//!   gone from the step;
//! * `C > 1` is chunked prefill (prefill seeding) — C prompt positions per
//!   pass, amortizing the weight stream across the chunk. Causality within
//!   the chunk is no longer vacuous; it enters through the same additive
//!   mask that erases unrealized bucket rows.
//!
//! The block, per kv-head group (g = heads/kv):
//!
//! * carried K/V enter as per-layer graph **inputs** `past_k_l`/`past_v_l`
//!   (`[kv, bucket, head_dim]`) — derived content through named ports, not a
//!   mutable cache. The engine splices each pass's `k_new_l`/`v_new_l`
//!   **outputs** (`[kv, C, head_dim]`, head-major so a per-head splice is
//!   one contiguous row-range copy) into rows `pos..pos+C` between passes;
//!   no scatter op exists in the graph.
//! * the additive `decode_mask` input (`[g·C, bucket+C]`, `0` / `-1e9`)
//!   carries BOTH visibility laws: bucket rows past the realized length are
//!   erased, and chunk position `i` sees only new columns `≤ i` (row
//!   `jj·C + i` is position `i`'s row for every head `jj` of the group).
//! * RoPE is plain arithmetic on **runtime** tables the engine synthesizes
//!   at the absolute positions, pre-expanded to the head-major row layout
//!   (`rope_cos_q`/`rope_sin_q` `[heads·C, head_dim]`,
//!   `rope_cos_k`/`rope_sin_k` `[kv·C, head_dim]`) — exact-shape `Mul`,
//!   zero broadcast assumptions. Rotate-half pairs `j ± head_dim/2`.
//! * attention decomposes into strictly 2-D matmuls
//!   (`q_group · Concat(past_k, k_new)ᵀ`, softmax, `probs · values`) — the
//!   substrate's MatMul kernel is 2-D, so nothing batched is assumed.
//!
//! `qk_norm` attention is rejected loud — the parametric safetensors recipe
//! never emits it, and silently dropping the norm would be a wrong number.

use anyhow::{bail, ensure, Context, Result};

use crate::ir::{
    shape_from_concrete, AiGraph, AiNode, AiOp, AiParam, DType, NodeId, TensorId, TensorInfo,
};

/// Runtime cosine RoPE table for Q, head-major (`[heads·C, head_dim]` f32).
pub const DECODE_ROPE_COS_Q_PORT: &str = "rope_cos_q";
/// Runtime sine RoPE table for Q, head-major.
pub const DECODE_ROPE_SIN_Q_PORT: &str = "rope_sin_q";
/// Runtime cosine RoPE table for K, head-major (`[kv·C, head_dim]` f32).
pub const DECODE_ROPE_COS_K_PORT: &str = "rope_cos_k";
/// Runtime sine RoPE table for K, head-major.
pub const DECODE_ROPE_SIN_K_PORT: &str = "rope_sin_k";
/// Additive decode mask (`[g·C, bucket+C]` f32).
pub const DECODE_MASK_PORT: &str = "decode_mask";
/// Runtime ring-write position (`cur_len`) for the v0.9.0 resident-KV form: a
/// single INT32 operand shared by every layer's `KvCacheWrite`. Content, not
/// structure — one compiled step-graph serves every step (ADR-0019).
pub const DECODE_POS_PORT: &str = "decode_pos";

/// Named input port carrying layer `l`'s past keys (`[kv, bucket, head_dim]`).
pub fn past_key_port(layer: usize) -> String {
    format!("past_k_{layer}")
}
/// Named input port carrying layer `l`'s past values (`[kv, bucket, head_dim]`).
pub fn past_value_port(layer: usize) -> String {
    format!("past_v_{layer}")
}
/// Named output port carrying layer `l`'s post-RoPE new key rows
/// (`[kv, C, head_dim]`, head-major).
pub fn new_key_port(layer: usize) -> String {
    format!("k_new_{layer}")
}
/// Named output port carrying layer `l`'s new value rows (`[kv, C, head_dim]`).
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
    /// Positions processed per pass (1 = generation step).
    pub chunk: u64,
    /// Head dim shared by every rewritten node (the rope table width).
    pub head_dim: u64,
    /// KV heads shared by every rewritten node (the K/V buffer row count).
    pub kv_heads: u64,
    /// Query heads shared by every rewritten node.
    pub heads: u64,
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

    /// `x · cos + rotate_half(x) · sin` on head-major flat rows
    /// (`[n, head_dim]` against exact-shape tables).
    fn rope(
        &mut self,
        x: TensorId,
        n: u64,
        head_dim: u64,
        cos: TensorId,
        sin: TensorId,
        name: &str,
    ) -> TensorId {
        let half = head_dim / 2;
        let x_lo = self.tensor(&format!("{name}_lo"), DType::F32, &[n, half]);
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
        let x_hi = self.tensor(&format!("{name}_hi"), DType::F32, &[n, half]);
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
        let neg_hi = self.tensor(&format!("{name}_neg_hi"), DType::F32, &[n, half]);
        self.node(AiOp::Neg, vec![x_hi], vec![neg_hi]);
        let rot = self.tensor(&format!("{name}_rot"), DType::F32, &[n, head_dim]);
        self.node(AiOp::Concat { axis: 1 }, vec![neg_hi, x_lo], vec![rot]);
        let x_cos = self.tensor(&format!("{name}_cos"), DType::F32, &[n, head_dim]);
        self.node(AiOp::Mul, vec![x, cos], vec![x_cos]);
        let rot_sin = self.tensor(&format!("{name}_sin"), DType::F32, &[n, head_dim]);
        self.node(AiOp::Mul, vec![rot, sin], vec![rot_sin]);
        let out = self.tensor(name, DType::F32, &[n, head_dim]);
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

    /// A seq-C 4-D projection (`[1,C,rows,dh]` or heads-first `[1,rows,C,dh]`)
    /// to head-major flat `[rows·C, dh]` — plus the `[rows, C, dh]` 3-D form.
    fn emit_head_major(
        &mut self,
        x4: TensorId,
        rows: u64,
        chunk: u64,
        dh: u64,
        heads_first: bool,
        name: &str,
    ) -> (TensorId, TensorId) {
        let hf3 = self.tensor(&format!("{name}_hf"), DType::F32, &[rows, chunk, dh]);
        if heads_first {
            // Already [1, rows, C, dh]: drop the batch dim.
            self.node(AiOp::Reshape { allow_zero: false }, vec![x4], vec![hf3]);
        } else {
            let pm3 = self.tensor(&format!("{name}_pm"), DType::F32, &[chunk, rows, dh]);
            self.node(AiOp::Reshape { allow_zero: false }, vec![x4], vec![pm3]);
            self.node(
                AiOp::Transpose {
                    perm: vec![1, 0, 2],
                },
                vec![pm3],
                vec![hf3],
            );
        }
        let flat = self.tensor(&format!("{name}_flat"), DType::F32, &[rows * chunk, dh]);
        self.node(AiOp::Reshape { allow_zero: false }, vec![hf3], vec![flat]);
        (hf3, flat)
    }

    /// One kv-group's masked scaled-dot-product attention over the fixed bucket
    /// plus this chunk's own keys: `softmax(q_jᵀ·[past_k ∥ k_new] + mask) ·
    /// [past_v ∥ v_new] → [g·chunk, dh]`, strictly 2-D.
    ///
    /// This whole per-group `Concat`/`Transpose`/`MatMul`/`Softmax`/`MatMul`
    /// chain is the **seam** the v0.9.0 fused `DecodeAttention` (κ119) collapses
    /// into a single pooled, split-KV kernel (ADR-0019). Keeping it in one named
    /// method makes that swap a localized change and keeps the layer emitter
    /// readable. `span = bucket + chunk` is the `[past ∥ new]` key count.
    #[allow(clippy::too_many_arguments)]
    fn emit_kv_group_attention(
        &mut self,
        q_s: TensorId,
        past_k: TensorId,
        past_v: TensorId,
        k_r: TensorId,
        v_flat: TensorId,
        mask: TensorId,
        j: u64,
        g: u64,
        chunk: u64,
        bucket: u64,
        dh: u64,
        span: u64,
        layer: usize,
    ) -> TensorId {
        let n = format!("l{layer}_kv{j}");
        // Head-major rows: group j's heads occupy rows j·g·C..(j+1)·g·C.
        let q_j = self.slice0(q_s, &format!("dqj_{n}"), j * g * chunk, g * chunk, &[dh]);

        let pk3 = self.slice0(past_k, &format!("dpk3_{n}"), j, 1, &[bucket, dh]);
        let pk = self.tensor(&format!("dpk_{n}"), DType::F32, &[bucket, dh]);
        self.node(AiOp::Reshape { allow_zero: false }, vec![pk3], vec![pk]);
        let kn = self.slice0(k_r, &format!("dkn_{n}"), j * chunk, chunk, &[dh]);
        let keys = self.tensor(&format!("dkeys_{n}"), DType::F32, &[span, dh]);
        self.node(AiOp::Concat { axis: 0 }, vec![pk, kn], vec![keys]);
        let keys_t = self.tensor(&format!("dkeyst_{n}"), DType::F32, &[dh, span]);
        self.node(
            AiOp::Transpose { perm: vec![1, 0] },
            vec![keys],
            vec![keys_t],
        );

        let scores = self.tensor(&format!("dscores_{n}"), DType::F32, &[g * chunk, span]);
        self.node(AiOp::MatMul, vec![q_j, keys_t], vec![scores]);
        let masked = self.tensor(&format!("dmasked_{n}"), DType::F32, &[g * chunk, span]);
        self.node(AiOp::Add, vec![scores, mask], vec![masked]);
        let probs = self.tensor(&format!("dprobs_{n}"), DType::F32, &[g * chunk, span]);
        self.node(AiOp::Softmax { axis: 1 }, vec![masked], vec![probs]);

        let pv3 = self.slice0(past_v, &format!("dpv3_{n}"), j, 1, &[bucket, dh]);
        let pv = self.tensor(&format!("dpv_{n}"), DType::F32, &[bucket, dh]);
        self.node(AiOp::Reshape { allow_zero: false }, vec![pv3], vec![pv]);
        let vn = self.slice0(v_flat, &format!("dvn_{n}"), j * chunk, chunk, &[dh]);
        let vals = self.tensor(&format!("dvals_{n}"), DType::F32, &[span, dh]);
        self.node(AiOp::Concat { axis: 0 }, vec![pv, vn], vec![vals]);

        let out_j = self.tensor(&format!("dout_{n}"), DType::F32, &[g * chunk, dh]);
        self.node(AiOp::MatMul, vec![probs, vals], vec![out_j]);
        out_j
    }

    /// Restore a head-major attention result (`src`, either flat `[h·C, dh]` or
    /// `[1, h, C, dh]` — same element count) to the fused node's own output
    /// tensor `out_tid`: a bare reshape when the model is heads-first, else a
    /// `[chunk, h, dh]` transpose back to the interleaved layout.
    #[allow(clippy::too_many_arguments)]
    fn restore_head_output(
        &mut self,
        src: TensorId,
        out_tid: TensorId,
        h: u64,
        chunk: u64,
        dh: u64,
        heads_first: bool,
        layer: usize,
    ) {
        if heads_first {
            self.node(
                AiOp::Reshape { allow_zero: false },
                vec![src],
                vec![out_tid],
            );
        } else {
            let hf3 = self.tensor(&format!("dout_hf_{layer}"), DType::F32, &[h, chunk, dh]);
            self.node(AiOp::Reshape { allow_zero: false }, vec![src], vec![hf3]);
            let pm3 = self.tensor(&format!("dout_pm_{layer}"), DType::F32, &[chunk, h, dh]);
            self.node(
                AiOp::Transpose {
                    perm: vec![1, 0, 2],
                },
                vec![hf3],
                vec![pm3],
            );
            self.node(
                AiOp::Reshape { allow_zero: false },
                vec![pm3],
                vec![out_tid],
            );
        }
    }

    /// Legacy per-group SDPA decomposition + head-major carried-K/V outputs
    /// (the NEW rows only; the driver splices them into the host cache). This is
    /// the substrate-version-agnostic path, unchanged in behavior.
    #[allow(clippy::too_many_arguments)]
    fn emit_legacy_attention(
        &mut self,
        q_r: TensorId,
        k_r: TensorId,
        v_flat: TensorId,
        v_hf3: TensorId,
        past_k: TensorId,
        past_v: TensorId,
        mask: TensorId,
        out_tid: TensorId,
        scale: Option<f32>,
        h: u64,
        kv: u64,
        g: u64,
        chunk: u64,
        bucket: u64,
        dh: u64,
        heads_first: bool,
        layer: usize,
    ) {
        // Attention scale folded into q once.
        let scale_val = scale.unwrap_or(1.0 / (dh as f32).sqrt());
        let scale_c = self.const_f32(&format!("dscale_{layer}"), &[scale_val], &[1, 1]);
        let q_s = self.tensor(&format!("dq_scaled_{layer}"), DType::F32, &[h * chunk, dh]);
        self.node(AiOp::Mul, vec![q_r, scale_c], vec![q_s]);

        let span = bucket + chunk;
        let mut group_outs = Vec::with_capacity(kv as usize);
        for j in 0..kv {
            group_outs.push(self.emit_kv_group_attention(
                q_s, past_k, past_v, k_r, v_flat, mask, j, g, chunk, bucket, dh, span, layer,
            ));
        }
        // Chain-concat the kv groups (head-major) → [h·C, dh].
        let mut acc = group_outs[0];
        for (j, &t) in group_outs.iter().enumerate().skip(1) {
            let rows = (j as u64 + 1) * g * chunk;
            let cat = self.tensor(&format!("dcat_l{layer}_{j}"), DType::F32, &[rows, dh]);
            self.node(AiOp::Concat { axis: 0 }, vec![acc, t], vec![cat]);
            acc = cat;
        }
        self.restore_head_output(acc, out_tid, h, chunk, dh, heads_first, layer);

        // Carried K/V rows leave head-major ([kv, C, dh]) so the engine's
        // per-head splice is contiguous.
        let k_out = self.tensor(&format!("dknew_{layer}"), DType::F32, &[kv, chunk, dh]);
        self.node(AiOp::Reshape { allow_zero: false }, vec![k_r], vec![k_out]);
        self.output(k_out, &new_key_port(layer));
        self.output(v_hf3, &new_value_port(layer));
    }

    /// The v0.9.0 fused **split-KV** decode attention (`DecodeAttention`, κ119)
    /// plus two `KvCacheWrite` ring updates (κ120). Fully parametric: every dim
    /// derives from the node, and the model's scale is honored for ANY head_dim.
    /// The kernel applies `1/√dh` internally (`scale_bits` 0 — the default and
    /// the common case, so q is untouched); a model-declared non-default scale
    /// `s` is folded into q as `f = s·√dh`, giving effective scale exactly `s`.
    /// The carried-K/V ports now carry the WHOLE updated cache (the driver binds
    /// it by κ-label next step, no host splice).
    #[allow(clippy::too_many_arguments)]
    fn emit_fused_attention(
        &mut self,
        q_r: TensorId,
        k_r: TensorId,
        v_flat: TensorId,
        past_k: TensorId,
        past_v: TensorId,
        mask: TensorId,
        pos: TensorId,
        out_tid: TensorId,
        scale: Option<f32>,
        h: u64,
        kv: u64,
        chunk: u64,
        bucket: u64,
        dh: u64,
        heads_first: bool,
        layer: usize,
    ) {
        let q_pre = match scale {
            None => q_r,
            Some(s) => {
                let f = s * (dh as f32).sqrt();
                let fc = self.const_f32(&format!("dscale_{layer}"), &[f], &[1, 1]);
                let qs = self.tensor(&format!("dq_scaled_{layer}"), DType::F32, &[h * chunk, dh]);
                self.node(AiOp::Mul, vec![q_r, fc], vec![qs]);
                qs
            }
        };
        // Head-major flat → rank-4 [1, heads, chunk, dh] the kernel expects.
        let q4 = self.tensor(&format!("dq4_{layer}"), DType::F32, &[1, h, chunk, dh]);
        self.node(AiOp::Reshape { allow_zero: false }, vec![q_pre], vec![q4]);
        let kn4 = self.tensor(&format!("dkn4_{layer}"), DType::F32, &[1, kv, chunk, dh]);
        self.node(AiOp::Reshape { allow_zero: false }, vec![k_r], vec![kn4]);
        let vn4 = self.tensor(&format!("dvn4_{layer}"), DType::F32, &[1, kv, chunk, dh]);
        self.node(AiOp::Reshape { allow_zero: false }, vec![v_flat], vec![vn4]);

        // One pooled split-KV masked attention over [past ∥ chunk].
        let attn4 = self.tensor(&format!("dattn_{layer}"), DType::F32, &[1, h, chunk, dh]);
        self.node(
            AiOp::DecodeAttention,
            vec![q4, past_k, past_v, kn4, vn4, mask],
            vec![attn4],
        );
        self.restore_head_output(attn4, out_tid, h, chunk, dh, heads_first, layer);

        // Ring-write the new rows; the UPDATED full caches leave through the
        // carried-K/V ports (bound by label next step — no host splice).
        let k_upd = self.tensor(&format!("dkupd_{layer}"), DType::F32, &[1, kv, bucket, dh]);
        self.node(AiOp::KvCacheWrite, vec![past_k, kn4, pos], vec![k_upd]);
        self.output(k_upd, &new_key_port(layer));
        let v_upd = self.tensor(&format!("dvupd_{layer}"), DType::F32, &[1, kv, bucket, dh]);
        self.node(AiOp::KvCacheWrite, vec![past_v, vn4, pos], vec![v_upd]);
        self.output(v_upd, &new_value_port(layer));
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
                format!("{what}: dimension {d:?} is symbolic — the decode rewrite requires a fully concretized seq-C graph")
            })
        })
        .collect()
}

/// Decompose every fused GQA node into the masked past-attention block.
///
/// The graph must already be built at `batch = 1, seq = chunk` with concrete
/// shapes at each attention node's ports. New ports are appended: shared
/// rope-table and `decode_mask` inputs, per-layer `past_k_l`/`past_v_l`
/// inputs and `k_new_l`/`v_new_l` outputs (layer index = attention-node
/// order, offset by `layer_base` — 0 for a monolithic graph, the stage's
/// first absolute layer for a stage graph).
pub fn rewrite_decode_attention(
    graph: &mut AiGraph,
    bucket: u64,
    chunk: u64,
    layer_base: usize,
    resident_kv: bool,
) -> Result<DecodeRewrite> {
    ensure!(bucket > 0, "decode bucket must be non-empty");
    ensure!(chunk > 0, "decode chunk must be non-empty");

    let gqa_positions: Vec<usize> = graph
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| matches!(n.op, AiOp::GroupedQueryAttention { .. }))
        .map(|(i, _)| i)
        .collect();
    if gqa_positions.is_empty() {
        bail!("decode rewrite found no GroupedQueryAttention node — not a fused-attention decoder graph");
    }

    let mut em = Emitter::new(graph);
    // (cos_q, sin_q, cos_k, sin_k, mask, pos, head_dim) — `pos` is 0/unused in
    // the legacy path, the runtime ring-write position in the resident-KV path.
    let mut shared: Option<(
        TensorId,
        TensorId,
        TensorId,
        TensorId,
        TensorId,
        TensorId,
        u64,
    )> = None;
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
            "decode rewrite does not support qk_norm attention (layer {layer}) — \
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

        let q_dims = concrete_dims(em.graph, q_tid, "decode attention Q")?;
        let expected = if heads_first {
            vec![1, h, chunk, dh]
        } else {
            vec![1, chunk, h, dh]
        };
        ensure!(
            q_dims == expected,
            "decode rewrite requires a batch-1, seq-{chunk} graph: layer {layer} Q is \
             {q_dims:?}, expected {expected:?}"
        );

        // Shared runtime ports, created at the first attention node. The mask is
        // the sole visibility authority; its rows are the query positions. The
        // legacy per-group decomposition replicates them per group (`g·chunk`
        // rows), while the fused kernel groups internally and applies one
        // per-position mask (`chunk` rows) — and adds a runtime `pos` operand for
        // the ring write. `pos` is unused (id 0) in the legacy path.
        let (cos_q, sin_q, cos_k, sin_k, mask, pos_port, shared_dh) =
            *shared.get_or_insert_with(|| {
                let cos_q = em.input(DECODE_ROPE_COS_Q_PORT, DType::F32, &[h * chunk, dh]);
                let sin_q = em.input(DECODE_ROPE_SIN_Q_PORT, DType::F32, &[h * chunk, dh]);
                let cos_k = em.input(DECODE_ROPE_COS_K_PORT, DType::F32, &[kv * chunk, dh]);
                let sin_k = em.input(DECODE_ROPE_SIN_K_PORT, DType::F32, &[kv * chunk, dh]);
                if resident_kv {
                    let mask = em.input(DECODE_MASK_PORT, DType::F32, &[chunk, bucket + chunk]);
                    let pos = em.input(DECODE_POS_PORT, DType::INT32, &[1]);
                    (cos_q, sin_q, cos_k, sin_k, mask, pos, dh)
                } else {
                    let mask = em.input(DECODE_MASK_PORT, DType::F32, &[g * chunk, bucket + chunk]);
                    (cos_q, sin_q, cos_k, sin_k, mask, 0, dh)
                }
            });
        ensure!(
            shared_dh == dh,
            "attention head_dim varies across layers ({shared_dh} vs {dh} at layer {layer}) — \
             the shared rope tables cannot serve both"
        );

        // Per-layer carried K/V ports. The fused kernel takes rank-4
        // `[1, kv, bucket, dh]`; the legacy decomposition slices a rank-3
        // `[kv, bucket, dh]` per group.
        let (past_k, past_v) = if resident_kv {
            (
                em.input(&past_key_port(layer), DType::F32, &[1, kv, bucket, dh]),
                em.input(&past_value_port(layer), DType::F32, &[1, kv, bucket, dh]),
            )
        } else {
            (
                em.input(&past_key_port(layer), DType::F32, &[kv, bucket, dh]),
                em.input(&past_value_port(layer), DType::F32, &[kv, bucket, dh]),
            )
        };

        em.nodes.clear();

        // Head-major forms: q/k flat for rope + group slicing, v both forms
        // (3-D for the output port, flat for per-group slicing).
        let (_, q_flat) =
            em.emit_head_major(q_tid, h, chunk, dh, heads_first, &format!("dq_{layer}"));
        let (_, k_flat) =
            em.emit_head_major(k_tid, kv, chunk, dh, heads_first, &format!("dk_{layer}"));
        let (v_hf3, v_flat) =
            em.emit_head_major(v_tid, kv, chunk, dh, heads_first, &format!("dv_{layer}"));

        // RoPE as data at the absolute positions (tables pre-expanded to the
        // head-major layout by the engine).
        let (q_r, k_r) = if rope {
            (
                em.rope(
                    q_flat,
                    h * chunk,
                    dh,
                    cos_q,
                    sin_q,
                    &format!("dq_rope_{layer}"),
                ),
                em.rope(
                    k_flat,
                    kv * chunk,
                    dh,
                    cos_k,
                    sin_k,
                    &format!("dk_rope_{layer}"),
                ),
            )
        } else {
            (q_flat, k_flat)
        };

        // Emit the attention core + carried-K/V outputs. Two forms, one seam:
        // the legacy per-group SDPA decomposition, or the v0.9.0 fused
        // split-KV DecodeAttention + KvCacheWrite (ADR-0019). Both consume the
        // same roped q/k and head-major v; they differ only in how the masked
        // attention and the cache update are expressed.
        if resident_kv {
            em.emit_fused_attention(
                q_r,
                k_r,
                v_flat,
                past_k,
                past_v,
                mask,
                pos_port,
                out_tid,
                scale,
                h,
                kv,
                chunk,
                bucket,
                dh,
                heads_first,
                layer,
            );
        } else {
            em.emit_legacy_attention(
                q_r,
                k_r,
                v_flat,
                v_hf3,
                past_k,
                past_v,
                mask,
                out_tid,
                scale,
                h,
                kv,
                g,
                chunk,
                bucket,
                dh,
                heads_first,
                layer,
            );
        }

        replacements.push((pos, std::mem::take(&mut em.nodes)));
        report = Some(DecodeRewrite {
            layers: order + 1,
            bucket,
            chunk,
            head_dim: dh,
            kv_heads: kv,
            heads: h,
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
