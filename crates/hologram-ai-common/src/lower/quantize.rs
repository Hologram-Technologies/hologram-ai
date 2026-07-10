//! Compile-time weight quantization pass: rewrites MatMul f32-weight constants
//! into i8 + per-channel scale + a `Dequantize` node. Gated on `QuantStrategy`.

use std::sync::Arc;

use anyhow::{bail, Result};
use hologram_ai_quant::encode_int8_per_channel;

use crate::ir::{
    shape_from_concrete, ActQuant, AiGraph, AiNode, AiOp, AiParam, DType, TensorId, TensorInfo,
    WeightLayout,
};
use crate::lower::QuantStrategy;

/// Read a tensor's concrete 2D dims `[k, n]`, or `None` if not rank-2 concrete.
fn concrete_2d(info: &TensorInfo) -> Option<(usize, usize)> {
    let k = info.shape.first()?.as_concrete()?;
    let n = info.shape.get(1)?.as_concrete()?;
    if info.shape.get(2).is_some() {
        return None; // higher rank — skip (batched matmul weight)
    }
    Some((k as usize, n as usize))
}

/// Rewrite each MatMul whose B (weight) operand is an inline f32 rank-2 constant
/// into `Dequantize(i8_weight, per_column_scale) → MatMul`. No-op unless
/// `strategy == Int8`. `Int4` is rejected until the int4 spec lands.
///
/// The emitted `Dequantize` node uses `axis: 1` (per output column for a
/// `[k, n]` weight) and carries only two operands — weight and scale; no
/// explicit zero-point operand is included (the lowering pass treats a missing
/// zero-point as all-zeros, matching the reference test in
/// `hologram-ai/tests/quantized_weight_memory.rs`).
pub fn quantize_weights(graph: &mut AiGraph, strategy: QuantStrategy) -> Result<()> {
    match strategy {
        QuantStrategy::Int8 => {}
        QuantStrategy::Int4 => bail!("int4 quantization is not yet implemented"),
        _ => return Ok(()),
    }

    // tensor_info and params share the TensorId namespace; max() over both
    // deduplicates. Three ids are allocated per rewritten MatMul: i8 weight,
    // per-channel scale, and the Dequantize output tensor.
    let mut next_tid: TensorId = graph
        .tensor_info
        .keys()
        .chain(graph.params.keys())
        .copied()
        .max()
        .unwrap_or(0)
        + 1;
    let mut next_nid = graph.nodes.iter().map(|n| n.id).max().unwrap_or(0) + 1;

    let mut new_nodes: Vec<AiNode> = Vec::new();
    let mut dead_params: Vec<TensorId> = Vec::new();
    let mut changed = false;
    let mut rewritten: usize = 0;

    for idx in 0..graph.nodes.len() {
        if !matches!(graph.nodes[idx].op, AiOp::MatMul) {
            continue;
        }
        let b_tid = match graph.nodes[idx].inputs.get(1).copied() {
            Some(t) => t,
            None => continue,
        };
        let (data, info) = match graph.params.get(&b_tid) {
            Some(AiParam::Inline { data, info }) => (data.clone(), info.clone()),
            _ => continue, // mmap or non-constant B: skip in this baseline
        };
        if info.logical_dtype != DType::F32 {
            continue;
        }
        let (k, n) = match concrete_2d(&info) {
            Some(kn) => kn,
            None => continue,
        };
        if data.len() != k * n * 4 {
            continue;
        }

        let wf: Vec<f32> = data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let (q, scales) = encode_int8_per_channel(&wf, k, n);

        let wq_tid = next_tid;
        let scale_tid = next_tid + 1;
        let deq_tid = next_tid + 2;
        next_tid += 3;

        // i8 weight constant [k, n].
        // Reinterpret the signed quantized values as raw bytes for storage in
        // AiParam::Inline; the lowering reads them back as i8 via INT8 dtype.
        let q_bytes: Vec<u8> = bytemuck::cast_slice::<i8, u8>(&q).to_vec();
        let wq_info = TensorInfo::new(DType::INT8, info.shape.clone());
        graph.params.insert(
            wq_tid,
            AiParam::Inline {
                data: Arc::new(q_bytes),
                info: wq_info.clone(),
            },
        );
        graph.tensor_info.insert(wq_tid, wq_info);

        // Per-column scale constant [n] f32.
        let scale_bytes: Vec<u8> = scales.iter().flat_map(|v| v.to_le_bytes()).collect();
        let scale_info = TensorInfo::new(DType::F32, shape_from_concrete(&[n as u64]));
        graph.params.insert(
            scale_tid,
            AiParam::Inline {
                data: Arc::new(scale_bytes),
                info: scale_info.clone(),
            },
        );
        graph.tensor_info.insert(scale_tid, scale_info);

        // Dequant output tensor [k, n] f32.
        graph
            .tensor_info
            .insert(deq_tid, TensorInfo::new(DType::F32, info.shape.clone()));

        // Dequantize node (zero-point omitted — lowering fills zeros), axis=1
        // (per output column for a [k, n] weight). This matches the per-channel
        // reference graph in `quantized_weight_memory.rs::per_channel_graph()`.
        //
        // This weight is a graph CONSTANT: its bytes are in the graph, `[k,n]`,
        // encoded by `encode_int8_per_channel` just above. `OutputMajor` would be
        // a false statement about them and the substrate rejects it. And it stays
        // `W8A32`, deliberately: the compile-time fusion only transposes a
        // constant at a decode shape (`m ≤ OMAJOR_W8A8_MAX_M`), so declaring W8A8
        // here would give this weight W8A8 at decode and W8A32 at prefill — the
        // same weight computing two different functions. Our chunked-prefill
        // seeder runs at `m = 64`; that split would break every equivalence in
        // `decode_plan.feature`. The κ-bound path below has no such gate.
        new_nodes.push(AiNode::new(
            next_nid,
            AiOp::Dequantize {
                axis: 1,
                layout: WeightLayout::RowMajor,
                act: ActQuant::W8A32,
            },
            vec![wq_tid, scale_tid],
            vec![deq_tid],
        ));
        next_nid += 1;

        // Rewire MatMul B → dequant output; retire the f32 weight constant.
        graph.nodes[idx].inputs[1] = deq_tid;
        dead_params.push(b_tid);
        changed = true;
        rewritten += 1;
    }

    graph.nodes.extend(new_nodes);
    for t in dead_params {
        graph.params.remove(&t);
        graph.tensor_info.remove(&t);
    }
    if changed {
        tracing::debug!(
            matmuls_rewritten = rewritten,
            "quantize_weights: int8 per-channel"
        );
        graph.invalidate_topo_cache();
    } else {
        // Int8 was requested but nothing matched (e.g. weights are mmap'd, not
        // inline f32 — which this baseline skips). Warn so the flag isn't a
        // silent no-op.
        tracing::warn!(
            "quantize_weights: --quantize int8 requested but no inline f32 MatMul \
             weights were found to quantize (mmap weights are skipped in this baseline)"
        );
    }
    Ok(())
}

/// **The one law.** Can a load-time-bound quantized weight of this tier and
/// shape take the fused output-major W8A8 decode GEMV?
///
/// This predicate decides two things that must never disagree: the byte layout
/// `hologram_ai::quantized::derive_quantized_artifact` authors, and the
/// `weight_layout`/`act_quant` the
/// binder declares on the `Dequantize` node. If they disagreed, the substrate
/// would read `[n,k]` bytes as `[k,n]` — a plausible, wrong answer. So both call
/// *this*, and neither restates its conditions.
///
/// Every condition is the substrate's own derived fact, never a literal of ours:
///
/// * `quant_tier(dtype)` — the codec registry. An unregistered tag is `None`,
///   never a guess. The tier states whether an output-major kernel exists for it
///   (`omajor_fusable`) and whether this `k` is addressable per output column
///   (`omajor_k_ok`: whole groups, and each column's unit span a whole number of
///   bytes — which is why `i4` needs an even `k` and `i8`/`e8cb` do not).
/// * `mm_act_quant::K_MAX` — the exact-accumulation ceiling, itself derived as
///   `⌊ACCUM_CAPACITY / ALPHABET_BOUND²⌋` from the declared alphabet. Under it
///   neither the final sum nor any intermediate accumulator state can overflow,
///   so the reduction is exact and the integer invariances are bit-level. Real
///   decode shapes sit three orders of magnitude below it; a 1M-context model's
///   projections do not come close.
/// * A VQ tier must bring a codebook operand. We author no codebook, so a tier
///   that needs one cannot be declared here.
///
/// `n` (the output-channel count) carries no condition of its own — the fused
/// call requires `channels == n`, which our `axis = 1` per-channel scales give by
/// construction — but it is taken so a future tier whose packing depends on the
/// column count cannot be added without revisiting this signature.
#[must_use]
pub fn omajor_w8a8_servable(quant_dtype: u8, k: usize, _n: usize) -> bool {
    use hologram_backend::kernel_call::mm_act_quant;
    if !SUBSTRATE_ACCEPTS_OUTPUT_MAJOR_ON_WEIGHTLESS_CONSTANTS {
        return false;
    }
    let Some(tier) = hologram_backend::quant_tier::quant_tier(hologram_types::DTypeId(quant_dtype))
    else {
        return false; // unregistered tier: never guess
    };
    tier.omajor_fusable && !tier.needs_codebook && tier.omajor_k_ok(k) && k <= mm_act_quant::K_MAX
}

/// Whether the substrate's compiler accepts `weight_layout = OUTPUT_MAJOR` on the
/// binding form our decode path actually uses: a **weightless constant** —
/// `ConstantEntry { bytes: vec![] }` plus a `holospaces.kappa_map` extension
/// naming the κ whose bytes arrive at materialization.
///
/// This is a fact about the *host substrate*, not about any model, input, or
/// use-case — the only kind of constant this codebase permits. It is witnessed,
/// not asserted: `a_weightless_kappa_constant_can_declare_output_major`
/// (`hologram-ai/tests/omajor_w8a8_substrate_contract.rs`) compiles that exact
/// graph and proves it.
///
/// **Was blocked through v0.8.0; fixed in v0.8.1 (rev `0120c94`).** The v0.8.0
/// validator rejected on `matches!(node.inputs.first(), Some(Constant(_)))` — any
/// graph constant, never asking whether it had bytes — which locked out precisely
/// the case `QuantAttrs::weight_layout`'s own docstring said the field exists to
/// serve. v0.8.1 narrowed it to `!e.bytes.is_empty()`, the same question
/// `fuse_const_i8_decode` asks before it transposes anything: a constant *with*
/// bytes carries them `[k,n]` and may not claim otherwise; a zero-byte constant is
/// a κ naming content that arrives at materialization, and may declare
/// OUTPUT_MAJOR. (We asked for this fix; the substrate commit cites the same
/// reasoning.)
///
/// **Still `false`, for exactly one commit.** The substrate accepts the
/// declaration now, but our pipeline turning it on is a *numerics* change: W8A8
/// per-token activation quantization re-keys every affected κ and re-bases the
/// reference transcript. That belongs in its own commit, with the byte-exact
/// oracle deliberately replaced by accuracy agreement — not folded into the
/// substrate bump, which must stay byte-neutral. The very next commit flips this
/// to `true` (or removes it) alongside the re-baseline.
///
/// Because this predicate gates the artifact's byte order *and* the declaration
/// together, `false` here means `derive_quantized_artifact` keeps authoring
/// row-major `[in, out]` bytes. Bytes and declaration cannot drift apart while
/// the flag is off, and cannot drift when it is turned on.
pub const SUBSTRATE_ACCEPTS_OUTPUT_MAJOR_ON_WEIGHTLESS_CONSTANTS: bool = false;

/// The quantized derived-artifact map (row `quantized-transit`): a
/// [`quant_key`] → `(artifact κ, out_features, in_features)` where the
/// artifact is the matmul-ready per-channel symmetric int8 form of the wide
/// `[out, in]` weight, layout `q_i8(in·out) ‖ scales_f32(4·out)` (one scale per
/// output channel).
///
/// The `q` block is **output-major `[out, in]`** when
/// [`omajor_w8a8_servable`] holds — which is the wide tensor's own order, so the
/// derivation costs no transpose — and row-major `[in, out]` otherwise. Either
/// way the block's length is `in·out` and the scales follow at that offset, so
/// the ranged κ bindings below are layout-independent.
///
/// A whole projection is keyed by its wide κ; a **head chunk** (a vocab-row
/// range of the LM-head/tied-embedding weight) is keyed by that κ AND its byte
/// range, so the several chunks that share one κ each map to their own per-chunk
/// artifact.
pub type QuantMap = std::collections::HashMap<String, (String, u64, u64)>;

/// The [`QuantMap`] key of a κ-bound projection weight: the bare κ for a
/// whole-tensor binding, or `κ@offset+len` for a ranged (head-chunk) binding.
/// The graph matcher and the derivation loop must mint the identical key for a
/// chunk to resolve its artifact — this one function is that shared law.
pub fn quant_key(kappa: &str, range: Option<(u64, u64)>) -> String {
    match range {
        Some((offset, len)) => format!("{kappa}@{offset}+{len}"),
        None => kappa.to_string(),
    }
}

/// One matched projection weight: the `MatMul` node, its wide param tensor id,
/// the κ, and the κ byte range when the binding is ranged (a head chunk).
struct WeightMatch {
    matmul_idx: usize,
    wide_tid: TensorId,
    kappa: String,
    range: Option<(u64, u64)>,
}

/// Matches of the projection chain `MatMul(x, Transpose{[1,0]}(wide))`
/// (optionally through a `Cast`) where `wide` is an [`AiParam::External`].
/// Whole-tensor bindings (`range: None`) and ranged head-chunk bindings are
/// both reported; a caller that only quantizes whole tensors (the retirement
/// prober) filters to `range.is_none()`.
fn transposed_external_matmul_weights(
    graph: &AiGraph,
    producer: &std::collections::HashMap<TensorId, usize>,
) -> Vec<WeightMatch> {
    let mut matches = Vec::new();
    for (m_idx, node) in graph.nodes.iter().enumerate() {
        if !matches!(node.op, AiOp::MatMul) {
            continue;
        }
        let Some(&b_tid) = node.inputs.get(1) else {
            continue;
        };
        let Some(&t_idx) = producer.get(&b_tid) else {
            continue;
        };
        let t_node = &graph.nodes[t_idx];
        if !matches!(&t_node.op, AiOp::Transpose { perm } if perm == &[1, 0]) {
            continue;
        }
        // The transpose input is the wide param, possibly through a Cast.
        let Some(&t_in) = t_node.inputs.first() else {
            continue;
        };
        let wide_tid = match producer.get(&t_in).map(|&i| &graph.nodes[i]) {
            Some(c) if matches!(c.op, AiOp::Cast { .. }) => match c.inputs.first() {
                Some(&w) => w,
                None => continue,
            },
            Some(_) => continue,
            None => t_in,
        };
        let Some(AiParam::External { kappa, range, .. }) = graph.params.get(&wide_tid) else {
            continue;
        };
        matches.push(WeightMatch {
            matmul_idx: m_idx,
            wide_tid,
            kappa: kappa.clone(),
            range: *range,
        });
    }
    matches
}

/// The wide κs this graph can rewrite onto quantized artifacts AND fully
/// retire — after the rewrite, no binding in the graph still carries the
/// wide κ. A κ with a consumer outside the projection chain (a tied
/// embedding's Gather, a chunked head's ranged bindings) is excluded: its
/// wide blob stays load-bearing and must not go gas-phase. Decided by
/// probing a clone with a dummy map — the same rewrite that will run,
/// never a parallel approximation of it.
pub fn quantizable_external_weights(graph: &AiGraph) -> Result<Vec<String>> {
    let producer: std::collections::HashMap<TensorId, usize> = graph
        .nodes
        .iter()
        .enumerate()
        .flat_map(|(i, n)| n.outputs.iter().map(move |t| (*t, i)))
        .collect();
    let mut dummy = QuantMap::new();
    // Retirement is a whole-κ decision: a ranged head-chunk binding never
    // retires its (shared, still-wide) κ, so the prober ignores ranged matches.
    for m in transposed_external_matmul_weights(graph, &producer)
        .into_iter()
        .filter(|m| m.range.is_none())
    {
        let (wide_tid, kappa) = (m.wide_tid, m.kappa);
        let Some(info) = graph.tensor_info.get(&wide_tid) else {
            continue;
        };
        let (Some(out), Some(inf)) = (
            info.shape.first().and_then(|d| d.as_concrete()),
            info.shape.get(1).and_then(|d| d.as_concrete()),
        ) else {
            continue;
        };
        if info.shape.len() != 2 {
            continue;
        }
        dummy.insert(kappa.clone(), (format!("probe:{kappa}"), out, inf));
    }
    if dummy.is_empty() {
        return Ok(Vec::new());
    }
    let mut probe = graph.clone();
    quantize_external_matmul_weights(&mut probe, &dummy)?;
    let still_wide: std::collections::HashSet<&String> = probe
        .params
        .values()
        .filter_map(|p| match p {
            AiParam::External { kappa, .. } if !kappa.starts_with("probe:") => Some(kappa),
            _ => None,
        })
        .collect();
    let mut eligible: Vec<String> = dummy
        .keys()
        .filter(|k| !still_wide.contains(k))
        .cloned()
        .collect();
    eligible.sort();
    Ok(eligible)
}

/// A head-chunk quantization target: a vocab-row range of the LM-head weight
/// the int8 tier can crystallize into its OWN per-chunk artifact. The several
/// chunks of one head share the wide κ (a tied head shares the embedding
/// table's κ, kept wide for the Gather) but each covers a distinct byte range,
/// so a whole-κ derivation cannot express them — each is derived from its slice
/// and keyed by [`quant_key`]`(κ, Some((offset, len)))`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeadChunkTarget {
    /// The wide κ the chunk ranges into (tied → the embedding table's κ).
    pub kappa: String,
    /// Byte offset of this chunk within the wide tensor.
    pub offset: u64,
    /// Byte length of this chunk (`rows · hidden · elem_bytes`).
    pub len: u64,
    /// Chunk rows — the vocab slice this chunk covers (the artifact's `out`).
    pub out_features: u64,
    /// Hidden size — the projection's `in`.
    pub in_features: u64,
}

/// The ranged (head-chunk) projection weights of a stage graph: the vocab-row
/// chunks of a large LM head that the int8 tier derives into per-chunk
/// artifacts. Whole-tensor projections (the 196 attention/MLP weights) are
/// reported by [`quantizable_external_weights`]; this is their ranged twin, so
/// a chunked head joins the int8 tier instead of staying a bf16 matmul whose
/// whole-panel F32 image thrashes residency.
pub fn ranged_external_matmul_weights(graph: &AiGraph) -> Vec<HeadChunkTarget> {
    let producer: std::collections::HashMap<TensorId, usize> = graph
        .nodes
        .iter()
        .enumerate()
        .flat_map(|(i, n)| n.outputs.iter().map(move |t| (*t, i)))
        .collect();
    transposed_external_matmul_weights(graph, &producer)
        .into_iter()
        .filter_map(|m| {
            let (offset, len) = m.range?;
            let info = graph.tensor_info.get(&m.wide_tid)?;
            if info.shape.len() != 2 {
                return None;
            }
            let out_features = info.shape.first()?.as_concrete()?;
            let in_features = info.shape.get(1)?.as_concrete()?;
            Some(HeadChunkTarget {
                kappa: m.kappa,
                offset,
                len,
                out_features,
                in_features,
            })
        })
        .collect()
}

/// Rewrite κ-bound MatMul weights onto their quantized derived artifacts
/// (row `quantized-transit`). The parametric recipe consumes a projection as
/// `MatMul(x, Transpose(weight))` (optionally through a `Cast` for narrow
/// storage) where `weight` is an un-ranged [`AiParam::External`] over the
/// wide κ. For each such weight whose κ has a recorded quantized derivation,
/// the wide binding is retired and replaced by TWO ranged bindings into the
/// artifact's κ (sub-tensor κ-resolution: the i8 block and the per-channel
/// f32 scales are ranges of one derived content) feeding
/// `Dequantize{axis:1} → MatMul` DIRECTLY — the artifact is stored in the
/// matmul orientation, so the compile-time transpose retires with the wide
/// binding and the dequant sits adjacent to its matmul (the fusable shape,
/// same as the inline int8 pass). The pass is structural — no weight bytes
/// are read; it runs on weightless stage graphs. Weights whose κ is absent
/// from the map, or consumed outside this chain (e.g. a tied embedding's
/// Gather), keep their wide binding — partial coverage is honest coverage.
/// Returns the number of rewritten weights.
pub fn quantize_external_matmul_weights(graph: &mut AiGraph, quant: &QuantMap) -> Result<usize> {
    let mut next_tid: TensorId = graph
        .tensor_info
        .keys()
        .chain(graph.params.keys())
        .copied()
        .max()
        .unwrap_or(0)
        + 1;
    let mut next_nid = graph.nodes.iter().map(|n| n.id).max().unwrap_or(0) + 1;

    // Producer index: output tid → node index.
    let producer: std::collections::HashMap<TensorId, usize> = graph
        .nodes
        .iter()
        .enumerate()
        .flat_map(|(i, n)| n.outputs.iter().map(move |t| (*t, i)))
        .collect();

    // Collect rewrites first (matmul node idx, wide param tid, quant entry)
    // so the mutation phase never fights the producer index.
    struct Rewrite {
        matmul_idx: usize,
        wide_tid: TensorId,
        kappa: String,
        out_features: u64,
        in_features: u64,
    }
    // The fused dequant-matmul kernel is 2D: rank>2 activations flatten to
    // [∏leading, in] around the matmul (a structural reshape pair — the
    // same computation, the kernel's proven shape) and unflatten after.
    let mut rewrites: Vec<Rewrite> = Vec::new();
    for m in transposed_external_matmul_weights(graph, &producer) {
        // Whole tensor → keyed by κ; head chunk → keyed by κ AND its range, so
        // the several chunks sharing one κ each resolve their own artifact.
        let key = quant_key(&m.kappa, m.range);
        let Some((quant_kappa, out_features, in_features)) = quant.get(&key) else {
            continue;
        };
        rewrites.push(Rewrite {
            matmul_idx: m.matmul_idx,
            wide_tid: m.wide_tid,
            kappa: quant_kappa.clone(),
            out_features: *out_features,
            in_features: *in_features,
        });
    }

    let rewritten = rewrites.len();
    for rw in rewrites {
        let name = graph
            .tensor_names
            .get(&rw.wide_tid)
            .cloned()
            .unwrap_or_else(|| format!("tensor_{}", rw.wide_tid));
        let (wq_tid, scale_tid, deq_tid) = (next_tid, next_tid + 1, next_tid + 2);
        next_tid += 3;
        let elems = rw.out_features * rw.in_features;

        // The artifact is stored in the matmul orientation: [in, out].
        let shape = shape_from_concrete(&[rw.in_features, rw.out_features]);
        let wq_info = TensorInfo::new(DType::INT8, shape.clone());
        graph.params.insert(
            wq_tid,
            AiParam::external_range(rw.kappa.clone(), wq_info.clone(), 0, elems),
        );
        graph.tensor_info.insert(wq_tid, wq_info);
        graph.tensor_names.insert(wq_tid, format!("{name}.q8"));

        let scale_info = TensorInfo::new(DType::F32, shape_from_concrete(&[rw.out_features]));
        graph.params.insert(
            scale_tid,
            AiParam::external_range(
                rw.kappa.clone(),
                scale_info.clone(),
                elems,
                rw.out_features * 4,
            ),
        );
        graph.tensor_info.insert(scale_tid, scale_info);
        graph
            .tensor_names
            .insert(scale_tid, format!("{name}.q8_scale"));

        graph
            .tensor_info
            .insert(deq_tid, TensorInfo::new(DType::F32, shape));
        graph.tensor_names.insert(deq_tid, format!("{name}.deq"));

        // The weight-slot declaration. This weight is LOAD-TIME BOUND — a κ the
        // binder materializes — which is the only kind that may assert
        // `OUTPUT_MAJOR`, and the assertion is what lets a weightless compile
        // reach the fused output-major decode GEMV at all.
        //
        // `omajor_w8a8_servable` is the same predicate `derive_quantized_artifact`
        // consults to choose the artifact's byte order, so the declaration and the
        // bytes cannot drift. When it holds, the artifact is `[out, in]` and this
        // node says so; when it does not, the artifact is `[in, out]` and this node
        // says that. There is no third state, and no fallback: a wrong pairing is
        // a wrong answer, not a slow one.
        //
        // W8A8 rides with the layout rather than being a separate knob, because the
        // output-major kernel and W8A8 are one call — the substrate pairs them and
        // rejects either alone.
        let servable = omajor_w8a8_servable(
            crate::lower::dtype::DTYPE_I8,
            rw.in_features as usize,
            rw.out_features as usize,
        );
        let (layout, act) = if servable {
            (WeightLayout::OutputMajor, ActQuant::W8A8TokenSym)
        } else {
            (WeightLayout::RowMajor, ActQuant::W8A32)
        };

        // Per-channel scales along the output columns of [in, out]: axis 1.
        // Feeding the matmul directly (no interposed transpose) is the
        // adjacency the substrate fuses in-register.
        graph.nodes.push(AiNode::new(
            next_nid,
            AiOp::Dequantize {
                axis: 1,
                layout,
                act,
            },
            vec![wq_tid, scale_tid],
            vec![deq_tid],
        ));
        next_nid += 1;
        graph.nodes[rw.matmul_idx].inputs[1] = deq_tid;

        // Flatten rank>2 activations to the kernel's 2D shape.
        let a_tid = graph.nodes[rw.matmul_idx].inputs[0];
        let a_shape = graph
            .tensor_info
            .get(&a_tid)
            .map(|i| i.shape.clone())
            .unwrap_or_default();
        if a_shape.len() > 2 {
            let (lead, last) = a_shape.split_at(a_shape.len() - 1);
            let flat_rows = lead
                .iter()
                .cloned()
                .reduce(|a, b| crate::ir::DimExpr::Mul(Box::new(a), Box::new(b)))
                .expect("rank>2 has leading dims");
            let (a_flat_tid, out_flat_tid) = (next_tid, next_tid + 1);
            next_tid += 2;
            graph.tensor_info.insert(
                a_flat_tid,
                TensorInfo::new(DType::F32, vec![flat_rows.clone(), last[0].clone()].into()),
            );
            graph
                .tensor_names
                .insert(a_flat_tid, format!("{name}.a_flat"));
            graph.tensor_info.insert(
                out_flat_tid,
                TensorInfo::new(
                    DType::F32,
                    vec![flat_rows, crate::ir::DimExpr::Concrete(rw.out_features)].into(),
                ),
            );
            graph
                .tensor_names
                .insert(out_flat_tid, format!("{name}.out_flat"));

            let out_tid = graph.nodes[rw.matmul_idx].outputs[0];
            graph.nodes.push(AiNode::new(
                next_nid,
                AiOp::Reshape { allow_zero: false },
                vec![a_tid],
                vec![a_flat_tid],
            ));
            graph.nodes.push(AiNode::new(
                next_nid + 1,
                AiOp::Reshape { allow_zero: false },
                vec![out_flat_tid],
                vec![out_tid],
            ));
            next_nid += 2;
            graph.nodes[rw.matmul_idx].inputs[0] = a_flat_tid;
            graph.nodes[rw.matmul_idx].outputs[0] = out_flat_tid;
        }
    }

    if rewritten > 0 {
        // Retire params and interior nodes (the bf16 Cast) that no longer
        // feed anything — a retired wide κ must leave the κ-map so it never
        // transits, and a dead Cast would fail lowering on a missing input.
        loop {
            let referenced: std::collections::HashSet<TensorId> = graph
                .nodes
                .iter()
                .flat_map(|n| n.inputs.iter().copied())
                .chain(graph.outputs.iter().copied())
                .collect();
            let dead_nodes: Vec<usize> = graph
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, n)| {
                    matches!(n.op, AiOp::Cast { .. } | AiOp::Transpose { .. })
                        && n.outputs.iter().all(|t| !referenced.contains(t))
                })
                .map(|(i, _)| i)
                .collect();
            let dead_params: Vec<TensorId> = graph
                .params
                .keys()
                .filter(|t| !referenced.contains(t) && !graph.inputs.contains(t))
                .copied()
                .collect();
            if dead_nodes.is_empty() && dead_params.is_empty() {
                break;
            }
            for i in dead_nodes.into_iter().rev() {
                let node = graph.nodes.remove(i);
                for t in node.outputs {
                    graph.tensor_info.remove(&t);
                    graph.tensor_names.remove(&t);
                }
            }
            for t in dead_params {
                graph.params.remove(&t);
                graph.tensor_info.remove(&t);
                graph.tensor_names.remove(&t);
            }
        }
        graph.invalidate_topo_cache();
    }
    Ok(rewritten)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{shape_from_concrete, DimVarTable, TensorInfo};
    use std::collections::HashMap;

    /// A minimal MatMul graph: `X[1,4] · W[4,2] → Y[1,2]` where W is an inline
    /// f32 constant.
    fn f32_weight_matmul_graph() -> AiGraph {
        let mut ti: HashMap<TensorId, TensorInfo> = HashMap::new();
        let mut params: HashMap<TensorId, AiParam> = HashMap::new();

        // X = input [1, 4]
        ti.insert(0, TensorInfo::new(DType::F32, shape_from_concrete(&[1, 4])));

        // W = f32 weight [4, 2]
        let w: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
        let w_bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
        let w_info = TensorInfo::new(DType::F32, shape_from_concrete(&[4, 2]));
        params.insert(1, AiParam::inline(w_bytes, w_info.clone()));
        ti.insert(1, w_info);

        // Y = output [1, 2]
        ti.insert(2, TensorInfo::new(DType::F32, shape_from_concrete(&[1, 2])));

        AiGraph {
            name: "qtest".into(),
            nodes: vec![AiNode::new(0, AiOp::MatMul, vec![0, 1], vec![2])],
            inputs: vec![0],
            outputs: vec![2],
            input_names: Vec::new(),
            output_names: Vec::new(),
            params,
            tensor_info: ti,
            metadata: HashMap::new(),
            warnings: Vec::new(),
            dim_vars: DimVarTable::default(),
            shape_constraints: crate::ir::ConstraintStore::default(),
            subgraphs: HashMap::new(),
            tensor_names: HashMap::new(),
            topo_cache: Default::default(),
        }
    }

    #[test]
    fn none_is_noop() {
        let mut g = f32_weight_matmul_graph();
        let before = g.nodes.len();
        quantize_weights(&mut g, QuantStrategy::None).unwrap();
        assert_eq!(g.nodes.len(), before);
        assert!(matches!(g.params.get(&1), Some(AiParam::Inline { .. })));
    }

    #[test]
    fn int8_rewrites_matmul_weight() {
        let mut g = f32_weight_matmul_graph();
        quantize_weights(&mut g, QuantStrategy::Int8).unwrap();

        // A Dequantize node was added.
        let deq = g
            .nodes
            .iter()
            .find(|n| matches!(n.op, AiOp::Dequantize { .. }))
            .expect("dequant node inserted");

        // Its weight operand is now an i8 constant of the original shape [4,2].
        let wq_tid = deq.inputs[0];
        match g.params.get(&wq_tid) {
            Some(AiParam::Inline { info, data }) => {
                assert_eq!(info.logical_dtype, DType::INT8);
                assert_eq!(data.len(), 4 * 2); // 1 byte/elem
            }
            _ => panic!("i8 weight constant missing"),
        }

        // Scale operand is an f32 vector of length n=2.
        let scale_tid = deq.inputs[1];
        match g.params.get(&scale_tid) {
            Some(AiParam::Inline { info, data }) => {
                assert_eq!(info.logical_dtype, DType::F32);
                assert_eq!(data.len(), 2 * 4);
            }
            _ => panic!("scale constant missing"),
        }

        // The MatMul's B now points at the dequant output; the old f32 const is gone.
        let mm = g
            .nodes
            .iter()
            .find(|n| matches!(n.op, AiOp::MatMul))
            .unwrap();
        assert_eq!(mm.inputs[1], deq.outputs[0]);
        assert!(!g.params.contains_key(&1), "old f32 weight retired");
    }

    /// A head-chunk projection: `MatMul(x, Transpose(Cast(external_ranged)))`,
    /// the exact shape the parametric head chunk emits — a ranged bf16 external
    /// (a vocab-row slice of the tied embedding), widened by `Cast`, transposed,
    /// matmul'd. `out` rows of `in` hidden, at byte `offset` of the wide κ.
    fn ranged_head_chunk_graph(kappa: &str, offset: u64, out: u64, inf: u64) -> AiGraph {
        let mut ti: HashMap<TensorId, TensorInfo> = HashMap::new();
        let mut params: HashMap<TensorId, AiParam> = HashMap::new();
        // x = input [1, in]
        ti.insert(
            0,
            TensorInfo::new(DType::F32, shape_from_concrete(&[1, inf])),
        );
        // W = ranged bf16 external [out, in]
        let w_info = TensorInfo::new(DType::BF16, shape_from_concrete(&[out, inf]));
        let len = out * inf * 2; // bf16 = 2 bytes
        params.insert(
            1,
            AiParam::external_range(kappa.to_string(), w_info.clone(), offset, len),
        );
        ti.insert(1, w_info);
        // Wf32 = Cast(W) [out, in]
        ti.insert(
            2,
            TensorInfo::new(DType::F32, shape_from_concrete(&[out, inf])),
        );
        // Wt = Transpose(Wf32) [in, out]
        ti.insert(
            3,
            TensorInfo::new(DType::F32, shape_from_concrete(&[inf, out])),
        );
        // y = MatMul(x, Wt) [1, out]
        ti.insert(
            4,
            TensorInfo::new(DType::F32, shape_from_concrete(&[1, out])),
        );
        AiGraph {
            name: "head_chunk".into(),
            nodes: vec![
                AiNode::new(0, AiOp::Cast { to: DType::F32 }, vec![1], vec![2]),
                AiNode::new(1, AiOp::Transpose { perm: vec![1, 0] }, vec![2], vec![3]),
                AiNode::new(2, AiOp::MatMul, vec![0, 3], vec![4]),
            ],
            inputs: vec![0],
            outputs: vec![4],
            input_names: Vec::new(),
            output_names: Vec::new(),
            params,
            tensor_info: ti,
            metadata: HashMap::new(),
            warnings: Vec::new(),
            dim_vars: DimVarTable::default(),
            shape_constraints: crate::ir::ConstraintStore::default(),
            subgraphs: HashMap::new(),
            tensor_names: HashMap::new(),
            topo_cache: Default::default(),
        }
    }

    #[test]
    fn ranged_head_chunk_is_a_quant_target_keyed_by_kappa_and_range() {
        // Two chunks of ONE head share the κ but cover distinct byte ranges —
        // each must be its own target under a distinct composite key.
        let g0 = ranged_head_chunk_graph("embed", 0, 3, 4);
        let g1 = ranged_head_chunk_graph("embed", 3 * 4 * 2, 3, 4);
        let t0 = ranged_external_matmul_weights(&g0);
        let t1 = ranged_external_matmul_weights(&g1);
        assert_eq!(t0.len(), 1);
        assert_eq!(t1.len(), 1);
        assert_eq!(t0[0].kappa, "embed");
        assert_eq!(
            (
                t0[0].offset,
                t0[0].len,
                t0[0].out_features,
                t0[0].in_features
            ),
            (0, 24, 3, 4)
        );
        assert_eq!(t1[0].offset, 24);
        // Distinct composite keys for chunks that share the κ.
        assert_ne!(
            quant_key(&t0[0].kappa, Some((t0[0].offset, t0[0].len))),
            quant_key(&t1[0].kappa, Some((t1[0].offset, t1[0].len))),
        );
        // A whole-tensor retirement prober must NOT see a ranged chunk.
        assert!(quantizable_external_weights(&g0).unwrap().is_empty());
    }

    #[test]
    fn ranged_head_chunk_rewrites_onto_its_per_chunk_artifact() {
        let mut g = ranged_head_chunk_graph("embed", 0, 3, 4);
        let key = quant_key("embed", Some((0, 24)));
        let mut quant = QuantMap::new();
        quant.insert(key, ("chunk-artifact-κ".to_string(), 3, 4));

        let rewritten = quantize_external_matmul_weights(&mut g, &quant).unwrap();
        assert_eq!(rewritten, 1, "the ranged head chunk must be rewritten");

        // A Dequantize node feeds the matmul directly; the bf16 Cast+Transpose
        // and the wide ranged binding are gone (no F32 head panel remains).
        let deq = g
            .nodes
            .iter()
            .find(|n| matches!(n.op, AiOp::Dequantize { .. }))
            .expect("dequant inserted");
        let mm = g
            .nodes
            .iter()
            .find(|n| matches!(n.op, AiOp::MatMul))
            .unwrap();
        assert_eq!(
            mm.inputs[1], deq.outputs[0],
            "matmul B is the dequant output"
        );
        assert!(
            !g.nodes
                .iter()
                .any(|n| matches!(n.op, AiOp::Cast { .. } | AiOp::Transpose { .. })),
            "the bf16 Cast and Transpose retire with the wide binding"
        );
        // Both artifact bindings range into the ONE per-chunk artifact κ.
        let art: Vec<_> = g
            .params
            .values()
            .filter_map(|p| match p {
                AiParam::External { kappa, .. } => Some(kappa.as_str()),
                _ => None,
            })
            .collect();
        assert!(art.iter().all(|k| *k == "chunk-artifact-κ"));
        assert_eq!(
            art.len(),
            2,
            "i8 block + per-channel scales range into one artifact"
        );
    }

    #[test]
    fn int4_errors() {
        let mut g = f32_weight_matmul_graph();
        assert!(quantize_weights(&mut g, QuantStrategy::Int4).is_err());
    }

    #[test]
    fn activation_b_is_not_quantized() {
        // B is a runtime activation (no param) → pass leaves the graph untouched.
        let mut g = f32_weight_matmul_graph();
        g.params.remove(&1); // W is no longer a constant
        let node_count = g.nodes.len();
        quantize_weights(&mut g, QuantStrategy::Int8).unwrap();
        assert_eq!(g.nodes.len(), node_count, "no Dequantize inserted");
        assert!(!g.params.contains_key(&1));
    }

    #[test]
    fn two_matmuls_get_distinct_ids() {
        // Build a graph with two MatMul nodes each with their own inline f32
        // weight constant. After Int8 quantization both must get a Dequantize
        // node and all newly-created TensorIds/NodeIds must be unique.
        let mut ti: HashMap<TensorId, TensorInfo> = HashMap::new();
        let mut params: HashMap<TensorId, AiParam> = HashMap::new();

        // X = input [1, 4]
        ti.insert(0, TensorInfo::new(DType::F32, shape_from_concrete(&[1, 4])));

        // W1 = f32 weight [4, 2] for first MatMul
        let w: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
        let w_bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
        let w_info = TensorInfo::new(DType::F32, shape_from_concrete(&[4, 2]));
        params.insert(1, AiParam::inline(w_bytes.clone(), w_info.clone()));
        ti.insert(1, w_info.clone());

        // Y1 = first MatMul output [1, 2]
        ti.insert(2, TensorInfo::new(DType::F32, shape_from_concrete(&[1, 2])));

        // W2 = separate f32 weight [4, 2] for second MatMul (tid=3)
        params.insert(3, AiParam::inline(w_bytes, w_info.clone()));
        ti.insert(3, w_info);

        // Y2 = second MatMul output [1, 2] (tid=4)
        ti.insert(4, TensorInfo::new(DType::F32, shape_from_concrete(&[1, 2])));

        let mut g = AiGraph {
            name: "two_mm".into(),
            nodes: vec![
                AiNode::new(0, AiOp::MatMul, vec![0, 1], vec![2]),
                AiNode::new(1, AiOp::MatMul, vec![0, 3], vec![4]),
            ],
            inputs: vec![0],
            outputs: vec![2, 4],
            input_names: Vec::new(),
            output_names: Vec::new(),
            params,
            tensor_info: ti,
            metadata: HashMap::new(),
            warnings: Vec::new(),
            dim_vars: crate::ir::DimVarTable::default(),
            shape_constraints: crate::ir::ConstraintStore::default(),
            subgraphs: HashMap::new(),
            tensor_names: HashMap::new(),
            topo_cache: Default::default(),
        };

        quantize_weights(&mut g, QuantStrategy::Int8).unwrap();

        // Both MatMuls must have a Dequantize node.
        let deq_nodes: Vec<_> = g
            .nodes
            .iter()
            .filter(|n| matches!(n.op, AiOp::Dequantize { .. }))
            .collect();
        assert_eq!(deq_nodes.len(), 2, "expected 2 Dequantize nodes");

        // The two Dequantize output TensorIds must be distinct.
        let deq_out_0 = deq_nodes[0].outputs[0];
        let deq_out_1 = deq_nodes[1].outputs[0];
        assert_ne!(
            deq_out_0, deq_out_1,
            "dequant outputs must have distinct TensorIds"
        );

        // The two Dequantize NodeIds must be distinct.
        assert_ne!(
            deq_nodes[0].id, deq_nodes[1].id,
            "dequant nodes must have distinct NodeIds"
        );

        // All tensor_info keys are unique (HashMap guarantees this); just
        // assert the two dequant output tids actually appear in tensor_info.
        assert!(g.tensor_info.contains_key(&deq_out_0));
        assert!(g.tensor_info.contains_key(&deq_out_1));
    }
}
