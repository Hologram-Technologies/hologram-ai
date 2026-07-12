//! The load-bearing migration gate for ADR-0019 increment 2b: the v0.9.0 fused
//! split-KV `DecodeAttention` path computes the SAME attention as the legacy
//! per-group SDPA decomposition — "we changed the schedule, not the numbers."
//!
//! Driver-independent: a synthetic single-GQA decode graph is rewritten BOTH
//! ways (`rewrite_decode_attention(.., resident_kv)`), each archive compiled and
//! driven directly via `HoloRunner` with matched inputs (the two forms differ
//! only in port SHAPE — same q/k/v/past, the same visibility encoded in each
//! form's mask, and the fused `pos`). We compare the attention output.
//!
//! Parametric: nothing is hardcoded to a family — the test sweeps a grid of
//! (heads, kv_heads, head_dim, bucket) so the wiring is proven for MHA (g=1),
//! GQA (g>1), and several sizes. A fuse reorders the f32 accumulation, so this
//! is numeric-equivalence within a tight relative tolerance (invariance-ladder
//! rung 2 bit-identity applies within ONE kernel's order, not across a fuse).

use std::collections::HashMap;

use hologram_ai::{HoloRunner, ModelCompiler, ModelSource};
use hologram_ai_common::opt::decode_plan::{
    past_key_port, past_value_port, rewrite_decode_attention, DECODE_MASK_PORT, DECODE_POS_PORT,
    DECODE_ROPE_COS_K_PORT, DECODE_ROPE_COS_Q_PORT, DECODE_ROPE_SIN_K_PORT, DECODE_ROPE_SIN_Q_PORT,
};
use hologram_ai_common::{shape_from_concrete, AiGraph, AiNode, AiOp, DType, TensorInfo};

fn ti(dt: DType, dims: &[u64]) -> TensorInfo {
    TensorInfo::new(dt, shape_from_concrete(dims))
}
fn f32s(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 31 + seed * 17) % 53) as f32 - 26.0) * 0.037)
        .collect()
}
fn to_le(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn le_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// A one-layer, batch-1, chunk-1 GQA decoder graph. RoPE is on but fed identity
/// tables (cos=1, sin=0) so it is an exact no-op — the fused/legacy comparison
/// is about the attention+cache path, not RoPE (which both share verbatim).
/// Tensor ids: q=0 k=1 v=2 attn=3.
fn gqa_graph(h: u64, kv: u64, dh: u64) -> AiGraph {
    let (q, k, v, attn) = (0u32, 1, 2, 3);
    let mut tinfo = HashMap::new();
    tinfo.insert(q, ti(DType::F32, &[1, h, 1, dh]));
    tinfo.insert(k, ti(DType::F32, &[1, kv, 1, dh]));
    tinfo.insert(v, ti(DType::F32, &[1, kv, 1, dh]));
    tinfo.insert(attn, ti(DType::F32, &[1, h, 1, dh]));
    AiGraph {
        name: "gqa1".into(),
        nodes: vec![AiNode::new(
            0,
            AiOp::GroupedQueryAttention {
                num_heads: h as u32,
                num_kv_heads: kv as u32,
                head_dim: dh as u32,
                scale: None,
                causal: true,
                heads_first: true,
                qk_norm: false,
                rope: true,
                rope_base: 10000.0,
            },
            vec![q, k, v],
            vec![attn],
        )],
        inputs: vec![q, k, v],
        outputs: vec![attn],
        input_names: vec!["q".into(), "k".into(), "v".into()],
        output_names: vec!["attn".into()],
        params: HashMap::new(),
        tensor_info: tinfo,
        metadata: HashMap::new(),
        warnings: Vec::new(),
        dim_vars: Default::default(),
        shape_constraints: Default::default(),
        subgraphs: HashMap::new(),
        tensor_names: HashMap::new(),
        topo_cache: Default::default(),
    }
}

/// Drive `runner` by resolving each input port's bytes by NAME.
fn run_named(runner: &mut HoloRunner, named: &HashMap<String, Vec<u8>>) -> Vec<f32> {
    let ports = runner.input_port_info();
    let bufs: Vec<Vec<u8>> = ports
        .iter()
        .map(|p| {
            named
                .get(&p.name)
                .unwrap_or_else(|| panic!("no test data for input port `{}`", p.name))
                .clone()
        })
        .collect();
    let refs: Vec<&[u8]> = bufs.iter().map(|b| b.as_slice()).collect();
    let out = runner.execute(&refs).expect("decode step executes");
    le_to_f32(&out[0].bytes) // output 0 is the attention result (out_tid)
}

/// Shared per-port inputs for both forms at a fixed realized length. `past_k`/
/// `past_v` are the same bytes; the fused form just views them rank-4 and adds
/// `pos`. Each form gets its own mask shape carrying the SAME visibility.
fn build_inputs(
    h: u64,
    kv: u64,
    dh: u64,
    bucket: u64,
    g: u64,
    realized: u64,
    resident_kv: bool,
) -> HashMap<String, Vec<u8>> {
    let mut m = HashMap::new();
    m.insert("q".into(), to_le(&f32s((h * dh) as usize, 1)));
    m.insert("k".into(), to_le(&f32s((kv * dh) as usize, 2)));
    m.insert("v".into(), to_le(&f32s((kv * dh) as usize, 3)));
    // Identity rope tables (cos=1, sin=0) → rope is a no-op.
    m.insert(
        DECODE_ROPE_COS_Q_PORT.into(),
        to_le(&vec![1.0; (h * dh) as usize]),
    );
    m.insert(
        DECODE_ROPE_SIN_Q_PORT.into(),
        to_le(&vec![0.0; (h * dh) as usize]),
    );
    m.insert(
        DECODE_ROPE_COS_K_PORT.into(),
        to_le(&vec![1.0; (kv * dh) as usize]),
    );
    m.insert(
        DECODE_ROPE_SIN_K_PORT.into(),
        to_le(&vec![0.0; (kv * dh) as usize]),
    );
    // Same carried K/V bytes; the unrealized rows are masked out identically.
    let kv_bytes = (kv * bucket * dh) as usize;
    m.insert(past_key_port(0), to_le(&f32s(kv_bytes, 4)));
    m.insert(past_value_port(0), to_le(&f32s(kv_bytes, 5)));
    // One query row's visibility over [bucket past ∥ 1 new key]: past cols
    // < realized visible, then masked, and the new key visible.
    let row: Vec<f32> = (0..bucket + 1)
        .map(|j| {
            if j < realized || j == bucket {
                0.0
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect();
    let mask_rows = if resident_kv { 1 } else { g };
    let mask: Vec<f32> = (0..mask_rows).flat_map(|_| row.clone()).collect();
    m.insert(DECODE_MASK_PORT.into(), to_le(&mask));
    if resident_kv {
        m.insert(
            DECODE_POS_PORT.into(),
            (realized as u32).to_le_bytes().to_vec(),
        );
    }
    m
}

fn compile_form(h: u64, kv: u64, dh: u64, bucket: u64, resident_kv: bool) -> HoloRunner {
    let mut graph = gqa_graph(h, kv, dh);
    rewrite_decode_attention(&mut graph, bucket, 1, 0, resident_kv)
        .expect("rewrite the single GQA layer");
    let archive = ModelCompiler::default()
        .compile(ModelSource::AiGraph(graph))
        .unwrap_or_else(|e| panic!("compile (resident_kv={resident_kv}): {e:?}"));
    HoloRunner::from_bytes(archive.bytes).expect("archive loads")
}

#[test]
fn fused_decode_attention_matches_the_legacy_decomposition_across_shapes() {
    // (heads, kv_heads, head_dim, bucket) — MHA (g=1) and GQA (g>1), several
    // sizes; head_dim even (RoPE splits it in half). Parametric, no family.
    // (The bare-synthetic LEGACY compile has a pre-existing completeness limit
    // on some MQA shapes unrelated to the fused path — the fused form compiles
    // them — so the fused==legacy grid uses shapes both forms compile.)
    let grid = [
        (4u64, 4u64, 8u64, 6u64), // MHA, g=1
        (4, 2, 8, 6),             // GQA, g=2
        (8, 2, 16, 10),           // GQA, g=4
        (6, 3, 32, 12),           // GQA, g=2, wider head
    ];
    for (h, kv, dh, bucket) in grid {
        let g = h / kv;
        let realized = (bucket / 2).max(1);
        let mut legacy = compile_form(h, kv, dh, bucket, false);
        let mut fused = compile_form(h, kv, dh, bucket, true);

        let out_l = run_named(
            &mut legacy,
            &build_inputs(h, kv, dh, bucket, g, realized, false),
        );
        let out_f = run_named(
            &mut fused,
            &build_inputs(h, kv, dh, bucket, g, realized, true),
        );

        assert_eq!(out_l.len(), out_f.len(), "shape {h}/{kv}/{dh}/{bucket}");
        assert_eq!(out_l.len(), (h * dh) as usize);
        let mut max_rel = 0.0f32;
        for (a, b) in out_l.iter().zip(&out_f) {
            assert!(
                a.is_finite() && b.is_finite(),
                "non-finite at {h}/{kv}/{dh}"
            );
            let denom = a.abs().max(b.abs()).max(1e-6);
            max_rel = max_rel.max((a - b).abs() / denom);
        }
        // A fuse reorders the f32 accumulation; the two SDPAs agree to a tight
        // relative bound (measured well under this — a wiring bug blows past it).
        assert!(
            max_rel < 2e-4,
            "fused != legacy for heads={h} kv={kv} dh={dh} bucket={bucket}: \
             max relative diff {max_rel:.2e} exceeds 2e-4 — a layout/scale/mask \
             wiring bug, not fp reordering"
        );
    }
}
