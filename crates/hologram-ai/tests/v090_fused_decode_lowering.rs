//! Lowering witness for ADR-0019 increment 2a: our IR's `AiOp::DecodeAttention`
//! and `AiOp::KvCacheWrite` lower, through the REAL `ModelCompiler` pipeline, to
//! the substrate's six-input `Attention` (κ discriminant 119, `DecodeAttention`)
//! and `KvCacheWrite` (κ120) kernel calls — and the compiled archive executes.
//!
//! This is the seam `rewrite_decode_attention` will emit onto (increment 2b):
//! before rewiring the crash-prone decode graph, prove the two new IR ops carry
//! from AiGraph → our lowering → compile → archive → session with the right
//! discriminants and the right bytes. The `KvCacheWrite` outputs are checked
//! against an independent host ring-write oracle (an exact byte move, so bitwise
//! on any lane); the attention output is checked for the right shape and
//! finiteness (its numeric contract is the substrate's `decode_attention_e2e`).

use std::collections::HashMap;

use hologram_ai::{HoloRunner, ModelCompiler, ModelSource};
use hologram_ai_common::{shape_from_concrete, AiGraph, AiNode, AiOp, DType, TensorInfo};
use hologram_archive::{decoder, format::SectionKind, HoloLoader};
use hologram_backend::KernelCall;

// b, query heads, kv heads, bucket rows, head dim. Small, GQA (h != hkv).
const B: u64 = 1;
const H: u64 = 4;
const HKV: u64 = 2;
const BUCKET: u64 = 8;
const D: u64 = 16;

fn ti(dt: DType, dims: &[u64]) -> TensorInfo {
    TensorInfo::new(dt, shape_from_concrete(dims))
}

fn f32s(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 13 + seed * 7) % 41) as f32 - 20.0) * 0.043)
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

/// One resident-KV decode step, authored in OUR IR:
/// `attn = DecodeAttention(q, k_past, v_past, k_new, v_new, mask)` plus two
/// `KvCacheWrite`s that ring-write the new rows into the caches at `pos`.
/// Tensor ids: q=0 kp=1 vp=2 kn=3 vn=4 mask=5 pos=6 attn=7 kc=8 vc=9.
fn fused_decode_step() -> AiGraph {
    let (q, kp, vp, kn, vn, mask, pos, attn, kc, vc) = (0u32, 1, 2, 3, 4, 5, 6, 7, 8, 9);
    let mut tinfo = HashMap::new();
    tinfo.insert(q, ti(DType::F32, &[B, H, 1, D]));
    tinfo.insert(kp, ti(DType::F32, &[B, HKV, BUCKET, D]));
    tinfo.insert(vp, ti(DType::F32, &[B, HKV, BUCKET, D]));
    tinfo.insert(kn, ti(DType::F32, &[B, HKV, 1, D]));
    tinfo.insert(vn, ti(DType::F32, &[B, HKV, 1, D]));
    tinfo.insert(mask, ti(DType::F32, &[1, BUCKET + 1]));
    tinfo.insert(pos, ti(DType::INT32, &[1]));
    tinfo.insert(attn, ti(DType::F32, &[B, H, 1, D]));
    tinfo.insert(kc, ti(DType::F32, &[B, HKV, BUCKET, D]));
    tinfo.insert(vc, ti(DType::F32, &[B, HKV, BUCKET, D]));
    AiGraph {
        name: "fused_decode_step".into(),
        nodes: vec![
            AiNode::new(
                0,
                AiOp::DecodeAttention,
                vec![q, kp, vp, kn, vn, mask],
                vec![attn],
            ),
            AiNode::new(1, AiOp::KvCacheWrite, vec![kp, kn, pos], vec![kc]),
            AiNode::new(2, AiOp::KvCacheWrite, vec![vp, vn, pos], vec![vc]),
        ],
        inputs: vec![q, kp, vp, kn, vn, mask, pos],
        outputs: vec![attn, kc, vc],
        input_names: Vec::new(),
        output_names: Vec::new(),
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

/// Independent host oracle for the ring write (see the increment-1 contract).
fn ring_write_oracle(
    cache: &[f32],
    new: &[f32],
    pos: u32,
    planes: usize,
    bucket: usize,
    d: usize,
) -> Vec<f32> {
    let mut out = cache.to_vec();
    let row = (pos as usize) % bucket;
    for p in 0..planes {
        out[p * bucket * d + row * d..p * bucket * d + row * d + d]
            .copy_from_slice(&new[p * d..p * d + d]);
    }
    out
}

#[test]
fn ir_decode_attention_and_kv_cache_write_lower_to_119_120_and_execute() {
    let archive = ModelCompiler::default()
        .compile(ModelSource::AiGraph(fused_decode_step()))
        .expect("the fused decode step must compile through our pipeline");

    // (a) The archive carries the substrate's decode calls, by discriminant.
    let plan = HoloLoader::from_bytes(&archive.bytes)
        .expect("archive loads")
        .into_plan()
        .expect("archive has a plan");
    let calls = decoder::decode_calls(
        plan.section(SectionKind::KernelCalls)
            .expect("kernel calls"),
    )
    .expect("kernel calls decode");
    let n_attn = calls
        .iter()
        .filter(|c| matches!(c, KernelCall::DecodeAttention(_)))
        .count();
    let n_write = calls
        .iter()
        .filter(|c| matches!(c, KernelCall::KvCacheWrite(_)))
        .count();
    assert_eq!(
        n_attn, 1,
        "AiOp::DecodeAttention must lower to exactly one DecodeAttention (κ119) call"
    );
    assert_eq!(
        n_write, 2,
        "two AiOp::KvCacheWrite must lower to two KvCacheWrite (κ120) calls"
    );

    // (b) It executes, and the writes equal the ring oracle (exact byte move).
    let planes = (B * HKV) as usize;
    let (bucket, d) = (BUCKET as usize, D as usize);
    let q = f32s((B * H * D) as usize, 1);
    let kp = f32s(planes * bucket * d, 2);
    let vp = f32s(planes * bucket * d, 3);
    let kn = f32s(planes * d, 4);
    let vn = f32s(planes * d, 5);
    let pos = 3u32;
    // Realized-length mask: rows past `pos` (and the padded tail) unreachable.
    let mask: Vec<f32> = (0..(bucket + 1))
        .map(|j| {
            if j <= pos as usize {
                0.0
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect();

    let mut runner =
        HoloRunner::from_bytes(archive.bytes).expect("archive loads through HoloRunner");
    let inputs: Vec<Vec<u8>> = vec![
        to_le(&q),
        to_le(&kp),
        to_le(&vp),
        to_le(&kn),
        to_le(&vn),
        to_le(&mask),
        pos.to_le_bytes().to_vec(),
    ];
    let refs: Vec<&[u8]> = inputs.iter().map(|v| v.as_slice()).collect();
    let out = runner
        .execute(&refs)
        .expect("the fused decode step executes");
    assert_eq!(out.len(), 3, "outputs: attn, k_cache', v_cache'");

    let attn = le_to_f32(&out[0].bytes);
    assert_eq!(
        attn.len(),
        (B * H * D) as usize,
        "attention output is q-shaped"
    );
    assert!(
        attn.iter().all(|x| x.is_finite()),
        "attention output is finite (mask erases exactly, no NaN)"
    );

    assert_eq!(
        le_to_f32(&out[1].bytes),
        ring_write_oracle(&kp, &kn, pos, planes, bucket, d),
        "k KvCacheWrite must equal the ring-write oracle"
    );
    assert_eq!(
        le_to_f32(&out[2].bytes),
        ring_write_oracle(&vp, &vn, pos, planes, bucket, d),
        "v KvCacheWrite must equal the ring-write oracle"
    );
}
