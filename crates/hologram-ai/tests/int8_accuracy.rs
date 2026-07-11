//! End-to-end accuracy gate for compile-time int8 weight quantization.
//!
//! Compiles the SAME plain-f32 linear model two ways — `QuantStrategy::None`
//! (f32 weights) and `QuantStrategy::Int8` (the quantize_weights pass rewrites
//! the MatMul weight to i8 + per-channel Dequantize) — runs both, and asserts
//! the int8 path (1) fused `Dequantize→MatMul` into `MatMulDequant` and
//! (2) matches the f32 logits to cosine ≥ 0.999.

use std::collections::HashMap;

use hologram_ai::{HoloRunner, ModelCompiler, ModelSource};
use hologram_ai_common::lower::QuantStrategy;
use hologram_ai_common::{shape_from_concrete, AiGraph, AiNode, AiOp, AiParam, DType, TensorInfo};

const K: usize = 64;
const N: usize = 48;

fn info(dtype: DType, dims: &[u64]) -> TensorInfo {
    TensorInfo::new(dtype, shape_from_concrete(dims))
}

/// `Y[1,N] = X[1,K] · W[K,N]` with `W` an inline f32 constant of varied
/// magnitude. The quantize pass (when enabled) rewrites `W`.
fn linear_graph() -> AiGraph {
    let w: Vec<f32> = (0..K * N)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.05 + ((i % 7) as f32) * 0.01)
        .collect();
    let w_bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();

    let mut params: HashMap<u32, AiParam> = HashMap::new();
    let mut ti: HashMap<u32, TensorInfo> = HashMap::new();
    // tids: 0 = X (input), 1 = W (f32 const), 2 = Y (matmul out)
    ti.insert(0, info(DType::F32, &[1, K as u64]));
    ti.insert(1, info(DType::F32, &[K as u64, N as u64]));
    ti.insert(2, info(DType::F32, &[1, N as u64]));
    params.insert(1, AiParam::inline(w_bytes, ti[&1].clone()));

    AiGraph {
        name: "int8_acc".into(),
        nodes: vec![AiNode::new(0, AiOp::MatMul, vec![0, 1], vec![2])],
        inputs: vec![0],
        outputs: vec![2],
        input_names: Vec::new(),
        output_names: Vec::new(),
        params,
        tensor_info: ti,
        metadata: HashMap::new(),
        warnings: Vec::new(),
        dim_vars: Default::default(),
        shape_constraints: Default::default(),
        subgraphs: HashMap::new(),
        tensor_names: HashMap::new(),
        topo_cache: Default::default(),
    }
}

fn run(strategy: QuantStrategy, x: &[f32]) -> (HoloRunner, Vec<f32>) {
    let compiler = ModelCompiler {
        quant_strategy: strategy,
        ..ModelCompiler::default()
    };
    let archive = compiler
        .compile(ModelSource::AiGraph(linear_graph()))
        .expect("compile failed");
    let mut runner = HoloRunner::from_bytes(archive.bytes).expect("load failed");
    let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let out = runner.execute(&[&x_bytes]).expect("execute failed");
    let y: Vec<f32> = out[0]
        .bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    (runner, y)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

#[test]
fn int8_matches_f32_within_cosine_999() {
    let x: Vec<f32> = (0..K).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
    let (_f, f_out) = run(QuantStrategy::None, &x);
    let (q_runner, q_out) = run(QuantStrategy::Int8, &x);

    assert_eq!(f_out.len(), N);
    assert_eq!(q_out.len(), N);

    // (1) The weight was quantized and the dequant fused into MatMulDequant.
    assert_eq!(
        q_runner.dequant_matmul_fused_count(),
        1,
        "int8 weight must fuse into MatMulDequant (weight stays packed)"
    );

    // (2) The int8 logits track the f32 logits.
    let cos = cosine(&f_out, &q_out);
    let max_abs: f32 = f_out
        .iter()
        .zip(&q_out)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(
        cos >= 0.999,
        "logit cosine {cos} < 0.999 (max abs delta {max_abs})"
    );
    assert!(
        q_out.iter().all(|v| v.is_finite()),
        "int8 output not finite"
    );
}

#[test]
fn int4_inline_decodes_and_tracks_f32() {
    // The INLINE int4 pass (`quantize_weights` → `Dequantize{I4}` → MatMul) must
    // DECODE end-to-end through the real compiler + backend — not merely rewrite
    // the graph. int4 is coarser than int8 (one symmetric scale per output
    // channel, 4-bit codes), so the bound is looser, but the logits must still
    // track f32 in DIRECTION (cosine) and stay finite. Fails-without: the
    // substrate cannot decode an inline i4 weight, or the nibble packing is wrong
    // (the logits would be garbage or the execute would error).
    let x: Vec<f32> = (0..K).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
    let (_f, f_out) = run(QuantStrategy::None, &x);
    let (_q4, q4_out) = run(QuantStrategy::Int4, &x);

    assert_eq!(q4_out.len(), N);
    assert!(
        q4_out.iter().all(|v| v.is_finite()),
        "int4 output not finite — inline i4 decode failed"
    );
    let cos = cosine(&f_out, &q4_out);
    assert!(
        cos >= 0.97,
        "int4 logit cosine {cos} < 0.97 — the inline i4 decode is wrong or absent"
    );
}
