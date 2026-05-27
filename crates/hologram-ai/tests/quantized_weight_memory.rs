//! Quantized-weight runtime memory (V&V class QZ + the memory-efficiency goal).
//!
//! The realized-information-content principle applied to weights: a weight's
//! realized domain is its quantization level (i8 = 1 byte/param, i4 = ½), not
//! the f32 storage width. hologram-ai carries quantized weights **packed** and
//! emits `Dequantize → MatMul`; hologram fuses that into `MatMulDequant`, which
//! dequantizes the packed weight **in-register** — the dense f32 weight is
//! never materialized. So the runtime weight footprint is the packed size.
//!
//! This test builds the same linear layer two ways — i8 weight + Dequantize vs
//! a plain f32 weight — and shows: (1) the fusion fires, (2) the i8 version's
//! resident pool is ~4× smaller, and (3) the output is numerically correct.

use std::collections::HashMap;

use hologram_ai::{HoloRunner, ModelCompiler, ModelSource};
use hologram_ai_common::{shape_from_concrete, AiGraph, AiNode, AiOp, AiParam, DType, TensorInfo};

const K: usize = 256; // input dim
const N: usize = 256; // output dim
const SCALE: f32 = 0.05;

fn info(dtype: DType, dims: &[u64]) -> TensorInfo {
    TensorInfo::new(dtype, shape_from_concrete(dims))
}

/// The reference weight matrix (row-major [K, N]) as small signed integers,
/// and its dequantized f32 form `w * SCALE` (symmetric, zero-point 0).
fn weight_i8() -> Vec<i8> {
    (0..K * N).map(|i| ((i as i64 % 7) - 3) as i8).collect()
}

/// `Dequantize(W_i8, scale, zp=0) → MatMul(X, W)`. The weight is a packed i8
/// constant; X is the graph input.
fn quantized_graph() -> AiGraph {
    let w = weight_i8();
    let w_bytes: Vec<u8> = w.iter().map(|&v| v as u8).collect();

    let mut params: HashMap<u32, AiParam> = HashMap::new();
    let mut ti: HashMap<u32, TensorInfo> = HashMap::new();

    // tids: 0=X(input), 1=W_i8(const), 2=scale(const), 3=zp(const),
    //       4=W_f32(dequant out), 5=Y(matmul out)
    ti.insert(0, info(DType::F32, &[1, K as u64]));
    ti.insert(1, info(DType::INT8, &[K as u64, N as u64]));
    ti.insert(2, info(DType::F32, &[]));
    ti.insert(3, info(DType::INT8, &[]));
    ti.insert(4, info(DType::F32, &[K as u64, N as u64]));
    ti.insert(5, info(DType::F32, &[1, N as u64]));

    params.insert(1, AiParam::inline(w_bytes, ti[&1].clone()));
    params.insert(2, AiParam::inline(SCALE.to_le_bytes().to_vec(), ti[&2].clone()));
    params.insert(3, AiParam::inline(vec![0u8], ti[&3].clone()));

    AiGraph {
        name: "quant_linear".into(),
        nodes: vec![
            AiNode::new(0, AiOp::Dequantize { axis: -1 }, vec![1, 2, 3], vec![4]),
            AiNode::new(1, AiOp::MatMul, vec![0, 4], vec![5]),
        ],
        inputs: vec![0],
        outputs: vec![5],
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

/// The same layer with a plain dense f32 weight (`w * SCALE`) — the baseline.
fn f32_graph() -> AiGraph {
    let w: Vec<f32> = weight_i8().iter().map(|&v| v as f32 * SCALE).collect();
    let w_bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();

    let mut params: HashMap<u32, AiParam> = HashMap::new();
    let mut ti: HashMap<u32, TensorInfo> = HashMap::new();
    ti.insert(0, info(DType::F32, &[1, K as u64]));
    ti.insert(1, info(DType::F32, &[K as u64, N as u64]));
    ti.insert(2, info(DType::F32, &[1, N as u64]));
    params.insert(1, AiParam::inline(w_bytes, ti[&1].clone()));

    AiGraph {
        name: "f32_linear".into(),
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

fn run(graph: AiGraph, x: &[f32]) -> (HoloRunner, Vec<f32>) {
    let archive = ModelCompiler::default()
        .compile(ModelSource::AiGraph(graph))
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

#[test]
fn quantized_weight_stays_packed_and_is_correct() {
    let x: Vec<f32> = (0..K).map(|i| ((i % 5) as f32 - 2.0) * 0.3).collect();

    // Reference: Y[j] = Σ_i x[i] · (w[i,j] · SCALE), computed in f64.
    let w = weight_i8();
    let reference: Vec<f32> = (0..N)
        .map(|j| {
            let acc: f64 = (0..K)
                .map(|i| x[i] as f64 * (w[i * N + j] as f64 * SCALE as f64))
                .sum();
            acc as f32
        })
        .collect();

    let (q_runner, q_out) = run(quantized_graph(), &x);
    let (f_runner, f_out) = run(f32_graph(), &x);

    // (1) The dequant→matmul fused: the weight is read packed, in-register.
    assert_eq!(
        q_runner.dequant_matmul_fused_count(),
        1,
        "Dequantize→MatMul must fuse into MatMulDequant (weight stays packed)"
    );

    // (2) Correctness: quantized path matches the f64 reference (and the f32 path).
    for (j, (&q, &r)) in q_out.iter().zip(reference.iter()).enumerate() {
        assert!((q - r).abs() <= 1e-3 + r.abs() * 1e-4, "quant out[{j}] {q} != ref {r}");
    }
    for (j, (&q, &f)) in q_out.iter().zip(f_out.iter()).enumerate() {
        assert!((q - f).abs() <= 1e-4, "quant[{j}] {q} != f32 baseline {f}");
    }

    // (3) Memory: the i8-weight pool is ~4× smaller than the f32 baseline —
    // the dense f32 weight is never materialized.
    let q_bytes = q_runner.resident_bytes();
    let f_bytes = f_runner.resident_bytes();
    println!(
        "linear [{K}×{N}]: i8-weight resident {q_bytes} B vs f32 resident {f_bytes} B  ({:.1}× smaller)",
        f_bytes as f64 / q_bytes.max(1) as f64
    );
    // i8 weight is 1 byte/param vs f32's 4; allow slack for the tiny X /
    // intermediates, but the reduction must be substantial (≥3×).
    assert!(
        q_bytes * 3 <= f_bytes,
        "quantized weight footprint ({q_bytes} B) should be ~4× under f32 ({f_bytes} B)"
    );
}

/// Per-channel (per-output-column) i8 weight: `Dequantize` carries the exact
/// ONNX `axis`, and a distinct scale per column. Verifies the per-channel path
/// is correct (no axis guessing) and still fuses + stays packed.
fn per_channel_graph() -> AiGraph {
    let w = weight_i8();
    let w_bytes: Vec<u8> = w.iter().map(|&v| v as u8).collect();
    let scales: Vec<f32> = (0..N).map(|n| 0.01 * (n as f32 + 1.0)).collect();
    let scale_bytes: Vec<u8> = scales.iter().flat_map(|v| v.to_le_bytes()).collect();

    let mut params: HashMap<u32, AiParam> = HashMap::new();
    let mut ti: HashMap<u32, TensorInfo> = HashMap::new();
    ti.insert(0, info(DType::F32, &[1, K as u64])); // X
    ti.insert(1, info(DType::INT8, &[K as u64, N as u64])); // W_i8
    ti.insert(2, info(DType::F32, &[N as u64])); // scale vector (per column)
    ti.insert(3, info(DType::INT8, &[N as u64])); // zero-point vector (zeros)
    ti.insert(4, info(DType::F32, &[K as u64, N as u64])); // W_f32 (dequant out)
    ti.insert(5, info(DType::F32, &[1, N as u64])); // Y
    params.insert(1, AiParam::inline(w_bytes, ti[&1].clone()));
    params.insert(2, AiParam::inline(scale_bytes, ti[&2].clone()));
    params.insert(3, AiParam::inline(vec![0u8; N], ti[&3].clone()));

    AiGraph {
        name: "per_channel_linear".into(),
        // axis = 1 ⇒ per-output-channel (one scale per column of W[K, N]).
        nodes: vec![
            AiNode::new(0, AiOp::Dequantize { axis: 1 }, vec![1, 2, 3], vec![4]),
            AiNode::new(1, AiOp::MatMul, vec![0, 4], vec![5]),
        ],
        inputs: vec![0],
        outputs: vec![5],
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

#[test]
fn per_channel_quantized_weight_is_correct() {
    let x: Vec<f32> = (0..K).map(|i| ((i % 5) as f32 - 2.0) * 0.3).collect();
    let w = weight_i8();
    let scales: Vec<f32> = (0..N).map(|n| 0.01 * (n as f32 + 1.0)).collect();
    // Y[n] = Σ_k x[k] · (w[k,n] · scale[n]).
    let reference: Vec<f32> = (0..N)
        .map(|n| {
            let acc: f64 = (0..K)
                .map(|k| x[k] as f64 * (w[k * N + n] as f64 * scales[n] as f64))
                .sum();
            acc as f32
        })
        .collect();

    let (runner, out) = run(per_channel_graph(), &x);
    assert_eq!(
        runner.dequant_matmul_fused_count(),
        1,
        "per-channel Dequantize→MatMul must fuse (weight stays packed)"
    );
    for (n, (&o, &r)) in out.iter().zip(reference.iter()).enumerate() {
        assert!(
            (o - r).abs() <= 1e-3 + r.abs() * 1e-3,
            "per-channel out[{n}] {o} != ref {r}"
        );
    }
}

/// Packed 4-bit (i4) weight: two nibbles per byte, sign-extended. Verifies true
/// ½-byte packing — the i4 weight is `K·N/2` bytes (8× under dense f32) — and
/// that the dequant is numerically correct.
fn i4_graph() -> (AiGraph, Vec<i8>) {
    // Small signed values in [-4, 3] so each fits a signed 4-bit nibble.
    let vals: Vec<i8> = (0..K * N).map(|i| ((i as i64 % 8) - 4) as i8).collect();
    // Pack: element 2k → low nibble, 2k+1 → high nibble.
    let packed: Vec<u8> = vals
        .chunks(2)
        .map(|c| {
            let lo = (c[0] as u8) & 0x0f;
            let hi = (c.get(1).copied().unwrap_or(0) as u8) & 0x0f;
            (hi << 4) | lo
        })
        .collect();

    let mut params: HashMap<u32, AiParam> = HashMap::new();
    let mut ti: HashMap<u32, TensorInfo> = HashMap::new();
    ti.insert(0, info(DType::F32, &[1, K as u64]));
    ti.insert(1, info(DType::INT4, &[K as u64, N as u64])); // packed i4 weight
    ti.insert(2, info(DType::F32, &[]));
    ti.insert(3, info(DType::INT8, &[]));
    ti.insert(4, info(DType::F32, &[K as u64, N as u64]));
    ti.insert(5, info(DType::F32, &[1, N as u64]));
    params.insert(1, AiParam::inline(packed, ti[&1].clone()));
    params.insert(2, AiParam::inline(SCALE.to_le_bytes().to_vec(), ti[&2].clone()));
    params.insert(3, AiParam::inline(vec![0u8], ti[&3].clone()));

    let g = AiGraph {
        name: "i4_linear".into(),
        nodes: vec![
            AiNode::new(0, AiOp::Dequantize { axis: -1 }, vec![1, 2, 3], vec![4]),
            AiNode::new(1, AiOp::MatMul, vec![0, 4], vec![5]),
        ],
        inputs: vec![0],
        outputs: vec![5],
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
    };
    (g, vals)
}

#[test]
fn i4_weight_packs_to_half_byte_and_is_correct() {
    let x: Vec<f32> = (0..K).map(|i| ((i % 5) as f32 - 2.0) * 0.3).collect();
    let (graph, vals) = i4_graph();
    let reference: Vec<f32> = (0..N)
        .map(|n| {
            let acc: f64 = (0..K)
                .map(|k| x[k] as f64 * (vals[k * N + n] as f64 * SCALE as f64))
                .sum();
            acc as f32
        })
        .collect();

    let (runner, out) = run(graph, &x);
    assert_eq!(runner.dequant_matmul_fused_count(), 1, "i4 Dequantize→MatMul must fuse");
    for (n, (&o, &r)) in out.iter().zip(reference.iter()).enumerate() {
        assert!((o - r).abs() <= 1e-3 + r.abs() * 1e-3, "i4 out[{n}] {o} != ref {r}");
    }

    // The i4 weight is K·N/2 bytes resident vs f32's K·N·4 → ~8× smaller.
    let (f_runner, _) = run(f32_graph(), &x);
    let q_bytes = runner.resident_bytes();
    let f_bytes = f_runner.resident_bytes();
    println!(
        "linear [{K}×{N}]: i4-weight resident {q_bytes} B vs f32 {f_bytes} B  ({:.1}× smaller)",
        f_bytes as f64 / q_bytes.max(1) as f64
    );
    assert!(
        q_bytes * 6 <= f_bytes,
        "i4 weight footprint ({q_bytes} B) should be ~8× under f32 ({f_bytes} B)"
    );
}
