//! Canonical quantized weights for *arbitrary* models (UOR-native, no panic,
//! no fallback). Exercises the weight-quant configurations real models use —
//! asymmetric and unsigned zero-points, per-channel along any axis, i4, and a
//! runtime (non-constant) scale — and checks each compiles, runs, and matches
//! an independent f64 reference. Nothing here may panic or bail.

use std::collections::HashMap;

use hologram_ai::{HoloRunner, ModelCompiler, ModelSource};
use hologram_ai_common::{shape_from_concrete, AiGraph, AiNode, AiOp, AiParam, DType, TensorInfo};

const K: usize = 64;
const N: usize = 48;

fn info(dtype: DType, dims: &[u64]) -> TensorInfo {
    TensorInfo::new(dtype, shape_from_concrete(dims))
}

/// i8 weight values [K, N], a deterministic spread.
fn weights() -> Vec<i8> {
    (0..K * N)
        .map(|i| ((i as i64 * 5 % 17) - 8) as i8)
        .collect()
}

fn input() -> Vec<f32> {
    (0..K).map(|i| ((i % 11) as f32 - 5.0) * 0.2).collect()
}

/// Compile `graph`, run with `x`, return the f32 output.
fn run(graph: AiGraph, x: &[f32], expect_fused: Option<usize>) -> Vec<f32> {
    let archive = ModelCompiler::default()
        .compile(ModelSource::AiGraph(graph))
        .expect("compile must not fail for a valid quantized model");
    let mut runner = HoloRunner::from_bytes(archive.bytes).expect("load");
    if let Some(f) = expect_fused {
        assert_eq!(
            runner.dequant_matmul_fused_count(),
            f,
            "MatMulDequant fusion count"
        );
    }
    let xb: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let out = runner.execute(&[&xb]).expect("execute must not fail");
    out[0]
        .bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn assert_close(got: &[f32], reference: &[f32], what: &str) {
    assert_eq!(got.len(), reference.len(), "{what}: length");
    for (j, (&g, &r)) in got.iter().zip(reference).enumerate() {
        assert!(
            (g - r).abs() <= 1e-2 + r.abs() * 1e-3,
            "{what}: out[{j}] {g} != ref {r}"
        );
    }
}

/// Y[n] = Σ_k x[k] · deq(k, n), with `deq` the per-element dequantizer.
fn reference(x: &[f32], deq: impl Fn(usize, usize) -> f64) -> Vec<f32> {
    (0..N)
        .map(|n| (0..K).map(|k| x[k] as f64 * deq(k, n)).sum::<f64>() as f32)
        .collect()
}

fn base_graph(name: &str) -> (HashMap<u32, AiParam>, HashMap<u32, TensorInfo>) {
    let mut ti = HashMap::new();
    ti.insert(0u32, info(DType::F32, &[1, K as u64])); // X (input 0)
    let _ = name;
    (HashMap::new(), ti)
}

/// Assemble `Dequantize(W, scale, zp[, ]) → MatMul(X, W)` with the given tids.
fn assemble(
    params: HashMap<u32, AiParam>,
    ti: HashMap<u32, TensorInfo>,
    inputs: Vec<u32>,
    dq_inputs: Vec<u32>,
    axis: i64,
) -> AiGraph {
    let mut ti = ti;
    ti.insert(90, info(DType::F32, &[K as u64, N as u64])); // dequant out
    ti.insert(91, info(DType::F32, &[1, N as u64])); // Y
    AiGraph {
        name: "quant_arb".into(),
        nodes: vec![
            AiNode::new(0, AiOp::Dequantize { axis }, dq_inputs, vec![90]),
            AiNode::new(1, AiOp::MatMul, vec![0, 90], vec![91]),
        ],
        inputs,
        outputs: vec![91],
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
fn per_tensor_asymmetric_and_unsigned_zero_points() {
    let w = weights();
    let x = input();
    let scale = 0.05f32;
    // (zp_dtype, zp_value as i32, stored bytes)
    let cases: &[(DType, i32, Vec<u8>)] = &[
        (DType::INT8, 7, vec![7u8]),           // asymmetric signed
        (DType::U8, 200, vec![200u8]),         // unsigned, large zp
        (DType::INT8, -4, vec![(-4i8) as u8]), // negative zp
    ];
    for (zp_dt, zp, zp_bytes) in cases {
        let (_p, ti) = base_graph("pt");
        let mut params = HashMap::new();
        let mut ti = ti;
        ti.insert(1, info(DType::INT8, &[K as u64, N as u64]));
        ti.insert(2, info(DType::F32, &[]));
        ti.insert(3, info(*zp_dt, &[]));
        params.insert(
            1,
            AiParam::inline(w.iter().map(|&v| v as u8).collect(), ti[&1].clone()),
        );
        params.insert(
            2,
            AiParam::inline(scale.to_le_bytes().to_vec(), ti[&2].clone()),
        );
        params.insert(3, AiParam::inline(zp_bytes.clone(), ti[&3].clone()));
        let g = assemble(params, ti, vec![0], vec![1, 2, 3], -1);
        let got = run(g, &x, Some(1)); // per-tensor const ⇒ packed, fuses
        let r = reference(&x, |k, n| (w[k * N + n] as f64 - *zp as f64) * scale as f64);
        assert_close(&got, &r, &format!("per-tensor zp={zp} ({zp_dt:?})"));
    }
}

#[test]
fn per_channel_along_axis_0_and_1() {
    let w = weights();
    let x = input();
    for axis in [0i64, 1] {
        let chan = if axis == 0 { K } else { N };
        let scales: Vec<f32> = (0..chan).map(|c| 0.01 * (c as f32 + 1.0)).collect();
        let zps: Vec<i32> = (0..chan).map(|c| (c as i32 % 5) - 2).collect();
        let mut params = HashMap::new();
        let mut ti = HashMap::new();
        ti.insert(0u32, info(DType::F32, &[1, K as u64]));
        ti.insert(1, info(DType::INT8, &[K as u64, N as u64]));
        ti.insert(2, info(DType::F32, &[chan as u64]));
        ti.insert(3, info(DType::INT8, &[chan as u64]));
        params.insert(
            1,
            AiParam::inline(w.iter().map(|&v| v as u8).collect(), ti[&1].clone()),
        );
        params.insert(
            2,
            AiParam::inline(
                scales.iter().flat_map(|v| v.to_le_bytes()).collect(),
                ti[&2].clone(),
            ),
        );
        params.insert(
            3,
            AiParam::inline(zps.iter().map(|&z| z as i8 as u8).collect(), ti[&3].clone()),
        );
        let g = assemble(params, ti, vec![0], vec![1, 2, 3], axis);
        let got = run(g, &x, Some(1)); // per-channel const ⇒ packed, fuses
        let c = |k: usize, n: usize| if axis == 0 { k } else { n };
        let r = reference(&x, |k, n| {
            (w[k * N + n] as f64 - zps[c(k, n)] as f64) * scales[c(k, n)] as f64
        });
        assert_close(&got, &r, &format!("per-channel axis={axis}"));
    }
}

#[test]
fn per_channel_i4_weight() {
    // Packed i4 weight, per-channel along axis 1 (one scale per output column).
    let vals: Vec<i8> = (0..K * N).map(|i| ((i as i64 % 8) - 4) as i8).collect();
    let packed: Vec<u8> = vals
        .chunks(2)
        .map(|c| ((c.get(1).copied().unwrap_or(0) as u8 & 0xf) << 4) | (c[0] as u8 & 0xf))
        .collect();
    let x = input();
    let scales: Vec<f32> = (0..N).map(|n| 0.02 * (n as f32 + 1.0)).collect();
    let mut params = HashMap::new();
    let mut ti = HashMap::new();
    ti.insert(0u32, info(DType::F32, &[1, K as u64]));
    ti.insert(1, info(DType::INT4, &[K as u64, N as u64]));
    ti.insert(2, info(DType::F32, &[N as u64]));
    ti.insert(3, info(DType::INT8, &[N as u64]));
    params.insert(1, AiParam::inline(packed, ti[&1].clone()));
    params.insert(
        2,
        AiParam::inline(
            scales.iter().flat_map(|v| v.to_le_bytes()).collect(),
            ti[&2].clone(),
        ),
    );
    params.insert(3, AiParam::inline(vec![0u8; N], ti[&3].clone()));
    let g = assemble(params, ti, vec![0], vec![1, 2, 3], 1);
    let got = run(g, &x, Some(1));
    let r = reference(&x, |k, n| vals[k * N + n] as f64 * scales[n] as f64);
    assert_close(&got, &r, "per-channel i4");
}

#[test]
fn runtime_scale_uses_primitive_path_without_panic() {
    // Scale is a graph INPUT (not a constant) — the packed kernel can't carry
    // it, so this must take the canonical primitive path and still be correct,
    // never panicking. (No MatMulDequant fusion is expected here.)
    let w = weights();
    let x = input();
    let scale = 0.05f32;
    let mut params = HashMap::new();
    let mut ti = HashMap::new();
    ti.insert(0u32, info(DType::F32, &[1, K as u64])); // X
    ti.insert(1, info(DType::INT8, &[K as u64, N as u64])); // W (const)
    ti.insert(2, info(DType::F32, &[])); // scale — a runtime INPUT (scalar)
    params.insert(
        1,
        AiParam::inline(w.iter().map(|&v| v as u8).collect(), ti[&1].clone()),
    );
    // inputs = [X, scale]; Dequantize(W, scale) with no zp.
    let g = assemble(params, ti, vec![0, 2], vec![1, 2], -1);

    let archive = ModelCompiler::default()
        .compile(ModelSource::AiGraph(g))
        .expect("compile must not fail for a runtime-scale model");
    let mut runner = HoloRunner::from_bytes(archive.bytes).expect("load");
    let xb: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let sb: Vec<u8> = scale.to_le_bytes().to_vec();
    // input order = graph.inputs = [X(0), scale(2)].
    let out = runner.execute(&[&xb, &sb]).expect("execute must not fail");
    let got: Vec<f32> = out[0]
        .bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let r = reference(&x, |k, n| w[k * N + n] as f64 * scale as f64);
    assert_close(&got, &r, "runtime-scale primitive");
}
