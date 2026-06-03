//! Full-model end-to-end validation against ONNX Runtime (V&V class EE).
//!
//! Compiles a complete ONNX model through hologram-ai's UOR-native pipeline,
//! runs a forward pass, and asserts the output matches **ONNX Runtime** — the
//! external authority — on the same inputs within tolerance. Unlike the
//! operator-spec harness (which checks single ops against the ONNX backend
//! node-test corpus), this exercises a whole multi-layer model
//! (`mini_transformer.onnx`: 18 nodes — MatMul, Softmax attention, Sigmoid-gated
//! FFN, residual Adds, Transposes), so it catches lowering / scheduling /
//! shape-concretization errors that only surface across a full graph.
//!
//! Gated behind the `conformance` feature (which pulls `ort` and downloads the
//! ONNX Runtime binary). Run with:
//!   cargo test -p hologram-ai-conformance --features conformance --test ort_full_model_e2e
#![cfg(feature = "conformance")]

use hologram_ai::{HoloRunner, ModelCompiler, ModelSource};
use hologram_ai_conformance::ort_runner::fixtures;
use hologram_ai_conformance::ort_runner::runner::{run_onnx_typed, OrtInputTyped};

fn f32_to_le(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn le_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn mini_transformer_matches_ort() {
    let model = fixtures::load_or_panic("mini_transformer");
    let (seq, hidden) = (4usize, 32usize);

    // Deterministic pseudo-random input X[seq, hidden] in roughly [-0.6, 0.6].
    let x: Vec<f32> = (0..seq * hidden)
        .map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.1)
        .collect();

    // ── hologram-ai: compile (concretizing seq=4) → load → forward ──────────
    let archive = ModelCompiler {
        seq_len_override: Some(seq as u64),
        ..Default::default()
    }
    .compile(ModelSource::OnnxBytes(model.clone()))
    .expect("hologram-ai compile failed");
    let mut runner = HoloRunner::from_bytes(archive.bytes).expect("load failed");
    let out = runner
        .execute(&[&f32_to_le(&x)])
        .expect("hologram-ai execute failed");
    assert_eq!(out.len(), 1, "expected one output");
    let holo = le_to_f32(&out[0].bytes);

    // ── ONNX Runtime: the external authority, same model + same input ───────
    let ort_out = run_onnx_typed(
        &model,
        vec![OrtInputTyped::F32 {
            name: "X".into(),
            shape: vec![seq, hidden],
            data: x.clone(),
        }],
    )
    .expect("ORT run failed");
    assert!(!ort_out.is_empty(), "ORT produced no f32 output");
    let reference = &ort_out[0].data;

    // ── Compare within tolerance ────────────────────────────────────────────
    assert_eq!(
        holo.len(),
        reference.len(),
        "output length: hologram-ai {} vs ORT {}",
        holo.len(),
        reference.len()
    );
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (i, (h, r)) in holo.iter().zip(reference.iter()).enumerate() {
        let diff = (h - r).abs();
        max_abs = max_abs.max(diff);
        max_rel = max_rel.max(diff / (r.abs() + 1e-6));
        // Relative tolerance: a full transformer's matmul/softmax chains differ
        // from ORT only by floating-point summation order.
        let tol = 1e-2 + 2e-3 * r.abs();
        assert!(
            diff <= tol,
            "element {i}: hologram-ai {h} vs ORT {r} (|diff| {diff} > tol {tol})"
        );
    }
    let ort_max = reference.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    println!(
        "mini_transformer vs ORT: {} elems, max |diff| = {max_abs:.2e}, max rel = {max_rel:.2e}, max |ORT| = {ort_max:.2e}",
        holo.len()
    );
}
