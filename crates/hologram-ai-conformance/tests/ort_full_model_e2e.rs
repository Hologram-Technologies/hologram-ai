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
use hologram_ai_conformance::ort_runner::runner::{run_onnx_typed, OrtInputTyped};
use hologram_ai_conformance::ort_runner::{fixtures, onnx_builder};

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

#[test]
fn batch_norm_nchw_matches_ort() {
    let model = onnx_builder::batch_norm_nchw(1, 3, 2, 2, 1e-5);
    let x: Vec<f32> = (0..12).map(|i| (i as f32) * 0.25 - 1.0).collect();

    let archive = ModelCompiler::default()
        .compile(ModelSource::OnnxBytes(model.clone()))
        .expect("hologram-ai compile failed");
    let mut runner = HoloRunner::from_bytes(archive.bytes).expect("load failed");
    let out = runner
        .execute(&[&f32_to_le(&x)])
        .expect("hologram-ai execute failed");
    assert_eq!(out.len(), 1, "expected one output");
    let holo = le_to_f32(&out[0].bytes);

    let ort_out = run_onnx_typed(
        &model,
        vec![OrtInputTyped::F32 {
            name: "X".into(),
            shape: vec![1, 3, 2, 2],
            data: x,
        }],
    )
    .expect("ORT run failed");
    assert_eq!(ort_out.len(), 1, "expected one ORT output");
    let reference = &ort_out[0].data;

    assert_eq!(
        holo.len(),
        reference.len(),
        "output length: hologram-ai {} vs ORT {}",
        holo.len(),
        reference.len()
    );
    for (i, (h, r)) in holo.iter().zip(reference.iter()).enumerate() {
        let diff = (h - r).abs();
        let tol = 1e-5 + 1e-5 * r.abs();
        assert!(
            diff <= tol,
            "element {i}: hologram-ai {h} vs ORT {r} (|diff| {diff} > tol {tol})"
        );
    }
}

#[test]
fn conv2d_with_padding_and_stride_matches_ort() {
    let model = onnx_builder::conv2d(
        onnx_builder::Conv2dSpec::new(1, 3, 7, 7, 4, 3, 3)
            .with_stride(2)
            .with_pad(1),
    );
    let x: Vec<f32> = (0..(3 * 7 * 7))
        .map(|i| ((i % 17) as f32) * 0.1 - 0.8)
        .collect();

    let archive = ModelCompiler::default()
        .compile(ModelSource::OnnxBytes(model.clone()))
        .expect("hologram-ai compile failed");
    let mut runner = HoloRunner::from_bytes(archive.bytes).expect("load failed");
    let out = runner
        .execute(&[&f32_to_le(&x)])
        .expect("hologram-ai execute failed");
    assert_eq!(out.len(), 1, "expected one output");
    let holo = le_to_f32(&out[0].bytes);

    let ort_out = run_onnx_typed(
        &model,
        vec![OrtInputTyped::F32 {
            name: "X".into(),
            shape: vec![1, 3, 7, 7],
            data: x,
        }],
    )
    .expect("ORT run failed");
    assert_eq!(ort_out.len(), 1, "expected one ORT output");
    let reference = &ort_out[0].data;

    assert_eq!(
        holo.len(),
        reference.len(),
        "output length: hologram-ai {} vs ORT {}",
        holo.len(),
        reference.len()
    );
    for (i, (h, r)) in holo.iter().zip(reference.iter()).enumerate() {
        let diff = (h - r).abs();
        let tol = 1e-4 + 1e-4 * r.abs();
        assert!(
            diff <= tol,
            "element {i}: hologram-ai {h} vs ORT {r} (|diff| {diff} > tol {tol})"
        );
    }
}

#[test]
fn relu_max_pool_with_padding_matches_ort() {
    let model = onnx_builder::relu_max_pool(2, 5, 5, 3, 2, 1);
    let x: Vec<f32> = (0..(2 * 5 * 5))
        .map(|i| ((i % 13) as f32) * 0.2 - 1.0)
        .collect();

    let archive = ModelCompiler::default()
        .compile(ModelSource::OnnxBytes(model.clone()))
        .expect("hologram-ai compile failed");
    let mut runner = HoloRunner::from_bytes(archive.bytes).expect("load failed");
    let out = runner
        .execute(&[&f32_to_le(&x)])
        .expect("hologram-ai execute failed");
    assert_eq!(out.len(), 1, "expected one output");
    let holo = le_to_f32(&out[0].bytes);

    let ort_out = run_onnx_typed(
        &model,
        vec![OrtInputTyped::F32 {
            name: "X".into(),
            shape: vec![1, 2, 5, 5],
            data: x,
        }],
    )
    .expect("ORT run failed");
    assert_eq!(ort_out.len(), 1, "expected one ORT output");
    let reference = &ort_out[0].data;

    assert_eq!(
        holo.len(),
        reference.len(),
        "output length: hologram-ai {} vs ORT {}",
        holo.len(),
        reference.len()
    );
    for (i, (h, r)) in holo.iter().zip(reference.iter()).enumerate() {
        let diff = (h - r).abs();
        let tol = 1e-6 + 1e-6 * r.abs();
        assert!(
            diff <= tol,
            "element {i}: hologram-ai {h} vs ORT {r} (|diff| {diff} > tol {tol})"
        );
    }
}

#[test]
fn relu_max_pool_without_padding_matches_ort() {
    let model = onnx_builder::relu_max_pool(2, 5, 5, 3, 2, 0);
    let x: Vec<f32> = (0..(2 * 5 * 5))
        .map(|i| ((i % 13) as f32) * 0.2 - 1.0)
        .collect();

    let archive = ModelCompiler::default()
        .compile(ModelSource::OnnxBytes(model.clone()))
        .expect("hologram-ai compile failed");
    let mut runner = HoloRunner::from_bytes(archive.bytes).expect("load failed");
    let out = runner
        .execute(&[&f32_to_le(&x)])
        .expect("hologram-ai execute failed");
    assert_eq!(out.len(), 1, "expected one output");
    let holo = le_to_f32(&out[0].bytes);

    let ort_out = run_onnx_typed(
        &model,
        vec![OrtInputTyped::F32 {
            name: "X".into(),
            shape: vec![1, 2, 5, 5],
            data: x,
        }],
    )
    .expect("ORT run failed");
    assert_eq!(ort_out.len(), 1, "expected one ORT output");
    let reference = &ort_out[0].data;

    assert_eq!(
        holo.len(),
        reference.len(),
        "output length: hologram-ai {} vs ORT {}",
        holo.len(),
        reference.len()
    );
    for (i, (h, r)) in holo.iter().zip(reference.iter()).enumerate() {
        let diff = (h - r).abs();
        let tol = 1e-6 + 1e-6 * r.abs();
        assert!(
            diff <= tol,
            "element {i}: hologram-ai {h} vs ORT {r} (|diff| {diff} > tol {tol})"
        );
    }
}

#[test]
fn batched_matmul_4d_matches_ort() {
    let (batch, heads, m, k, n) = (1usize, 12usize, 8usize, 64usize, 8usize);
    let model = onnx_builder::batched_matmul_4d(batch, heads, m, k, n);
    let a: Vec<f32> = (0..batch * heads * m * k)
        .map(|i| ((i % 97) as f32 - 48.0) * 0.03125)
        .collect();
    let b: Vec<f32> = (0..batch * heads * k * n)
        .map(|i| ((i % 89) as f32 - 44.0) * 0.041)
        .collect();

    let archive = ModelCompiler::default()
        .compile(ModelSource::OnnxBytes(model.clone()))
        .expect("hologram-ai compile failed");
    let mut runner = HoloRunner::from_bytes(archive.bytes).expect("load failed");
    let out = runner
        .execute(&[&f32_to_le(&a), &f32_to_le(&b)])
        .expect("hologram-ai execute failed");
    assert_eq!(out.len(), 1, "expected one output");
    let holo = le_to_f32(&out[0].bytes);

    let ort_out = run_onnx_typed(
        &model,
        vec![
            OrtInputTyped::F32 {
                name: "A".into(),
                shape: vec![batch, heads, m, k],
                data: a,
            },
            OrtInputTyped::F32 {
                name: "B".into(),
                shape: vec![batch, heads, k, n],
                data: b,
            },
        ],
    )
    .expect("ORT run failed");
    assert_eq!(ort_out.len(), 1, "expected one ORT output");
    let reference = &ort_out[0].data;

    assert_eq!(
        holo.len(),
        reference.len(),
        "output length: hologram-ai {} vs ORT {}",
        holo.len(),
        reference.len()
    );
    for (i, (h, r)) in holo.iter().zip(reference.iter()).enumerate() {
        let diff = (h - r).abs();
        let tol = 1e-4 + 1e-4 * r.abs();
        assert!(
            diff <= tol,
            "element {i}: hologram-ai {h} vs ORT {r} (|diff| {diff} > tol {tol})"
        );
    }
}

#[test]
fn mini_vision_classifier_matches_ort() {
    let model = onnx_builder::mini_vision_classifier(3, 8, 8, 5, 7);
    let x: Vec<f32> = (0..(3 * 8 * 8))
        .map(|i| ((i % 29) as f32) * 0.05 - 0.7)
        .collect();

    let archive = ModelCompiler::default()
        .compile(ModelSource::OnnxBytes(model.clone()))
        .expect("hologram-ai compile failed");
    let mut runner = HoloRunner::from_bytes(archive.bytes).expect("load failed");
    let out = runner
        .execute(&[&f32_to_le(&x)])
        .expect("hologram-ai execute failed");
    assert_eq!(out.len(), 1, "expected one output");
    let holo = le_to_f32(&out[0].bytes);

    let ort_out = run_onnx_typed(
        &model,
        vec![OrtInputTyped::F32 {
            name: "X".into(),
            shape: vec![1, 3, 8, 8],
            data: x,
        }],
    )
    .expect("ORT run failed");
    assert_eq!(ort_out.len(), 1, "expected one ORT output");
    let reference = &ort_out[0].data;

    assert_eq!(
        holo.len(),
        reference.len(),
        "output length: hologram-ai {} vs ORT {}",
        holo.len(),
        reference.len()
    );
    for (i, (h, r)) in holo.iter().zip(reference.iter()).enumerate() {
        let diff = (h - r).abs();
        let tol = 1e-4 + 1e-4 * r.abs();
        assert!(
            diff <= tol,
            "element {i}: hologram-ai {h} vs ORT {r} (|diff| {diff} > tol {tol})"
        );
    }
}
