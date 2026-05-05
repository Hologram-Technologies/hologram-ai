//! ORT cross-validation tests: run single-op ONNX models through ORT
//! and compare against dispatch_float().
//!
//! Gated behind `conformance` feature (requires ORT shared library at runtime).
//!
//! Run: `cargo test -p hologram-ai-conformance --features=conformance`

#![cfg(feature = "conformance")]

use hologram::hologram_exec::float_dispatch::dispatch_float;
use hologram::hologram_core::op::FloatOp;
use hologram_ai_conformance::ort_runner::{onnx_builder, runner::OrtInput};

fn dispatch_f32(op: &FloatOp, inputs: &[&[f32]]) -> Vec<f32> {
    let byte_inputs: Vec<Vec<u8>> = inputs
        .iter()
        .map(|f| bytemuck::cast_slice::<f32, u8>(f).to_vec())
        .collect();
    let byte_refs: Vec<&[u8]> = byte_inputs.iter().map(|v| v.as_slice()).collect();
    let result = dispatch_float(op, &byte_refs).expect("dispatch_float failed");
    bytemuck::cast_slice::<u8, f32>(&result).to_vec()
}

fn assert_close(actual: &[f32], expected: &[f32], atol: f32, rtol: f32, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch: {} vs {}",
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        let diff = (a - e).abs();
        let tol = atol + rtol * e.abs();
        assert!(
            diff <= tol,
            "{label}[{i}]: hologram={a}, ort={e}, diff={diff}, tol={tol}"
        );
    }
}

fn run_ort(model_bytes: &[u8], inputs: Vec<OrtInput>) -> Vec<f32> {
    hologram_ai_conformance::ort_runner::runner::run_onnx_bytes(model_bytes, inputs)
        .expect("ORT run failed")
}

fn inp(name: &str, shape: Vec<usize>, data: Vec<f32>) -> OrtInput {
    OrtInput {
        name: name.into(),
        shape,
        data,
    }
}

// ── Unary elementwise ───────────────────────────────────────────────────────

macro_rules! ort_unary_test {
    ($name:ident, $onnx_op:expr, $float_op:expr, $data:expr) => {
        #[test]
        fn $name() {
            let data: Vec<f32> = $data;
            let model = onnx_builder::unary_op($onnx_op, data.len());
            let ort_out = run_ort(&model, vec![inp("X", vec![1, data.len()], data.clone())]);
            let holo_out = dispatch_f32(&$float_op, &[&data]);
            assert_close(&holo_out, &ort_out, 1e-6, 1e-5, $onnx_op);
        }
    };
}

ort_unary_test!(
    ort_relu,
    "Relu",
    FloatOp::Relu,
    vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, -0.5, 0.5]
);
ort_unary_test!(
    ort_sigmoid,
    "Sigmoid",
    FloatOp::Sigmoid,
    vec![-3.0, -1.0, 0.0, 1.0, 3.0, 5.0, -5.0, 0.5]
);
ort_unary_test!(
    ort_tanh,
    "Tanh",
    FloatOp::Tanh,
    vec![-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, -0.5, 3.0]
);
ort_unary_test!(
    ort_abs,
    "Abs",
    FloatOp::Abs,
    vec![-3.0, -1.5, 0.0, 1.5, 3.0, -0.1, 0.1, -100.0]
);
ort_unary_test!(
    ort_neg,
    "Neg",
    FloatOp::Neg,
    vec![-3.0, 0.0, 1.5, -0.5, 100.0, -100.0, 0.001, -0.001]
);
ort_unary_test!(
    ort_sqrt,
    "Sqrt",
    FloatOp::Sqrt,
    vec![0.0, 1.0, 4.0, 9.0, 16.0, 0.25, 100.0, 0.01]
);
ort_unary_test!(
    ort_exp,
    "Exp",
    FloatOp::Exp,
    vec![0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0]
);
ort_unary_test!(
    ort_log,
    "Log",
    FloatOp::Log,
    vec![0.01, 0.1, 1.0, 2.0, 10.0, 100.0, 0.5, 3.15]
);

// ── Binary elementwise ──────────────────────────────────────────────────────

macro_rules! ort_binary_test {
    ($name:ident, $onnx_op:expr, $float_op:expr) => {
        #[test]
        fn $name() {
            let a = vec![1.0, 2.0, 3.0, -1.0, 0.0, 5.0, -3.0, 0.5];
            let b = vec![0.5, -1.0, 2.0, 3.0, 1.0, -2.0, 4.0, -0.5];
            let model = onnx_builder::binary_op($onnx_op, a.len());
            let ort_out = run_ort(
                &model,
                vec![
                    inp("A", vec![1, a.len()], a.clone()),
                    inp("B", vec![1, b.len()], b.clone()),
                ],
            );
            let holo_out = dispatch_f32(&$float_op, &[&a, &b]);
            assert_close(&holo_out, &ort_out, 1e-6, 1e-5, $onnx_op);
        }
    };
}

ort_binary_test!(ort_add, "Add", FloatOp::Add);
ort_binary_test!(ort_sub, "Sub", FloatOp::Sub);
ort_binary_test!(ort_mul, "Mul", FloatOp::Mul);
ort_binary_test!(ort_div, "Div", FloatOp::Div);

// ── Softmax ─────────────────────────────────────────────────────────────────

#[test]
fn ort_softmax() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let len = data.len();
    let model = onnx_builder::softmax(1, len);
    let ort_out = run_ort(&model, vec![inp("input", vec![1, len], data.clone())]);
    let holo_out = dispatch_f32(&FloatOp::Softmax { size: len as u32 }, &[&data]);
    assert_close(&holo_out, &ort_out, 1e-5, 1e-4, "Softmax");
}

#[test]
fn ort_softmax_multi_row() {
    let data = vec![
        1.0, 2.0, 3.0, 4.0, // row 0
        5.0, 6.0, 7.0, 8.0, // row 1
    ];
    let rows = 2;
    let cols = 4;
    let model = onnx_builder::softmax(rows, cols);
    let ort_out = run_ort(&model, vec![inp("input", vec![rows, cols], data.clone())]);
    // dispatch_float Softmax operates on total_len / size rows automatically
    let holo_out = dispatch_f32(&FloatOp::Softmax { size: cols as u32 }, &[&data]);
    assert_close(&holo_out, &ort_out, 1e-5, 1e-4, "Softmax_multi_row");
}

// ── MatMul ──────────────────────────────────────────────────────────────────

#[test]
fn ort_matmul() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
    let b = vec![
        1.0, 0.0, 0.5, -1.0, 0.0, 1.0, -0.5, 2.0, -1.0, 0.5, 1.0, 0.0,
    ]; // 3x4
    let model = onnx_builder::matmul(2, 3, 4);
    let ort_out = run_ort(
        &model,
        vec![
            inp("A", vec![2, 3], a.clone()),
            inp("B", vec![3, 4], b.clone()),
        ],
    );
    let holo_out = dispatch_f32(&FloatOp::MatMul { m: 2, k: 3, n: 4 }, &[&a, &b]);
    assert_close(&holo_out, &ort_out, 1e-4, 1e-3, "MatMul");
}

// ── Gemm ────────────────────────────────────────────────────────────────────

#[test]
fn ort_gemm() {
    let (m, k, n) = (2, 3, 4);
    let alpha: f32 = 1.0;
    let beta: f32 = 1.0;
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![
        1.0, 0.0, 0.5, -1.0, 0.0, 1.0, -0.5, 2.0, -1.0, 0.5, 1.0, 0.0,
    ];
    let c = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

    let model = onnx_builder::gemm(m, k, n, alpha, beta, false, false);
    let ort_out = run_ort(
        &model,
        vec![
            inp("A", vec![m, k], a.clone()),
            inp("B", vec![k, n], b.clone()),
            inp("C", vec![m, n], c.clone()),
        ],
    );
    let holo_out = dispatch_f32(
        &FloatOp::Gemm {
            m: m as u32,
            k: k as u32,
            n: n as u32,
            alpha: alpha.to_bits(),
            beta: beta.to_bits(),
            trans_a: false,
            trans_b: false,
            quant_b: 0,
        },
        &[&a, &b, &c],
    );
    assert_close(&holo_out, &ort_out, 1e-4, 1e-3, "Gemm");
}

#[test]
fn ort_gemm_transpose_b() {
    let (m, k, n) = (2, 3, 4);
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    // B stored as [n, k] because transB=1
    let b = vec![
        1.0, 0.0, -1.0, 0.0, 1.0, 0.5, 0.5, -0.5, 1.0, -1.0, 2.0, 0.0,
    ];
    let c = vec![0.0; m * n];

    let model = onnx_builder::gemm(m, k, n, alpha, beta, false, true);
    let ort_out = run_ort(
        &model,
        vec![
            inp("A", vec![m, k], a.clone()),
            inp("B", vec![n, k], b.clone()),
            inp("C", vec![m, n], c.clone()),
        ],
    );
    let holo_out = dispatch_f32(
        &FloatOp::Gemm {
            m: m as u32,
            k: k as u32,
            n: n as u32,
            alpha: alpha.to_bits(),
            beta: beta.to_bits(),
            trans_a: false,
            trans_b: true,
            quant_b: 0,
        },
        &[&a, &b, &c],
    );
    assert_close(&holo_out, &ort_out, 1e-4, 1e-3, "Gemm_transB");
}

// ── Composite ops (multi-node ONNX graphs) ──────────────────────────────

#[test]
fn ort_rms_norm() {
    let size = 8;
    let rows = 2;
    let eps: f32 = 1e-5;
    let x: Vec<f32> = (0..rows * size).map(|i| (i as f32 - 8.0) * 0.3).collect();
    let weight: Vec<f32> = (0..size).map(|i| 1.0 + i as f32 * 0.1).collect();

    let model = onnx_builder::rms_norm(rows, size, eps);
    let ort_out = run_ort(
        &model,
        vec![
            inp("X", vec![rows, size], x.clone()),
            inp("Weight", vec![size], weight.clone()),
        ],
    );

    let holo_out = dispatch_f32(
        &FloatOp::RmsNorm {
            size: size as u32,
            epsilon: eps.to_bits(),
        },
        &[&x, &weight],
    );

    assert_close(&holo_out, &ort_out, 1e-4, 1e-3, "RmsNorm_composite");
}

#[test]
fn ort_layer_norm() {
    let size = 8;
    let rows = 2;
    let eps: f32 = 1e-5;
    let x: Vec<f32> = (0..rows * size).map(|i| (i as f32 - 8.0) * 0.3).collect();
    let weight: Vec<f32> = (0..size).map(|i| 1.0 + i as f32 * 0.1).collect();
    let bias: Vec<f32> = (0..size).map(|i| i as f32 * 0.05).collect();

    let model = onnx_builder::layer_norm(rows, size, eps);
    let ort_out = run_ort(
        &model,
        vec![
            inp("X", vec![rows, size], x.clone()),
            inp("Weight", vec![size], weight.clone()),
            inp("Bias", vec![size], bias.clone()),
        ],
    );

    let holo_out = dispatch_f32(
        &FloatOp::LayerNorm {
            size: size as u32,
            epsilon: eps.to_bits(),
        },
        &[&x, &weight, &bias],
    );

    assert_close(&holo_out, &ort_out, 1e-4, 1e-3, "LayerNorm_composite");
}

// ── Model-level ORT validation ──────────────────────────────────────────

#[test]
fn ort_validate_identity_fixture() {
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop();
    path.pop();
    path.push("tests/fixtures/onnx/identity.onnx");
    if !path.exists() {
        eprintln!("skipping: {} not found", path.display());
        return;
    }
    let report = hologram_ai_conformance::validate_ort::validate_model_with_ort(&path);
    assert!(report.ort_ok, "ORT should load identity.onnx");
    assert!(
        report.hologram_ok,
        "hologram should compile identity.onnx: {:?}",
        report.error
    );
}

#[test]
#[ignore] // TODO(conformance): compiled_nodes == 0 — fixture model not compiling
fn ort_validate_tiny_mlp_fixture() {
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop();
    path.pop();
    path.push("tests/fixtures/onnx/tiny-mlp.onnx");
    if !path.exists() {
        eprintln!("skipping: {} not found", path.display());
        return;
    }
    let report = hologram_ai_conformance::validate_ort::validate_model_with_ort(&path);
    assert!(report.ort_ok, "ORT should load tiny-mlp.onnx");
    assert!(
        report.hologram_ok,
        "hologram should compile tiny-mlp.onnx: {:?}",
        report.error
    );
    assert!(report.compiled_nodes > 0);
}
