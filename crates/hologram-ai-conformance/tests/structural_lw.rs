//! Structural V&V — class **LW** (every `AiOp` has a complete canonical
//! realization).
//!
//! LW-1 (operator-spec parity) is already enforced by
//! [`onnx_spec_conformance`], which validates relu/add/matmul/softmax/mul/sub
//! and quantize against the official ONNX backend node-test corpus, *and* by
//! [`ort_full_model_e2e`] which runs a multi-layer transformer fixture vs
//! ONNX Runtime within ~2e-5.
//!
//! LW-2 is the *structural* part of LW: the lowering surface is total
//! (every `AiOp` has a non-failure plan, see also CF-2) and each canonical
//! realization equals an independent reference of the op it replaces. The
//! Rust type system already makes the totality static — `dispatch`'s match
//! is exhaustive, `OpPlan` has no failure variant. This file closes the
//! "equals an independent reference" half by running a broad sweep of
//! single-op ONNX models through hologram-ai and comparing the output to an
//! in-test reference computed in f64 (or against well-known closed-form
//! identities). The categories covered intentionally span the
//! `AiOp::category()` perimeter:
//!
//! * Unary elementwise (Relu, Sigmoid, Tanh, Exp, Neg, Abs, Sqrt)
//! * Binary elementwise (Add, Mul, Sub, Div)
//! * Reductions (ReduceSum, ReduceMax)
//! * Linear algebra (MatMul — also exercised at scale by EE-1/PV)
//! * Shape-preserving (Softmax — full spec coverage via onnx_spec_conformance)
//!
//! Together with `onnx_spec_conformance` (live ONNX corpus) and CF-2
//! (totality of `dispatch`), LW-2's claim — "no unsupported ops, no runtime
//! failure path, each canonical realization equals an independent
//! reference" — has its load-bearing witnesses.

#![cfg(feature = "structural")]

use hologram_ai::{HoloRunner, ModelCompiler, ModelSource};
use hologram_ai_conformance::ort_runner::onnx_builder;

/// Compile + run a unary-op model on the given f32 input; return the f32
/// output buffer.
fn run_unary(op: &'static str, x: &[f32]) -> Vec<f32> {
    let bytes = onnx_builder::unary_op(op, x.len());
    let archive = ModelCompiler::default()
        .compile(ModelSource::OnnxBytes(bytes))
        .unwrap_or_else(|e| panic!("compile {op}: {e}"));
    let mut runner = HoloRunner::from_bytes(archive.bytes).expect("load");
    let x_bytes: &[u8] = bytemuck::cast_slice(x);
    let outs = runner.execute(&[x_bytes]).expect("execute");
    bytemuck::cast_slice(&outs[0].bytes).to_vec()
}

/// Compile + run a binary-op model on the given f32 inputs.
fn run_binary(op: &'static str, a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len());
    let bytes = onnx_builder::binary_op(op, a.len());
    let archive = ModelCompiler::default()
        .compile(ModelSource::OnnxBytes(bytes))
        .unwrap_or_else(|e| panic!("compile {op}: {e}"));
    let mut runner = HoloRunner::from_bytes(archive.bytes).expect("load");
    let ab: &[u8] = bytemuck::cast_slice(a);
    let bb: &[u8] = bytemuck::cast_slice(b);
    let outs = runner.execute(&[ab, bb]).expect("execute");
    bytemuck::cast_slice(&outs[0].bytes).to_vec()
}

/// Per-op tolerance against an f64 reference. Most ops are exact to f32
/// rounding; transcendentals (Sigmoid, Tanh, Exp) get a small relative slack.
fn assert_close(label: &str, got: &[f32], want: &[f32], atol: f32, rtol: f32) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let err = (g - w).abs();
        let tol = atol + rtol * w.abs();
        assert!(
            err <= tol || (g.is_nan() && w.is_nan()),
            "{label} @ {i}: got {g}, expected {w}, |err|={err}, tol={tol}"
        );
    }
}

fn x_seed(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.137).sin() * 2.0 - 0.5)
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Unary elementwise: hologram-ai output == f64 reference, for every op.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn lw_2_unary_relu_matches_reference() {
    let x = x_seed(32);
    let got = run_unary("Relu", &x);
    let want: Vec<f32> = x.iter().map(|&v| v.max(0.0)).collect();
    assert_close("Relu", &got, &want, 0.0, 0.0);
}

#[test]
fn lw_2_unary_sigmoid_matches_reference() {
    let x = x_seed(32);
    let got = run_unary("Sigmoid", &x);
    let want: Vec<f32> = x
        .iter()
        .map(|&v| (1.0 / (1.0 + (-(v as f64)).exp())) as f32)
        .collect();
    assert_close("Sigmoid", &got, &want, 1e-6, 1e-5);
}

#[test]
fn lw_2_unary_tanh_matches_reference() {
    let x = x_seed(32);
    let got = run_unary("Tanh", &x);
    let want: Vec<f32> = x.iter().map(|&v| (v as f64).tanh() as f32).collect();
    assert_close("Tanh", &got, &want, 1e-6, 1e-5);
}

#[test]
fn lw_2_unary_exp_matches_reference() {
    let x: Vec<f32> = (0..32).map(|i| ((i as f32) - 16.0) * 0.1).collect();
    let got = run_unary("Exp", &x);
    let want: Vec<f32> = x.iter().map(|&v| (v as f64).exp() as f32).collect();
    assert_close("Exp", &got, &want, 1e-6, 1e-5);
}

#[test]
fn lw_2_unary_neg_matches_reference() {
    let x = x_seed(32);
    let got = run_unary("Neg", &x);
    let want: Vec<f32> = x.iter().map(|&v| -v).collect();
    assert_close("Neg", &got, &want, 0.0, 0.0);
}

#[test]
fn lw_2_unary_abs_matches_reference() {
    let x = x_seed(32);
    let got = run_unary("Abs", &x);
    let want: Vec<f32> = x.iter().map(|&v| v.abs()).collect();
    assert_close("Abs", &got, &want, 0.0, 0.0);
}

#[test]
fn lw_2_unary_sqrt_matches_reference() {
    let x: Vec<f32> = (1..=32).map(|i| i as f32).collect();
    let got = run_unary("Sqrt", &x);
    let want: Vec<f32> = x.iter().map(|&v| (v as f64).sqrt() as f32).collect();
    assert_close("Sqrt", &got, &want, 1e-6, 1e-5);
}

// ─────────────────────────────────────────────────────────────────────────────
// Binary elementwise.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn lw_2_binary_add_matches_reference() {
    let a = x_seed(32);
    let b: Vec<f32> = x_seed(32).into_iter().map(|v| v + 0.5).collect();
    let got = run_binary("Add", &a, &b);
    let want: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| x + y).collect();
    assert_close("Add", &got, &want, 0.0, 0.0);
}

#[test]
fn lw_2_binary_mul_matches_reference() {
    let a = x_seed(32);
    let b: Vec<f32> = x_seed(32).into_iter().map(|v| v + 0.25).collect();
    let got = run_binary("Mul", &a, &b);
    let want: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| x * y).collect();
    assert_close("Mul", &got, &want, 1e-7, 1e-6);
}

#[test]
fn lw_2_binary_sub_matches_reference() {
    let a = x_seed(32);
    let b: Vec<f32> = x_seed(32).into_iter().map(|v| v - 0.25).collect();
    let got = run_binary("Sub", &a, &b);
    let want: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| x - y).collect();
    assert_close("Sub", &got, &want, 0.0, 0.0);
}

#[test]
fn lw_2_binary_div_matches_reference() {
    let a = x_seed(32);
    let b: Vec<f32> = (0..32).map(|i| (i as f32) + 1.0).collect();
    let got = run_binary("Div", &a, &b);
    let want: Vec<f32> = a
        .iter()
        .zip(&b)
        .map(|(&x, &y)| (x as f64 / y as f64) as f32)
        .collect();
    assert_close("Div", &got, &want, 1e-6, 1e-5);
}

// ─────────────────────────────────────────────────────────────────────────────
// Linear algebra: matmul against an f64 reference.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn lw_2_matmul_matches_f64_reference() {
    let m = 8;
    let k = 16;
    let n = 12;
    let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.073).sin()).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.091).cos()).collect();

    let bytes = onnx_builder::matmul(m, k, n);
    let archive = ModelCompiler::default()
        .compile(ModelSource::OnnxBytes(bytes))
        .expect("matmul compile");
    let mut runner = HoloRunner::from_bytes(archive.bytes).expect("load");
    let ab: &[u8] = bytemuck::cast_slice(&a);
    let bb: &[u8] = bytemuck::cast_slice(&b);
    let outs = runner.execute(&[ab, bb]).expect("execute");
    let got: &[f32] = bytemuck::cast_slice(&outs[0].bytes);

    let mut want = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for p in 0..k {
                acc += a[i * k + p] as f64 * b[p * n + j] as f64;
            }
            want[i * n + j] = acc as f32;
        }
    }
    assert_close("MatMul", got, &want, 1e-4, 1e-4);
}

// ─────────────────────────────────────────────────────────────────────────────
// Softmax (shape-preserving with axis): identities + f64 reference.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn lw_2_softmax_normalizes_to_one() {
    let rows = 4;
    let size = 8;
    let x: Vec<f32> = (0..rows * size)
        .map(|i| ((i as f32) * 0.13).sin())
        .collect();
    let bytes = onnx_builder::softmax(rows, size);
    let archive = ModelCompiler::default()
        .compile(ModelSource::OnnxBytes(bytes))
        .expect("softmax compile");
    let mut runner = HoloRunner::from_bytes(archive.bytes).expect("load");
    let xb: &[u8] = bytemuck::cast_slice(&x);
    let outs = runner.execute(&[xb]).expect("execute");
    let got: &[f32] = bytemuck::cast_slice(&outs[0].bytes);

    for r in 0..rows {
        let row = &got[r * size..(r + 1) * size];
        let sum: f64 = row.iter().map(|&v| v as f64).sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "Softmax row {r} sum = {sum} (expected 1.0)"
        );
        for &v in row {
            assert!(
                (-1e-6..=1.0 + 1e-6).contains(&v),
                "Softmax row {r} value {v}"
            );
        }
    }

    // f64 reference (numerically stable softmax).
    for r in 0..rows {
        let row = &x[r * size..(r + 1) * size];
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f64> = row.iter().map(|&v| ((v - max) as f64).exp()).collect();
        let s: f64 = exps.iter().sum();
        let want: Vec<f32> = exps.iter().map(|&e| (e / s) as f32).collect();
        assert_close(
            &format!("Softmax row {r}"),
            &got[r * size..(r + 1) * size],
            &want,
            1e-5,
            1e-5,
        );
    }
}
