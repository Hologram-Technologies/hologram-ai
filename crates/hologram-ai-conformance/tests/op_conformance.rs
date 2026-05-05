//! Cross-validation tests: dispatch_float vs pure-Rust reference implementations.
//!
//! These tests run each complex FloatOp through both `dispatch_float()` and the
//! corresponding reference implementation in `hologram_ai_conformance::reference`,
//! then compare outputs using per-op tolerances.

use hologram::hologram_exec::float_dispatch::dispatch_float;
use hologram::hologram_core::op::FloatOp;
use hologram_ai_conformance::reference;
use hologram_ai_conformance::tolerance::{compare_outputs, tolerance_for, Tolerance};

// ── Helpers ─────────────────────────────────────────────────────────────────

fn f32_bytes(vals: &[f32]) -> Vec<u8> {
    bytemuck::cast_slice(vals).to_vec()
}

fn run_dispatch(op: &FloatOp, inputs: &[&[u8]]) -> Vec<f32> {
    let result = dispatch_float(op, inputs).expect("dispatch_float failed");
    bytemuck::cast_slice::<u8, f32>(&result).to_vec()
}

fn assert_close(actual: &[f32], expected: &[f32], tol: Tolerance, op_name: &str) {
    let result = compare_outputs(actual, expected, tol);
    assert!(
        result.passed,
        "{op_name}: {}\n  actual[..8]:   {:?}\n  expected[..8]: {:?}",
        result.message,
        &actual[..actual.len().min(8)],
        &expected[..expected.len().min(8)],
    );
}

// ── Softmax ─────────────────────────────────────────────────────────────────

#[test]
fn conformance_softmax() {
    let input: Vec<f32> = (0..20).map(|i| (i as f32 - 10.0) * 0.3).collect();
    let size = 5;
    let op = FloatOp::Softmax { size };
    let tol = tolerance_for(&op);

    let actual = run_dispatch(&op, &[&f32_bytes(&input)]);
    let expected = reference::softmax(&input, size as usize);

    assert_close(&actual, &expected, tol, "Softmax");
}

#[test]
fn conformance_softmax_large_values() {
    // Numerical stability: large values that would overflow naive exp
    let input = vec![1000.0, 1001.0, 999.0, 1002.0];
    let size = 4;
    let op = FloatOp::Softmax { size };
    let tol = tolerance_for(&op);

    let actual = run_dispatch(&op, &[&f32_bytes(&input)]);
    let expected = reference::softmax(&input, size as usize);

    assert_close(&actual, &expected, tol, "Softmax-large");
    // Should not produce NaN
    assert!(actual.iter().all(|v| !v.is_nan()), "NaN in softmax output");
}

// ── LogSoftmax ──────────────────────────────────────────────────────────────

#[test]
fn conformance_log_softmax() {
    let input: Vec<f32> = (0..12).map(|i| (i as f32 - 6.0) * 0.5).collect();
    let size = 4;
    let op = FloatOp::LogSoftmax { size };
    let tol = tolerance_for(&op);

    let actual = run_dispatch(&op, &[&f32_bytes(&input)]);
    let expected = reference::log_softmax(&input, size as usize);

    assert_close(&actual, &expected, tol, "LogSoftmax");
    // All log-softmax outputs should be <= 0
    assert!(
        actual.iter().all(|&v| v <= 0.0),
        "LogSoftmax should be <= 0"
    );
}

// ── RmsNorm ─────────────────────────────────────────────────────────────────

#[test]
fn conformance_rms_norm() {
    let size = 16;
    let input: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
    let weight: Vec<f32> = (0..size).map(|i| 1.0 + i as f32 * 0.01).collect();
    let epsilon = 1e-5_f32;
    let op = FloatOp::RmsNorm {
        size: size as u32,
        epsilon: epsilon.to_bits(),
    };
    let tol = tolerance_for(&op);

    let actual = run_dispatch(&op, &[&f32_bytes(&input), &f32_bytes(&weight)]);
    let expected = reference::rms_norm(&input, &weight, size, epsilon);

    assert_close(&actual, &expected, tol, "RmsNorm");
}

#[test]
fn conformance_rms_norm_uniform() {
    // When all inputs are the same, RMS = |x|, so output = sign(x) * weight
    let size = 16;
    let val = 2.0_f32;
    let input = vec![val; size * 2];
    let weight = vec![1.0_f32; size];
    let epsilon = 1e-6_f32;
    let op = FloatOp::RmsNorm {
        size: size as u32,
        epsilon: epsilon.to_bits(),
    };
    let tol = tolerance_for(&op);

    let actual = run_dispatch(&op, &[&f32_bytes(&input), &f32_bytes(&weight)]);
    let expected = reference::rms_norm(&input, &weight, size, epsilon);

    assert_close(&actual, &expected, tol, "RmsNorm-uniform");
}

// ── LayerNorm ───────────────────────────────────────────────────────────────

#[test]
fn conformance_layer_norm() {
    let size = 16;
    let input: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.2).collect();
    let weight: Vec<f32> = (0..size).map(|i| 1.0 + i as f32 * 0.02).collect();
    let bias: Vec<f32> = (0..size).map(|i| i as f32 * 0.01).collect();
    let epsilon = 1e-5_f32;
    let op = FloatOp::LayerNorm {
        size: size as u32,
        epsilon: epsilon.to_bits(),
    };
    let tol = tolerance_for(&op);

    let actual = run_dispatch(
        &op,
        &[&f32_bytes(&input), &f32_bytes(&weight), &f32_bytes(&bias)],
    );
    let expected = reference::layer_norm(&input, &weight, &bias, size, epsilon);

    assert_close(&actual, &expected, tol, "LayerNorm");
}

#[test]
fn conformance_layer_norm_zero_mean_property() {
    // With weight=1, bias=0: output should have zero mean (within tolerance)
    let size = 16;
    let input: Vec<f32> = (0..size).map(|i| i as f32 * 3.7 - 12.0).collect();
    let weight = vec![1.0_f32; size];
    let bias = vec![0.0_f32; size];
    let epsilon = 1e-5_f32;
    let op = FloatOp::LayerNorm {
        size: size as u32,
        epsilon: epsilon.to_bits(),
    };

    let actual = run_dispatch(
        &op,
        &[&f32_bytes(&input), &f32_bytes(&weight), &f32_bytes(&bias)],
    );
    let mean: f32 = actual.iter().sum::<f32>() / actual.len() as f32;
    assert!(
        mean.abs() < 1e-4,
        "LayerNorm output should have ~zero mean, got {mean}"
    );
}

// ── MatMul ──────────────────────────────────────────────────────────────────

#[test]
fn conformance_matmul() {
    let m = 4;
    let k = 3;
    let n = 5;
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32 - 6.0) * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32 - 7.0) * 0.2).collect();
    let op = FloatOp::MatMul {
        m: m as u32,
        k: k as u32,
        n: n as u32,
    };
    let tol = tolerance_for(&op);

    let actual = run_dispatch(&op, &[&f32_bytes(&a), &f32_bytes(&b)]);
    let expected = reference::matmul(&a, &b, m, k, n);

    assert_close(&actual, &expected, tol, "MatMul");
}

#[test]
fn conformance_matmul_square() {
    let n = 8;
    let a: Vec<f32> = (0..n * n).map(|i| (i as f32).sin()).collect();
    let b: Vec<f32> = (0..n * n).map(|i| (i as f32 * 0.7).cos()).collect();
    let op = FloatOp::MatMul {
        m: n as u32,
        k: n as u32,
        n: n as u32,
    };
    let tol = tolerance_for(&op);

    let actual = run_dispatch(&op, &[&f32_bytes(&a), &f32_bytes(&b)]);
    let expected = reference::matmul(&a, &b, n, n, n);

    assert_close(&actual, &expected, tol, "MatMul-square");
}

// ── Gemm ────────────────────────────────────────────────────────────────────

#[test]
fn conformance_gemm_identity() {
    let m = 3;
    let k = 4;
    let n = 2;
    let alpha = 1.0_f32;
    let beta = 0.0_f32;
    let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.5).collect();
    let b: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.3).collect();
    let op = FloatOp::Gemm {
        m: m as u32,
        k: k as u32,
        n: n as u32,
        alpha: alpha.to_bits(),
        beta: beta.to_bits(),
        trans_a: false,
        trans_b: false,
        quant_b: 0,
    };
    let tol = tolerance_for(&op);

    let actual = run_dispatch(&op, &[&f32_bytes(&a), &f32_bytes(&b)]);
    let expected = reference::gemm(reference::GemmParams::new(&a, &b, m, k, n).with_alpha(alpha));

    assert_close(&actual, &expected, tol, "Gemm-identity");
}

#[test]
fn conformance_gemm_alpha_beta() {
    let m = 3;
    let k = 4;
    let n = 2;
    let alpha = 2.0_f32;
    let beta = 0.5_f32;
    let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32 - 3.0) * 0.2).collect();
    let c: Vec<f32> = (0..m * n).map(|i| i as f32).collect();
    let op = FloatOp::Gemm {
        m: m as u32,
        k: k as u32,
        n: n as u32,
        alpha: alpha.to_bits(),
        beta: beta.to_bits(),
        trans_a: false,
        trans_b: false,
        quant_b: 0,
    };
    let tol = tolerance_for(&op);

    let actual = run_dispatch(&op, &[&f32_bytes(&a), &f32_bytes(&b), &f32_bytes(&c)]);
    let expected = reference::gemm(
        reference::GemmParams::new(&a, &b, m, k, n)
            .with_c(&c)
            .with_alpha(alpha)
            .with_beta(beta),
    );

    assert_close(&actual, &expected, tol, "Gemm-alpha-beta");
}

#[test]
fn conformance_gemm_trans_b() {
    let m = 3;
    let k = 4;
    let n = 2;
    let alpha = 1.0_f32;
    let beta = 0.0_f32;
    let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.1).collect();
    // B is [n, k] when trans_b=true
    let b: Vec<f32> = (0..n * k).map(|i| (i as f32 - 3.0) * 0.2).collect();
    let op = FloatOp::Gemm {
        m: m as u32,
        k: k as u32,
        n: n as u32,
        alpha: alpha.to_bits(),
        beta: beta.to_bits(),
        trans_a: false,
        trans_b: true,
        quant_b: 0,
    };
    let tol = tolerance_for(&op);

    let actual = run_dispatch(&op, &[&f32_bytes(&a), &f32_bytes(&b)]);
    let expected = reference::gemm(
        reference::GemmParams::new(&a, &b, m, k, n)
            .with_alpha(alpha)
            .with_trans_b(true),
    );

    assert_close(&actual, &expected, tol, "Gemm-trans_b");
}

// ── Reductions ──────────────────────────────────────────────────────────────

#[test]
fn conformance_reduce_sum() {
    let input: Vec<f32> = (0..20).map(|i| (i as f32 - 10.0) * 0.7).collect();
    let size = 5;
    let op = FloatOp::ReduceSum { size };
    let tol = tolerance_for(&op);

    let actual = run_dispatch(&op, &[&f32_bytes(&input)]);
    let expected = reference::reduce_sum(&input, size as usize);

    assert_close(&actual, &expected, tol, "ReduceSum");
}

#[test]
fn conformance_reduce_mean() {
    let input: Vec<f32> = (0..20).map(|i| (i as f32) * 0.3).collect();
    let size = 5;
    let op = FloatOp::ReduceMean { size };
    let tol = tolerance_for(&op);

    let actual = run_dispatch(&op, &[&f32_bytes(&input)]);
    let expected = reference::reduce_mean(&input, size as usize);

    assert_close(&actual, &expected, tol, "ReduceMean");
}

#[test]
fn conformance_reduce_max() {
    let input: Vec<f32> = (0..20).map(|i| (i as f32 - 10.0).sin()).collect();
    let size = 5;
    let op = FloatOp::ReduceMax { size };
    let tol = tolerance_for(&op);

    let actual = run_dispatch(&op, &[&f32_bytes(&input)]);
    let expected = reference::reduce_max(&input, size as usize);

    assert_close(&actual, &expected, tol, "ReduceMax");
}

#[test]
fn conformance_reduce_min() {
    let input: Vec<f32> = (0..20).map(|i| (i as f32 - 10.0).cos()).collect();
    let size = 5;
    let op = FloatOp::ReduceMin { size };
    let tol = tolerance_for(&op);

    let actual = run_dispatch(&op, &[&f32_bytes(&input)]);
    let expected = reference::reduce_min(&input, size as usize);

    assert_close(&actual, &expected, tol, "ReduceMin");
}

#[test]
fn conformance_reduce_prod() {
    // Use small values to avoid overflow
    let input: Vec<f32> = (0..12).map(|i| 0.5 + (i as f32) * 0.1).collect();
    let size = 3;
    let op = FloatOp::ReduceProd { size };
    let tol = tolerance_for(&op);

    let actual = run_dispatch(&op, &[&f32_bytes(&input)]);
    let expected = reference::reduce_prod(&input, size as usize);

    assert_close(&actual, &expected, tol, "ReduceProd");
}

// ── Elementwise activations (cross-validated against reference) ─────────────

#[test]
fn conformance_gelu() {
    let input: Vec<f32> = (-10..=10).map(|i| i as f32 * 0.3).collect();
    let op = FloatOp::Gelu;
    let tol = tolerance_for(&op);

    let actual = run_dispatch(&op, &[&f32_bytes(&input)]);
    let expected: Vec<f32> = input.iter().map(|&x| reference::gelu(x)).collect();

    assert_close(&actual, &expected, tol, "GELU");
}

#[test]
fn conformance_silu() {
    let input: Vec<f32> = (-10..=10).map(|i| i as f32 * 0.3).collect();
    let op = FloatOp::Silu;
    let tol = tolerance_for(&op);

    let actual = run_dispatch(&op, &[&f32_bytes(&input)]);
    let expected: Vec<f32> = input.iter().map(|&x| reference::silu(x)).collect();

    assert_close(&actual, &expected, tol, "SiLU");
}

#[test]
fn conformance_sigmoid() {
    let input: Vec<f32> = (-10..=10).map(|i| i as f32 * 0.5).collect();
    let op = FloatOp::Sigmoid;
    let tol = tolerance_for(&op);

    let actual = run_dispatch(&op, &[&f32_bytes(&input)]);
    let expected: Vec<f32> = input.iter().map(|&x| reference::sigmoid(x)).collect();

    assert_close(&actual, &expected, tol, "Sigmoid");
    // Sigmoid output always in (0, 1)
    assert!(actual.iter().all(|&v| v > 0.0 && v < 1.0));
}

// ── Fused SwiGLU ────────────────────────────────────────────────────────────

#[test]
fn conformance_fused_swiglu() {
    let gate: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.3).collect();
    let up: Vec<f32> = (0..16).map(|i| (i as f32 - 4.0) * 0.2).collect();
    let op = FloatOp::FusedSwiGLU;
    let tol = tolerance_for(&op);

    let actual = run_dispatch(&op, &[&f32_bytes(&gate), &f32_bytes(&up)]);
    let expected = reference::fused_swiglu(&gate, &up);

    assert_close(&actual, &expected, tol, "FusedSwiGLU");
}

// ── Rotary Embedding (RoPE) ─────────────────────────────────────────────────

#[test]
fn conformance_rope_position_zero() {
    // At position 0, all angles are 0 → cos=1, sin=0 → identity
    let dim = 8;
    let input: Vec<f32> = (0..dim).map(|i| (i + 1) as f32).collect();
    let op = FloatOp::RotaryEmbedding {
        dim: dim as u32,
        base: 10000.0_f32.to_bits(),
        n_heads: 1,
    };
    let tol = tolerance_for(&op);

    let actual = run_dispatch(&op, &[&f32_bytes(&input)]);
    let expected = reference::rotary_embedding(&input, dim, 10000.0, 1, 0);

    assert_close(&actual, &expected, tol, "RoPE-pos0");
    // At position 0, output should equal input (identity)
    for (a, &e) in actual.iter().zip(input.iter()) {
        assert!(
            (a - e).abs() < 1e-5,
            "RoPE at pos 0 should be identity: {a} vs {e}"
        );
    }
}

#[test]
fn conformance_rope_multi_token() {
    let dim = 8;
    let n_heads = 2;
    // 3 tokens * 2 heads = 6 chunks of dim=8
    let input: Vec<f32> = (0..48).map(|i| (i as f32 * 0.1).sin()).collect();
    let op = FloatOp::RotaryEmbedding {
        dim: dim as u32,
        base: 10000.0_f32.to_bits(),
        n_heads: n_heads as u32,
    };
    let tol = tolerance_for(&op);

    let actual = run_dispatch(&op, &[&f32_bytes(&input)]);
    let expected = reference::rotary_embedding(&input, dim, 10000.0, n_heads, 0);

    assert_close(&actual, &expected, tol, "RoPE-multi-token");
}

#[test]
fn conformance_rope_preserves_norm() {
    // Rotation preserves vector norm (within each pair)
    let dim = 8;
    let input: Vec<f32> = (0..dim).map(|i| (i + 1) as f32).collect();
    let op = FloatOp::RotaryEmbedding {
        dim: dim as u32,
        base: 10000.0_f32.to_bits(),
        n_heads: 1,
    };

    let actual = run_dispatch(&op, &[&f32_bytes(&input)]);
    let input_norm: f32 = input.iter().map(|x| x * x).sum::<f32>().sqrt();
    let output_norm: f32 = actual.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (input_norm - output_norm).abs() < 1e-4,
        "RoPE should preserve norm: {input_norm} vs {output_norm}"
    );
}

// ── Attention ───────────────────────────────────────────────────────────────

/// dispatch_attention expects [seq, n_heads, head_dim] layout.
/// reference::attention expects [n_heads, seq, head_dim] layout.
/// dispatch_attention also returns [seq, n_heads, head_dim].
/// We generate in [seq, n_heads, head_dim], transpose for reference, then
/// transpose reference output back to [seq, n_heads, head_dim] for comparison.
fn transpose_seq_heads(data: &[f32], seq: usize, n_heads: usize, head_dim: usize) -> Vec<f32> {
    // [seq, n_heads, head_dim] -> [n_heads, seq, head_dim]
    let mut out = vec![0.0f32; data.len()];
    for t in 0..seq {
        for h in 0..n_heads {
            for d in 0..head_dim {
                out[h * seq * head_dim + t * head_dim + d] =
                    data[t * n_heads * head_dim + h * head_dim + d];
            }
        }
    }
    out
}

fn transpose_heads_to_seq(data: &[f32], seq: usize, n_heads: usize, head_dim: usize) -> Vec<f32> {
    // [n_heads, seq, head_dim] -> [seq, n_heads, head_dim]
    let mut out = vec![0.0f32; data.len()];
    for h in 0..n_heads {
        for t in 0..seq {
            for d in 0..head_dim {
                out[t * n_heads * head_dim + h * head_dim + d] =
                    data[h * seq * head_dim + t * head_dim + d];
            }
        }
    }
    out
}

fn run_attention_test(
    head_dim: usize,
    seq: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    causal: bool,
    name: &str,
) {
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Generate in [seq, n_heads, head_dim] format (what dispatch expects)
    let q_snh: Vec<f32> = (0..seq * num_q_heads * head_dim)
        .map(|i| (i as f32 * 0.1).sin())
        .collect();
    let k_snh: Vec<f32> = (0..seq * num_kv_heads * head_dim)
        .map(|i| (i as f32 * 0.2).cos())
        .collect();
    let v_snh: Vec<f32> = (0..seq * num_kv_heads * head_dim)
        .map(|i| i as f32 * 0.15)
        .collect();

    let op = FloatOp::Attention {
        head_dim: head_dim as u32,
        num_q_heads: num_q_heads as u32,
        num_kv_heads: num_kv_heads as u32,
        scale: scale.to_bits(),
        causal,
        heads_first: false,
        qk_norm: false,
        rope: false,
        rope_base: 0,
        sparse_v: true,
    };
    let tol = tolerance_for(&op);

    // dispatch expects [seq, n_heads, head_dim], returns [seq, n_heads, head_dim]
    let actual = run_dispatch(
        &op,
        &[&f32_bytes(&q_snh), &f32_bytes(&k_snh), &f32_bytes(&v_snh)],
    );

    // reference expects [n_heads, seq, head_dim]
    let q_hns = transpose_seq_heads(&q_snh, seq, num_q_heads, head_dim);
    let k_hns = transpose_seq_heads(&k_snh, seq, num_kv_heads, head_dim);
    let v_hns = transpose_seq_heads(&v_snh, seq, num_kv_heads, head_dim);
    let ref_out_hns = reference::attention(
        reference::AttentionParams::new(
            &q_hns,
            &k_hns,
            &v_hns,
            head_dim,
            num_q_heads,
            num_kv_heads,
        )
        .with_scale(scale)
        .with_causal(causal),
    );
    // transpose reference output back to [seq, n_heads, head_dim]
    let expected = transpose_heads_to_seq(&ref_out_hns, seq, num_q_heads, head_dim);

    assert_close(&actual, &expected, tol, name);
}

#[test]
fn conformance_attention_single_head() {
    run_attention_test(4, 3, 1, 1, false, "Attention-single-head");
}

#[test]
fn conformance_attention_causal() {
    run_attention_test(4, 4, 2, 2, true, "Attention-causal");
}

#[test]
fn conformance_attention_gqa() {
    run_attention_test(4, 3, 4, 2, false, "Attention-GQA");
}

// ── Conv2d ──────────────────────────────────────────────────────────────────

#[test]
fn conformance_conv2d_simple() {
    let in_h = 5;
    let in_w = 5;
    let k_h = 3;
    let k_w = 3;
    let input: Vec<f32> = (0..in_h * in_w).map(|i| i as f32 * 0.1).collect();
    let kernel: Vec<f32> = (0..k_h * k_w).map(|i| (i as f32 - 4.0) * 0.2).collect();

    let op = FloatOp::Conv2d {
        kernel_h: k_h as u32,
        kernel_w: k_w as u32,
        stride_h: 1,
        stride_w: 1,
        pad_h: 0,
        pad_w: 0,
        dilation_h: 1,
        dilation_w: 1,
        group: 1,
        input_h: in_h as u32,
        input_w: in_w as u32,
    };
    let tol = tolerance_for(&op);

    let actual = run_dispatch(&op, &[&f32_bytes(&input), &f32_bytes(&kernel)]);
    let expected = reference::conv2d_simple(reference::Conv2dSimpleParams::new(
        &input, &kernel, in_h, in_w, k_h, k_w,
    ));

    assert_close(&actual, &expected, tol, "Conv2d-simple");
}

#[test]
fn conformance_conv2d_with_padding() {
    let in_h = 4;
    let in_w = 4;
    let k_h = 3;
    let k_w = 3;
    let pad_h = 1;
    let pad_w = 1;
    let input: Vec<f32> = (0..in_h * in_w).map(|i| (i as f32).sin()).collect();
    let kernel: Vec<f32> = (0..k_h * k_w).map(|i| (i as f32 * 0.3).cos()).collect();

    let op = FloatOp::Conv2d {
        kernel_h: k_h as u32,
        kernel_w: k_w as u32,
        stride_h: 1,
        stride_w: 1,
        pad_h: pad_h as u32,
        pad_w: pad_w as u32,
        dilation_h: 1,
        dilation_w: 1,
        group: 1,
        input_h: in_h as u32,
        input_w: in_w as u32,
    };
    let tol = tolerance_for(&op);

    let actual = run_dispatch(&op, &[&f32_bytes(&input), &f32_bytes(&kernel)]);
    let expected = reference::conv2d_simple(
        reference::Conv2dSimpleParams::new(&input, &kernel, in_h, in_w, k_h, k_w)
            .with_padding(pad_h, pad_w),
    );

    assert_close(&actual, &expected, tol, "Conv2d-padded");
}

// ── Multi-row tests (batched behavior) ──────────────────────────────────────

#[test]
fn conformance_softmax_multi_row() {
    // 4 rows of 5 elements each — tests correct row-independent processing
    let size = 5;
    let input: Vec<f32> = (0..20).map(|i| ((i as f32) * 1.7 - 15.0).sin()).collect();
    let op = FloatOp::Softmax { size };
    let tol = tolerance_for(&op);

    let actual = run_dispatch(&op, &[&f32_bytes(&input)]);
    let expected = reference::softmax(&input, size as usize);

    assert_close(&actual, &expected, tol, "Softmax-multi-row");
    // Each row should sum to ~1.0
    for (i, row) in actual.chunks(size as usize).enumerate() {
        let sum: f32 = row.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Row {i} sum = {sum}, expected ~1.0"
        );
    }
}

#[test]
fn conformance_rms_norm_multi_row() {
    let size = 16;
    let rows = 4;
    let input: Vec<f32> = (0..rows * size).map(|i| ((i as f32) * 0.7).sin()).collect();
    let weight: Vec<f32> = (0..size).map(|i| 0.8 + i as f32 * 0.025).collect();
    let epsilon = 1e-5_f32;
    let op = FloatOp::RmsNorm {
        size: size as u32,
        epsilon: epsilon.to_bits(),
    };
    let tol = tolerance_for(&op);

    let actual = run_dispatch(&op, &[&f32_bytes(&input), &f32_bytes(&weight)]);
    let expected = reference::rms_norm(&input, &weight, size, epsilon);

    assert_close(&actual, &expected, tol, "RmsNorm-multi-row");
}

// ── GroupNorm ─────────────────────────────────────────────────────────────

/// Reference GroupNorm implementation for conformance testing.
/// data: [N, C, spatial...] flattened, scale/bias: [C]
fn reference_group_norm(
    data: &[f32],
    scale: &[f32],
    bias: &[f32],
    num_groups: usize,
    epsilon: f32,
) -> Vec<f32> {
    let n_channels = scale.len();
    let channels_per_group = n_channels / num_groups;
    let spatial = data.len() / n_channels;
    let mut out = data.to_vec();

    for g in 0..num_groups {
        let group_size = channels_per_group * spatial;
        let mut sum: f64 = 0.0;
        let mut sum_sq: f64 = 0.0;
        for c_local in 0..channels_per_group {
            let c = g * channels_per_group + c_local;
            let start = c * spatial;
            let end = (start + spatial).min(out.len());
            for &v in &out[start..end] {
                let v64 = v as f64;
                sum += v64;
                sum_sq += v64 * v64;
            }
        }
        let mean = (sum / group_size as f64) as f32;
        let var = (sum_sq / group_size as f64 - (mean as f64 * mean as f64)) as f32;
        let inv_std = 1.0 / (var + epsilon).sqrt();

        for c_local in 0..channels_per_group {
            let c = g * channels_per_group + c_local;
            let s = scale[c];
            let b = bias[c];
            let start = c * spatial;
            let end = (start + spatial).min(out.len());
            for v in out[start..end].iter_mut() {
                *v = (*v - mean) * inv_std * s + b;
            }
        }
    }
    out
}

#[test]
fn conformance_group_norm() {
    // 4 channels, 2 groups, spatial=8 (like a [1, 4, 2, 4] NCHW tensor)
    let num_groups = 2u32;
    let n_channels = 4;
    let spatial = 8;
    let epsilon = 1e-5_f32;

    let data: Vec<f32> = (0..n_channels * spatial)
        .map(|i| (i as f32 - 16.0) * 0.1)
        .collect();
    let scale: Vec<f32> = (0..n_channels).map(|i| 1.0 + i as f32 * 0.1).collect();
    let bias: Vec<f32> = (0..n_channels).map(|i| i as f32 * 0.05).collect();

    let op = FloatOp::GroupNorm {
        num_groups,
        epsilon: epsilon.to_bits(),
    };
    let tol = Tolerance {
        atol: 1e-5,
        rtol: 1e-4,
    };

    let actual = run_dispatch(
        &op,
        &[&f32_bytes(&data), &f32_bytes(&scale), &f32_bytes(&bias)],
    );
    let expected = reference_group_norm(&data, &scale, &bias, num_groups as usize, epsilon);

    assert_close(&actual, &expected, tol, "GroupNorm");
}

#[test]
fn conformance_group_norm_single_group() {
    // Single group = LayerNorm-like behavior across all channels
    let num_groups = 1u32;
    let n_channels = 4;
    let spatial = 4;
    let epsilon = 1e-5_f32;

    let data: Vec<f32> = (0..n_channels * spatial)
        .map(|i| ((i as f32) * 0.3).sin())
        .collect();
    let scale: Vec<f32> = vec![1.0; n_channels];
    let bias: Vec<f32> = vec![0.0; n_channels];

    let op = FloatOp::GroupNorm {
        num_groups,
        epsilon: epsilon.to_bits(),
    };
    let tol = Tolerance {
        atol: 1e-5,
        rtol: 1e-4,
    };

    let actual = run_dispatch(
        &op,
        &[&f32_bytes(&data), &f32_bytes(&scale), &f32_bytes(&bias)],
    );
    let expected = reference_group_norm(&data, &scale, &bias, num_groups as usize, epsilon);

    assert_close(&actual, &expected, tol, "GroupNorm-single-group");
}
