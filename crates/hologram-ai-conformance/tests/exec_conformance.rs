//! Execution conformance tests — compile ONNX → hologram, run both, compare.
//!
//! These tests validate the full compile → lower → execute pipeline by
//! comparing hologram executor output against ORT on multi-node ONNX models.
//!
//! Feature-gated behind `conformance` (requires ORT runtime).
//!
//! Run with:
//!   ORT_STRATEGY=system cargo test -p hologram-ai-conformance --features conformance

#![cfg(feature = "conformance")]

use hologram_ai::{ModelCompiler, ModelSource};
use hologram_ai_conformance::ort_runner::onnx_builder;
use hologram_ai_conformance::ort_runner::runner::{run_onnx_all_outputs, run_onnx_file_typed, OrtInput, OrtInputTyped};
use hologram_ai_conformance::tolerance::Tolerance;

/// Default tolerance for execution conformance (slightly looser than kernel tests
/// since errors can accumulate across nodes).
fn exec_tol() -> Tolerance {
    Tolerance {
        atol: 1e-4,
        rtol: 1e-3,
    }
}

/// Helper: compile ONNX bytes, execute through hologram, return final output f32s.
fn compile_and_execute(
    model_bytes: &[u8],
    inputs: &[(&str, Vec<usize>, Vec<f32>)],
) -> Vec<f32> {
    let compiler = ModelCompiler::default();
    let (archive, _debug_map) = compiler
        .compile_with_debug_info(ModelSource::OnnxBytes(model_bytes.to_vec()))
        .expect("compilation failed");

    // Build GraphInputs.
    let mut graph_inputs = hologram::GraphInputs::new();
    for (i, (_name, shape, data)) in inputs.iter().enumerate() {
        let bytes: Vec<u8> = bytemuck::cast_slice(data).to_vec();
        graph_inputs.set_with_shape(i as u32, bytes, shape.clone());
    }

    // Execute.
    let plan = hologram::load_from_bytes(&archive.bytes).expect("loading archive");
    let outputs = hologram::execute_plan(&plan, &graph_inputs).expect("execution failed");

    // Extract first output as f32 (safe — no alignment requirement).
    let (_, out_bytes) = outputs.get(0).expect("no outputs");
    out_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("chunk is 4 bytes")))
        .collect()
}

/// Test: debug map is populated for a compiled model.
#[test]
fn debug_map_populated_for_matmul() {
    let model_bytes = onnx_builder::matmul(2, 4, 3);
    let compiler = ModelCompiler::default();
    let (_archive, debug_map) = compiler
        .compile_with_debug_info(ModelSource::OnnxBytes(model_bytes))
        .expect("compile failed");
    assert!(
        !debug_map.name_to_idx.is_empty(),
        "debug map should not be empty"
    );
}

/// Test: MatMul output matches ORT.
#[test]
fn matmul_matches_ort() {
    let m = 2;
    let k = 4;
    let n = 3;
    let model_bytes = onnx_builder::matmul(m, k, n);

    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.05 + 0.1).collect();

    // ORT reference.
    let ort_outputs = run_onnx_all_outputs(
        &model_bytes,
        vec![
            OrtInput { name: "A".into(), shape: vec![m, k], data: a_data.clone() },
            OrtInput { name: "B".into(), shape: vec![k, n], data: b_data.clone() },
        ],
    )
    .expect("ORT failed");

    // Hologram.
    let holo_out = compile_and_execute(
        &model_bytes,
        &[("A", vec![m, k], a_data), ("B", vec![k, n], b_data)],
    );

    let cmp = hologram_ai_conformance::tolerance::compare_outputs(
        &holo_out,
        &ort_outputs[0].data,
        exec_tol(),
    );
    assert!(cmp.passed, "MatMul mismatch: {}", cmp.message);
}

/// Test: Softmax output matches ORT.
#[test]
fn softmax_matches_ort() {
    let rows = 2;
    let size = 8;
    let model_bytes = onnx_builder::softmax(rows, size);

    let input_data: Vec<f32> = (0..rows * size).map(|i| (i as f32) * 0.5 - 3.0).collect();

    let ort_outputs = run_onnx_all_outputs(
        &model_bytes,
        vec![OrtInput { name: "input".into(), shape: vec![rows, size], data: input_data.clone() }],
    )
    .expect("ORT failed");

    let holo_out = compile_and_execute(
        &model_bytes,
        &[("input", vec![rows, size], input_data)],
    );

    let cmp = hologram_ai_conformance::tolerance::compare_outputs(
        &holo_out,
        &ort_outputs[0].data,
        exec_tol(),
    );
    assert!(cmp.passed, "Softmax mismatch: {}", cmp.message);
}

/// Test: RmsNorm composite model (6 nodes) matches ORT.
#[test]
fn rmsnorm_composite_matches_ort() {
    let rows = 2;
    let size = 16;
    let eps = 1e-6;
    let model_bytes = onnx_builder::rms_norm(rows, size, eps);

    let x_data: Vec<f32> = (0..rows * size).map(|i| (i as f32) * 0.1 - 0.8).collect();
    let w_data: Vec<f32> = (0..size).map(|i| 1.0 + (i as f32) * 0.01).collect();

    let ort_outputs = run_onnx_all_outputs(
        &model_bytes,
        vec![
            OrtInput { name: "X".into(), shape: vec![rows, size], data: x_data.clone() },
            OrtInput { name: "Weight".into(), shape: vec![size], data: w_data.clone() },
        ],
    )
    .expect("ORT failed");

    let holo_out = compile_and_execute(
        &model_bytes,
        &[("X", vec![rows, size], x_data), ("Weight", vec![size], w_data)],
    );

    // Composite ops: slightly looser tolerance.
    let tol = Tolerance { atol: 1e-3, rtol: 1e-2 };
    let cmp = hologram_ai_conformance::tolerance::compare_outputs(
        &holo_out, &ort_outputs[0].data, tol,
    );
    assert!(cmp.passed, "RmsNorm mismatch: {}", cmp.message);
}

/// Test: Gemm with transB matches ORT.
#[test]
fn gemm_trans_b_matches_ort() {
    let m = 3;
    let k = 4;
    let n = 2;
    let model_bytes = onnx_builder::gemm(m, k, n, 1.0, 1.0, false, true);

    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
    let b_data: Vec<f32> = (0..n * k).map(|i| (i as f32) * 0.05).collect();
    let c_data: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.01).collect();

    let ort_outputs = run_onnx_all_outputs(
        &model_bytes,
        vec![
            OrtInput { name: "A".into(), shape: vec![m, k], data: a_data.clone() },
            OrtInput { name: "B".into(), shape: vec![n, k], data: b_data.clone() },
            OrtInput { name: "C".into(), shape: vec![m, n], data: c_data.clone() },
        ],
    )
    .expect("ORT failed");

    let holo_out = compile_and_execute(
        &model_bytes,
        &[("A", vec![m, k], a_data), ("B", vec![n, k], b_data), ("C", vec![m, n], c_data)],
    );

    let cmp = hologram_ai_conformance::tolerance::compare_outputs(
        &holo_out, &ort_outputs[0].data, exec_tol(),
    );
    assert!(cmp.passed, "Gemm transB mismatch: {}", cmp.message);
}

/// Test: LayerNorm composite model (9 nodes) matches ORT.
#[test]
fn layernorm_composite_matches_ort() {
    let rows = 2;
    let size = 16;
    let eps = 1e-5;
    let model_bytes = onnx_builder::layer_norm(rows, size, eps);

    let x_data: Vec<f32> = (0..rows * size).map(|i| (i as f32) * 0.1 - 0.8).collect();
    let w_data: Vec<f32> = (0..size).map(|i| 1.0 + (i as f32) * 0.01).collect();
    let b_data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001).collect();

    let ort_outputs = run_onnx_all_outputs(
        &model_bytes,
        vec![
            OrtInput { name: "X".into(), shape: vec![rows, size], data: x_data.clone() },
            OrtInput { name: "Weight".into(), shape: vec![size], data: w_data.clone() },
            OrtInput { name: "Bias".into(), shape: vec![size], data: b_data.clone() },
        ],
    )
    .expect("ORT failed");

    let holo_out = compile_and_execute(
        &model_bytes,
        &[
            ("X", vec![rows, size], x_data),
            ("Weight", vec![size], w_data),
            ("Bias", vec![size], b_data),
        ],
    );

    let tol = Tolerance { atol: 1e-3, rtol: 1e-2 };
    let cmp = hologram_ai_conformance::tolerance::compare_outputs(
        &holo_out, &ort_outputs[0].data, tol,
    );
    assert!(cmp.passed, "LayerNorm mismatch: {}", cmp.message);
}

/// Test: 4D batched MatMul matches ORT.
///
/// Covers the Q@K^T pattern in multi-head attention where A and B are 4D:
/// A [batch, heads, seq_q, head_dim] × B [batch, heads, head_dim, seq_k] → [batch, heads, seq_q, seq_k]
///
/// This test exercises the batched matmul dispatch path with 4D inputs and validates
/// that shape tracking through the pipeline produces correct outputs. A failure here
/// indicates a bug in shape propagation or batched matmul dispatch for attention ops.
#[test]
fn batched_4d_matmul_matches_ort() {
    let batch = 1;
    let heads = 4;
    let seq_q = 6;
    let head_dim = 8;
    let seq_k = 6;
    // A: [batch, heads, seq_q, head_dim], B: [batch, heads, head_dim, seq_k]
    // → Y: [batch, heads, seq_q, seq_k]
    let model_bytes = onnx_builder::batched_matmul_4d(batch, heads, seq_q, head_dim, seq_k);

    let a_elems = batch * heads * seq_q * head_dim;
    let b_elems = batch * heads * head_dim * seq_k;
    let a_data: Vec<f32> = (0..a_elems).map(|i| (i as f32) * 0.05 - 1.0).collect();
    let b_data: Vec<f32> = (0..b_elems).map(|i| (i as f32) * 0.03 + 0.1).collect();

    let ort_outputs = run_onnx_all_outputs(
        &model_bytes,
        vec![
            OrtInput { name: "A".into(), shape: vec![batch, heads, seq_q, head_dim], data: a_data.clone() },
            OrtInput { name: "B".into(), shape: vec![batch, heads, head_dim, seq_k], data: b_data.clone() },
        ],
    )
    .expect("ORT failed for batched_4d_matmul");

    let holo_out = compile_and_execute(
        &model_bytes,
        &[
            ("A", vec![batch, heads, seq_q, head_dim], a_data),
            ("B", vec![batch, heads, head_dim, seq_k], b_data),
        ],
    );

    let cmp = hologram_ai_conformance::tolerance::compare_outputs(
        &holo_out,
        &ort_outputs[0].data,
        exec_tol(),
    );
    assert!(cmp.passed, "4D batched MatMul mismatch: {}", cmp.message);
}

/// Test: Concat along last axis (axis=3) of 4D tensors matches ORT.
///
/// Covers the rotate_half concat pattern in RoPE where two [batch, heads, seq, half_dim]
/// tensors are concatenated along the last axis to produce [batch, heads, seq, head_dim].
///
/// A failure here indicates that `concrete_concat_row_size` in strategy.rs computes the
/// wrong row size (missing the axis-dim factor), causing incorrect concat lowering.
#[test]
fn concat_4d_last_axis_matches_ort() {
    let batch = 1;
    let heads = 4;
    let seq = 6;
    let half_dim = 8; // each half; concatenated = 16
    let model_bytes = onnx_builder::concat_4d_last_axis(batch, heads, seq, half_dim, half_dim);

    let elems_per_half = batch * heads * seq * half_dim;
    // First half: identity pattern, second half: negated offset pattern (like rotate_half)
    let a_data: Vec<f32> = (0..elems_per_half).map(|i| (i as f32) * 0.1).collect();
    let b_data: Vec<f32> = (0..elems_per_half).map(|i| -(i as f32) * 0.1 - 0.5).collect();

    let ort_outputs = run_onnx_all_outputs(
        &model_bytes,
        vec![
            OrtInput { name: "A".into(), shape: vec![batch, heads, seq, half_dim], data: a_data.clone() },
            OrtInput { name: "B".into(), shape: vec![batch, heads, seq, half_dim], data: b_data.clone() },
        ],
    )
    .expect("ORT failed for concat_4d_last_axis");

    let holo_out = compile_and_execute(
        &model_bytes,
        &[
            ("A", vec![batch, heads, seq, half_dim], a_data),
            ("B", vec![batch, heads, seq, half_dim], b_data),
        ],
    );

    let cmp = hologram_ai_conformance::tolerance::compare_outputs(
        &holo_out,
        &ort_outputs[0].data,
        exec_tol(),
    );
    assert!(cmp.passed, "Concat last-axis mismatch: {}", cmp.message);
}

/// Test: Scaled Dot-Product Attention matches ORT.
///
/// Full attention pattern: Q@K^T → scale → softmax → scores@V
/// Q, K, V: [batch, heads, seq, head_dim] → AttnOut: [batch, heads, seq, head_dim]
///
/// This is the core attention pattern from TinyLlama. A failure indicates a bug in
/// one or more of: 4D transpose lowering, 4D batched matmul shape tracking, or softmax.
#[test]
fn scaled_dot_product_attention_matches_ort() {
    let batch = 1;
    let heads = 4;
    let seq = 6;
    let head_dim = 8;
    let model_bytes =
        onnx_builder::scaled_dot_product_attention(batch, heads, seq, head_dim);

    let elems = batch * heads * seq * head_dim;
    let q_data: Vec<f32> = (0..elems).map(|i| (i as f32) * 0.02 - 0.5).collect();
    let k_data: Vec<f32> = (0..elems).map(|i| (i as f32) * 0.015 + 0.1).collect();
    let v_data: Vec<f32> = (0..elems).map(|i| ((i % 16) as f32) * 0.1 - 0.7).collect();

    let ort_outputs = run_onnx_all_outputs(
        &model_bytes,
        vec![
            OrtInput { name: "Q".into(), shape: vec![batch, heads, seq, head_dim], data: q_data.clone() },
            OrtInput { name: "K".into(), shape: vec![batch, heads, seq, head_dim], data: k_data.clone() },
            OrtInput { name: "V".into(), shape: vec![batch, heads, seq, head_dim], data: v_data.clone() },
        ],
    )
    .expect("ORT failed for scaled_dot_product_attention");

    let holo_out = compile_and_execute(
        &model_bytes,
        &[
            ("Q", vec![batch, heads, seq, head_dim], q_data),
            ("K", vec![batch, heads, seq, head_dim], k_data),
            ("V", vec![batch, heads, seq, head_dim], v_data),
        ],
    );

    // Attention involves softmax, so slightly looser tolerance.
    let tol = Tolerance { atol: 1e-3, rtol: 1e-2 };
    let cmp = hologram_ai_conformance::tolerance::compare_outputs(
        &holo_out,
        &ort_outputs[0].data,
        tol,
    );
    assert!(cmp.passed, "Scaled dot-product attention mismatch: {}", cmp.message);
}

/// Test: GQA (Grouped Query Attention) with Expand matches ORT.
///
/// Models TinyLlama's GQA pattern: n_heads=32, n_kv_heads=4, head_dim=64.
/// K and V are first projected to [batch, n_kv_heads, seq, head_dim], then
/// Expand-ed to [batch, n_heads, seq, head_dim] before attention.
///
/// A failure here exposes the GQA Expand shape doubling bug: the Expand op
/// resolves the target shape incorrectly, producing K/V with head_dim×2
/// (e.g. [1,32,seq,128]) instead of [1,32,seq,64]. This propagates through
/// scores@V → Reshape → downstream MatMul as A=[40,4096] instead of [40,2048].
///
/// Dimensions scaled down for test speed; ratio matches TinyLlama (8:2 = 4:1 ratio).
#[test]
fn gqa_expand_attention_matches_ort() {
    let batch = 1;
    let n_heads = 8;
    let n_kv_heads = 2;
    let seq = 6;
    let head_dim = 8;
    let model_bytes =
        onnx_builder::gqa_expand_attention(batch, n_heads, n_kv_heads, seq, head_dim);

    let q_elems = batch * n_heads * seq * head_dim;
    let kv_elems = batch * n_kv_heads * seq * head_dim;

    let q_data: Vec<f32> = (0..q_elems).map(|i| (i as f32) * 0.02 - 0.5).collect();
    let k_data: Vec<f32> = (0..kv_elems).map(|i| (i as f32) * 0.015 + 0.1).collect();
    let v_data: Vec<f32> = (0..kv_elems).map(|i| ((i % 16) as f32) * 0.1 - 0.7).collect();

    let ort_outputs = run_onnx_all_outputs(
        &model_bytes,
        vec![
            OrtInput {
                name: "Q".into(),
                shape: vec![batch, n_heads, seq, head_dim],
                data: q_data.clone(),
            },
            OrtInput {
                name: "K_compact".into(),
                shape: vec![batch, n_kv_heads, seq, head_dim],
                data: k_data.clone(),
            },
            OrtInput {
                name: "V_compact".into(),
                shape: vec![batch, n_kv_heads, seq, head_dim],
                data: v_data.clone(),
            },
        ],
    )
    .expect("ORT failed for gqa_expand_attention");

    let holo_out = compile_and_execute(
        &model_bytes,
        &[
            ("Q", vec![batch, n_heads, seq, head_dim], q_data),
            ("K_compact", vec![batch, n_kv_heads, seq, head_dim], k_data),
            ("V_compact", vec![batch, n_kv_heads, seq, head_dim], v_data),
        ],
    );

    // GQA attention: slightly looser tolerance (softmax + multiple matmuls).
    let tol = Tolerance { atol: 1e-3, rtol: 1e-2 };
    let cmp = hologram_ai_conformance::tolerance::compare_outputs(
        &holo_out,
        &ort_outputs[0].data,
        tol,
    );
    assert!(cmp.passed, "GQA Expand attention mismatch: {}", cmp.message);
}

/// Test: GQA attention with TinyLlama-scale dims at seq=1 AND seq=2.
///
/// The existing gqa_expand_attention test uses small dims (8 heads, seq=6).
/// This test uses TinyLlama's actual dimensions (32 heads, 4 KV heads, head_dim=64)
/// to catch precision or stride bugs that only manifest at scale.
/// Tests both seq=1 (baseline) and seq=2 (the failing case for the full model).
#[test]
fn gqa_tinyllama_dims_seq1_and_seq2_match_ort() {
    let batch = 1;
    let n_heads = 32;
    let n_kv_heads = 4;
    let head_dim = 64;

    // Test with smaller dims to narrow the bug threshold.
    // n_heads=32, n_kv=4, head_dim=64 fails at seq=2.
    // n_heads=8, n_kv=2, head_dim=8 passes at seq=6.
    // Try: n_heads=32, n_kv=4, head_dim=8 (large heads, small dim)
    //      n_heads=8, n_kv=2, head_dim=64 (small heads, large dim)
    for seq in [1, 2] {
        let model_bytes =
            onnx_builder::gqa_expand_attention(batch, n_heads, n_kv_heads, seq, head_dim);

        let q_elems = batch * n_heads * seq * head_dim;
        let kv_elems = batch * n_kv_heads * seq * head_dim;

        // Use deterministic pseudo-random data scaled to typical activation range.
        let q_data: Vec<f32> = (0..q_elems).map(|i| ((i * 7 + 3) % 100) as f32 * 0.02 - 1.0).collect();
        let k_data: Vec<f32> = (0..kv_elems).map(|i| ((i * 13 + 7) % 100) as f32 * 0.015 - 0.5).collect();
        let v_data: Vec<f32> = (0..kv_elems).map(|i| ((i * 11 + 5) % 100) as f32 * 0.01 - 0.3).collect();

        let ort_outputs = run_onnx_all_outputs(
            &model_bytes,
            vec![
                OrtInput { name: "Q".into(), shape: vec![batch, n_heads, seq, head_dim], data: q_data.clone() },
                OrtInput { name: "K_compact".into(), shape: vec![batch, n_kv_heads, seq, head_dim], data: k_data.clone() },
                OrtInput { name: "V_compact".into(), shape: vec![batch, n_kv_heads, seq, head_dim], data: v_data.clone() },
            ],
        )
        .unwrap_or_else(|e| panic!("ORT failed for GQA seq={seq}: {e}"));

        let holo_out = compile_and_execute(
            &model_bytes,
            &[
                ("Q", vec![batch, n_heads, seq, head_dim], q_data.clone()),
                ("K_compact", vec![batch, n_kv_heads, seq, head_dim], k_data.clone()),
                ("V_compact", vec![batch, n_kv_heads, seq, head_dim], v_data.clone()),
            ],
        );

        let tol = Tolerance { atol: 1e-3, rtol: 1e-2 };

        // (intermediate tracing removed — use sub-pattern tests to isolate)

        let cmp = hologram_ai_conformance::tolerance::compare_outputs(
            &holo_out,
            &ort_outputs[0].data,
            tol,
        );
        eprintln!(
            "[gqa-tinyllama seq={seq}] max_abs_err={:.6} max_rel_err={:.6} mismatches={}/{}",
            cmp.max_abs_error, cmp.max_rel_error, cmp.num_mismatches, cmp.total_elements,
        );
        assert!(
            cmp.passed,
            "GQA TinyLlama-dims attention mismatch at seq={seq}: {}",
            cmp.message,
        );
    }
}

/// Test: KV head expansion (Unsqueeze→Expand→Reshape) at TinyLlama dims.
/// Isolates the first 3 ops of the GQA pattern to find where seq>1 diverges.
#[test]
fn kv_expand_tinyllama_dims_matches_ort() {
    use hologram_ai_conformance::ort_runner::onnx_builder::{
        Node as ONode, Initializer,
    };

    let batch = 1;
    let n_heads = 32;
    let n_kv_heads = 4;
    let seq = 2;
    let head_dim = 64;
    let group_size = n_heads / n_kv_heads;

    let expand_shape: Vec<i64> = vec![
        batch as i64, n_kv_heads as i64, group_size as i64, seq as i64, head_dim as i64,
    ];
    let attn_shape: Vec<i64> = vec![batch as i64, n_heads as i64, seq as i64, head_dim as i64];

    let nodes = vec![
        ONode::new("Unsqueeze", &["K_compact", "unsq_axes"], &["K_unsq"]),
        ONode::new("Expand", &["K_unsq", "expand_shape"], &["K_5d"]),
        ONode::new("Reshape", &["K_5d", "attn_shape"], &["K_exp"]),
    ];
    let inits = vec![
        Initializer::int64_1d("unsq_axes", vec![2]),
        Initializer::int64_1d("expand_shape", expand_shape),
        Initializer::int64_1d("attn_shape", attn_shape),
    ];
    // Build the model using gqa_expand_attention (but only K expansion + reshape).
    // Use onnx_builder::gqa_expand_attention as template — it outputs AttnOut.
    // Instead, build a simpler K-only expansion model.
    let model_bytes = onnx_builder::gqa_expand_attention(batch, n_heads, n_kv_heads, seq, head_dim);

    let q_elems = batch * n_heads * seq * head_dim;
    let kv_elems = batch * n_kv_heads * seq * head_dim;
    let q_data: Vec<f32> = (0..q_elems).map(|i| ((i * 7 + 3) % 100) as f32 * 0.02 - 1.0).collect();
    let k_data: Vec<f32> = (0..kv_elems).map(|i| ((i * 13 + 7) % 100) as f32 * 0.015 - 0.5).collect();
    let v_data: Vec<f32> = (0..kv_elems).map(|i| ((i * 11 + 5) % 100) as f32 * 0.01 - 0.3).collect();

    let ort_out = run_onnx_all_outputs(
        &model_bytes,
        vec![
            OrtInput { name: "Q".into(), shape: vec![batch, n_heads, seq, head_dim], data: q_data.clone() },
            OrtInput { name: "K_compact".into(), shape: vec![batch, n_kv_heads, seq, head_dim], data: k_data.clone() },
            OrtInput { name: "V_compact".into(), shape: vec![batch, n_kv_heads, seq, head_dim], data: v_data.clone() },
        ],
    ).expect("ORT failed");

    let holo_out = compile_and_execute(
        &model_bytes,
        &[
            ("Q", vec![batch, n_heads, seq, head_dim], q_data),
            ("K_compact", vec![batch, n_kv_heads, seq, head_dim], k_data),
            ("V_compact", vec![batch, n_kv_heads, seq, head_dim], v_data),
        ],
    );

    eprintln!("[kv-expand] ORT: elems={} hologram: elems={}", ort_out[0].data.len(), holo_out.len());

    let tol = Tolerance { atol: 1e-3, rtol: 1e-2 };
    let cmp = hologram_ai_conformance::tolerance::compare_outputs(
        &holo_out,
        &ort_out[0].data,
        tol,
    );
    eprintln!(
        "[kv-expand] max_abs_err={:.6} mismatches={}/{}",
        cmp.max_abs_error, cmp.num_mismatches, cmp.total_elements,
    );
    assert!(cmp.passed, "KV expand mismatch at seq={seq}: {}", cmp.message);
}

/// Test: `Shape` op returns per-axis dimension values (not a scalar element count).
///
/// Graph: X [2, 6, 32] → Shape → INT64 [3] → Cast(to=FLOAT) → Y [3]
///
/// Expected: Y = [2.0, 6.0, 32.0].
///
/// A failure means `Shape` returns a 1-element scalar (e.g. 384.0 = 2*6*32)
/// instead of the three individual dims. This is the root-cause regression test
/// for the TinyLlama `A=[40,4096]` shape-doubling bug: if Shape is wrong, the
/// Shape → Slice → Concat → Expand chains in GQA models produce garbage shapes.
#[test]
fn shape_op_returns_correct_dims_matches_ort() {
    let batch = 2usize;
    let seq = 6usize;
    let hidden = 32usize;
    let model_bytes = onnx_builder::shape_then_cast_to_float(batch, seq, hidden);

    let n_elems = batch * seq * hidden;
    let x_data: Vec<f32> = (0..n_elems).map(|i| i as f32 * 0.1).collect();

    let ort_outputs = run_onnx_all_outputs(
        &model_bytes,
        vec![OrtInput { name: "X".into(), shape: vec![batch, seq, hidden], data: x_data.clone() }],
    )
    .expect("ORT failed for shape_op_returns_correct_dims");

    let holo_out = compile_and_execute(
        &model_bytes,
        &[("X", vec![batch, seq, hidden], x_data)],
    );

    // ORT returns [2.0, 6.0, 32.0]; hologram must agree exactly.
    let cmp = hologram_ai_conformance::tolerance::compare_outputs(
        &holo_out,
        &ort_outputs[0].data,
        exec_tol(),
    );
    assert!(
        cmp.passed,
        "Shape op returned wrong dims: expected [{batch}.0, {seq}.0, {hidden}.0], got {:?}. {}",
        holo_out,
        cmp.message
    );
}

/// Test: Expand where the target shape is built at runtime via Shape → Slice → Concat.
///
/// Graph:
///   X [batch, seq, hidden]
///   Shape(X) → x_shape INT64[3]
///   Slice(x_shape, 0:2) → first_two INT64[2] = [batch, seq]
///   Concat([first_two, hidden_c]) → reshape_tgt INT64[3] = [batch, seq, hidden]
///   Expand(X, reshape_tgt) → Y [batch, seq, hidden]
///
/// Since the target matches X's shape this is identity, but the path exercises
/// the runtime Shape → Slice → Concat chain. A bug in `Shape` that returns a
/// scalar element count will propagate into garbage expand shape and either fail
/// outright or produce a mis-shaped output whose values differ from ORT.
#[test]
fn expand_with_dynamic_shape_tensor_matches_ort() {
    let batch = 2usize;
    let seq = 6usize;
    let hidden = 32usize;
    let model_bytes = onnx_builder::expand_via_dynamic_shape(batch, seq, hidden);

    let n_elems = batch * seq * hidden;
    let x_data: Vec<f32> = (0..n_elems).map(|i| (i as f32) * 0.05 - 1.5).collect();

    let ort_outputs = run_onnx_all_outputs(
        &model_bytes,
        vec![OrtInput { name: "X".into(), shape: vec![batch, seq, hidden], data: x_data.clone() }],
    )
    .expect("ORT failed for expand_with_dynamic_shape_tensor");

    let holo_out = compile_and_execute(
        &model_bytes,
        &[("X", vec![batch, seq, hidden], x_data)],
    );

    let cmp = hologram_ai_conformance::tolerance::compare_outputs(
        &holo_out,
        &ort_outputs[0].data,
        exec_tol(),
    );
    assert!(cmp.passed, "Expand with dynamic shape tensor mismatch: {}", cmp.message);
}

/// Test: GQA K-expand where Expand and Reshape targets are computed at runtime
/// via Shape → Slice → Concat, exactly as in TinyLlama's ONNX graph.
///
/// Unlike `gqa_expand_attention_matches_ort` (constant INT64 shape initializers),
/// this model extracts `[seq, head_dim]` from `Shape(K_compact)` at runtime via
/// Slice and concatenates with constant dims to build the 5-D expand target.
///
/// Expected (batch=1, n_heads=8, n_kv_heads=2, seq=6, head_dim=8):
///   K_compact [1,2,6,8] → K_exp [1,8,6,8]  (each of 8 heads = one of 2 KV heads × 4)
///
/// A failure here directly reproduces the TinyLlama regression where
/// Shape → Slice → Concat produces a wrong 5-D expand shape, causing V to
/// expand to [1,32,40,128] (head_dim doubled) instead of [1,32,40,64], which
/// propagates to A=[40,4096] at the output-projection MatMul (NodeId 336).
#[test]
fn gqa_k_expand_with_dynamic_shape_matches_ort() {
    let batch = 1usize;
    let n_heads = 8usize;
    let n_kv_heads = 2usize;
    let seq = 6usize;
    let head_dim = 8usize;
    let model_bytes =
        onnx_builder::gqa_k_expand_with_dynamic_shape(batch, n_heads, n_kv_heads, seq, head_dim);

    let kv_elems = batch * n_kv_heads * seq * head_dim;
    let k_data: Vec<f32> = (0..kv_elems).map(|i| (i as f32) * 0.015 + 0.1).collect();

    let ort_outputs = run_onnx_all_outputs(
        &model_bytes,
        vec![OrtInput {
            name: "K_compact".into(),
            shape: vec![batch, n_kv_heads, seq, head_dim],
            data: k_data.clone(),
        }],
    )
    .expect("ORT failed for gqa_k_expand_with_dynamic_shape");

    let holo_out = compile_and_execute(
        &model_bytes,
        &[("K_compact", vec![batch, n_kv_heads, seq, head_dim], k_data)],
    );

    // Output is K_exp [1, 8, 6, 8] = 384 elements.
    // Element count mismatch would panic earlier; value mismatch here means
    // the wrong KV head was used for one or more query heads.
    let tol = Tolerance { atol: 1e-5, rtol: 1e-4 };
    let cmp = hologram_ai_conformance::tolerance::compare_outputs(
        &holo_out,
        &ort_outputs[0].data,
        tol,
    );
    assert!(
        cmp.passed,
        "GQA K-expand with dynamic shape mismatch (TinyLlama regression): {}",
        cmp.message
    );
}

/// Test: `Shape` op respects `start`/`end` attributes (opset 15 style).
///
/// Graph:
///   K [1, 2, 6, 8] → Shape(start=0, end=1) → INT64[1] → Cast → batch_f32 [1]
///   K [1, 2, 6, 8] → Shape(start=2, end=4) → INT64[2] → Cast → seqhd_f32 [2]
///   Concat([batch_f32, seqhd_f32]) → Y [3]
///
/// Expected: Y = [1.0, 6.0, 8.0].
///
/// A failure means `FloatOp::Shape` returns all 4 dims ([1.0, 2.0, 6.0, 8.0])
/// instead of the sliced dims, breaking the Shape → Concat → Expand chain in
/// GQA models like TinyLlama (root cause of the A=[40,4096] regression).
#[test]
fn shape_with_start_end_attrs_matches_ort() {
    let batch = 1usize;
    let n_kv_heads = 2usize;
    let seq = 6usize;
    let head_dim = 8usize;
    let model_bytes =
        onnx_builder::shape_with_start_end_attrs(batch, n_kv_heads, seq, head_dim);

    let kv_elems = batch * n_kv_heads * seq * head_dim;
    let k_data: Vec<f32> = (0..kv_elems).map(|i| i as f32 * 0.1).collect();

    let ort_outputs = run_onnx_all_outputs(
        &model_bytes,
        vec![OrtInput {
            name: "K".into(),
            shape: vec![batch, n_kv_heads, seq, head_dim],
            data: k_data.clone(),
        }],
    )
    .expect("ORT failed for shape_with_start_end_attrs");

    let holo_out = compile_and_execute(
        &model_bytes,
        &[("K", vec![batch, n_kv_heads, seq, head_dim], k_data)],
    );

    // ORT returns [1.0, 6.0, 8.0]; hologram must agree.
    // A wrong result like [1.0, 2.0, 6.0, 8.0] means start/end was ignored.
    let cmp = hologram_ai_conformance::tolerance::compare_outputs(
        &holo_out,
        &ort_outputs[0].data,
        exec_tol(),
    );
    assert!(
        cmp.passed,
        "Shape(start,end) returned wrong dims: got {:?}, expected [{batch}.0, {seq}.0, {head_dim}.0]. \
         If hologram returns 4 values instead of 3, FloatOp::Shape is ignoring start/end. {}",
        holo_out,
        cmp.message
    );
}

/// Test: GQA K-expand using `Shape` with `start`/`end` attrs — TinyLlama's exact pattern.
///
/// Matches TinyLlama's attention layer where:
///   batch_dim = Shape(K_compact, start=0, end=1)  → only dim 0
///   seq_hdim  = Shape(K_compact, start=2, end=4)  → only dims 2..3
///   expand_shape = Concat([batch_dim, nkv_c, group_c, seq_hdim])
///   K_exp = Reshape(Expand(Unsqueeze(K_compact, 2), expand_shape), reshape_tgt)
///
/// If `Shape` ignores `start`/`end`, `batch_dim` returns all 4 dims (corrupting
/// the Concat) and the Expand produces a wrong-count tensor, causing `K_exp`
/// to have 163,840 elements instead of 384 — the TinyLlama NodeId(336) error.
#[test]
fn gqa_k_expand_with_shape_start_end_matches_ort() {
    let batch = 1usize;
    let n_heads = 8usize;
    let n_kv_heads = 2usize;
    let seq = 6usize;
    let head_dim = 8usize;
    let model_bytes = onnx_builder::gqa_k_expand_with_shape_start_end(
        batch, n_heads, n_kv_heads, seq, head_dim,
    );

    let kv_elems = batch * n_kv_heads * seq * head_dim;
    let k_data: Vec<f32> = (0..kv_elems).map(|i| (i as f32) * 0.015 + 0.1).collect();

    let ort_outputs = run_onnx_all_outputs(
        &model_bytes,
        vec![OrtInput {
            name: "K_compact".into(),
            shape: vec![batch, n_kv_heads, seq, head_dim],
            data: k_data.clone(),
        }],
    )
    .expect("ORT failed for gqa_k_expand_with_shape_start_end");

    let holo_out = compile_and_execute(
        &model_bytes,
        &[("K_compact", vec![batch, n_kv_heads, seq, head_dim], k_data)],
    );

    // Output K_exp is [1, 8, 6, 8] = 384 elements.
    let tol = Tolerance { atol: 1e-5, rtol: 1e-4 };
    let cmp = hologram_ai_conformance::tolerance::compare_outputs(
        &holo_out,
        &ort_outputs[0].data,
        tol,
    );
    assert!(
        cmp.passed,
        "GQA K-expand with Shape(start,end) mismatch — TinyLlama NodeId(336) regression: {}",
        cmp.message
    );
}

/// Test: `Shape` with `start`/`end` when seq dim is declared dynamic — hologram
/// constant-folds to the concretized seq=1 value (Dynamic→1 in `concretize_all_dims`).
///
/// Hologram compiles for a **fixed** sequence length: symbolic/dynamic dims are
/// baked in via `concretize_all_dims` (Dynamic → 1, Var → lower bound), then a
/// second DataProp pass materializes the Shape node as a compile-time constant.
/// The compiled output is therefore [1.0, 8.0] (seq=1, head_dim=8), not [6.0, 8.0].
///
/// This is a known architectural limitation: end-to-end Shape start/end correctness
/// is verified at the executor kernel level in `hologram-exec/tests/shape_chain.rs`
/// (`shape_start_end_extracts_seq_and_head_dim`), and at the AiGraph level with
/// concrete input dims in `shape_with_start_end_attrs_matches_ort`.
///
/// This test documents the current behavior and ensures compilation + execution
/// succeed without panicking.
#[test]
fn shape_start_end_with_dynamic_seq_compiles_and_runs() {
    let batch = 1usize;
    let n_kv_heads = 2usize;
    let seq = 6usize;
    let head_dim = 8usize;
    let model_bytes =
        onnx_builder::shape_start_end_with_dynamic_seq(batch, n_kv_heads, seq, head_dim);

    let kv_elems = batch * n_kv_heads * seq * head_dim;
    let k_data: Vec<f32> = (0..kv_elems).map(|i| i as f32 * 0.1).collect();

    // Just verify compilation and execution succeed; do not compare to ORT since
    // hologram bakes in seq=1 (Dynamic→1) while ORT uses the actual runtime seq=6.
    let holo_out = compile_and_execute(
        &model_bytes,
        &[("K", vec![batch, n_kv_heads, seq, head_dim], k_data)],
    );

    // Output must be non-empty and not contain NaN (Shape output is shape dims).
    assert!(
        !holo_out.is_empty(),
        "Shape(start=2,end=4) with dynamic seq must produce non-empty output"
    );
    for v in &holo_out {
        assert!(!v.is_nan(), "Shape output must not contain NaN");
    }
}

/// Test: SwiGLU (silu(gate) * up) matches ORT reference.
///
/// Verifies the fused SwiGLU activation used in TinyLlama/LLaMA GGUF FFN blocks.
/// The ONNX graph computes `silu(gate) * up = Sigmoid(gate) * gate * up` via 3 nodes.
/// hologram compiles this as `FloatOp::FusedSwiGLU` (fused kernel).
///
/// A mismatch here indicates either:
/// (a) the ONNX-to-AiOp fusion for SwiGLU is incorrect, or
/// (b) the `FusedSwiGLU` kernel computes `gate * up` instead of `silu(gate) * up`.
#[test]
fn swiglu_matches_ort() {
    let rows = 4;
    let cols = 16;
    let model_bytes = onnx_builder::swiglu(rows, cols);

    let n = rows * cols;
    // Use varied values including negatives to exercise the sigmoid non-linearity.
    let gate: Vec<f32> = (0..n).map(|i| (i as f32) * 0.3 - 2.0).collect();
    let up: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 + 0.5).collect();

    let ort_outputs = run_onnx_all_outputs(
        &model_bytes,
        vec![
            OrtInput { name: "gate".into(), shape: vec![rows, cols], data: gate.clone() },
            OrtInput { name: "up".into(), shape: vec![rows, cols], data: up.clone() },
        ],
    )
    .expect("ORT failed for swiglu");

    let holo_out = compile_and_execute(
        &model_bytes,
        &[("gate", vec![rows, cols], gate), ("up", vec![rows, cols], up)],
    );

    let cmp = hologram_ai_conformance::tolerance::compare_outputs(
        &holo_out,
        &ort_outputs[0].data,
        exec_tol(),
    );
    assert!(
        cmp.passed,
        "SwiGLU mismatch: expected silu(gate)*up but got different values. \
         If hologram returns gate*up (missing silu), check FusedSwiGLU dispatch. {}",
        cmp.message
    );
}

/// Test: GQA with flat inputs and causal mask matches ORT.
///
/// Models TinyLlama's GGUF attention path where Q/K/V arrive as flat
/// `[seq, n_heads*head_dim]` tensors from the Gemm projection, rather than
/// pre-split `[batch, n_heads, seq, head_dim]` as in the ONNX path.
///
/// Uses n_kv_heads=1 (single shared KV head) to keep the ONNX reference graph
/// expressible without a loop. The ratio n_q_heads:n_kv_heads = 4:1 exercises
/// the GQA head-repeat (group_size=4) logic.
///
/// A mismatch here indicates a bug in:
/// (a) the flat-input reshape/transpose within dispatch_attention, or
/// (b) the GQA head mapping (kh = qh / group_size) in the kernel.
///
/// Note: hologram compiles the flat Reshape+Transpose+Expand+SDPA ONNX chain
/// (not via FloatOp::Attention) — this test validates the ONNX-path ops that
/// produce equivalent GQA results, cross-validating the overall computation.
#[test]
fn gqa_flat_single_kv_matches_ort() {
    let n_q_heads = 4;
    let seq = 5;
    let head_dim = 8;
    let model_bytes = onnx_builder::gqa_flat_single_kv(n_q_heads, seq, head_dim);

    let q_elems = seq * n_q_heads * head_dim;
    let kv_elems = seq * head_dim;
    let q_data: Vec<f32> = (0..q_elems).map(|i| (i as f32) * 0.02 - 0.5).collect();
    let k_data: Vec<f32> = (0..kv_elems).map(|i| (i as f32) * 0.015 + 0.1).collect();
    let v_data: Vec<f32> = (0..kv_elems).map(|i| ((i % 8) as f32) * 0.1 - 0.3).collect();

    let ort_outputs = run_onnx_all_outputs(
        &model_bytes,
        vec![
            OrtInput {
                name: "Q_flat".into(),
                shape: vec![seq, n_q_heads * head_dim],
                data: q_data.clone(),
            },
            OrtInput { name: "K_flat".into(), shape: vec![seq, head_dim], data: k_data.clone() },
            OrtInput { name: "V_flat".into(), shape: vec![seq, head_dim], data: v_data.clone() },
        ],
    )
    .expect("ORT failed for gqa_flat_single_kv");

    let holo_out = compile_and_execute(
        &model_bytes,
        &[
            ("Q_flat", vec![seq, n_q_heads * head_dim], q_data),
            ("K_flat", vec![seq, head_dim], k_data),
            ("V_flat", vec![seq, head_dim], v_data),
        ],
    );

    // Multiple matmuls + softmax — slightly looser tolerance.
    let tol = Tolerance { atol: 1e-3, rtol: 1e-2 };
    let cmp = hologram_ai_conformance::tolerance::compare_outputs(
        &holo_out,
        &ort_outputs[0].data,
        tol,
    );
    assert!(
        cmp.passed,
        "GQA flat-input single-KV-head mismatch (GGUF attention path regression): {}",
        cmp.message
    );
}

/// Regression: Range op must correctly read i64 scalar inputs.
///
/// In ONNX, when the Range `limit` comes from a Shape op, it is an 8-byte i64
/// scalar.  The old `dispatch_range` called `cast_f32()` which reinterpreted
/// those bytes as IEEE 754 f32, producing a subnormal ≈0 (e.g. i64(8) → f32
/// ≈1.1e-44).  The result was a 1-element output `[0.0]` instead of
/// `[0.0, 1.0, ..., n-1.0]`.
///
/// The fix: detect 8-byte inputs and read them as i64, then convert to f32.
///
/// Graph: Range(i64_zero, i64_n, i64_one) → Cast(to=float) → output [n]
#[test]
fn range_i64_inputs_matches_ort() {
    let n = 8usize; // small enough to be fast but exercises the bug
    let model_bytes = onnx_builder::range_i64_then_cast(n);

    let ort_outputs = run_onnx_all_outputs(&model_bytes, vec![])
        .expect("ORT failed for range_i64_inputs");

    let holo_out = compile_and_execute(&model_bytes, &[]);

    // Expected: [0.0, 1.0, 2.0, ..., n-1.0]
    let expected: Vec<f32> = (0..n).map(|i| i as f32).collect();
    assert_eq!(
        holo_out.len(),
        expected.len(),
        "Range i64 output length mismatch: hologram produced {} elements, expected {} (= n={}). \
         This indicates Range is misreading i64 inputs as f32 subnormals.",
        holo_out.len(),
        expected.len(),
        n
    );

    let cmp = hologram_ai_conformance::tolerance::compare_outputs(
        &holo_out,
        &ort_outputs[0].data,
        Tolerance { atol: 1e-5, rtol: 1e-5 },
    );
    assert!(
        cmp.passed,
        "Range i64 input mismatch (regression: i64 bytes misread as f32): {}",
        cmp.message
    );
}

/// Regression: `binary_compare_broadcast` must perform full orthogonal broadcast.
///
/// `LessOrEqual([seq,1], [1,seq])` should produce `[seq,seq]` (out_len = seq²),
/// not `[seq]` (element cycling).  The stale-shape guard that was in
/// `binary_compare_broadcast` incorrectly triggered because `out_len > max(a.len(), b.len())` —
/// which is the hallmark of a valid orthogonal broadcast, not stale-shape inflation.
///
/// TinyLlama `model_causal.onnx` uses this exact pattern to build its causal
/// attention mask.  Without the fix, hologram produces an all-1.0 `[seq]` mask,
/// turning the model into non-causal (bidirectional) attention and generating
/// completely wrong logits for every transformer layer.
///
/// Graph: LessOrEqual(row=[seq,1], col=[1,seq]) → bool [seq,seq] → Cast → float [seq,seq].
#[test]
fn causal_mask_orthogonal_broadcast_matches_ort() {
    let seq = 4usize;
    let model_bytes = onnx_builder::causal_mask_less_equal(seq);

    let ort_outputs = run_onnx_all_outputs(&model_bytes, vec![])
        .expect("ORT failed for causal_mask_less_equal");

    let holo_out = compile_and_execute(&model_bytes, &[]);

    // Expected: upper-triangular [seq, seq] float mask.
    // entry[i,j] = 1.0 if i <= j else 0.0.
    let expected: Vec<f32> = (0..seq)
        .flat_map(|i| (0..seq).map(move |j| if i <= j { 1.0_f32 } else { 0.0_f32 }))
        .collect();

    assert_eq!(
        holo_out.len(),
        seq * seq,
        "causal mask output length mismatch: hologram produced {} elements, expected {}={}². \
         binary_compare_broadcast fell back to element cycling instead of [seq,1]×[1,seq]→[seq,seq] \
         orthogonal broadcast.",
        holo_out.len(),
        seq * seq,
        seq
    );

    let cmp = hologram_ai_conformance::tolerance::compare_outputs(
        &holo_out,
        &ort_outputs[0].data,
        Tolerance { atol: 1e-6, rtol: 0.0 },
    );
    assert!(
        cmp.passed,
        "causal_mask LessOrEqual orthogonal broadcast mismatch — \
         binary_compare_broadcast stale-shape guard regression.\n\
         Expected (upper-triangular): {:?}\n\
         Got: {:?}\n{}",
        expected,
        holo_out,
        cmp.message
    );
}

/// Test: model_causal.onnx top-1 logit at last token matches ORT.
///
/// Loads the actual TinyLlama model_causal.onnx from disk, runs it through both
/// ORT and hologram with a 2-token input, and compares the top-5 predicted token IDs
/// at position 1 (last real token). A mismatch reveals that hologram's full-model
/// computation diverges from ORT.
///
/// Skipped if models/TinyLlama-1.1B-Chat-v1.0/model_causal.onnx is not present.
/// Ignored by default: takes 4+ minutes (loads 1GB model). Use
/// `cargo test --test exec_conformance -- --ignored tinyllama` to run manually.
#[test]
#[ignore]
fn tinyllama_causal_onnx_top1_matches_ort() {
    // Resolve model path relative to workspace root.
    let mut model_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    model_path.pop(); // hologram-ai-conformance → crates/
    model_path.pop(); // crates/ → workspace root
    model_path.push("models/TinyLlama-1.1B-Chat-v1.0/model_causal.onnx");

    if !model_path.exists() {
        eprintln!("SKIP: {:?} not found", model_path);
        return;
    }

    let vocab = 32000usize;

    // Hologram: compile as single graph (not pipeline) to isolate compilation
    // path issues. Uses compile_with_debug_info + run_with_shape_context.
    let compiler = hologram_ai::ModelCompiler::default();
    let (archive, _debug_map) = compiler
        .compile_with_debug_info(hologram_ai::ModelSource::OnnxPath(model_path.clone()))
        .expect("hologram compilation failed");

    // Helper closure: run hologram with shape context (single-graph path).
    let run_hologram = |archive: &hologram_ai::HoloArchive, ids: &[i64], mask: &[i64]| {
        let seq = ids.len();
        let id_bytes: Vec<u8> = ids.iter().flat_map(|&v| v.to_le_bytes()).collect();
        let mask_bytes: Vec<u8> = mask.iter().flat_map(|&v| v.to_le_bytes()).collect();
        let mut graph_inputs = hologram::GraphInputs::new();
        graph_inputs.set_with_shape(0, id_bytes, vec![1, seq]);
        graph_inputs.set_with_shape(1, mask_bytes, vec![1, seq]);
        let outputs = hologram_ai::run_with_shape_context(archive, &graph_inputs)
            .expect("hologram execution failed");
        let (_, holo_bytes) = outputs.get(0).expect("no hologram outputs");
        bytemuck::cast_slice::<u8, f32>(holo_bytes).to_vec()
    };

    // ── Diagnostic: seq=1 ────────────────────────────────────────────────────
    // With seq=1, compiled shapes (concretized to seq=1) match runtime shapes exactly.
    // If this diverges from ORT, there is a fundamental computation bug.
    {
        let ids1: Vec<i64> = vec![1];
        let mask1: Vec<i64> = vec![1];
        let ort1 = run_onnx_file_typed(
            &model_path,
            vec![
                OrtInputTyped::I64 { name: "input_ids".into(), shape: vec![1, 1], data: ids1.clone() },
                OrtInputTyped::I64 { name: "attention_mask".into(), shape: vec![1, 1], data: mask1.clone() },
            ],
        )
        .expect("ORT seq=1 failed");
        let ort1_logits = &ort1[0].data;
        let holo1_logits = run_hologram(&archive, &ids1, &mask1);
        let top5_ort1 = top_k(ort1_logits, 5);
        let top5_holo1 = if holo1_logits.len() >= vocab {
            top_k(&holo1_logits[..vocab], 5)
        } else {
            vec![]
        };
        eprintln!(
            "[diag seq=1] ORT top-5: {:?}, hologram top-5: {:?}, match={}",
            top5_ort1,
            top5_holo1,
            !top5_holo1.is_empty() && top5_ort1[0] == top5_holo1[0]
        );
    }

    // ── Main test: seq=2 ─────────────────────────────────────────────────────
    // KNOWN BUG: Shape projection for attention head-split Reshape nodes
    // produces wrong shapes at seq > 1. The ShapeContextGraph's Concat
    // i64 propagation chain includes Shape(attention_mask) which contributes
    // [batch, seq] (2 values) instead of just [seq] (1 value), creating a
    // 5-dim Reshape target where the Gather index picks the wrong dimension.
    // Root cause: missing Gather between Shape(mask) and Concat in the
    // shape computation chain — the ONNX model's Concat directly consumes
    // Shape output, and the ShapeSpecRepr::Shape projection returns the
    // input tensor's shape as its OWN shape (not [rank] as a 1-D i64 tensor).
    // See specs/SPRINT.md Plan 014 for full diagnosis.
    let seq = 2usize;
    let input_ids: Vec<i64> = vec![1, 2]; // BOS + second token
    let attention_mask: Vec<i64> = vec![1, 1];

    let ort_outputs = run_onnx_file_typed(
        &model_path,
        vec![
            OrtInputTyped::I64 {
                name: "input_ids".into(),
                shape: vec![1, seq],
                data: input_ids.clone(),
            },
            OrtInputTyped::I64 {
                name: "attention_mask".into(),
                shape: vec![1, seq],
                data: attention_mask.clone(),
            },
        ],
    )
    .expect("ORT failed for tinyllama_causal_onnx_top1");

    assert!(!ort_outputs.is_empty(), "ORT produced no outputs");
    let ort_logits = &ort_outputs[0].data;
    assert!(
        ort_logits.len() >= seq * vocab,
        "ORT logit output too small: {} < {}",
        ort_logits.len(),
        seq * vocab
    );
    let ort_last_pos = &ort_logits[(seq - 1) * vocab..seq * vocab];

    let holo_logits = run_hologram(&archive, &input_ids, &attention_mask);
    assert!(
        holo_logits.len() >= seq * vocab,
        "hologram logit output too small: {} < {}",
        holo_logits.len(),
        seq * vocab
    );
    let holo_last_pos = &holo_logits[(seq - 1) * vocab..seq * vocab];

    let top5_ort = top_k(ort_last_pos, 5);
    let top5_holo = top_k(holo_last_pos, 5);

    eprintln!("ORT top-5:     {:?}", top5_ort);
    eprintln!("hologram top-5: {:?}", top5_holo);

    // Log actual logit values for the top tokens.
    eprintln!("ORT logit values:     {:?}",
        top5_ort.iter().map(|&i| ort_last_pos[i]).collect::<Vec<_>>());
    eprintln!("hologram logit values: {:?}",
        top5_holo.iter().map(|&i| holo_last_pos[i]).collect::<Vec<_>>());
    // Also show hologram's logit for ORT's top-1.
    let ort_top1 = top5_ort[0];
    eprintln!(
        "hologram logit for ORT's top-1 ({}): {:.4} (hologram rank: {})",
        ort_top1,
        holo_last_pos[ort_top1],
        top5_holo.iter().position(|&i| i == ort_top1).map(|p| p + 1).unwrap_or(999),
    );

    // The top-1 token must match.
    assert_eq!(
        top5_ort[0], top5_holo[0],
        "top-1 token mismatch: ORT predicted {:?} but hologram predicted {:?}. \
         This indicates hologram's full TinyLlama computation diverges from ORT. \
         ORT top-5: {:?}, hologram top-5: {:?}",
        top5_ort[0], top5_holo[0], top5_ort, top5_holo
    );
}

/// Node-by-node divergence finder for TinyLlama ONNX at seq=2.
///
/// Compiles TinyLlama, executes with intermediate capture, loads ORT
/// intermediates from npz, and finds the first node where hologram diverges.
///
/// Prerequisites:
///   python3 scripts/ort_intermediates.py 2  (creates /tmp/ort_intermediates_seq2.npz)
///
/// Run with:
///   ORT_STRATEGY=system cargo test -p hologram-ai-conformance \
///     --features "conformance,profile" --test exec_conformance \
///     -- --ignored tinyllama_node_divergence --nocapture
#[test]
#[ignore]
#[cfg(feature = "profile")]
fn tinyllama_node_divergence_finder() {
    let mut model_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    model_path.pop();
    model_path.pop();
    model_path.push("models/TinyLlama-1.1B-Chat-v1.0/model_causal.onnx");
    if !model_path.exists() {
        eprintln!("SKIP: model not found");
        return;
    }

    let npz_path = "/tmp/ort_intermediates_seq2.npz";
    if !std::path::Path::new(npz_path).exists() {
        eprintln!("SKIP: run `python3 scripts/ort_intermediates.py 2` first");
        return;
    }

    // Compile with debug info (single-graph, gives ONNX name → NodeId map).
    let compiler = hologram_ai::ModelCompiler::default();
    let (archive, debug_map) = compiler
        .compile_with_debug_info(hologram_ai::ModelSource::OnnxPath(model_path))
        .expect("compilation failed");

    // Execute with intermediate capture.
    let seq = 2usize;
    let input_ids: Vec<i64> = vec![1, 2];
    let attention_mask: Vec<i64> = vec![1, 1];
    let id_bytes: Vec<u8> = input_ids.iter().flat_map(|&v| v.to_le_bytes()).collect();
    let mask_bytes: Vec<u8> = attention_mask.iter().flat_map(|&v| v.to_le_bytes()).collect();
    let mut graph_inputs = hologram::GraphInputs::new();
    graph_inputs.set_with_shape(0, id_bytes, vec![1, seq]);
    graph_inputs.set_with_shape(1, mask_bytes, vec![1, seq]);

    let plan = hologram::load_from_bytes(&archive.bytes).expect("load failed");
    let capture = hologram::execute_plan_with_intermediates(&plan, &graph_inputs)
        .expect("execution with intermediates failed");

    eprintln!("hologram: {} node buffers captured", capture.node_buffers.len());
    eprintln!("debug_map: {} ONNX name → NodeId mappings", debug_map.name_to_idx.len());

    // Load ORT intermediates from npz (pre-generated by Python script).
    // Read the npz as raw bytes and parse manually (no numpy crate in Rust).
    // Instead, we'll compare just the FINAL logits and report the shape/range
    // for the first few captured nodes.
    // Dump ALL f32 nodes with their ONNX names and value ranges.
    // Sorted by node ID (topological order) so the first divergent node is easy to find.
    let mut node_ids: Vec<_> = capture.node_buffers.keys().collect();
    node_ids.sort_by_key(|id| id.index());

    // Build reverse map: node_idx → ONNX name.
    let idx_to_name: std::collections::HashMap<usize, &str> = debug_map.name_to_idx.iter()
        .map(|(name, &idx)| (idx, name.as_str()))
        .collect();

    eprintln!("\n=== Hologram node intermediates (f32 only, sorted by ID) ===");
    let mut printed = 0;
    for &nid in &node_ids {
        let (buf, es) = &capture.node_buffers[nid];
        // Show all nodes with data, but focus on f32 (es=4) and skip tiny (<4 bytes).
        if buf.len() < 4 { continue; }
        // Skip constants (< node 200) unless they're small activation-shaped.
        if (nid.index() as usize) < 200 && buf.len() > 100000 { continue; }
        let shape = capture.node_shapes.get(nid).map(|s| format!("{s:?}")).unwrap_or_default();
        let name = idx_to_name.get(&(nid.index() as usize)).copied().unwrap_or("?");
        if *es == 4 && buf.len() >= 4 {
            let floats: &[f32] = bytemuck::cast_slice(buf);
            let max = floats.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min = floats.iter().cloned().fold(f32::INFINITY, f32::min);
            eprintln!("  node {:4} {:30} shape={:20} es={} range=[{:12.4}, {:12.4}]",
                nid.index(), name, shape, es, min, max);
        } else {
            eprintln!("  node {:4} {:30} shape={:20} es={} bytes={}",
                nid.index(), name, shape, es, buf.len());
        }
        printed += 1;
        if printed > 2000 { eprintln!("  ... ({} more)", node_ids.len() - printed); break; }
    }
    eprintln!("=== end ({printed} nodes shown) ===\n");

    // Trace the specific Cast node that produces garbage.
    // Node 390 (_to_copy_1) should cast attention_mask from i64 to f32.
    {
        let graph = plan.graph();
        for node in &graph.nodes {
            let idx = node.id.index();
            if (387..=395).contains(&idx) {
                let op = format!("{:?}", node.op);
                let op_short = if op.len() > 80 { &op[..80] } else { &op };
                let inputs: Vec<_> = node.inputs.iter().map(|s| format!("{:?}", s.source)).collect();
                eprintln!("[node-op] {} op={} inputs={:?}", idx, op_short, inputs);
            }
        }
    }

    // Check final logits.
    let (_, logit_bytes) = capture.outputs.get(0).expect("no outputs");
    let logits: &[f32] = bytemuck::cast_slice(logit_bytes);
    let last_pos = &logits[(seq - 1) * 32000..seq * 32000];
    let top5 = top_k(last_pos, 5);
    eprintln!("hologram top-5: {:?}", top5);
    eprintln!("hologram logit values: {:?}",
        top5.iter().map(|&i| last_pos[i]).collect::<Vec<_>>());
}

/// Return indices of the top-k largest values.
fn top_k(logits: &[f32], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.into_iter().take(k).map(|(i, _)| i).collect()
}

/// Node-by-node inspector for TinyLlama ONNX.
///
/// Compiles with `compile_with_shape_context` to get debug map + shape context,
/// executes with shape-aware intermediate capture, and dumps all node outputs
/// to `/tmp/hologram_intermediates/` for comparison with ORT.
///
/// This is the primary debugging tool for finding where hologram diverges from
/// ORT on the full TinyLlama model.
///
/// Usage:
///   1. Generate ORT intermediates:
///      python3 scripts/ort_intermediates.py 5
///   2. Run this test:
///      ORT_STRATEGY=system cargo test -p hologram-ai-conformance \
///        --features "conformance,profile" --test exec_conformance \
///        -- --ignored tinyllama_node_inspector --nocapture
///   3. Compare:
///      python3 scripts/compare_node_by_node.py
#[test]
#[ignore]
#[cfg(feature = "profile")]
fn tinyllama_node_inspector() {
    use hologram_ai_common::lower::shape_spec_bridge::walk_shape_context;
    use std::collections::HashMap;

    let mut model_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    model_path.pop();
    model_path.pop();
    model_path.push("models/TinyLlama-1.1B-Chat-v1.0/model_causal.onnx");
    if !model_path.exists() {
        eprintln!("SKIP: model not found at {}", model_path.display());
        return;
    }

    // ── Step 1: Compile with shape context + debug map ──
    // Compile at seq=5 to match ORT intermediates (python3 scripts/ort_intermediates.py 5).
    let compiled_seq = 5usize;
    eprintln!("=== Compiling TinyLlama ONNX with seq_len={compiled_seq} ===");
    let compiler = ModelCompiler {
        seq_len_override: Some(compiled_seq as u64),
        ..Default::default()
    };
    let (archive, debug_map, shape_ctx) = compiler
        .compile_with_shape_context(ModelSource::OnnxPath(model_path.clone()))
        .expect("compilation failed");

    eprintln!(
        "compiled: {} nodes, {:.1} MB weights, {} debug mappings",
        archive.stats.node_count,
        archive.stats.total_weight_bytes as f64 / 1_048_576.0,
        debug_map.name_to_idx.len(),
    );

    let shape_ctx = shape_ctx.expect("no ShapeContextGraph in compiled archive");
    eprintln!(
        "shape context: {} seeds, {} projections",
        shape_ctx.seeds.len(),
        shape_ctx.projections.len(),
    );

    // ── Step 2: Prepare inputs ──
    // Compiled at seq=5, inputs match exactly — no padding needed.
    let input_ids: Vec<i64> = (1..=compiled_seq as i64).collect();
    let attention_mask: Vec<i64> = vec![1; compiled_seq];

    let id_bytes: Vec<u8> = input_ids.iter().flat_map(|&v| v.to_le_bytes()).collect();
    let mask_bytes: Vec<u8> = attention_mask.iter().flat_map(|&v| v.to_le_bytes()).collect();

    let mut graph_inputs = hologram::GraphInputs::new();
    graph_inputs.set_with_shape(0, id_bytes, vec![1, compiled_seq]);
    graph_inputs.set_with_shape(1, mask_bytes, vec![1, compiled_seq]);

    // Shape hints from ShapeContextGraph (still used for intermediate capture).
    let mut input_node_shapes: HashMap<u32, Vec<usize>> = HashMap::new();
    input_node_shapes.insert(0, vec![1, compiled_seq]);
    input_node_shapes.insert(1, vec![1, compiled_seq]);

    let mut shape_map: HashMap<u32, Vec<usize>> = HashMap::new();
    walk_shape_context(&shape_ctx, &input_node_shapes, &HashMap::new(), &mut shape_map);
    eprintln!("shape_map: {} entries projected", shape_map.len());

    // ── Step 4: Execute with shape-aware intermediate capture ──
    eprintln!("=== Executing with intermediate capture (seq={compiled_seq}) ===");
    let plan = hologram::load_from_bytes(&archive.bytes).expect("load failed");
    let capture = hologram::execute_plan_with_intermediates_and_shape_hints(
        &plan, &graph_inputs, &shape_map,
    )
    .expect("execution with intermediates failed");

    eprintln!(
        "captured: {} node buffers, {} node shapes",
        capture.node_buffers.len(),
        capture.node_shapes.len(),
    );

    // ── Step 5: Also run ORT for direct comparison ──
    eprintln!("=== Running ORT for final logits comparison ===");
    let ort_outputs = run_onnx_file_typed(
        &model_path,
        vec![
            OrtInputTyped::I64 {
                name: "input_ids".into(),
                shape: vec![1, compiled_seq],
                data: input_ids.clone(),
            },
            OrtInputTyped::I64 {
                name: "attention_mask".into(),
                shape: vec![1, compiled_seq],
                data: attention_mask.clone(),
            },
        ],
    )
    .expect("ORT execution failed");

    // Compare final logits at the last REAL token position.
    let ort_logits = &ort_outputs[0];
    let (_, holo_logit_bytes) = capture.outputs.get(0).expect("no hologram output");
    let holo_logits: &[f32] = bytemuck::cast_slice(holo_logit_bytes);

    eprintln!("\n=== Final logits comparison (last position = {}) ===", compiled_seq - 1);
    let vocab = 32000usize;
    let ort_last = &ort_logits.data[(compiled_seq - 1) * vocab..compiled_seq * vocab];
    let holo_last = &holo_logits[(compiled_seq - 1) * vocab..compiled_seq * vocab];

    let ort_top5 = top_k(ort_last, 5);
    let holo_top5 = top_k(holo_last, 5);
    eprintln!("ORT  top-5: {:?}  values: {:?}", ort_top5,
        ort_top5.iter().map(|&i| format!("{:.4}", ort_last[i])).collect::<Vec<_>>());
    eprintln!("Holo top-5: {:?}  values: {:?}", holo_top5,
        holo_top5.iter().map(|&i| format!("{:.4}", holo_last[i])).collect::<Vec<_>>());

    let match_status = if ort_top5[0] == holo_top5[0] { "MATCH" } else { "MISMATCH" };
    eprintln!("Top-1: {} (ORT={}, Holo={})", match_status, ort_top5[0], holo_top5[0]);

    // ── Step 6: Build reverse maps and dump all intermediates ──
    let idx_to_name: HashMap<usize, &str> = debug_map
        .name_to_idx
        .iter()
        .map(|(name, &idx)| (idx, name.as_str()))
        .collect();

    let dump_dir = std::path::Path::new("/tmp/hologram_intermediates");
    std::fs::create_dir_all(dump_dir).expect("creating dump dir");

    // Write metadata file for the Python comparator.
    let mut meta_lines = Vec::new();
    meta_lines.push(format!("seq={compiled_seq}"));
    meta_lines.push(format!("vocab={vocab}"));
    meta_lines.push(format!("input_ids={input_ids:?}"));

    let mut node_ids: Vec<_> = capture.node_buffers.keys().collect();
    node_ids.sort_by_key(|id| id.index());

    eprintln!("\n=== Node-by-node inspection (sorted by ID) ===");
    let mut dumped = 0usize;

    for &nid in &node_ids {
        let idx = nid.index() as usize;
        let (buf, es) = &capture.node_buffers[nid];
        let shape = capture.node_shapes.get(nid);
        let name = idx_to_name.get(&idx).copied().unwrap_or("?");

        // Skip tiny buffers (likely scalar constants or empty).
        if buf.len() < 4 {
            continue;
        }

        // Dump f32 data.
        if *es == 4 && buf.len() >= 4 {
            let floats: &[f32] = bytemuck::cast_slice(buf);
            let min_v = floats.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_v = floats.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let nan_count = floats.iter().filter(|f| f.is_nan()).count();
            let inf_count = floats.iter().filter(|f| f.is_infinite()).count();

            let shape_str = shape
                .map(|s| format!("{s:?}"))
                .unwrap_or_else(|| "?".into());

            // Only print first 50 and any with issues.
            if dumped < 50 || nan_count > 0 || inf_count > 0 {
                eprintln!(
                    "  node {:5} {:40} shape={:25} elems={:8} range=[{:12.4}, {:12.4}]{}",
                    idx,
                    name,
                    shape_str,
                    floats.len(),
                    min_v,
                    max_v,
                    if nan_count > 0 { format!(" NaN={nan_count}") } else { String::new() },
                );
            }

            // Write binary dump file.
            let safe_name = name.replace('/', "_").replace(':', "_");
            let filename = format!("{:05}_{}.f32", idx, safe_name);
            std::fs::write(dump_dir.join(&filename), buf).expect("writing dump file");

            // Write shape file.
            if let Some(s) = shape {
                let shape_str = s.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(",");
                std::fs::write(
                    dump_dir.join(format!("{:05}_{}.shape", idx, safe_name)),
                    shape_str,
                )
                .expect("writing shape file");
            }

            meta_lines.push(format!("{idx}\t{name}\t{}\t{}", floats.len(),
                shape.map(|s| format!("{s:?}")).unwrap_or_default()));
        } else {
            // Non-f32 data (i64 shape tensors, etc.)
            let safe_name = name.replace('/', "_").replace(':', "_");
            let filename = format!("{:05}_{}.raw", idx, safe_name);
            std::fs::write(dump_dir.join(&filename), buf).expect("writing raw dump file");

            meta_lines.push(format!("{idx}\t{name}\traw_es={}\tbytes={}", es, buf.len()));
        }

        dumped += 1;
    }

    // Write metadata index.
    std::fs::write(dump_dir.join("index.tsv"), meta_lines.join("\n"))
        .expect("writing index.tsv");

    eprintln!("\n=== Dumped {dumped} node buffers to {} ===", dump_dir.display());
    eprintln!("Run: python3 scripts/compare_node_by_node.py");
}

// ── Dynamic-seq tests ────────────────────────────────────────────────────────

/// Softmax at dynamic seq lengths (1, 2, 6) matches ORT.
#[test]
fn softmax_dynamic_seq_at_seq2_matches_ort() {
    let hidden = 16usize;
    let model_bytes = onnx_builder::softmax_dyn_seq(hidden);

    for seq in [1usize, 2, 6] {
        let x_data: Vec<f32> = (0..seq * hidden).map(|i| (i as f32) * 0.1 - 0.5).collect();
        let ort_out = run_onnx_all_outputs(
            &model_bytes,
            vec![OrtInput { name: "X".into(), shape: vec![1, seq, hidden], data: x_data.clone() }],
        )
        .unwrap_or_else(|e| panic!("ORT failed at seq={seq}: {e}"));

        let holo_out = compile_and_execute(
            &model_bytes,
            &[("X", vec![1, seq, hidden], x_data)],
        );

        let cmp = hologram_ai_conformance::tolerance::compare_outputs(
            &holo_out, &ort_out[0].data, exec_tol(),
        );
        assert!(cmp.passed, "softmax dyn seq={seq}: {}", cmp.message);
    }
}

/// MatMul at dynamic seq lengths (1, 2, 6) matches ORT.
#[test]
fn matmul_dynamic_seq_at_multiple_seq_matches_ort() {
    let k = 8usize;
    let n = 4usize;
    let model_bytes = onnx_builder::matmul_dyn_seq(k, n);

    let w_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.05 + 0.1).collect();

    for seq in [1usize, 2, 6] {
        let x_data: Vec<f32> = (0..seq * k).map(|i| (i as f32) * 0.05 - 0.3).collect();
        let ort_out = run_onnx_all_outputs(
            &model_bytes,
            vec![
                OrtInput { name: "X".into(), shape: vec![1, seq, k], data: x_data.clone() },
                OrtInput { name: "W".into(), shape: vec![k, n], data: w_data.clone() },
            ],
        )
        .unwrap_or_else(|e| panic!("ORT failed at seq={seq}: {e}"));

        let holo_out = compile_and_execute(
            &model_bytes,
            &[("X", vec![1, seq, k], x_data), ("W", vec![k, n], w_data.clone())],
        );

        let cmp = hologram_ai_conformance::tolerance::compare_outputs(
            &holo_out, &ort_out[0].data, exec_tol(),
        );
        assert!(cmp.passed, "matmul dyn seq={seq}: {}", cmp.message);
    }
}

/// Reshape unpack-heads at dynamic seq=2 matches ORT.
#[test]
fn reshape_unpack_heads_dynamic_seq_at_seq2_matches_ort() {
    let num_heads = 4usize;
    let head_dim = 8usize;
    let hidden = num_heads * head_dim;
    let model_bytes = onnx_builder::reshape_unpack_heads_dyn_seq(num_heads, head_dim);

    for seq in [1usize, 2] {
        let x_data: Vec<f32> = (0..seq * hidden).map(|i| (i as f32) * 0.01).collect();
        let ort_out = run_onnx_all_outputs(
            &model_bytes,
            vec![OrtInput { name: "X".into(), shape: vec![1, seq, hidden], data: x_data.clone() }],
        )
        .unwrap_or_else(|e| panic!("ORT failed at seq={seq}: {e}"));

        let holo_out = compile_and_execute(
            &model_bytes,
            &[("X", vec![1, seq, hidden], x_data)],
        );

        let cmp = hologram_ai_conformance::tolerance::compare_outputs(
            &holo_out, &ort_out[0].data, exec_tol(),
        );
        assert!(cmp.passed, "reshape unpack heads seq={seq}: {}", cmp.message);
    }
}

// ── ShapeContextGraph projection tests ───────────────────────────────────────

/// Verify walk_shape_context projects correct output shape for MatMul.
#[test]
fn walk_shape_context_matmul_projects_output_shape() {
    let m = 2usize;
    let k = 4usize;
    let n = 3usize;
    let model_bytes = onnx_builder::matmul(m, k, n);
    let compiler = ModelCompiler::default();
    let (archive, _debug_map, shape_ctx) = compiler
        .compile_with_shape_context(ModelSource::OnnxBytes(model_bytes))
        .expect("compile failed");

    let ctx = shape_ctx.expect("shape context should be present");
    let mut shape_map = std::collections::HashMap::new();
    let input_shapes: std::collections::HashMap<u32, Vec<usize>> = [
        (0, vec![m, k]),
        (1, vec![k, n]),
    ]
    .into();
    hologram_ai_common::lower::shape_spec_bridge::walk_shape_context(
        &ctx,
        &input_shapes,
        &std::collections::HashMap::new(),
        &mut shape_map,
    );

    // The output node should have shape [m, n].
    let output_shapes: Vec<&Vec<usize>> = shape_map
        .values()
        .filter(|s| *s == &vec![m, n])
        .collect();
    assert!(
        !output_shapes.is_empty(),
        "walk_shape_context should project MatMul output to [{m}, {n}]"
    );
    let _ = archive;
}

/// Verify walk_shape_context projects SameAs(0) for RmsNorm.
#[test]
fn walk_shape_context_rmsnorm_same_as_input() {
    let rows = 2usize;
    let size = 16usize;
    let model_bytes = onnx_builder::rms_norm(rows, size, 1e-6);
    let compiler = ModelCompiler::default();
    let (_archive, _debug_map, shape_ctx) = compiler
        .compile_with_shape_context(ModelSource::OnnxBytes(model_bytes))
        .expect("compile failed");

    let ctx = shape_ctx.expect("shape context should be present");
    let mut shape_map = std::collections::HashMap::new();
    let input_shapes: std::collections::HashMap<u32, Vec<usize>> = [
        (0, vec![rows, size]),
        (1, vec![size]),
    ]
    .into();
    hologram_ai_common::lower::shape_spec_bridge::walk_shape_context(
        &ctx,
        &input_shapes,
        &std::collections::HashMap::new(),
        &mut shape_map,
    );

    // The output should preserve [rows, size].
    let has_output = shape_map.values().any(|s| *s == vec![rows, size]);
    assert!(has_output, "RmsNorm output should be [{rows}, {size}]");
}

// ── Mini transformer CI fixture ──────────────────────────────────────────────

const MINI_HIDDEN: usize = 32;
const MINI_HEADS: usize = 2;
const MINI_FFN: usize = 64;
const MINI_VOCAB: usize = 32;

/// Mini transformer output matches ORT for seq = 1 and 7.
///
/// Compiles the mini transformer and verifies hologram's output matches ORT
/// for two sequence lengths. Uses `run_with_shape_context()` for the hologram
/// path to exercise the ShapeContextGraph projection end-to-end.
///
/// This is the fast CI replacement for `tinyllama_causal_onnx_top1_matches_ort`.
#[test]
fn mini_transformer_matches_ort() {
    let model_bytes = onnx_builder::mini_transformer_dyn(MINI_HIDDEN, MINI_HEADS, MINI_FFN, MINI_VOCAB);

    let archive = ModelCompiler::default()
        .compile(ModelSource::OnnxBytes(model_bytes.clone()))
        .expect("mini transformer compilation failed");

    for seq in [1usize, 7] {
        let x: Vec<f32> = (0..seq * MINI_HIDDEN).map(|i| (i as f32) * 0.01 - 0.32).collect();

        let ort_outputs = run_onnx_all_outputs(
            &model_bytes,
            vec![OrtInput { name: "X".into(), shape: vec![seq, MINI_HIDDEN], data: x.clone() }],
        )
        .unwrap_or_else(|e| panic!("ORT failed at seq={seq}: {e}"));

        let x_bytes: Vec<u8> = bytemuck::cast_slice(&x).to_vec();
        let mut graph_inputs = hologram::GraphInputs::new();
        graph_inputs.set_with_shape(0, x_bytes, vec![seq, MINI_HIDDEN]);

        let outputs = hologram_ai::run_with_shape_context(&archive, &graph_inputs)
            .unwrap_or_else(|e| panic!("hologram failed at seq={seq}: {e}"));
        let (_, out_bytes) = outputs.get(0).expect("no output");
        let holo_out: Vec<f32> = bytemuck::cast_slice(out_bytes).to_vec();

        let tol = Tolerance { atol: 1e-3, rtol: 1e-2 };
        let cmp = hologram_ai_conformance::tolerance::compare_outputs(&holo_out, &ort_outputs[0].data, tol);
        assert!(cmp.passed, "mini transformer seq={seq} mismatch: {}", cmp.message);
    }
}

// ── File-based fixture tests ─────────────────────────────────────────────────
//
// These tests load real `.onnx` files from the `fixtures/` directory,
// mimicking how production models are loaded from disk.

use hologram_ai_conformance::ort_runner::fixtures;

/// Multi-head attention with Unsqueeze/Reshape/Transpose patterns matches ORT.
///
/// Loads `fixtures/multihead_attention.onnx` — exercises the identity-op shape
/// changes (Unsqueeze, Squeeze) that caused TinyLlama failures. Tests
/// `run_with_shape_context()` at variable seq_len.
///
/// hidden=32, n_heads=4, head_dim=8. ~17 KB fixture.
#[test]
fn multihead_attention_fixture_matches_ort() {
    let hidden = 32usize;
    let model_bytes = fixtures::load_or_panic("multihead_attention");

    let archive = ModelCompiler::default()
        .compile(ModelSource::OnnxBytes(model_bytes.clone()))
        .expect("multihead attention compilation failed");

    for seq in [1usize, 3, 7] {
        let x: Vec<f32> = (0..seq * hidden)
            .map(|i| (i as f32) * 0.01 - 0.16)
            .collect();

        let ort_outputs = run_onnx_all_outputs(
            &model_bytes,
            vec![OrtInput {
                name: "X".into(),
                shape: vec![seq, hidden],
                data: x.clone(),
            }],
        )
        .unwrap_or_else(|e| panic!("ORT failed at seq={seq}: {e}"));

        let x_bytes: Vec<u8> = bytemuck::cast_slice(&x).to_vec();
        let mut graph_inputs = hologram::GraphInputs::new();
        graph_inputs.set_with_shape(0, x_bytes, vec![seq, hidden]);

        let runner = hologram_ai::HoloRunner::from_bytes(archive.bytes.clone())
            .expect("failed to load HoloRunner for multihead fixture");
        let outputs = runner.execute(&graph_inputs)
            .unwrap_or_else(|e| panic!("hologram failed at seq={seq}: {e}"));
        let (_, out_bytes) = outputs.get(0).expect("no output");
        let holo_out: Vec<f32> = bytemuck::cast_slice(out_bytes).to_vec();

        let tol = Tolerance { atol: 1e-3, rtol: 1e-2 };
        let cmp = hologram_ai_conformance::tolerance::compare_outputs(
            &holo_out,
            &ort_outputs[0].data,
            tol,
        );
        assert!(
            cmp.passed,
            "multihead attention seq={seq} mismatch: {}",
            cmp.message
        );
    }
}

/// GQA with Unsqueeze→Expand→Reshape KV head repetition matches ORT.
///
/// Loads `fixtures/gqa_expand_attention.onnx` — reproduces the exact TinyLlama
/// GQA pattern: n_heads=8, n_kv_heads=2, group_size=4. Includes the full
/// Shape→Gather→Unsqueeze→Concat→Expand chain for dynamic seq.
///
/// hidden=128, head_dim=16. ~165 KB fixture.
#[test]
fn gqa_expand_attention_fixture_matches_ort() {
    let hidden = 128usize;
    let model_bytes = fixtures::load_or_panic("gqa_expand_attention");

    let archive = ModelCompiler::default()
        .compile(ModelSource::OnnxBytes(model_bytes.clone()))
        .expect("GQA attention compilation failed");

    for seq in [1usize, 3, 7] {
        let x: Vec<f32> = (0..1 * seq * hidden)
            .map(|i| (i as f32) * 0.005 - 0.32)
            .collect();

        let ort_outputs = run_onnx_all_outputs(
            &model_bytes,
            vec![OrtInput {
                name: "X".into(),
                shape: vec![1, seq, hidden],
                data: x.clone(),
            }],
        )
        .unwrap_or_else(|e| panic!("ORT failed at seq={seq}: {e}"));

        let x_bytes: Vec<u8> = bytemuck::cast_slice(&x).to_vec();
        let mut graph_inputs = hologram::GraphInputs::new();
        graph_inputs.set_with_shape(0, x_bytes, vec![1, seq, hidden]);

        let outputs = hologram_ai::run_with_shape_context(&archive, &graph_inputs)
            .unwrap_or_else(|e| panic!("hologram failed at seq={seq}: {e}"));
        let (_, out_bytes) = outputs.get(0).expect("no output");
        let holo_out: Vec<f32> = bytemuck::cast_slice(out_bytes).to_vec();

        let tol = Tolerance { atol: 1e-3, rtol: 1e-2 };
        let cmp = hologram_ai_conformance::tolerance::compare_outputs(
            &holo_out,
            &ort_outputs[0].data,
            tol,
        );
        assert!(
            cmp.passed,
            "GQA expand attention seq={seq} mismatch: {}",
            cmp.message
        );
    }
}

/// Slice on shape tensor picking dim index 2 — reproduces TinyLlama bug.
///
/// TinyLlama's ONNX uses `Shape(X) → Slice(start=2, end=3)` to extract the
/// hidden dim from the shape vector. hologram's `SliceToGather` pass converts
/// this to `Gather(shape, [2], axis=0)` where the shape tensor is 1-D i64.
/// The lowering computes `dim = 1` (row width of a 1-D tensor) and the executor
/// bounds-checks `index < dim` → `2 < 1` → failure.
///
/// Loads `fixtures/slice_shape_to_gather.onnx`. X=[1, seq, 64], output=[64.0].
#[test]
fn slice_shape_to_gather_fixture_matches_ort() {
    let hidden = 64usize;
    let model_bytes = fixtures::load_or_panic("slice_shape_to_gather");

    let archive = ModelCompiler::default()
        .compile(ModelSource::OnnxBytes(model_bytes.clone()))
        .expect("slice_shape_to_gather compilation failed");

    for seq in [1usize, 3] {
        let x: Vec<f32> = (0..seq * hidden)
            .map(|i| (i as f32) * 0.01)
            .collect();

        let ort_outputs = run_onnx_all_outputs(
            &model_bytes,
            vec![OrtInput {
                name: "X".into(),
                shape: vec![1, seq, hidden],
                data: x.clone(),
            }],
        )
        .unwrap_or_else(|e| panic!("ORT failed at seq={seq}: {e}"));

        let x_bytes: Vec<u8> = bytemuck::cast_slice(&x).to_vec();
        let mut graph_inputs = hologram::GraphInputs::new();
        graph_inputs.set_with_shape(0, x_bytes, vec![1, seq, hidden]);

        let outputs = hologram_ai::run_with_shape_context(&archive, &graph_inputs)
            .unwrap_or_else(|e| panic!(
                "hologram failed at seq={seq}: {e}\n\
                 This is the TinyLlama NodeId(498) regression: SliceToGather \
                 converts Slice(shape, 2:3) to Gather with dim=1, then \
                 executor bounds-checks index(2) < dim(1) and fails."
            ));
        let (_, out_bytes) = outputs.get(0).expect("no output");
        let holo_out: Vec<f32> = bytemuck::cast_slice(out_bytes).to_vec();

        let tol = Tolerance { atol: 1e-4, rtol: 1e-4 };
        let cmp = hologram_ai_conformance::tolerance::compare_outputs(
            &holo_out,
            &ort_outputs[0].data,
            tol,
        );
        assert!(
            cmp.passed,
            "slice_shape_to_gather seq={seq}: expected [{hidden}.0], got {:?}. {}",
            holo_out,
            cmp.message
        );
    }
}

/// Gather on 1-D i64 tensor with index > row-width — TinyLlama regression.
///
/// Reproduces the TinyLlama NodeId(498) failure. In hologram's Gather kernel,
/// `dim` stores the row width (product of dims after the gather axis). For
/// a 1-D tensor, dim=1. The executor bounds-checks `index < dim` instead of
/// `index < num_rows`, so any index > 0 on a 1-D i64 tensor fails.
///
/// Loads `fixtures/gather_i64_index_gt_dim.onnx`. data=[10,20,30] i64,
/// Gather(idx=2) → 30 → Cast → 30.0.
#[test]
fn gather_i64_index_gt_dim_fixture_matches_ort() {
    let model_bytes = fixtures::load_or_panic("gather_i64_index_gt_dim");

    let archive = ModelCompiler::default()
        .compile(ModelSource::OnnxBytes(model_bytes.clone()))
        .expect("gather_i64_index_gt_dim compilation failed");

    // data = [10, 20, 30] as i64
    let data_vals: Vec<i64> = vec![10, 20, 30];
    let data_bytes: Vec<u8> = data_vals.iter().flat_map(|&v| v.to_le_bytes()).collect();

    // ORT reference
    // Use typed runner for i64 input
    let ort_out = hologram_ai_conformance::ort_runner::runner::run_onnx_typed(
        &model_bytes,
        vec![hologram_ai_conformance::ort_runner::runner::OrtInputTyped::I64 {
            name: "data".into(),
            shape: vec![3],
            data: data_vals,
        }],
    )
    .expect("ORT failed");
    let ort_val = ort_out[0].data[0]; // should be 30.0

    // Hologram
    let mut graph_inputs = hologram::GraphInputs::new();
    graph_inputs.set_with_shape(0, data_bytes, vec![3]);

    let result = hologram_ai::run_with_shape_context(&archive, &graph_inputs);
    match result {
        Ok(outputs) => {
            let (_, out_bytes) = outputs.get(0).expect("no output");
            let holo_out: Vec<f32> = bytemuck::cast_slice(out_bytes).to_vec();
            assert!(
                (holo_out[0] - ort_val).abs() < 1e-4,
                "gather_i64 mismatch: hologram={}, ORT={ort_val}",
                holo_out[0]
            );
        }
        Err(e) => {
            panic!(
                "hologram Gather i64 failed: {e}\n\
                 This is the TinyLlama NodeId(498) regression: Gather on 1-D i64 \
                 with dim=1 bounds-checks index(2) < dim(1). Fix needed in \
                 hologram executor's Gather kernel: check against num_rows, not dim."
            );
        }
    }
}

/// Shape(4-D)→Slice(2:4) with dynamic seq — TinyLlama SliceToGather regression.
///
/// K=[1, 4, seq, 16]. Shape(K)→Slice(start=2, end=4) picks [seq, 16].
/// With dynamic seq, the result contains a 0-sentinel and can't be constant-folded.
/// SliceToGather converts to Gather with indices=[2,3] on a 1-D i64[4] shape tensor.
/// Gather dim=1 (row width), executor checks index < dim → 2 < 1 or 3 < 1 → fails.
///
/// Loads `fixtures/slice_shape_4d_dynamic.onnx`.
#[test]
fn slice_shape_4d_dynamic_fixture_matches_ort() {
    let model_bytes = fixtures::load_or_panic("slice_shape_4d_dynamic");
    let n_kv_heads = 4usize;
    let head_dim = 16usize;

    let archive = ModelCompiler::default()
        .compile(ModelSource::OnnxBytes(model_bytes.clone()))
        .expect("slice_shape_4d compilation failed");

    for seq in [1usize, 3] {
        let k: Vec<f32> = (0..n_kv_heads * seq * head_dim)
            .map(|i| (i as f32) * 0.01)
            .collect();

        let ort_outputs = run_onnx_all_outputs(
            &model_bytes,
            vec![OrtInput {
                name: "K".into(),
                shape: vec![1, n_kv_heads, seq, head_dim],
                data: k.clone(),
            }],
        )
        .unwrap_or_else(|e| panic!("ORT failed at seq={seq}: {e}"));

        let k_bytes: Vec<u8> = bytemuck::cast_slice(&k).to_vec();
        let mut graph_inputs = hologram::GraphInputs::new();
        graph_inputs.set_with_shape(0, k_bytes, vec![1, n_kv_heads, seq, head_dim]);

        let outputs = hologram_ai::run_with_shape_context(&archive, &graph_inputs)
            .unwrap_or_else(|e| panic!(
                "hologram failed at seq={seq}: {e}\n\
                 TinyLlama NodeId(498) regression: SliceToGather on 4-D shape \
                 with dynamic seq creates Gather(dim=1) with index >= 2."
            ));
        let (_, out_bytes) = outputs.get(0).expect("no output");
        let holo_out: Vec<f32> = bytemuck::cast_slice(out_bytes).to_vec();

        let tol = Tolerance { atol: 1e-4, rtol: 1e-4 };
        let cmp = hologram_ai_conformance::tolerance::compare_outputs(
            &holo_out,
            &ort_outputs[0].data,
            tol,
        );
        assert!(
            cmp.passed,
            "slice_shape_4d seq={seq}: expected [{seq}.0, {head_dim}.0], got {:?}. {}",
            holo_out, cmp.message
        );
    }
}

/// Shape→Slice on dynamic-seq tensor: SliceToGather regression.
///
/// The dynamic seq dim prevents Shape from being constant-folded.
/// hologram's SliceToGather pass converts `Slice(Shape(X), 2:3)` to
/// `Gather(Shape(X), [2], axis=0)` where Shape output is 1-D i64 `[3]`.
/// Gather gets `dim=1` (row width), and the executor checks `index < dim`
/// → `2 < 1` → failure. This is the root cause of TinyLlama NodeId(498).
///
/// Loads `fixtures/slice_shape_dynamic_seq.onnx`.
/// X=[1, seq, 64] → Shape→Slice(2:3)→Cast→sqrt→X/sqrt(hidden) → Y.
#[test]
fn slice_shape_dynamic_seq_fixture_matches_ort() {
    let hidden = 64usize;
    let model_bytes = fixtures::load_or_panic("slice_shape_dynamic_seq");

    let archive = ModelCompiler::default()
        .compile(ModelSource::OnnxBytes(model_bytes.clone()))
        .expect("slice_shape_dynamic_seq compilation failed");

    for seq in [1usize, 3] {
        let x: Vec<f32> = vec![2.0; seq * hidden];

        let ort_outputs = run_onnx_all_outputs(
            &model_bytes,
            vec![OrtInput {
                name: "X".into(),
                shape: vec![1, seq, hidden],
                data: x.clone(),
            }],
        )
        .unwrap_or_else(|e| panic!("ORT failed at seq={seq}: {e}"));

        let x_bytes: Vec<u8> = bytemuck::cast_slice(&x).to_vec();
        let mut graph_inputs = hologram::GraphInputs::new();
        graph_inputs.set_with_shape(0, x_bytes, vec![1, seq, hidden]);

        let outputs = hologram_ai::run_with_shape_context(&archive, &graph_inputs)
            .unwrap_or_else(|e| panic!(
                "hologram failed at seq={seq}: {e}\n\
                 TinyLlama regression: SliceToGather creates Gather with dim=1 \
                 on a 1-D i64 shape tensor. Executor bounds-checks index < dim \
                 instead of index < num_rows."
            ));
        let (_, out_bytes) = outputs.get(0).expect("no output");
        let holo_out: Vec<f32> = bytemuck::cast_slice(out_bytes).to_vec();

        let tol = Tolerance { atol: 1e-4, rtol: 1e-3 };
        let cmp = hologram_ai_conformance::tolerance::compare_outputs(
            &holo_out,
            &ort_outputs[0].data,
            tol,
        );
        assert!(
            cmp.passed,
            "slice_shape_dynamic_seq seq={seq} mismatch: {}",
            cmp.message
        );
    }
}

/// Gather on i64 shape tensor: Shape(X) → Gather(idx=2) → Cast → scalar.
///
/// Reproduces the TinyLlama NodeId(498) failure where a Gather on a shape
/// tensor (dtype=I64, dim=1) bounds-checks the index against `dim` (1)
/// instead of the table length (3), causing "expected i64 index < 1, got
/// index = 2".
///
/// Loads `fixtures/gather_shape_i64.onnx`. X=[1, seq, 64], output=64.0.
#[test]
fn gather_shape_i64_fixture_matches_ort() {
    let hidden = 64usize;
    let model_bytes = fixtures::load_or_panic("gather_shape_i64");

    let archive = ModelCompiler::default()
        .compile(ModelSource::OnnxBytes(model_bytes.clone()))
        .expect("gather_shape_i64 compilation failed");

    for seq in [1usize, 3, 7] {
        let x: Vec<f32> = (0..seq * hidden)
            .map(|i| (i as f32) * 0.01 - 0.32)
            .collect();

        let ort_outputs = run_onnx_all_outputs(
            &model_bytes,
            vec![OrtInput {
                name: "X".into(),
                shape: vec![1, seq, hidden],
                data: x.clone(),
            }],
        )
        .unwrap_or_else(|e| panic!("ORT failed at seq={seq}: {e}"));

        let x_bytes: Vec<u8> = bytemuck::cast_slice(&x).to_vec();
        let mut graph_inputs = hologram::GraphInputs::new();
        graph_inputs.set_with_shape(0, x_bytes, vec![1, seq, hidden]);

        let outputs = hologram_ai::run_with_shape_context(&archive, &graph_inputs)
            .unwrap_or_else(|e| panic!("hologram failed at seq={seq}: {e}"));
        let (_, out_bytes) = outputs.get(0).expect("no output");
        let holo_out: Vec<f32> = bytemuck::cast_slice(out_bytes).to_vec();

        let tol = Tolerance { atol: 1e-4, rtol: 1e-4 };
        let cmp = hologram_ai_conformance::tolerance::compare_outputs(
            &holo_out,
            &ort_outputs[0].data,
            tol,
        );
        assert!(
            cmp.passed,
            "gather_shape_i64 seq={seq}: expected {hidden}.0, got {:?}. {}",
            holo_out,
            cmp.message
        );
    }
}

// ── Fused kernel conformance tests ──────────────────────────────────────────
//
// These tests exercise the fused AiOp paths used by GGUF import
// (GroupedQueryAttention, FusedSwiGLU) by building an AiGraph directly
// and compiling via ModelSource::AiGraph. The reference output comes from
// the decomposed ONNX equivalent run through ORT.
//
// This is the "fixture-driven regression testing" methodology: when a full
// model (TinyLlama GGUF) produces incorrect output, isolate the suspect
// fused kernel into a minimal AiGraph fixture and compare against ORT.

/// Helper: compile an AiGraph through the full pipeline and execute,
/// returning the final output as f32.
fn compile_and_execute_aigraph(
    graph: hologram_ai_common::AiGraph,
    inputs: &[(&str, Vec<usize>, Vec<f32>)],
) -> Vec<f32> {
    let compiler = ModelCompiler::default();
    let archive = compiler
        .compile(ModelSource::AiGraph(graph))
        .expect("AiGraph compilation failed");

    let mut graph_inputs = hologram::GraphInputs::new();
    for (i, (_name, shape, data)) in inputs.iter().enumerate() {
        let bytes: Vec<u8> = bytemuck::cast_slice(data).to_vec();
        graph_inputs.set_with_shape(i as u32, bytes, shape.clone());
    }

    let plan = hologram::load_from_bytes(&archive.bytes).expect("loading archive");
    let outputs = hologram::execute_plan(&plan, &graph_inputs).expect("execution failed");

    let (_, out_bytes) = outputs.get(0).expect("no outputs");
    bytemuck::cast_slice::<u8, f32>(out_bytes).to_vec()
}

/// Build a minimal AiGraph with a single GroupedQueryAttention op.
///
/// Inputs: Q [seq, n_q_heads * head_dim], K [seq, n_kv_heads * head_dim], V [seq, n_kv_heads * head_dim]
/// Output: [seq, n_q_heads * head_dim]
fn build_gqa_aigraph(
    n_q_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    seq: usize,
    causal: bool,
) -> hologram_ai_common::AiGraph {
    use hologram_ai_common::{
        AiGraph, AiNode, AiOp, DType, TensorInfo, shape_from_concrete,
    };
    use std::collections::HashMap;

    let q_dim = (n_q_heads * head_dim) as usize;
    let kv_dim = (n_kv_heads * head_dim) as usize;

    // TensorIds: 0=Q, 1=K, 2=V, 3=output
    let mut tensor_info = HashMap::new();
    tensor_info.insert(0u32, TensorInfo::new(DType::F32, shape_from_concrete(&[seq as u64, q_dim as u64])));
    tensor_info.insert(1u32, TensorInfo::new(DType::F32, shape_from_concrete(&[seq as u64, kv_dim as u64])));
    tensor_info.insert(2u32, TensorInfo::new(DType::F32, shape_from_concrete(&[seq as u64, kv_dim as u64])));
    tensor_info.insert(3u32, TensorInfo::new(DType::F32, shape_from_concrete(&[seq as u64, q_dim as u64])));

    AiGraph {
        name: "gqa_fused_fixture".into(),
        nodes: vec![AiNode::new(
            0,
            AiOp::GroupedQueryAttention {
                num_heads: n_q_heads,
                num_kv_heads: n_kv_heads,
                head_dim,
                scale: None,
                causal,
                heads_first: false, // Fixture inputs are [seq, n_heads*head_dim] (flat/GGUF)
            },
            vec![0, 1, 2],
            vec![3],
        )],
        inputs: vec![0, 1, 2],
        outputs: vec![3],
        input_names: vec!["Q_flat".into(), "K_flat".into(), "V_flat".into()],
        output_names: vec!["output".into()],
        params: HashMap::new(),
        tensor_info,
        metadata: HashMap::new(),
        warnings: vec![],
        dim_vars: Default::default(),
        shape_constraints: Default::default(),
        subgraphs: HashMap::new(),
        tensor_names: HashMap::new(),
    }
}

/// Test: Fused GQA kernel (AiOp::GroupedQueryAttention) matches decomposed ONNX reference.
///
/// This exercises the hologram-exec `dispatch_attention` kernel with
/// num_q_heads=8, num_kv_heads=2 (group_size=4) — the TinyLlama GQA ratio.
/// The reference is the decomposed ONNX graph run through ORT.
///
/// A mismatch here is the primary suspect for GGUF token degeneration.
#[test]
fn gqa_fused_kernel_matches_decomposed() {
    let n_q_heads: usize = 8;
    let n_kv_heads: usize = 2;
    let seq: usize = 4;
    let head_dim: usize = 8;

    let q_dim = n_q_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;

    // Generate deterministic test data.
    let q_data: Vec<f32> = (0..seq * q_dim).map(|i| (i as f32) * 0.02 - 0.5).collect();
    let k_data: Vec<f32> = (0..seq * kv_dim).map(|i| (i as f32) * 0.015 + 0.1).collect();
    let v_data: Vec<f32> = (0..seq * kv_dim).map(|i| ((i % 16) as f32) * 0.1 - 0.7).collect();

    // Reference: decomposed ONNX fixture loaded from file (generated by generate.py).
    let onnx_bytes = hologram_ai_conformance::ort_runner::fixtures::load("gqa_fused_reference")
        .expect("fixture gqa_fused_reference.onnx not found — run generate.py");
    let ort_outputs = run_onnx_all_outputs(
        &onnx_bytes,
        vec![
            OrtInput { name: "Q_flat".into(), shape: vec![seq, q_dim], data: q_data.clone() },
            OrtInput { name: "K_flat".into(), shape: vec![seq, kv_dim], data: k_data.clone() },
            OrtInput { name: "V_flat".into(), shape: vec![seq, kv_dim], data: v_data.clone() },
        ],
    )
    .expect("ORT failed for gqa_flat_multi_kv reference");

    // Fused: AiGraph with GroupedQueryAttention compiled through hologram.
    let graph = build_gqa_aigraph(
        n_q_heads as u32,
        n_kv_heads as u32,
        head_dim as u32,
        seq,
        true, // causal
    );
    let holo_out = compile_and_execute_aigraph(
        graph,
        &[
            ("Q_flat", vec![seq, q_dim], q_data),
            ("K_flat", vec![seq, kv_dim], k_data),
            ("V_flat", vec![seq, kv_dim], v_data),
        ],
    );

    // GQA attention: looser tolerance due to softmax + multiple matmuls.
    let tol = Tolerance { atol: 1e-3, rtol: 1e-2 };
    let cmp = hologram_ai_conformance::tolerance::compare_outputs(
        &holo_out,
        &ort_outputs[0].data,
        tol,
    );
    assert!(
        cmp.passed,
        "Fused GQA kernel mismatch vs decomposed ORT reference: {}",
        cmp.message
    );
}

/// Build a minimal AiGraph with a single FusedSwiGLU op.
///
/// Inputs: gate [rows, cols], up [rows, cols]
/// Output: [rows, cols]
fn build_swiglu_aigraph(rows: usize, cols: usize) -> hologram_ai_common::AiGraph {
    use hologram_ai_common::{
        AiGraph, AiNode, AiOp, DType, TensorInfo, shape_from_concrete,
    };
    use std::collections::HashMap;

    // TensorIds: 0=gate, 1=up, 2=output
    let mut tensor_info = HashMap::new();
    tensor_info.insert(0u32, TensorInfo::new(DType::F32, shape_from_concrete(&[rows as u64, cols as u64])));
    tensor_info.insert(1u32, TensorInfo::new(DType::F32, shape_from_concrete(&[rows as u64, cols as u64])));
    tensor_info.insert(2u32, TensorInfo::new(DType::F32, shape_from_concrete(&[rows as u64, cols as u64])));

    AiGraph {
        name: "swiglu_fused_fixture".into(),
        nodes: vec![AiNode::new(
            0,
            AiOp::FusedSwiGLU,
            vec![0, 1],
            vec![2],
        )],
        inputs: vec![0, 1],
        outputs: vec![2],
        input_names: vec!["gate".into(), "up".into()],
        output_names: vec!["output".into()],
        params: HashMap::new(),
        tensor_info,
        metadata: HashMap::new(),
        warnings: vec![],
        dim_vars: Default::default(),
        shape_constraints: Default::default(),
        subgraphs: HashMap::new(),
        tensor_names: HashMap::new(),
    }
}

/// Test: Fused SwiGLU kernel (AiOp::FusedSwiGLU) matches decomposed ONNX reference.
///
/// SwiGLU = silu(gate) * up = gate * sigmoid(gate) * up
/// The decomposed ONNX graph (Sigmoid + Mul + Mul) is run through ORT as reference.
/// The fused AiGraph path compiles to FloatOp::FusedSwiGLU in hologram.
///
/// A mismatch indicates the fused kernel implementation differs from the
/// standard decomposed computation.
#[test]
fn swiglu_fused_kernel_matches_decomposed() {
    let rows = 4;
    let cols = 16;

    let gate: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.05 - 1.5).collect();
    let up: Vec<f32> = (0..rows * cols).map(|i| ((i % 8) as f32) * 0.2 - 0.7).collect();

    // Reference: decomposed ONNX fixture loaded from file (generated by generate.py).
    let onnx_bytes = hologram_ai_conformance::ort_runner::fixtures::load("swiglu_fused_reference")
        .expect("fixture swiglu_fused_reference.onnx not found — run generate.py");
    let ort_outputs = run_onnx_all_outputs(
        &onnx_bytes,
        vec![
            OrtInput { name: "gate".into(), shape: vec![rows, cols], data: gate.clone() },
            OrtInput { name: "up".into(), shape: vec![rows, cols], data: up.clone() },
        ],
    )
    .expect("ORT failed for swiglu reference");

    // Fused: AiGraph with FusedSwiGLU compiled through hologram.
    let graph = build_swiglu_aigraph(rows, cols);
    let holo_out = compile_and_execute_aigraph(
        graph,
        &[
            ("gate", vec![rows, cols], gate),
            ("up", vec![rows, cols], up),
        ],
    );

    let tol = Tolerance { atol: 1e-5, rtol: 1e-4 };
    let cmp = hologram_ai_conformance::tolerance::compare_outputs(
        &holo_out,
        &ort_outputs[0].data,
        tol,
    );
    assert!(
        cmp.passed,
        "Fused SwiGLU kernel mismatch vs decomposed ORT reference: {}",
        cmp.message
    );
}

// ── Vision / Conv2d conformance ─────────────────────────────────────────────

/// Test: single Conv2d node matches ORT.
#[test]
fn conv2d_matches_ort() {
    let (n, ic, h, w) = (1, 3, 8, 8);
    let (oc, kh, kw) = (4, 3, 3);
    let (stride, pad) = (1, 1);
    let model_bytes = onnx_builder::conv2d(n, ic, h, w, oc, kh, kw, stride, pad);

    let input_data: Vec<f32> = (0..n * ic * h * w)
        .map(|i| ((i as f32) * 0.02).sin())
        .collect();

    let ort_outputs = run_onnx_all_outputs(
        &model_bytes,
        vec![OrtInput { name: "X".into(), shape: vec![n, ic, h, w], data: input_data.clone() }],
    )
    .expect("ORT failed");

    let holo_out = compile_and_execute(
        &model_bytes,
        &[("X", vec![n, ic, h, w], input_data)],
    );

    let tol = Tolerance { atol: 1e-4, rtol: 1e-3 };
    let cmp = hologram_ai_conformance::tolerance::compare_outputs(
        &holo_out, &ort_outputs[0].data, tol,
    );
    assert!(cmp.passed, "Conv2d mismatch: {}", cmp.message);
}

/// Test: Conv2d with stride=2 and no padding matches ORT.
#[test]
fn conv2d_stride2_matches_ort() {
    let (n, ic, h, w) = (1, 3, 8, 8);
    let (oc, kh, kw) = (8, 3, 3);
    let (stride, pad) = (2, 0);
    let model_bytes = onnx_builder::conv2d(n, ic, h, w, oc, kh, kw, stride, pad);

    let input_data: Vec<f32> = (0..n * ic * h * w)
        .map(|i| ((i as f32) * 0.03).cos())
        .collect();

    let ort_outputs = run_onnx_all_outputs(
        &model_bytes,
        vec![OrtInput { name: "X".into(), shape: vec![n, ic, h, w], data: input_data.clone() }],
    )
    .expect("ORT failed");

    let holo_out = compile_and_execute(
        &model_bytes,
        &[("X", vec![n, ic, h, w], input_data)],
    );

    let tol = Tolerance { atol: 1e-4, rtol: 1e-3 };
    let cmp = hologram_ai_conformance::tolerance::compare_outputs(
        &holo_out, &ort_outputs[0].data, tol,
    );
    assert!(cmp.passed, "Conv2d stride=2 mismatch: {}", cmp.message);
}

/// Test: Conv+Relu+GlobalAvgPool matches ORT (no FC layer).
#[test]
fn conv_relu_gap_matches_ort() {
    let (ic, h, w) = (3, 8, 8);
    let oc = 4;
    let model_bytes = onnx_builder::conv_relu_gap(ic, h, w, oc);

    let input_data: Vec<f32> = (0..1 * ic * h * w)
        .map(|i| ((i as f32) * 0.01).sin())
        .collect();

    let ort_outputs = run_onnx_all_outputs(
        &model_bytes,
        vec![OrtInput { name: "X".into(), shape: vec![1, ic, h, w], data: input_data.clone() }],
    )
    .expect("ORT failed");

    let holo_out = compile_and_execute(
        &model_bytes,
        &[("X", vec![1, ic, h, w], input_data)],
    );

    assert!(
        !holo_out.is_empty(),
        "Conv+Relu+GAP produced empty output (ORT got {} floats)",
        ort_outputs[0].data.len()
    );

    let tol = Tolerance { atol: 1e-4, rtol: 1e-3 };
    let cmp = hologram_ai_conformance::tolerance::compare_outputs(
        &holo_out, &ort_outputs[0].data, tol,
    );
    assert!(cmp.passed, "Conv+Relu+GAP mismatch: {}", cmp.message);
}

/// Test: mini vision classifier (Conv+Relu+GlobalAvgPool+Flatten+Gemm) matches ORT.
#[test]
fn mini_vision_classifier_matches_ort() {
    let (ic, h, w) = (3, 8, 8);
    let oc = 4;
    let num_classes = 5;
    let model_bytes = onnx_builder::mini_vision_classifier(ic, h, w, oc, num_classes);

    let input_data: Vec<f32> = (0..1 * ic * h * w)
        .map(|i| ((i as f32) * 0.01).sin())
        .collect();

    let ort_outputs = run_onnx_all_outputs(
        &model_bytes,
        vec![OrtInput { name: "X".into(), shape: vec![1, ic, h, w], data: input_data.clone() }],
    )
    .expect("ORT failed");

    let holo_out = compile_and_execute(
        &model_bytes,
        &[("X", vec![1, ic, h, w], input_data)],
    );

    assert!(
        !holo_out.is_empty(),
        "hologram produced empty output for mini vision classifier"
    );

    let tol = Tolerance { atol: 1e-3, rtol: 1e-2 };
    let cmp = hologram_ai_conformance::tolerance::compare_outputs(
        &holo_out, &ort_outputs[0].data, tol,
    );
    assert!(cmp.passed, "Mini vision classifier mismatch: {}", cmp.message);
}
