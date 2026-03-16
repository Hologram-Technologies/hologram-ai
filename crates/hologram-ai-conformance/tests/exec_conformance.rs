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

    // Extract first output as f32.
    let (_, out_bytes) = outputs.get(0).expect("no outputs");
    bytemuck::cast_slice::<u8, f32>(out_bytes).to_vec()
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
#[test]
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

    // Helper closure: run hologram with given seq-len inputs (using a compiled archive).
    let run_hologram = |archive: &hologram_ai::HoloArchive, ids: &[i64], mask: &[i64]| {
        let seq = ids.len();
        let id_bytes: Vec<u8> = ids.iter().flat_map(|&v| v.to_le_bytes()).collect();
        let mask_bytes: Vec<u8> = mask.iter().flat_map(|&v| v.to_le_bytes()).collect();
        let mut graph_inputs = hologram::GraphInputs::new();
        graph_inputs.set_with_shape(0, id_bytes, vec![1, seq]);
        graph_inputs.set_with_shape(1, mask_bytes, vec![1, seq]);
        let plan = hologram::load_from_bytes(&archive.bytes).expect("loading archive");
        let outputs = hologram::execute_plan(&plan, &graph_inputs).expect("hologram execution failed");
        let (_, holo_bytes) = outputs.get(0).expect("no hologram outputs");
        bytemuck::cast_slice::<u8, f32>(holo_bytes).to_vec()
    };

    // Hologram: compile from path (same as CLI to avoid OnnxBytes parsing differences).
    // Compiled shapes are concretized with seq=1.
    let compiler = hologram_ai::ModelCompiler::default();
    let (archive, _) = compiler
        .compile_with_debug_info(hologram_ai::ModelSource::OnnxPath(model_path.clone()))
        .expect("hologram compilation failed");

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
    let seq = 2usize;
    let input_ids: Vec<i64> = vec![1, 2]; // BOS + second token
    let attention_mask: Vec<i64> = vec![1, 1];

    // ORT reference: run from file (large model — file-based loading avoids memory issues).
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
    // logits shape [1, seq, vocab] = [1, 2, 32000]. Take last position.
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

    // Compare top-5 token IDs (not requiring exact logit values, just ranking agreement).
    let top5_ort = top_k(ort_last_pos, 5);
    let top5_holo = top_k(holo_last_pos, 5);

    eprintln!("ORT top-5:     {:?}", top5_ort);
    eprintln!("hologram top-5: {:?}", top5_holo);

    // The top-1 token must match.
    assert_eq!(
        top5_ort[0], top5_holo[0],
        "top-1 token mismatch: ORT predicted {:?} but hologram predicted {:?}. \
         This indicates hologram's full TinyLlama computation diverges from ORT. \
         ORT top-5: {:?}, hologram top-5: {:?}",
        top5_ort[0], top5_holo[0], top5_ort, top5_holo
    );
}

/// Return indices of the top-k largest values.
fn top_k(logits: &[f32], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.into_iter().take(k).map(|(i, _)| i).collect()
}
