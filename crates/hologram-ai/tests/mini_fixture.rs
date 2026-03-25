//! Mini transformer fixture tests — fast CI replacement for TinyLlama file tests.
//!
//! Uses an in-memory 32-node synthetic transformer (hidden=32, heads=2, ffn=64,
//! vocab=32) to test variable seq_len and shape projection without needing model
//! files or ORT. Compile time: <100ms. Execute at seq=128: <5ms.
//!
//! Run with:
//!   cargo test -p hologram-ai -- mini_fixture --nocapture

use hologram_ai_conformance::ort_runner::onnx_builder;

const MINI_HIDDEN: usize = 32;
const MINI_HEADS: usize = 2;
const MINI_FFN: usize = 64;
const MINI_VOCAB: usize = 32;

#[cfg(feature = "e2e")]
fn workspace_path(rel: &str) -> std::path::PathBuf {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p.push(rel);
    p
}

#[cfg(feature = "e2e")]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in 0..n {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

/// Mini transformer at multiple seq_len values.
///
/// Compiles the mini transformer separately for each seq_len (1, 7, 128) and
/// verifies output shape `[seq * vocab]` with no NaN/Inf values.
///
/// The tape executor bakes MatMul output dimensions at compile time, so each
/// seq_len requires its own compilation. Variable-length prefill within a single
/// compiled model is handled by `resolve_size()` for ops like Softmax/RmsNorm,
/// but the final output shape matches the compiled seq_len.
#[test]
fn mini_transformer_variable_seq_len_runs() {
    let model_bytes =
        onnx_builder::mini_transformer_dyn(MINI_HIDDEN, MINI_HEADS, MINI_FFN, MINI_VOCAB);

    for seq in [1usize, 7, 128] {
        let compiler = hologram_ai::ModelCompiler {
            seq_len_override: Some(seq as u64),
            ..Default::default()
        };
        let archive = compiler
            .compile(hologram_ai::ModelSource::OnnxBytes(model_bytes.clone()))
            .unwrap_or_else(|e| panic!("mini transformer compilation for seq={seq} failed: {e}"));

        let x: Vec<f32> = (0..seq * MINI_HIDDEN).map(|i| (i as f32) * 0.01 - 0.32).collect();
        let x_bytes: Vec<u8> = bytemuck::cast_slice(&x).to_vec();

        let mut graph_inputs = hologram::GraphInputs::new();
        graph_inputs.set_with_shape(0, x_bytes, vec![seq, MINI_HIDDEN]);

        let outputs = hologram_ai::run_with_shape_context(&archive, &graph_inputs)
            .unwrap_or_else(|e| panic!("mini transformer seq={seq} failed: {e}"));

        let (_, out_bytes) = outputs.get(0).expect("no output");
        let out_f32: &[f32] = bytemuck::cast_slice(out_bytes);

        assert_eq!(
            out_f32.len(),
            seq * MINI_VOCAB,
            "seq={seq}: expected {} output elements, got {}",
            seq * MINI_VOCAB,
            out_f32.len()
        );
        assert!(
            out_f32.iter().all(|v| v.is_finite()),
            "seq={seq}: output contains NaN/Inf"
        );
        eprintln!("seq={seq}: OK ({} output f32s)", out_f32.len());
    }
}

/// GGUF causal logit consistency — logits at position P must be identical
/// whether the sequence has length P+1 or P+2 (causal attention invariant).
///
/// Compiles TinyLlama GGUF, runs at seq=6 and seq=7 with the same first 6 tokens.
/// Compares logits at position 5 between both runs. If cosine similarity < 0.99,
/// the executor's variable-seq computation is broken.
#[test]
#[cfg(feature = "e2e")]
fn gguf_causal_logit_consistency() {
    let model = workspace_path(
        "models/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_0.gguf",
    );
    if !model.exists() {
        eprintln!("SKIP: {} not found", model.display());
        return;
    }

    let archive = hologram_ai::ModelCompiler::default()
        .compile(hologram_ai::ModelSource::GgufPath(model))
        .expect("GGUF compilation failed");

    let runner = hologram_ai::compiler::HoloRunner::from_bytes(archive.bytes.clone())
        .expect("HoloRunner creation failed");

    let vocab_size = 32000usize;
    let bytes_per_pos = vocab_size * 4;

    // Token IDs: just use sequential small values.
    // Test with both small gap (6→7) and large gap (6→46) to catch
    // degeneration at larger seq_len.
    let base_tokens: Vec<i64> = vec![1, 4007, 526, 263, 14225, 20255];
    let extended_tokens: Vec<i64> = {
        let mut v = base_tokens.clone();
        v.push(29871); // space token
        v
    };
    let far_tokens: Vec<i64> = {
        let mut v = base_tokens.clone();
        // Add 40 more tokens (simulating a 46-token sequence).
        for i in 0..40 {
            v.push(100 + i);
        }
        v
    };

    // Run at seq=6.
    let mut inputs_6 = hologram::GraphInputs::new();
    let bytes_6: Vec<u8> = bytemuck::cast_slice(&base_tokens).to_vec();
    inputs_6.set_with_shape(0, bytes_6, vec![1, base_tokens.len()]);
    let out_6 = runner.execute(&inputs_6).expect("seq=6 failed");
    let (_, logit_data_6) = out_6.get(0).expect("no output at seq=6");

    // Run at seq=7.
    let mut inputs_7 = hologram::GraphInputs::new();
    let bytes_7: Vec<u8> = bytemuck::cast_slice(&extended_tokens).to_vec();
    inputs_7.set_with_shape(0, bytes_7, vec![1, extended_tokens.len()]);
    let out_7 = runner.execute(&inputs_7).expect("seq=7 failed");
    let (_, logit_data_7) = out_7.get(0).expect("no output at seq=7");

    // Extract logits at position 5 from both runs.
    let pos = 5;
    let offset_6 = pos * bytes_per_pos;
    let offset_7 = pos * bytes_per_pos;

    assert!(
        logit_data_6.len() >= offset_6 + bytes_per_pos,
        "seq=6 output too short: {} < {}",
        logit_data_6.len(),
        offset_6 + bytes_per_pos
    );
    assert!(
        logit_data_7.len() >= offset_7 + bytes_per_pos,
        "seq=7 output too short: {} < {}",
        logit_data_7.len(),
        offset_7 + bytes_per_pos
    );

    let logits_6: &[f32] =
        bytemuck::cast_slice(&logit_data_6[offset_6..offset_6 + bytes_per_pos]);
    let logits_7: &[f32] =
        bytemuck::cast_slice(&logit_data_7[offset_7..offset_7 + bytes_per_pos]);

    let cos_sim = cosine_similarity(logits_6, logits_7);
    eprintln!(
        "causal consistency: cos_sim(seq=6[pos=5], seq=7[pos=5]) = {cos_sim:.6}"
    );

    // Also compare top-1 tokens.
    let top1_6 = logits_6
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("NaN in logits"))
        .map(|(i, v)| (i, *v));
    let top1_7 = logits_7
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("NaN in logits"))
        .map(|(i, v)| (i, *v));
    eprintln!("  seq=6 top-1: {:?}", top1_6);
    eprintln!("  seq=7 top-1: {:?}", top1_7);

    assert!(
        cos_sim > 0.99,
        "causal invariant violated: cos_sim={cos_sim:.6} (expected > 0.99). \
         Logits at position 5 should be identical regardless of sequence length."
    );

    // Also test with far sequence (seq=46) — catches degeneration at larger seq_len.
    let mut inputs_far = hologram::GraphInputs::new();
    let bytes_far: Vec<u8> = bytemuck::cast_slice(&far_tokens).to_vec();
    inputs_far.set_with_shape(0, bytes_far, vec![1, far_tokens.len()]);
    let out_far = runner.execute(&inputs_far).expect("seq=46 failed");
    let (_, logit_data_far) = out_far.get(0).expect("no output at seq=46");

    let offset_far = pos * bytes_per_pos;
    assert!(
        logit_data_far.len() >= offset_far + bytes_per_pos,
        "seq=46 output too short"
    );
    let logits_far: &[f32] =
        bytemuck::cast_slice(&logit_data_far[offset_far..offset_far + bytes_per_pos]);
    let cos_sim_far = cosine_similarity(logits_6, logits_far);
    eprintln!(
        "far causal consistency: cos_sim(seq=6[pos=5], seq=46[pos=5]) = {cos_sim_far:.6}"
    );
    assert!(
        cos_sim_far > 0.99,
        "far causal invariant violated: cos_sim={cos_sim_far:.6}"
    );
}
