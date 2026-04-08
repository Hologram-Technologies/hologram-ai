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

        let x: Vec<f32> = (0..seq * MINI_HIDDEN)
            .map(|i| (i as f32) * 0.01 - 0.32)
            .collect();
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

/// KV cache decode conformance — logits from decode step must match
/// full-prefill logits at the same position.
///
/// Compiles TinyLlama ONNX at seq=7. Runs two scenarios:
/// 1. "Prefill reference": all 7 tokens as a single prefill → logits at pos 6
/// 2. "KV decode": 6 tokens as prefill, then 1 token via KV cache decode → decode logits
///
/// If the KV cache read/write cycle is correct, both produce identical logits
/// at the last position (cosine similarity > 0.99, top-1 token match).
#[test]
#[cfg(feature = "e2e")]
fn onnx_kv_decode_matches_full_prefill() {
    let causal = workspace_path("models/TinyLlama-1.1B-Chat-v1.0/model_causal.onnx");
    let fallback = workspace_path("models/TinyLlama-1.1B-Chat-v1.0/model.onnx");
    let model = if causal.exists() { causal } else { fallback };
    if !model.exists() {
        eprintln!("SKIP: no TinyLlama ONNX model found");
        return;
    }

    let vocab_size = 32000usize;
    let bytes_per_pos = vocab_size * 4;

    // "The capital of France is" + token 24241 ("verte")
    let tokens_7: Vec<i64> = vec![1, 450, 7483, 310, 3444, 338, 24241];
    let tokens_6: Vec<i64> = tokens_7[..6].to_vec();
    let decode_token: i64 = tokens_7[6];

    // ── Compile at seq=7 ────────────────────────────────────────────────
    let compiler = hologram_ai::ModelCompiler {
        seq_len_override: Some(7),
        ..Default::default()
    };
    let archive = compiler
        .compile(hologram_ai::ModelSource::OnnxPath(model))
        .unwrap_or_else(|e| panic!("compilation failed: {e}"));
    let runner = hologram_ai::compiler::HoloRunner::from_bytes(archive.bytes)
        .unwrap_or_else(|e| panic!("HoloRunner creation failed: {e}"));

    let graph = runner.plan().graph();
    let input_slot = graph
        .input_names
        .iter()
        .position(|n| n == "input_ids")
        .unwrap_or(0) as u32;
    let mask_slot = graph
        .input_names
        .iter()
        .position(|n| n == "attention_mask")
        .map(|i| i as u32);
    let pos_slot = graph
        .input_names
        .iter()
        .position(|n| n == "position_ids")
        .map(|i| i as u32);

    // Read KV metadata — embedded by compile() in the pipeline wrapper.
    let meta = {
        use hologram::hologram_archive::section::model_meta::{
            ModelMetaSection, SECTION_MODEL_META,
        };
        let bytes = runner.archive_bytes();
        // Try runner's plan first, then re-parse full archive (pipeline wrapper).
        runner
            .plan()
            .sections()
            .find(SECTION_MODEL_META)
            .and_then(|entry| {
                let s = entry.offset as usize;
                let e = s + entry.size as usize;
                if e <= bytes.len() {
                    ModelMetaSection::deserialize_from(&bytes[s..e]).ok()
                } else {
                    None
                }
            })
            .or_else(|| {
                let full_plan = hologram::load_from_bytes(bytes).ok()?;
                let entry = full_plan.sections().find(SECTION_MODEL_META)?;
                let s = entry.offset as usize;
                let e = s + entry.size as usize;
                if e <= bytes.len() {
                    ModelMetaSection::deserialize_from(&bytes[s..e]).ok()
                } else {
                    None
                }
            })
    };
    // Fall back to known TinyLlama architecture params if metadata not found.
    let (n_layers, n_kv_heads, head_dim, max_seq) = match &meta {
        Some(m) if m.n_layers > 0 => (m.n_layers, m.n_kv_heads, m.head_dim, m.max_seq_len as usize),
        _ => (22, 4, 64, 2048), // TinyLlama 1.1B defaults
    };

    let build_inputs = |tokens: &[i64], pos_offset: usize| {
        let seq = tokens.len();
        let mut inputs = hologram::GraphInputs::new();
        inputs.set_with_shape(
            input_slot,
            tokens.iter().flat_map(|&v| v.to_le_bytes()).collect(),
            vec![1, seq],
        );
        if let Some(slot) = mask_slot {
            inputs.set_with_shape(
                slot,
                (0..seq).flat_map(|_| 1i64.to_le_bytes()).collect(),
                vec![1, seq],
            );
        }
        if let Some(slot) = pos_slot {
            inputs.set_with_shape(
                slot,
                (0..seq as i64)
                    .map(|i| pos_offset as i64 + i)
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
                vec![1, seq],
            );
        }
        inputs
    };

    // ── Scenario 1: Full prefill (7 tokens) ─────────────────────────────
    let inputs_7 = build_inputs(&tokens_7, 0);
    let mut kv_ref = hologram::KvCacheState::new(n_layers, n_kv_heads, head_dim, max_seq);
    let out_ref = runner
        .execute_with_kv(&inputs_7, &mut kv_ref)
        .unwrap_or_else(|e| panic!("prefill-7 failed: {e}"));
    let (_, ref_bytes) = out_ref.get(0).expect("no prefill output");

    let pos6_offset = 6 * bytes_per_pos;
    assert!(
        ref_bytes.len() >= pos6_offset + bytes_per_pos,
        "prefill output too short"
    );
    let ref_logits: &[f32] =
        bytemuck::cast_slice(&ref_bytes[pos6_offset..pos6_offset + bytes_per_pos]);

    // ── Scenario 2: 6-token prefill + 1-token KV decode ────────────────
    let inputs_6 = build_inputs(&tokens_6, 0);
    let mut kv_dec = hologram::KvCacheState::new(n_layers, n_kv_heads, head_dim, max_seq);
    let _out_6 = runner
        .execute_with_kv(&inputs_6, &mut kv_dec)
        .unwrap_or_else(|e| panic!("prefill-6 failed: {e}"));
    assert_eq!(
        kv_dec.write_pos(),
        6,
        "KV write_pos should be 6 after prefill"
    );

    let inputs_dec = build_inputs(&[decode_token], kv_dec.write_pos());
    let out_dec = runner
        .execute_with_kv(&inputs_dec, &mut kv_dec)
        .unwrap_or_else(|e| panic!("decode step failed: {e}"));
    let (_, dec_bytes) = out_dec.get(0).expect("no decode output");
    assert!(dec_bytes.len() >= bytes_per_pos, "decode output too short");
    let dec_logits: &[f32] = bytemuck::cast_slice(&dec_bytes[..bytes_per_pos]);

    // ── Compare ─────────────────────────────────────────────────────────
    let cos = cosine_similarity(ref_logits, dec_logits);

    let top1 = |logits: &[f32]| -> (usize, f32) {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, &v)| (i, v))
            .unwrap_or((0, 0.0))
    };
    let (ref_id, ref_val) = top1(ref_logits);
    let (dec_id, dec_val) = top1(dec_logits);

    eprintln!("KV decode conformance:");
    eprintln!("  prefill-7 top-1: id={ref_id} val={ref_val:.4}");
    eprintln!("  decode    top-1: id={dec_id} val={dec_val:.4}");
    eprintln!("  cosine similarity: {cos:.6}");

    assert_eq!(
        ref_id, dec_id,
        "KV decode top-1 mismatch: prefill=({ref_id}, {ref_val:.4}) decode=({dec_id}, {dec_val:.4}). \
         cos_sim={cos:.6}. The decode path produces different predictions than full prefill.",
    );
    assert!(
        cos > 0.99,
        "KV decode logit divergence: cos_sim={cos:.6} (expected > 0.99).",
    );
}

/// Same as above but compiled at seq=32 (mismatched from actual seq=6/1).
/// This tests variable-length runtime shape resolution during KV decode.
#[test]
#[cfg(feature = "e2e")]
fn onnx_kv_decode_variable_length() {
    let causal = workspace_path("models/TinyLlama-1.1B-Chat-v1.0/model_causal.onnx");
    let fallback = workspace_path("models/TinyLlama-1.1B-Chat-v1.0/model.onnx");
    let model = if causal.exists() { causal } else { fallback };
    if !model.exists() {
        eprintln!("SKIP: no TinyLlama ONNX model found");
        return;
    }

    let vocab_size = 32000usize;
    let bytes_per_pos = vocab_size * 4;

    let tokens_7: Vec<i64> = vec![1, 450, 7483, 310, 3444, 338, 24241];
    let tokens_6: Vec<i64> = tokens_7[..6].to_vec();
    let decode_token: i64 = tokens_7[6];

    // Compile at seq=32 — actual usage will be seq=6 and seq=1 (variable).
    let compiler = hologram_ai::ModelCompiler {
        seq_len_override: Some(32),
        ..Default::default()
    };
    let archive = compiler
        .compile(hologram_ai::ModelSource::OnnxPath(model))
        .unwrap_or_else(|e| panic!("compilation failed: {e}"));
    let runner = hologram_ai::compiler::HoloRunner::from_bytes(archive.bytes)
        .unwrap_or_else(|e| panic!("HoloRunner creation failed: {e}"));

    let graph = runner.plan().graph();
    let input_slot = graph
        .input_names
        .iter()
        .position(|n| n == "input_ids")
        .unwrap_or(0) as u32;
    let mask_slot = graph
        .input_names
        .iter()
        .position(|n| n == "attention_mask")
        .map(|i| i as u32);
    let pos_slot = graph
        .input_names
        .iter()
        .position(|n| n == "position_ids")
        .map(|i| i as u32);

    let (n_layers, n_kv_heads, head_dim, max_seq) = (22u32, 4u32, 64u32, 2048usize);

    let build_inputs = |tokens: &[i64], pos_offset: usize| {
        let seq = tokens.len();
        let mut inputs = hologram::GraphInputs::new();
        inputs.set_with_shape(
            input_slot,
            tokens.iter().flat_map(|&v| v.to_le_bytes()).collect(),
            vec![1, seq],
        );
        if let Some(slot) = mask_slot {
            inputs.set_with_shape(
                slot,
                (0..seq).flat_map(|_| 1i64.to_le_bytes()).collect(),
                vec![1, seq],
            );
        }
        if let Some(slot) = pos_slot {
            inputs.set_with_shape(
                slot,
                (0..seq as i64)
                    .map(|i| pos_offset as i64 + i)
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
                vec![1, seq],
            );
        }
        inputs
    };

    // Reference: 7-token prefill at seq=32 (variable-length).
    let inputs_7 = build_inputs(&tokens_7, 0);
    let mut kv_ref = hologram::KvCacheState::new(n_layers, n_kv_heads, head_dim, max_seq);
    let out_ref = runner
        .execute_with_kv(&inputs_7, &mut kv_ref)
        .unwrap_or_else(|e| panic!("prefill-7 failed: {e}"));
    let (_, ref_bytes) = out_ref.get(0).expect("no prefill output");
    let pos6_offset = 6 * bytes_per_pos;
    assert!(
        ref_bytes.len() >= pos6_offset + bytes_per_pos,
        "prefill output too short"
    );
    let ref_logits: &[f32] =
        bytemuck::cast_slice(&ref_bytes[pos6_offset..pos6_offset + bytes_per_pos]);

    // Decode: 6-token prefill + 1-token decode at seq=32.
    let inputs_6 = build_inputs(&tokens_6, 0);
    let mut kv_dec = hologram::KvCacheState::new(n_layers, n_kv_heads, head_dim, max_seq);
    let _out_6 = runner
        .execute_with_kv(&inputs_6, &mut kv_dec)
        .unwrap_or_else(|e| panic!("prefill-6 failed: {e}"));
    assert_eq!(kv_dec.write_pos(), 6);

    let inputs_dec = build_inputs(&[decode_token], kv_dec.write_pos());
    let out_dec = runner
        .execute_with_kv(&inputs_dec, &mut kv_dec)
        .unwrap_or_else(|e| panic!("decode step failed: {e}"));
    let (_, dec_bytes) = out_dec.get(0).expect("no decode output");
    assert!(dec_bytes.len() >= bytes_per_pos, "decode output too short");
    let dec_logits: &[f32] = bytemuck::cast_slice(&dec_bytes[..bytes_per_pos]);

    let cos = cosine_similarity(ref_logits, dec_logits);
    let top1 = |logits: &[f32]| -> (usize, f32) {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, &v)| (i, v))
            .unwrap_or((0, 0.0))
    };
    let (ref_id, ref_val) = top1(ref_logits);
    let (dec_id, dec_val) = top1(dec_logits);

    eprintln!("KV decode variable-length conformance (compiled seq=32, actual seq=6+1):");
    eprintln!("  prefill-7 top-1: id={ref_id} val={ref_val:.4}");
    eprintln!("  decode    top-1: id={dec_id} val={dec_val:.4}");
    eprintln!("  cosine similarity: {cos:.6}");

    assert_eq!(
        ref_id, dec_id,
        "Variable-length KV decode top-1 mismatch: prefill=({ref_id}, {ref_val:.4}) \
         decode=({dec_id}, {dec_val:.4}). cos_sim={cos:.6}.",
    );
    assert!(
        cos > 0.99,
        "Variable-length KV decode logit divergence: cos_sim={cos:.6}.",
    );
}
