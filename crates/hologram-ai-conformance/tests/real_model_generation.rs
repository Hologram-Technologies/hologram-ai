//! **Real pretrained-LLM generation (V&V class EE).**
//!
//! Compiles the authoritative onnx-community SmolLM2-135M-Instruct export and
//! drives the actual `generate_stream` loop (the `run --prompt` path) to produce
//! real text, asserting it is coherent, prompt-relevant, and deterministic under
//! greedy decoding. This is the end-to-end witness that hologram-ai runs a real
//! Llama-family decoder — RoPE + grouped-query causal attention + RMSNorm +
//! SwiGLU + tied LM head — and that the with-past export runs as an empty-past
//! full-recompute prefill (hologram-ai has no external KV-cache; reuse is
//! content-addressed κ-label elision).
//!
//! Skip-safe: runs only with `HOLOGRAM_AI_LIVE=1` and the (git-ignored, ≈540 MB)
//! model present via `HOLOGRAM_AI_SMOLLM2_ONNX` or
//! `<workspace>/models/smollm2-135m/{model.onnx,tokenizer.json}`. Build with
//! `--features onnx-spec`.
#![cfg(feature = "onnx-spec")]

use std::path::PathBuf;

use hologram_ai::commands::generate::{generate_stream, GenConfig};
use hologram_ai::{GrowableSession, ModelCompiler, ModelSource};
use hologram_ai_tokenizer::NativeTokenizer;

fn model_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("HOLOGRAM_AI_SMOLLM2_ONNX") {
        let p = PathBuf::from(p);
        return p.exists().then_some(p);
    }
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/smollm2-135m/model.onnx");
    p.exists().then_some(p)
}

#[test]
fn smollm2_generates_coherent_text() {
    if std::env::var("HOLOGRAM_AI_LIVE").as_deref() != Ok("1") {
        eprintln!("skipping: set HOLOGRAM_AI_LIVE=1 to run real-model generation");
        return;
    }
    let Some(onnx) = model_path() else {
        eprintln!("skipping: SmolLM2 model not present");
        return;
    };
    let tok_path = onnx.with_file_name("tokenizer.json");
    if !tok_path.exists() {
        eprintln!("skipping: tokenizer.json not next to the model");
        return;
    }

    // Drive the length-adaptive engine straight from the model source: the
    // window is compiled on demand and grows up to the model's real context —
    // no baked seq_len, no artificial cap on prompt or output length. This is
    // the production `run <source> --prompt` path.
    let prepared = ModelCompiler::default()
        .prepare(ModelSource::OnnxPath(onnx))
        .expect("prepare SmolLM2");
    let mut provider = GrowableSession::new(prepared);
    let tok = NativeTokenizer::from_tokenizer_json(&tok_path).expect("load tokenizer");

    // Greedy (temperature 0) ⇒ deterministic, so the output is a stable witness.
    let cfg = GenConfig {
        max_tokens: 10,
        temperature: 0.0,
        ..Default::default()
    };

    let mut gen = |prompt: &str| {
        let mut sink = Vec::new();
        generate_stream(&mut provider, &tok, prompt, &cfg, &mut sink).expect("generate")
    };

    // Correctness: greedy decoding must produce the factually-correct answer
    // (this exact continuation is verified to match ONNX Runtime token-for-token
    // by EE-3 / scripts/verify_logits_vs_ort.py).
    let france = gen("The capital of France is");
    println!("[gen] The capital of France is →{france}");
    assert!(
        france.contains("Paris"),
        "expected the correct answer (Paris), got: {france:?}"
    );

    // Coherence: a real continuation is multi-word English (not a single token,
    // not whitespace), and every char is valid (generate_stream returns a String).
    let sun = gen("The sun rises in the");
    println!("[gen] The sun rises in the →{sun}");
    assert!(
        sun.split_whitespace().count() >= 3,
        "expected a multi-word continuation, got: {sun:?}"
    );

    // Determinism: greedy decoding is reproducible run-to-run.
    assert_eq!(
        gen("The sun rises in the"),
        sun,
        "greedy decoding must be deterministic"
    );
}
