//! **Real Qwen2-0.5B-Instruct generation (V&V class EE).**
//!
//! Sibling of `real_model_generation.rs` (SmolLM2). Compiles the
//! authoritative Qwen2-0.5B-Instruct ONNX export and drives the
//! `generate_stream` loop. Qwen2 exercises code paths SmolLM2 does
//! not: grouped-query attention with kv_heads=2 (Q heads=14 → 7:1
//! GQA ratio), an explicit `logits` graph output (no LM-head
//! injection), and the full rank-3 norm-projection / down-projection
//! chain through `desugar_norm_projection`. This is the regression
//! witness behind the UOR-native fix that eradicated rank-3 MatMul
//! shape corruption and Gemm `trans_b` silent drop (commit 61b26de).
//!
//! Skip-safe: runs only with `HOLOGRAM_AI_LIVE=1` and the (git-ignored)
//! model present via `HOLOGRAM_AI_QWEN2_ONNX` or
//! `<workspace>/models/Qwen2-0.5B-Instruct/{model.onnx,tokenizer.json}`.
//! Build with `--features onnx-spec`.
#![cfg(feature = "onnx-spec")]

use std::path::PathBuf;

use hologram_ai::commands::generate::{generate_stream, GenConfig};
use hologram_ai::{GrowableSession, ModelCompiler, ModelSource};
use hologram_ai_tokenizer::NativeTokenizer;

fn model_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("HOLOGRAM_AI_QWEN2_ONNX") {
        let p = PathBuf::from(p);
        return p.exists().then_some(p);
    }
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../models/Qwen2-0.5B-Instruct/model.onnx");
    p.exists().then_some(p)
}

#[test]
fn qwen2_generates_coherent_text() {
    if std::env::var("HOLOGRAM_AI_LIVE").as_deref() != Ok("1") {
        eprintln!("skipping: set HOLOGRAM_AI_LIVE=1 to run real-model generation");
        return;
    }
    let Some(onnx) = model_path() else {
        eprintln!("skipping: Qwen2 model not present");
        return;
    };
    let tok_path = onnx.with_file_name("tokenizer.json");
    if !tok_path.exists() {
        eprintln!("skipping: tokenizer.json not next to the model");
        return;
    }

    let prepared = ModelCompiler::default()
        .prepare(ModelSource::OnnxPath(onnx))
        .expect("prepare Qwen2");
    let mut provider = GrowableSession::new(prepared);
    let tok = NativeTokenizer::from_tokenizer_json(&tok_path).expect("load tokenizer");

    let cfg = GenConfig {
        max_tokens: 20,
        temperature: 0.0,
        ..Default::default()
    };

    let mut gen = |prompt: &str| {
        let mut sink = Vec::new();
        generate_stream(&mut provider, &tok, prompt, &cfg, &mut sink).expect("generate")
    };

    // Correctness: greedy decoding must produce the factually-correct
    // answer. Qwen2-0.5B-Instruct answers "The capital of France is"
    // with an instruct-style multiple-choice that lists Paris first.
    let france = gen("The capital of France is");
    println!("[gen] The capital of France is →{france}");
    assert!(
        france.contains("Paris"),
        "expected the correct answer (Paris), got: {france:?}"
    );

    // Coherence: a real continuation is multi-word English.
    let hello = gen("Hello, my name is");
    println!("[gen] Hello, my name is →{hello}");
    assert!(
        hello.split_whitespace().count() >= 3,
        "expected a multi-word continuation, got: {hello:?}"
    );

    // Determinism: greedy decoding is reproducible run-to-run.
    assert_eq!(
        gen("Hello, my name is"),
        hello,
        "greedy decoding must be deterministic"
    );
}
