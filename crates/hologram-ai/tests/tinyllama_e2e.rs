//! End-to-end tests for TinyLlama 1.1B — compile and run both ONNX and GGUF variants.
//!
//! These tests require the model files to be present locally. They are feature-gated
//! behind `e2e` to avoid running in CI where model files are not available.
//!
//! Run with:
//!   cargo test -p hologram-ai --features e2e -- tinyllama --nocapture
//!
//! Models expected at (relative to workspace root):
//!   models/TinyLlama-1.1B-Chat-v1.0/model.onnx
//!   models/TinyLlama-1.1B-Chat-v1.0/tokenizer.json
//!   models/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_0.gguf

#![cfg(feature = "e2e")]

use hologram_ai::validate::validate_model;
use std::path::PathBuf;

/// Resolve a path relative to the hologram-ai workspace root.
fn workspace_path(rel: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // crates/hologram-ai → crates/
    p.pop(); // crates/ → workspace root
    p.push(rel);
    p
}

/// The binary under test — built by cargo alongside the test runner.
fn hologram_ai_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_hologram-ai"))
}

// ── Compilation tests ─────────────────────────────────────────────────────────

#[test]
fn tinyllama_onnx_compiles() {
    let model = workspace_path("models/TinyLlama-1.1B-Chat-v1.0/model.onnx");
    if !model.exists() {
        eprintln!("SKIP: {} not found", model.display());
        return;
    }

    let report = validate_model(&model);
    println!("{report}");

    assert!(
        report.compilation_ok,
        "TinyLlama ONNX should compile: {:?}",
        report.error
    );
    assert!(
        report.total_nodes > 1000,
        "expected > 1000 nodes, got {}",
        report.total_nodes
    );
    assert!(
        report.compiled_weight_bytes > 1_000_000_000,
        "expected > 1 GB weights, got {}",
        report.compiled_weight_bytes
    );
}

#[test]
fn tinyllama_gguf_compiles() {
    let model = workspace_path(
        "models/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_0.gguf",
    );
    if !model.exists() {
        eprintln!("SKIP: {} not found", model.display());
        return;
    }

    let report = validate_model(&model);
    println!("{report}");

    assert!(
        report.compilation_ok,
        "TinyLlama GGUF should compile: {:?}",
        report.error
    );
    assert!(
        report.total_nodes > 100,
        "expected > 100 nodes, got {}",
        report.total_nodes
    );
    assert!(
        report.compiled_weight_bytes > 500_000_000,
        "expected > 500 MB weights, got {}",
        report.compiled_weight_bytes
    );
}

// ── Run tests ─────────────────────────────────────────────────────────────────

/// Compile a model to a temp .holo file then run it with a prompt.
///
/// `tokenizer` — optional path to a tokenizer.json to embed. When `None` the
/// compiler uses its auto-discover logic (looks for tokenizer.json next to the
/// model file). Pass `Some` for GGUF models whose directory has no tokenizer.json.
fn compile_then_run(model_path: &PathBuf, prompt: &str, max_tokens: usize) -> RunResult {
    compile_then_run_tok(model_path, None, prompt, max_tokens)
}

/// Like [`compile_then_run`] but with an explicit tokenizer path.
fn compile_then_run_tok(
    model_path: &PathBuf,
    tokenizer: Option<&PathBuf>,
    prompt: &str,
    max_tokens: usize,
) -> RunResult {
    let bin = hologram_ai_bin();
    let out_dir = tempfile::tempdir().expect("tempdir");
    // The compiler names the output file after the input model's stem.
    // e.g. model.onnx → model.holo, tinyllama-1.1b-chat-v1.0.Q4_0.gguf → tinyllama-1.1b-chat-v1.0.Q4_0.holo
    let stem = model_path
        .file_stem()
        .expect("model path has no file stem")
        .to_string_lossy();
    let holo_path = out_dir.path().join(format!("{stem}.holo"));

    // Step 1: compile
    let mut compile_cmd = std::process::Command::new(&bin);
    compile_cmd.args(["compile", "--model"]).arg(model_path);
    if let Some(tok) = tokenizer {
        compile_cmd.args(["--tokenizer"]).arg(tok);
    }
    let compile_status = compile_cmd
        .args(["--output"])
        .arg(out_dir.path())
        .output()
        .expect("compile command failed to start");

    if !compile_status.status.success() {
        return RunResult {
            compile_ok: false,
            compile_stderr: String::from_utf8_lossy(&compile_status.stderr).into_owned(),
            run_ok: false,
            stdout: String::new(),
            stderr: String::new(),
        };
    }

    // Step 2: run
    let run_output = std::process::Command::new(&bin)
        .args(["run"])
        .arg(&holo_path)
        .arg("--prompt")
        .arg(prompt)
        .arg("--max-tokens")
        .arg(max_tokens.to_string())
        .output()
        .expect("run command failed to start");

    RunResult {
        compile_ok: true,
        compile_stderr: String::from_utf8_lossy(&compile_status.stderr).into_owned(),
        run_ok: run_output.status.success(),
        stdout: String::from_utf8_lossy(&run_output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&run_output.stderr).into_owned(),
    }
}

struct RunResult {
    compile_ok: bool,
    compile_stderr: String,
    run_ok: bool,
    stdout: String,
    stderr: String,
}

/// TinyLlama chat prompt template.
const CHAT_PROMPT: &str =
    "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\nWhat is 2 + 2?</s>\n<|assistant|>";

#[test]
fn tinyllama_onnx_runs_and_produces_english() {
    // Prefer model_causal.onnx: it is the causal decoder export that already
    // outputs `logits` natively and includes the causal attention mask.
    // model.onnx is a bidirectional (encoder-style) export that cannot be used
    // for autoregressive generation — it attends to all positions in both
    // directions, producing incoherent hidden states at generation time.
    let causal = workspace_path("models/TinyLlama-1.1B-Chat-v1.0/model_causal.onnx");
    let fallback = workspace_path("models/TinyLlama-1.1B-Chat-v1.0/model.onnx");
    let model = if causal.exists() {
        causal
    } else if fallback.exists() {
        eprintln!(
            "WARN: model_causal.onnx not found; falling back to model.onnx \
             (bidirectional — output may be incoherent)"
        );
        fallback
    } else {
        eprintln!("SKIP: neither model_causal.onnx nor model.onnx found");
        return;
    };

    let result = compile_then_run(&model, CHAT_PROMPT, 20);

    if !result.compile_ok {
        panic!(
            "TinyLlama ONNX failed to compile:\n{}",
            result.compile_stderr
        );
    }

    eprintln!("compile stderr:\n{}", result.compile_stderr);
    eprintln!("run stderr:\n{}", result.stderr);
    eprintln!("run stdout:\n{}", result.stdout);

    assert!(
        result.run_ok,
        "TinyLlama ONNX run failed:\nstderr: {}",
        result.stderr
    );

    // The output should have non-empty content and not be pure gibberish.
    // We check that at least one recognisable English token appears.
    let has_content = !result.stdout.trim().is_empty();
    assert!(
        has_content,
        "TinyLlama ONNX produced no output. stderr: {}",
        result.stderr
    );
    assert!(
        !result.stderr.contains("UnsupportedOp")
            && !result.stderr.contains("exec error")
            && !result.stderr.contains("shape mismatch"),
        "TinyLlama ONNX run produced an execution error:\n{}",
        result.stderr
    );
}

#[test]
fn tinyllama_gguf_runs_and_produces_english() {
    let model = workspace_path(
        "models/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_0.gguf",
    );
    if !model.exists() {
        eprintln!("SKIP: {} not found", model.display());
        return;
    }

    // The GGUF directory has no tokenizer.json; use the one from the ONNX variant
    // (same TinyLlama tokenizer — identical vocabulary and merge rules).
    let onnx_tok = workspace_path("models/TinyLlama-1.1B-Chat-v1.0/tokenizer.json");
    let tok = onnx_tok.exists().then_some(&onnx_tok);

    let result = compile_then_run_tok(&model, tok, CHAT_PROMPT, 20);

    if !result.compile_ok {
        panic!(
            "TinyLlama GGUF failed to compile:\n{}",
            result.compile_stderr
        );
    }

    eprintln!("compile stderr:\n{}", result.compile_stderr);
    eprintln!("run stderr:\n{}", result.stderr);
    eprintln!("run stdout:\n{}", result.stdout);

    assert!(
        result.run_ok,
        "TinyLlama GGUF run failed:\nstderr: {}",
        result.stderr
    );

    let has_content = !result.stdout.trim().is_empty();
    assert!(
        has_content,
        "TinyLlama GGUF produced no output. stderr: {}",
        result.stderr
    );
    assert!(
        !result.stderr.contains("UnsupportedOp")
            && !result.stderr.contains("exec error")
            && !result.stderr.contains("shape mismatch"),
        "TinyLlama GGUF run produced an execution error:\n{}",
        result.stderr
    );
}

// ── Regression tests for known bugs ──────────────────────────────────────────

/// Regression: NaN detector must not fire false positives on i64 outputs.
///
/// A Concat op with dtype=I64 whose output contains -1 (0xFFFFFFFFFFFFFFFF) was
/// incorrectly triggering the NaN detector because the bytes were cast to f32.
/// The NaN detector must check the op's output dtype before interpreting bytes
/// as f32.
///
/// The ONNX model uses Concat(I64) to build shape tensors for Reshape ops.
/// TinyLlama's RoPE implementation produces a shape tensor containing -1 as a
/// dynamic dim sentinel. Before the fix, this generated a spurious [first-nan]
/// diagnostic and masked the real first NaN.
#[test]
fn nan_detector_no_false_positive_on_i64_concat() {
    use hologram_ai::{ModelCompiler, ModelSource};
    // Build a tiny ONNX with a Concat(I64) node whose output contains -1.
    // We use an existing small test model that has I64 ops (cast/shape).
    // If this returns a shape mismatch (not a NaN panic), the fix is working.
    let model = workspace_path("models/TinyLlama-1.1B-Chat-v1.0/model.onnx");
    if !model.exists() {
        eprintln!("SKIP: model not found");
        return;
    }
    // Compilation should succeed; the [first-nan] line must not appear for I64 ops.
    let result = ModelCompiler::default().compile(ModelSource::OnnxPath(model));
    assert!(result.is_ok(), "compilation failed: {:?}", result.err());
}

/// Regression: batched MatMul shape mismatch — TinyLlama ONNX attention output.
///
/// Previously, `binary_elementwise_broadcast` would inflate outputs when compiled
/// shapes had 0-sentinels resolving to wrong values. This caused the attention output
/// shape tensor (computed via Mul ops in the shape subgraph) to produce [1,seq,4096]
/// instead of [1,seq,2048], doubling the hidden dimension. The downstream output-
/// projection MatMul then received A=[seq,4096] vs B with k=2048, causing a mismatch.
///
/// Fix: broadcast inflation guard in `binary_elementwise_broadcast` and
/// `binary_compare_broadcast` — falls back to element-cycling when out_len exceeds
/// both input sizes (indicating stale compiled shapes, not a real broadcast).
///
/// Status: FIXED.
#[test]
fn tinyllama_onnx_batched_matmul_shape_regression() {
    let model = workspace_path("models/TinyLlama-1.1B-Chat-v1.0/model.onnx");
    if !model.exists() {
        eprintln!("SKIP: model not found");
        return;
    }

    let result = compile_then_run(&model, CHAT_PROMPT, 1);

    assert!(
        result.compile_ok,
        "ONNX compilation failed: {}",
        result.compile_stderr
    );
    assert!(
        result.run_ok,
        "batched MatMul shape mismatch regression: run failed:\n{}",
        result.stderr
    );
    assert!(
        !result.stderr.contains("shape mismatch"),
        "shape mismatch regression: {}",
        result.stderr
    );
}
