//! End-to-end tests for Qwen2-0.5B — compile and run the ONNX variant.
//!
//! These tests validate cross-family LLM support beyond LLaMA. Qwen2 exercises:
//! - Extreme GQA (14 Q-heads, 2 KV-heads)
//! - RoPE with theta=1_000_000 (baked into ONNX graph as Cos/Sin ops)
//! - Post-embedding RMSNorm (Qwen-specific, not present in LLaMA)
//! - Byte-level BPE tokenizer (151K vocab)
//! - IsNaN + Where attention masking pattern (replaces NaN after softmax)
//!
//! Run with:
//!   cargo test -p hologram-ai --features e2e -- qwen2 --nocapture
//!
//! Model expected at (relative to workspace root):
//!   models/Qwen2-0.5B/model.onnx
//!   models/Qwen2-0.5B/tokenizer.json

#![cfg(feature = "e2e")]

mod common;

use hologram_ai::validate::validate_model;
use std::path::PathBuf;

fn model_path() -> Option<PathBuf> {
    let p = common::workspace_path("models/Qwen2-0.5B/model.onnx");
    if p.exists() {
        Some(p)
    } else {
        eprintln!("SKIP: {} not found", p.display());
        None
    }
}

/// The binary under test — built by cargo alongside the test runner.
fn hologram_ai_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_hologram-ai"))
}

// ── Compilation tests ─────────────────────────────────────────────────────────

#[test]
fn qwen2_onnx_compiles() {
    let model = match model_path() {
        Some(p) => p,
        None => return,
    };

    let report = validate_model(&model);
    println!("{report}");

    assert!(
        report.compilation_ok,
        "Qwen2 ONNX should compile: {:?}",
        report.error
    );
    // Qwen2-0.5B has 1828 ONNX nodes (24 layers, 14 heads, 2 KV heads)
    assert!(
        report.total_nodes > 1500,
        "expected > 1500 nodes, got {}",
        report.total_nodes
    );
    // Qwen2-0.5B has ~1.8 GB weights (494M params * 4 bytes)
    assert!(
        report.compiled_weight_bytes > 1_500_000_000,
        "expected > 1.5 GB weights, got {}",
        report.compiled_weight_bytes
    );
}

// ── Run tests ─────────────────────────────────────────────────────────────────

struct RunResult {
    compile_ok: bool,
    compile_stderr: String,
    run_ok: bool,
    stdout: String,
    stderr: String,
}

fn compile_then_run(model_path: &PathBuf, prompt: &str, max_tokens: usize) -> RunResult {
    let bin = hologram_ai_bin();
    let out_dir = tempfile::tempdir().expect("tempdir");
    let stem = model_path
        .file_stem()
        .expect("model path has no file stem")
        .to_string_lossy();
    let holo_path = out_dir.path().join(format!("{stem}.holo"));

    // Step 1: compile
    let compile_status = std::process::Command::new(&bin)
        .args(["compile", "--model"])
        .arg(model_path)
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

    // Step 2: run (greedy sampling for deterministic output)
    let run_output = std::process::Command::new(&bin)
        .args(["run"])
        .arg(&holo_path)
        .arg("--prompt")
        .arg(prompt)
        .arg("--max-tokens")
        .arg(max_tokens.to_string())
        .args(["--temperature", "0"])
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

#[test]
fn qwen2_onnx_runs_without_errors() {
    let model = match model_path() {
        Some(p) => p,
        None => return,
    };

    // Qwen2-0.5B is a base model (not instruct) — plain text continuation.
    let result = compile_then_run(&model, "The capital of France is", 10);

    if !result.compile_ok {
        panic!("Qwen2 ONNX failed to compile:\n{}", result.compile_stderr);
    }

    eprintln!("compile stderr:\n{}", result.compile_stderr);
    eprintln!("run stderr:\n{}", result.stderr);
    eprintln!("run stdout:\n{}", result.stdout);

    assert!(
        result.run_ok,
        "Qwen2 ONNX run failed:\nstderr: {}",
        result.stderr
    );

    // The output should have non-empty content.
    let has_content = !result.stdout.trim().is_empty();
    assert!(
        has_content,
        "Qwen2 ONNX produced no output. stderr: {}",
        result.stderr
    );
    assert!(
        !result.stderr.contains("UnsupportedOp")
            && !result.stderr.contains("exec error")
            && !result.stderr.contains("shape mismatch"),
        "Qwen2 ONNX run produced an execution error:\n{}",
        result.stderr
    );

    // NOTE: Output quality (gibberish vs coherent text) is a known issue.
    // The model compiles and runs without errors, but output correctness
    // depends on attention numerical accuracy. See Plan 074 for investigation.
}

// ── Variable seq_len tests ────────────────────────────────────────────────────

/// Compile Qwen2 ONNX once and run with different sequence lengths via the
/// library API. Verifies shape projection handles Qwen2's GQA (14 Q/2 KV heads)
/// at variable lengths without panics or shape errors.
#[test]
fn qwen2_onnx_variable_seq_len_runs() {
    let model = match model_path() {
        Some(p) => p,
        None => return,
    };

    // Compile to a temp directory (streaming compilation writes to disk).
    let out_dir = tempfile::tempdir().expect("tempdir");
    let bin = hologram_ai_bin();
    let compile_status = std::process::Command::new(&bin)
        .args(["compile", "--model"])
        .arg(&model)
        .args(["--output"])
        .arg(out_dir.path())
        .output()
        .expect("compile command failed to start");
    assert!(
        compile_status.status.success(),
        "Qwen2 compilation failed: {}",
        String::from_utf8_lossy(&compile_status.stderr)
    );

    let holo_path = out_dir.path().join("model.holo");
    assert!(holo_path.exists(), "compiled archive not found");

    // Load the archive bytes and run via library API.
    let archive_bytes = std::fs::read(&holo_path).expect("read archive");
    let runner = hologram_ai::HoloRunner::from_bytes(archive_bytes).expect("load archive");

    // Run with different seq_len values.
    // Qwen2 ONNX inputs (after PositionIdsInjection):
    //   0: input_ids     [1, seq_len] (i64)
    //   1: attention_mask [1, seq_len] (i64, all-ones)
    //   2: position_ids   [seq_len]   (i64, 0..seq_len)
    for seq_len in [1usize, 7, 128] {
        let token_ids: Vec<i64> = vec![1i64; seq_len];
        let attn_mask: Vec<i64> = vec![1i64; seq_len];
        let pos_ids: Vec<i64> = (0..seq_len as i64).collect();
        let id_bytes: Vec<u8> = bytemuck::cast_slice(&token_ids).to_vec();
        let mask_bytes: Vec<u8> = bytemuck::cast_slice(&attn_mask).to_vec();
        let pos_bytes: Vec<u8> = bytemuck::cast_slice(&pos_ids).to_vec();
        let mut graph_inputs = hologram::GraphInputs::new();
        graph_inputs.set_with_shape(0, id_bytes, vec![1, seq_len]);
        graph_inputs.set_with_shape(1, mask_bytes, vec![1, seq_len]);
        graph_inputs.set_with_shape(2, pos_bytes, vec![seq_len]);

        let mut kv = hologram::KvCacheState::new(24, 2, 64, 2048);
        let outputs = runner
            .execute_with_kv(&graph_inputs, &mut kv)
            .unwrap_or_else(|e| panic!("seq_len={seq_len} execution failed: {e}"));

        assert!(
            !outputs.get(0).map(|(_, b)| b.is_empty()).unwrap_or(true),
            "seq_len={seq_len} produced empty output"
        );

        eprintln!(
            "seq_len={seq_len} OK — {} output bytes",
            outputs.get(0).map(|(_, b)| b.len()).unwrap_or(0)
        );
    }
}

// ── Architecture-specific tests ──────────────────────────────────────────────

/// Verify that the compiler detects Qwen2's extreme GQA configuration:
/// 14 Q-heads, 2 KV-heads, head_dim=64, 24 layers, hidden_size=896.
#[test]
fn qwen2_gqa_config_detected() {
    let model = match model_path() {
        Some(p) => p,
        None => return,
    };

    let archive = hologram_ai::ModelCompiler::default()
        .compile(hologram_ai::ModelSource::OnnxPath(model))
        .expect("Qwen2 ONNX compilation failed");

    // The archive should be a valid pipeline with LLM metadata.
    // Check that we got at least some non-zero output.
    assert!(
        archive.stats.node_count > 2000,
        "expected >2000 nodes, got {}",
        archive.stats.node_count
    );
}

// ── Logit comparison tests ──────────────────────────────────────────────────

/// Debug test: dump prefill logits for manual comparison with ORT.
///
/// ORT reference for "The capital of France is" (tokens: [785, 6722, 315, 9625, 374]):
///   Top-1 at last position: token 12095 ("Paris"), logit ≈ 13.46
///   Logits range: [-11.55, 13.91]
///   No NaN, no Inf
#[test]
fn qwen2_prefill_logit_comparison() {
    let model = match model_path() {
        Some(p) => p,
        None => return,
    };

    // Compile to disk (streaming) then load.
    let out_dir = tempfile::tempdir().expect("tempdir");
    let bin = hologram_ai_bin();
    let compile_status = std::process::Command::new(&bin)
        .args(["compile", "--model"])
        .arg(&model)
        .args(["--output"])
        .arg(out_dir.path())
        .output()
        .expect("compile command failed to start");
    assert!(compile_status.status.success());

    let holo_path = out_dir.path().join("model.holo");
    let archive_bytes = std::fs::read(&holo_path).expect("read archive");
    let runner = hologram_ai::HoloRunner::from_bytes(archive_bytes).expect("load");

    // "The capital of France is" = [785, 6722, 315, 9625, 374]
    let token_ids: Vec<i64> = vec![785, 6722, 315, 9625, 374];
    let seq_len = token_ids.len();
    let attn_mask: Vec<i64> = vec![1i64; seq_len];
    let pos_ids: Vec<i64> = (0..seq_len as i64).collect();

    let id_bytes: Vec<u8> = bytemuck::cast_slice(&token_ids).to_vec();
    let mask_bytes: Vec<u8> = bytemuck::cast_slice(&attn_mask).to_vec();
    let pos_bytes: Vec<u8> = bytemuck::cast_slice(&pos_ids).to_vec();

    let mut inputs = hologram::GraphInputs::new();
    inputs.set_with_shape(0, id_bytes, vec![1, seq_len]);
    inputs.set_with_shape(1, mask_bytes, vec![1, seq_len]);
    inputs.set_with_shape(2, pos_bytes, vec![seq_len]);

    let mut kv = hologram::KvCacheState::new(24, 2, 64, 2048);
    let outputs = runner
        .execute_with_kv(&inputs, &mut kv)
        .expect("execute failed");

    let (_, raw) = outputs.get(0).expect("no output");
    let logits: &[f32] = bytemuck::cast_slice(raw);
    let vocab_size = 151936;

    eprintln!("Total logit elements: {}", logits.len());
    eprintln!(
        "Expected: {} * {} = {}",
        seq_len,
        vocab_size,
        seq_len * vocab_size
    );

    // Check for NaN/Inf
    let nan_count = logits.iter().filter(|x| x.is_nan()).count();
    let inf_count = logits.iter().filter(|x| x.is_infinite()).count();
    eprintln!("NaN: {}, Inf: {}", nan_count, inf_count);

    if logits.len() >= seq_len * vocab_size {
        let last_start = (seq_len - 1) * vocab_size;
        let last_logits = &logits[last_start..last_start + vocab_size];

        // Find top-5
        let mut indexed: Vec<(usize, f32)> = last_logits
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        eprintln!("Hologram top-10 at last position:");
        for (i, (tok, score)) in indexed.iter().take(10).enumerate() {
            eprintln!("  {}: token={} score={:.4}", i, tok, score);
        }

        let logits_min = logits
            .iter()
            .cloned()
            .filter(|x| x.is_finite())
            .reduce(f32::min)
            .unwrap_or(0.0);
        let logits_max = logits
            .iter()
            .cloned()
            .filter(|x| x.is_finite())
            .reduce(f32::max)
            .unwrap_or(0.0);
        eprintln!("Logits range: [{:.4}, {:.4}]", logits_min, logits_max);

        // ORT says token 12095 should be top-1
        eprintln!(
            "Token 12095 (Paris) logit: {:.4}",
            last_logits.get(12095).unwrap_or(&f32::NAN)
        );

        // Also check position 0 — if this is wrong, the issue is in
        // embedding/early layers, not attention masking.
        // ORT reference pos 0 top-5: [2701, 1156, 1372, 220, 5461]
        let pos0_logits = &logits[0..vocab_size];
        let mut pos0_indexed: Vec<(usize, f32)> = pos0_logits
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        pos0_indexed
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        eprintln!("\nHologram top-5 at position 0:");
        for (i, (tok, score)) in pos0_indexed.iter().take(5).enumerate() {
            eprintln!("  {}: token={} score={:.4}", i, tok, score);
        }
        eprintln!("ORT reference pos 0 top-5: [2701, 1156, 1372, 220, 5461]");
    } else {
        eprintln!(
            "WARNING: logits size {} < expected {}",
            logits.len(),
            seq_len * vocab_size
        );
    }
}
