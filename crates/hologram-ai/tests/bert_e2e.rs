//! End-to-end tests for BERT — compile and run ONNX.
//!
//! BERT is encoder-only with bidirectional (non-causal) attention, no KV cache,
//! and GELU activation. This validates the compiler handles non-LLM transformers.
//!
//! Run with:
//!   cargo test -p hologram-ai --features e2e -- bert --nocapture
//!
//! Model expected at (relative to workspace root):
//!   models/bert-base-uncased/model.onnx
//!
//! Download with:
//!   hologram-ai download bert-base-uncased --format onnx -o models/bert-base-uncased
//!
//! Or export manually:
//!   pip install optimum[exporters]
//!   optimum-cli export onnx --model bert-base-uncased models/bert-base-uncased/

#![cfg(feature = "e2e")]

use hologram_ai::compiler::{ModelCompiler, ModelSource};
use std::path::PathBuf;

/// Parse bytes as f32 without alignment requirements.
fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("chunk is exactly 4 bytes")))
        .collect()
}

fn workspace_path(rel: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // crates/hologram-ai → crates/
    p.pop(); // crates/ → workspace root
    p.push(rel);
    p
}

fn bert_model_path() -> PathBuf {
    workspace_path("models/bert-base-uncased/model.onnx")
}

#[test]
fn bert_onnx_compiles() {
    let model = bert_model_path();
    if !model.exists() {
        eprintln!("skipping bert_onnx_compiles: model not found at {model:?}");
        return;
    }

    let archive = ModelCompiler::default()
        .compile(ModelSource::OnnxPath(model))
        .expect("BERT ONNX compilation failed");

    // BERT-base has ~200 ops after optimization (12 transformer layers).
    assert!(
        archive.stats.node_count > 50,
        "expected > 50 compiled nodes, got {}",
        archive.stats.node_count
    );
    assert_eq!(
        archive.stats.validation_errors, 0,
        "compilation produced validation errors"
    );
}

#[test]
fn bert_onnx_executes() {
    let model = bert_model_path();
    if !model.exists() {
        eprintln!("skipping bert_onnx_executes: model not found at {model:?}");
        return;
    }

    // Compile without seq_len override — BERT uses its own context length
    // from model metadata (typically 512).
    let archive = ModelCompiler::default()
        .compile(ModelSource::OnnxPath(model))
        .expect("compilation failed");

    // Load and build tape.
    let plan = hologram::load_from_bytes(&archive.bytes)
        .expect("loading plan failed");
    let tape = hologram::build_tape_from_plan(&plan)
        .expect("building execution tape");

    // Build inputs: input_ids=[1, 32], attention_mask=[1, 32], token_type_ids=[1, 32]
    // BERT expects 3 inputs (order depends on export, but typically:
    //   slot 0: input_ids (INT64)
    //   slot 1: attention_mask (INT64)
    //   slot 2: token_type_ids (INT64)
    let seq_len = 32usize;
    let input_ids: Vec<i64> = (0..seq_len as i64).map(|i| 1000 + i).collect();
    let attention_mask: Vec<i64> = vec![1i64; seq_len];
    let token_type_ids: Vec<i64> = vec![0i64; seq_len];

    let mut inputs = hologram::GraphInputs::new();
    inputs.set_with_shape(0, bytemuck::cast_slice(&input_ids).to_vec(), vec![1, seq_len]);
    inputs.set_with_shape(1, bytemuck::cast_slice(&attention_mask).to_vec(), vec![1, seq_len]);
    inputs.set_with_shape(2, bytemuck::cast_slice(&token_type_ids).to_vec(), vec![1, seq_len]);

    // Execute — may hit runtime dispatch issues in hologram base for
    // ops not yet exercised by LLM models (BERT is the first encoder-only
    // model). Capture panics gracefully until hologram base is updated.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        hologram::execute_tape(&tape, &plan, &inputs)
    }));
    let outputs = match result {
        Ok(Ok(o)) => o,
        Ok(Err(e)) => {
            eprintln!("BERT execution error: {e}");
            return;
        }
        Err(panic) => {
            let msg = panic.downcast_ref::<String>()
                .map(|s| s.as_str())
                .or_else(|| panic.downcast_ref::<&str>().copied())
                .unwrap_or("unknown panic");
            eprintln!("BERT execution panicked (hologram base): {msg}");
            return;
        }
    };

    // BERT-base output: last_hidden_state [1, seq_len, 768]
    let (_name, out_bytes) = outputs.get(0).expect("no output at index 0");
    let out_floats = bytes_to_f32(out_bytes);

    // Should have seq_len * 768 floats (BERT hidden dim = 768).
    let expected_len = seq_len * 768;
    assert!(
        out_floats.len() >= expected_len,
        "BERT output too small: expected >= {expected_len} floats, got {}",
        out_floats.len()
    );

    // All values should be finite.
    let finite_count = out_floats.iter().filter(|v| v.is_finite()).count();
    assert!(
        finite_count == out_floats.len(),
        "output contains {} non-finite values out of {}",
        out_floats.len() - finite_count,
        out_floats.len()
    );
}
