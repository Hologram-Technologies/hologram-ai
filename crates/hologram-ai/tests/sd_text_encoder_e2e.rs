//! End-to-end test for Stable Diffusion text encoder (CLIP).
//!
//! Run with:
//!   cargo test -p hologram-ai --features e2e -- sd_text_encoder --nocapture

#![cfg(feature = "e2e")]

use hologram_ai::compiler::{ModelCompiler, ModelSource};
use std::path::PathBuf;

fn workspace_path(rel: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p.push(rel);
    p
}

fn text_encoder_onnx_path() -> PathBuf {
    workspace_path("models/stable-diffusion-v1-5/text_encoder/model.onnx")
}

fn text_encoder_holo_path() -> PathBuf {
    workspace_path("models/stable-diffusion-v1-5/text_encoder/model.holo")
}

fn ensure_compiled() -> bool {
    let holo = text_encoder_holo_path();
    if holo.exists() {
        return true;
    }
    let onnx = text_encoder_onnx_path();
    if !onnx.exists() {
        return false;
    }
    let archive = ModelCompiler::default()
        .compile(ModelSource::OnnxPath(onnx))
        .expect("text encoder compilation failed");
    std::fs::write(&holo, &archive.bytes).expect("writing archive");
    true
}

#[test]
fn sd_text_encoder_compiles() {
    if !text_encoder_onnx_path().exists() {
        eprintln!("skipping: text encoder not found");
        return;
    }
    assert!(ensure_compiled());
}

#[test]
fn sd_text_encoder_executes() {
    if !text_encoder_onnx_path().exists() {
        eprintln!("skipping: text encoder not found");
        return;
    }
    assert!(ensure_compiled(), "compilation failed");

    let holo_path = text_encoder_holo_path();
    let loader = hologram::HoloLoader::open(&holo_path).expect("mmap open failed");
    let pipeline = unsafe {
        hologram::LoadedPipeline::from_bytes_zero_copy(loader.as_bytes())
    }
    .expect("loading pipeline failed");
    let plan = pipeline.into_first_model().expect("no model in pipeline");

    eprintln!("graph nodes: {}", plan.graph().nodes.len());
    eprintln!("weights: {} bytes", plan.weights().len());

    let tape = hologram::build_tape_from_plan(&plan).expect("building tape");

    // CLIP text encoder input: input_ids [1, 77] (INT64)
    // SD v1.5 uses max 77 tokens.
    let seq_len = 77;
    let input_ids: Vec<i64> = (0..seq_len).map(|i| (i % 49408) as i64).collect();
    let input_bytes: Vec<u8> = input_ids
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();

    let mut inputs = hologram::GraphInputs::new();
    inputs.set_with_shape(0, input_bytes, vec![1, seq_len]);

    eprintln!("starting execution...");
    let start = std::time::Instant::now();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        hologram::execute_tape(&tape, &plan, &inputs)
    }));
    eprintln!("execution took: {:.2?}", start.elapsed());

    let outputs = match result {
        Ok(Ok(o)) => o,
        Ok(Err(e)) => {
            eprintln!("text encoder execution error: {e}");
            return;
        }
        Err(panic) => {
            let msg = panic
                .downcast_ref::<String>()
                .map(|s| s.as_str())
                .or_else(|| panic.downcast_ref::<&str>().copied())
                .unwrap_or("unknown panic");
            eprintln!("text encoder panicked: {msg}");
            return;
        }
    };

    // Output: last_hidden_state [1, 77, 768]
    eprintln!("output count: {}", outputs.len());
    if let Some((_name, out_bytes)) = outputs.get(0) {
        let n_floats = out_bytes.len() / 4;
        let expected = 1 * seq_len * 768;
        eprintln!("output[0]: {} floats (expected {})", n_floats, expected);

        assert!(
            n_floats >= expected,
            "text encoder output too small: expected >= {expected}, got {n_floats}"
        );

        // Check all finite
        let floats: &[f32] = if out_bytes.is_empty() {
            &[]
        } else {
            bytemuck::cast_slice(out_bytes)
        };
        let finite = floats.iter().filter(|v| v.is_finite()).count();
        assert_eq!(finite, floats.len(), "output contains non-finite values");

        eprintln!("text encoder: {} output floats, all finite", n_floats);
    } else {
        eprintln!("no output at index 0");
    }
}
