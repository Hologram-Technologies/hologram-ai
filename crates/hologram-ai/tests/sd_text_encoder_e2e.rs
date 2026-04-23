//! End-to-end test for Stable Diffusion text encoder (CLIP).
//!
//! Run with:
//!   cargo test -p hologram-ai --features e2e -- sd_text_encoder --nocapture

#![cfg(feature = "e2e")]

mod common;

use hologram_ai::compiler::{ModelCompiler, ModelSource};
use std::path::PathBuf;

fn text_encoder_onnx_path() -> PathBuf {
    common::workspace_path("models/stable-diffusion-v1-5/text_encoder/model.onnx")
}

fn text_encoder_holo_path() -> PathBuf {
    common::workspace_path("models/stable-diffusion-v1-5/text_encoder/model.holo")
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
    let pipeline = unsafe { hologram::LoadedPipeline::from_bytes_zero_copy(loader.as_bytes()) }
        .expect("loading pipeline failed");
    let plan = pipeline.into_first_model().expect("no model in pipeline");

    eprintln!("graph nodes: {}", plan.graph().nodes.len());
    eprintln!("weights: {} bytes", plan.weights().len());

    let tape = hologram::build_tape_from_plan(&plan).expect("building tape");

    // CLIP text encoder input: input_ids [1, 77] (INT64)
    // SD v1.5 uses max 77 tokens.
    let seq_len = 77;
    let input_ids: Vec<i64> = (0..seq_len).map(|i| (i % 49408) as i64).collect();
    let input_bytes: Vec<u8> = input_ids.iter().flat_map(|v| v.to_le_bytes()).collect();

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

    // ONNX declares two outputs: `last_hidden_state` [1, seq, 768] and
    // `index` (pooler_output) [batch, 768]. Dump both by name + stats so
    // we can diagnose the 48× seq-axis bloat bug.
    eprintln!("output count: {}", outputs.len());
    for i in 0..outputs.len() {
        if let Some((name, out_bytes)) = outputs.get(i) {
            let n_floats = out_bytes.len() / 4;
            let floats: Vec<f32> = out_bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))
                .collect();
            let mn = floats.iter().cloned().fold(f32::MAX, f32::min);
            let mx = floats.iter().cloned().fold(f32::MIN, f32::max);
            let mean = floats.iter().sum::<f32>() / n_floats.max(1) as f32;
            let finite = floats.iter().filter(|v| v.is_finite()).count();
            eprintln!(
                "  [{i}] name={:?} floats={n_floats} finite={finite} min={mn:.4} max={mx:.4} mean={mean:.6}",
                name
            );
        }
    }

    // Lookup graph metadata to cross-reference declared output shapes.
    let graph = plan.graph();
    eprintln!("graph output_names: {:?}", graph.output_names);
    eprintln!("graph output_node_ids: {:?}", graph.output_node_ids);

    // Dump the layer header's declared input/output port shapes.
    if let Some(lh) = plan.layer_header() {
        for layer in &lh.layers {
            eprintln!("layer {:?}:", layer.name);
            for (i, port) in layer.inputs.iter().enumerate() {
                eprintln!(
                    "  input[{i}] name={:?} shape={:?} dtype={}",
                    port.name,
                    port.shape,
                    port.dtype.name()
                );
            }
            for (i, port) in layer.outputs.iter().enumerate() {
                eprintln!(
                    "  output[{i}] name={:?} shape={:?} dtype={}",
                    port.name,
                    port.shape,
                    port.dtype.name()
                );
            }
        }
    } else {
        eprintln!("no layer header");
    }

    // The first declared output should be last_hidden_state with 59136 floats.
    let expected = 1 * seq_len * 768;
    let last_hidden = outputs
        .by_name("last_hidden_state")
        .expect("no last_hidden_state output");
    let n = last_hidden.len() / 4;
    eprintln!(
        "last_hidden_state by_name: {} floats (expected {}, ratio {})",
        n,
        expected,
        n as f64 / expected as f64
    );
    assert_eq!(
        n, expected,
        "CLIP last_hidden_state should be [1, 77, 768] = {} floats, got {}",
        expected, n
    );
}
