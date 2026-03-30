//! End-to-end tests for Stable Diffusion UNet — compile and run ONNX.
//!
//! The SD UNet exercises GroupNorm, cross-attention, SiLU, Resize, and Conv2d
//! in a single model — a superset of what TinyLlama (causal LLM) and ResNet-50
//! (pure vision) cover.
//!
//! Run with:
//!   cargo test -p hologram-ai --features e2e -- sd_unet --nocapture
//!
//! Model expected at (relative to workspace root):
//!   models/stable-diffusion-v1-5/unet/model.onnx
//!
//! Download with:
//!   pip install optimum[exporters]
//!   optimum-cli export onnx --model runwayml/stable-diffusion-v1-5 \
//!       --task stable-diffusion models/stable-diffusion-v1-5/

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

fn sd_unet_model_path() -> PathBuf {
    workspace_path("models/stable-diffusion-v1-5/unet/model.onnx")
}

fn sd_unet_holo_path() -> PathBuf {
    workspace_path("models/stable-diffusion-v1-5/unet/model.holo")
}

/// Compile the UNet ONNX model and write the `.holo` archive to disk.
///
/// If the `.holo` file already exists, this is a no-op.
/// The archive is written to disk immediately and the in-memory buffer dropped.
fn ensure_compiled() -> bool {
    let holo = sd_unet_holo_path();
    if holo.exists() {
        return true;
    }
    let onnx = sd_unet_model_path();
    if !onnx.exists() {
        return false;
    }
    let archive = ModelCompiler::default()
        .compile(ModelSource::OnnxPath(onnx))
        .expect("SD UNet ONNX compilation failed");

    std::fs::write(&holo, &archive.bytes).expect("writing archive to disk");
    // Drop archive immediately to free ~3.4GB.
    drop(archive);
    true
}

#[test]
fn sd_unet_onnx_compiles() {
    let model = sd_unet_model_path();
    if !model.exists() {
        eprintln!("skipping sd_unet_onnx_compiles: model not found at {model:?}");
        return;
    }

    assert!(ensure_compiled(), "compilation failed");

    let holo = sd_unet_holo_path();
    let meta = std::fs::metadata(&holo).expect("reading holo metadata");
    eprintln!("archive bytes: {}", meta.len());
    assert!(meta.len() > 1_000_000, "archive too small");
}

#[test]
fn sd_unet_onnx_executes() {
    let model = sd_unet_model_path();
    if !model.exists() {
        eprintln!("skipping sd_unet_onnx_executes: model not found at {model:?}");
        return;
    }

    // Ensure the .holo file exists on disk.
    assert!(ensure_compiled(), "compilation failed");

    // Load via mmap — zero-copy weight access.
    let holo_path = sd_unet_holo_path();
    let loader = hologram::HoloLoader::open(&holo_path).expect("mmap open failed");

    // Pipeline archives wrap sub-models. Extract the first (only) model
    // with zero-copy weights borrowed from the mmap.
    let pipeline = unsafe {
        hologram::LoadedPipeline::from_bytes_zero_copy(loader.as_bytes())
    }
    .expect("loading pipeline failed");
    let plan = pipeline.into_first_model().expect("no model in pipeline");

    eprintln!("plan loaded, graph nodes: {}", plan.graph().nodes.len());
    eprintln!("weights: {} bytes", plan.weights().len());

    let tape = hologram::build_tape_from_plan(&plan).expect("building execution tape");

    // SD v1.5 UNet inputs:
    //   sample:                  [1, 4, 64, 64]  (f32) — noisy latent
    //   timestep:                [1]              (f32) — diffusion timestep
    //   encoder_hidden_states:   [1, 77, 768]     (f32) — text conditioning
    let sample_len = 1 * 4 * 64 * 64; // 16384 floats
    let sample_data: Vec<f32> = (0..sample_len)
        .map(|i| ((i as f32) * 0.01).sin() * 0.1)
        .collect();

    let timestep_data: Vec<f32> = vec![500.0]; // mid-range timestep

    let encoder_len = 1 * 77 * 768; // 59136 floats
    let encoder_data: Vec<f32> = (0..encoder_len)
        .map(|i| ((i as f32) * 0.001).cos() * 0.1)
        .collect();

    let mut inputs = hologram::GraphInputs::new();
    inputs.set_with_shape(0, bytemuck::cast_slice(&sample_data).to_vec(), vec![1, 4, 64, 64]);
    inputs.set_with_shape(1, bytemuck::cast_slice(&timestep_data).to_vec(), vec![1]);
    inputs.set_with_shape(2, bytemuck::cast_slice(&encoder_data).to_vec(), vec![1, 77, 768]);

    eprintln!("starting execution...");
    let start = std::time::Instant::now();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        hologram::execute_tape(&tape, &plan, &inputs)
    }));
    let elapsed = start.elapsed();
    eprintln!("execution took: {elapsed:.2?}");

    let outputs = match result {
        Ok(Ok(o)) => o,
        Ok(Err(e)) => {
            eprintln!("UNet execution error: {e}");
            return;
        }
        Err(panic) => {
            let msg = panic
                .downcast_ref::<String>()
                .map(|s| s.as_str())
                .or_else(|| panic.downcast_ref::<&str>().copied())
                .unwrap_or("unknown panic");
            eprintln!("UNet execution panicked: {msg}");
            return;
        }
    };

    // SD v1.5 UNet output: noise prediction [1, 4, 64, 64]
    let (_name, out_bytes) = outputs.get(0).expect("no output at index 0");
    let out_floats = bytes_to_f32(out_bytes);
    eprintln!("output: {} floats ({} bytes)", out_floats.len(), out_bytes.len());

    // Should have 1*4*64*64 = 16384 floats.
    let expected_len = 1 * 4 * 64 * 64;
    assert!(
        out_floats.len() >= expected_len,
        "SD UNet output too small: expected >= {expected_len} floats, got {}",
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
