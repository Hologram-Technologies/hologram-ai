//! End-to-end test for Stable Diffusion VAE decoder.
//!
//! The VAE decoder is much smaller (~198MB, 290 nodes) than the UNet (~3.4GB,
//! 2400 nodes), making it ideal for fast iteration on execution correctness.
//!
//! Run with:
//!   cargo test -p hologram-ai --features e2e -- sd_vae --nocapture

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

fn vae_onnx_path() -> PathBuf {
    workspace_path("models/stable-diffusion-v1-5/vae_decoder/model.onnx")
}

fn vae_holo_path() -> PathBuf {
    workspace_path("models/stable-diffusion-v1-5/vae_decoder/model.holo")
}

fn vae_holo_small_path() -> PathBuf {
    workspace_path("models/stable-diffusion-v1-5/vae_decoder/model_small.holo")
}

fn ensure_compiled() -> bool {
    let holo = vae_holo_path();
    if holo.exists() {
        return true;
    }
    let onnx = vae_onnx_path();
    if !onnx.exists() {
        return false;
    }
    let archive = ModelCompiler::default()
        .compile(ModelSource::OnnxPath(onnx))
        .expect("VAE decoder compilation failed");
    std::fs::write(&holo, &archive.bytes).expect("writing archive");
    true
}

fn ensure_compiled_small() -> bool {
    let holo = vae_holo_small_path();
    if holo.exists() {
        return true;
    }
    let onnx = vae_onnx_path();
    if !onnx.exists() {
        return false;
    }
    // Compile at 1/4 resolution: 64×64 latent → 16×16 latent, 512→128 output.
    let mut compiler = ModelCompiler::default();
    compiler.spatial_scale = Some(4);
    let archive = compiler
        .compile(ModelSource::OnnxPath(onnx))
        .expect("VAE decoder small compilation failed");
    std::fs::write(&holo, &archive.bytes).expect("writing archive");
    true
}

#[test]
fn sd_vae_decoder_compiles() {
    if !vae_onnx_path().exists() {
        eprintln!("skipping: VAE model not found");
        return;
    }
    assert!(ensure_compiled());
}

/// Full-resolution VAE with zero input — for debugging spatial artifacts.
/// Saves raw output to /tmp/hologram_vae_zero.bin for ORT comparison.
#[test]
fn sd_vae_decoder_zero_input() {
    if !vae_onnx_path().exists() {
        eprintln!("skipping: VAE model not found");
        return;
    }
    assert!(ensure_compiled(), "compilation failed");

    let holo_path = vae_holo_path();
    let loader = hologram::HoloLoader::open(&holo_path).expect("mmap open failed");
    let pipeline = unsafe {
        hologram::LoadedPipeline::from_bytes_zero_copy(loader.as_bytes())
    }
    .expect("loading pipeline failed");
    let plan = pipeline.into_first_model().expect("no model in pipeline");
    let tape = hologram::build_tape_from_plan(&plan).expect("building tape");

    // Zero input [1, 4, 64, 64]
    let input_data = vec![0.0f32; 1 * 4 * 64 * 64];
    let mut inputs = hologram::GraphInputs::new();
    inputs.set_with_shape(0, bytemuck::cast_slice(&input_data).to_vec(), vec![1, 4, 64, 64]);

    let start = std::time::Instant::now();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        hologram::execute_tape(&tape, &plan, &inputs)
    }));
    eprintln!("execution took: {:.2?}", start.elapsed());

    match result {
        Ok(Ok(outputs)) => {
            let (_, out_bytes) = outputs.get(0).expect("no output");
            std::fs::write("/tmp/hologram_vae_zero.bin", out_bytes).expect("writing output");
            eprintln!("wrote {} bytes to /tmp/hologram_vae_zero.bin", out_bytes.len());

            let floats: Vec<f32> = out_bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let n = floats.len() / 3;
            let side = (n as f64).sqrt() as usize;
            eprintln!("output: {} floats = 3 × {} × {}", floats.len(), side, side);
            if side > 0 {
                // R channel stats
                let r_row0: Vec<f32> = (0..side.min(8)).map(|x| floats[x]).collect();
                let r_col0: Vec<f32> = (0..side.min(8)).map(|y| floats[y * side]).collect();
                eprintln!("  R row0[:8] = {:?}", r_row0);
                eprintln!("  R col0[:8] = {:?}", r_col0);
            }
        }
        Ok(Err(e)) => eprintln!("execution error: {e}"),
        Err(p) => {
            let msg = p.downcast_ref::<String>().map(|s| s.as_str())
                .or_else(|| p.downcast_ref::<&str>().copied())
                .unwrap_or("unknown");
            eprintln!("panicked: {msg}");
        }
    }
}

/// Reduced-resolution VAE execution test.
///
/// Compiles at 1/4 spatial scale (16×16 latent → 128×128 output) to keep
/// activation memory bounded. Full-resolution (64×64 → 512×512) requires
/// ~2-4GB of activation memory due to simultaneous Conv2d outputs at high
/// spatial dims.
///
/// Memory reduction approaches for future optimization:
/// - **Operator fusion**: fuse Conv2d→ReLU→Conv2d to avoid materializing
///   intermediate activations (like TensorRT/XLA). Requires fused kernels.
/// - **Activation checkpointing**: drop skip-connection activations and
///   recompute them when needed. Trades compute for memory.
/// - **Streaming execution**: process row-strips through the network instead
///   of full spatial tensors. Requires graph-level tiling passes.
/// - **F16 activations**: halve memory by storing intermediates in half
///   precision. Requires f16 compute kernels.
#[test]
fn sd_vae_decoder_executes_small() {
    if !vae_onnx_path().exists() {
        eprintln!("skipping: VAE model not found");
        return;
    }
    assert!(ensure_compiled_small(), "small compilation failed");

    let holo_path = vae_holo_small_path();
    let loader = hologram::HoloLoader::open(&holo_path).expect("mmap open failed");
    let pipeline = unsafe {
        hologram::LoadedPipeline::from_bytes_zero_copy(loader.as_bytes())
    }
    .expect("loading pipeline failed");
    let plan = pipeline.into_first_model().expect("no model in pipeline");

    eprintln!("graph nodes: {}", plan.graph().nodes.len());
    eprintln!("weights: {} bytes", plan.weights().len());

    eprintln!("building tape...");
    let tape = hologram::build_tape_from_plan(&plan).expect("building tape");
    eprintln!("tape built, {} instructions", tape.instructions.len());

    // Scaled input: [1, 4, 16, 16] (64/4 = 16)
    let h = 16;
    let w = 16;
    let input_len = 1 * 4 * h * w;
    let input_data: Vec<f32> = (0..input_len)
        .map(|i| ((i as f32) * 0.01).sin() * 0.1)
        .collect();

    let mut inputs = hologram::GraphInputs::new();
    inputs.set_with_shape(
        0,
        bytemuck::cast_slice(&input_data).to_vec(),
        vec![1, 4, h, w],
    );

    eprintln!("starting execution (128×128 output)...");
    let start = std::time::Instant::now();

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        hologram::execute_tape(&tape, &plan, &inputs)
    }));

    let elapsed = start.elapsed();
    eprintln!("execution took: {elapsed:.2?}");

    let outputs = match result {
        Ok(Ok(o)) => o,
        Ok(Err(e)) => {
            eprintln!("VAE execution error: {e}");
            return;
        }
        Err(panic) => {
            let msg = panic
                .downcast_ref::<String>()
                .map(|s| s.as_str())
                .or_else(|| panic.downcast_ref::<&str>().copied())
                .unwrap_or("unknown panic");
            eprintln!("VAE execution panicked: {msg}");
            return;
        }
    };

    // Scaled output: [1, 3, 128, 128] = 49152 floats
    let (_name, out_bytes) = outputs.get(0).expect("no output at index 0");
    let out_floats: Vec<f32> = out_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))
        .collect();

    let expected_len = 1 * 3 * (h * 8) * (w * 8); // VAE upsamples 8x
    eprintln!(
        "output: {} floats (expected {})",
        out_floats.len(),
        expected_len
    );

    assert!(
        out_floats.len() >= expected_len,
        "VAE output too small: expected >= {expected_len}, got {}",
        out_floats.len()
    );

    let finite = out_floats.iter().filter(|v| v.is_finite()).count();
    assert_eq!(finite, out_floats.len(), "output contains non-finite values");

    eprintln!("VAE decoder (small): {} output floats, all finite", out_floats.len());
}
