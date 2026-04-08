//! End-to-end Stable Diffusion pipeline test.
//!
//! Runs the full text-to-image pipeline:
//!   tokenize → CLIP text encoder → UNet denoising (20 steps) → VAE decoder → PNG
//!
//! Run with:
//!   cargo test -p hologram-ai --features e2e -- sd_pipeline --nocapture

#![cfg(feature = "e2e")]

use hologram_ai::compiler::{ModelCompiler, ModelSource};
use hologram_ai_common::lower::QuantStrategy;
use std::path::PathBuf;

fn workspace_path(rel: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p.push(rel);
    p
}

// ── Model paths ──────────────────────────────────────────────────────────────

fn text_encoder_holo() -> PathBuf {
    workspace_path("models/stable-diffusion-v1-5/text_encoder/model.holo")
}
fn text_encoder_q8_holo() -> PathBuf {
    workspace_path("models/stable-diffusion-v1-5/text_encoder/model_q8.holo")
}
fn unet_holo() -> PathBuf {
    workspace_path("models/stable-diffusion-v1-5/unet/model.holo")
}
fn vae_holo() -> PathBuf {
    workspace_path("models/stable-diffusion-v1-5/vae_decoder/model_pipeline.holo")
}

fn text_encoder_onnx() -> PathBuf {
    workspace_path("models/stable-diffusion-v1-5/text_encoder/model.onnx")
}
fn unet_onnx() -> PathBuf {
    workspace_path("models/stable-diffusion-v1-5/unet/model.onnx")
}
fn vae_onnx() -> PathBuf {
    workspace_path("models/stable-diffusion-v1-5/vae_decoder/model.onnx")
}

fn output_path() -> PathBuf {
    workspace_path("models/stable-diffusion-v1-5/output.ppm")
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn ensure_compiled(onnx: &std::path::Path, holo: &std::path::Path) -> bool {
    ensure_compiled_with(onnx, holo, None, None)
}

fn ensure_compiled_with(
    onnx: &std::path::Path,
    holo: &std::path::Path,
    seq_len: Option<u64>,
    spatial_scale: Option<u32>,
) -> bool {
    ensure_compiled_full(onnx, holo, seq_len, spatial_scale, None)
}

fn ensure_compiled_full(
    onnx: &std::path::Path,
    holo: &std::path::Path,
    seq_len: Option<u64>,
    spatial_scale: Option<u32>,
    quant: Option<hologram_ai_common::lower::QuantStrategy>,
) -> bool {
    if holo.exists() {
        return true;
    }
    if !onnx.exists() {
        return false;
    }
    let mut compiler = ModelCompiler::default();
    compiler.seq_len_override = seq_len;
    compiler.spatial_scale = spatial_scale;
    if let Some(q) = quant {
        compiler.quant_strategy = q;
    }
    let archive = compiler
        .compile(ModelSource::OnnxPath(onnx.to_path_buf()))
        .expect("compilation failed");
    std::fs::write(holo, &archive.bytes).expect("writing archive");
    true
}

fn load_model(
    holo_path: &std::path::Path,
) -> (
    hologram::HoloLoader,
    hologram::LoadedPlan,
    hologram::hologram_exec::tape::EnumTape,
) {
    let loader = hologram::HoloLoader::open(holo_path).expect("mmap open failed");
    let pipeline = unsafe { hologram::LoadedPipeline::from_bytes_zero_copy(loader.as_bytes()) }
        .expect("loading pipeline failed");
    let plan = pipeline.into_first_model().expect("no model in pipeline");
    let tape = hologram::build_tape_from_plan(&plan).expect("building tape");
    (loader, plan, tape)
}

fn execute(
    tape: &hologram::hologram_exec::tape::EnumTape,
    plan: &hologram::LoadedPlan,
    inputs: &hologram::GraphInputs,
) -> hologram::GraphOutputs {
    hologram::execute_tape(tape, plan, inputs).expect("execution failed")
}

fn f32_to_bytes(data: &[f32]) -> Vec<u8> {
    bytemuck::cast_slice(data).to_vec()
}

fn bytes_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

// ── Euler-a scheduler ────────────────────────────────────────────────────────

/// Compute DDPM alpha_bar schedule for SD v1.5 (1000 timesteps).
///
/// SD v1.5's `scheduler_config.json` specifies `beta_schedule: "scaled_linear"`,
/// which interpolates the *square roots* of beta and squares the result. This
/// matches `diffusers.DDIMScheduler` exactly and is **not** the same as a plain
/// linear schedule — using plain linear here produces wrong alpha_bar values
/// (compounding error at high t) and a washed-out final image.
///
/// betas[i] = (sqrt(beta_start) + (sqrt(beta_end) - sqrt(beta_start)) * i / (n - 1))^2
/// alpha_bar[t] = product(1 - beta_i for i=0..=t)
fn ddpm_alpha_bars() -> Vec<f32> {
    let n = 1000;
    let beta_start = 0.00085f32;
    let beta_end = 0.012f32;
    let s_start = beta_start.sqrt();
    let s_end = beta_end.sqrt();
    let mut alpha_bar = Vec::with_capacity(n);
    let mut cumulative = 1.0f32;
    for i in 0..n {
        let s = s_start + (s_end - s_start) * i as f32 / (n - 1) as f32;
        let beta = s * s;
        cumulative *= 1.0 - beta;
        alpha_bar.push(cumulative);
    }
    alpha_bar
}

/// Select timesteps for n_steps evenly spaced from 999 to 0.
fn ddpm_timesteps(n_steps: usize) -> Vec<usize> {
    (0..n_steps)
        .map(|i| (n_steps - 1 - i) * 999 / (n_steps - 1).max(1))
        .collect()
}

/// One DDIM deterministic step.
///
/// Given x_t at timestep t and noise prediction eps:
///   x0_pred = (x_t - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t)
///   x_{t-1} = sqrt(alpha_bar_{t-1}) * x0_pred + sqrt(1 - alpha_bar_{t-1}) * eps
fn ddim_step(latent: &mut [f32], noise_pred: &[f32], alpha_bar_t: f32, alpha_bar_prev: f32) {
    let sqrt_ab = alpha_bar_t.sqrt();
    let sqrt_1m_ab = (1.0 - alpha_bar_t).sqrt();
    let sqrt_ab_prev = alpha_bar_prev.sqrt();
    let sqrt_1m_ab_prev = (1.0 - alpha_bar_prev).sqrt();

    for (l, &eps) in latent.iter_mut().zip(noise_pred.iter()) {
        // Predict x0 from current noisy sample.
        let x0 = (*l - sqrt_1m_ab * eps) / sqrt_ab.max(1e-8);
        // Compute x_{t-1} deterministically.
        *l = sqrt_ab_prev * x0 + sqrt_1m_ab_prev * eps;
    }
}

// ── Tokenizer ────────────────────────────────────────────────────────────────

/// Tokenize a prompt using the CLIP BPE tokenizer.
///
/// Loads the HuggingFace tokenizer.json, encodes the text, pads/truncates
/// to 77 tokens, and returns INT64 token IDs.
fn tokenize_clip(prompt: &str) -> Vec<i64> {
    use hologram_ai_tokenizer::{NativeTokenizer, Tokenizer};

    let tokenizer_path = workspace_path("models/stable-diffusion-v1-5/tokenizer.json");
    let tokenizer =
        NativeTokenizer::from_tokenizer_json(&tokenizer_path).expect("loading CLIP tokenizer");

    let mut ids: Vec<u32> = tokenizer.encode(prompt);

    // CLIP uses max 77 tokens. Truncate or pad with end_token (49407).
    let max_len = 77;
    let end_token = 49407u32;
    ids.truncate(max_len);
    while ids.len() < max_len {
        ids.push(end_token);
    }

    // Convert to i64 for the model.
    ids.iter().map(|&id| id as i64).collect()
}

// ── Save image ───────────────────────────────────────────────────────────────

/// Save [1, 3, H, W] f32 tensor as PPM image.
/// Values are clamped to [0, 1] and mapped to [0, 255].
fn save_ppm(data: &[f32], h: usize, w: usize, path: &std::path::Path) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).expect("creating output file");
    write!(f, "P6\n{w} {h}\n255\n").expect("writing PPM header");

    let hw = h * w;
    for y in 0..h {
        for x in 0..w {
            let r = (data[0 * hw + y * w + x].clamp(0.0, 1.0) * 255.0) as u8;
            let g = (data[1 * hw + y * w + x].clamp(0.0, 1.0) * 255.0) as u8;
            let b = (data[2 * hw + y * w + x].clamp(0.0, 1.0) * 255.0) as u8;
            f.write_all(&[r, g, b]).expect("writing pixel");
        }
    }
    eprintln!("saved {w}×{h} image to {}", path.display());
}

// ── Pipeline test ────────────────────────────────────────────────────────────

#[test]
fn sd_pipeline_generates_image() {
    // Check all models exist.
    if !text_encoder_onnx().exists() || !unet_onnx().exists() || !vae_onnx().exists() {
        eprintln!("skipping: SD v1.5 models not found");
        return;
    }

    // Compile all components.
    // Text encoder: compile at seq_len=77 (CLIP's max for SD v1.5).
    assert!(ensure_compiled_with(
        &text_encoder_onnx(),
        &text_encoder_holo(),
        Some(77),
        None,
    ));
    // Also compile a Q8 variant for ~2× faster CLIP inference.
    let _ = ensure_compiled_full(
        &text_encoder_onnx(),
        &text_encoder_q8_holo(),
        Some(77),
        None,
        Some(QuantStrategy::Q8_0),
    );
    assert!(ensure_compiled(&unet_onnx(), &unet_holo()));
    // VAE at spatial_scale=2: 256×256 output, ~3GB peak memory.
    assert!(ensure_compiled_with(
        &vae_onnx(),
        &vae_holo(),
        None,
        Some(2)
    ));
    eprintln!("all 3 components compiled");

    let total_start = std::time::Instant::now();

    // ── Step 1: Tokenize ──────────────────────────────────────────────────
    let prompt = "dog";
    let token_ids = tokenize_clip(prompt);
    let token_bytes: Vec<u8> = token_ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    eprintln!("tokenized: {} tokens", token_ids.len());

    // ── Step 2: Text Encoder ──────────────────────────────────────────────
    // Prefer Q8 variant if available (faster via fused dequant-matmul).
    let te_path = if text_encoder_q8_holo().exists() {
        eprintln!("using Q8 text encoder");
        text_encoder_q8_holo()
    } else {
        text_encoder_holo()
    };
    let (_te_loader, te_plan, te_tape) = load_model(&te_path);
    let mut te_inputs = hologram::GraphInputs::new();
    te_inputs.set_with_shape(0, token_bytes, vec![1, 77]);

    let start = std::time::Instant::now();
    let te_outputs = execute(&te_tape, &te_plan, &te_inputs);
    eprintln!("text encoder: {:.2?}", start.elapsed());

    let (_name, hidden_bytes) = te_outputs.get(0).expect("no text encoder output");
    let hidden_states = bytes_to_f32(hidden_bytes);
    let expected_len = 77 * 768;
    let clip_len = expected_len.min(hidden_states.len());
    let hidden_77_768: Vec<f32> = hidden_states[..clip_len].to_vec();
    // Debug: check hidden state statistics
    let hs_min = hidden_77_768.iter().cloned().fold(f32::MAX, f32::min);
    let hs_max = hidden_77_768.iter().cloned().fold(f32::MIN, f32::max);
    let hs_mean = hidden_77_768.iter().sum::<f32>() / hidden_77_768.len() as f32;
    eprintln!(
        "hidden states: {} floats (using {} for UNet)",
        hidden_states.len(),
        clip_len
    );
    eprintln!("  min={hs_min:.4} max={hs_max:.4} mean={hs_mean:.6}");

    // ── Step 3: UNet Denoising Loop ───────────────────────────────────────
    let (_unet_loader, unet_plan, unet_tape) = load_model(&unet_holo());

    let n_steps = 10; // 10 DDIM steps — balance between quality and speed.
    let alpha_bars = ddpm_alpha_bars();
    let timesteps = ddpm_timesteps(n_steps);

    // Initialize latent with deterministic Gaussian noise (seed=42).
    let latent_len = 1 * 4 * 64 * 64;
    let mut latent: Vec<f32> = (0..latent_len)
        .map(|i| {
            // Simple LCG PRNG → approximate Gaussian via Box-Muller.
            let s1 = (i as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let s2 = s1.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u1 = (s1 >> 11) as f32 / (1u64 << 53) as f32 + 1e-10;
            let u2 = (s2 >> 11) as f32 / (1u64 << 53) as f32;
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
        })
        .collect();

    eprintln!("starting denoising: {} steps (DDIM)", n_steps);
    let denoise_start = std::time::Instant::now();

    // Unconditional (empty prompt) hidden states for classifier-free guidance.
    let uncond_tokens = tokenize_clip("");
    let uncond_bytes: Vec<u8> = uncond_tokens.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut uncond_inputs = hologram::GraphInputs::new();
    uncond_inputs.set_with_shape(0, uncond_bytes, vec![1, 77]);
    let uncond_outputs = execute(&te_tape, &te_plan, &uncond_inputs);
    let (_, uncond_hidden_bytes) = uncond_outputs.get(0).expect("no uncond output");
    let uncond_states = bytes_to_f32(uncond_hidden_bytes);
    let uncond_77_768: Vec<f32> = uncond_states[..clip_len].to_vec();
    eprintln!("unconditional hidden states ready");

    let guidance_scale: f32 = 7.5; // Standard CFG scale for SD v1.5.

    for (step_idx, &t) in timesteps.iter().enumerate() {
        let alpha_bar_t = alpha_bars[t];
        let alpha_bar_prev = if step_idx + 1 < timesteps.len() {
            alpha_bars[timesteps[step_idx + 1]]
        } else {
            1.0
        };

        let timestep_f32 = t as f32;
        let step_start = std::time::Instant::now();

        // Conditional prediction (with text prompt).
        let mut cond_inputs = hologram::GraphInputs::new();
        cond_inputs.set_with_shape(0, f32_to_bytes(&latent), vec![1, 4, 64, 64]);
        cond_inputs.set_with_shape(1, f32_to_bytes(&[timestep_f32]), vec![1]);
        cond_inputs.set_with_shape(2, f32_to_bytes(&hidden_77_768), vec![1, 77, 768]);
        let cond_out = execute(&unet_tape, &unet_plan, &cond_inputs);
        let cond_noise = bytes_to_f32(cond_out.get(0).expect("no cond output").1);

        // Unconditional prediction (empty prompt).
        let mut uncond_unet_inputs = hologram::GraphInputs::new();
        uncond_unet_inputs.set_with_shape(0, f32_to_bytes(&latent), vec![1, 4, 64, 64]);
        uncond_unet_inputs.set_with_shape(1, f32_to_bytes(&[timestep_f32]), vec![1]);
        uncond_unet_inputs.set_with_shape(2, f32_to_bytes(&uncond_77_768), vec![1, 77, 768]);
        let uncond_out = execute(&unet_tape, &unet_plan, &uncond_unet_inputs);
        let uncond_noise = bytes_to_f32(uncond_out.get(0).expect("no uncond output").1);

        // Classifier-free guidance: noise = uncond + scale * (cond - uncond)
        let noise_pred: Vec<f32> = uncond_noise
            .iter()
            .zip(cond_noise.iter())
            .map(|(&u, &c)| u + guidance_scale * (c - u))
            .collect();

        let step_time = step_start.elapsed();

        if noise_pred.len() >= latent_len {
            ddim_step(
                &mut latent,
                &noise_pred[..latent_len],
                alpha_bar_t,
                alpha_bar_prev,
            );
        }

        if step_idx < 3 || step_idx == n_steps - 1 {
            let np_min = noise_pred.iter().cloned().fold(f32::MAX, f32::min);
            let np_max = noise_pred.iter().cloned().fold(f32::MIN, f32::max);
            let lat_min = latent.iter().cloned().fold(f32::MAX, f32::min);
            let lat_max = latent.iter().cloned().fold(f32::MIN, f32::max);
            eprintln!("  step {}/{} (t={}): {:.2?} noise=[{np_min:.3}..{np_max:.3}] latent=[{lat_min:.3}..{lat_max:.3}]",
                step_idx + 1, n_steps, t, step_time);
        }
    }
    eprintln!("denoising done: {:.2?}", denoise_start.elapsed());

    // ── Step 4: VAE Decode ────────────────────────────────────────────────
    // Scale latent by 1/0.18215 (SD v1.5 scaling factor).
    let scaling_factor = 1.0 / 0.18215;
    let scaled_latent: Vec<f32> = latent.iter().map(|v| v * scaling_factor).collect();

    let (_vae_loader, vae_plan, mut vae_tape) = load_model(&vae_holo());
    // Enable activation checkpointing: force-evict skip-connection buffers
    // after first consumer and recompute when distant consumers need them.
    // Trades ~30% extra compute for dramatically lower peak memory (51GB → ~2-3GB).
    vae_tape.checkpoint_enabled = true;
    let mut vae_inputs = hologram::GraphInputs::new();
    vae_inputs.set_with_shape(0, f32_to_bytes(&scaled_latent), vec![1, 4, 64, 64]);

    let start = std::time::Instant::now();
    let vae_outputs = execute(&vae_tape, &vae_plan, &vae_inputs);
    eprintln!("VAE decode: {:.2?}", start.elapsed());

    let (_name, image_bytes) = vae_outputs.get(0).expect("no VAE output");
    let image = bytes_to_f32(image_bytes);
    eprintln!("image: {} floats", image.len());

    // ── Step 5: Save Image ────────────────────────────────────────────────
    // Output is [1, 3, H, W]. Determine H×W from total.
    let n_pixels = image.len() / 3;
    let side = (n_pixels as f64).sqrt() as usize;
    let (h, w) = if side * side == n_pixels {
        (side, side)
    } else {
        // Non-square: try common SD ratios.
        (512, n_pixels / 512)
    };

    // Normalize from model output range to [0, 1].
    // SD VAE outputs in roughly [-1, 1], so map: pixel = (x + 1) / 2.
    let normalized: Vec<f32> = image.iter().map(|v| (v + 1.0) / 2.0).collect();

    save_ppm(&normalized, h, w, &output_path());

    let total_time = total_start.elapsed();
    eprintln!("total pipeline: {:.2?}", total_time);

    // Verify output exists and is reasonable size.
    let meta = std::fs::metadata(&output_path()).expect("output file missing");
    assert!(
        meta.len() > 1000,
        "output file too small: {} bytes",
        meta.len()
    );
    eprintln!("SD pipeline complete: {} bytes", meta.len());
}
