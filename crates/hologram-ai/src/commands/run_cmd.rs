//! `hologram-ai run` — execute a `.holo` archive with `ShapeContextGraph` shape hints.
//!
//! Mirrors `hologram run` but replaces every execution call with
//! [`HoloRunner::execute`], which projects shapes through the embedded
//! `ShapeContextGraph` before dispatch. This fixes shape mismatches at seq>1
//! that occur when compiled shapes have 0-sentinels for symbolic dims.

use anyhow::Context as _;
use clap::Args;
use hologram::hologram_archive::section::model_meta::{ModelMetaSection, SECTION_MODEL_META};
use hologram::hologram_archive::section::tokenizer::{
    MiniBpeEncoder, TokenizerSection, SECTION_TOKENIZER,
};
use hologram::hologram_archive::weight::WeightDType;
use hologram::GraphInputs;
use std::io::Write;
use std::path::PathBuf;

use crate::compiler::HoloRunner;

// ── CLI args ───────────────────────────────────────────────────────────────

/// Arguments for the `hologram-ai run` subcommand.
#[derive(Args)]
pub struct RunArgs {
    /// Path to the `.holo` file to execute.
    pub file: PathBuf,
    /// Input values as `INDEX:HEX` pairs (e.g. `--input 0:deadbeef`).
    #[arg(long = "input", value_name = "INDEX:HEX")]
    pub inputs: Vec<String>,
    /// Input from file as `SLOT:PATH` pairs (e.g. `--input-file 0:input.bin`).
    #[arg(long = "input-file", value_name = "SLOT:PATH")]
    pub input_files: Vec<String>,
    /// Text prompt for autoregressive generation (requires embedded tokenizer).
    #[arg(long)]
    pub prompt: Option<String>,
    /// Maximum tokens to generate when using `--prompt` (default: 128).
    #[arg(long, default_value = "128")]
    pub max_tokens: usize,
    /// Sampling temperature (0.0 = greedy argmax, default: 0.7).
    #[arg(long, default_value = "0.7")]
    pub temperature: f32,
    /// Top-k sampling: only consider the top K tokens (default: 40, 0 = disabled).
    #[arg(long, default_value = "40")]
    pub top_k: usize,
    /// Print per-step logit diagnostics.
    #[arg(long)]
    pub verbose: bool,
}

// ── Entry point ────────────────────────────────────────────────────────────

/// Execute the run command using shape-aware inference.
pub fn execute(args: RunArgs) -> anyhow::Result<()> {
    let archive_bytes = std::fs::read(&args.file)
        .with_context(|| format!("reading archive {}", args.file.display()))?;

    let runner = HoloRunner::from_bytes(archive_bytes.clone())
        .with_context(|| format!("loading archive {}", args.file.display()))?;

    // Load optional metadata sections.
    // Try the runner's effective bytes first (sub-archive for pipeline),
    // then the raw archive bytes (pipeline wrapper where CLI embeds sections).
    let effective = runner.archive_bytes();
    let tokenizer = load_section::<TokenizerSection>(effective, runner.plan(), SECTION_TOKENIZER)
        .or_else(|| load_section_from_raw::<TokenizerSection>(&archive_bytes, SECTION_TOKENIZER));
    let model_meta = load_section::<ModelMetaSection>(effective, runner.plan(), SECTION_MODEL_META)
        .or_else(|| load_section_from_raw::<ModelMetaSection>(&archive_bytes, SECTION_MODEL_META));

    print_model_info(runner.plan(), &model_meta);

    if let Some(prompt) = &args.prompt {
        if let Some(meta) = &model_meta {
            if !meta.supports_prompt {
                anyhow::bail!(
                    "model kind {:?} does not support --prompt (arch: {})",
                    meta.kind,
                    meta.arch
                );
            }
        }
        let tok = tokenizer.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "archive has no embedded tokenizer section; \
                 recompile with --tokenizer to use --prompt"
            )
        })?;
        let gen_config = GenerationConfig {
            max_tokens: args.max_tokens,
            temperature: args.temperature,
            top_k: args.top_k,
            verbose: args.verbose,
        };
        run_generation(&runner, tok, prompt, &gen_config, model_meta.as_ref())?;
    } else {
        let mut graph_inputs = parse_inputs(&args.inputs)?;
        load_file_inputs(&args.input_files, &mut graph_inputs)?;

        if args.inputs.is_empty() && args.input_files.is_empty() {
            print_input_help(runner.plan());
        }

        let start = std::time::Instant::now();
        let outputs = runner.execute(&graph_inputs)?;
        let elapsed = start.elapsed();

        if let Some(tok) = &tokenizer {
            print_decoded_outputs(&outputs, tok);
        } else {
            print_typed_outputs(&outputs, runner.plan());
        }
        eprintln!(
            "executed in {:.3}ms (weights {})",
            elapsed.as_secs_f64() * 1000.0,
            format_bytes(runner.plan().weights().len() as u64),
        );
    }
    Ok(())
}

// ── Generation config ────────────────────────────────────────────────────

struct GenerationConfig {
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    verbose: bool,
}

// ── Sequence mode ─────────────────────────────────────────────────────────

/// How the generation loop handles input sequence length.
///
/// All shapes are fully baked at compile time — inputs are padded to the
/// compiled seq_len, and the attention_mask handles which positions are real.
enum SeqMode {
    /// Pad inputs to the compiled sequence length.
    FixedPad(usize),
}

fn resolve_seq_mode(runner: &HoloRunner) -> SeqMode {
    // All shapes are fully baked at compile time — no runtime shape resolution.
    // Read the compiled seq_len from the layer header (input_ids shape[1]).
    let compiled_seq = runner
        .plan()
        .layer_header()
        .into_iter()
        .flat_map(|lh| lh.layers.iter())
        .flat_map(|l| l.inputs.iter())
        .find(|p| p.name == "input_ids")
        .and_then(|p| p.shape.get(1).copied())
        .filter(|&s| s > 0)
        .map(|s| s as usize);

    if let Some(seq) = compiled_seq {
        // Pad inputs to the compiled seq_len. The attention_mask (ONNX models)
        // zeroes out padding positions so the model ignores them.
        SeqMode::FixedPad(seq)
    } else {
        // Fallback: read from model metadata or default to 2048.
        let meta_seq = load_meta_seq_len(runner);
        SeqMode::FixedPad(meta_seq.unwrap_or(2048))
    }
}

/// Try to read max_seq_len from embedded ModelMetaSection.
fn load_meta_seq_len(runner: &HoloRunner) -> Option<usize> {
    let meta: ModelMetaSection = load_section(runner.archive_bytes(), runner.plan(), SECTION_MODEL_META)?;
    if meta.max_seq_len > 0 {
        Some(meta.max_seq_len as usize)
    } else {
        None
    }
}

// ── Generation loop ────────────────────────────────────────────────────────

/// Autoregressive text generation loop using shape-aware execution.
fn run_generation(
    runner: &HoloRunner,
    tok_section: &TokenizerSection,
    prompt: &str,
    config: &GenerationConfig,
    model_meta: Option<&ModelMetaSection>,
) -> anyhow::Result<()> {
    let plan = runner.plan();
    let encoder = MiniBpeEncoder::from_tokenizer_section(tok_section);
    let input_dtype = resolve_input_dtype(plan, "input_ids");
    let seq_mode = resolve_seq_mode(runner);

    let mut token_ids = encoder.encode(prompt);
    let prompt_len = token_ids.len();

    // Startup diagnostics.
    eprintln!(
        "prompt: {} tokens (vocab_size={}, input_dtype={})",
        prompt_len,
        encoder.vocab_size(),
        input_dtype.name(),
    );
    let SeqMode::FixedPad(n) = &seq_mode;
    eprintln!("seq_mode: fixed pad to {n}");
    eprintln!(
        "sampling: temperature={:.2}, top_k={}, rep_penalty=1.3 (generated tokens only)",
        config.temperature,
        config.top_k,
    );

    let input_slot = plan
        .graph()
        .input_names
        .iter()
        .position(|n| n == "input_ids")
        .unwrap_or(0) as u32;

    let mask_slot = plan
        .graph()
        .input_names
        .iter()
        .position(|n| n == "attention_mask")
        .map(|i| i as u32);
    let mask_dtype = mask_slot.map(|_| resolve_input_dtype(plan, "attention_mask"));

    // position_ids: injected by PositionIdsInjection pass for KV cache decode.
    let pos_slot = plan
        .graph()
        .input_names
        .iter()
        .position(|n| n == "position_ids")
        .map(|i| i as u32);

    let vocab_size = encoder.vocab_size();
    let bytes_per_pos = vocab_size * 4;
    let start = std::time::Instant::now();

    // KV cache state for pipeline archives.
    // Created on first use with architecture params from the compiled model.
    let mut kv_state: Option<hologram::KvCacheState> = None;
    let use_kv_cache = runner.is_pipeline();
    if use_kv_cache {
        eprintln!("kv_cache: enabled (pipeline archive)");
    }

    for step in 0..config.max_tokens {
        // Build the effective token sequence for this step.
        // With KV cache: step 0 = full prompt (prefill), step 1+ = single token (decode).
        // The executor handles RoPE position offset and KV expansion internally.
        // With KV cache: step 0 = full prompt (prefill), step 1+ = single token (decode).
        // The executor handles RoPE position offset and KV expansion internally.
        let (effective_tokens, actual_len) = if use_kv_cache && step > 0 {
            let last = *token_ids.last().expect("no tokens");
            (vec![last], 1)
        } else {
            build_step_tokens(&token_ids, &seq_mode)
        };
        let padded_len = effective_tokens.len();

        let input_bytes = serialize_token_ids(&effective_tokens, input_dtype);
        let mut inputs = GraphInputs::new();
        inputs.set_with_shape(input_slot, input_bytes, vec![1, padded_len]);

        // Attention mask (only for models with an explicit mask input, e.g. ONNX).
        if let Some(slot) = mask_slot {
            let mask_dtype_val = mask_dtype.unwrap_or(WeightDType::I64);
            let mask_bytes = if actual_len < padded_len {
                serialize_mask(actual_len, padded_len, mask_dtype_val)
            } else {
                serialize_ones(padded_len, mask_dtype_val)
            };
            inputs.set_with_shape(slot, mask_bytes, vec![1, padded_len]);
        }

        // position_ids: absolute positions for each token in the input.
        // For prefill: [0, 1, 2, ..., actual_len-1, 0, 0, ..., 0] (padded)
        // For KV cache decode: [kv_write_pos] (single token at absolute position)
        if let Some(slot) = pos_slot {
            let pos_offset = if use_kv_cache && step > 0 {
                kv_state.as_ref().map(|kv| kv.write_pos()).unwrap_or(0)
            } else {
                0
            };
            let position_ids: Vec<i64> = (0..padded_len as i64)
                .map(|i| {
                    if (i as usize) < actual_len {
                        pos_offset as i64 + i
                    } else {
                        0 // padding positions
                    }
                })
                .collect();
            let pos_bytes: Vec<u8> = position_ids
                .iter()
                .flat_map(|&v| v.to_le_bytes())
                .collect();
            inputs.set_with_shape(slot, pos_bytes, vec![1, padded_len]);
        }

        // Execute: use KV cache if available, otherwise standard shape-aware.
        let outputs = if use_kv_cache {
            // Lazy-init KV cache on first call.
            if kv_state.is_none() {
                // Read KV architecture params from the already-loaded metadata.
                let meta = model_meta.as_ref()
                    .expect("KV cache requires ModelMetaSection in archive");
                let max_seq = meta.max_seq_len as usize;
                let n_layers = meta.n_layers;
                let n_kv_heads = meta.n_kv_heads;
                let head_dim = meta.head_dim;
                eprintln!(
                    "kv_cache: n_layers={n_layers} n_kv_heads={n_kv_heads} head_dim={head_dim} max_seq={max_seq}"
                );
                kv_state = Some(hologram::KvCacheState::new(
                    n_layers, n_kv_heads, head_dim, max_seq,
                ));
            }
            runner.execute_with_kv(
                &inputs,
                kv_state.as_mut().expect("kv_state initialized above"),
            )?
        } else {
            runner.execute(&inputs)?
        };

        let logit_data = match outputs.get(0) {
            Some((_, data)) => data,
            None => anyhow::bail!("model produced no output"),
        };

        let target_pos = actual_len.saturating_sub(1);

        // Per-step diagnostics (step 0 always, others if --verbose).
        if step == 0 || config.verbose {
            print_logit_diagnostics(logit_data, target_pos, vocab_size, tok_section, step);
        }

        // Extract logits at target_pos and sample.
        let logits_slice = extract_logits_at_pos(logit_data, target_pos, bytes_per_pos);
        let next_token = sample_next_token(
            logits_slice,
            &token_ids,
            prompt_len,
            config.temperature,
            config.top_k,
        );

        let next_token = match next_token {
            Some(id) => id,
            None => {
                eprintln!("\n[no logits in output]");
                break;
            }
        };

        if next_token == encoder.eos_id() {
            break;
        }

        let text = encoder.decode(&[next_token]);
        print!("{text}");
        std::io::stdout().flush().ok();

        token_ids.push(next_token);

        if step == 0 {
            let prefill_ms = start.elapsed().as_secs_f64() * 1000.0;
            eprintln!("\n[prefill {prefill_ms:.0}ms]");
        }
    }

    let elapsed = start.elapsed();
    let generated = token_ids.len() - prompt_len;
    let tok_per_sec = if elapsed.as_secs_f64() > 0.0 {
        generated as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };
    eprintln!(
        "\n[{generated} tokens in {:.1}s ({tok_per_sec:.1} tok/s), weights {}]",
        elapsed.as_secs_f64(),
        format_bytes(runner.plan().weights().len() as u64),
    );
    Ok(())
}

/// Build the effective token sequence for one generation step.
///
/// Returns `(tokens, actual_len)` where `actual_len` is the number of
/// real (non-padding) tokens. For variable seq mode these are equal.
fn build_step_tokens(token_ids: &[u32], mode: &SeqMode) -> (Vec<u32>, usize) {
    let SeqMode::FixedPad(max_seq) = mode;
    let actual_len = token_ids.len().min(*max_seq);
    if token_ids.len() > *max_seq {
        let truncated = token_ids[token_ids.len() - max_seq..].to_vec();
        (truncated, actual_len)
    } else {
        let mut padded = token_ids.to_vec();
        padded.resize(*max_seq, 0);
        (padded, actual_len)
    }
}

/// Extract the logits slice for a given target position from the output buffer.
///
/// For output shape `[1, seq, vocab]`, the logits at position `pos` start at
/// `pos * vocab * 4` bytes. Falls back to the last `vocab * 4` bytes if the
/// offset is out of range.
fn extract_logits_at_pos(
    logit_data: &[u8],
    target_pos: usize,
    bytes_per_pos: usize,
) -> &[u8] {
    let offset = target_pos * bytes_per_pos;
    if logit_data.len() >= offset + bytes_per_pos {
        &logit_data[offset..offset + bytes_per_pos]
    } else if logit_data.len() >= bytes_per_pos {
        &logit_data[logit_data.len() - bytes_per_pos..]
    } else {
        logit_data
    }
}

// ── Sampling ──────────────────────────────────────────────────────────────

/// Sample the next token from logits with temperature, top-k, and repetition penalty.
///
/// Repetition penalty is applied ONLY to generated tokens (after `prompt_len`),
/// not to prompt tokens. This prevents the model from being biased against common
/// words that appeared in the prompt.
fn sample_next_token(
    logit_bytes: &[u8],
    token_ids: &[u32],
    prompt_len: usize,
    temperature: f32,
    top_k: usize,
) -> Option<u32> {
    const PENALTY: f32 = 1.3;
    const WINDOW: usize = 64;

    if !logit_bytes.len().is_multiple_of(4) {
        return None;
    }
    let mut logits: Vec<f32> = logit_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().expect("4-byte chunk")))
        .collect();

    // Repetition penalty on generated tokens only (not prompt).
    let gen_start = prompt_len.max(token_ids.len().saturating_sub(WINDOW));
    for &tok in &token_ids[gen_start..] {
        let idx = tok as usize;
        if idx < logits.len() {
            if logits[idx] > 0.0 {
                logits[idx] /= PENALTY;
            } else {
                logits[idx] *= PENALTY;
            }
        }
    }

    if temperature <= 0.0 || temperature < 1e-6 {
        // Greedy argmax.
        return logits
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_finite())
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32);
    }

    // Temperature-scaled top-k sampling.
    // 1. Apply temperature.
    for v in &mut logits {
        if v.is_finite() {
            *v /= temperature;
        }
    }

    // 2. Find top-k candidates.
    let k = if top_k == 0 || top_k >= logits.len() {
        logits.len()
    } else {
        top_k
    };

    let mut indexed: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);

    if indexed.is_empty() {
        return None;
    }

    // 3. Softmax over top-k.
    let max_logit = indexed[0].1;
    let mut probs: Vec<(usize, f32)> = indexed
        .iter()
        .map(|&(i, v)| (i, (v - max_logit).exp()))
        .collect();
    let sum: f32 = probs.iter().map(|(_, p)| p).sum();
    if sum <= 0.0 || !sum.is_finite() {
        return Some(indexed[0].0 as u32);
    }
    for (_, p) in &mut probs {
        *p /= sum;
    }

    // 4. Sample from the distribution using a simple LCG PRNG.
    // Use a per-step seed derived from token_ids to be reproducible-ish
    // but different each step.
    let seed = token_ids.iter().fold(0x517cc1b7u64, |h, &t| {
        h.wrapping_mul(6364136223846793005).wrapping_add(t as u64)
    });
    let r = ((seed >> 16) as f32) / (u32::MAX as f32);
    let r = r.clamp(0.0, 1.0);

    let mut cumulative = 0.0_f32;
    for &(idx, p) in &probs {
        cumulative += p;
        if r <= cumulative {
            return Some(idx as u32);
        }
    }

    // Fallback: return the highest-probability token.
    Some(probs[0].0 as u32)
}

// ── Diagnostics ───────────────────────────────────────────────────────────

fn print_logit_diagnostics(
    logit_data: &[u8],
    target_pos: usize,
    vocab_size: usize,
    tok_section: &TokenizerSection,
    step: usize,
) {
    let bytes_per_pos = vocab_size * 4;
    let offset = target_pos * bytes_per_pos;
    if logit_data.len() < offset + bytes_per_pos {
        return;
    }
    let slice = &logit_data[offset..offset + bytes_per_pos];
    let floats: Vec<f32> = slice
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))
        .collect();

    let nan_count = floats.iter().filter(|f| f.is_nan()).count();
    let inf_count = floats.iter().filter(|f| f.is_infinite()).count();
    let zero_count = floats.iter().filter(|&&f| f == 0.0).count();
    let min = floats.iter().copied().reduce(f32::min).unwrap_or(0.0);
    let max = floats.iter().copied().reduce(f32::max).unwrap_or(0.0);
    let mean = floats.iter().sum::<f32>() / floats.len() as f32;

    eprintln!(
        "[logit-debug] step={step} pos={target_pos} vocab={vocab_size} \
         total_bytes={} nan={nan_count} inf={inf_count} zero={zero_count} \
         min={min:.4} max={max:.4} mean={mean:.6}",
        logit_data.len()
    );

    let mut indexed: Vec<(usize, f32)> = floats.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (i, (tok_id, val)) in indexed.iter().take(5).enumerate() {
        let tok_str = tok_section.id_to_token(*tok_id as u32).unwrap_or("<unk>");
        eprintln!(
            "[logit-debug] top-{}: id={tok_id} val={val:.6} \"{tok_str}\"",
            i + 1
        );
    }
}

// ── Section loading ────────────────────────────────────────────────────────

fn load_section<T>(
    archive_bytes: &[u8],
    plan: &hologram::LoadedPlan,
    kind: u32,
) -> Option<T>
where
    T: SectionDeserialize,
{
    let entry = plan.sections().find(kind)?;
    let offset = entry.offset as usize;
    let size = entry.size as usize;
    if offset + size > archive_bytes.len() {
        return None;
    }
    T::deserialize_section(&archive_bytes[offset..offset + size]).ok()
}

/// Load a section from raw archive bytes (loads the plan on-the-fly).
/// Used as fallback for pipeline archives where sections are in the wrapper.
fn load_section_from_raw<T: SectionDeserialize>(raw: &[u8], kind: u32) -> Option<T> {
    let plan = hologram::load_from_bytes(raw).ok()?;
    load_section(raw, &plan, kind)
}

trait SectionDeserialize: Sized {
    fn deserialize_section(bytes: &[u8]) -> anyhow::Result<Self>;
}

impl SectionDeserialize for TokenizerSection {
    fn deserialize_section(bytes: &[u8]) -> anyhow::Result<Self> {
        TokenizerSection::deserialize_from(bytes)
            .map_err(|e| anyhow::anyhow!("deserialize TokenizerSection: {e}"))
    }
}

impl SectionDeserialize for ModelMetaSection {
    fn deserialize_section(bytes: &[u8]) -> anyhow::Result<Self> {
        ModelMetaSection::deserialize_from(bytes)
            .map_err(|e| anyhow::anyhow!("deserialize ModelMetaSection: {e}"))
    }
}

// ── Input parsing ──────────────────────────────────────────────────────────

fn parse_inputs(raw: &[String]) -> anyhow::Result<GraphInputs> {
    let mut inputs = GraphInputs::new();
    for s in raw {
        let (idx, bytes) = parse_input_pair(s)?;
        inputs.set(idx, bytes);
    }
    Ok(inputs)
}

fn parse_input_pair(s: &str) -> anyhow::Result<(u32, Vec<u8>)> {
    let (idx_str, hex_str) = s
        .split_once(':')
        .ok_or_else(|| anyhow::anyhow!("expected INDEX:HEX, got {s:?}"))?;
    let idx: u32 = idx_str
        .parse()
        .map_err(|_| anyhow::anyhow!("invalid index {idx_str:?} in {s:?}"))?;
    let bytes = decode_hex(hex_str)
        .map_err(|e| anyhow::anyhow!("invalid hex in {s:?}: {e}"))?;
    Ok((idx, bytes))
}

fn load_file_inputs(raw: &[String], inputs: &mut GraphInputs) -> anyhow::Result<()> {
    for s in raw {
        let (idx_str, path_str) = s
            .split_once(':')
            .ok_or_else(|| anyhow::anyhow!("expected SLOT:PATH, got {s:?}"))?;
        let idx: u32 = idx_str
            .parse()
            .map_err(|_| anyhow::anyhow!("invalid slot {idx_str:?} in {s:?}"))?;
        let bytes = std::fs::read(path_str)
            .with_context(|| format!("reading input file {path_str:?}"))?;
        eprintln!("loaded slot {idx} from {path_str:?} ({} bytes)", bytes.len());
        inputs.set(idx, bytes);
    }
    Ok(())
}

fn decode_hex(s: &str) -> Result<Vec<u8>, String> {
    if !s.len().is_multiple_of(2) {
        return Err(format!("odd-length hex string: {s:?}"));
    }
    (0..s.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&s[i..i + 2], 16)
                .map_err(|_| format!("invalid hex byte {:?}", &s[i..i + 2]))
        })
        .collect()
}

// ── Output formatting ──────────────────────────────────────────────────────

fn print_model_info(
    plan: &hologram::LoadedPlan,
    model_meta: &Option<ModelMetaSection>,
) {
    if let Some(meta) = model_meta {
        eprintln!(
            "model: {:?} arch={} seq_len={} prompt={}",
            meta.kind, meta.arch, meta.max_seq_len, meta.supports_prompt,
        );
        if !meta.description.is_empty() {
            eprintln!("  {}", meta.description);
        }
    }

    let lh = match plan.layer_header() {
        Some(lh) => lh,
        None => {
            eprintln!("no layer header; using direct graph execution");
            return;
        }
    };
    for layer in &lh.layers {
        let inputs: Vec<String> = layer
            .inputs
            .iter()
            .map(|p| format!("{}:{:?}:{}", p.name, p.shape, p.dtype.name()))
            .collect();
        let outputs: Vec<String> = layer
            .outputs
            .iter()
            .map(|p| format!("{}:{:?}:{}", p.name, p.shape, p.dtype.name()))
            .collect();
        eprintln!(
            "layer {:?}: {:?} [{}] -> [{}]",
            layer.name,
            layer.entrypoint,
            inputs.join(", "),
            outputs.join(", "),
        );
    }
}

fn print_input_help(plan: &hologram::LoadedPlan) {
    let lh = match plan.layer_header() {
        Some(lh) => lh,
        None => {
            eprintln!("inputs (by graph name):");
            for (i, name) in plan.graph().input_names.iter().enumerate() {
                eprintln!("  slot {i}: \"{name}\"");
            }
            return;
        }
    };
    eprintln!("expected inputs:");
    for layer in &lh.layers {
        for (i, port) in layer.inputs.iter().enumerate() {
            let elem_bytes = port.dtype.byte_size();
            let total_elems: u64 = port.shape.iter().product();
            let total_bytes = if elem_bytes > 0 && total_elems > 0 {
                format!("{} bytes", total_elems as usize * elem_bytes)
            } else {
                "dynamic".into()
            };
            eprintln!(
                "  slot {i} '{}': shape {:?} dtype {} ({})",
                port.name,
                port.shape,
                port.dtype.name(),
                total_bytes,
            );
        }
    }
}

fn print_typed_outputs(
    outputs: &hologram::GraphOutputs,
    plan: &hologram::LoadedPlan,
) {
    use hologram::hologram_archive::entrypoint::TensorPort;
    let output_ports: Vec<TensorPort> = plan
        .layer_header()
        .into_iter()
        .flat_map(|lh| lh.layers.iter())
        .flat_map(|l| l.outputs.iter().cloned())
        .collect();

    for i in 0..outputs.len() {
        if let Some((name, data)) = outputs.get(i) {
            let dtype = output_ports.get(i).map(|p| p.dtype);
            match dtype {
                Some(WeightDType::F32) if data.len() >= 4 => {
                    let n = data.len() / 4;
                    let floats: Vec<f32> = (0..n)
                        .map(|j| f32::from_le_bytes(data[j * 4..(j + 1) * 4].try_into().expect("4 bytes")))
                        .collect();
                    if floats.len() <= 16 {
                        println!("{name}: {floats:?}");
                    } else {
                        let min = floats.iter().copied().reduce(f32::min).unwrap_or(0.0);
                        let max = floats.iter().copied().reduce(f32::max).unwrap_or(0.0);
                        let mean = floats.iter().sum::<f32>() / floats.len() as f32;
                        println!(
                            "{name}: [{} f32] min={min:.4} max={max:.4} mean={mean:.4}",
                            floats.len(),
                        );
                    }
                }
                Some(WeightDType::I64) if data.len() >= 8 => {
                    let n = data.len() / 8;
                    let ints: Vec<i64> = (0..n)
                        .map(|j| i64::from_le_bytes(data[j * 8..(j + 1) * 8].try_into().expect("8 bytes")))
                        .collect();
                    if ints.len() <= 32 {
                        println!("{name}: {ints:?}");
                    } else {
                        println!("{name}: [{} i64 values]", ints.len());
                    }
                }
                Some(WeightDType::I32) if data.len() >= 4 => {
                    let n = data.len() / 4;
                    let ints: Vec<i32> = (0..n)
                        .map(|j| i32::from_le_bytes(data[j * 4..(j + 1) * 4].try_into().expect("4 bytes")))
                        .collect();
                    if ints.len() <= 32 {
                        println!("{name}: {ints:?}");
                    } else {
                        println!("{name}: [{} i32 values]", ints.len());
                    }
                }
                _ => {
                    let hex: String = data.iter().take(64).map(|b| format!("{b:02x}")).collect();
                    let suffix = if data.len() > 64 { "..." } else { "" };
                    println!("{name}: {hex}{suffix} ({} bytes)", data.len());
                }
            }
        }
    }
}

fn print_decoded_outputs(outputs: &hologram::GraphOutputs, tok: &TokenizerSection) {
    for i in 0..outputs.len() {
        if let Some((name, data)) = outputs.get(i) {
            if let Some(token_id) = TokenizerSection::argmax_f32(data) {
                let text = tok.id_to_token(token_id).unwrap_or("<unk>");
                println!("{name}: token_id={token_id} \"{text}\"");
            } else {
                let hex: String = data.iter().take(64).map(|b| format!("{b:02x}")).collect();
                let suffix = if data.len() > 64 { "..." } else { "" };
                println!("{name}: {hex}{suffix} ({} bytes)", data.len());
            }
        }
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

fn resolve_input_dtype(
    plan: &hologram::LoadedPlan,
    name: &str,
) -> WeightDType {
    plan.layer_header()
        .into_iter()
        .flat_map(|lh| lh.layers.iter())
        .flat_map(|l| l.inputs.iter())
        .find(|p| p.name == name)
        .map(|p| {
            if p.dtype == WeightDType::U8 && p.shape == [1] {
                WeightDType::I64
            } else {
                p.dtype
            }
        })
        .unwrap_or(WeightDType::I64)
}

fn serialize_token_ids(ids: &[u32], dtype: WeightDType) -> Vec<u8> {
    match dtype {
        WeightDType::I32 => ids
            .iter()
            .flat_map(|&id| (id as i32).to_le_bytes())
            .collect(),
        _ => ids
            .iter()
            .flat_map(|&id| (id as i64).to_le_bytes())
            .collect(),
    }
}

fn serialize_ones(n: usize, dtype: WeightDType) -> Vec<u8> {
    match dtype {
        WeightDType::I32 => (0..n).flat_map(|_| 1i32.to_le_bytes()).collect(),
        WeightDType::F32 => (0..n).flat_map(|_| 1.0f32.to_le_bytes()).collect(),
        _ => (0..n).flat_map(|_| 1i64.to_le_bytes()).collect(),
    }
}

fn serialize_mask(real_len: usize, total_len: usize, dtype: WeightDType) -> Vec<u8> {
    match dtype {
        WeightDType::I32 => (0..total_len)
            .flat_map(|i| if i < real_len { 1i32.to_le_bytes() } else { 0i32.to_le_bytes() })
            .collect(),
        WeightDType::F32 => (0..total_len)
            .flat_map(|i| if i < real_len { 1.0f32.to_le_bytes() } else { 0.0f32.to_le_bytes() })
            .collect(),
        _ => (0..total_len)
            .flat_map(|i| if i < real_len { 1i64.to_le_bytes() } else { 0i64.to_le_bytes() })
            .collect(),
    }
}

fn format_bytes(n: u64) -> String {
    if n >= 1024 * 1024 * 1024 {
        format!("{:.1} GiB", n as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if n >= 1024 * 1024 {
        format!("{:.1} MiB", n as f64 / (1024.0 * 1024.0))
    } else if n >= 1024 {
        format!("{:.1} KiB", n as f64 / 1024.0)
    } else {
        format!("{n} B")
    }
}
