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
use tracing::{debug, info, warn};

use crate::compiler::HoloRunner;

// ── CLI args ───────────────────────────────────────────────────────────────

/// Arguments for the `hologram-ai run` subcommand.
#[derive(Args)]
pub struct RunArgs {
    /// Path to the `.holo` file to execute.
    /// If `--model-config` is provided, this is optional and inferred from the
    /// config's `[model].output` directory.
    pub file: Option<PathBuf>,
    /// Path to a hologram.toml model config file. Reads `[run].prompt`,
    /// `[run].max_tokens`, and infers the `.holo` path from `[model].output`.
    /// CLI flags override config values.
    #[arg(long, value_name = "FILE")]
    pub model_config: Option<PathBuf>,
    /// Input values as `INDEX:HEX` pairs (e.g. `--input 0:deadbeef`).
    #[arg(long = "input", value_name = "INDEX:HEX")]
    pub inputs: Vec<String>,
    /// Input from file as `SLOT:PATH` pairs (e.g. `--input-file 0:input.bin`).
    #[arg(long = "input-file", value_name = "SLOT:PATH")]
    pub input_files: Vec<String>,
    /// Text prompt for autoregressive generation (requires embedded tokenizer).
    #[arg(long)]
    pub prompt: Option<String>,
    /// Maximum tokens to generate when using `--prompt`.
    /// Defaults to the model's max_seq_len if not specified.
    #[arg(long)]
    pub max_tokens: Option<usize>,
    /// Sampling temperature (0.0 = greedy argmax, default: 0.7).
    #[arg(long, default_value = "0.7")]
    pub temperature: f32,
    /// Top-k sampling: only consider the top K tokens (default: 40, 0 = disabled).
    #[arg(long, default_value = "40")]
    pub top_k: usize,
    /// Print per-step logit diagnostics.
    #[arg(long)]
    pub verbose: bool,
    /// Directory for decompressed archive cache. Compressed archives are
    /// decompressed once and cached here for instant mmap loading.
    /// Default: from config file, or next to the archive file.
    #[arg(long, value_name = "DIR")]
    pub cache_dir: Option<PathBuf>,
    /// Path to a hologram config file (TOML). Overrides the default
    /// config search (~/.hologram/config.toml, .hologram/config.toml).
    #[arg(long, value_name = "FILE")]
    pub config: Option<PathBuf>,
    /// KV cache quantization: `f32` (default), `q8`, `q4`, or `q8:q4` for
    /// asymmetric K:V precision. V compression is nearly free; K precision
    /// dominates quality.
    #[arg(long, value_name = "TYPE", default_value = "f32")]
    pub kv_cache: String,
    /// Number of boundary layers (at start and end) kept at f32 regardless
    /// of `--kv-cache` setting. Default: 2.
    #[arg(long, default_value = "2")]
    pub kv_boundary_layers: usize,
    /// Enable Walsh-Hadamard rotation before V quantization. Gaussianizes
    /// the distribution for better quantization efficiency.
    #[arg(long)]
    pub kv_wht: bool,
    /// Enable speculative decoding: generate N draft tokens greedily, then
    /// verify against the target model. Accepted tokens are emitted as a
    /// batch, giving up to N+1 tokens per cycle.
    #[arg(long)]
    pub speculative: bool,
    /// Number of draft tokens per speculative decode cycle (default: 4).
    #[arg(long, default_value = "4")]
    pub draft_steps: usize,
    /// Stop generation when the decoded output ends with this string. Can be
    /// passed multiple times for multiple stop sequences. The EOS token from
    /// the tokenizer is always a stop condition in addition to these.
    #[arg(long = "stop", value_name = "STRING")]
    pub stop: Vec<String>,
}

// ── KV cache config parsing ───────────────────────────────────────────────

/// Parse a `--kv-cache` string into a `KvCacheConfig`.
///
/// Accepted formats: `f32`, `q8`, `q4`, `q8:q4` (asymmetric K:V).
pub fn parse_kv_cache_config(
    kv_cache: &str,
    boundary_layers: usize,
    wht: bool,
) -> anyhow::Result<hologram::kv_cache::KvCacheConfig> {
    use hologram::kv_cache::{KvBits, KvCacheConfig};

    let parse_bits = |s: &str| -> anyhow::Result<KvBits> {
        match s.trim() {
            "f32" => Ok(KvBits::F32),
            "q8" => Ok(KvBits::Q8),
            "q4" => Ok(KvBits::Q4),
            other => anyhow::bail!("unknown KV cache type '{other}', expected f32/q8/q4"),
        }
    };

    let (k_bits, v_bits) = if let Some((k, v)) = kv_cache.split_once(':') {
        (parse_bits(k)?, parse_bits(v)?)
    } else {
        let bits = parse_bits(kv_cache)?;
        (bits, bits)
    };

    Ok(KvCacheConfig {
        k_bits,
        v_bits,
        boundary_layers,
        wht_rotation: wht,
    })
}

// ── Entry point ────────────────────────────────────────────────────────────

/// Execute the run command using shape-aware inference.
pub fn execute(mut args: RunArgs) -> anyhow::Result<()> {
    // Load model config if provided — fill in missing CLI args.
    if let Some(ref config_path) = args.model_config {
        let text = std::fs::read_to_string(config_path)
            .with_context(|| format!("reading model config {}", config_path.display()))?;
        let cfg: toml::Value = toml::from_str(&text)
            .with_context(|| format!("parsing model config {}", config_path.display()))?;
        let config_dir = config_path.parent().unwrap_or(std::path::Path::new("."));

        // Infer .holo path from [model].output if --file not given.
        if args.file.is_none() {
            if let Some(output) = cfg.get("model").and_then(|m| m.get("output")).and_then(|v| v.as_str()) {
                let output_dir = config_dir.join(output);
                // Find the .holo file in the output directory.
                let holo = if output_dir.is_dir() {
                    std::fs::read_dir(&output_dir)?
                        .filter_map(|e| e.ok())
                        .find(|e| e.path().extension().map_or(false, |ext| ext == "holo"))
                        .map(|e| e.path())
                        .with_context(|| format!("no .holo file found in {}", output_dir.display()))?
                } else if output_dir.extension().map_or(false, |ext| ext == "holo") {
                    output_dir
                } else {
                    anyhow::bail!("cannot infer .holo path from config output: {}", output_dir.display());
                };
                args.file = Some(holo);
            }
        }

        // Fill in [run] defaults.
        if let Some(run) = cfg.get("run") {
            if args.prompt.is_none() {
                if let Some(prompt) = run.get("prompt").and_then(|v| v.as_str()) {
                    args.prompt = Some(prompt.to_string());
                }
            }
            if args.max_tokens.is_none() {
                if let Some(max) = run.get("max_tokens").and_then(|v| v.as_integer()) {
                    args.max_tokens = Some(max as usize);
                }
            }
        }
    }

    let file = args.file.as_ref()
        .context("no .holo file specified — use a positional argument or --model-config")?;

    let load_start = std::time::Instant::now();
    let runner = HoloRunner::from_path(
        file,
        args.cache_dir.as_deref(),
        args.config.as_deref(),
    )
    .with_context(|| format!("loading archive {}", file.display()))?;
    info!(
        "archive loaded in {:.1}ms",
        load_start.elapsed().as_secs_f64() * 1000.0
    );

    // Load optional metadata sections.
    // Try the effective bytes (sub-archive) first, then fall back to the raw
    // bytes (pipeline wrapper where sections are embedded outside the sub-archive).
    // Pre-parse the raw plan once to avoid re-parsing per section.
    let effective = runner.archive_bytes();
    let raw = runner.raw_bytes();
    let raw_plan = hologram::load_from_bytes(raw).ok();
    let tokenizer = load_section::<TokenizerSection>(effective, runner.plan(), SECTION_TOKENIZER)
        .or_else(|| {
            raw_plan
                .as_ref()
                .and_then(|p| load_section::<TokenizerSection>(raw, p, SECTION_TOKENIZER))
        });
    let model_meta = load_section::<ModelMetaSection>(effective, runner.plan(), SECTION_MODEL_META)
        .or_else(|| {
            raw_plan
                .as_ref()
                .and_then(|p| load_section::<ModelMetaSection>(raw, p, SECTION_MODEL_META))
        });

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
            kv_cache: args.kv_cache.clone(),
            kv_boundary_layers: args.kv_boundary_layers,
            kv_wht: args.kv_wht,
            speculative: args.speculative,
            draft_steps: args.draft_steps,
            stop: args.stop.clone(),
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
        info!(
            "executed in {:.3}ms (weights {})",
            elapsed.as_secs_f64() * 1000.0,
            format_bytes(runner.plan().weights().len() as u64),
        );
    }
    Ok(())
}

// ── Generation config ────────────────────────────────────────────────────

struct GenerationConfig {
    max_tokens: Option<usize>,
    temperature: f32,
    top_k: usize,
    verbose: bool,
    kv_cache: String,
    kv_boundary_layers: usize,
    kv_wht: bool,
    speculative: bool,
    draft_steps: usize,
    stop: Vec<String>,
}

// ── Sequence mode ─────────────────────────────────────────────────────────

/// How the generation loop handles input sequence length.
///
/// All shapes are fully baked at compile time — inputs are padded to the
/// compiled seq_len, and the attention_mask handles which positions are real.
enum SeqMode {
    /// Pad inputs to the compiled sequence length.
    #[allow(dead_code)]
    FixedPad(usize),
    /// Variable-length: use actual prompt length (no padding).
    /// Requires hologram executor to resolve baked FloatOp params from
    /// runtime buffer sizes (resolve_size in dispatch_float_ctx/into).
    Variable { max_seq: usize },
}

fn resolve_seq_mode(runner: &HoloRunner, prompt_len: Option<usize>) -> SeqMode {
    let max_seq = load_meta_seq_len(runner).unwrap_or(2048);

    // Detect compiled seq_len from the graph's first 2D input shape [1, seq].
    let graph = runner.plan().graph();
    let compiled_seq: Option<usize> = graph
        .node_shapes
        .iter()
        .find(|(_, shape)| shape.len() == 2 && shape[0] == 1 && shape[1] > 1)
        .map(|(_, shape)| shape[1]);

    // When a ShapeContextGraph is available AND the prompt fits within the
    // compiled seq_len, use variable-length mode (no padding waste).
    //
    // Prompts exceeding compiled seq_len require 0-sentinel op parameters
    // (Plan 045) — shape metadata overrides alone can't fix baked op params.
    if runner.has_shape_context() {
        let prompt = prompt_len.unwrap_or(0);
        let fits = compiled_seq.is_none_or(|c| prompt <= c);
        if fits {
            info!("shape context available — using variable-length execution");
            return SeqMode::Variable { max_seq };
        }
        warn!(
            "prompt length={prompt} exceeds compiled seq_len={}. \
             Recompile with a larger `--seq-len` for correct results.",
            compiled_seq.unwrap_or(0),
        );
    }

    if let (Some(compiled), Some(prompt)) = (compiled_seq, prompt_len) {
        if compiled != prompt && !runner.has_shape_context() {
            warn!(
                "compiled seq_len={compiled} does not match prompt length={prompt}. \
                 Recompile with `--seq-len {prompt}` for correct results. \
                 Using FixedPad={compiled} (output may be incorrect)."
            );
        }
    }

    if let Some(seq) = compiled_seq {
        SeqMode::FixedPad(seq)
    } else {
        SeqMode::Variable { max_seq }
    }
}

/// Try to read max_seq_len from embedded ModelMetaSection.
fn load_meta_seq_len(runner: &HoloRunner) -> Option<usize> {
    let meta: ModelMetaSection =
        load_section(runner.archive_bytes(), runner.plan(), SECTION_MODEL_META)?;
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

    let mut token_ids = encoder.encode(prompt);
    let prompt_len = token_ids.len();
    let seq_mode = resolve_seq_mode(runner, Some(prompt_len));
    info!("token_ids: {:?}", &token_ids);

    // Startup diagnostics.
    info!(
        "prompt: {} tokens (vocab_size={}, input_dtype={})",
        prompt_len,
        encoder.vocab_size(),
        input_dtype.name(),
    );
    match &seq_mode {
        SeqMode::FixedPad(n) => info!("seq_mode: fixed pad to {n}"),
        SeqMode::Variable { max_seq } => info!("seq_mode: variable (max {max_seq})"),
    }
    info!(
        "sampling: temperature={:.2}, top_k={}, rep_penalty=1.3 (generated tokens only)",
        config.temperature, config.top_k,
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

    // Infer vocab size from the model's compiled output shape rather than the
    // tokenizer. Models often pad vocab to a multiple (e.g., Qwen2 has
    // tokenizer vocab=151643 but model logit dim=151936). Using the tokenizer
    // vocab for position extraction causes a misaligned read at multi-token
    // prompts where the offset accumulates per position.
    let tokenizer_vocab = encoder.vocab_size();
    let mut vocab_size = tokenizer_vocab;
    let mut bytes_per_pos = vocab_size * 4;
    let start = std::time::Instant::now();

    // KV cache state — used for any model with attention layers (pipeline
    // or single-graph). Detected from ModelMetaSection: if n_layers > 0,
    // the model has KvWrite/KvRead ops that require cache state.
    let mut kv_state: Option<hologram::KvCacheState> = None;
    let use_kv_cache = model_meta.as_ref().is_some_and(|m| m.n_layers > 0);
    if use_kv_cache {
        info!("kv_cache: enabled");
    }

    let max_tokens = config.max_tokens.unwrap_or_else(|| {
        let limit = model_meta.map(|m| m.max_seq_len as usize).unwrap_or(2048);
        info!("max_tokens not set, using model max_seq_len: {limit}");
        limit
    });

    let mut decode_start: Option<std::time::Instant> = None;

    for step in 0..max_tokens {
        // With KV cache: step 0 = full prompt (prefill), step 1+ = single token (decode).
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
            let pos_bytes: Vec<u8> = position_ids.iter().flat_map(|&v| v.to_le_bytes()).collect();
            inputs.set_with_shape(slot, pos_bytes, vec![1, padded_len]);
        }

        // Execute: use KV cache if available, otherwise standard shape-aware.
        let step_start = std::time::Instant::now();
        let outputs = if use_kv_cache {
            // Lazy-init KV cache on first call.
            if kv_state.is_none() {
                // Read KV architecture params from the already-loaded metadata.
                let meta = model_meta
                    .as_ref()
                    .expect("KV cache requires ModelMetaSection in archive");
                let max_seq = meta.max_seq_len as usize;
                let n_layers = meta.n_layers;
                let n_kv_heads = meta.n_kv_heads;
                let head_dim = meta.head_dim;
                info!(
                    "kv_cache: n_layers={n_layers} n_kv_heads={n_kv_heads} head_dim={head_dim} max_seq={max_seq}"
                );
                let kv_config = parse_kv_cache_config(
                    &config.kv_cache,
                    config.kv_boundary_layers,
                    config.kv_wht,
                )
                .expect("invalid --kv-cache value");
                info!(
                    "kv_config: k={:?} v={:?} boundary={} wht={}",
                    kv_config.k_bits,
                    kv_config.v_bits,
                    kv_config.boundary_layers,
                    kv_config.wht_rotation
                );
                kv_state = Some(hologram::KvCacheState::with_config(
                    n_layers, n_kv_heads, head_dim, max_seq, kv_config,
                ));
            }
            let kv = kv_state.as_mut().expect("kv_state initialized above");
            // For padded prefill (step 0), only advance KV cache by actual
            // prompt length — padding positions are meaningless for K/V.
            if step == 0 && padded_len > actual_len {
                kv.set_advance_override(actual_len);
            }
            runner.execute_with_kv(&inputs, kv)?
        } else {
            runner.execute(&inputs)?
        };

        let logit_data = match outputs.get(0) {
            Some((_, data)) => data,
            None => anyhow::bail!("model produced no output"),
        };

        // On first step, infer the model's actual vocab dimension from the
        // output buffer. This handles models where the logit dimension differs
        // from the tokenizer vocab (e.g., Qwen2: vocab=151643, logit=151936).
        if step == 0 && actual_len > 0 {
            let total_f32 = logit_data.len() / 4;
            let inferred = total_f32 / actual_len;
            if inferred > 0 && inferred != vocab_size && total_f32 == inferred * actual_len {
                tracing::info!(
                    tokenizer_vocab,
                    model_vocab = inferred,
                    "using model output dim for logit extraction (differs from tokenizer)"
                );
                vocab_size = inferred;
                bytes_per_pos = vocab_size * 4;
            }
        }

        let target_pos = actual_len.saturating_sub(1);

        // Per-step diagnostics (always for first 3 steps for debugging).
        if step < 3 || config.verbose {
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
                warn!("no logits in output");
                break;
            }
        };

        // Allow at least one generated token before accepting EOS — the model
        // should never legitimately end the response on the very first token.
        if next_token == encoder.eos_id() && step > 0 {
            break;
        }

        // Stream the new token text. decode() strips leading ▁-spaces
        // which are word boundaries. Decode the growing suffix to preserve them.
        //
        // A multi-byte codepoint (e.g. an emoji) can span two BPE tokens, so
        // `prev_len` (byte length of the prior decode) may land mid-character
        // in the new decode. Defer printing until `prev_len` lands on a UTF-8
        // char boundary — the next token will flush the completed character.
        let prev_len = encoder.decode(&token_ids[prompt_len..]).len();
        token_ids.push(next_token);
        let full = encoder.decode(&token_ids[prompt_len..]);
        if prev_len <= full.len() && full.is_char_boundary(prev_len) {
            let new_text = &full[prev_len..];
            print!("{new_text}");
            std::io::stdout().flush().ok();
        }

        // Stop if the decoded output ends with any user-provided stop string.
        // The stop string is not stripped from output; the user sees it and
        // can split on it in post-processing.
        if !config.stop.is_empty()
            && config
                .stop
                .iter()
                .any(|s| !s.is_empty() && full.ends_with(s))
        {
            break;
        }

        let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;
        if step == 0 {
            let ttft_ms = start.elapsed().as_secs_f64() * 1000.0;
            debug!("TTFT {ttft_ms:.0}ms | prefill {step_ms:.1}ms");
            decode_start = Some(std::time::Instant::now());

            // Speculative decode: after prefill, switch to speculative loop.
            if config.speculative && use_kv_cache {
                let spec_config = crate::speculative::SpeculativeConfig {
                    draft_steps: config.draft_steps,
                    input_slot,
                    pos_slot,
                    mask_slot,
                    mask_dtype,
                    vocab_size,
                    bytes_per_pos,
                    input_dtype,
                };
                let kv = kv_state.as_mut().expect("kv initialized during prefill");
                let remaining = max_tokens.saturating_sub(1); // prefill used 1 token
                let mut speculative_tokens = 0usize;

                for _cycle in 0..remaining {
                    let last = *token_ids.last().expect("no tokens");
                    let result = crate::speculative::speculative_decode_step(
                        runner,
                        kv,
                        last,
                        &spec_config,
                    )?;

                    if result.accepted_tokens.is_empty() {
                        break;
                    }

                    let mut hit_stop = false;
                    for &tok in &result.accepted_tokens {
                        if tok == encoder.eos_id() {
                            // Stream what we have, then stop.
                            break;
                        }
                        let prev_len = encoder.decode(&token_ids[prompt_len..]).len();
                        token_ids.push(tok);
                        let full = encoder.decode(&token_ids[prompt_len..]);
                        let new_text = &full[prev_len..];
                        print!("{new_text}");
                        std::io::stdout().flush().ok();
                        speculative_tokens += 1;
                        if !config.stop.is_empty()
                            && config
                                .stop
                                .iter()
                                .any(|s| !s.is_empty() && full.ends_with(s))
                        {
                            hit_stop = true;
                            break;
                        }
                    }

                    if hit_stop
                        || result
                            .accepted_tokens
                            .iter()
                            .any(|&t| t == encoder.eos_id())
                    {
                        break;
                    }
                    if token_ids.len() - prompt_len >= remaining {
                        break;
                    }
                }

                info!(
                    "speculative: {speculative_tokens} tokens generated via speculative decoding"
                );
                break; // exit the standard decode loop
            }
        } else if config.verbose {
            debug!("step {step}: {step_ms:.1}ms");
        }
    }

    let _elapsed = start.elapsed();
    let generated = token_ids.len() - prompt_len;
    let decode_elapsed = decode_start
        .map(|d| d.elapsed().as_secs_f64())
        .unwrap_or(0.0);
    let decode_tokens = generated.saturating_sub(1); // first token is from prefill
    let decode_tok_s = if decode_elapsed > 0.0 && decode_tokens > 0 {
        decode_tokens as f64 / decode_elapsed
    } else {
        0.0
    };
    println!(
        "\n\n[{generated} tokens | TTFT {:.0}ms | decode {decode_tokens} tok in {decode_elapsed:.2}s ({decode_tok_s:.1} tok/s) | weights {}]",
        start.elapsed().as_secs_f64() * 1000.0 - decode_elapsed * 1000.0,
        format_bytes(runner.plan().weights().len() as u64),
    );
    Ok(())
}

/// Build the effective token sequence for one generation step.
///
/// Returns `(tokens, actual_len)` where `actual_len` is the number of
/// real (non-padding) tokens. For variable seq mode these are equal.
fn build_step_tokens(token_ids: &[u32], mode: &SeqMode) -> (Vec<u32>, usize) {
    match mode {
        SeqMode::FixedPad(max_seq) => {
            let max_seq = *max_seq;
            let actual_len = token_ids.len().min(max_seq);
            if token_ids.len() > max_seq {
                let truncated = token_ids[token_ids.len() - max_seq..].to_vec();
                (truncated, actual_len)
            } else {
                let mut padded = token_ids.to_vec();
                padded.resize(max_seq, 0);
                (padded, actual_len)
            }
        }
        SeqMode::Variable { max_seq } => {
            let actual_len = token_ids.len().min(*max_seq);
            let tokens = if token_ids.len() > *max_seq {
                token_ids[token_ids.len() - max_seq..].to_vec()
            } else {
                token_ids.to_vec()
            };
            (tokens, actual_len)
        }
    }
}

/// Extract the logits slice for a given target position from the output buffer.
///
/// For output shape `[1, seq, vocab]`, the logits at position `pos` start at
/// `pos * vocab * 4` bytes. Falls back to the last `vocab * 4` bytes if the
/// offset is out of range.
fn extract_logits_at_pos(logit_data: &[u8], target_pos: usize, bytes_per_pos: usize) -> &[u8] {
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

    if !logit_bytes.len().is_multiple_of(4) {
        return None;
    }
    let mut logits: Vec<f32> = logit_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().expect("4-byte chunk")))
        .collect();

    // Repetition penalty on ALL generated tokens (not prompt).
    // Covers the full output to prevent long-range repetition loops.
    for &tok in &token_ids[prompt_len..] {
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

    // 4. Sample from the distribution using xorshift64* PRNG.
    // Seed from token_ids + time nanos for non-deterministic sampling.
    let time_nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let mut s = token_ids
        .iter()
        .fold(0x517cc1b7u64.wrapping_add(time_nanos), |h, &t| {
            h.wrapping_mul(6364136223846793005).wrapping_add(t as u64)
        });
    // xorshift64* — three rounds for good avalanche.
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    s = s.wrapping_mul(0x2545F4914F6CDD1D);
    let r = (s >> 11) as f32 / ((1u64 << 53) as f32);
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

    debug!(
        "[logit-debug] step={step} pos={target_pos} vocab={vocab_size} \
         total_bytes={} nan={nan_count} inf={inf_count} zero={zero_count} \
         min={min:.4} max={max:.4} mean={mean:.6}",
        logit_data.len()
    );

    let mut indexed: Vec<(usize, f32)> = floats.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (i, (tok_id, val)) in indexed.iter().take(5).enumerate() {
        let tok_str = tok_section.id_to_token(*tok_id as u32).unwrap_or("<unk>");
        debug!(
            "[logit-debug] top-{}: id={tok_id} val={val:.6} \"{tok_str}\"",
            i + 1
        );
    }
}

// ── Section loading ────────────────────────────────────────────────────────

fn load_section<T>(archive_bytes: &[u8], plan: &hologram::LoadedPlan, kind: u32) -> Option<T>
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
#[allow(dead_code)]
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
    let bytes = decode_hex(hex_str).map_err(|e| anyhow::anyhow!("invalid hex in {s:?}: {e}"))?;
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
        let bytes =
            std::fs::read(path_str).with_context(|| format!("reading input file {path_str:?}"))?;
        info!(
            "loaded slot {idx} from {path_str:?} ({} bytes)",
            bytes.len()
        );
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

fn print_model_info(plan: &hologram::LoadedPlan, model_meta: &Option<ModelMetaSection>) {
    if let Some(meta) = model_meta {
        info!(
            "model: {:?} arch={} seq_len={} prompt={}",
            meta.kind, meta.arch, meta.max_seq_len, meta.supports_prompt,
        );
        if !meta.description.is_empty() {
            info!("  {}", meta.description);
        }
    }

    let lh = match plan.layer_header() {
        Some(lh) => lh,
        None => {
            warn!("no layer header; using direct graph execution");
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
        info!(
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
            info!("inputs (by graph name):");
            for (i, name) in plan.graph().input_names.iter().enumerate() {
                info!("  slot {i}: \"{name}\"");
            }
            return;
        }
    };
    info!("expected inputs:");
    for layer in &lh.layers {
        for (i, port) in layer.inputs.iter().enumerate() {
            let elem_bytes = port.dtype.byte_size();
            let total_elems: u64 = port.shape.iter().product();
            let total_bytes = if elem_bytes > 0 && total_elems > 0 {
                format!("{} bytes", total_elems as usize * elem_bytes)
            } else {
                "dynamic".into()
            };
            info!(
                "  slot {i} '{}': shape {:?} dtype {} ({})",
                port.name,
                port.shape,
                port.dtype.name(),
                total_bytes,
            );
        }
    }
}

fn print_typed_outputs(outputs: &hologram::GraphOutputs, plan: &hologram::LoadedPlan) {
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
                        .map(|j| {
                            f32::from_le_bytes(
                                data[j * 4..(j + 1) * 4].try_into().expect("4 bytes"),
                            )
                        })
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
                        .map(|j| {
                            i64::from_le_bytes(
                                data[j * 8..(j + 1) * 8].try_into().expect("8 bytes"),
                            )
                        })
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
                        .map(|j| {
                            i32::from_le_bytes(
                                data[j * 4..(j + 1) * 4].try_into().expect("4 bytes"),
                            )
                        })
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

fn resolve_input_dtype(plan: &hologram::LoadedPlan, name: &str) -> WeightDType {
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
            .flat_map(|i| {
                if i < real_len {
                    1i32.to_le_bytes()
                } else {
                    0i32.to_le_bytes()
                }
            })
            .collect(),
        WeightDType::F32 => (0..total_len)
            .flat_map(|i| {
                if i < real_len {
                    1.0f32.to_le_bytes()
                } else {
                    0.0f32.to_le_bytes()
                }
            })
            .collect(),
        _ => (0..total_len)
            .flat_map(|i| {
                if i < real_len {
                    1i64.to_le_bytes()
                } else {
                    0i64.to_le_bytes()
                }
            })
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
