//! `hologram-ai run` — execute a compiled `.holo` archive.
//!
//! Loads the archive into a [`HoloRunner`] (a thin `InferenceSession` wrapper)
//! and runs a forward pass over caller-supplied input buffers. The UOR-native
//! runtime needs no KV-cache, shape projection, or host config — the compiled
//! archive carries concrete shapes and a schedule, and content-addressed
//! elision handles repeated computation (architecture §5.3, §7).

use anyhow::{Context as _, Result};
use clap::Args;
use std::io::Write;
use std::path::PathBuf;

use crate::commands::generate::{self, GenConfig};
use crate::runner::HoloRunner;
use hologram_ai_tokenizer::NativeTokenizer;

/// Arguments for the `run` subcommand.
#[derive(Args, Debug)]
pub struct RunArgs {
    /// Path to the `.holo` file to execute.
    pub file: PathBuf,
    /// Input values as `INDEX:HEX` pairs (e.g. `--input 0:deadbeef`).
    #[arg(long = "input", value_name = "INDEX:HEX")]
    pub inputs: Vec<String>,
    /// Input from file as `INDEX:PATH` pairs (e.g. `--input-file 0:input.bin`).
    #[arg(long = "input-file", value_name = "INDEX:PATH")]
    pub input_files: Vec<String>,
    /// Print output bytes as hex (otherwise only shapes/sizes are printed).
    #[arg(long)]
    pub verbose: bool,

    // ── Text generation (causal LM) ──────────────────────────────────────────
    // When `--prompt` is given, `run` performs autoregressive generation instead
    // of a single raw forward pass. The `.holo` carries no tokenizer (closed
    // section set), so `--tokenizer` supplies the HuggingFace `tokenizer.json`.
    /// Prompt text — switches `run` into text-generation mode.
    #[arg(long)]
    pub prompt: Option<String>,
    /// Path to the model's HuggingFace `tokenizer.json` (required with `--prompt`).
    #[arg(long, value_name = "FILE")]
    pub tokenizer: Option<PathBuf>,
    /// Prompt template with a `{prompt}` placeholder (e.g. a chat template).
    #[arg(long, value_name = "TEMPLATE")]
    pub prompt_template: Option<String>,
    /// Maximum number of new tokens to generate.
    #[arg(long, default_value_t = 64)]
    pub max_tokens: usize,
    /// Sampling temperature; `0.0` is greedy/deterministic argmax.
    #[arg(long, default_value_t = 0.0)]
    pub temperature: f32,
    /// Restrict sampling to the `k` most-likely tokens.
    #[arg(long, value_name = "K")]
    pub top_k: Option<usize>,
    /// Stop string(s); generation halts when the decoded suffix contains one.
    #[arg(long)]
    pub stop: Vec<String>,
    /// Override the end-of-sequence token id (default: tokenizer's eos).
    #[arg(long, value_name = "ID")]
    pub eos: Option<u32>,
    /// RNG seed for temperature sampling (reproducibility).
    #[arg(long, default_value_t = 0x9E3779B97F4A7C15)]
    pub seed: u64,
}

/// Execute the `run` subcommand.
pub fn execute(args: RunArgs) -> Result<()> {
    if args.prompt.is_some() {
        return generate_cmd(args);
    }

    let mut runner = HoloRunner::from_path(&args.file, None)
        .with_context(|| format!("loading model {:?}", args.file))?;

    let n_inputs = runner.input_count();
    println!(
        "Loaded {:?}: {} input(s), {} output(s)",
        args.file,
        n_inputs,
        runner.output_count()
    );

    // Collect input byte-buffers indexed by graph-input position.
    let mut slots: Vec<Option<Vec<u8>>> = vec![None; n_inputs];
    for pair in &args.inputs {
        let (idx, bytes) = parse_hex_input(pair)?;
        store_slot(&mut slots, idx, bytes)?;
    }
    for pair in &args.input_files {
        let (idx, path) = split_index_path(pair)?;
        let bytes = std::fs::read(&path).with_context(|| format!("reading input file {path:?}"))?;
        store_slot(&mut slots, idx, bytes)?;
    }

    let missing: Vec<usize> = slots
        .iter()
        .enumerate()
        .filter_map(|(i, s)| s.is_none().then_some(i))
        .collect();
    if !missing.is_empty() {
        anyhow::bail!(
            "missing input(s) {missing:?}; supply each graph input via --input INDEX:HEX or --input-file INDEX:PATH"
        );
    }

    let owned: Vec<Vec<u8>> = slots.into_iter().map(|s| s.unwrap()).collect();
    let refs: Vec<&[u8]> = owned.iter().map(|v| v.as_slice()).collect();

    let outputs = runner.execute(&refs).context("inference failed")?;

    for (i, out) in outputs.iter().enumerate() {
        if args.verbose {
            println!(
                "output[{i}] ({} bytes): {}",
                out.bytes.len(),
                hex(&out.bytes)
            );
        } else {
            println!("output[{i}]: {} bytes", out.bytes.len());
        }
    }
    Ok(())
}

/// `run --prompt …` — autoregressive text generation over a causal LM.
fn generate_cmd(args: RunArgs) -> Result<()> {
    let prompt = args.prompt.as_deref().expect("generate_cmd requires --prompt");
    let tok_path = args.tokenizer.as_ref().context(
        "text generation needs the model tokenizer: pass --tokenizer path/to/tokenizer.json",
    )?;

    let tokenizer = NativeTokenizer::from_tokenizer_json(tok_path)
        .with_context(|| format!("loading tokenizer {tok_path:?}"))?;
    let mut runner = HoloRunner::from_path(&args.file, None)
        .with_context(|| format!("loading model {:?}", args.file))?;

    let cfg = GenConfig {
        max_tokens: args.max_tokens,
        temperature: args.temperature,
        top_k: args.top_k,
        stop: args.stop.clone(),
        eos: args.eos,
        seed: args.seed,
    };

    let templated = generate::apply_template(args.prompt_template.as_deref(), prompt);

    let mut stdout = std::io::stdout();
    // Echo the prompt so a streamed transcript reads coherently, then stream
    // the generated continuation token-by-token from inside generate_stream.
    print!("{prompt}");
    stdout.flush().ok();
    generate::generate_stream(&mut runner, &tokenizer, &templated, &cfg, &mut stdout)?;
    println!();
    Ok(())
}

// ── input parsing ─────────────────────────────────────────────────────────────

fn store_slot(slots: &mut [Option<Vec<u8>>], idx: usize, bytes: Vec<u8>) -> Result<()> {
    let n = slots.len();
    let slot = slots
        .get_mut(idx)
        .with_context(|| format!("input index {idx} out of range (model has {n} inputs)"))?;
    *slot = Some(bytes);
    Ok(())
}

/// Parse an `INDEX:HEX` pair.
fn parse_hex_input(s: &str) -> Result<(usize, Vec<u8>)> {
    let (idx, hex_str) = s
        .split_once(':')
        .with_context(|| format!("malformed --input {s:?}; expected INDEX:HEX"))?;
    let idx: usize = idx
        .trim()
        .parse()
        .with_context(|| format!("bad input index in {s:?}"))?;
    Ok((
        idx,
        decode_hex(hex_str.trim()).map_err(|e| anyhow::anyhow!("{e} in {s:?}"))?,
    ))
}

/// Parse an `INDEX:PATH` pair.
fn split_index_path(s: &str) -> Result<(usize, PathBuf)> {
    let (idx, path) = s
        .split_once(':')
        .with_context(|| format!("malformed --input-file {s:?}; expected INDEX:PATH"))?;
    let idx: usize = idx
        .trim()
        .parse()
        .with_context(|| format!("bad input index in {s:?}"))?;
    Ok((idx, PathBuf::from(path.trim())))
}

fn decode_hex(s: &str) -> Result<Vec<u8>, String> {
    let s = s.strip_prefix("0x").unwrap_or(s);
    if !s.len().is_multiple_of(2) {
        return Err("hex string has odd length".into());
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(|e| e.to_string()))
        .collect()
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}
