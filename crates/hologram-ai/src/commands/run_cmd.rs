//! `hologram-ai run` — execute a compiled `.holo` archive.
//!
//! Loads the archive into a [`HoloRunner`] (a thin `InferenceSession` wrapper)
//! and runs a forward pass over caller-supplied input buffers. The UOR-native
//! runtime needs no KV-cache, shape projection, or host config — the compiled
//! archive carries concrete shapes and a schedule, and content-addressed
//! elision handles repeated computation (architecture §5.3, §7).

use anyhow::{bail, Context as _, Result};
use clap::Args;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use crate::commands::generate::{self, GenConfig};
use crate::engine::{FixedSession, GrowableSession, SessionProvider};
use crate::runner::HoloRunner;
use crate::stats::ForwardStats;
use hologram_ai_tokenizer::{NativeTokenizer, Tokenizer};

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
    /// Synthesize any input not given via `--input`/`--input-file`, so an
    /// arbitrary model runs with one command. `zeros` (all-zero bytes, valid for
    /// every dtype), `ones`, or a numeric constant (e.g. `--fill 1.5`).
    #[arg(long, value_name = "zeros|ones|N")]
    pub fill: Option<String>,
    /// Print timing stats to stderr. In generation mode this reports prompt
    /// encode, prefill latency, and decode throughput.
    #[arg(long)]
    pub stats: bool,

    // ── Text generation (causal LM) ──────────────────────────────────────────
    // When `--prompt` is given, `run` performs autoregressive generation instead
    // of a single raw forward pass. The tokenizer is read from the archive's
    // baked-in extension; `--tokenizer` overrides it with an external file.
    /// Prompt text — switches `run` into text-generation mode.
    #[arg(long)]
    pub prompt: Option<String>,
    /// HuggingFace `tokenizer.json` override; defaults to the one baked into the
    /// archive (a compiled model is self-describing), or the one beside the model
    /// source.
    #[arg(long, value_name = "FILE")]
    pub tokenizer: Option<PathBuf>,
    /// Weight quantization for the growable (model-source) generation path:
    /// `none`/`f32`, `int8`, `int4`. Ignored when running a precompiled
    /// `.holo` (it is already quantized as compiled).
    #[arg(long, value_name = "SCHEME")]
    pub quantize: Option<String>,
    /// Prompt template with a `{prompt}` placeholder (e.g. a chat template).
    #[arg(long, value_name = "TEMPLATE")]
    pub prompt_template: Option<String>,
    /// Path to a HuggingFace `chat_template.jinja` override. When omitted,
    /// `run --prompt` auto-discovers an embedded or companion chat template.
    #[arg(long, value_name = "FILE")]
    pub chat_template: Option<PathBuf>,
    /// Maximum number of new tokens to generate.
    #[arg(long, default_value_t = 64)]
    pub max_tokens: usize,
    /// Sampling temperature; `0.0` is greedy/deterministic argmax.
    #[arg(long, default_value_t = 0.0)]
    pub temperature: f32,
    /// Restrict sampling to the `k` most-likely tokens.
    #[arg(long, value_name = "K")]
    pub top_k: Option<usize>,
    /// Print the first-step top-k token candidates from the logits row before
    /// sampling. Useful for debugging prompt formatting and output sanity.
    #[arg(long, value_name = "K")]
    pub decode_top_k: Option<usize>,
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

    let total_start = Instant::now();
    let load_start = Instant::now();
    let mut runner = HoloRunner::from_path(&args.file, None)
        .with_context(|| format!("loading model {:?}", args.file))?;
    let load = load_start.elapsed();

    let n_inputs = runner.input_count();
    let in_ports = runner.input_port_info();
    let in_sizes = runner.input_byte_sizes();
    println!(
        "Loaded {:?}: {} input(s), {} output(s)",
        args.file,
        n_inputs,
        runner.output_count()
    );
    for (i, (p, &bytes)) in in_ports.iter().zip(in_sizes.iter()).enumerate() {
        println!(
            "  input[{i}]: {} × {} = {bytes} bytes",
            dtype_name(p.dtype),
            p.element_count
        );
    }

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

    // Validate any explicitly-supplied input against the port's expected size —
    // a clear error beats a downstream `InputMismatch`.
    for (i, slot) in slots.iter().enumerate() {
        if let Some(buf) = slot {
            let want = in_sizes[i];
            if buf.len() != want {
                anyhow::bail!(
                    "input[{i}] is {} bytes but the model expects {want} ({} × {})",
                    buf.len(),
                    dtype_name(in_ports[i].dtype),
                    in_ports[i].element_count
                );
            }
        }
    }

    // Synthesize any unspecified input from `--fill`, so an arbitrary model runs
    // with a single command. Without `--fill`, missing inputs are a hard error
    // listing what the model expects (no silent zero-fill).
    let fill = args.fill.as_deref().map(parse_fill).transpose()?;
    let missing: Vec<usize> = slots
        .iter()
        .enumerate()
        .filter_map(|(i, s)| s.is_none().then_some(i))
        .collect();
    if !missing.is_empty() {
        let Some(fill) = fill else {
            anyhow::bail!(
                "missing input(s) {missing:?}; supply each via --input INDEX:HEX / \
                 --input-file INDEX:PATH, or synthesize them with --fill zeros"
            );
        };
        for &i in &missing {
            slots[i] = Some(synth_input(
                in_sizes[i],
                in_ports[i].element_count,
                in_ports[i].dtype,
                fill,
            )?);
        }
    }

    let owned: Vec<Vec<u8>> = slots.into_iter().map(|s| s.unwrap()).collect();
    let refs: Vec<&[u8]> = owned.iter().map(|v| v.as_slice()).collect();

    let execute_start = Instant::now();
    let outputs = runner.execute(&refs).context("inference failed")?;
    let execute = execute_start.elapsed();

    let out_ports = runner.output_port_info();
    for (i, out) in outputs.iter().enumerate() {
        let dt = out_ports.get(i).map(|p| p.dtype).unwrap_or(8);
        let elems = out_ports.get(i).map(|p| p.element_count).unwrap_or(0);
        println!(
            "output[{i}]: {} × {elems} ({} bytes)",
            dtype_name(dt),
            out.bytes.len()
        );
        if args.verbose {
            println!("  {}", preview(&out.bytes, dt));
        }
    }
    if args.stats {
        let mut stderr = std::io::stderr();
        ForwardStats {
            load,
            execute,
            total: total_start.elapsed(),
        }
        .display(&mut stderr)
        .ok();
    }
    Ok(())
}

/// Fill mode for synthesizing absent inputs (`--fill`).
#[derive(Clone, Copy)]
enum Fill {
    /// All-zero bytes — a valid encoding of 0 for every dtype.
    Zeros,
    /// A numeric constant, encoded per the port's dtype.
    Value(f64),
}

fn parse_fill(s: &str) -> Result<Fill> {
    match s.trim().to_ascii_lowercase().as_str() {
        "zeros" | "zero" => Ok(Fill::Zeros),
        "ones" | "one" => Ok(Fill::Value(1.0)),
        other => other
            .parse::<f64>()
            .map(Fill::Value)
            .map_err(|_| anyhow::anyhow!("--fill must be `zeros`, `ones`, or a number, got {s:?}")),
    }
}

/// Build an input buffer of `byte_size` bytes for `element_count` elements of
/// `dtype`, per the fill mode. `Zeros` is dtype-agnostic; a numeric value is
/// encoded element-wise for the common dtypes.
fn synth_input(byte_size: usize, element_count: usize, dtype: u8, fill: Fill) -> Result<Vec<u8>> {
    let v = match fill {
        Fill::Zeros => return Ok(vec![0u8; byte_size]),
        Fill::Value(v) => v,
    };
    let mut out = Vec::with_capacity(byte_size);
    for _ in 0..element_count {
        match dtype {
            1 => out.push(v as u8),                                   // U8
            2 => out.push(v as i8 as u8),                             // I8
            3 => out.extend_from_slice(&(v as u64).to_le_bytes()),    // U64
            4 => out.extend_from_slice(&(v as i32).to_le_bytes()),    // I32
            5 => out.extend_from_slice(&(v as i64).to_le_bytes()),    // I64
            8 => out.extend_from_slice(&(v as f32).to_le_bytes()),    // F32
            9 => out.extend_from_slice(&v.to_le_bytes()),             // F64
            _ => anyhow::bail!(
                "--fill {v} is not supported for dtype {} (use --fill zeros, or supply the input directly)",
                dtype_name(dtype)
            ),
        }
    }
    Ok(out)
}

/// Human-readable name for a backend dtype tag (`hologram_backend::cpu::dtype`).
fn dtype_name(tag: u8) -> &'static str {
    match tag {
        0 => "bool",
        1 => "u8",
        2 => "i8",
        3 => "u64",
        4 => "i32",
        5 => "i64",
        6 => "f16",
        7 => "bf16",
        8 => "f32",
        9 => "f64",
        10 => "i4",
        _ => "?",
    }
}

/// A short typed preview of an output buffer (first few elements), falling back
/// to hex for dtypes without a simple host decode.
fn preview(bytes: &[u8], dtype: u8) -> String {
    const MAX: usize = 16;
    match dtype {
        8 => fmt_vals(bytes, 4, MAX, |c| {
            f32::from_le_bytes(c.try_into().unwrap()) as f64
        }),
        9 => fmt_vals(bytes, 8, MAX, |c| f64::from_le_bytes(c.try_into().unwrap())),
        4 => fmt_vals(bytes, 4, MAX, |c| {
            i32::from_le_bytes(c.try_into().unwrap()) as f64
        }),
        5 => fmt_vals(bytes, 8, MAX, |c| {
            i64::from_le_bytes(c.try_into().unwrap()) as f64
        }),
        _ => {
            let shown = bytes.len().min(MAX * 4);
            let more = if bytes.len() > shown { " …" } else { "" };
            format!("{}{more}", hex(&bytes[..shown]))
        }
    }
}

fn fmt_vals(bytes: &[u8], width: usize, max: usize, decode: impl Fn(&[u8]) -> f64) -> String {
    let vals: Vec<String> = bytes
        .chunks_exact(width)
        .take(max)
        .map(|c| {
            let v = decode(c);
            if v == v.trunc() && v.abs() < 1e15 {
                format!("{v}")
            } else {
                format!("{v:.4}")
            }
        })
        .collect();
    let n = bytes.len() / width;
    let more = if n > max {
        format!(" … ({n} total)")
    } else {
        String::new()
    };
    format!("[{}{more}]", vals.join(", "))
}

/// `run --prompt …` — autoregressive text generation over a causal LM.
///
/// The model argument may be either a precompiled `.holo` (a fixed window) or a
/// model **source** — an `.onnx` file or a directory holding `model.onnx` +
/// `tokenizer.json`. A source drives the length-adaptive [`GrowableSession`]:
/// the window grows with the sequence up to the model's context length, so the
/// prompt and the continuation are bounded only by the model.
fn generate_cmd(args: RunArgs) -> Result<()> {
    let prompt = args
        .prompt
        .as_deref()
        .expect("generate_cmd requires --prompt");

    // Resolve the model argument and the tokenizer, then build the matching
    // session provider. `--tokenizer` always overrides; otherwise a `.holo`
    // self-describes (baked tokenizer) and a source uses the `tokenizer.json`
    // beside it.
    let (mut provider, tokenizer, auto_chat_template): (
        Box<dyn SessionProvider>,
        NativeTokenizer,
        Option<String>,
    ) = match resolve_model_arg(&args.file)? {
        ModelArg::Holo(path) => {
            let runner = HoloRunner::from_path(&path, None)
                .with_context(|| format!("loading model {path:?}"))?;
            let tokenizer = match args.tokenizer.as_ref() {
                Some(p) => NativeTokenizer::from_tokenizer_json(p)
                    .with_context(|| format!("loading tokenizer {p:?}"))?,
                None => load_archived_tokenizer(&runner)?,
            };
            let chat_template = load_holo_chat_template(&path, &runner)?;
            (
                Box::new(FixedSession::new(runner)),
                tokenizer,
                chat_template,
            )
        }
        ModelArg::Source {
            onnx,
            tokenizer_json,
            chat_template,
        } => {
            let tok_path = args.tokenizer.clone().unwrap_or(tokenizer_json);
            let tokenizer = NativeTokenizer::from_tokenizer_json(&tok_path).with_context(|| {
                format!(
                    "loading tokenizer {tok_path:?} (a model source needs a tokenizer.json \
                         beside it, or pass --tokenizer)"
                )
            })?;
            let compiler = crate::compiler::ModelCompiler {
                quant_strategy: parse_quant(args.quantize.as_deref())?,
                ..Default::default()
            };
            let prepared = compiler
                .prepare(crate::compiler::ModelSource::OnnxPath(onnx.clone()))
                .with_context(|| format!("importing model {onnx:?}"))?;
            let chat_template =
                load_model_chat_template(chat_template.as_deref(), args.chat_template.as_deref())?;
            (
                Box::new(GrowableSession::new(prepared)),
                tokenizer,
                chat_template,
            )
        }
    };

    let cfg = GenConfig {
        max_tokens: args.max_tokens,
        temperature: args.temperature,
        top_k: args.top_k,
        decode_top_k: args.decode_top_k,
        stop: args.stop.clone(),
        eos: args.eos,
        seed: args.seed,
    };

    let explicit_chat_template = load_explicit_chat_template(args.chat_template.as_deref())?;
    let templated = render_generation_prompt(
        &tokenizer,
        prompt,
        args.prompt_template.as_deref(),
        explicit_chat_template
            .as_deref()
            .or(auto_chat_template.as_deref()),
    )?;

    let mut stdout = std::io::stdout();
    // Echo the prompt so a streamed transcript reads coherently, then stream
    // the generated continuation token-by-token from inside generate_stream.
    print!("{prompt}");
    stdout.flush().ok();
    let outcome = generate::generate_stream_with_stats(
        provider.as_mut(),
        &tokenizer,
        &templated,
        &cfg,
        &mut stdout,
    )?;
    println!();
    if args.stats {
        let mut stderr = std::io::stderr();
        outcome.stats.display(&mut stderr).ok();
    }
    Ok(())
}

/// How the model argument to `run --prompt` was interpreted.
enum ModelArg {
    /// A precompiled `.holo` archive (fixed window).
    Holo(PathBuf),
    /// A model source — an importable `.onnx` plus the `tokenizer.json` to use.
    Source {
        onnx: PathBuf,
        tokenizer_json: PathBuf,
        chat_template: Option<PathBuf>,
    },
}

/// Classify the `run` model argument: a `.holo` file, an `.onnx` file, or a
/// directory laid out as `model.onnx` + `tokenizer.json` (the `download`
/// layout). A directory is searched for `model.onnx` then `onnx/model.onnx`.
fn resolve_model_arg(path: &std::path::Path) -> Result<ModelArg> {
    if path.is_dir() {
        let candidates = [
            path.join("model.onnx"),
            path.join("onnx").join("model.onnx"),
        ];
        let onnx = candidates
            .iter()
            .find(|p| p.exists())
            .cloned()
            .with_context(|| {
                format!("no model.onnx (or onnx/model.onnx) found in directory {path:?}")
            })?;
        let tokenizer_json = path.join("tokenizer.json");
        let chat_template = find_chat_template(path);
        return Ok(ModelArg::Source {
            onnx,
            tokenizer_json,
            chat_template,
        });
    }
    match path.extension().and_then(|e| e.to_str()) {
        Some("holo") => Ok(ModelArg::Holo(path.to_path_buf())),
        Some("onnx") => {
            let dir = path.parent().unwrap_or_else(|| std::path::Path::new("."));
            Ok(ModelArg::Source {
                onnx: path.to_path_buf(),
                tokenizer_json: dir.join("tokenizer.json"),
                chat_template: find_chat_template(dir),
            })
        }
        _ => bail!(
            "unrecognized model {path:?}; expected a compiled .holo, an .onnx file, or a \
             directory containing model.onnx + tokenizer.json"
        ),
    }
}

fn render_generation_prompt(
    tokenizer: &NativeTokenizer,
    prompt: &str,
    prompt_template: Option<&str>,
    chat_template: Option<&str>,
) -> Result<String> {
    if let Some(template) = prompt_template {
        return Ok(generate::apply_template(Some(template), prompt));
    }
    let eos_token = tokenizer
        .id_to_token(tokenizer.eos_token_id())
        .unwrap_or("</s>");
    if let Some(template) = chat_template {
        return generate::apply_chat_template(template, prompt, eos_token)
            .context("rendering chat template");
    }
    Ok(generate::apply_template(None, prompt))
}

fn load_explicit_chat_template(path: Option<&std::path::Path>) -> Result<Option<String>> {
    path.map(read_chat_template).transpose()
}

fn load_holo_chat_template(path: &std::path::Path, runner: &HoloRunner) -> Result<Option<String>> {
    if let Some(template) = load_explicit_chat_template(
        find_chat_template(path.parent().unwrap_or_else(|| std::path::Path::new("."))).as_deref(),
    )? {
        return Ok(Some(template));
    }
    Ok(runner.embedded_chat_template().map(ToOwned::to_owned))
}

fn load_model_chat_template(
    auto_path: Option<&std::path::Path>,
    explicit_path: Option<&std::path::Path>,
) -> Result<Option<String>> {
    if let Some(template) = load_explicit_chat_template(explicit_path)? {
        return Ok(Some(template));
    }
    auto_path.map(read_chat_template).transpose()
}

fn find_chat_template(dir: &std::path::Path) -> Option<PathBuf> {
    let jinja = dir.join("chat_template.jinja");
    if jinja.exists() {
        return Some(jinja);
    }
    let config = dir.join("tokenizer_config.json");
    config.exists().then_some(config)
}

fn read_chat_template(path: &std::path::Path) -> Result<String> {
    if path.file_name().and_then(|name| name.to_str()) == Some("chat_template.jinja") {
        return std::fs::read_to_string(path)
            .with_context(|| format!("reading chat template {path:?}"));
    }
    let raw = std::fs::read(path).with_context(|| format!("reading tokenizer config {path:?}"))?;
    let json: serde_json::Value = serde_json::from_slice(&raw)
        .with_context(|| format!("parsing tokenizer config {path:?}"))?;
    json.get("chat_template")
        .or_else(|| json.get("default_chat_template"))
        .and_then(|value| value.as_str())
        .map(ToOwned::to_owned)
        .with_context(|| format!("no chat_template found in {path:?}"))
}

/// Parse a quantization scheme name (matches the `compile` subcommand's flag).
fn parse_quant(s: Option<&str>) -> Result<hologram_ai_common::lower::QuantStrategy> {
    use hologram_ai_common::lower::QuantStrategy;
    Ok(match s.map(|s| s.to_ascii_lowercase()).as_deref() {
        None | Some("none") | Some("f32") => QuantStrategy::None,
        Some("int8") => QuantStrategy::Int8,
        Some("int4") => QuantStrategy::Int4,
        Some(other) => {
            bail!("unknown quantization scheme {other:?} (expected none/int8/int4)")
        }
    })
}

/// Load the tokenizer baked into the archive (canonical `tokenizer.json`
/// extension), verifying its content address against the stored κ-label so a
/// corrupted tokenizer is caught rather than silently producing wrong tokens.
fn load_archived_tokenizer(runner: &HoloRunner) -> Result<NativeTokenizer> {
    runner.embedded_tokenizer()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_fill_modes() {
        assert!(matches!(parse_fill("zeros").unwrap(), Fill::Zeros));
        assert!(matches!(parse_fill("ones").unwrap(), Fill::Value(v) if v == 1.0));
        assert!(matches!(parse_fill("2.5").unwrap(), Fill::Value(v) if v == 2.5));
        assert!(parse_fill("nonsense").is_err());
    }

    #[test]
    fn zeros_fill_is_dtype_agnostic() {
        // Any dtype: zeros is just zero bytes of the right length (here i4 packs
        // 6 elems into 3 bytes — the caller passes the exact byte_size).
        assert_eq!(synth_input(3, 6, 10, Fill::Zeros).unwrap(), vec![0u8; 3]);
        assert_eq!(synth_input(32, 8, 8, Fill::Zeros).unwrap(), vec![0u8; 32]);
    }

    #[test]
    fn numeric_fill_encodes_per_dtype() {
        // f32 ones: 4 elems × 1.0
        let f = synth_input(16, 4, 8, Fill::Value(1.0)).unwrap();
        let v: Vec<f32> = f
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(v, vec![1.0; 4]);
        // i64 value 5
        let g = synth_input(16, 2, 5, Fill::Value(5.0)).unwrap();
        let w: Vec<i64> = g
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(w, vec![5i64; 2]);
        // unsupported numeric dtype (bf16) errors rather than mis-encoding
        assert!(synth_input(4, 2, 7, Fill::Value(1.0)).is_err());
    }

    #[test]
    fn f32_preview_decodes_values() {
        let bytes: Vec<u8> = [1.0f32, 2.0, 3.5]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(preview(&bytes, 8), "[1, 2, 3.5000]");
    }

    #[test]
    fn dtype_names() {
        assert_eq!(dtype_name(8), "f32");
        assert_eq!(dtype_name(5), "i64");
        assert_eq!(dtype_name(10), "i4");
    }
}
