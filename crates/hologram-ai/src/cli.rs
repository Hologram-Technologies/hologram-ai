//! CLI entry point for hologram-ai.

use clap::Parser;
use hologram_ai::download;
use hologram_ai::session::{ModelCompiler, ModelSource, InferenceSession};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "hologram-ai", about = "AI model inference via hologram runtime")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand)]
enum Command {
    /// Run inference on a model file.
    Run {
        /// Path to a model file (.holo, .onnx, or .gguf).
        #[arg(short, long)]
        model: PathBuf,
        /// Input token IDs for AI models (comma-separated, for ONNX/GGUF).
        #[arg(short, long, value_delimiter = ',')]
        tokens: Vec<u32>,
        /// Raw input values as INDEX:HEX pairs (for .holo files).
        #[arg(long = "input", value_name = "INDEX:HEX")]
        inputs: Vec<String>,
    },
    /// Inspect a `.holo` archive or ONNX model file.
    Info {
        /// Path to a `.holo` or `.onnx` file.
        #[arg(short = 'f', long)]
        file: PathBuf,
        /// Levels of detail (for `.holo` files, may be repeated).
        #[arg(long, value_enum, default_values_t = [hologram::hologram_cli::commands::inspect::DetailLevel::Summary])]
        detail: Vec<hologram::hologram_cli::commands::inspect::DetailLevel>,
    },
    /// Compile a model to a `.holo` archive file.
    Compile {
        /// Path to the input model (ONNX or GGUF).
        #[arg(short, long)]
        model: PathBuf,
        /// Output directory for the compiled `.holo` archive.
        #[arg(short, long, value_name = "DIR")]
        output: PathBuf,
    },
    /// Download a model from HuggingFace Hub.
    Download(download::DownloadArgs),
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber_init();
    let cli = Cli::parse();

    match cli.command {
        Command::Run { model, tokens, inputs } => {
            match model.extension().and_then(|e: &std::ffi::OsStr| e.to_str()).unwrap_or("") {
                "holo" => run_holo(model, inputs)?,
                _ => {
                    let t0 = std::time::Instant::now();
                    let source = model_source_from_path(&model)?;
                    let compiled = ModelCompiler::default().compile(source)?;
                    eprintln!("[compile] {:.2?}", t0.elapsed());
                    let mut sess = InferenceSession::new(Arc::new(compiled));
                    let t1 = std::time::Instant::now();
                    let logits = sess.run(&tokens)?;
                    eprintln!("[inference] {:.2?}", t1.elapsed());
                    println!("logits shape: [{}]", logits.len());
                }
            }
        }
        Command::Info { file, detail } => {
            let ext = file.extension().and_then(|e: &std::ffi::OsStr| e.to_str()).unwrap_or("");
            match ext {
                "holo" => inspect_holo(file, detail)?,
                "onnx" => inspect_onnx(&file)?,
                other => anyhow::bail!(
                    "info supports .holo and .onnx files, got '.{other}'"
                ),
            }
        }
        Command::Compile { model, output } => {
            let source = model_source_from_path(&model)?;
            let compiled = ModelCompiler::default().compile(source)?;
            if output.exists() && !output.is_dir() {
                anyhow::bail!(
                    "'{}' exists and is not a directory. Remove it or choose a different --output path.",
                    output.display()
                );
            }
            std::fs::create_dir_all(&output)?;
            let stem = model.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
            let holo_path = output.join(format!("{stem}.holo"));
            compiled.save_archive(&holo_path)?;
            println!("wrote {}", holo_path.display());
        }
        Command::Download(args) => {
            download::run(args)?;
        }
    }

    Ok(())
}

// ── Run sub-commands ─────────────────────────────────────────────────────────

/// Run a compiled `.holo` archive.
///
/// If any input looks like text (not valid hex), we tokenize it and run
/// through the AI inference pipeline (which has custom op handlers).
/// Otherwise we delegate to the generic `hologram run`.
fn run_holo(file: PathBuf, inputs: Vec<String>) -> anyhow::Result<()> {
    if has_text_inputs(&inputs) {
        run_ai_inference(&file, &inputs)
    } else {
        run_holo_raw(file, inputs)
    }
}

/// Raw `.holo` execution — delegates to generic `hologram run`.
fn run_holo_raw(file: PathBuf, inputs: Vec<String>) -> anyhow::Result<()> {
    use hologram::hologram_cli::commands::run_cmd::{RunArgs, execute};
    let args = RunArgs { file, inputs };
    tokio::runtime::Builder::new_current_thread()
        .build()?
        .block_on(execute(args))
        .map_err(|e| anyhow::anyhow!("{e}"))
}

/// AI inference with tokenization.
///
/// Tokenizes text input, finds the source model (ONNX/GGUF) next to the
/// `.holo` file, compiles it (to get the custom op registry), and runs
/// inference via `InferenceSession`.
fn run_ai_inference(holo_path: &std::path::Path, inputs: &[String]) -> anyhow::Result<()> {
    use hologram_ai_tokenizer::{NativeTokenizer, Tokenizer};

    let dir = holo_path.parent().unwrap_or(std::path::Path::new("."));

    // Find text to tokenize (first non-hex input value)
    let text = inputs
        .iter()
        .find_map(|s| {
            let (_, value) = s.split_once(':')?;
            if is_valid_hex(value) { None } else { Some(value) }
        })
        .ok_or_else(|| anyhow::anyhow!("no text input found"))?;

    // Discover and load tokenizer
    let tok_path = dir.join("tokenizer.json");
    anyhow::ensure!(tok_path.exists(),
        "no tokenizer.json found in {}. Use --tokenizer to specify one.", dir.display());
    let tokenizer = NativeTokenizer::from_tokenizer_json(&tok_path)?;
    let token_ids = tokenizer.encode(text);

    eprintln!(
        "tokenized {} tokens (vocab={}, tokenizer={})",
        token_ids.len(),
        tokenizer.vocab_size(),
        tok_path.display()
    );

    // Find source model to compile (needed for custom op registry)
    let source_path = find_source_model(dir)?;
    eprintln!("compiling {} ...", source_path.display());

    let source = model_source_from_path(&source_path)?;
    let compiled = ModelCompiler::default().compile(source)?;
    let mut sess = InferenceSession::new(Arc::new(compiled));

    let logits = sess.run(&token_ids)?;

    let vocab_size = sess.model().metadata.vocab_size as usize;
    let seq_len = token_ids.len();
    eprintln!("logits: [{seq_len} x {vocab_size}]");

    // Print top-5 predictions for the last token position
    if vocab_size > 0 && logits.len() >= vocab_size {
        let last_logits = &logits[logits.len() - vocab_size..];
        let mut indexed: Vec<(usize, f32)> = last_logits.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        println!("top predictions:");
        for (id, score) in indexed.iter().take(5) {
            let token = tokenizer.id_to_token(*id as u32).unwrap_or("?");
            println!("  {id:>6} {token:>20}  {score:.4}");
        }
    }

    Ok(())
}

/// Find a source model (ONNX or GGUF) in the given directory.
fn find_source_model(dir: &std::path::Path) -> anyhow::Result<PathBuf> {
    for ext in &["onnx", "gguf"] {
        for entry in std::fs::read_dir(dir)? {
            let path = entry?.path();
            if path.extension().and_then(|e| e.to_str()) == Some(ext) {
                return Ok(path);
            }
        }
    }
    anyhow::bail!(
        "no source model (.onnx or .gguf) found in {}. \
         AI models need the source model for custom op handlers.",
        dir.display()
    )
}

/// Check if any input value looks like text (not valid hex).
fn has_text_inputs(inputs: &[String]) -> bool {
    inputs.iter().any(|s| {
        let Some((_, value)) = s.split_once(':') else { return false };
        !is_valid_hex(value)
    })
}

/// Check if a string is valid hex (even length, all hex digits).
fn is_valid_hex(s: &str) -> bool {
    s.len() % 2 == 0 && !s.is_empty() && s.bytes().all(|b| b.is_ascii_hexdigit())
}

// ── Info sub-commands ────────────────────────────────────────────────────────

/// Inspect a compiled `.holo` archive — delegates to `hologram inspect`.
fn inspect_holo(
    file: PathBuf,
    detail: Vec<hologram::hologram_cli::commands::inspect::DetailLevel>,
) -> anyhow::Result<()> {
    use hologram::hologram_cli::commands::inspect::{InspectArgs, execute};
    let args = InspectArgs { file, detail };
    tokio::runtime::Builder::new_current_thread()
        .build()?
        .block_on(execute(args))
        .map_err(|e| anyhow::anyhow!("{e}"))
}

/// Inspect an ONNX model file (import + print metadata without compilation).
fn inspect_onnx(path: &std::path::Path) -> anyhow::Result<()> {
    let ai_graph = hologram_ai_onnx::import_onnx_path(path, Default::default())?;

    println!("file:      {:?}", path);
    println!("format:    ONNX");
    println!("nodes:     {}", ai_graph.nodes.len());
    println!("params:    {}", ai_graph.params.len());
    println!("inputs:    {}", ai_graph.inputs.len());
    println!("outputs:   {}", ai_graph.outputs.len());

    // Print model metadata if available.
    use hologram_ai_common::MetaValue;
    for (key, val) in &ai_graph.metadata {
        let s = match val {
            MetaValue::Str(s) => s.clone(),
            MetaValue::Int(i) => i.to_string(),
            MetaValue::Float(f) => format!("{f:.4}"),
            MetaValue::Bool(b) => b.to_string(),
            MetaValue::Ints(v) => format!("{v:?}"),
        };
        println!("{key:<11}{s}");
    }

    Ok(())
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn model_source_from_path(path: &std::path::Path) -> anyhow::Result<ModelSource> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    match ext {
        "onnx" => Ok(ModelSource::OnnxPath(path.to_owned())),
        "gguf" => Ok(ModelSource::GgufPath(path.to_owned())),
        other  => anyhow::bail!("unsupported model extension: '.{other}'"),
    }
}

fn tracing_subscriber_init() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();
}
