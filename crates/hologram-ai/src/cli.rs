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
                    let source = model_source_from_path(&model)?;
                    let compiled = ModelCompiler::default().compile(source)?;
                    let mut sess = InferenceSession::new(Arc::new(compiled));
                    let logits = sess.run(&tokens)?;
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

/// Run a compiled `.holo` archive — delegates to `hologram run`.
///
/// Input values that aren't valid hex are treated as UTF-8 text and
/// encoded to hex automatically (e.g. `0:"hello world"` → `0:68656c6c6f20776f726c64`).
fn run_holo(file: PathBuf, inputs: Vec<String>) -> anyhow::Result<()> {
    use hologram::hologram_cli::commands::run_cmd::{RunArgs, execute};
    let inputs = inputs.into_iter().map(encode_input_if_text).collect();
    let args = RunArgs { file, inputs };
    tokio::runtime::Builder::new_current_thread()
        .build()?
        .block_on(execute(args))
        .map_err(|e| anyhow::anyhow!("{e}"))
}

/// If the value portion of an `INDEX:VALUE` string isn't valid hex,
/// re-encode it as UTF-8 hex bytes.
fn encode_input_if_text(s: String) -> String {
    let Some((idx, value)) = s.split_once(':') else { return s };
    if is_valid_hex(value) {
        return s;
    }
    let hex: String = value.bytes().map(|b| format!("{b:02x}")).collect();
    format!("{idx}:{hex}")
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
