//! CLI entry point for hologram-ai.

use clap::Parser;
use hologram_ai::download;
use hologram_ai::compiler::{ModelCompiler, ModelSource};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "hologram-ai", about = "AI model compiler for hologram runtime")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand)]
enum Command {
    /// Run a model file (facade to `hologram run`).
    ///
    /// For `.holo` files, delegates directly. For `.onnx`/`.gguf`, compiles
    /// to a temporary `.holo` first, then delegates.
    Run {
        /// Path to a model file (.holo, .onnx, or .gguf).
        #[arg(short, long)]
        model: PathBuf,
        /// Raw input values as INDEX:HEX pairs.
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
        Command::Run { model, inputs } => {
            let holo_path = match model.extension().and_then(|e| e.to_str()).unwrap_or("") {
                "holo" => model,
                _ => {
                    // Compile to a temp .holo, then run that.
                    let t0 = std::time::Instant::now();
                    let source = model_source_from_path(&model)?;
                    let compiled = ModelCompiler::default().compile(source)?;
                    let tmp_dir = std::env::temp_dir().join("hologram-ai");
                    std::fs::create_dir_all(&tmp_dir)?;
                    let stem = model.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
                    let tmp_holo = tmp_dir.join(format!("{stem}.holo"));
                    compiled.save_archive(&tmp_holo)?;
                    eprintln!("[compile] {:.2?} → {}", t0.elapsed(), tmp_holo.display());
                    tmp_holo
                }
            };
            run_holo(holo_path, inputs)?;
        }
        Command::Info { file, detail } => {
            let ext = file.extension().and_then(|e| e.to_str()).unwrap_or("");
            match ext {
                "holo" => inspect_holo(file, detail)?,
                "onnx" => inspect_onnx(&file)?,
                "gguf" => inspect_gguf(&file)?,
                other => anyhow::bail!(
                    "info supports .holo, .onnx, and .gguf files, got '.{other}'"
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
            compiled.save(&holo_path)?;
            println!("wrote {} ({} nodes, {} weight bytes, {} warnings)",
                holo_path.display(),
                compiled.stats.node_count,
                compiled.stats.total_weight_bytes,
                compiled.stats.import_warnings,
            );
        }
        Command::Download(args) => {
            download::run(args)?;
        }
    }

    Ok(())
}

// ── Run ──────────────────────────────────────────────────────────────────────

/// Run a compiled `.holo` archive — delegates to `hologram run`.
fn run_holo(file: PathBuf, inputs: Vec<String>) -> anyhow::Result<()> {
    use hologram::hologram_cli::commands::run_cmd::{RunArgs, execute};
    let args = RunArgs { file, inputs };
    tokio::runtime::Builder::new_current_thread()
        .build()?
        .block_on(execute(args))
        .map_err(|e| anyhow::anyhow!("{e}"))
}

// ── Info ─────────────────────────────────────────────────────────────────────

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

/// Inspect a GGUF model file (parse header + print metadata).
fn inspect_gguf(path: &std::path::Path) -> anyhow::Result<()> {
    let data = std::fs::read(path)?;
    let gguf = hologram_ai_gguf::parser::parse_gguf(&data)?;
    let arch = hologram_ai_gguf::metadata::ArchParams::from_gguf(&gguf, None)?;

    println!("file:        {:?}", path);
    println!("format:      GGUF v{}", gguf.version);
    println!("arch:        {}", arch.arch);
    println!("tensors:     {}", gguf.tensors.len());
    println!("context:     {}", arch.context_length);
    println!("embedding:   {}", arch.embedding_length);
    println!("layers:      {}", arch.block_count);
    println!("heads:       {} (kv: {})", arch.head_count, arch.head_count_kv);
    println!("ffn:         {}", arch.feed_forward_length);
    println!("vocab:       {}", arch.vocab_size);
    println!("rope_base:   {:.1}", arch.rope_freq_base);
    println!("rms_eps:     {:.1e}", arch.layer_norm_rms_epsilon);

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
