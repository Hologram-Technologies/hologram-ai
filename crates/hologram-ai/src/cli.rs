//! CLI entry point for hologram-ai — the UOR-native AI model compiler + runner.
//!
//! Three commands: `compile` (model → `.holo`), `run` (execute a `.holo`), and
//! `download` (fetch a model). The compiler lowers the model to a canonical
//! hologram graph and hands it to `hologram_compiler::compile`; the runner
//! loads the archive into an `InferenceSession` (architecture §5, §7).

use anyhow::Context as _;
use clap::Parser;
use hologram_ai::commands::run_cmd::{execute as run_execute, RunArgs};
use hologram_ai::compiler::{ModelCompiler, ModelSource};
#[cfg(feature = "native")]
use hologram_ai::download::{self, DownloadArgs};
use hologram_ai_common::lower::QuantStrategy;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "hologram-ai",
    about = "UOR-native AI model compiler + runner for the hologram runtime"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand)]
enum Command {
    /// Compile a model (ONNX) into a `.holo` archive.
    Compile {
        /// Path to the input ONNX model file.
        #[arg(short, long, value_name = "FILE")]
        model: PathBuf,
        /// Output directory for the compiled `.holo` archive.
        #[arg(short, long, value_name = "DIR", default_value = ".")]
        output: PathBuf,
        /// Archive filename stem (the `.holo` extension is appended).
        /// Defaults to the model file stem.
        #[arg(long, value_name = "STEM")]
        name: Option<String>,
        /// Fixed sequence length for compilation (default: model's context_length).
        #[arg(long, value_name = "N")]
        seq_len: Option<u64>,
        /// Weight quantization scheme: 'none'/'f32', 'int8', or 'int4'.
        #[arg(long, value_name = "SCHEME")]
        quantize: Option<String>,
        /// Scale spatial dims (H, W) of 4-D inputs by this factor for lower
        /// activation memory (vision/diffusion models).
        #[arg(long, value_name = "N")]
        spatial_scale: Option<u32>,
    },
    /// Execute a compiled `.holo` archive.
    Run(RunArgs),
    /// Download a model.
    #[cfg(feature = "native")]
    Download(DownloadArgs),
}

fn main() -> anyhow::Result<()> {
    #[cfg(feature = "native")]
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    match Cli::parse().command {
        Command::Compile {
            model,
            output,
            name,
            seq_len,
            quantize,
            spatial_scale,
        } => compile(model, output, name, seq_len, quantize, spatial_scale),
        Command::Run(args) => run_execute(args),
        #[cfg(feature = "native")]
        Command::Download(args) => download::run(args),
    }
}

fn compile(
    model: PathBuf,
    output: PathBuf,
    name: Option<String>,
    seq_len: Option<u64>,
    quantize: Option<String>,
    spatial_scale: Option<u32>,
) -> anyhow::Result<()> {
    let quant_strategy = parse_quant(quantize.as_deref())?;
    let compiler = ModelCompiler {
        seq_len_override: seq_len,
        quant_strategy,
        spatial_scale,
        ..ModelCompiler::default()
    };

    let stem = name.unwrap_or_else(|| {
        model
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "model".to_string())
    });
    let out_path = output.join(format!("{stem}.holo"));

    let archive = compiler
        .compile(ModelSource::OnnxPath(model.clone()))
        .with_context(|| format!("compiling {model:?}"))?;
    archive.save(&out_path)?;

    println!(
        "Compiled {model:?} → {out_path:?} ({} nodes, {} archive bytes)",
        archive.stats.node_count,
        archive.bytes.len()
    );
    Ok(())
}

fn parse_quant(s: Option<&str>) -> anyhow::Result<QuantStrategy> {
    Ok(match s.map(|s| s.to_ascii_lowercase()).as_deref() {
        None | Some("none") | Some("f32") => QuantStrategy::None,
        Some("int8") => QuantStrategy::Int8,
        // int4 is a per-channel symmetric tier (packed nibbles): the inline
        // `quantize_weights` pass emits it and the substrate's `matmul_i4_pc_omajor`
        // decodes it (proven in `int8_accuracy.rs` / `omajor_i4_substrate_contract.rs`).
        // It is materially lossier than int8 (≈16% vs ≈1% GEMV error) — a size-first
        // choice — but it runs, so it is offered.
        Some("int4") => QuantStrategy::Int4,
        Some(other) => {
            anyhow::bail!("unknown quantization scheme {other:?} (expected none/f32/int8/int4)")
        }
    })
}
