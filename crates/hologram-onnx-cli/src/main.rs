//! hologram-onnx CLI tool
//!
//! Command-line interface for compiling ONNX models to hologram's .holo format
//! and downloading models from Hugging Face.

use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod compile;
mod download;
mod info;
mod translator;
mod validate;

use compile::compile_command;
use download::download_command;
use info::info_command;
use validate::validate_command;

#[derive(Parser)]
#[command(name = "hologram-onnx")]
#[command(about = "Production ONNX runtime for hologram", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable verbose logging
    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Compile ONNX model to .holo format
    Compile {
        /// Input ONNX model file
        input: PathBuf,

        /// Output path (without extension, .holo and .weights will be added)
        #[arg(short, long)]
        output: PathBuf,

        /// Enable graph partitioning for large models
        #[arg(long)]
        partition: bool,

        /// Partition size (number of nodes per partition)
        #[arg(long, default_value = "500")]
        partition_size: usize,

        /// Memory budget in MB
        #[arg(long)]
        memory_budget: Option<usize>,

        /// Weight threshold for external storage (bytes)
        #[arg(long, default_value = "4096")]
        weight_threshold: usize,
    },

    /// Download ONNX model from Hugging Face
    Download {
        /// Model ID on Hugging Face (e.g., "stable-diffusion-v1-5")
        model_id: String,

        /// Output directory
        #[arg(short, long)]
        output: PathBuf,

        /// Git revision/branch
        #[arg(short, long)]
        revision: Option<String>,
    },

    /// Display ONNX model information
    Info {
        /// ONNX model file
        model: PathBuf,

        /// Show detailed operation list
        #[arg(long)]
        detailed: bool,
    },

    /// Validate ONNX model
    Validate {
        /// ONNX model file
        model: PathBuf,

        /// Check for unsupported operations
        #[arg(long)]
        check_ops: bool,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    let log_level = if cli.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(log_level))
        )
        .init();

    match cli.command {
        Commands::Compile {
            input,
            output,
            partition,
            partition_size,
            memory_budget,
            weight_threshold,
        } => compile_command(
            &input,
            &output,
            partition,
            partition_size,
            memory_budget,
            weight_threshold,
        ),

        Commands::Download {
            model_id,
            output,
            revision,
        } => download_command(&model_id, &output, revision.as_deref()),

        Commands::Info { model, detailed } => info_command(&model, detailed),

        Commands::Validate { model, check_ops } => validate_command(&model, check_ops),
    }
}
