//! hologram-onnx CLI tool
//!
//! Command-line interface for compiling ONNX models to hologram's .holo format,
//! downloading models from Hugging Face, and running pipelines with unified configs.

use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod compile;
mod download;
mod info;
mod run;
mod translator;
mod validate;

use compile::compile_command;
use download::download_command;
use info::info_command;
use run::run_command;
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
        /// Input ONNX model file (required if --config not specified)
        #[arg(required_unless_present = "config")]
        input: Option<PathBuf>,

        /// Output path (without extension, .holo and .weights will be added)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Load compiler settings from a unified config file
        #[arg(short, long)]
        config: Option<PathBuf>,

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

    /// Run an ONNX pipeline from a unified config file
    Run {
        /// Path to the unified config file (TOML)
        #[arg(short, long)]
        config: PathBuf,

        /// Runtime inputs as key=value pairs
        #[arg(short, long = "input", value_name = "NAME=VALUE")]
        inputs: Vec<String>,

        /// Output directory for generated files
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Force recompilation of all models
        #[arg(long)]
        recompile: bool,
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
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(log_level)),
        )
        .init();

    match cli.command {
        Commands::Compile {
            input,
            output,
            config,
            partition,
            partition_size,
            memory_budget,
            weight_threshold,
        } => {
            // If a config file is provided, use it for compiler settings
            if let Some(config_path) = config {
                compile_with_config(
                    &config_path,
                    input.as_deref(),
                    output.as_deref(),
                    partition,
                    partition_size,
                    memory_budget,
                    weight_threshold,
                )
            } else {
                // Traditional compile with explicit input
                let input = input.ok_or_else(|| {
                    anyhow::anyhow!("Input ONNX file required when --config not specified")
                })?;
                let output = output.unwrap_or_else(|| {
                    input.with_extension("")
                });
                compile_command(
                    &input,
                    &output,
                    partition,
                    partition_size,
                    memory_budget,
                    weight_threshold,
                )
            }
        }

        Commands::Run {
            config,
            inputs,
            output,
            recompile,
        } => run_command(&config, &inputs, output.as_deref(), recompile),

        Commands::Download {
            model_id,
            output,
            revision,
        } => download_command(&model_id, &output, revision.as_deref()),

        Commands::Info { model, detailed } => info_command(&model, detailed),

        Commands::Validate { model, check_ops } => validate_command(&model, check_ops),
    }
}

/// Compile models using settings from a unified config file.
fn compile_with_config(
    config_path: &std::path::Path,
    input_override: Option<&std::path::Path>,
    output_override: Option<&std::path::Path>,
    partition_override: bool,
    partition_size_override: usize,
    memory_budget_override: Option<usize>,
    weight_threshold_override: usize,
) -> anyhow::Result<()> {
    use hologram_onnx_config::UnifiedConfig;
    use hologram_onnx_core::OnnxConfig;
    use tracing::info;

    info!("Loading config from: {}", config_path.display());

    let config = UnifiedConfig::from_file(config_path)
        .map_err(|e| anyhow::anyhow!("Failed to load config: {}", e))?;

    // Get compiler settings from config
    let compiler_config: OnnxConfig = (&config.compiler).into();

    // Apply CLI overrides (CLI takes precedence)
    let partition = partition_override || compiler_config.enable_partitioning;
    let partition_size = if partition_size_override != 500 {
        partition_size_override
    } else {
        compiler_config.partition_size
    };
    let memory_budget = memory_budget_override.or(compiler_config.memory_budget);
    let weight_threshold = if weight_threshold_override != 4096 {
        weight_threshold_override
    } else {
        compiler_config.weight_threshold
    };

    // Get config directory for resolving relative paths
    let config_dir = config_path.parent().unwrap_or_else(|| std::path::Path::new("."));

    // If input is provided, compile just that model
    if let Some(input) = input_override {
        let output = output_override
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| input.with_extension(""));

        return compile_command(
            input,
            &output,
            partition,
            partition_size,
            memory_budget,
            weight_threshold,
        );
    }

    // Otherwise, compile all models in the config
    if config.models.is_empty() {
        anyhow::bail!("No models specified in config");
    }

    for (name, model_def) in &config.models {
        let onnx_path = {
            let path = std::path::Path::new(model_def.path());
            if path.is_absolute() {
                path.to_path_buf()
            } else {
                config_dir.join(path)
            }
        };

        // Skip if model file doesn't exist (might need to be downloaded first)
        if !onnx_path.exists() {
            tracing::warn!(
                "Model file not found, skipping: {} ({})",
                name,
                onnx_path.display()
            );
            continue;
        }

        let output = output_override
            .map(|p| p.join(name))
            .unwrap_or_else(|| onnx_path.with_extension(""));

        info!("Compiling model '{}': {}", name, onnx_path.display());

        compile_command(
            &onnx_path,
            &output,
            partition,
            partition_size,
            memory_budget,
            weight_threshold,
        )?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_parse_compile() {
        let args = vec![
            "hologram-onnx",
            "compile",
            "input.onnx",
            "-o",
            "output",
        ];
        let cli = Cli::try_parse_from(args).unwrap();
        match cli.command {
            Commands::Compile { input, output, .. } => {
                assert_eq!(input, Some(PathBuf::from("input.onnx")));
                assert_eq!(output, Some(PathBuf::from("output")));
            }
            _ => panic!("Expected Compile command"),
        }
    }

    #[test]
    fn test_cli_parse_compile_with_config() {
        let args = vec![
            "hologram-onnx",
            "compile",
            "--config",
            "pipeline.toml",
        ];
        let cli = Cli::try_parse_from(args).unwrap();
        match cli.command {
            Commands::Compile { config, input, .. } => {
                assert_eq!(config, Some(PathBuf::from("pipeline.toml")));
                assert_eq!(input, None);
            }
            _ => panic!("Expected Compile command"),
        }
    }

    #[test]
    fn test_cli_parse_run() {
        let args = vec![
            "hologram-onnx",
            "run",
            "--config",
            "pipeline.toml",
            "-i",
            "prompt=hello",
            "-i",
            "steps=50",
        ];
        let cli = Cli::try_parse_from(args).unwrap();
        match cli.command {
            Commands::Run { config, inputs, .. } => {
                assert_eq!(config, PathBuf::from("pipeline.toml"));
                assert_eq!(inputs.len(), 2);
                assert!(inputs.contains(&"prompt=hello".to_string()));
                assert!(inputs.contains(&"steps=50".to_string()));
            }
            _ => panic!("Expected Run command"),
        }
    }

    #[test]
    fn test_cli_parse_run_with_output() {
        let args = vec![
            "hologram-onnx",
            "run",
            "--config",
            "pipeline.toml",
            "-o",
            "output_dir",
            "--recompile",
        ];
        let cli = Cli::try_parse_from(args).unwrap();
        match cli.command {
            Commands::Run {
                config,
                output,
                recompile,
                ..
            } => {
                assert_eq!(config, PathBuf::from("pipeline.toml"));
                assert_eq!(output, Some(PathBuf::from("output_dir")));
                assert!(recompile);
            }
            _ => panic!("Expected Run command"),
        }
    }
}
