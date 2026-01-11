//! hologram-onnx CLI tool
//!
//! Command-line interface for compiling ONNX models to hologram's .holo format,
//! downloading models from Hugging Face, and running pipelines with unified configs.

use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod bundle;
mod compile;
mod compile_pipeline;
mod compile_tokenizer;
mod download;
mod info;
mod run;
mod translator;
mod validate;

use bundle::{
    bundle_command, bundle_from_config, bundle_pipeline_command, bundle_pipeline_from_config,
    extract_command, list_pipeline_command,
};
use compile::compile_command;
use compile_pipeline::compile_pipeline_command;
pub use compile_tokenizer::{compile_tokenizer_command, compile_tokenizer_from_config};
pub use download::download_command;
use info::info_command;
use run::{run_command, run_pipeline_bundle_command};
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

        /// Input shapes for dynamic dimensions (e.g., "input_name=1,4,8,8")
        #[arg(long = "input-shape", value_name = "NAME=DIMS")]
        input_shapes: Vec<String>,

        /// Create a unified bundle with embedded weights (default: true)
        ///
        /// When enabled, produces a single .holo file with page-aligned weights
        /// section that can be memory-mapped for efficient loading of large models.
        /// Disable with --no-bundle for separate .holo + .weights files.
        #[arg(long, default_value = "true", action = clap::ArgAction::Set)]
        bundle: bool,

        /// Embed files in the bundle (vocabulary, tokenizer config, etc.)
        ///
        /// Format: type:path or type:path:custom_id
        ///
        /// Types:
        ///   vocabulary       - Line-based vocabulary (vocab.txt)
        ///   vocabulary_json  - JSON vocabulary (vocab.json)
        ///   tokenizer_config - Tokenizer config (tokenizer_config.json)
        ///   model_config     - Model config (config.json)
        ///   special_tokens   - Special tokens map (special_tokens_map.json)
        ///   preprocessor     - Preprocessor config (preprocessor_config.json)
        ///   sentencepiece    - SentencePiece model (*.model)
        ///   generation       - Generation config (generation_config.json)
        ///   raw:content_type - Raw file with custom content type
        ///
        /// Examples:
        ///   --embed vocabulary:vocab.txt
        ///   --embed tokenizer_config:tokenizer_config.json
        ///   --embed raw:application/octet-stream:data.bin:my_data
        #[arg(long = "embed", value_name = "TYPE:PATH[:ID]")]
        embed_files: Vec<String>,

        /// Compile transformer model as layer-wise HOLM pipeline
        ///
        /// Detects transformer layer structure and compiles each layer as a separate
        /// HOLB model, packaged into a HOLM pipeline bundle. This enables layer-by-layer
        /// execution with madvise-based prefetching for large models on memory-constrained
        /// systems.
        ///
        /// Memory reduction for 70B models: ~130GB → ~2GB peak memory
        #[arg(long)]
        layer_wise: bool,
    },

    /// Compile tokenizer to .holo format for hologram execution
    ///
    /// Converts tokenizer.json files to compiled .holo format that can be executed
    /// via the hologram backend with SIMD-accelerated vocabulary lookups.
    ///
    /// # Examples
    ///
    /// From tokenizer.json file:
    ///   hologram-onnx compile-tokenizer models/t5-small/tokenizer.json \
    ///     -o models/t5-small/tokenizer.holo \
    ///     --type sentencepiece \
    ///     --max-length 512
    ///
    /// From config file:
    ///   hologram-onnx compile-tokenizer --config tokenizer.toml
    CompileTokenizer {
        /// Input tokenizer.json file (required if --config not specified)
        #[arg(required_unless_present = "config")]
        vocab_path: Option<PathBuf>,

        /// Output .holo file path
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Load tokenizer settings from a config file
        #[arg(short, long)]
        config: Option<PathBuf>,

        /// Tokenizer type (sentencepiece)
        #[arg(long, default_value = "sentencepiece")]
        tokenizer_type: String,

        /// Maximum sequence length for padding
        #[arg(long, default_value = "512")]
        max_length: usize,

        /// Padding token ID
        #[arg(long, default_value = "0")]
        pad_token_id: u32,

        /// End-of-sequence token ID
        #[arg(long, default_value = "1")]
        eos_token_id: u32,

        /// Unknown token ID
        #[arg(long, default_value = "2")]
        unk_token_id: u32,
    },

    /// Run an ONNX pipeline from a unified config file or directly execute a .holo model
    ///
    /// Models must be pre-compiled with `hologram-onnx compile` before running.
    ///
    /// # Config-based execution (for Stable Diffusion pipelines):
    /// hologram-onnx run --config pipeline.toml -i prompt="a dog" -i steps=20
    ///
    /// # Direct execution (for T5 models):
    /// hologram-onnx run encoder.holo --prompt "Tell me a joke"
    Run {
        /// Direct .holo model file (for simple inference)
        #[arg(required_unless_present = "config")]
        model: Option<PathBuf>,

        /// Path to the unified config file (TOML) for multi-model pipelines
        #[arg(short, long)]
        config: Option<PathBuf>,

        /// Text prompt for T5/text models (simpler than --input)
        #[arg(short, long)]
        prompt: Option<String>,

        /// Path to tokenizer.json for T5 models
        #[arg(long, default_value = "tokenizer.json")]
        tokenizer: PathBuf,

        /// Maximum sequence length for tokenization
        #[arg(long, default_value = "512")]
        max_length: usize,

        /// Runtime inputs as key=value pairs (for config-based pipelines)
        #[arg(short, long = "input", value_name = "NAME=VALUE")]
        inputs: Vec<String>,

        /// Output directory for generated files
        #[arg(short, long)]
        output: Option<PathBuf>,
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

    /// Bundle multiple .holo files into a single distributable file
    Bundle {
        /// Input .holo files to bundle
        #[arg(required_unless_present = "config")]
        inputs: Vec<PathBuf>,

        /// Output bundle file path
        #[arg(short, long)]
        output: PathBuf,

        /// Load models from a unified config file
        #[arg(short, long)]
        config: Option<PathBuf>,

        /// Custom names for models (parallel to inputs)
        #[arg(long = "name", value_name = "NAME")]
        names: Vec<String>,
    },

    /// Extract models from a bundle
    Extract {
        /// Bundle file to extract
        bundle: PathBuf,

        /// Output directory for extracted models
        #[arg(short, long)]
        output: PathBuf,
    },

    /// List models in a bundle
    List {
        /// Bundle file to list
        bundle: PathBuf,
    },

    /// Run a T5 pipeline bundle for text generation
    ///
    /// Executes a pre-compiled HOLM pipeline bundle containing encoder, decoder,
    /// and tokenizer for encoder-decoder text generation.
    ///
    /// # Examples
    ///
    /// Generate text:
    ///   hologram-onnx run-pipeline t5-pipeline.holo --prompt "Tell me a joke"
    ///
    /// With custom max tokens:
    ///   hologram-onnx run-pipeline t5-pipeline.holo --prompt "Translate to French: Hello" --max-tokens 100
    RunPipeline {
        /// Pipeline bundle file (HOLM format containing encoder, decoder, tokenizer)
        pipeline: PathBuf,

        /// Text prompt for generation
        #[arg(short, long)]
        prompt: String,

        /// Maximum number of new tokens to generate
        #[arg(long, default_value = "50")]
        max_tokens: usize,
    },

    /// Create a pipeline bundle (HOLM format) from HOLB model bundles
    ///
    /// Pipeline bundles package multiple models with their embedded weights
    /// into a single file for easy deployment. Each model's weights section
    /// remains page-aligned for efficient memory-mapping.
    ///
    /// # Examples
    ///
    /// Bundle encoder and decoder:
    ///   hologram-onnx bundle-pipeline \
    ///     --encoder models/encoder_bundle.holo \
    ///     --decoder models/decoder_bundle.holo \
    ///     -o models/pipeline.holo
    ///
    /// Bundle from config:
    ///   hologram-onnx bundle-pipeline --config pipeline.toml -o pipeline.holo
    BundlePipeline {
        /// Encoder model HOLB bundle
        #[arg(long)]
        encoder: Option<PathBuf>,

        /// Decoder model HOLB bundle
        #[arg(long)]
        decoder: Option<PathBuf>,

        /// Tokenizer model HOLB bundle
        #[arg(long)]
        tokenizer: Option<PathBuf>,

        /// Additional named models (format: name=path)
        #[arg(long = "model", value_name = "NAME=PATH")]
        models: Vec<String>,

        /// Load models from a unified config file
        #[arg(short, long)]
        config: Option<PathBuf>,

        /// Output path for the pipeline bundle
        #[arg(short, long)]
        output: PathBuf,
    },

    /// Compile ONNX models and tokenizer into a single pipeline bundle
    ///
    /// This command compiles ONNX models directly to a HOLM pipeline bundle,
    /// combining compilation and bundling into a single step. Intermediate
    /// HOLB files are created in a temporary directory and cleaned up after.
    ///
    /// # Examples
    ///
    /// Compile T5 encoder and decoder:
    ///   hologram-onnx compile-pipeline \
    ///     --encoder models/encoder.onnx \
    ///     --decoder models/decoder.onnx \
    ///     --tokenizer models/tokenizer.json \
    ///     -o models/t5-pipeline.holo
    ///
    /// With compiler options:
    ///   hologram-onnx compile-pipeline \
    ///     --encoder models/encoder.onnx \
    ///     --decoder models/decoder.onnx \
    ///     --weight-threshold 8192 \
    ///     --partition \
    ///     -o pipeline.holo
    ///
    /// Keep intermediate files for debugging:
    ///   hologram-onnx compile-pipeline \
    ///     --encoder models/encoder.onnx \
    ///     --decoder models/decoder.onnx \
    ///     --keep-intermediates \
    ///     -o pipeline.holo
    ///
    /// From config file:
    ///   hologram-onnx compile-pipeline --config pipeline.toml -o pipeline.holo
    CompilePipeline {
        /// Encoder ONNX model file
        #[arg(long)]
        encoder: Option<PathBuf>,

        /// Decoder ONNX model file
        #[arg(long)]
        decoder: Option<PathBuf>,

        /// Tokenizer JSON file (tokenizer.json or spiece.model)
        #[arg(long)]
        tokenizer: Option<PathBuf>,

        /// Additional models (format: name=path.onnx or name=tokenizer:path.json)
        #[arg(long = "model", value_name = "NAME=PATH")]
        models: Vec<String>,

        /// Load models and settings from a config file
        #[arg(short, long)]
        config: Option<PathBuf>,

        /// Output pipeline bundle path
        #[arg(short, long)]
        output: PathBuf,

        /// Weight threshold for external storage (bytes)
        #[arg(long, default_value = "4096")]
        weight_threshold: usize,

        /// Enable graph partitioning for large models
        #[arg(long)]
        partition: bool,

        /// Partition size (number of nodes per partition)
        #[arg(long, default_value = "500")]
        partition_size: usize,

        /// Memory budget in MB
        #[arg(long)]
        memory_budget: Option<usize>,

        /// Keep intermediate HOLB files (for debugging)
        #[arg(long)]
        keep_intermediates: bool,
    },
}

/// Run the hologram-onnx CLI with standard argument parsing.
pub fn run() -> anyhow::Result<()> {
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
            input_shapes,
            bundle,
            embed_files,
            layer_wise,
        } => {
            // Parse input shapes from "name=d1,d2,d3" format
            let parsed_shapes: std::collections::HashMap<String, Vec<usize>> = input_shapes
                .iter()
                .filter_map(|s| {
                    let parts: Vec<&str> = s.splitn(2, '=').collect();
                    if parts.len() == 2 {
                        let name = parts[0].to_string();
                        let dims: Result<Vec<usize>, _> = parts[1]
                            .split(',')
                            .map(|d| d.trim().parse::<usize>())
                            .collect();
                        dims.ok().map(|d| (name, d))
                    } else {
                        None
                    }
                })
                .collect();

            // Parse embedded files
            let parsed_embed_files = parse_embed_files(&embed_files)?;

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
                    bundle,
                    &parsed_embed_files,
                    layer_wise,
                )
            } else {
                // Traditional compile with explicit input
                let input = input.ok_or_else(|| {
                    anyhow::anyhow!("Input ONNX file required when --config not specified")
                })?;
                let output = output.unwrap_or_else(|| input.with_extension(""));
                compile_command(
                    &input,
                    &output,
                    partition,
                    partition_size,
                    memory_budget,
                    weight_threshold,
                    true,
                    true,
                    true, // enable_resize_upscaling
                    &parsed_shapes,
                    bundle,
                    &parsed_embed_files,
                    layer_wise,
                )
            }
        }

        Commands::CompileTokenizer {
            vocab_path,
            output,
            config,
            tokenizer_type,
            max_length,
            pad_token_id,
            eos_token_id,
            unk_token_id,
        } => {
            if let Some(config_path) = config {
                // Config-based compilation
                compile_tokenizer_from_config(&config_path, output.as_deref())
            } else {
                // Direct compilation with explicit parameters
                let vocab = vocab_path.ok_or_else(|| {
                    anyhow::anyhow!("Tokenizer vocab_path required when --config not specified")
                })?;
                let output = output.unwrap_or_else(|| vocab.with_extension("holo"));
                compile_tokenizer_command(
                    &vocab,
                    &output,
                    &tokenizer_type,
                    max_length,
                    pad_token_id,
                    eos_token_id,
                    unk_token_id,
                )
            }
        }

        Commands::Run {
            model,
            config,
            prompt,
            tokenizer,
            max_length,
            inputs,
            output,
        } => {
            if let Some(config_path) = config {
                // Config-based pipeline execution (existing mode)
                run_command(&config_path, &inputs, output.as_deref())
            } else if let Some(model_path) = model {
                // Direct .holo execution with --prompt (new T5 mode)
                run::run_direct_command(
                    &model_path,
                    prompt.as_deref(),
                    &tokenizer,
                    max_length,
                    output.as_deref(),
                )
            } else {
                anyhow::bail!("Either --config or model path must be specified")
            }
        }

        Commands::Download {
            model_id,
            output,
            revision,
        } => download_command(&model_id, &output, revision.as_deref()),

        Commands::Info { model, detailed } => info_command(&model, detailed),

        Commands::Validate { model, check_ops } => validate_command(&model, check_ops),

        Commands::Bundle {
            inputs,
            output,
            config,
            names,
        } => {
            if let Some(config_path) = config {
                bundle_from_config(&config_path, &output)
            } else {
                let names_opt = if names.is_empty() {
                    None
                } else {
                    Some(names.as_slice())
                };
                bundle_command(&inputs, &output, names_opt)
            }
        }

        Commands::Extract { bundle, output } => extract_command(&bundle, &output),

        Commands::List { bundle } => list_pipeline_command(&bundle),

        Commands::RunPipeline {
            pipeline,
            prompt,
            max_tokens,
        } => {
            let result = run_pipeline_bundle_command(&pipeline, &prompt, max_tokens)?;
            println!("\n=== Generated Output ===\n{}\n", result);
            Ok(())
        }

        Commands::BundlePipeline {
            encoder,
            decoder,
            tokenizer,
            models,
            config,
            output,
        } => {
            if let Some(config_path) = config {
                bundle_pipeline_from_config(&config_path, &output)
            } else {
                // Collect all model paths from explicit flags and --model args
                let mut inputs: Vec<(String, PathBuf)> = Vec::new();

                if let Some(path) = encoder {
                    inputs.push(("encoder".to_string(), path));
                }
                if let Some(path) = decoder {
                    inputs.push(("decoder".to_string(), path));
                }
                if let Some(path) = tokenizer {
                    inputs.push(("tokenizer".to_string(), path));
                }

                // Parse name=path from --model args
                for model_spec in &models {
                    let parts: Vec<&str> = model_spec.splitn(2, '=').collect();
                    if parts.len() == 2 {
                        inputs.push((parts[0].to_string(), PathBuf::from(parts[1])));
                    } else {
                        anyhow::bail!(
                            "Invalid model specification '{}'. Expected format: name=path",
                            model_spec
                        );
                    }
                }

                if inputs.is_empty() {
                    anyhow::bail!(
                        "No models specified. Use --encoder, --decoder, --tokenizer, or --model name=path"
                    );
                }

                let input_refs: Vec<(&str, &std::path::Path)> = inputs
                    .iter()
                    .map(|(name, path)| (name.as_str(), path.as_path()))
                    .collect();

                bundle_pipeline_command(&input_refs, &output)
            }
        }

        Commands::CompilePipeline {
            encoder,
            decoder,
            tokenizer,
            models,
            config,
            output,
            weight_threshold,
            partition,
            partition_size,
            memory_budget,
            keep_intermediates,
        } => compile_pipeline_command(
            encoder.as_deref(),
            decoder.as_deref(),
            tokenizer.as_deref(),
            &models,
            config.as_deref(),
            &output,
            weight_threshold,
            partition,
            partition_size,
            memory_budget,
            keep_intermediates,
        ),
    }
}

/// Parse embed file specifications from CLI arguments.
///
/// Format: type:path or type:path:custom_id
/// For raw type: raw:content_type:path or raw:content_type:path:custom_id
#[cfg(feature = "onnx")]
fn parse_embed_files(
    specs: &[String],
) -> anyhow::Result<Vec<hologram_ai_onnx::core::EmbeddedFileConfig>> {
    use hologram_ai_onnx::core::{EmbeddedFileConfig, SectionType};

    let mut result = Vec::new();

    for spec in specs {
        let parts: Vec<&str> = spec.split(':').collect();
        if parts.len() < 2 {
            anyhow::bail!(
                "Invalid embed format '{}'. Expected type:path or type:path:custom_id",
                spec
            );
        }

        let type_str = parts[0];
        let (section_type, path, custom_id) = if type_str == "raw" {
            // raw:content_type:path[:custom_id]
            if parts.len() < 3 {
                anyhow::bail!(
                    "Invalid raw embed format '{}'. Expected raw:content_type:path[:custom_id]",
                    spec
                );
            }
            let content_type = parts[1].to_string();
            let path = parts[2];
            let custom_id = if parts.len() > 3 {
                Some(parts[3].to_string())
            } else {
                None
            };
            (SectionType::Raw { content_type }, path, custom_id)
        } else {
            // type:path[:custom_id]
            let path = parts[1];
            let custom_id = if parts.len() > 2 {
                Some(parts[2].to_string())
            } else {
                None
            };

            let section_type = match type_str {
                "vocabulary" | "vocab" => SectionType::Vocabulary,
                "vocabulary_json" | "vocab_json" => SectionType::VocabularyJson,
                "tokenizer_config" | "tokenizer" => SectionType::TokenizerConfig,
                "model_config" | "config" => SectionType::ModelConfig,
                "special_tokens" => SectionType::SpecialTokensMap,
                "preprocessor" | "preprocessor_config" => SectionType::PreprocessorConfig,
                "sentencepiece" | "spm" => SectionType::SentencePiece,
                "generation" | "generation_config" => SectionType::GenerationConfig,
                _ => {
                    anyhow::bail!(
                        "Unknown embed type '{}'. Valid types: vocabulary, vocabulary_json, \
                         tokenizer_config, model_config, special_tokens, preprocessor, \
                         sentencepiece, generation, raw",
                        type_str
                    );
                }
            };
            (section_type, path, custom_id)
        };

        let mut config = EmbeddedFileConfig::new(PathBuf::from(path), section_type);
        config.custom_id = custom_id;
        result.push(config);
    }

    Ok(result)
}

#[cfg(not(feature = "onnx"))]
fn parse_embed_files(_specs: &[String]) -> anyhow::Result<Vec<()>> {
    Ok(Vec::new())
}

/// Compile models using settings from a unified config file.
#[allow(clippy::too_many_arguments)] // CLI override helper mirrors flag surface.
fn compile_with_config(
    config_path: &std::path::Path,
    input_override: Option<&std::path::Path>,
    output_override: Option<&std::path::Path>,
    partition_override: bool,
    partition_size_override: usize,
    memory_budget_override: Option<usize>,
    weight_threshold_override: usize,
    bundle: bool,
    #[cfg(feature = "onnx")] embed_files: &[hologram_ai_onnx::core::EmbeddedFileConfig],
    #[cfg(not(feature = "onnx"))] _embed_files: &[()],
    layer_wise: bool,
) -> anyhow::Result<()> {
    use crate::config::UnifiedConfig;
    #[cfg(feature = "onnx")]
    use hologram_ai_onnx::core::OnnxConfig;
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
    let config_dir = config_path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));

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
            compiler_config.decompose_conv2d,
            compiler_config.decompose_pooling,
            compiler_config.enable_resize_upscaling,
            &std::collections::HashMap::new(), // No input shapes from config yet
            bundle,
            embed_files,
            layer_wise,
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
            compiler_config.decompose_conv2d,
            compiler_config.decompose_pooling,
            compiler_config.enable_resize_upscaling,
            &std::collections::HashMap::new(), // No input shapes from config yet
            bundle,
            embed_files,
            layer_wise,
        )?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_parse_compile() {
        let args = vec!["hologram-onnx", "compile", "input.onnx", "-o", "output"];
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
        let args = vec!["hologram-onnx", "compile", "--config", "pipeline.toml"];
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
                assert_eq!(config, Some(PathBuf::from("pipeline.toml")));
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
        ];
        let cli = Cli::try_parse_from(args).unwrap();
        match cli.command {
            Commands::Run { config, output, .. } => {
                assert_eq!(config, Some(PathBuf::from("pipeline.toml")));
                assert_eq!(output, Some(PathBuf::from("output_dir")));
            }
            _ => panic!("Expected Run command"),
        }
    }
}
