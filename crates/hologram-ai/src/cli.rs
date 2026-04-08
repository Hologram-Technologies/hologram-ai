//! CLI entry point for hologram-ai.

use clap::Parser;
use hologram_ai::compiler::{ModelCompiler, ModelSource};
use hologram_ai::download;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "hologram-ai", about = "AI model compiler for hologram runtime")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand)]
enum Command {
    /// Inspect a `.holo` archive or ONNX model file.
    Info {
        /// Path to a `.holo` or `.onnx` file.
        file: PathBuf,
        /// Levels of detail (for `.holo` files, may be repeated).
        #[arg(long, value_enum, default_values_t = [hologram::hologram_cli::commands::inspect::DetailLevel::Summary])]
        detail: Vec<hologram::hologram_cli::commands::inspect::DetailLevel>,
    },
    /// Compile a model to a `.holo` archive file.
    ///
    /// Single model: `hologram-ai compile -m model.onnx -o out/`
    /// Multi-component: `hologram-ai compile --manifest pipeline.toml -o out/`
    Compile {
        /// Path to the input ONNX model file. Mutually exclusive with --manifest.
        #[arg(
            short,
            long,
            required_unless_present = "manifest",
            conflicts_with = "manifest"
        )]
        model: Option<PathBuf>,
        /// Path to a TOML manifest for multi-component compilation.
        /// Mutually exclusive with --model.
        #[arg(long, value_name = "FILE", conflicts_with = "model")]
        manifest: Option<PathBuf>,
        /// Output directory for the compiled `.holo` archive.
        #[arg(short, long, value_name = "DIR")]
        output: PathBuf,
        /// Path to tokenizer.json (auto-detected from model directory if omitted).
        #[arg(long, value_name = "FILE")]
        tokenizer: Option<PathBuf>,
        /// Fixed sequence length for compilation (default: model's context_length).
        /// All shapes are baked to this value. Inputs are padded at runtime.
        #[arg(long, value_name = "N")]
        seq_len: Option<u64>,
        /// Quantize f32 weights at compile time for LUT-GEMM acceleration.
        /// Supported: q2_0 (2-bit, 4 centroids), q4_0 (4-bit, 16 centroids), q8_0 (8-bit, 256 centroids).
        #[arg(long, value_name = "SCHEME")]
        quantize: Option<String>,
    },
    /// Run a compiled `.holo` archive with shape-aware inference.
    Run(hologram_ai::commands::run_cmd::RunArgs),
    /// Download a model from HuggingFace Hub.
    Download(download::DownloadArgs),
    /// Validate a model: import, optimize, compile, and report results.
    Validate {
        /// Path to the ONNX model file.
        #[arg(short, long)]
        model: PathBuf,
    },
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber_init();
    let cli = Cli::parse();

    match cli.command {
        Command::Info { file, detail } => {
            let ext = file.extension().and_then(|e| e.to_str()).unwrap_or("");
            match ext {
                "holo" => inspect_holo(file, detail)?,
                "onnx" => inspect_onnx(&file)?,
                other => {
                    anyhow::bail!("info supports .holo and .onnx files, got '.{other}'")
                }
            }
        }
        Command::Compile {
            model,
            manifest,
            output,
            tokenizer,
            seq_len,
            quantize,
        } => {
            let (source, model_path, manifest_kind) = if let Some(manifest_path) = &manifest {
                let (source, kind) = parse_manifest(manifest_path)?;
                (source, manifest_path.clone(), kind)
            } else {
                let model_path = model.as_ref().expect("--model or --manifest required");
                let source = model_source_from_path(model_path)?;
                (source, model_path.clone(), None)
            };

            let quant_strategy = match quantize.as_deref() {
                Some("q2_0" | "Q2_0") => hologram_ai_common::lower::QuantStrategy::Q2_0,
                Some("q4_0" | "Q4_0") => hologram_ai_common::lower::QuantStrategy::Q4_0,
                Some("q8_0" | "Q8_0") => hologram_ai_common::lower::QuantStrategy::Q8_0,
                Some("none" | "f32") => hologram_ai_common::lower::QuantStrategy::Auto,
                Some(other) => anyhow::bail!(
                    "unsupported quantization scheme '{other}' (supported: q2_0, q4_0, q8_0, none)"
                ),
                None => hologram_ai_common::lower::QuantStrategy::Q4_0,
            };
            let compiler = ModelCompiler {
                seq_len_override: seq_len,
                quant_strategy,
                ..Default::default()
            };
            let compiled = compiler.compile(source)?;
            if output.exists() && !output.is_dir() {
                anyhow::bail!(
                    "'{}' exists and is not a directory. Remove it or choose a different --output path.",
                    output.display()
                );
            }
            std::fs::create_dir_all(&output)?;
            let stem = model_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("model");
            let holo_path = output.join(format!("{stem}.holo"));

            // Resolve tokenizer path: explicit flag or auto-detect from model dir.
            let tok_path = tokenizer.or_else(|| {
                let dir = model_path.parent()?;
                let candidate = dir.join("tokenizer.json");
                candidate.exists().then_some(candidate)
            });

            // Embed model metadata section.
            // Use manifest kind if provided, otherwise auto-detect.
            let kind = if let Some(k) = manifest_kind {
                k
            } else {
                // Detect LLM: either the metadata says so (GGUF sets arch/n_layers)
                // or we have a tokenizer (strong signal for text models).
                let is_llm = (compiled.metadata.arch != "unknown"
                    && compiled.metadata.n_layers > 0)
                    || tok_path.is_some();
                if is_llm {
                    hologram::hologram_archive::section::model_meta::ModelKind::TextLlm
                } else {
                    hologram::hologram_archive::section::model_meta::ModelKind::Generic
                }
            };
            let model_meta = hologram::hologram_archive::section::model_meta::ModelMetaSection {
                kind,
                arch: compiled.metadata.arch.clone(),
                description: format!("{} ({})", compiled.metadata.arch, model_path.display()),
                max_seq_len: compiled.metadata.context_len,
                supports_prompt: tok_path.is_some(),
                n_layers: compiled.metadata.n_layers,
                n_kv_heads: compiled.metadata.n_kv_heads,
                head_dim: compiled.metadata.head_dim,
                kv_k_bits: 0,
                kv_v_bits: 0,
                kv_boundary_layers: 2,
                kv_wht: false,
            };
            let mut final_bytes =
                hologram_ai::compiler::rebuild_archive_with_section(&compiled.bytes, &model_meta)?;

            // Embed tokenizer section if available.
            if let Some(tok_path) = &tok_path {
                let section =
                    hologram_ai_tokenizer::archive::TokenizerSectionData::from_tokenizer_json(
                        tok_path,
                    )?;
                eprintln!(
                    "embedding tokenizer ({} tokens) from {}",
                    section.vocab.len(),
                    tok_path.display()
                );
                final_bytes =
                    hologram_ai::compiler::rebuild_archive_with_section(&final_bytes, &section)?;
            }

            std::fs::write(&holo_path, &final_bytes)?;
            println!(
                "wrote {} ({} nodes, {} weight bytes, {} warnings)",
                holo_path.display(),
                compiled.stats.node_count,
                compiled.stats.total_weight_bytes,
                compiled.stats.import_warnings,
            );
        }
        Command::Run(args) => {
            hologram_ai::commands::run_cmd::execute(args)?;
        }
        Command::Download(args) => {
            download::run(args)?;
        }
        Command::Validate { model } => {
            let report = hologram_ai::validate::validate_model(&model);
            println!("{report}");
            if !report.compilation_ok {
                std::process::exit(1);
            }
        }
    }

    Ok(())
}

// ── Info ─────────────────────────────────────────────────────────────────────

/// Inspect a compiled `.holo` archive — delegates to `hologram inspect`.
fn inspect_holo(
    file: PathBuf,
    detail: Vec<hologram::hologram_cli::commands::inspect::DetailLevel>,
) -> anyhow::Result<()> {
    use hologram::hologram_cli::commands::inspect::{execute, InspectArgs};
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

// ── Helpers ──────────────────────────────────────────────────────────────────

fn model_source_from_path(path: &std::path::Path) -> anyhow::Result<ModelSource> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    match ext {
        "onnx" => Ok(ModelSource::OnnxPath(path.to_owned())),
        other => anyhow::bail!("unsupported model extension: '.{other}'"),
    }
}

// ── Multi-component manifest ─────────────────────────────────────────────────

/// TOML manifest for multi-component compilation.
///
/// Example:
/// ```toml
/// [[component]]
/// name = "encoder"
/// path = "encoder.onnx"
/// role = "encoder"
/// weight_group = "shared"
///
/// [[connection]]
/// from = "encoder:hidden_states"
/// to = "decoder:encoder_hidden_states"
/// ```
#[derive(serde::Deserialize)]
struct Manifest {
    /// Optional model kind override (e.g. "image-gen", "text-llm", "vision").
    /// When set, overrides the auto-detection heuristic.
    kind: Option<String>,
    component: Vec<ManifestComponent>,
    #[serde(default)]
    connection: Vec<ManifestConnection>,
}

#[derive(serde::Deserialize)]
struct ManifestComponent {
    name: String,
    path: String,
    role: String,
    weight_group: String,
}

#[derive(serde::Deserialize)]
struct ManifestConnection {
    from: String,
    to: String,
}

/// Parse a model kind string from a manifest into the hologram ModelKind enum.
fn parse_model_kind(s: &str) -> hologram::hologram_archive::section::model_meta::ModelKind {
    use hologram::hologram_archive::section::model_meta::ModelKind;
    match s {
        "text-llm" | "llm" => ModelKind::TextLlm,
        "text-encoder" | "encoder" => ModelKind::TextEncoder,
        "vision" => ModelKind::Vision,
        "audio" => ModelKind::Audio,
        "image-gen" | "diffusion" => ModelKind::ImageGen,
        "audio-gen" | "tts" => ModelKind::AudioGen,
        "video-gen" => ModelKind::VideoGen,
        "multi-modal" => ModelKind::MultiModal,
        _ => ModelKind::Generic,
    }
}

fn parse_manifest(
    path: &std::path::Path,
) -> anyhow::Result<(
    ModelSource,
    Option<hologram::hologram_archive::section::model_meta::ModelKind>,
)> {
    use anyhow::Context as _;
    use hologram_ai::compiler::ComponentInput;
    use hologram_ai_common::sections::meta::{ComponentConnection, ComponentRole};

    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading manifest {}", path.display()))?;
    let manifest: Manifest =
        toml::from_str(&text).with_context(|| format!("parsing manifest {}", path.display()))?;
    let manifest_kind = manifest.kind.as_deref().map(parse_model_kind);

    let manifest_dir = path.parent().unwrap_or(std::path::Path::new("."));

    let components: Vec<ComponentInput> = manifest
        .component
        .into_iter()
        .map(|c| {
            let role = match c.role.as_str() {
                "prefill" => ComponentRole::Prefill,
                "decode" => ComponentRole::Decode,
                "encoder" => ComponentRole::Encoder,
                "decoder" => ComponentRole::Decoder,
                "backbone" => ComponentRole::Backbone,
                "generative_head" => ComponentRole::GenerativeHead,
                "forward" => ComponentRole::Forward,
                other => ComponentRole::Custom(other.to_string()),
            };
            ComponentInput {
                name: c.name,
                path: manifest_dir.join(&c.path),
                role,
                weight_group: c.weight_group,
            }
        })
        .collect();

    let connections: Vec<ComponentConnection> = manifest
        .connection
        .into_iter()
        .map(|c| {
            let (from_component, from_output) =
                c.from.split_once(':').unwrap_or((&c.from, "output"));
            let (to_component, to_input) = c.to.split_once(':').unwrap_or((&c.to, "input"));
            ComponentConnection {
                from_component: from_component.to_string(),
                from_output: from_output.to_string(),
                to_component: to_component.to_string(),
                to_input: to_input.to_string(),
            }
        })
        .collect();

    Ok((
        ModelSource::MultiOnnx {
            components,
            connections,
        },
        manifest_kind,
    ))
}

fn tracing_subscriber_init() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();
}
