//! CLI entry point for hologram-ai.

use anyhow::Context as _;
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
    /// Config file:  `hologram-ai compile --config models/model/hologram.toml`
    /// Multi-component: `hologram-ai compile --manifest pipeline.toml -o out/`
    Compile {
        /// Path to a hologram.toml config file. All paths in the config are
        /// resolved relative to the config file's directory. CLI flags override
        /// config values.
        #[arg(short, long, value_name = "FILE")]
        config: Option<PathBuf>,
        /// Path to the input ONNX model file. Mutually exclusive with --manifest.
        #[arg(
            short,
            long,
            required_unless_present_any = &["manifest", "config"],
            conflicts_with = "manifest"
        )]
        model: Option<PathBuf>,
        /// Path to a TOML manifest for multi-component compilation.
        /// Mutually exclusive with --model.
        #[arg(long, value_name = "FILE", conflicts_with = "model")]
        manifest: Option<PathBuf>,
        /// Output directory for the compiled `.holo` archive.
        #[arg(short, long, value_name = "DIR")]
        output: Option<PathBuf>,
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

        // ── Host metadata flags (Plan 060) ───────────────────────────────
        // All flags are optional. Manifest `[host]` values take precedence
        // over flags. Omitting all of these and having no manifest `[host]`
        // table means no `HostMetaSection` is written.
        /// Single-turn prompt template, e.g. `"<|user|>{prompt}<|assistant|>"`.
        #[arg(long, value_name = "TEMPLATE")]
        prompt_template: Option<String>,
        /// Jinja-style multi-turn chat template. Often auto-populated from
        /// GGUF v3 `tokenizer.chat_template` at import time.
        #[arg(long, value_name = "TEMPLATE")]
        chat_template: Option<String>,
        /// Default sampling temperature (f32).
        #[arg(long, value_name = "F32")]
        temperature: Option<f32>,
        /// Default top-k sampling cutoff.
        #[arg(long, value_name = "N")]
        top_k: Option<u32>,
        /// Default top-p (nucleus) sampling cutoff.
        #[arg(long, value_name = "F32")]
        top_p: Option<f32>,
        /// Default repetition penalty (f32, typically >= 1.0).
        #[arg(long, value_name = "F32")]
        repetition_penalty: Option<f32>,
        /// Stop strings for generation (repeatable).
        #[arg(long = "stop", value_name = "STR")]
        stop: Vec<String>,
        /// Model card: author.
        #[arg(long, value_name = "NAME")]
        author: Option<String>,
        /// Model card: SPDX license identifier.
        #[arg(long, value_name = "SPDX")]
        license: Option<String>,
        /// Model card: source URL (HuggingFace Hub, etc.).
        #[arg(long, value_name = "URL")]
        source_url: Option<String>,
        /// Model card: tags (repeatable).
        #[arg(long = "tag", value_name = "TAG")]
        tag: Vec<String>,

        // ── ViT patch pruning (Plan 063) ─────────────────────────────────
        /// Patch budget ratio for ViT models (PixelPrune). Controls what
        /// fraction of image patches the compiled ViT retains. A runtime
        /// kernel selects the most informative patches before execution.
        /// Range: (0.0, 1.0]. Default: 0.75. Use `--no-patch-prune` to disable.
        #[arg(long, value_name = "RATIO")]
        patch_budget: Option<f32>,
        /// Disable ViT patch pruning entirely (overrides --patch-budget).
        #[arg(long)]
        no_patch_prune: bool,
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
            config,
            model,
            manifest,
            output,
            tokenizer,
            seq_len,
            quantize,
            prompt_template,
            chat_template,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            stop,
            author,
            license,
            source_url,
            tag,
            patch_budget,
            no_patch_prune,
        } => {
            // Load config file if provided. Paths are resolved relative to
            // the config file's directory.
            let cfg = if let Some(config_path) = &config {
                let cfg = load_model_config(config_path)
                    .with_context(|| format!("loading config from {}", config_path.display()))?;
                Some(cfg)
            } else {
                None
            };
            let config_dir = config
                .as_ref()
                .and_then(|p| p.parent())
                .map(|p| p.to_path_buf());

            // Merge: CLI flags override config values.
            let effective_model = model.or_else(|| {
                cfg.as_ref()
                    .and_then(|c| c.model.path.as_ref())
                    .map(|p| resolve_config_path(&config_dir, p))
            });
            let effective_output = output.or_else(|| {
                cfg.as_ref()
                    .and_then(|c| c.model.output.as_ref())
                    .map(|p| resolve_config_path(&config_dir, p))
            });
            let effective_quantize = quantize.or_else(|| {
                cfg.as_ref().and_then(|c| c.model.quantize.clone())
            });
            let effective_seq_len = seq_len.or_else(|| {
                cfg.as_ref().and_then(|c| c.model.seq_len)
            });
            let effective_tokenizer = tokenizer.or_else(|| {
                cfg.as_ref()
                    .and_then(|c| c.model.tokenizer.as_ref())
                    .map(|p| resolve_config_path(&config_dir, p))
            });

            let host_cli = HostMetaCliArgs {
                prompt_template: prompt_template.or_else(|| {
                    cfg.as_ref().and_then(|c| c.model.prompt_template.clone())
                }),
                chat_template,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                stop,
                author,
                license,
                source_url,
                tags: tag,
            };
            let (source, model_path, manifest_kind, manifest_host) =
                if let Some(manifest_path) = &manifest {
                    let (source, kind, host) = parse_manifest(manifest_path)?;
                    (source, manifest_path.clone(), kind, host)
                } else {
                    let model_path = effective_model
                        .as_ref()
                        .context("--model, --manifest, or --config with [model].path required")?;
                    let source = model_source_from_path(model_path)?;
                    (source, model_path.clone(), None, None)
                };
            let output = effective_output
                .context("--output or --config with [model].output required")?;
            let tokenizer = effective_tokenizer;
            let seq_len = effective_seq_len;
            let quantize = effective_quantize;

            // These sections are collected here so that streaming compilation
            // can include them in the single build pass (no rebuild-with-section
            // round-trips on a 14 GB archive).
            let quant_strategy = match quantize.as_deref() {
                Some("q2_0" | "Q2_0") => hologram_ai_common::lower::QuantStrategy::Q2_0,
                Some("q4_0" | "Q4_0") => hologram_ai_common::lower::QuantStrategy::Q4_0,
                Some("q8_0" | "Q8_0") => hologram_ai_common::lower::QuantStrategy::Q8_0,
                Some("none" | "f32") => hologram_ai_common::lower::QuantStrategy::None,
                Some(other) => anyhow::bail!(
                    "unsupported quantization scheme '{other}' (supported: q2_0, q4_0, q8_0, none)"
                ),
                None => hologram_ai_common::lower::QuantStrategy::Q4_0,
            };
            let patch_budget_ratio = if no_patch_prune {
                None
            } else {
                patch_budget.map(Some).unwrap_or(Some(0.75))
            };
            let compiler = ModelCompiler {
                seq_len_override: seq_len,
                quant_strategy,
                patch_budget_ratio,
                ..Default::default()
            };

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

            // ── Collect all sections BEFORE compilation ──────────────────
            // Single-pass: sections are embedded during archive build.
            // No post-processing rebuild_archive_with_section round-trips.
            let mut sections = hologram_ai::compiler::ArchiveSections::new();

            // Tokenizer section.
            if let Some(tok_path) = &tok_path {
                let tok_section =
                    hologram_ai_tokenizer::archive::TokenizerSectionData::from_tokenizer_json(
                        tok_path,
                    )?;
                eprintln!(
                    "embedding tokenizer ({} tokens) from {}",
                    tok_section.vocab.len(),
                    tok_path.display()
                );
                sections.add(&tok_section);
            }

            // Host metadata section.
            let host_section = build_host_meta(&host_cli, manifest_host.as_ref(), None);
            if !host_section.is_empty() {
                sections.add(&host_section);
                eprintln!("embedded host metadata section");
            }

            // ── Compile with all sections in a single pass ──────────────
            let compiled = compiler.compile_with_sections(source, sections)?;

            if let Some(archive_path) = &compiled.path {
                // Streaming path: archive is already on disk. Move it.
                std::fs::rename(archive_path, &holo_path)
                    .or_else(|_| std::fs::copy(archive_path, &holo_path).map(|_| ()))
                    .with_context(|| {
                        format!(
                            "moving archive from {} to {}",
                            archive_path.display(),
                            holo_path.display(),
                        )
                    })?;
            } else {
                // In-memory path: add model_meta section and write.
                let kind = if let Some(k) = manifest_kind {
                    k
                } else {
                    let is_llm = (compiled.metadata.arch != "unknown"
                        && compiled.metadata.n_layers > 0)
                        || tok_path.is_some();
                    if is_llm {
                        hologram::hologram_archive::section::model_meta::ModelKind::TextLlm
                    } else {
                        hologram::hologram_archive::section::model_meta::ModelKind::Generic
                    }
                };
                let model_meta =
                    hologram::hologram_archive::section::model_meta::ModelMetaSection {
                        kind,
                        arch: compiled.metadata.arch.clone(),
                        description: format!(
                            "{} ({})",
                            compiled.metadata.arch,
                            model_path.display()
                        ),
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
                let final_bytes = hologram_ai::compiler::rebuild_archive_with_section(
                    &compiled.bytes,
                    &model_meta,
                )?;
                std::fs::write(&holo_path, &final_bytes)?;
            }
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
    /// Optional host-facing metadata (prompt/chat template, sampling
    /// defaults, model card). See `HostMetaManifest` for fields.
    #[serde(default)]
    host: Option<HostMetaManifest>,
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

/// `[host]` table in a TOML manifest — the manifest-level counterpart to
/// `HostMetaSection`. All fields optional.
#[derive(Default, serde::Deserialize)]
struct HostMetaManifest {
    prompt_template: Option<String>,
    chat_template: Option<String>,
    #[serde(default)]
    sampling: Option<SamplingManifest>,
    /// `[host.ports]` — logical name → graph port id.
    #[serde(default)]
    ports: std::collections::BTreeMap<String, String>,
    #[serde(default)]
    model_card: Option<ModelCardManifest>,
}

#[derive(Default, serde::Deserialize)]
struct SamplingManifest {
    temperature: Option<f32>,
    top_k: Option<u32>,
    top_p: Option<f32>,
    repetition_penalty: Option<f32>,
    #[serde(default)]
    stop: Vec<String>,
}

#[derive(Default, serde::Deserialize)]
struct ModelCardManifest {
    author: Option<String>,
    license: Option<String>,
    source_url: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
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
    Option<HostMetaManifest>,
)> {
    use anyhow::Context as _;
    use hologram_ai::compiler::ComponentInput;
    use hologram_ai_common::sections::meta::{ComponentConnection, ComponentRole};

    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading manifest {}", path.display()))?;
    let manifest: Manifest =
        toml::from_str(&text).with_context(|| format!("parsing manifest {}", path.display()))?;
    let manifest_kind = manifest.kind.as_deref().map(parse_model_kind);
    let host = manifest.host;

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
        host,
    ))
}

fn tracing_subscriber_init() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();
}

// ── Host metadata (Plan 060) ─────────────────────────────────────────────────

/// Host metadata values passed via the `compile` subcommand's CLI flags.
/// Mirrors `HostMetaManifest` but with explicit `Vec`s for repeatable flags.
#[derive(Default)]
struct HostMetaCliArgs {
    prompt_template: Option<String>,
    chat_template: Option<String>,
    temperature: Option<f32>,
    top_k: Option<u32>,
    top_p: Option<f32>,
    repetition_penalty: Option<f32>,
    stop: Vec<String>,
    author: Option<String>,
    license: Option<String>,
    source_url: Option<String>,
    tags: Vec<String>,
}

/// Merge host metadata sources into a `HostMetaSection` honoring the
/// documented precedence order: **manifest > CLI flags > importer
/// auto-populated > unset**.
///
/// `imported_chat_template` is the Phase 4 integration point for GGUF v3
/// `tokenizer.chat_template` auto-population. It is passed as a plain
/// argument so that phase 2 can ship before phase 4 lands. Pass `None`
/// until then.
fn build_host_meta(
    cli: &HostMetaCliArgs,
    manifest: Option<&HostMetaManifest>,
    imported_chat_template: Option<String>,
) -> hologram::hologram_archive::section::host_meta::HostMetaSection {
    use hologram::hologram_archive::section::host_meta::{
        HostMetaSection, ModelCard, PortBinding, SamplingDefaults, HOST_META_VERSION,
    };

    // Helper: pick manifest over CLI over fallback.
    fn pick<T: Clone>(manifest: Option<T>, cli: Option<T>, fallback: Option<T>) -> Option<T> {
        manifest.or(cli).or(fallback)
    }

    let (manifest_prompt, manifest_chat, manifest_sampling, manifest_ports, manifest_card) =
        match manifest {
            Some(m) => (
                m.prompt_template.clone(),
                m.chat_template.clone(),
                m.sampling.as_ref(),
                Some(&m.ports),
                m.model_card.as_ref(),
            ),
            None => (None, None, None, None, None),
        };

    let sampling = {
        let temperature = pick(
            manifest_sampling.and_then(|s| s.temperature),
            cli.temperature,
            None,
        );
        let top_k = pick(manifest_sampling.and_then(|s| s.top_k), cli.top_k, None);
        let top_p = pick(manifest_sampling.and_then(|s| s.top_p), cli.top_p, None);
        let repetition_penalty = pick(
            manifest_sampling.and_then(|s| s.repetition_penalty),
            cli.repetition_penalty,
            None,
        );
        // Stop list: manifest overrides CLI entirely when set (not merged),
        // mirroring how we treat other fields. Empty vec == not set.
        let stop = match manifest_sampling {
            Some(s) if !s.stop.is_empty() => s.stop.clone(),
            _ => cli.stop.clone(),
        };
        if temperature.is_none()
            && top_k.is_none()
            && top_p.is_none()
            && repetition_penalty.is_none()
            && stop.is_empty()
        {
            None
        } else {
            Some(SamplingDefaults {
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                stop,
            })
        }
    };

    let ports: Vec<PortBinding> = manifest_ports
        .map(|m| {
            m.iter()
                .map(|(k, v)| PortBinding {
                    logical_name: k.clone(),
                    graph_port: v.clone(),
                })
                .collect()
        })
        .unwrap_or_default();

    let model_card = {
        let author = pick(
            manifest_card.and_then(|c| c.author.clone()),
            cli.author.clone(),
            None,
        );
        let license = pick(
            manifest_card.and_then(|c| c.license.clone()),
            cli.license.clone(),
            None,
        );
        let source_url = pick(
            manifest_card.and_then(|c| c.source_url.clone()),
            cli.source_url.clone(),
            None,
        );
        let tags = match manifest_card {
            Some(c) if !c.tags.is_empty() => c.tags.clone(),
            _ => cli.tags.clone(),
        };
        if author.is_none() && license.is_none() && source_url.is_none() && tags.is_empty() {
            None
        } else {
            Some(ModelCard {
                author,
                license,
                source_url,
                tags,
            })
        }
    };

    HostMetaSection {
        version: HOST_META_VERSION,
        prompt_template: pick(manifest_prompt, cli.prompt_template.clone(), None),
        chat_template: pick(
            manifest_chat,
            cli.chat_template.clone(),
            imported_chat_template,
        ),
        sampling,
        ports,
        model_card,
    }
}

// ── Model config file (hologram.toml) ────────────────────────────────────────

/// `hologram.toml` config file for per-model compilation settings.
///
/// All paths are resolved relative to the config file's directory.
/// CLI flags override config values.
#[derive(serde::Deserialize)]
struct ModelConfig {
    #[serde(default)]
    model: ModelConfigModel,
    #[serde(default)]
    #[allow(dead_code)] // Used when `--config` is passed to the `run` command (future).
    run: Option<ModelConfigRun>,
}

#[derive(Default, serde::Deserialize)]
struct ModelConfigModel {
    path: Option<String>,
    output: Option<String>,
    quantize: Option<String>,
    seq_len: Option<u64>,
    tokenizer: Option<String>,
    prompt_template: Option<String>,
}

#[allow(dead_code)] // Fields used when `--config` is passed to the `run` command (future).
#[derive(Default, serde::Deserialize)]
struct ModelConfigRun {
    prompt: Option<String>,
    max_tokens: Option<u32>,
}

fn load_model_config(path: &std::path::Path) -> anyhow::Result<ModelConfig> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading config {}", path.display()))?;
    toml::from_str(&text).with_context(|| format!("parsing config {}", path.display()))
}

fn resolve_config_path(config_dir: &Option<PathBuf>, relative: &str) -> PathBuf {
    match config_dir {
        Some(dir) => dir.join(relative),
        None => PathBuf::from(relative),
    }
}

#[cfg(test)]
mod host_meta_tests {
    use super::*;
    use hologram::hologram_archive::section::host_meta::HostMetaSection;

    #[test]
    fn empty_inputs_produce_empty_section() {
        let cli = HostMetaCliArgs::default();
        let section = build_host_meta(&cli, None, None);
        assert!(section.is_empty());
    }

    #[test]
    fn cli_flags_populate_all_fields() {
        let cli = HostMetaCliArgs {
            prompt_template: Some("<|u|>{prompt}<|a|>".into()),
            chat_template: None,
            temperature: Some(0.8),
            top_k: Some(50),
            top_p: Some(0.9),
            repetition_penalty: Some(1.2),
            stop: vec!["</s>".into()],
            author: Some("me".into()),
            license: Some("MIT".into()),
            source_url: Some("https://example.com/model".into()),
            tags: vec!["test".into()],
        };
        let section = build_host_meta(&cli, None, None);
        assert_eq!(
            section.prompt_template.as_deref(),
            Some("<|u|>{prompt}<|a|>")
        );
        let s = section.sampling.expect("sampling");
        assert_eq!(s.temperature, Some(0.8));
        assert_eq!(s.top_k, Some(50));
        assert_eq!(s.stop, vec!["</s>".to_string()]);
        let c = section.model_card.expect("card");
        assert_eq!(c.author.as_deref(), Some("me"));
        assert_eq!(c.license.as_deref(), Some("MIT"));
        assert_eq!(c.tags, vec!["test".to_string()]);
    }

    #[test]
    fn manifest_overrides_cli_for_scalars() {
        let cli = HostMetaCliArgs {
            prompt_template: Some("cli-template".into()),
            temperature: Some(0.1),
            ..Default::default()
        };
        let manifest = HostMetaManifest {
            prompt_template: Some("manifest-template".into()),
            sampling: Some(SamplingManifest {
                temperature: Some(0.9),
                ..Default::default()
            }),
            ..Default::default()
        };
        let section = build_host_meta(&cli, Some(&manifest), None);
        assert_eq!(
            section.prompt_template.as_deref(),
            Some("manifest-template")
        );
        assert_eq!(section.sampling.expect("sampling").temperature, Some(0.9),);
    }

    #[test]
    fn cli_fills_gap_when_manifest_silent() {
        let cli = HostMetaCliArgs {
            temperature: Some(0.1),
            ..Default::default()
        };
        let manifest = HostMetaManifest {
            prompt_template: Some("manifest-template".into()),
            sampling: None, // manifest says nothing about sampling
            ..Default::default()
        };
        let section = build_host_meta(&cli, Some(&manifest), None);
        assert_eq!(
            section.prompt_template.as_deref(),
            Some("manifest-template")
        );
        assert_eq!(
            section.sampling.expect("sampling").temperature,
            Some(0.1), // CLI fills the gap
        );
    }

    #[test]
    fn imported_chat_template_is_lowest_priority() {
        // Imported value used when nothing else supplies it.
        let cli = HostMetaCliArgs::default();
        let section = build_host_meta(&cli, None, Some("imported".into()));
        assert_eq!(section.chat_template.as_deref(), Some("imported"));

        // CLI beats imported.
        let cli_with_flag = HostMetaCliArgs {
            chat_template: Some("cli".into()),
            ..Default::default()
        };
        let section2 = build_host_meta(&cli_with_flag, None, Some("imported".into()));
        assert_eq!(section2.chat_template.as_deref(), Some("cli"));

        // Manifest beats CLI (which beats imported).
        let manifest = HostMetaManifest {
            chat_template: Some("manifest".into()),
            ..Default::default()
        };
        let section3 = build_host_meta(&cli_with_flag, Some(&manifest), Some("imported".into()));
        assert_eq!(section3.chat_template.as_deref(), Some("manifest"));
    }

    #[test]
    fn manifest_stop_list_replaces_cli_stop_list() {
        let cli = HostMetaCliArgs {
            stop: vec!["cli-stop".into()],
            ..Default::default()
        };
        let manifest = HostMetaManifest {
            sampling: Some(SamplingManifest {
                stop: vec!["manifest-stop-a".into(), "manifest-stop-b".into()],
                ..Default::default()
            }),
            ..Default::default()
        };
        let section = build_host_meta(&cli, Some(&manifest), None);
        let s = section.sampling.expect("sampling");
        assert_eq!(
            s.stop,
            vec!["manifest-stop-a".to_string(), "manifest-stop-b".to_string()],
        );
    }

    #[test]
    fn ports_from_manifest_round_trip() {
        let mut ports = std::collections::BTreeMap::new();
        ports.insert("logits".to_string(), "output_0".to_string());
        ports.insert("hidden".to_string(), "output_1".to_string());
        let manifest = HostMetaManifest {
            ports,
            ..Default::default()
        };
        let section = build_host_meta(&HostMetaCliArgs::default(), Some(&manifest), None);
        assert_eq!(section.ports.len(), 2);
        // BTreeMap iteration is alphabetical, so "hidden" comes before "logits".
        assert_eq!(section.ports[0].logical_name, "hidden");
        assert_eq!(section.ports[0].graph_port, "output_1");
        assert_eq!(section.ports[1].logical_name, "logits");
        assert_eq!(section.ports[1].graph_port, "output_0");
    }

    #[test]
    fn full_cli_path_rkyv_round_trip() {
        // End-to-end: build a section from CLI flags, serialize to bytes
        // via the archive-level API, deserialize, and confirm every field
        // survives the round trip. This catches `EmbeddableSection` impl
        // regressions that the unit tests in hologram-archive wouldn't
        // see because they don't exercise the compile-path struct.
        let cli = HostMetaCliArgs {
            prompt_template: Some("tmpl".into()),
            chat_template: Some("{{msg}}".into()),
            temperature: Some(0.7),
            top_k: Some(40),
            top_p: Some(0.95),
            repetition_penalty: Some(1.3),
            stop: vec!["</s>".into()],
            author: Some("ari".into()),
            license: Some("Apache-2.0".into()),
            source_url: Some("https://x/y".into()),
            tags: vec!["chat".into(), "test".into()],
        };
        let section = build_host_meta(&cli, None, None);
        assert!(!section.is_empty());

        use hologram::hologram_archive::section::EmbeddableSection;
        let bytes = section.to_bytes();
        let de = HostMetaSection::deserialize_from(&bytes).expect("deserialize");

        assert_eq!(de.prompt_template.as_deref(), Some("tmpl"));
        assert_eq!(de.chat_template.as_deref(), Some("{{msg}}"));
        let s = de.sampling.expect("sampling");
        assert_eq!(s.temperature, Some(0.7));
        assert_eq!(s.top_k, Some(40));
        assert_eq!(s.top_p, Some(0.95));
        assert_eq!(s.repetition_penalty, Some(1.3));
        assert_eq!(s.stop, vec!["</s>".to_string()]);
        let c = de.model_card.expect("card");
        assert_eq!(c.author.as_deref(), Some("ari"));
        assert_eq!(c.license.as_deref(), Some("Apache-2.0"));
        assert_eq!(c.source_url.as_deref(), Some("https://x/y"));
        assert_eq!(c.tags, vec!["chat".to_string(), "test".to_string()]);
    }
}
