mod convert;
mod hf_api;
mod progress;

use std::path::{Path, PathBuf};

use hf_api::{HfClient, ModelInfo};

const COMPANION_FILES: &[&str] = &[
    "tokenizer.json",
    "config.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "special_tokens_map.json",
];

// ── CLI args ─────────────────────────────────────────────────────────────────

#[derive(clap::Args)]
pub struct DownloadArgs {
    /// HuggingFace model identifier (e.g., meta-llama/Llama-3.2-1B).
    pub model_id: String,

    /// Output directory where model files will be saved [default: ./models/<model-name>].
    #[arg(short, long, value_name = "DIR")]
    pub output: Option<PathBuf>,

    /// Preferred format.
    #[arg(short, long, default_value = "auto")]
    pub format: DownloadFormat,

    /// Git revision on HF Hub.
    #[arg(long, default_value = "main")]
    pub revision: String,

    /// Quantization variant (e.g., Q4_0, Q4_K_M).
    #[arg(long)]
    pub quantization: Option<String>,

    /// Keep Python virtualenv after conversion.
    #[arg(long)]
    pub keep_venv: bool,

    /// HuggingFace API token (or set HF_TOKEN env var).
    #[arg(long)]
    pub token: Option<String>,
}

#[derive(Clone, clap::ValueEnum)]
pub enum DownloadFormat {
    Auto,
    Onnx,
}

// ── Format resolution ────────────────────────────────────────────────────────

enum ResolvedDownload {
    Onnx { filenames: Vec<String> },
    ConvertToOnnx,
    ConvertDiffusionToOnnx,
}

fn resolve_format(
    info: &ModelInfo,
    format: &DownloadFormat,
    _quantization: Option<&str>,
) -> ResolvedDownload {
    match format {
        DownloadFormat::Onnx | DownloadFormat::Auto => {
            if let Some(r) = try_resolve_onnx(info) {
                r
            } else if is_diffusion_pipeline(info) {
                ResolvedDownload::ConvertDiffusionToOnnx
            } else {
                ResolvedDownload::ConvertToOnnx
            }
        }
    }
}

/// Detect diffusion pipelines by looking for `model_index.json` — the marker
/// file that `diffusers` uses to describe a multi-component pipeline.
fn is_diffusion_pipeline(info: &ModelInfo) -> bool {
    info.siblings
        .iter()
        .any(|f| f.filename == "model_index.json")
}

fn try_resolve_onnx(info: &ModelInfo) -> Option<ResolvedDownload> {
    let onnx_files: Vec<String> = info
        .siblings
        .iter()
        .filter(|f| f.filename.ends_with(".onnx"))
        .map(|f| f.filename.clone())
        .collect();

    if onnx_files.is_empty() {
        return None;
    }

    Some(ResolvedDownload::Onnx {
        filenames: onnx_files,
    })
}

// ── Entrypoint ───────────────────────────────────────────────────────────────

pub fn run(args: DownloadArgs) -> anyhow::Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(run_async(args))
}

async fn run_async(args: DownloadArgs) -> anyhow::Result<()> {
    let client = HfClient::new(args.token.clone());

    eprintln!("Fetching model info for {}...", args.model_id);
    let info = client.model_info(&args.model_id).await?;

    let model_name = args
        .model_id
        .split('/')
        .next_back()
        .unwrap_or(&args.model_id);
    let output_dir = args
        .output
        .clone()
        .unwrap_or_else(|| PathBuf::from("models").join(model_name));
    std::fs::create_dir_all(&output_dir)?;

    let resolved = resolve_format(&info, &args.format, args.quantization.as_deref());

    match resolved {
        ResolvedDownload::Onnx { filenames } => {
            download_onnx(
                &client,
                &args.model_id,
                &args.revision,
                &filenames,
                &output_dir,
                &info,
            )
            .await?;
        }
        ResolvedDownload::ConvertToOnnx => {
            eprintln!("No pre-built ONNX found. Converting via Python...");
            let result = convert::convert_to_onnx(&args.model_id, &output_dir, args.keep_venv)?;
            eprintln!("Converted: {}", result.model_path.display());
        }
        ResolvedDownload::ConvertDiffusionToOnnx => {
            eprintln!("Diffusion pipeline detected. Exporting components to ONNX via optimum...");
            let result =
                convert::convert_diffusion_to_onnx(&args.model_id, &output_dir, args.keep_venv)?;
            eprintln!("Exported: {}", result.model_path.display());
            for f in &result.companion_files {
                eprintln!("  component: {}", f.display());
            }
        }
    }

    print_summary(&args.model_id, &output_dir)?;
    Ok(())
}

// ── Download helpers ─────────────────────────────────────────────────────────

async fn download_onnx(
    client: &HfClient,
    model_id: &str,
    revision: &str,
    filenames: &[String],
    output_dir: &Path,
    info: &ModelInfo,
) -> anyhow::Result<()> {
    let dp = progress::DownloadProgress::new();

    for filename in filenames {
        let dest = output_dir.join(filename);
        let file_info = info.siblings.iter().find(|f| &f.filename == filename);
        let total = file_info.and_then(|f| f.size).unwrap_or(0);
        let pb = dp.add_file(filename, total);
        client
            .download_file(model_id, filename, revision, &dest, &pb)
            .await?;

        if let Some(lfs) = file_info.and_then(|f| f.lfs.as_ref()) {
            verify_checksum(&dest, &lfs.sha256)?;
        }
    }

    // Also download any external data files (e.g., model.onnx_data)
    for sibling in &info.siblings {
        if sibling.filename.ends_with(".onnx_data") || sibling.filename.ends_with(".onnx.data") {
            let dest = output_dir.join(&sibling.filename);
            let total = sibling.size.unwrap_or(0);
            let pb = dp.add_file(&sibling.filename, total);
            client
                .download_file(model_id, &sibling.filename, revision, &dest, &pb)
                .await?;
        }
    }

    download_companions(client, model_id, revision, output_dir, info, &dp).await?;
    Ok(())
}

async fn download_companions(
    client: &HfClient,
    model_id: &str,
    revision: &str,
    output_dir: &Path,
    info: &ModelInfo,
    dp: &progress::DownloadProgress,
) -> anyhow::Result<()> {
    for companion in COMPANION_FILES {
        if info.siblings.iter().any(|f| f.filename == *companion) {
            let dest = output_dir.join(companion);
            let file_info = info.siblings.iter().find(|f| f.filename == *companion);
            let total = file_info.and_then(|f| f.size).unwrap_or(0);
            let pb = dp.add_file(companion, total);
            client
                .download_file(model_id, companion, revision, &dest, &pb)
                .await?;
        }
    }
    Ok(())
}

// ── Checksum verification ────────────────────────────────────────────────────

fn verify_checksum(path: &Path, expected_sha256: &str) -> anyhow::Result<()> {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    let mut file = std::fs::File::open(path)?;
    std::io::copy(&mut file, &mut hasher)?;
    let hash = format!("{:x}", hasher.finalize());
    if hash != expected_sha256 {
        anyhow::bail!(
            "Checksum mismatch for {}: expected {expected_sha256}, got {hash}",
            path.display()
        );
    }
    Ok(())
}

// ── Summary ──────────────────────────────────────────────────────────────────

fn print_summary(model_id: &str, output_dir: &Path) -> anyhow::Result<()> {
    eprintln!();
    eprintln!("Downloaded: {model_id}");
    eprintln!("Location:   {}", output_dir.display());
    eprintln!("Files:");
    for entry in std::fs::read_dir(output_dir)? {
        let entry = entry?;
        let meta = entry.metadata()?;
        let size = meta.len();
        let name = entry.file_name().to_string_lossy().to_string();
        eprintln!("  {name:<40} {}", format_size(size));
    }
    Ok(())
}

fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;
    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use hf_api::FileInfo;

    fn mock_info(filenames: &[&str]) -> ModelInfo {
        ModelInfo {
            id: "test/model".into(),
            siblings: filenames
                .iter()
                .map(|f| FileInfo {
                    filename: f.to_string(),
                    size: Some(1000),
                    lfs: None,
                })
                .collect(),
            private: false,
            gated: None,
        }
    }

    #[test]
    fn resolve_falls_back_to_onnx() {
        let info = mock_info(&["model.onnx"]);
        let resolved = resolve_format(&info, &DownloadFormat::Auto, None);
        assert!(matches!(resolved, ResolvedDownload::Onnx { .. }));
    }

    #[test]
    fn resolve_triggers_conversion_when_no_formats() {
        let info = mock_info(&["pytorch_model.bin"]);
        let resolved = resolve_format(&info, &DownloadFormat::Auto, None);
        assert!(matches!(resolved, ResolvedDownload::ConvertToOnnx));
    }

    #[test]
    fn resolve_explicit_onnx_with_no_onnx_triggers_convert() {
        let info = mock_info(&["pytorch_model.bin"]);
        let resolved = resolve_format(&info, &DownloadFormat::Onnx, None);
        assert!(matches!(resolved, ResolvedDownload::ConvertToOnnx));
    }

    #[test]
    fn format_size_display() {
        assert_eq!(format_size(500), "500 B");
        assert_eq!(format_size(1536), "1.5 KB");
        assert_eq!(format_size(1_048_576), "1.0 MB");
        assert_eq!(format_size(1_610_612_736), "1.5 GB");
    }
}
