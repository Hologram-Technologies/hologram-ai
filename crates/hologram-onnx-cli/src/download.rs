//! Download ONNX models from Hugging Face.
//!
//! This module provides functionality to download ONNX models from Hugging Face
//! Model Hub, with progress indication.

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::blocking::Client;
use serde::Deserialize;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use tracing::{debug, info, warn};

/// File information from Hugging Face API
#[derive(Debug, Deserialize)]
struct FileInfo {
    #[serde(rename = "path")]
    path: String,

    #[serde(rename = "size")]
    size: Option<u64>,
}

/// Download an ONNX model from Hugging Face.
///
/// # Arguments
///
/// * `model_id` - Hugging Face model ID (e.g., "stable-diffusion-v1-5")
/// * `output_dir` - Directory to save downloaded files
/// * `revision` - Optional git revision/branch (defaults to "main")
///
/// # Returns
///
/// Returns Ok(()) on success, or an error if download fails.
///
/// # Example
///
/// ```no_run
/// use std::path::Path;
/// # use anyhow::Result;
/// # fn main() -> Result<()> {
/// hologram_onnx_cli::download::download_command(
///     "CompVis/stable-diffusion-v1-4",
///     Path::new("./models"),
///     Some("main"),
/// )?;
/// # Ok(())
/// # }
/// ```
pub fn download_command(model_id: &str, output_dir: &Path, revision: Option<&str>) -> Result<()> {
    let revision = revision.unwrap_or("main");

    info!("Downloading model: {}", model_id);
    info!("Revision: {}", revision);
    info!("Output directory: {}", output_dir.display());

    // Create output directory
    fs::create_dir_all(output_dir).with_context(|| {
        format!(
            "Failed to create output directory: {}",
            output_dir.display()
        )
    })?;

    // Create HTTP client
    let client = Client::builder()
        .user_agent("hologram-onnx-cli/0.1.0")
        .build()
        .context("Failed to create HTTP client")?;

    // Fetch file list from Hugging Face API
    let api_url = format!(
        "https://huggingface.co/api/models/{}/tree/{}",
        model_id, revision
    );

    info!("Fetching file list from Hugging Face...");
    debug!("API URL: {}", api_url);

    let response = client
        .get(&api_url)
        .send()
        .context("Failed to fetch model file list from Hugging Face")?;

    if !response.status().is_success() {
        anyhow::bail!(
            "Failed to fetch model info: HTTP {} - {}",
            response.status(),
            response
                .text()
                .unwrap_or_else(|_| "Unknown error".to_string())
        );
    }

    let files: Vec<FileInfo> = response
        .json()
        .context("Failed to parse Hugging Face API response")?;

    // Filter for ONNX files
    let onnx_files: Vec<&FileInfo> = files.iter().filter(|f| f.path.ends_with(".onnx")).collect();

    if onnx_files.is_empty() {
        warn!("No ONNX files found in model repository: {}", model_id);
        info!("Available files:");
        for file in files.iter().take(10) {
            info!("  - {}", file.path);
        }
        if files.len() > 10 {
            info!("  ... and {} more files", files.len() - 10);
        }
        anyhow::bail!("No ONNX files found in repository");
    }

    info!("Found {} ONNX file(s):", onnx_files.len());
    for file in &onnx_files {
        let size_str = file
            .size
            .map(|s| format!(" ({} bytes)", s))
            .unwrap_or_default();
        info!("  - {}{}", file.path, size_str);
    }

    // Download each ONNX file
    for file in onnx_files {
        download_file(
            &client, model_id, revision, &file.path, output_dir, file.size,
        )?;
    }

    info!("✓ Download complete!");
    info!("  Files saved to: {}", output_dir.display());

    Ok(())
}

/// Download a single file from Hugging Face
fn download_file(
    client: &Client,
    model_id: &str,
    revision: &str,
    file_path: &str,
    output_dir: &Path,
    file_size: Option<u64>,
) -> Result<()> {
    let download_url = format!(
        "https://huggingface.co/{}/resolve/{}/{}",
        model_id, revision, file_path
    );

    let output_path = output_dir.join(file_path);

    // Create parent directories
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create directory: {}", parent.display()))?;
    }

    info!("Downloading: {} → {}", file_path, output_path.display());
    debug!("URL: {}", download_url);

    // Start download
    let mut response = client
        .get(&download_url)
        .send()
        .with_context(|| format!("Failed to download file: {}", file_path))?;

    if !response.status().is_success() {
        anyhow::bail!(
            "Failed to download {}: HTTP {}",
            file_path,
            response.status()
        );
    }

    // Get content length
    let total_size = response.content_length().or(file_size).unwrap_or(0);

    // Create progress bar
    let pb = if total_size > 0 {
        let pb = ProgressBar::new(total_size);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
                .expect("Failed to create progress bar template")
                .progress_chars("#>-"),
        );
        Some(pb)
    } else {
        None
    };

    // Write file with progress tracking
    let mut file = File::create(&output_path)
        .with_context(|| format!("Failed to create file: {}", output_path.display()))?;

    let mut downloaded = 0u64;
    let mut buffer = vec![0; 8192];

    loop {
        use std::io::Read;
        let n = response
            .read(&mut buffer)
            .context("Failed to read from download stream")?;

        if n == 0 {
            break;
        }

        file.write_all(&buffer[..n])
            .with_context(|| format!("Failed to write to file: {}", output_path.display()))?;

        downloaded += n as u64;
        if let Some(ref pb) = pb {
            pb.set_position(downloaded);
        }
    }

    if let Some(pb) = pb {
        pb.finish_with_message(format!("Downloaded {}", file_path));
    }

    info!("✓ Downloaded: {} ({} bytes)", file_path, downloaded);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_info_deserialization() {
        let json = r#"{"path": "model.onnx", "size": 1024}"#;
        let info: FileInfo = serde_json::from_str(json).unwrap();
        assert_eq!(info.path, "model.onnx");
        assert_eq!(info.size, Some(1024));
    }

    #[test]
    fn test_file_info_without_size() {
        let json = r#"{"path": "model.onnx"}"#;
        let info: FileInfo = serde_json::from_str(json).unwrap();
        assert_eq!(info.path, "model.onnx");
        assert_eq!(info.size, None);
    }
}
