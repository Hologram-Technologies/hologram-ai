//! Download ONNX models from Hugging Face.
//!
//! This module provides functionality to download ONNX models from Hugging Face
//! Model Hub, with progress indication. Supports recursive directory scanning
//! to find ONNX files in subdirectories (e.g., for Stable Diffusion models).
//!
//! ## Authentication
//!
//! For gated/private models, set one of these environment variables:
//! - `HF_TOKEN` - HuggingFace access token
//! - `HUGGING_FACE_HUB_TOKEN` - Alternative token variable
//!
//! Get your token from: https://huggingface.co/settings/tokens

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::blocking::Client;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION};
use serde::Deserialize;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use tracing::{debug, info, warn};

/// File information from Hugging Face API
#[derive(Debug, Deserialize, Clone)]
struct FileInfo {
    /// File or directory path
    #[serde(rename = "path")]
    path: String,

    /// File size in bytes (None for directories)
    #[serde(rename = "size")]
    size: Option<u64>,

    /// Entry type: "file" or "directory"
    #[serde(rename = "type", default)]
    entry_type: String,
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
/// ```ignore
/// use std::path::Path;
/// use anyhow::Result;
/// fn main() -> Result<()> {
///     hologram_ai::cli::download_command(
///         "CompVis/stable-diffusion-v1-4",
///         Path::new("./models"),
///         Some("main"),
///     )?;
///     Ok(())
/// }
/// ```
/// Get HuggingFace token from environment variables.
fn get_hf_token() -> Option<String> {
    std::env::var("HF_TOKEN")
        .ok()
        .or_else(|| std::env::var("HUGGING_FACE_HUB_TOKEN").ok())
}

/// Download a model from HuggingFace Hub.
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

    // Check for authentication token
    let hf_token = get_hf_token();
    if hf_token.is_some() {
        info!("Using HuggingFace authentication token");
    } else {
        debug!("No HF_TOKEN or HUGGING_FACE_HUB_TOKEN found - proceeding without authentication");
    }

    // Create HTTP client with optional auth header
    let mut headers = HeaderMap::new();
    if let Some(ref token) = hf_token {
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", token))
                .context("Invalid HuggingFace token format")?,
        );
    }

    let client = Client::builder()
        .user_agent("hologram-onnx-cli/0.1.0")
        .default_headers(headers)
        .build()
        .context("Failed to create HTTP client")?;

    info!("Fetching file list from Hugging Face (scanning subdirectories)...");

    // Recursively fetch all files including subdirectories
    let all_files = fetch_files_recursive(&client, model_id, revision, "")?;

    // Filter for ONNX files and their external data
    let onnx_files: Vec<&FileInfo> = all_files
        .iter()
        .filter(|f| {
            f.path.ends_with(".onnx") ||
            f.path.ends_with(".onnx_data") ||
            f.path.ends_with(".pb") ||
            f.path.ends_with(".bin") && f.path.contains("model")
        })
        .collect();

    if onnx_files.is_empty() {
        warn!("No ONNX files found in model repository: {}", model_id);
        info!("Available files (showing first 20):");
        for file in all_files.iter().take(20) {
            let type_marker = if file.entry_type == "directory" {
                "📁"
            } else {
                "📄"
            };
            info!("  {} {}", type_marker, file.path);
        }
        if all_files.len() > 20 {
            info!("  ... and {} more files", all_files.len() - 20);
        }
        anyhow::bail!("No ONNX files found in repository");
    }

    info!("Found {} ONNX file(s):", onnx_files.len());
    let mut total_size: u64 = 0;
    for file in &onnx_files {
        let size_str = file
            .size
            .map(|s| {
                total_size += s;
                format_size(s)
            })
            .unwrap_or_else(|| "unknown size".to_string());
        info!("  - {} ({})", file.path, size_str);
    }
    if total_size > 0 {
        info!("Total download size: {}", format_size(total_size));
    }

    // Download each ONNX file
    for (idx, file) in onnx_files.iter().enumerate() {
        info!(
            "[{}/{}] Downloading: {}",
            idx + 1,
            onnx_files.len(),
            file.path
        );
        download_file(&client, model_id, revision, &file.path, output_dir, file.size)?;
    }

    info!("✓ Download complete!");
    info!("  Files saved to: {}", output_dir.display());

    Ok(())
}

/// Format file size in human-readable form
fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}

/// Recursively fetch all files from a HuggingFace repository
fn fetch_files_recursive(
    client: &Client,
    model_id: &str,
    revision: &str,
    path: &str,
) -> Result<Vec<FileInfo>> {
    let api_url = if path.is_empty() {
        format!(
            "https://huggingface.co/api/models/{}/tree/{}",
            model_id, revision
        )
    } else {
        format!(
            "https://huggingface.co/api/models/{}/tree/{}/{}",
            model_id, revision, path
        )
    };

    debug!("Fetching: {}", api_url);

    let response = client
        .get(&api_url)
        .send()
        .with_context(|| format!("Failed to fetch file list from: {}", api_url))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response
            .text()
            .unwrap_or_else(|_| "Unknown error".to_string());

        if status == reqwest::StatusCode::UNAUTHORIZED {
            anyhow::bail!(
                "Authentication required for model: {}\n\
                 Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN environment variable.\n\
                 Get your token from: https://huggingface.co/settings/tokens\n\
                 \n\
                 Example: HF_TOKEN=hf_xxx cargo run -p hologram-onnx-cli -- download ...\n\
                 \n\
                 Original error: HTTP {} - {}",
                model_id,
                status,
                body
            );
        }

        anyhow::bail!(
            "Failed to fetch model info: HTTP {} - {}",
            status,
            body
        );
    }

    let entries: Vec<FileInfo> = response
        .json()
        .context("Failed to parse Hugging Face API response")?;

    let mut all_files = Vec::new();

    for entry in entries {
        if entry.entry_type == "directory" {
            // Recursively fetch files from subdirectory
            debug!("Scanning subdirectory: {}", entry.path);
            let subdir_files = fetch_files_recursive(client, model_id, revision, &entry.path)?;
            all_files.extend(subdir_files);
        } else {
            all_files.push(entry);
        }
    }

    Ok(all_files)
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
