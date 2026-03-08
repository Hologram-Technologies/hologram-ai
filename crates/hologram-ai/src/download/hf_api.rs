use std::path::Path;
use std::time::Duration;

use futures::StreamExt;
use indicatif::ProgressBar;
use reqwest::Client;
use serde::{Deserialize, Deserializer};
use tokio::io::AsyncWriteExt;

pub struct HfClient {
    client: Client,
    token: Option<String>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
pub struct ModelInfo {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub siblings: Vec<FileInfo>,
    #[serde(default)]
    pub private: bool,
    #[serde(default, deserialize_with = "deserialize_gated")]
    pub gated: Option<String>,
}

#[derive(Deserialize, Clone)]
pub struct FileInfo {
    #[serde(rename = "rfilename")]
    pub filename: String,
    pub size: Option<u64>,
    pub lfs: Option<LfsInfo>,
}

#[derive(Deserialize, Clone)]
#[allow(dead_code)]
pub struct LfsInfo {
    pub sha256: String,
    pub size: u64,
}

impl HfClient {
    pub fn new(token: Option<String>) -> Self {
        let token = token
            .or_else(|| std::env::var("HF_TOKEN").ok())
            .or_else(read_hf_token_file);
        HfClient {
            client: Client::new(),
            token,
        }
    }

    pub async fn model_info(&self, model_id: &str) -> anyhow::Result<ModelInfo> {
        let url = format!("https://huggingface.co/api/models/{model_id}");
        let mut req = self.client.get(&url);
        if let Some(ref token) = self.token {
            req = req.bearer_auth(token);
        }
        let resp = req.send().await?;

        match resp.status().as_u16() {
            401 | 403 => anyhow::bail!(
                "This model requires authentication. Use --token or set HF_TOKEN env var."
            ),
            404 => anyhow::bail!("Model not found: {model_id}"),
            s if s >= 400 => anyhow::bail!("HuggingFace API error: HTTP {s}"),
            _ => {}
        }

        Ok(resp.json().await?)
    }

    pub async fn download_file(
        &self,
        model_id: &str,
        filename: &str,
        revision: &str,
        dest: &Path,
        progress: &ProgressBar,
    ) -> anyhow::Result<()> {
        let mut attempts = 0u32;
        loop {
            attempts += 1;
            match self
                .download_file_once(model_id, filename, revision, dest, progress)
                .await
            {
                Ok(()) => return Ok(()),
                Err(e) if attempts < 3 => {
                    let delay = Duration::from_secs(1 << attempts);
                    eprintln!(
                        "Download failed (attempt {attempts}/3): {e}. Retrying in {delay:?}..."
                    );
                    tokio::time::sleep(delay).await;
                    progress.reset();
                }
                Err(e) => return Err(e),
            }
        }
    }

    async fn download_file_once(
        &self,
        model_id: &str,
        filename: &str,
        revision: &str,
        dest: &Path,
        progress: &ProgressBar,
    ) -> anyhow::Result<()> {
        let url = format!("https://huggingface.co/{model_id}/resolve/{revision}/{filename}");
        let mut req = self.client.get(&url);
        if let Some(ref token) = self.token {
            req = req.bearer_auth(token);
        }

        let resp = req.send().await?.error_for_status()?;
        let total = resp.content_length().unwrap_or(0);
        if total > 0 {
            progress.set_length(total);
        }

        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut file = tokio::fs::File::create(dest).await?;
        let mut stream = resp.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            file.write_all(&chunk).await?;
            progress.inc(chunk.len() as u64);
        }
        file.flush().await?;
        progress.finish();
        Ok(())
    }
}

/// HF API returns `false` (bool) for non-gated models, `"auto"` or `"manual"` (string) for gated.
fn deserialize_gated<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let value: serde_json::Value = Deserialize::deserialize(deserializer)?;
    match value {
        serde_json::Value::String(s) => Ok(Some(s)),
        serde_json::Value::Bool(false) | serde_json::Value::Null => Ok(None),
        serde_json::Value::Bool(true) => Ok(Some("true".to_string())),
        _ => Ok(None),
    }
}

fn read_hf_token_file() -> Option<String> {
    let path = dirs::cache_dir()?.join("huggingface/token");
    std::fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}
