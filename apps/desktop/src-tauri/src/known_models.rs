//! Curated list of models known to work end-to-end with `hologram-ai`.
//!
//! Each entry pairs a HuggingFace identifier with the compile settings the
//! existing examples (`examples/*.toml`) and validation runs have proven
//! correct. The desktop UI surfaces this list rather than a freeform
//! HuggingFace input, so the user only ever picks a model that actually
//! produces sensible output.

use std::path::PathBuf;

use serde::Serialize;

use crate::paths;

#[derive(Clone, Serialize)]
pub struct KnownModel {
    /// Stable id used by the UI as a key.
    pub id: &'static str,
    /// HuggingFace repo id passed to `hologram-ai download`.
    pub hf_id: &'static str,
    /// Display name for the UI.
    pub display_name: &'static str,
    /// One-line description.
    pub description: &'static str,
    /// Modality category — used to drive screen pickers.
    pub modality: Modality,
    /// Approximate parameter count (e.g. "1.1B").
    pub size: &'static str,
    /// Approximate `.holo` archive size after compile, for the "you'll need
    /// ~X GB free" hint.
    pub approx_archive_mb: u32,
    /// Quantization scheme passed to `--quantize` (one of `none`, `int8`, `int4`).
    pub quantize: &'static str,
    /// Optional prompt-template suggestion (Jinja-style or plain).
    pub prompt_template: Option<&'static str>,
    /// Optional default stop strings for generation.
    pub stop: &'static [&'static str],
    /// Multi-turn turn-separator: text inserted *between* prior user/assistant
    /// pairs when the desktop UI builds a multi-turn `{prompt}` slot. The
    /// literal `{response}` placeholder is substituted with the prior
    /// assistant message. The `prompt_template`'s prefix/suffix wrap the
    /// outer turn unchanged. `None` means single-turn only.
    pub chat_turn_separator: Option<&'static str>,
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Modality {
    TextChat,
}

/// Static catalogue. Add a new entry only after you've verified the full
/// download → compile → run loop succeeds.
pub const CATALOGUE: &[KnownModel] = &[
    KnownModel {
        id: "tinyllama-1.1b-chat",
        hf_id: "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        display_name: "TinyLlama 1.1B Chat",
        description: "Lightweight chat model — fastest path to a working demo.",
        modality: Modality::TextChat,
        size: "1.1B",
        approx_archive_mb: 700,
        quantize: "none",
        prompt_template: Some("<|user|>\n{prompt}</s>\n<|assistant|>\n"),
        stop: &["</s>"],
        chat_turn_separator: Some("</s>\n<|assistant|>\n{response}</s>\n<|user|>\n"),
    },
    KnownModel {
        id: "qwen2.5-0.5b-instruct",
        // onnx-community publishes the ONNX export; the official Qwen
        // org repo ships PyTorch/safetensors only. Qwen2 was removed
        // from the Hub (RepoNotFound), Qwen2.5 is the in-family
        // successor at the same 0.5B scale with the same ChatML
        // template.
        hf_id: "onnx-community/Qwen2.5-0.5B-Instruct",
        display_name: "Qwen2.5 0.5B Instruct",
        description: "Small chat-tuned model — follows instructions and answers questions.",
        modality: Modality::TextChat,
        size: "0.5B",
        approx_archive_mb: 350,
        quantize: "none",
        prompt_template: Some(
            "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        ),
        stop: &["<|im_end|>"],
        chat_turn_separator: Some(
            "<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>\n<|im_start|>user\n",
        ),
    },
];

#[derive(Serialize)]
pub struct KnownModelStatus {
    #[serde(flatten)]
    pub model: KnownModel,
    /// Local model directory if downloaded.
    pub local_dir: Option<PathBuf>,
    /// True if a `.onnx` file exists somewhere under `local_dir`.
    pub downloaded: bool,
    /// Path to a compiled `.holo` archive, if one exists in the conventional
    /// location (`models/<name>/*.holo` or `output/*.holo`).
    pub compiled_archive: Option<PathBuf>,
}

pub async fn list_with_status() -> Vec<KnownModelStatus> {
    let models_dir = paths::models_dir();
    let output_dir = paths::output_dir();

    let mut out = Vec::with_capacity(CATALOGUE.len());
    for m in CATALOGUE {
        // Conventional layout: `hologram-ai download org/model` puts files
        // under `models/<repo-name>/`. Strip the org prefix.
        let local_name = m.hf_id.split('/').next_back().unwrap_or(m.hf_id);
        let local_dir = models_dir.join(local_name);

        let downloaded = local_dir.exists() && find_extension(&local_dir, "onnx").await;

        // Migrate any pre-existing generic `model.holo` (from compiles done
        // before the CLI grew a `--name` flag) to the catalogue id, so the
        // Chat picker shows descriptive filenames.
        migrate_generic_archive(&local_dir, m.id).await;

        // Look for a compiled archive in two places: alongside the model
        // (the `examples/*.toml` convention) and in the workspace `output/`.
        let compiled_archive = first_archive_for(&local_dir, &output_dir, local_name, m.id).await;

        out.push(KnownModelStatus {
            model: m.clone(),
            local_dir: if local_dir.exists() {
                Some(local_dir)
            } else {
                None
            },
            downloaded,
            compiled_archive,
        });
    }
    out
}

/// Rename a generic `model.holo` to `<catalogue-id>.holo` if no descriptive
/// archive already exists. Idempotent: a second call is a no-op.
async fn migrate_generic_archive(local_dir: &std::path::Path, id: &str) {
    let generic = local_dir.join("model.holo");
    if !generic.exists() {
        return;
    }
    let descriptive = local_dir.join(format!("{id}.holo"));
    if descriptive.exists() {
        // Both exist — keep the descriptive one and leave the generic in
        // place. The user can delete it; we won't risk losing data.
        return;
    }
    if let Err(e) = tokio::fs::rename(&generic, &descriptive).await {
        tracing::warn!(
            "migrate {} → {} failed: {e}",
            generic.display(),
            descriptive.display(),
        );
    } else {
        tracing::info!("renamed {} → {}", generic.display(), descriptive.display(),);
    }
}

async fn find_extension(dir: &std::path::Path, ext: &str) -> bool {
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let mut rd = match tokio::fs::read_dir(&d).await {
            Ok(rd) => rd,
            Err(_) => continue,
        };
        while let Ok(Some(entry)) = rd.next_entry().await {
            let p = entry.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().and_then(|e| e.to_str()) == Some(ext) {
                return true;
            }
        }
    }
    false
}

async fn first_archive_for(
    local_dir: &std::path::Path,
    output_dir: &std::path::Path,
    local_name: &str,
    id: &str,
) -> Option<PathBuf> {
    // Prefer the catalogue-id archive (`<id>.holo`) above any other match —
    // ambiguity here would mean the wrong archive shows up in Chat.
    let preferred = local_dir.join(format!("{id}.holo"));
    if preferred.exists() {
        return Some(preferred);
    }
    if let Some(p) = first_holo_in(local_dir).await {
        return Some(p);
    }
    // Fall back to `output/` if the user compiled to the shared output dir
    // rather than the model dir.
    let mut rd = tokio::fs::read_dir(output_dir).await.ok()?;
    while let Ok(Some(entry)) = rd.next_entry().await {
        let p = entry.path();
        if p.extension().and_then(|e| e.to_str()) != Some("holo") {
            continue;
        }
        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        if stem == id || stem.contains(local_name) {
            return Some(p);
        }
    }
    None
}

async fn first_holo_in(dir: &std::path::Path) -> Option<PathBuf> {
    let mut rd = tokio::fs::read_dir(dir).await.ok()?;
    while let Ok(Some(entry)) = rd.next_entry().await {
        let p = entry.path();
        if p.extension().and_then(|e| e.to_str()) == Some("holo") {
            return Some(p);
        }
    }
    None
}
