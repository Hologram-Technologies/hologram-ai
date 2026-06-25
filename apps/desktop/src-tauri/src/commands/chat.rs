//! Chat / generation commands.
//!
//! `generate` shells out to `hologram-ai run --prompt ...` and streams every
//! stdout/stderr line as a `chat://line` event. The frontend reassembles a
//! transcript from those lines. `cancel_generation` kills the active child.

use std::path::PathBuf;

use serde::Deserialize;
use tauri::{AppHandle, State};
use tokio::sync::oneshot;

use crate::paths;
use crate::state::AppState;

use super::process_runner;

#[derive(Deserialize)]
pub struct GenerateRequest {
    pub archive: PathBuf,
    pub prompt: String,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub stop: Vec<String>,
    /// HuggingFace `tokenizer.json`. The `.holo` carries no tokenizer (closed
    /// section set), so generation needs it at run time. Defaults to
    /// `tokenizer.json` beside the archive when omitted.
    #[serde(default)]
    pub tokenizer: Option<PathBuf>,
    /// Prompt template with a `{prompt}` placeholder (chat template).
    #[serde(default)]
    pub prompt_template: Option<String>,
}

#[tauri::command]
pub async fn generate(
    app: AppHandle,
    state: State<'_, AppState>,
    req: GenerateRequest,
) -> Result<i32, String> {
    let bin = paths::hologram_ai_bin().map_err(|e| e.to_string())?;
    let cwd = paths::workspace_root();

    // Pass the model *directory* (parent of the `.holo`) instead of the
    // archive itself. When the CLI sees a directory it uses the growable
    // session (window starts at 64 and grows as needed) rather than a fixed
    // 2048-token window, which is dramatically faster for short prompts.
    // The directory contains `tokenizer.json` and the ONNX source, so the
    // CLI auto-discovers everything it needs.
    let model_path = req
        .archive
        .parent()
        .filter(|p| p.join("tokenizer.json").exists())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| req.archive.clone());
    let mut args = vec![
        "run".to_string(),
        model_path.to_string_lossy().into_owned(),
        "--no-echo".into(),
        "--prompt".into(),
        req.prompt,
    ];
    if let Some(t) = &req.prompt_template {
        args.push("--prompt-template".into());
        args.push(t.clone());
    }
    if let Some(n) = req.max_tokens {
        args.push("--max-tokens".into());
        args.push(n.to_string());
    }
    if let Some(t) = req.temperature {
        args.push("--temperature".into());
        args.push(t.to_string());
    }
    if let Some(k) = req.top_k {
        args.push("--top-k".into());
        args.push(k.to_string());
    }
    for s in req.stop {
        args.push("--stop".into());
        args.push(s);
    }

    let (cancel_tx, cancel_rx) = oneshot::channel();
    {
        let mut slot = state.active_generation.lock();
        if let Some(prev) = slot.take() {
            let _ = prev.send(());
        }
        *slot = Some(cancel_tx);
    }

    let logs = state.logs.clone();
    let result =
        process_runner::spawn_streaming(app, bin, args, cwd, logs, "chat://line", cancel_rx)
            .await
            .map_err(|e| e.to_string());

    state.active_generation.lock().take();
    result
}

#[tauri::command]
pub fn cancel_generation(state: State<'_, AppState>) -> bool {
    if let Some(tx) = state.active_generation.lock().take() {
        let _ = tx.send(());
        true
    } else {
        false
    }
}
