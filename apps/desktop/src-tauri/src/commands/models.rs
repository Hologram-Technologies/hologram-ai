//! Model management commands: known-model listing, download, and compile.
//!
//! All operations target the curated catalogue in `known_models::CATALOGUE`.
//! Freeform HuggingFace ids are intentionally not exposed to the UI — only
//! models that have been verified to work end-to-end appear here.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, State};
use tokio::sync::oneshot;

use crate::known_models::{self, KnownModelStatus};
use crate::paths;
use crate::state::AppState;

use super::process_runner;

#[derive(Serialize)]
pub struct WorkspacePaths {
    pub root: PathBuf,
    pub models_dir: PathBuf,
    pub output_dir: PathBuf,
    pub hologram_ai_bin: Option<PathBuf>,
}

#[tauri::command]
pub fn workspace_paths() -> WorkspacePaths {
    WorkspacePaths {
        root: paths::workspace_root(),
        models_dir: paths::models_dir(),
        output_dir: paths::output_dir(),
        hologram_ai_bin: paths::hologram_ai_bin().ok(),
    }
}

#[tauri::command]
pub async fn list_known_models() -> Vec<KnownModelStatus> {
    known_models::list_with_status().await
}

#[derive(Serialize)]
pub struct CompiledArchive {
    pub name: String,
    pub path: PathBuf,
    pub size_bytes: u64,
}

#[tauri::command]
pub async fn list_compiled_archives() -> Result<Vec<CompiledArchive>, String> {
    let mut out = Vec::new();
    let mut stack = vec![paths::output_dir(), paths::models_dir()];
    while let Some(d) = stack.pop() {
        let mut rd = match tokio::fs::read_dir(&d).await {
            Ok(rd) => rd,
            Err(_) => continue,
        };
        while let Ok(Some(entry)) = rd.next_entry().await {
            let p = entry.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().and_then(|e| e.to_str()) == Some("holo") {
                let size = entry.metadata().await.map(|m| m.len()).unwrap_or(0);
                out.push(CompiledArchive {
                    name: p.file_name().and_then(|s| s.to_str()).unwrap_or("?").to_string(),
                    path: p,
                    size_bytes: size,
                });
            }
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

#[derive(Deserialize)]
pub struct ByIdRequest {
    pub id: String,
}

#[tauri::command]
pub async fn download_known_model(
    app: AppHandle,
    state: State<'_, AppState>,
    req: ByIdRequest,
) -> Result<i32, String> {
    let model = known_models::CATALOGUE
        .iter()
        .find(|m| m.id == req.id)
        .ok_or_else(|| format!("unknown model id: {}", req.id))?;

    let bin = paths::hologram_ai_bin().map_err(|e| e.to_string())?;
    let cwd = paths::workspace_root();
    let args = vec!["download".to_string(), model.hf_id.to_string()];

    let (_tx, rx) = oneshot::channel();
    process_runner::spawn_streaming(
        app,
        bin,
        args,
        cwd,
        state.logs.clone(),
        "models://download-line",
        rx,
    )
    .await
    .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn compile_known_model(
    app: AppHandle,
    state: State<'_, AppState>,
    req: ByIdRequest,
) -> Result<i32, String> {
    let model = known_models::CATALOGUE
        .iter()
        .find(|m| m.id == req.id)
        .ok_or_else(|| format!("unknown model id: {}", req.id))?;

    // Locate the downloaded ONNX file. The CLI's download command lays the
    // model out under `models/<repo-name>/...` with `model.onnx` at the top
    // level for converted/auto-resolved models.
    let local_name = model.hf_id.split('/').next_back().unwrap_or(model.hf_id);
    let model_dir = paths::models_dir().join(local_name);
    let onnx = find_first_onnx(&model_dir).await.ok_or_else(|| {
        format!(
            "no .onnx file found under {}; download the model first",
            model_dir.display()
        )
    })?;

    let bin = paths::hologram_ai_bin().map_err(|e| e.to_string())?;
    let cwd = paths::workspace_root();
    let mut args = vec![
        "compile".to_string(),
        "--model".into(),
        onnx.to_string_lossy().into_owned(),
        "--output".into(),
        model_dir.to_string_lossy().into_owned(),
        "--name".into(),
        model.id.to_string(),
        "--quantize".into(),
        model.quantize.to_string(),
    ];
    // Pass the prompt template through to the archive so the runtime can
    // apply it without the desktop UI needing to know the format.
    if let Some(template) = model.prompt_template {
        args.push("--prompt-template".into());
        args.push(template.to_string());
    }
    for s in model.stop {
        args.push("--stop".into());
        args.push((*s).to_string());
    }

    let (_tx, rx) = oneshot::channel();
    process_runner::spawn_streaming(
        app,
        bin,
        args,
        cwd,
        state.logs.clone(),
        "models://compile-line",
        rx,
    )
    .await
    .map_err(|e| e.to_string())
}

async fn find_first_onnx(dir: &std::path::Path) -> Option<PathBuf> {
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
            } else if p.extension().and_then(|e| e.to_str()) == Some("onnx") {
                return Some(p);
            }
        }
    }
    None
}
