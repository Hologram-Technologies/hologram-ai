//! Path resolution helpers — locate the hologram-ai CLI binary, the workspace
//! root, and the conventional `models/` and `output/` directories.

use std::path::{Path, PathBuf};

/// Workspace root: the directory two levels above this crate (apps/desktop/src-tauri → repo root).
pub fn workspace_root() -> PathBuf {
    let here = Path::new(env!("CARGO_MANIFEST_DIR"));
    here.ancestors()
        .nth(3)
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| here.to_path_buf())
}

pub fn models_dir() -> PathBuf {
    workspace_root().join("models")
}

pub fn output_dir() -> PathBuf {
    workspace_root().join("output")
}

/// Resolve the `hologram-ai` binary. Preference order:
///   1. `HOLOGRAM_AI_BIN` env var override
///   2. Sidecar shipped alongside the desktop binary (the production path —
///      Tauri places `externalBin` files in the same dir as the main exe)
///   3. `<workspace>/target/release/hologram-ai` (dev)
///   4. `<workspace>/target/debug/hologram-ai` (dev)
///   5. anything on `PATH`
pub fn hologram_ai_bin() -> anyhow::Result<PathBuf> {
    if let Ok(p) = std::env::var("HOLOGRAM_AI_BIN") {
        let path = PathBuf::from(p);
        if path.exists() {
            return Ok(path);
        }
    }
    if let Some(p) = sidecar_bin() {
        return Ok(p);
    }
    let release = workspace_root().join("target/release/hologram-ai");
    if release.exists() {
        return Ok(release);
    }
    let debug = workspace_root().join("target/debug/hologram-ai");
    if debug.exists() {
        return Ok(debug);
    }
    which::which("hologram-ai").map_err(|e| {
        anyhow::anyhow!(
            "hologram-ai binary not found (tried env, sidecar, target/, PATH): {e}. \
             Build it with `cargo build --release -p hologram-ai`."
        )
    })
}

/// Look for the bundled CLI sidecar next to the running desktop binary.
///
/// Tauri's `externalBin` configuration places sidecars alongside the main
/// executable in every bundle target (macOS `.app/Contents/MacOS/`, Linux
/// AppImage/`/usr/bin/`, Windows next to the `.exe`). We resolve via
/// `current_exe()` rather than the Tauri sidecar API to keep this helper
/// usable from any context (no `AppHandle` needed).
fn sidecar_bin() -> Option<PathBuf> {
    let our_exe = std::env::current_exe().ok()?;
    let dir = our_exe.parent()?;
    let name = if cfg!(windows) {
        "hologram-ai.exe"
    } else {
        "hologram-ai"
    };
    let candidate = dir.join(name);
    if candidate.exists() && candidate != our_exe {
        Some(candidate)
    } else {
        None
    }
}
