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
///
/// In dev builds, `tauri-build` copies the externalBin entry from
/// `binaries/hologram-ai-<triple>` to `target/debug/hologram-ai` next to
/// the desktop binary — and `build.rs` writes a shell-script placeholder
/// there when no real binary has been staged. We must skip that placeholder
/// so the lookup falls through to `target/release/hologram-ai`.
fn sidecar_bin() -> Option<PathBuf> {
    let our_exe = std::env::current_exe().ok()?;
    let dir = our_exe.parent()?;
    let name = if cfg!(windows) {
        "hologram-ai.exe"
    } else {
        "hologram-ai"
    };
    let candidate = dir.join(name);
    if !candidate.exists() || candidate == our_exe {
        return None;
    }
    if is_placeholder_stub(&candidate) {
        return None;
    }
    Some(candidate)
}

/// Returns true if `path` is the build-time placeholder stub written by
/// `build.rs`. The stub is a tiny shell script; a real CLI binary is a
/// platform executable (Mach-O / ELF / PE), never a `#!`-prefixed script.
fn is_placeholder_stub(path: &Path) -> bool {
    use std::io::Read;
    let Ok(mut f) = std::fs::File::open(path) else {
        return false;
    };
    let mut buf = [0u8; 2];
    matches!(f.read(&mut buf), Ok(2)) && &buf == b"#!"
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_path(stem: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("hologram-ai-paths-{stem}-{}", std::process::id()));
        p
    }

    #[test]
    fn detects_shell_script_stub() {
        let p = scratch_path("stub");
        std::fs::write(&p, b"#!/bin/sh\necho stub\nexit 127\n").expect("write stub");
        let detected = is_placeholder_stub(&p);
        let _ = std::fs::remove_file(&p);
        assert!(detected);
    }

    #[test]
    fn binary_is_not_a_stub() {
        let p = scratch_path("bin");
        // Mach-O magic for a 64-bit arm64 binary; any non-`#!` start works.
        std::fs::write(&p, [0xCFu8, 0xFA, 0xED, 0xFE, 0x0C, 0x00, 0x00, 0x01])
            .expect("write fake binary");
        let detected = is_placeholder_stub(&p);
        let _ = std::fs::remove_file(&p);
        assert!(!detected);
    }

    #[test]
    fn missing_file_is_not_a_stub() {
        assert!(!is_placeholder_stub(Path::new("/nonexistent/hologram-ai")));
    }
}
