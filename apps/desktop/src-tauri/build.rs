use std::path::Path;

fn main() {
    ensure_sidecar_placeholder();
    tauri_build::build()
}

/// Drop a no-op placeholder for the `hologram-ai` sidecar if the real
/// binary hasn't been staged yet. Tauri-build refuses to compile when an
/// `externalBin` entry is missing, which would otherwise break `cargo
/// check` / `clippy -D warnings` for anyone working on the desktop crate
/// without first running `cargo build -p hologram-ai`.
///
/// CI stages the real binaries before invoking `cargo build`, so the
/// `if !path.exists()` guard means the real binary is never overwritten.
fn ensure_sidecar_placeholder() {
    let target = match std::env::var("TARGET") {
        Ok(t) if !t.is_empty() => t,
        _ => return,
    };
    let dir = Path::new("binaries");
    if !dir.exists() {
        if let Err(e) = std::fs::create_dir_all(dir) {
            println!("cargo:warning=create binaries/: {e}");
            return;
        }
    }
    let suffix = if target.contains("windows") { ".exe" } else { "" };
    let path = dir.join(format!("hologram-ai-{target}{suffix}"));
    if path.exists() {
        return;
    }
    let stub = b"#!/bin/sh\necho 'hologram-ai sidecar placeholder - rebuild for production' >&2\nexit 127\n";
    if let Err(e) = std::fs::write(&path, stub) {
        println!("cargo:warning=write sidecar placeholder {}: {e}", path.display());
        return;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(meta) = std::fs::metadata(&path) {
            let mut perms = meta.permissions();
            perms.set_mode(0o755);
            let _ = std::fs::set_permissions(&path, perms);
        }
    }
    println!("cargo:warning=staged sidecar placeholder at {}", path.display());
}
