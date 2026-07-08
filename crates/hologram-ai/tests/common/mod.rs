//! Shared test utilities for hologram-ai integration tests.

/// Parametric decoder-family definitions (config + manifest + weights) shared by
/// the family-coverage decode test and the parametric family memory sweep.
pub mod families;

use std::path::PathBuf;

/// Resolve a path relative to the workspace root.
#[allow(dead_code)]
pub fn workspace_path(rel: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // crates/hologram-ai → crates/
    p.pop(); // crates/ → workspace root
    p.push(rel);
    p
}

/// Parse a byte slice as little-endian `f32` values.
#[allow(dead_code)]
pub fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("chunk is exactly 4 bytes")))
        .collect()
}
