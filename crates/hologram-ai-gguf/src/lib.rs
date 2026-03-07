//! GGUF importer — Phase 2 (deferred; ONNX is the priority importer in Sprint 001).
//!
//! The GGUF importer will be implemented in Sprint 002, targeting:
//! - LLaMA architecture (Q4_0 weights)
//! - GGUF v1 / v2 / v3 binary format

use std::path::Path;
use hologram_ai_common::AiGraph;

/// Options for GGUF import.
pub struct GgufImportOptions {
    pub mmap: bool,
    pub arch_override: Option<String>,
}

impl Default for GgufImportOptions {
    fn default() -> Self { Self { mmap: true, arch_override: None } }
}

/// Import a GGUF model file.
///
/// **Not yet implemented** — returns an error. Implemented in Sprint 002.
pub fn import_gguf(_path: &Path, _opts: GgufImportOptions) -> anyhow::Result<AiGraph> {
    anyhow::bail!("GGUF importer not yet implemented (Sprint 002)")
}

/// Import a GGUF model from bytes.
///
/// **Not yet implemented** — returns an error. Implemented in Sprint 002.
pub fn import_gguf_bytes(_bytes: &[u8], _opts: GgufImportOptions) -> anyhow::Result<AiGraph> {
    anyhow::bail!("GGUF importer not yet implemented (Sprint 002)")
}
