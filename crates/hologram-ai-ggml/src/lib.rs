//! GGML importer — Phase 3 (stub).
//!
//! GGML format support is deferred. ONNX and GGUF are higher priority.

use hologram_ai_common::AiGraph;

/// Import a GGML model from bytes.
///
/// **Not yet implemented** — returns an error.
pub fn import_ggml(_bytes: &[u8]) -> anyhow::Result<AiGraph> {
    anyhow::bail!("GGML importer not yet implemented (Phase 3)")
}
