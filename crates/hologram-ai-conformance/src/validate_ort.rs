//! ORT-based model validation: run an ONNX model through both ORT and hologram,
//! compare compilation results.
//!
//! Validates that hologram can compile a model that ORT accepts, ensuring the
//! import → optimize → lower → compile pipeline produces valid output.

use anyhow::Result;
use ort::session::Session;
use std::path::Path;

/// Validation result comparing ORT vs hologram.
pub struct OrtValidationReport {
    pub model_path: String,
    /// ORT loaded the model successfully.
    pub ort_ok: bool,
    /// Hologram compiled the model successfully.
    pub hologram_ok: bool,
    /// Number of hologram compiled nodes.
    pub compiled_nodes: usize,
    /// Error message if either side failed.
    pub error: Option<String>,
}

impl std::fmt::Display for OrtValidationReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== ORT Cross-Validation Report ===")?;
        writeln!(f, "Model: {}", self.model_path)?;
        writeln!(f, "ORT load:     {}", if self.ort_ok { "PASS" } else { "FAIL" })?;
        writeln!(f, "Hologram compile: {}", if self.hologram_ok { "PASS" } else { "FAIL" })?;
        if self.hologram_ok {
            writeln!(f, "Compiled nodes:   {}", self.compiled_nodes)?;
        }
        if let Some(ref err) = self.error {
            writeln!(f, "Error: {err}")?;
        }
        Ok(())
    }
}

/// Validate that an ONNX model can be loaded by ORT and compiled by hologram.
///
/// This is a compilation-level cross-check: if ORT accepts the model, hologram
/// should too. Full output comparison requires runtime execution which is
/// outside the scope of the compiler crate.
pub fn validate_model_with_ort(model_path: &Path) -> OrtValidationReport {
    let model_str = model_path.display().to_string();

    // Step 1: Can ORT load it?
    let ort_ok = match Session::builder()
        .and_then(|mut b| b.commit_from_file(model_path))
    {
        Ok(_session) => true,
        Err(e) => {
            return OrtValidationReport {
                model_path: model_str,
                ort_ok: false,
                hologram_ok: false,
                compiled_nodes: 0,
                error: Some(format!("ORT load failed: {e}")),
            };
        }
    };

    // Step 2: Can hologram compile it?
    use hologram_ai::compiler::{ModelCompiler, ModelSource};
    let source = ModelSource::OnnxPath(model_path.to_owned());
    match ModelCompiler::default().compile(source) {
        Ok(compiled) => OrtValidationReport {
            model_path: model_str,
            ort_ok,
            hologram_ok: true,
            compiled_nodes: compiled.stats.node_count,
            error: None,
        },
        Err(e) => OrtValidationReport {
            model_path: model_str,
            ort_ok,
            hologram_ok: false,
            compiled_nodes: 0,
            error: Some(format!("hologram compile failed: {e:#}")),
        },
    }
}

/// Run an in-memory ONNX model through ORT and return f32 outputs.
///
/// For single-op models built by `onnx_builder`, this is the primary
/// cross-validation path.
pub fn run_ort_model_bytes(
    model_bytes: &[u8],
    inputs: Vec<crate::ort_runner::runner::OrtInput>,
) -> Result<Vec<f32>> {
    crate::ort_runner::runner::run_onnx_bytes(model_bytes, inputs)
}
