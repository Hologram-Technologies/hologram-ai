//! Compile ONNX models and tokenizers directly into a pipeline bundle.
//!
//! TEMPORARILY STUBBED during hologram API migration.
//! The pipeline compilation functionality requires advanced features from the old API
//! (partitioning, memory budgets, etc.) that are not yet available.

use anyhow::Result;
use std::path::Path;
use tracing::warn;

/// Compile ONNX models and tokenizer into a single pipeline bundle.
///
/// STUB: This function is temporarily disabled during API migration.
#[allow(clippy::too_many_arguments)]
pub fn compile_pipeline_command(
    _encoder: Option<&Path>,
    _decoder: Option<&Path>,
    _tokenizer: Option<&Path>,
    _models: &[String],
    _config_path: Option<&Path>,
    _output: &Path,
    _weight_threshold: usize,
    _partition: bool,
    _partition_size: usize,
    _memory_budget: Option<usize>,
    _keep_intermediates: bool,
) -> Result<()> {
    warn!("Pipeline compilation is temporarily disabled during API migration");
    anyhow::bail!(
        "Pipeline compilation is temporarily disabled. \
         Use 'compile' command for individual models."
    )
}
