use thiserror::Error;

/// Errors produced by the hologram-ai-common library layer.
#[derive(Debug, Error)]
pub enum CommonError {
    #[error("lowering failed: {0}")]
    Lowering(String),

    #[error("graph validation failed: {count} error(s)")]
    Validation { count: usize },

    #[error("unsupported op: {op_type}")]
    UnsupportedOp { op_type: String },

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}
