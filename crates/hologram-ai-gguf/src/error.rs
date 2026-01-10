//! GGUF error types.

use thiserror::Error;

/// Error type for GGUF operations.
#[derive(Error, Debug)]
pub enum GgufError {
    /// IO error reading file.
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Invalid GGUF magic bytes.
    #[error("Invalid GGUF magic bytes")]
    InvalidMagic,

    /// Unsupported GGUF version.
    #[error("Unsupported GGUF version: {0}")]
    UnsupportedVersion(u32),

    /// Missing required metadata.
    #[error("Missing required metadata: {0}")]
    MissingMetadata(String),

    /// Invalid metadata value.
    #[error("Invalid metadata value for {key}: {message}")]
    InvalidMetadata {
        /// Metadata key.
        key: String,
        /// Error message.
        message: String,
    },

    /// Unsupported quantization type.
    #[error("Unsupported quantization type: {0}")]
    UnsupportedQuantization(String),

    /// Unsupported architecture.
    #[error("Unsupported architecture: {0}")]
    UnsupportedArchitecture(String),

    /// Graph building error.
    #[error("Graph building error: {0}")]
    GraphBuildError(String),

    /// Compilation error.
    #[error("Compilation error: {0}")]
    CompilationError(String),

    /// Serialization error.
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Common error from hologram-ai-common.
    #[error("Common error: {0}")]
    CommonError(#[from] hologram_ai_common::CommonError),
}

/// Result type for GGUF operations.
pub type Result<T> = std::result::Result<T, GgufError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_magic_error() {
        let err = GgufError::InvalidMagic;
        assert!(err.to_string().contains("Invalid GGUF magic"));
    }

    #[test]
    fn test_unsupported_version_error() {
        let err = GgufError::UnsupportedVersion(1);
        assert!(err.to_string().contains("Unsupported GGUF version: 1"));
    }

    #[test]
    fn test_missing_metadata_error() {
        let err = GgufError::MissingMetadata("llama.block_count".into());
        assert!(err.to_string().contains("Missing required metadata"));
    }

    #[test]
    fn test_unsupported_quantization_error() {
        let err = GgufError::UnsupportedQuantization("Q2_K".into());
        assert!(err.to_string().contains("Unsupported quantization"));
    }

    #[test]
    fn test_error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<GgufError>();
    }
}
