//! SafeTensors error types.

use thiserror::Error;

/// Error type for SafeTensors operations.
#[derive(Error, Debug)]
pub enum SafeTensorsError {
    /// IO error.
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Not a directory.
    #[error("Not a directory: {0}")]
    NotADirectory(String),

    /// Missing config.json.
    #[error("Missing config.json in model directory")]
    MissingConfig,

    /// Missing SafeTensors files.
    #[error("No .safetensors files found in model directory")]
    MissingSafeTensors,

    /// Invalid JSON in config.
    #[error("Invalid JSON in config: {0}")]
    InvalidJson(#[from] serde_json::Error),

    /// Missing required configuration field.
    #[error("Missing required config field: {0}")]
    MissingConfigField(String),

    /// Unsupported model architecture.
    #[error("Unsupported model architecture: {0}")]
    UnsupportedArchitecture(String),

    /// Invalid SafeTensors header.
    #[error("Invalid SafeTensors header: {0}")]
    InvalidHeader(String),

    /// Invalid tensor data.
    #[error("Invalid tensor data: {0}")]
    InvalidTensorData(String),

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

/// Result type for SafeTensors operations.
pub type Result<T> = std::result::Result<T, SafeTensorsError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_not_a_directory_error() {
        let err = SafeTensorsError::NotADirectory("/path/to/file".into());
        assert!(err.to_string().contains("Not a directory"));
    }

    #[test]
    fn test_missing_config_error() {
        let err = SafeTensorsError::MissingConfig;
        assert!(err.to_string().contains("Missing config.json"));
    }

    #[test]
    fn test_missing_safetensors_error() {
        let err = SafeTensorsError::MissingSafeTensors;
        assert!(err.to_string().contains("No .safetensors files"));
    }

    #[test]
    fn test_error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SafeTensorsError>();
    }
}
