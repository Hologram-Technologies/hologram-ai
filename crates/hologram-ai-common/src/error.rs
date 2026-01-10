//! Unified error types for hologram-ai.

use thiserror::Error;

/// Common error type for hologram-ai operations.
#[derive(Error, Debug)]
pub enum CommonError {
    /// Weight not found in weight map.
    #[error("Weight not found: {0}")]
    WeightNotFound(String),

    /// Invalid weight shape.
    #[error("Invalid weight shape for {name}: expected {expected:?}, got {actual:?}")]
    InvalidWeightShape {
        /// Weight name.
        name: String,
        /// Expected shape.
        expected: Vec<usize>,
        /// Actual shape.
        actual: Vec<usize>,
    },

    /// Invalid configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Unsupported feature.
    #[error("Unsupported feature: {0}")]
    Unsupported(String),

    /// IR graph building error.
    #[error("Graph building error: {0}")]
    GraphBuildError(String),

    /// Serialization error.
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// IO error.
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Result type for hologram-ai operations.
pub type Result<T> = std::result::Result<T, CommonError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_not_found_error() {
        let err = CommonError::WeightNotFound("model.layers.0.attention.wq.weight".into());
        assert!(err.to_string().contains("Weight not found"));
        assert!(err.to_string().contains("wq.weight"));
    }

    #[test]
    fn test_invalid_weight_shape_error() {
        let err = CommonError::InvalidWeightShape {
            name: "embed_tokens".into(),
            expected: vec![32000, 4096],
            actual: vec![32000, 2048],
        };
        let msg = err.to_string();
        assert!(msg.contains("Invalid weight shape"));
        assert!(msg.contains("embed_tokens"));
    }

    #[test]
    fn test_invalid_config_error() {
        let err = CommonError::InvalidConfig("num_layers must be > 0".into());
        assert!(err.to_string().contains("Invalid configuration"));
    }

    #[test]
    fn test_unsupported_error() {
        let err = CommonError::Unsupported("MoE routing".into());
        assert!(err.to_string().contains("Unsupported feature"));
    }

    #[test]
    fn test_error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CommonError>();
    }
}
