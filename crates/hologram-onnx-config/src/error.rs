//! Error types for configuration and output handling.

use std::io;
use thiserror::Error;

/// Errors that can occur during config parsing and output handling.
#[derive(Error, Debug)]
pub enum ConfigError {
    /// TOML parsing error
    #[error("Failed to parse TOML config: {0}")]
    TomlParse(#[from] toml::de::Error),

    /// TOML serialization error
    #[error("Failed to serialize TOML config: {0}")]
    TomlSerialize(#[from] toml::ser::Error),

    /// IO error during config file reading
    #[error("IO error reading config file: {0}")]
    Io(#[from] io::Error),

    /// Missing required field in config
    #[error("Missing required field in config: {field}")]
    MissingField {
        /// Field name that was missing
        field: String,
    },

    /// Invalid value in config
    #[error("Invalid value for field '{field}': {reason}")]
    InvalidValue {
        /// Field name with invalid value
        field: String,
        /// Reason why the value is invalid
        reason: String,
    },

    /// Unknown output handler type
    #[error("Unknown output handler type: {handler_type}")]
    UnknownHandlerType {
        /// The unknown handler type
        handler_type: String,
    },

    /// Feature not enabled for handler
    #[error("Handler type '{handler_type}' requires feature '{feature}' to be enabled")]
    FeatureNotEnabled {
        /// Handler type that requires the feature
        handler_type: String,
        /// Feature flag name
        feature: String,
    },

    /// Missing output tensor
    #[error("Missing expected output tensor: {tensor_name}")]
    MissingOutputTensor {
        /// Name of the missing tensor
        tensor_name: String,
    },

    /// Invalid tensor shape
    #[error("Invalid tensor shape for output '{output_name}': expected {expected}, got {actual}")]
    InvalidTensorShape {
        /// Output name
        output_name: String,
        /// Expected shape description
        expected: String,
        /// Actual shape description
        actual: String,
    },

    /// Invalid image format
    #[error("Invalid image format: {0}")]
    InvalidImageFormat(String),

    /// Invalid audio format
    #[error("Invalid audio format: {0}")]
    InvalidAudioFormat(String),

    /// Image processing error
    #[cfg(feature = "image-output")]
    #[error("Image processing error: {0}")]
    ImageError(#[from] image::ImageError),

    /// Audio processing error
    #[cfg(feature = "audio-output")]
    #[error("Audio processing error: {0}")]
    HoundError(#[from] hound::Error),

    /// Tokenizer error
    #[cfg(feature = "text-output")]
    #[error("Tokenizer error: {0}")]
    TokenizerError(String),

    /// Generic error
    #[error("Config error: {0}")]
    Other(String),
}

impl ConfigError {
    /// Create a missing field error.
    pub fn missing_field(field: impl Into<String>) -> Self {
        Self::MissingField {
            field: field.into(),
        }
    }

    /// Create an invalid value error.
    pub fn invalid_value(field: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::InvalidValue {
            field: field.into(),
            reason: reason.into(),
        }
    }

    /// Create an unknown handler type error.
    pub fn unknown_handler_type(handler_type: impl Into<String>) -> Self {
        Self::UnknownHandlerType {
            handler_type: handler_type.into(),
        }
    }

    /// Create a feature not enabled error.
    pub fn feature_not_enabled(
        handler_type: impl Into<String>,
        feature: impl Into<String>,
    ) -> Self {
        Self::FeatureNotEnabled {
            handler_type: handler_type.into(),
            feature: feature.into(),
        }
    }

    /// Create a missing output tensor error.
    pub fn missing_output_tensor(tensor_name: impl Into<String>) -> Self {
        Self::MissingOutputTensor {
            tensor_name: tensor_name.into(),
        }
    }

    /// Create an invalid tensor shape error.
    pub fn invalid_tensor_shape(
        output_name: impl Into<String>,
        expected: impl Into<String>,
        actual: impl Into<String>,
    ) -> Self {
        Self::InvalidTensorShape {
            output_name: output_name.into(),
            expected: expected.into(),
            actual: actual.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_missing_field_error() {
        let err = ConfigError::missing_field("output_name");
        assert!(err.to_string().contains("output_name"));
    }

    #[test]
    fn test_invalid_value_error() {
        let err = ConfigError::invalid_value("format", "must be 'rgb' or 'rgba'");
        assert!(err.to_string().contains("format"));
        assert!(err.to_string().contains("rgb"));
    }

    #[test]
    fn test_unknown_handler_type_error() {
        let err = ConfigError::unknown_handler_type("video");
        assert!(err.to_string().contains("video"));
    }

    #[test]
    fn test_feature_not_enabled_error() {
        let err = ConfigError::feature_not_enabled("image", "image-output");
        assert!(err.to_string().contains("image"));
        assert!(err.to_string().contains("image-output"));
    }

    #[test]
    fn test_missing_output_tensor_error() {
        let err = ConfigError::missing_output_tensor("sample");
        assert!(err.to_string().contains("sample"));
    }

    #[test]
    fn test_invalid_tensor_shape_error() {
        let err =
            ConfigError::invalid_tensor_shape("image", "[1, 3, 224, 224]", "[1, 3, 512, 512]");
        assert!(err.to_string().contains("image"));
        assert!(err.to_string().contains("224"));
        assert!(err.to_string().contains("512"));
    }
}
