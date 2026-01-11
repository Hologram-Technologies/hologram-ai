//! Error types for embeddable sections.
//!
//! This module defines error types used when embedding or extracting
//! sections from .holo bundles.

use thiserror::Error;

/// Result type for embedding operations.
pub type EmbedResult<T> = Result<T, EmbedError>;

/// Errors that can occur during section embedding/extraction.
#[derive(Error, Debug)]
pub enum EmbedError {
    /// Section data is invalid or malformed.
    #[error("Invalid section data: {0}")]
    InvalidData(String),

    /// Section format version is unsupported.
    #[error("Unsupported section version: {version} for section '{section_id}'")]
    UnsupportedVersion {
        /// The section identifier.
        section_id: String,
        /// The unsupported version number.
        version: u32,
    },

    /// Section not found in bundle.
    #[error("Section not found: {0}")]
    SectionNotFound(String),

    /// Section checksum verification failed.
    #[error("Checksum mismatch for section '{section_id}': expected {expected:08x}, got {actual:08x}")]
    ChecksumMismatch {
        /// The section identifier.
        section_id: String,
        /// Expected checksum value.
        expected: u32,
        /// Actual computed checksum.
        actual: u32,
    },

    /// I/O error during section read/write.
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// JSON serialization/deserialization error.
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// UTF-8 encoding error.
    #[error("UTF-8 encoding error: {0}")]
    Utf8Error(#[from] std::string::FromUtf8Error),

    /// Section table is truncated or corrupt.
    #[error("Section table truncated: expected {expected} bytes, got {actual}")]
    TableTruncated {
        /// Expected size in bytes.
        expected: usize,
        /// Actual available bytes.
        actual: usize,
    },
}

impl EmbedError {
    /// Create an invalid data error with a message.
    pub fn invalid_data(msg: impl Into<String>) -> Self {
        Self::InvalidData(msg.into())
    }

    /// Create an unsupported version error.
    pub fn unsupported_version(section_id: impl Into<String>, version: u32) -> Self {
        Self::UnsupportedVersion {
            section_id: section_id.into(),
            version,
        }
    }

    /// Create a section not found error.
    pub fn section_not_found(id: impl Into<String>) -> Self {
        Self::SectionNotFound(id.into())
    }

    /// Create a checksum mismatch error.
    pub fn checksum_mismatch(section_id: impl Into<String>, expected: u32, actual: u32) -> Self {
        Self::ChecksumMismatch {
            section_id: section_id.into(),
            expected,
            actual,
        }
    }

    /// Check if this is a recoverable error (section not found, version mismatch).
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::SectionNotFound(_) | Self::UnsupportedVersion { .. }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_data_error() {
        let err = EmbedError::invalid_data("test error");
        assert!(err.to_string().contains("test error"));
    }

    #[test]
    fn test_unsupported_version_error() {
        let err = EmbedError::unsupported_version("vocabulary", 99);
        let msg = err.to_string();
        assert!(msg.contains("vocabulary"));
        assert!(msg.contains("99"));
    }

    #[test]
    fn test_section_not_found_error() {
        let err = EmbedError::section_not_found("missing_section");
        assert!(err.to_string().contains("missing_section"));
    }

    #[test]
    fn test_checksum_mismatch_error() {
        let err = EmbedError::checksum_mismatch("vocab", 0x12345678, 0xABCDEF01);
        let msg = err.to_string();
        assert!(msg.contains("vocab"));
        assert!(msg.contains("12345678"));
        assert!(msg.contains("abcdef01"));
    }

    #[test]
    fn test_is_recoverable() {
        assert!(EmbedError::section_not_found("x").is_recoverable());
        assert!(EmbedError::unsupported_version("x", 1).is_recoverable());
        assert!(!EmbedError::invalid_data("x").is_recoverable());
        assert!(!EmbedError::checksum_mismatch("x", 0, 1).is_recoverable());
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let embed_err: EmbedError = io_err.into();
        assert!(matches!(embed_err, EmbedError::IoError(_)));
    }

    #[test]
    fn test_json_error_conversion() {
        let json_err = serde_json::from_str::<serde_json::Value>("invalid json").unwrap_err();
        let embed_err: EmbedError = json_err.into();
        assert!(matches!(embed_err, EmbedError::JsonError(_)));
    }

    #[test]
    fn test_utf8_error_conversion() {
        let invalid_bytes = vec![0xff, 0xfe];
        let utf8_err = String::from_utf8(invalid_bytes).unwrap_err();
        let embed_err: EmbedError = utf8_err.into();
        assert!(matches!(embed_err, EmbedError::Utf8Error(_)));
    }
}
