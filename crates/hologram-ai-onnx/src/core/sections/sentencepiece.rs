//! SentencePiece model section.
//!
//! This module provides the [`SentencePieceSection`] type for embedding
//! SentencePiece binary model files (.model) in .holo bundles.

use super::error::EmbedResult;
use super::traits::{EmbeddableSection, FromEmbeddedSection};

/// SentencePiece model section (binary .model file).
///
/// Contains the raw bytes of a SentencePiece model file, which uses
/// protobuf format internally.
///
/// # Example
///
/// ```rust,ignore
/// use hologram_ai_onnx::core::sections::SentencePieceSection;
/// use std::fs;
///
/// // Load from file
/// let model_bytes = fs::read("tokenizer.model")?;
/// let section = SentencePieceSection::new(model_bytes);
///
/// // Get the raw bytes
/// println!("Model size: {} bytes", section.as_bytes().len());
/// ```
#[derive(Debug, Clone)]
pub struct SentencePieceSection {
    /// Raw SentencePiece model bytes (protobuf format).
    model_bytes: Vec<u8>,
}

impl SentencePieceSection {
    /// Create from raw model bytes.
    ///
    /// # Arguments
    /// * `model_bytes` - The raw bytes of a SentencePiece .model file
    pub fn new(model_bytes: Vec<u8>) -> Self {
        Self { model_bytes }
    }

    /// Get the raw model bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.model_bytes
    }

    /// Get the model size in bytes.
    pub fn len(&self) -> usize {
        self.model_bytes.len()
    }

    /// Check if the model is empty.
    pub fn is_empty(&self) -> bool {
        self.model_bytes.is_empty()
    }

    /// Consume self and return the model bytes.
    pub fn into_bytes(self) -> Vec<u8> {
        self.model_bytes
    }

    /// Check if this looks like a valid SentencePiece model.
    ///
    /// Performs basic validation by checking for protobuf markers.
    /// This is a heuristic check, not a full validation.
    pub fn is_valid_format(&self) -> bool {
        // SentencePiece models are protobuf encoded
        // They typically start with field tags in the 0x08-0x7F range
        // and have a minimum size
        if self.model_bytes.len() < 10 {
            return false;
        }

        // Check for common protobuf field tag patterns
        // Field 1 (trainer_spec) with wire type 2 (length-delimited) = 0x0A
        // Field 2 (normalizer_spec) with wire type 2 = 0x12
        let first_byte = self.model_bytes[0];
        matches!(first_byte, 0x08..=0x7F)
    }
}

impl EmbeddableSection for SentencePieceSection {
    fn section_id(&self) -> &'static str {
        "sentencepiece_model"
    }

    fn to_bytes(&self) -> Vec<u8> {
        self.model_bytes.clone()
    }

    fn content_type(&self) -> &'static str {
        "application/x-sentencepiece"
    }
}

impl FromEmbeddedSection for SentencePieceSection {
    const SECTION_ID: &'static str = "sentencepiece_model";

    fn from_bytes(bytes: &[u8]) -> EmbedResult<Self> {
        Ok(Self::new(bytes.to_vec()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let data = vec![0x0A, 0x10, 0x12, 0x20, 0x30, 0x40];
        let section = SentencePieceSection::new(data.clone());

        assert_eq!(section.as_bytes(), &data);
        assert_eq!(section.len(), 6);
        assert!(!section.is_empty());
    }

    #[test]
    fn test_empty() {
        let section = SentencePieceSection::new(vec![]);

        assert!(section.is_empty());
        assert_eq!(section.len(), 0);
    }

    #[test]
    fn test_section_id() {
        let section = SentencePieceSection::new(vec![0x0A]);

        assert_eq!(section.section_id(), "sentencepiece_model");
        assert_eq!(SentencePieceSection::SECTION_ID, "sentencepiece_model");
    }

    #[test]
    fn test_content_type() {
        let section = SentencePieceSection::new(vec![]);

        assert_eq!(section.content_type(), "application/x-sentencepiece");
    }

    #[test]
    fn test_roundtrip() {
        let original_data = vec![0x0A, 0x08, 0x12, 0x06, 0x1A, 0x04, 0x22, 0x02];
        let section = SentencePieceSection::new(original_data.clone());

        let bytes = section.to_bytes();
        let restored = SentencePieceSection::from_bytes(&bytes).unwrap();

        assert_eq!(restored.as_bytes(), &original_data);
    }

    #[test]
    fn test_into_bytes() {
        let data = vec![1, 2, 3, 4, 5];
        let section = SentencePieceSection::new(data.clone());

        let bytes = section.into_bytes();
        assert_eq!(bytes, data);
    }

    #[test]
    fn test_is_valid_format() {
        // Too small
        let small = SentencePieceSection::new(vec![0x0A]);
        assert!(!small.is_valid_format());

        // Valid-looking header (starts with protobuf field tag)
        let valid = SentencePieceSection::new(vec![
            0x0A, 0x10, 0x08, 0x01, 0x12, 0x0C, 0x1A, 0x0A, 0x22, 0x08,
        ]);
        assert!(valid.is_valid_format());

        // Invalid first byte (outside protobuf field tag range)
        let invalid = SentencePieceSection::new(vec![
            0xFF, 0x10, 0x08, 0x01, 0x12, 0x0C, 0x1A, 0x0A, 0x22, 0x08,
        ]);
        assert!(!invalid.is_valid_format());
    }

    #[test]
    fn test_clone() {
        let section = SentencePieceSection::new(vec![10, 20, 30]);
        let cloned = section.clone();

        assert_eq!(section.as_bytes(), cloned.as_bytes());
    }

    #[test]
    fn test_large_model() {
        // Simulate a larger model (100KB)
        let large_data: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();
        let section = SentencePieceSection::new(large_data.clone());

        assert_eq!(section.len(), 100_000);

        let bytes = section.to_bytes();
        let restored = SentencePieceSection::from_bytes(&bytes).unwrap();

        assert_eq!(restored.as_bytes(), &large_data);
    }
}
