//! Raw file section for arbitrary data.
//!
//! This module provides the [`RawFileSection`] type for embedding
//! arbitrary files in .holo bundles when no specific section type exists.

use super::error::EmbedResult;
use super::traits::{EmbeddableSection, FromEmbeddedSection};

/// Raw file section for arbitrary data.
///
/// Use this for files that don't have a specific section type.
/// Allows custom section IDs and content types.
///
/// # Example
///
/// ```rust,ignore
/// use hologram_ai_onnx::core::sections::RawFileSection;
/// use std::fs;
///
/// // Embed a custom binary file
/// let data = fs::read("custom_weights.bin")?;
/// let section = RawFileSection::new(
///     "custom_weights",
///     "application/octet-stream",
///     data
/// );
///
/// // Embed a text file
/// let text = fs::read("readme.txt")?;
/// let readme = RawFileSection::new(
///     "readme",
///     "text/plain",
///     text
/// );
/// ```
#[derive(Debug, Clone)]
pub struct RawFileSection {
    /// Section identifier (custom).
    id: String,
    /// Content type.
    content_type: String,
    /// Raw data bytes.
    data: Vec<u8>,
}

impl RawFileSection {
    /// Create a new raw file section.
    ///
    /// # Arguments
    /// * `id` - Custom section identifier
    /// * `content_type` - MIME content type for the data
    /// * `data` - Raw file bytes
    pub fn new(id: impl Into<String>, content_type: impl Into<String>, data: Vec<u8>) -> Self {
        Self {
            id: id.into(),
            content_type: content_type.into(),
            data,
        }
    }

    /// Create a raw file section with binary content type.
    ///
    /// Uses "application/octet-stream" as the content type.
    pub fn binary(id: impl Into<String>, data: Vec<u8>) -> Self {
        Self::new(id, "application/octet-stream", data)
    }

    /// Create a raw file section with text content type.
    ///
    /// Uses "text/plain" as the content type.
    pub fn text(id: impl Into<String>, data: Vec<u8>) -> Self {
        Self::new(id, "text/plain", data)
    }

    /// Create a raw file section with JSON content type.
    ///
    /// Uses "application/json" as the content type.
    pub fn json(id: impl Into<String>, data: Vec<u8>) -> Self {
        Self::new(id, "application/json", data)
    }

    /// Get the custom section ID.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get the custom content type.
    pub fn get_content_type(&self) -> &str {
        &self.content_type
    }

    /// Get the raw data.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Get the data size in bytes.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the data is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Consume self and return the data.
    pub fn into_data(self) -> Vec<u8> {
        self.data
    }

    /// Try to interpret data as UTF-8 text.
    pub fn as_text(&self) -> Option<&str> {
        std::str::from_utf8(&self.data).ok()
    }
}

impl EmbeddableSection for RawFileSection {
    fn section_id(&self) -> &'static str {
        // This is a limitation - static str is required by the trait
        // For RawFileSection, we use "raw" as the base identifier
        // The actual custom ID is stored in the section table entry
        "raw"
    }

    fn to_bytes(&self) -> Vec<u8> {
        self.data.clone()
    }

    fn content_type(&self) -> &'static str {
        // Same limitation as section_id
        "application/octet-stream"
    }
}

impl FromEmbeddedSection for RawFileSection {
    const SECTION_ID: &'static str = "raw";

    fn from_bytes(bytes: &[u8]) -> EmbedResult<Self> {
        Ok(Self::new("raw", "application/octet-stream", bytes.to_vec()))
    }
}

/// Extended raw file section that can store dynamic ID and content type.
///
/// This wrapper allows the bundle writer to access the actual custom ID
/// and content type when writing the section table.
#[derive(Debug, Clone)]
pub struct DynamicRawSection {
    inner: RawFileSection,
}

impl DynamicRawSection {
    /// Create a new dynamic raw section.
    pub fn new(id: impl Into<String>, content_type: impl Into<String>, data: Vec<u8>) -> Self {
        Self {
            inner: RawFileSection::new(id, content_type, data),
        }
    }

    /// Get the custom section ID.
    pub fn custom_id(&self) -> &str {
        &self.inner.id
    }

    /// Get the custom content type.
    pub fn custom_content_type(&self) -> &str {
        &self.inner.content_type
    }

    /// Get the inner RawFileSection.
    pub fn inner(&self) -> &RawFileSection {
        &self.inner
    }

    /// Get the data.
    pub fn data(&self) -> &[u8] {
        &self.inner.data
    }
}

impl EmbeddableSection for DynamicRawSection {
    fn section_id(&self) -> &'static str {
        "raw"
    }

    fn to_bytes(&self) -> Vec<u8> {
        self.inner.data.clone()
    }

    fn content_type(&self) -> &'static str {
        "application/octet-stream"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let section = RawFileSection::new("my_section", "application/custom", vec![1, 2, 3]);

        assert_eq!(section.id(), "my_section");
        assert_eq!(section.get_content_type(), "application/custom");
        assert_eq!(section.data(), &[1, 2, 3]);
        assert_eq!(section.len(), 3);
        assert!(!section.is_empty());
    }

    #[test]
    fn test_binary() {
        let section = RawFileSection::binary("binary_data", vec![0xFF, 0xFE]);

        assert_eq!(section.id(), "binary_data");
        assert_eq!(section.get_content_type(), "application/octet-stream");
    }

    #[test]
    fn test_text() {
        let section = RawFileSection::text("text_data", b"hello world".to_vec());

        assert_eq!(section.id(), "text_data");
        assert_eq!(section.get_content_type(), "text/plain");
        assert_eq!(section.as_text(), Some("hello world"));
    }

    #[test]
    fn test_json() {
        let section = RawFileSection::json("json_data", b"{\"key\": \"value\"}".to_vec());

        assert_eq!(section.id(), "json_data");
        assert_eq!(section.get_content_type(), "application/json");
    }

    #[test]
    fn test_empty() {
        let section = RawFileSection::new("empty", "text/plain", vec![]);

        assert!(section.is_empty());
        assert_eq!(section.len(), 0);
    }

    #[test]
    fn test_into_data() {
        let data = vec![10, 20, 30, 40, 50];
        let section = RawFileSection::new("test", "application/octet-stream", data.clone());

        let result = section.into_data();
        assert_eq!(result, data);
    }

    #[test]
    fn test_as_text_valid_utf8() {
        let section = RawFileSection::new("text", "text/plain", b"valid utf8".to_vec());
        assert_eq!(section.as_text(), Some("valid utf8"));
    }

    #[test]
    fn test_as_text_invalid_utf8() {
        let section = RawFileSection::new("binary", "application/octet-stream", vec![0xFF, 0xFE]);
        assert_eq!(section.as_text(), None);
    }

    #[test]
    fn test_section_trait() {
        let section = RawFileSection::new("custom", "custom/type", vec![1, 2, 3]);

        // Trait methods return static strings
        assert_eq!(section.section_id(), "raw");
        assert_eq!(section.content_type(), "application/octet-stream");
        assert_eq!(section.to_bytes(), vec![1, 2, 3]);
    }

    #[test]
    fn test_roundtrip() {
        let original = RawFileSection::new("test", "application/octet-stream", vec![5, 10, 15]);

        let bytes = original.to_bytes();
        let restored = RawFileSection::from_bytes(&bytes).unwrap();

        assert_eq!(restored.data(), original.data());
    }

    #[test]
    fn test_clone() {
        let section = RawFileSection::new("test", "text/plain", vec![1, 2, 3]);
        let cloned = section.clone();

        assert_eq!(section.id(), cloned.id());
        assert_eq!(section.get_content_type(), cloned.get_content_type());
        assert_eq!(section.data(), cloned.data());
    }

    #[test]
    fn test_dynamic_raw_section() {
        let section = DynamicRawSection::new("custom_id", "custom/type", vec![1, 2, 3]);

        assert_eq!(section.custom_id(), "custom_id");
        assert_eq!(section.custom_content_type(), "custom/type");
        assert_eq!(section.data(), &[1, 2, 3]);

        // Trait methods still return static strings
        assert_eq!(section.section_id(), "raw");
        assert_eq!(section.content_type(), "application/octet-stream");
    }

    #[test]
    fn test_large_data() {
        // Test with 1MB of data
        let large_data: Vec<u8> = (0..1_000_000).map(|i| (i % 256) as u8).collect();
        let section = RawFileSection::binary("large", large_data.clone());

        assert_eq!(section.len(), 1_000_000);

        let bytes = section.to_bytes();
        let restored = RawFileSection::from_bytes(&bytes).unwrap();

        assert_eq!(restored.data(), &large_data);
    }
}
