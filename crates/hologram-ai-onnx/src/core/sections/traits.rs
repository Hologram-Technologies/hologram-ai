//! Core traits for embeddable sections.
//!
//! This module defines the traits that all embeddable section types must implement.
//! The trait-based design allows for extensibility while maintaining type safety.

use super::error::EmbedResult;

/// Trait for data that can be embedded in a .holo bundle.
///
/// Implement this trait to add new embeddable section types. Each section type
/// has a unique identifier and can serialize itself to bytes.
///
/// # Built-in Implementations
///
/// - [`VocabularySection`](super::VocabularySection) - WordPiece/BPE vocabularies
/// - [`TokenizerConfigSection`](super::TokenizerConfigSection) - Tokenizer configuration
/// - [`ModelConfigSection`](super::ModelConfigSection) - Model architecture config
/// - [`SpecialTokensSection`](super::SpecialTokensSection) - Special token mappings
/// - [`SentencePieceSection`](super::SentencePieceSection) - SentencePiece models
/// - [`RawFileSection`](super::RawFileSection) - Arbitrary file data
///
/// # Example
///
/// ```rust,ignore
/// use hologram_ai_onnx::core::sections::{EmbeddableSection, EmbedResult};
///
/// struct MyCustomSection {
///     data: Vec<u8>,
/// }
///
/// impl EmbeddableSection for MyCustomSection {
///     fn section_id(&self) -> &'static str {
///         "my_custom_section"
///     }
///
///     fn to_bytes(&self) -> Vec<u8> {
///         self.data.clone()
///     }
///
///     fn content_type(&self) -> &'static str {
///         "application/x-custom"
///     }
/// }
/// ```
pub trait EmbeddableSection: Send + Sync {
    /// Unique section identifier (e.g., "vocabulary", "tokenizer_config").
    ///
    /// This ID is used to locate the section in the bundle and must be
    /// unique across all section types. Common IDs include:
    ///
    /// - `"vocabulary"` - Token vocabulary
    /// - `"tokenizer_config"` - Tokenizer parameters
    /// - `"model_config"` - Model architecture
    /// - `"special_tokens_map"` - Special token mappings
    /// - `"sentencepiece_model"` - SentencePiece binary
    /// - `"generation_config"` - LLM generation parameters
    fn section_id(&self) -> &'static str;

    /// Serialize section data to bytes.
    ///
    /// The format of the bytes is section-specific:
    /// - For text-based sections (vocab.txt): UTF-8 encoded text
    /// - For JSON sections: UTF-8 encoded JSON
    /// - For binary sections (SentencePiece): Raw binary data
    fn to_bytes(&self) -> Vec<u8>;

    /// Content type for this section (for tooling/debugging).
    ///
    /// Common values:
    /// - `"text/plain"` for vocabulary files (vocab.txt)
    /// - `"application/json"` for config files
    /// - `"application/x-sentencepiece"` for SentencePiece models
    /// - `"application/octet-stream"` for unknown/binary data
    fn content_type(&self) -> &'static str {
        "application/octet-stream"
    }

    /// Version number for this section format.
    ///
    /// Increment this when making breaking changes to the section format.
    /// Readers should check the version and handle migrations appropriately.
    fn version(&self) -> u32 {
        1
    }
}

/// Trait for deserializing embedded sections.
///
/// Types implementing this trait can be extracted from a .holo bundle.
/// The `SECTION_ID` constant must match the `section_id()` of the
/// corresponding [`EmbeddableSection`] implementation.
///
/// # Example
///
/// ```rust,ignore
/// use hologram_ai_onnx::core::sections::{FromEmbeddedSection, EmbedResult, EmbedError};
///
/// struct MyCustomSection {
///     data: Vec<u8>,
/// }
///
/// impl FromEmbeddedSection for MyCustomSection {
///     const SECTION_ID: &'static str = "my_custom_section";
///
///     fn from_bytes(bytes: &[u8]) -> EmbedResult<Self> {
///         Ok(Self {
///             data: bytes.to_vec(),
///         })
///     }
/// }
/// ```
pub trait FromEmbeddedSection: Sized {
    /// Section identifier this type handles.
    ///
    /// Must match the `section_id()` of the corresponding `EmbeddableSection`.
    const SECTION_ID: &'static str;

    /// Deserialize from bytes.
    ///
    /// # Arguments
    /// * `bytes` - Raw section data from the bundle
    ///
    /// # Errors
    /// Returns `EmbedError` if deserialization fails due to:
    /// - Invalid data format
    /// - Encoding errors
    /// - Missing required fields
    fn from_bytes(bytes: &[u8]) -> EmbedResult<Self>;

    /// Expected version for this section format.
    ///
    /// Override to support multiple versions during migration.
    /// The default implementation expects version 1.
    fn expected_version() -> u32 {
        1
    }
}

/// Helper trait for sections that can be cloned and boxed.
///
/// This is automatically implemented for all types that implement
/// `EmbeddableSection + Clone + 'static`.
pub trait CloneableSection: EmbeddableSection {
    /// Clone this section into a boxed trait object.
    fn clone_boxed(&self) -> Box<dyn EmbeddableSection>;
}

impl<T: EmbeddableSection + Clone + 'static> CloneableSection for T {
    fn clone_boxed(&self) -> Box<dyn EmbeddableSection> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test implementation for testing traits
    #[derive(Clone)]
    struct TestSection {
        data: Vec<u8>,
    }

    impl EmbeddableSection for TestSection {
        fn section_id(&self) -> &'static str {
            "test_section"
        }

        fn to_bytes(&self) -> Vec<u8> {
            self.data.clone()
        }

        fn content_type(&self) -> &'static str {
            "application/x-test"
        }

        fn version(&self) -> u32 {
            2
        }
    }

    impl FromEmbeddedSection for TestSection {
        const SECTION_ID: &'static str = "test_section";

        fn from_bytes(bytes: &[u8]) -> EmbedResult<Self> {
            Ok(Self {
                data: bytes.to_vec(),
            })
        }

        fn expected_version() -> u32 {
            2
        }
    }

    #[test]
    fn test_embeddable_section_trait() {
        let section = TestSection {
            data: vec![1, 2, 3, 4],
        };

        assert_eq!(section.section_id(), "test_section");
        assert_eq!(section.to_bytes(), vec![1, 2, 3, 4]);
        assert_eq!(section.content_type(), "application/x-test");
        assert_eq!(section.version(), 2);
    }

    #[test]
    fn test_from_embedded_section_trait() {
        let bytes = vec![5, 6, 7, 8];
        let section = TestSection::from_bytes(&bytes).unwrap();

        assert_eq!(section.data, vec![5, 6, 7, 8]);
        assert_eq!(TestSection::SECTION_ID, "test_section");
        assert_eq!(TestSection::expected_version(), 2);
    }

    #[test]
    fn test_cloneable_section() {
        let section = TestSection {
            data: vec![9, 10, 11],
        };

        let boxed = section.clone_boxed();
        assert_eq!(boxed.section_id(), "test_section");
        assert_eq!(boxed.to_bytes(), vec![9, 10, 11]);
    }

    #[test]
    fn test_default_content_type() {
        struct MinimalSection;

        impl EmbeddableSection for MinimalSection {
            fn section_id(&self) -> &'static str {
                "minimal"
            }
            fn to_bytes(&self) -> Vec<u8> {
                vec![]
            }
        }

        let section = MinimalSection;
        assert_eq!(section.content_type(), "application/octet-stream");
        assert_eq!(section.version(), 1);
    }

    #[test]
    fn test_section_roundtrip() {
        let original = TestSection {
            data: b"hello world".to_vec(),
        };

        let bytes = original.to_bytes();
        let restored = TestSection::from_bytes(&bytes).unwrap();

        assert_eq!(original.data, restored.data);
    }
}
