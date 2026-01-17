//! Embeddable sections for single-file model distribution.
//!
//! This module provides a trait-based system for embedding auxiliary data
//! (vocabulary, configs, preprocessor settings) in `.holo` bundles.
//!
//! # Overview
//!
//! When compiling an ONNX model, you often need additional files for inference:
//! - Vocabulary files (`vocab.txt`, `vocab.json`)
//! - Tokenizer configuration (`tokenizer_config.json`)
//! - Model configuration (`config.json`)
//! - Special tokens map (`special_tokens_map.json`)
//! - SentencePiece models (`.model`)
//!
//! This module enables embedding all these files directly in the `.holo` bundle,
//! enabling true single-file distribution.
//!
//! # Architecture
//!
//! The system uses two core traits:
//!
//! - [`EmbeddableSection`]: For types that can be serialized and embedded
//! - [`FromEmbeddedSection`]: For types that can be deserialized from embedded data
//!
//! # Built-in Section Types
//!
//! | Type | Section ID | Content Type | Use Case |
//! |------|------------|--------------|----------|
//! | [`VocabularySection`] | `vocabulary` | text/plain or application/json | WordPiece/BPE vocab |
//! | [`TokenizerConfigSection`] | `tokenizer_config` | application/json | Tokenizer parameters |
//! | [`ModelConfigSection`] | `model_config` | application/json | Model architecture |
//! | [`SpecialTokensSection`] | `special_tokens_map` | application/json | Special token mappings |
//! | [`GenerationConfigSection`] | `generation_config` | application/json | LLM generation params |
//! | [`PreprocessorConfigSection`] | `preprocessor_config` | application/json | Vision preprocessing |
//! | [`SentencePieceSection`] | `sentencepiece_model` | application/x-sentencepiece | SentencePiece binary |
//! | [`RawFileSection`] | (custom) | (custom) | Arbitrary files |
//!
//! # Example: Embedding Sections
//!
//! ```rust,ignore
//! use hologram_ai_onnx::core::sections::{VocabularySection, TokenizerConfigSection};
//! use hologram_ai_onnx::core::UnifiedBundleWriter;
//! use serde_json::json;
//!
//! let mut writer = UnifiedBundleWriter::new();
//! writer.set_graph_bytes(graph_data);
//! writer.set_weights_bytes(weights_data);
//!
//! // Add vocabulary
//! let vocab = VocabularySection::from_lines(vec![
//!     "[PAD]".to_string(),
//!     "[UNK]".to_string(),
//!     "[CLS]".to_string(),
//! ]);
//! writer.add_section(vocab);
//!
//! // Add tokenizer config
//! let config = TokenizerConfigSection::new(json!({
//!     "do_lower_case": true,
//!     "max_length": 512
//! }));
//! writer.add_section(config);
//!
//! let bundle = writer.finish();
//! ```
//!
//! # Example: Reading Sections
//!
//! ```rust,ignore
//! use hologram_ai_onnx::core::sections::{VocabularySection, TokenizerConfigSection};
//! use hologram_ai_onnx::core::UnifiedBundleReader;
//!
//! let reader = UnifiedBundleReader::from_bytes(&bundle_data)?;
//!
//! // List all sections
//! for section in reader.sections() {
//!     println!("{}: {} bytes", section.id, section.size);
//! }
//!
//! // Get typed sections
//! if let Some(vocab) = reader.vocabulary() {
//!     println!("Vocabulary size: {}", vocab.len());
//! }
//!
//! if let Some(config) = reader.tokenizer_config() {
//!     println!("Max length: {:?}", config.get_i64("max_length"));
//! }
//! ```
//!
//! # Implementing Custom Sections
//!
//! ```rust,ignore
//! use hologram_ai_onnx::core::sections::{EmbeddableSection, FromEmbeddedSection, EmbedResult};
//!
//! struct MyCustomSection {
//!     data: Vec<u8>,
//! }
//!
//! impl EmbeddableSection for MyCustomSection {
//!     fn section_id(&self) -> &'static str { "my_custom_section" }
//!     fn to_bytes(&self) -> Vec<u8> { self.data.clone() }
//!     fn content_type(&self) -> &'static str { "application/x-custom" }
//! }
//!
//! impl FromEmbeddedSection for MyCustomSection {
//!     const SECTION_ID: &'static str = "my_custom_section";
//!
//!     fn from_bytes(bytes: &[u8]) -> EmbedResult<Self> {
//!         Ok(Self { data: bytes.to_vec() })
//!     }
//! }
//! ```

mod config;
mod error;
mod input_order;
mod raw;
mod sentencepiece;
mod traits;
mod vocabulary;

// Re-export core traits
pub use traits::{CloneableSection, EmbeddableSection, FromEmbeddedSection};

// Re-export layer header section from hologram bundle
pub use hologram::bundle::LayerHeaderSection;

// Re-export error types
pub use error::{EmbedError, EmbedResult};

// Re-export vocabulary section
pub use vocabulary::{VocabularyJsonSection, VocabularySection};

// Re-export config sections
pub use config::{
    GenerationConfigSection, ModelConfigSection, PreprocessorConfigSection, SpecialTokensSection,
    TokenizerConfigSection,
};
pub use input_order::InputOrderSection;

// Re-export sentencepiece section
pub use sentencepiece::SentencePieceSection;

// Re-export raw file section
pub use raw::{DynamicRawSection, RawFileSection};

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_all_section_ids_unique() {
        // Verify that all built-in section types have unique IDs
        let ids = [
            VocabularySection::SECTION_ID,
            VocabularyJsonSection::SECTION_ID,
            TokenizerConfigSection::SECTION_ID,
            ModelConfigSection::SECTION_ID,
            SpecialTokensSection::SECTION_ID,
            GenerationConfigSection::SECTION_ID,
            PreprocessorConfigSection::SECTION_ID,
            SentencePieceSection::SECTION_ID,
            InputOrderSection::SECTION_ID,
            RawFileSection::SECTION_ID,
            LayerHeaderSection::SECTION_ID,
        ];

        let mut unique_ids = std::collections::HashSet::new();
        for id in &ids {
            assert!(unique_ids.insert(*id), "Duplicate section ID: {}", id);
        }
    }

    #[test]
    fn test_section_trait_object_usage() {
        // Verify sections can be used as trait objects
        let sections: Vec<Box<dyn EmbeddableSection>> = vec![
            Box::new(VocabularySection::from_lines(vec!["test".to_string()])),
            Box::new(TokenizerConfigSection::new(json!({"key": "value"}))),
            Box::new(ModelConfigSection::new(json!({"model_type": "bert"}))),
            Box::new(SentencePieceSection::new(vec![1, 2, 3])),
            Box::new(RawFileSection::binary("test", vec![4, 5, 6])),
        ];

        for section in &sections {
            // All trait methods should work
            let _ = section.section_id();
            let _ = section.to_bytes();
            let _ = section.content_type();
            let _ = section.version();
        }

        assert_eq!(sections.len(), 5);
    }

    #[test]
    fn test_cloneable_section() {
        let section = VocabularySection::from_lines(vec!["a".to_string(), "b".to_string()]);
        let boxed = section.clone_boxed();

        assert_eq!(boxed.section_id(), "vocabulary");
        assert_eq!(boxed.to_bytes(), section.to_bytes());
    }

    #[test]
    fn test_all_sections_have_roundtrip() {
        // VocabularySection
        let vocab = VocabularySection::from_lines(vec!["token".to_string()]);
        let restored_vocab = VocabularySection::from_bytes(&vocab.to_bytes()).unwrap();
        assert_eq!(vocab.len(), restored_vocab.len());

        // TokenizerConfigSection
        let tok_config = TokenizerConfigSection::new(json!({"key": "value"}));
        let restored_tok = TokenizerConfigSection::from_bytes(&tok_config.to_bytes()).unwrap();
        assert!(restored_tok.get("key").is_some());

        // ModelConfigSection
        let model_config = ModelConfigSection::new(json!({"hidden_size": 768}));
        let restored_model = ModelConfigSection::from_bytes(&model_config.to_bytes()).unwrap();
        assert_eq!(restored_model.hidden_size(), Some(768));

        // SpecialTokensSection
        let special = SpecialTokensSection::new(json!({"bos_token": "<s>"}));
        let restored_special = SpecialTokensSection::from_bytes(&special.to_bytes()).unwrap();
        assert_eq!(restored_special.bos_token(), Some("<s>"));

        // GenerationConfigSection
        let gen_config = GenerationConfigSection::new(json!({"max_length": 100}));
        let restored_gen = GenerationConfigSection::from_bytes(&gen_config.to_bytes()).unwrap();
        assert_eq!(restored_gen.max_length(), Some(100));

        // SentencePieceSection
        let sp = SentencePieceSection::new(vec![1, 2, 3]);
        let restored_sp = SentencePieceSection::from_bytes(&sp.to_bytes()).unwrap();
        assert_eq!(restored_sp.as_bytes(), sp.as_bytes());

        // RawFileSection
        let raw = RawFileSection::binary("test", vec![4, 5, 6]);
        let restored_raw = RawFileSection::from_bytes(&raw.to_bytes()).unwrap();
        assert_eq!(restored_raw.data(), raw.data());
    }
}
