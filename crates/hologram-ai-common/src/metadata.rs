//! Metadata sections for embedding in .holo bundles.
//!
//! This module provides embeddable sections that store tokenizer, model,
//! and generation metadata directly in compiled .holo files, reducing the need
//! for external configuration.

use hologram_bundle::error::{EmbedError, EmbedResult};
use hologram_bundle::traits::{EmbeddableSection, FromEmbeddedSection};
use serde::{Deserialize, Serialize};

/// Tokenizer metadata for embedding in .holo files.
///
/// This contains all the information needed to tokenize/detokenize text
/// for the model, eliminating the need to specify these parameters in
/// runtime config files.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct TokenizerMetadata {
    /// Tokenizer type (e.g., "sentencepiece", "bpe", "wordpiece")
    pub tokenizer_type: String,

    /// Path to vocabulary file (optional - vocab may be embedded separately)
    pub vocab_path: Option<String>,

    /// Maximum sequence length
    pub max_length: usize,

    /// Padding token ID
    pub pad_token_id: i64,

    /// End-of-sequence token ID
    pub eos_token_id: i64,

    /// Unknown token ID
    pub unk_token_id: i64,

    /// Beginning-of-sequence token ID (optional)
    pub bos_token_id: Option<i64>,

    /// Separator token ID (optional, used by some models)
    pub sep_token_id: Option<i64>,

    /// Additional special tokens (name -> token_id mapping)
    pub special_tokens: Vec<(String, i64)>,
}

impl Default for TokenizerMetadata {
    fn default() -> Self {
        Self {
            tokenizer_type: "unknown".to_string(),
            vocab_path: None,
            max_length: 512,
            pad_token_id: 0,
            eos_token_id: 1,
            unk_token_id: 2,
            bos_token_id: None,
            sep_token_id: None,
            special_tokens: Vec::new(),
        }
    }
}

/// Model metadata for embedding in .holo files.
///
/// This contains high-level information about the model architecture
/// and capabilities.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct ModelMetadata {
    /// Model name (e.g., "t5-small", "bert-base-uncased")
    pub name: String,

    /// Model architecture (e.g., "T5", "BERT", "GPT2")
    pub architecture: String,

    /// Model version or variant
    pub version: Option<String>,

    /// Model tasks (e.g., ["translation", "summarization"])
    pub tasks: Vec<String>,

    /// Number of parameters (approximate)
    pub num_parameters: Option<u64>,

    /// Additional metadata (key-value pairs)
    pub extra: Vec<(String, String)>,
}

impl Default for ModelMetadata {
    fn default() -> Self {
        Self {
            name: "unknown".to_string(),
            architecture: "unknown".to_string(),
            version: None,
            tasks: Vec::new(),
            num_parameters: None,
            extra: Vec::new(),
        }
    }
}

/// Generation configuration for embedding in .holo files.
///
/// These are **suggested defaults** that can be overridden at runtime.
/// Unlike TokenizerMetadata (which defines the model's fixed properties),
/// these parameters control runtime behavior and should be treated as hints.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct GenerationConfig {
    /// Default maximum new tokens to generate
    pub max_new_tokens: usize,

    /// Default start token ID for generation
    pub start_token_id: i64,

    /// Default sampling temperature (1.0 = neutral)
    pub temperature: f32,

    /// Default top-K filtering
    pub top_k: usize,

    /// Default nucleus sampling threshold
    pub top_p: f32,

    /// Default sampling mode (false = greedy decoding)
    pub do_sample: bool,

    /// Default repetition penalty
    pub repetition_penalty: Option<f32>,

    /// Default length penalty
    pub length_penalty: Option<f32>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 50,
            start_token_id: 0,
            temperature: 1.0,
            top_k: 50,
            top_p: 0.9,
            do_sample: false,
            repetition_penalty: None,
            length_penalty: None,
        }
    }
}

// =============================================================================
// EmbeddableSection implementations for hologram bundle format
// =============================================================================

/// Embeddable section for tokenizer metadata.
#[derive(Clone, Debug)]
pub struct TokenizerSection {
    /// Tokenizer metadata
    pub metadata: TokenizerMetadata,
}

impl EmbeddableSection for TokenizerSection {
    fn section_id(&self) -> &'static str {
        "tokenizer_config"
    }

    fn content_type(&self) -> &'static str {
        "application/json"
    }

    fn to_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(&self.metadata).expect("Failed to serialize TokenizerMetadata to JSON")
    }
}

impl FromEmbeddedSection for TokenizerSection {
    const SECTION_ID: &'static str = "tokenizer_config";

    fn from_bytes(bytes: &[u8]) -> EmbedResult<Self> {
        let metadata = serde_json::from_slice(bytes).map_err(|e| {
            EmbedError::invalid_data(format!("Failed to parse tokenizer config: {}", e))
        })?;

        Ok(Self { metadata })
    }
}

/// Embeddable section for model metadata.
#[derive(Clone, Debug)]
pub struct ModelMetadataSection {
    /// Model metadata
    pub metadata: ModelMetadata,
}

impl EmbeddableSection for ModelMetadataSection {
    fn section_id(&self) -> &'static str {
        "model_config"
    }

    fn content_type(&self) -> &'static str {
        "application/json"
    }

    fn to_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(&self.metadata).expect("Failed to serialize ModelMetadata to JSON")
    }
}

impl FromEmbeddedSection for ModelMetadataSection {
    const SECTION_ID: &'static str = "model_config";

    fn from_bytes(bytes: &[u8]) -> EmbedResult<Self> {
        let metadata = serde_json::from_slice(bytes).map_err(|e| {
            EmbedError::invalid_data(format!("Failed to parse model config: {}", e))
        })?;

        Ok(Self { metadata })
    }
}

/// Embeddable section for generation configuration.
#[derive(Clone, Debug)]
pub struct GenerationConfigSection {
    /// Generation configuration
    pub config: GenerationConfig,
}

impl EmbeddableSection for GenerationConfigSection {
    fn section_id(&self) -> &'static str {
        "generation_config"
    }

    fn content_type(&self) -> &'static str {
        "application/json"
    }

    fn to_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(&self.config).expect("Failed to serialize GenerationConfig to JSON")
    }
}

impl FromEmbeddedSection for GenerationConfigSection {
    const SECTION_ID: &'static str = "generation_config";

    fn from_bytes(bytes: &[u8]) -> EmbedResult<Self> {
        let config = serde_json::from_slice(bytes).map_err(|e| {
            EmbedError::invalid_data(format!("Failed to parse generation config: {}", e))
        })?;

        Ok(Self { config })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_metadata_default() {
        let metadata = TokenizerMetadata::default();
        assert_eq!(metadata.tokenizer_type, "unknown");
        assert_eq!(metadata.max_length, 512);
        assert_eq!(metadata.pad_token_id, 0);
        assert_eq!(metadata.eos_token_id, 1);
    }

    #[test]
    fn test_tokenizer_section_roundtrip() {
        let metadata = TokenizerMetadata {
            tokenizer_type: "sentencepiece".to_string(),
            vocab_path: Some("/path/to/vocab".to_string()),
            max_length: 1024,
            pad_token_id: 0,
            eos_token_id: 1,
            unk_token_id: 2,
            bos_token_id: Some(3),
            sep_token_id: None,
            special_tokens: vec![("mask".to_string(), 4)],
        };

        let section = TokenizerSection {
            metadata: metadata.clone(),
        };

        // Serialize
        let bytes = section.to_bytes();

        // Deserialize
        let deserialized = TokenizerSection::from_bytes(&bytes).unwrap();

        assert_eq!(deserialized.metadata, metadata);
        assert_eq!(TokenizerSection::SECTION_ID, "tokenizer_config");
    }

    #[test]
    fn test_model_metadata_default() {
        let metadata = ModelMetadata::default();
        assert_eq!(metadata.name, "unknown");
        assert_eq!(metadata.architecture, "unknown");
        assert!(metadata.tasks.is_empty());
    }

    #[test]
    fn test_model_metadata_section_roundtrip() {
        let metadata = ModelMetadata {
            name: "t5-small".to_string(),
            architecture: "T5".to_string(),
            version: Some("1.0".to_string()),
            tasks: vec!["translation".to_string(), "summarization".to_string()],
            num_parameters: Some(60_000_000),
            extra: vec![("param1".to_string(), "value1".to_string())],
        };

        let section = ModelMetadataSection {
            metadata: metadata.clone(),
        };

        // Serialize
        let bytes = section.to_bytes();

        // Deserialize
        let deserialized = ModelMetadataSection::from_bytes(&bytes).unwrap();

        assert_eq!(deserialized.metadata, metadata);
        assert_eq!(ModelMetadataSection::SECTION_ID, "model_config");
    }

    #[test]
    fn test_generation_config_default() {
        let config = GenerationConfig::default();
        assert_eq!(config.max_new_tokens, 50);
        assert_eq!(config.temperature, 1.0);
        assert!(!config.do_sample);
    }

    #[test]
    fn test_generation_config_section_roundtrip() {
        let config = GenerationConfig {
            max_new_tokens: 100,
            start_token_id: 0,
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            do_sample: true,
            repetition_penalty: Some(1.2),
            length_penalty: Some(1.0),
        };

        let section = GenerationConfigSection {
            config: config.clone(),
        };

        // Serialize
        let bytes = section.to_bytes();

        // Deserialize
        let deserialized = GenerationConfigSection::from_bytes(&bytes).unwrap();

        assert_eq!(deserialized.config.max_new_tokens, config.max_new_tokens);
        assert_eq!(deserialized.config.temperature, config.temperature);
        assert_eq!(deserialized.config.do_sample, config.do_sample);
        assert_eq!(GenerationConfigSection::SECTION_ID, "generation_config");
    }

    #[test]
    fn test_tokenizer_section_as_trait_object() {
        let section = TokenizerSection {
            metadata: TokenizerMetadata::default(),
        };

        let boxed: Box<dyn EmbeddableSection> = Box::new(section);
        assert_eq!(boxed.section_id(), "tokenizer_config");
        assert_eq!(boxed.content_type(), "application/json");
    }

    #[test]
    fn test_json_serialization_format() {
        let metadata = TokenizerMetadata {
            tokenizer_type: "test".to_string(),
            vocab_path: None,
            max_length: 512,
            pad_token_id: 0,
            eos_token_id: 1,
            unk_token_id: 2,
            bos_token_id: None,
            sep_token_id: None,
            special_tokens: Vec::new(),
        };

        let section = TokenizerSection { metadata };
        let bytes = section.to_bytes();

        // Should be valid JSON
        let json_str = String::from_utf8(bytes).unwrap();
        assert!(json_str.contains("\"tokenizer_type\""));
        assert!(json_str.contains("\"test\""));
    }
}
