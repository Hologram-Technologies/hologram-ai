//! JSON configuration sections.
//!
//! This module provides section types for embedding JSON configuration files
//! commonly used with AI models (tokenizer config, model config, etc.).

use super::error::{EmbedError, EmbedResult};
use super::traits::{EmbeddableSection, FromEmbeddedSection};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Tokenizer configuration section.
///
/// Contains tokenizer parameters like `do_lower_case`, `max_length`,
/// special token IDs, etc.
///
/// # Example
///
/// ```rust,ignore
/// use hologram_ai_onnx::core::sections::TokenizerConfigSection;
/// use serde_json::json;
///
/// let config = TokenizerConfigSection::new(json!({
///     "do_lower_case": true,
///     "max_length": 512,
///     "pad_token": "[PAD]",
///     "unk_token": "[UNK]"
/// }));
///
/// assert_eq!(config.get_bool("do_lower_case"), Some(true));
/// assert_eq!(config.get_i64("max_length"), Some(512));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfigSection {
    /// Raw JSON configuration.
    #[serde(flatten)]
    config: Value,
}

impl TokenizerConfigSection {
    /// Create from a JSON value.
    pub fn new(config: Value) -> Self {
        Self { config }
    }

    /// Create from a JSON string.
    ///
    /// # Errors
    /// Returns an error if the JSON is invalid.
    pub fn from_json_str(json: &str) -> EmbedResult<Self> {
        let config = serde_json::from_str(json)
            .map_err(|e| EmbedError::invalid_data(format!("JSON parse error: {e}")))?;
        Ok(Self { config })
    }

    /// Get the raw JSON configuration.
    pub fn config(&self) -> &Value {
        &self.config
    }

    /// Get a configuration value by key.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.config.get(key)
    }

    /// Get a string value by key.
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.config.get(key).and_then(|v| v.as_str())
    }

    /// Get a boolean value by key.
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.config.get(key).and_then(|v| v.as_bool())
    }

    /// Get an integer value by key.
    pub fn get_i64(&self, key: &str) -> Option<i64> {
        self.config.get(key).and_then(|v| v.as_i64())
    }

    /// Get a floating-point value by key.
    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.config.get(key).and_then(|v| v.as_f64())
    }
}

impl EmbeddableSection for TokenizerConfigSection {
    fn section_id(&self) -> &'static str {
        "tokenizer_config"
    }

    fn to_bytes(&self) -> Vec<u8> {
        serde_json::to_vec_pretty(&self.config).unwrap_or_default()
    }

    fn content_type(&self) -> &'static str {
        "application/json"
    }
}

impl FromEmbeddedSection for TokenizerConfigSection {
    const SECTION_ID: &'static str = "tokenizer_config";

    fn from_bytes(bytes: &[u8]) -> EmbedResult<Self> {
        let text = String::from_utf8(bytes.to_vec())?;
        Self::from_json_str(&text)
    }
}

/// Model configuration section.
///
/// Contains model architecture parameters like `hidden_size`, `num_layers`,
/// `num_attention_heads`, etc.
///
/// # Example
///
/// ```rust,ignore
/// use hologram_ai_onnx::core::sections::ModelConfigSection;
/// use serde_json::json;
///
/// let config = ModelConfigSection::new(json!({
///     "hidden_size": 768,
///     "num_hidden_layers": 12,
///     "num_attention_heads": 12,
///     "intermediate_size": 3072,
///     "model_type": "bert"
/// }));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfigSection {
    /// Raw JSON configuration.
    #[serde(flatten)]
    config: Value,
}

impl ModelConfigSection {
    /// Create from a JSON value.
    pub fn new(config: Value) -> Self {
        Self { config }
    }

    /// Create from a JSON string.
    pub fn from_json_str(json: &str) -> EmbedResult<Self> {
        let config = serde_json::from_str(json)
            .map_err(|e| EmbedError::invalid_data(format!("JSON parse error: {e}")))?;
        Ok(Self { config })
    }

    /// Get the raw JSON configuration.
    pub fn config(&self) -> &Value {
        &self.config
    }

    /// Get a configuration value by key.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.config.get(key)
    }

    /// Get a string value by key.
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.config.get(key).and_then(|v| v.as_str())
    }

    /// Get an integer value by key.
    pub fn get_i64(&self, key: &str) -> Option<i64> {
        self.config.get(key).and_then(|v| v.as_i64())
    }

    /// Get the model type (e.g., "bert", "gpt2", "t5").
    pub fn model_type(&self) -> Option<&str> {
        self.get_str("model_type")
    }

    /// Get the hidden size.
    pub fn hidden_size(&self) -> Option<i64> {
        self.get_i64("hidden_size")
    }

    /// Get the number of hidden layers.
    pub fn num_hidden_layers(&self) -> Option<i64> {
        self.get_i64("num_hidden_layers")
    }

    /// Get the number of attention heads.
    pub fn num_attention_heads(&self) -> Option<i64> {
        self.get_i64("num_attention_heads")
    }
}

impl EmbeddableSection for ModelConfigSection {
    fn section_id(&self) -> &'static str {
        "model_config"
    }

    fn to_bytes(&self) -> Vec<u8> {
        serde_json::to_vec_pretty(&self.config).unwrap_or_default()
    }

    fn content_type(&self) -> &'static str {
        "application/json"
    }
}

impl FromEmbeddedSection for ModelConfigSection {
    const SECTION_ID: &'static str = "model_config";

    fn from_bytes(bytes: &[u8]) -> EmbedResult<Self> {
        let text = String::from_utf8(bytes.to_vec())?;
        Self::from_json_str(&text)
    }
}

/// Special tokens mapping section.
///
/// Contains the mapping of special token names to their string values.
///
/// # Example
///
/// ```rust,ignore
/// use hologram_ai_onnx::core::sections::SpecialTokensSection;
/// use serde_json::json;
///
/// let tokens = SpecialTokensSection::new(json!({
///     "bos_token": "<s>",
///     "eos_token": "</s>",
///     "unk_token": "<unk>",
///     "pad_token": "<pad>",
///     "mask_token": "<mask>"
/// }));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialTokensSection {
    /// Raw JSON token mapping.
    #[serde(flatten)]
    tokens: Value,
}

impl SpecialTokensSection {
    /// Create from a JSON value.
    pub fn new(tokens: Value) -> Self {
        Self { tokens }
    }

    /// Create from a JSON string.
    pub fn from_json_str(json: &str) -> EmbedResult<Self> {
        let tokens = serde_json::from_str(json)
            .map_err(|e| EmbedError::invalid_data(format!("JSON parse error: {e}")))?;
        Ok(Self { tokens })
    }

    /// Get the raw JSON token mapping.
    pub fn tokens(&self) -> &Value {
        &self.tokens
    }

    /// Get a special token by name.
    pub fn get(&self, name: &str) -> Option<&str> {
        self.tokens.get(name).and_then(|v| v.as_str())
    }

    /// Get the BOS (beginning of sequence) token.
    pub fn bos_token(&self) -> Option<&str> {
        self.get("bos_token")
    }

    /// Get the EOS (end of sequence) token.
    pub fn eos_token(&self) -> Option<&str> {
        self.get("eos_token")
    }

    /// Get the UNK (unknown) token.
    pub fn unk_token(&self) -> Option<&str> {
        self.get("unk_token")
    }

    /// Get the PAD (padding) token.
    pub fn pad_token(&self) -> Option<&str> {
        self.get("pad_token")
    }

    /// Get the MASK token.
    pub fn mask_token(&self) -> Option<&str> {
        self.get("mask_token")
    }

    /// Get the CLS (classification) token.
    pub fn cls_token(&self) -> Option<&str> {
        self.get("cls_token")
    }

    /// Get the SEP (separator) token.
    pub fn sep_token(&self) -> Option<&str> {
        self.get("sep_token")
    }
}

impl EmbeddableSection for SpecialTokensSection {
    fn section_id(&self) -> &'static str {
        "special_tokens_map"
    }

    fn to_bytes(&self) -> Vec<u8> {
        serde_json::to_vec_pretty(&self.tokens).unwrap_or_default()
    }

    fn content_type(&self) -> &'static str {
        "application/json"
    }
}

impl FromEmbeddedSection for SpecialTokensSection {
    const SECTION_ID: &'static str = "special_tokens_map";

    fn from_bytes(bytes: &[u8]) -> EmbedResult<Self> {
        let tokens = serde_json::from_slice(bytes)
            .map_err(|e| EmbedError::invalid_data(format!("JSON parse error: {e}")))?;
        Ok(Self { tokens })
    }
}

/// Generation configuration section.
///
/// Contains parameters for text generation with language models,
/// such as `max_length`, `temperature`, `top_k`, `top_p`, etc.
///
/// # Example
///
/// ```rust,ignore
/// use hologram_ai_onnx::core::sections::GenerationConfigSection;
/// use serde_json::json;
///
/// let config = GenerationConfigSection::new(json!({
///     "max_length": 100,
///     "min_length": 10,
///     "do_sample": true,
///     "temperature": 0.7,
///     "top_k": 50,
///     "top_p": 0.9
/// }));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfigSection {
    /// Raw JSON configuration.
    #[serde(flatten)]
    config: Value,
}

impl GenerationConfigSection {
    /// Create from a JSON value.
    pub fn new(config: Value) -> Self {
        Self { config }
    }

    /// Create from a JSON string.
    pub fn from_json_str(json: &str) -> EmbedResult<Self> {
        let config = serde_json::from_str(json)
            .map_err(|e| EmbedError::invalid_data(format!("JSON parse error: {e}")))?;
        Ok(Self { config })
    }

    /// Get the raw JSON configuration.
    pub fn config(&self) -> &Value {
        &self.config
    }

    /// Get a configuration value by key.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.config.get(key)
    }

    /// Get maximum generation length.
    pub fn max_length(&self) -> Option<i64> {
        self.config.get("max_length").and_then(|v| v.as_i64())
    }

    /// Get minimum generation length.
    pub fn min_length(&self) -> Option<i64> {
        self.config.get("min_length").and_then(|v| v.as_i64())
    }

    /// Get temperature for sampling.
    pub fn temperature(&self) -> Option<f64> {
        self.config.get("temperature").and_then(|v| v.as_f64())
    }

    /// Get top-k value for sampling.
    pub fn top_k(&self) -> Option<i64> {
        self.config.get("top_k").and_then(|v| v.as_i64())
    }

    /// Get top-p (nucleus sampling) value.
    pub fn top_p(&self) -> Option<f64> {
        self.config.get("top_p").and_then(|v| v.as_f64())
    }

    /// Check if sampling is enabled.
    pub fn do_sample(&self) -> Option<bool> {
        self.config.get("do_sample").and_then(|v| v.as_bool())
    }

    /// Get the EOS token ID.
    pub fn eos_token_id(&self) -> Option<i64> {
        self.config.get("eos_token_id").and_then(|v| v.as_i64())
    }

    /// Get the PAD token ID.
    pub fn pad_token_id(&self) -> Option<i64> {
        self.config.get("pad_token_id").and_then(|v| v.as_i64())
    }
}

impl EmbeddableSection for GenerationConfigSection {
    fn section_id(&self) -> &'static str {
        "generation_config"
    }

    fn to_bytes(&self) -> Vec<u8> {
        serde_json::to_vec_pretty(&self.config).unwrap_or_default()
    }

    fn content_type(&self) -> &'static str {
        "application/json"
    }
}

impl FromEmbeddedSection for GenerationConfigSection {
    const SECTION_ID: &'static str = "generation_config";

    fn from_bytes(bytes: &[u8]) -> EmbedResult<Self> {
        let config = serde_json::from_slice(bytes)
            .map_err(|e| EmbedError::invalid_data(format!("JSON parse error: {e}")))?;
        Ok(Self { config })
    }
}

/// Preprocessor configuration section.
///
/// Contains preprocessing parameters for vision models (image size,
/// normalization values, etc.) or audio models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessorConfigSection {
    /// Raw JSON configuration.
    #[serde(flatten)]
    config: Value,
}

impl PreprocessorConfigSection {
    /// Create from a JSON value.
    pub fn new(config: Value) -> Self {
        Self { config }
    }

    /// Create from a JSON string.
    pub fn from_json_str(json: &str) -> EmbedResult<Self> {
        let config = serde_json::from_str(json)
            .map_err(|e| EmbedError::invalid_data(format!("JSON parse error: {e}")))?;
        Ok(Self { config })
    }

    /// Get the raw JSON configuration.
    pub fn config(&self) -> &Value {
        &self.config
    }

    /// Get a configuration value by key.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.config.get(key)
    }

    /// Get image size (for vision models).
    pub fn image_size(&self) -> Option<i64> {
        self.config.get("size").and_then(|v| v.as_i64())
    }

    /// Check if resizing is enabled.
    pub fn do_resize(&self) -> Option<bool> {
        self.config.get("do_resize").and_then(|v| v.as_bool())
    }

    /// Check if normalization is enabled.
    pub fn do_normalize(&self) -> Option<bool> {
        self.config.get("do_normalize").and_then(|v| v.as_bool())
    }
}

impl EmbeddableSection for PreprocessorConfigSection {
    fn section_id(&self) -> &'static str {
        "preprocessor_config"
    }

    fn to_bytes(&self) -> Vec<u8> {
        serde_json::to_vec_pretty(&self.config).unwrap_or_default()
    }

    fn content_type(&self) -> &'static str {
        "application/json"
    }
}

impl FromEmbeddedSection for PreprocessorConfigSection {
    const SECTION_ID: &'static str = "preprocessor_config";

    fn from_bytes(bytes: &[u8]) -> EmbedResult<Self> {
        let config = serde_json::from_slice(bytes)
            .map_err(|e| EmbedError::invalid_data(format!("JSON parse error: {e}")))?;
        Ok(Self { config })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_tokenizer_config() {
        let config = TokenizerConfigSection::new(json!({
            "do_lower_case": true,
            "max_length": 512,
            "pad_token": "[PAD]"
        }));

        assert_eq!(config.section_id(), "tokenizer_config");
        assert_eq!(config.get_bool("do_lower_case"), Some(true));
        assert_eq!(config.get_i64("max_length"), Some(512));
        assert_eq!(config.get_str("pad_token"), Some("[PAD]"));
        assert_eq!(config.get("missing"), None);
    }

    #[test]
    fn test_tokenizer_config_roundtrip() {
        let original = TokenizerConfigSection::new(json!({
            "do_lower_case": false,
            "vocab_size": 30522
        }));

        let bytes = original.to_bytes();
        let restored = TokenizerConfigSection::from_bytes(&bytes).unwrap();

        assert_eq!(
            original.get_bool("do_lower_case"),
            restored.get_bool("do_lower_case")
        );
        assert_eq!(
            original.get_i64("vocab_size"),
            restored.get_i64("vocab_size")
        );
    }

    #[test]
    fn test_model_config() {
        let config = ModelConfigSection::new(json!({
            "model_type": "bert",
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12
        }));

        assert_eq!(config.section_id(), "model_config");
        assert_eq!(config.model_type(), Some("bert"));
        assert_eq!(config.hidden_size(), Some(768));
        assert_eq!(config.num_hidden_layers(), Some(12));
        assert_eq!(config.num_attention_heads(), Some(12));
    }

    #[test]
    fn test_model_config_roundtrip() {
        let original = ModelConfigSection::new(json!({
            "model_type": "gpt2",
            "hidden_size": 1024
        }));

        let bytes = original.to_bytes();
        let restored = ModelConfigSection::from_bytes(&bytes).unwrap();

        assert_eq!(original.model_type(), restored.model_type());
        assert_eq!(original.hidden_size(), restored.hidden_size());
    }

    #[test]
    fn test_special_tokens() {
        let tokens = SpecialTokensSection::new(json!({
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "mask_token": "<mask>",
            "cls_token": "[CLS]",
            "sep_token": "[SEP]"
        }));

        assert_eq!(tokens.section_id(), "special_tokens_map");
        assert_eq!(tokens.bos_token(), Some("<s>"));
        assert_eq!(tokens.eos_token(), Some("</s>"));
        assert_eq!(tokens.unk_token(), Some("<unk>"));
        assert_eq!(tokens.pad_token(), Some("<pad>"));
        assert_eq!(tokens.mask_token(), Some("<mask>"));
        assert_eq!(tokens.cls_token(), Some("[CLS]"));
        assert_eq!(tokens.sep_token(), Some("[SEP]"));
    }

    #[test]
    fn test_special_tokens_roundtrip() {
        let original = SpecialTokensSection::new(json!({
            "bos_token": "<bos>",
            "eos_token": "<eos>"
        }));

        let bytes = original.to_bytes();
        let restored = SpecialTokensSection::from_bytes(&bytes).unwrap();

        assert_eq!(original.bos_token(), restored.bos_token());
        assert_eq!(original.eos_token(), restored.eos_token());
    }

    #[test]
    fn test_generation_config() {
        let config = GenerationConfigSection::new(json!({
            "max_length": 100,
            "min_length": 10,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.9,
            "do_sample": true,
            "eos_token_id": 2,
            "pad_token_id": 0
        }));

        assert_eq!(config.section_id(), "generation_config");
        assert_eq!(config.max_length(), Some(100));
        assert_eq!(config.min_length(), Some(10));
        assert!((config.temperature().unwrap() - 0.7).abs() < 0.001);
        assert_eq!(config.top_k(), Some(50));
        assert!((config.top_p().unwrap() - 0.9).abs() < 0.001);
        assert_eq!(config.do_sample(), Some(true));
        assert_eq!(config.eos_token_id(), Some(2));
        assert_eq!(config.pad_token_id(), Some(0));
    }

    #[test]
    fn test_generation_config_roundtrip() {
        let original = GenerationConfigSection::new(json!({
            "max_length": 200,
            "temperature": 1.0
        }));

        let bytes = original.to_bytes();
        let restored = GenerationConfigSection::from_bytes(&bytes).unwrap();

        assert_eq!(original.max_length(), restored.max_length());
        assert_eq!(original.temperature(), restored.temperature());
    }

    #[test]
    fn test_preprocessor_config() {
        let config = PreprocessorConfigSection::new(json!({
            "size": 224,
            "do_resize": true,
            "do_normalize": true,
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225]
        }));

        assert_eq!(config.section_id(), "preprocessor_config");
        assert_eq!(config.image_size(), Some(224));
        assert_eq!(config.do_resize(), Some(true));
        assert_eq!(config.do_normalize(), Some(true));
    }

    #[test]
    fn test_preprocessor_config_roundtrip() {
        let original = PreprocessorConfigSection::new(json!({
            "size": 384,
            "do_resize": false
        }));

        let bytes = original.to_bytes();
        let restored = PreprocessorConfigSection::from_bytes(&bytes).unwrap();

        assert_eq!(original.image_size(), restored.image_size());
        assert_eq!(original.do_resize(), restored.do_resize());
    }

    #[test]
    fn test_content_types() {
        let tokenizer = TokenizerConfigSection::new(json!({}));
        let model = ModelConfigSection::new(json!({}));
        let special = SpecialTokensSection::new(json!({}));
        let generation = GenerationConfigSection::new(json!({}));
        let preprocessor = PreprocessorConfigSection::new(json!({}));

        assert_eq!(tokenizer.content_type(), "application/json");
        assert_eq!(model.content_type(), "application/json");
        assert_eq!(special.content_type(), "application/json");
        assert_eq!(generation.content_type(), "application/json");
        assert_eq!(preprocessor.content_type(), "application/json");
    }

    #[test]
    fn test_from_json_str() {
        let json = r#"{"key": "value", "number": 42}"#;

        let tokenizer = TokenizerConfigSection::from_json_str(json).unwrap();
        let model = ModelConfigSection::from_json_str(json).unwrap();
        let generation = GenerationConfigSection::from_json_str(json).unwrap();
        let preprocessor = PreprocessorConfigSection::from_json_str(json).unwrap();

        assert_eq!(tokenizer.get_str("key"), Some("value"));
        assert!(model.get("number").is_some());
        assert!(generation.get("key").is_some());
        assert!(preprocessor.get("number").is_some());
    }

    #[test]
    fn test_invalid_json() {
        let invalid = "not json";

        assert!(TokenizerConfigSection::from_json_str(invalid).is_err());
        assert!(ModelConfigSection::from_json_str(invalid).is_err());
        assert!(GenerationConfigSection::from_json_str(invalid).is_err());
        assert!(PreprocessorConfigSection::from_json_str(invalid).is_err());
    }
}
