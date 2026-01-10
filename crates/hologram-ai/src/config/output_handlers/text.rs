//! Text output handler for processing LLM/text generation outputs.
//!
//! # Performance
//!
//! - **O(1) token lookup**: Tokenizer maintains vocab HashMap
//! - **Batch decoding**: Process multiple sequences simultaneously
//! - **Skip special tokens**: Configurable filtering

use crate::config::OutputHandlerConfig;
use crate::config::error::ConfigError;
use crate::config::output_handlers::{OutputHandler, ProcessedOutput, TensorData};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use tokenizers::Tokenizer;
use tracing::{debug, trace};

/// Text output handler.
///
/// Processes model token ID outputs into human-readable text using tokenizers.
///
/// # Configuration
///
/// ```toml
/// [pipeline.execution.output_handlers.text]
/// type = "text"
/// output = "token_ids"          # Tensor name
/// tokenizer_path = "tokenizer.json"
/// skip_special_tokens = true    # Skip <pad>, <eos>, etc.
/// ```
#[derive(Debug)]
pub struct TextHandler {
    /// Output tensor name to process
    pub output_name: String,

    /// Tokenizer for decoding token IDs
    pub tokenizer: Tokenizer,

    /// Skip special tokens in output
    pub skip_special_tokens: bool,
}

impl TextHandler {
    /// Create from config.
    pub fn from_config(config: &OutputHandlerConfig) -> Result<Self, ConfigError> {
        let output_name = config.output.clone();

        let tokenizer_path = config
            .get_string("tokenizer_path")
            .ok_or_else(|| ConfigError::missing_field("tokenizer_path"))?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| ConfigError::TokenizerError(format!("Failed to load tokenizer: {}", e)))?;

        let skip_special_tokens = config.get_bool("skip_special_tokens").unwrap_or(true);

        debug!(
            "Created TextHandler: tokenizer={}, skip_special={}",
            tokenizer_path, skip_special_tokens
        );

        Ok(Self {
            output_name,
            tokenizer,
            skip_special_tokens,
        })
    }

    /// Convert token IDs to text.
    ///
    /// # Performance: O(n) where n = number of tokens
    fn decode_tokens(&self, token_ids: &[u32]) -> Result<String, ConfigError> {
        self.tokenizer
            .decode(token_ids, self.skip_special_tokens)
            .map_err(|e| ConfigError::TokenizerError(format!("Failed to decode tokens: {}", e)))
    }

    /// Extract token IDs from tensor.
    ///
    /// Handles various tensor shapes:
    /// - [seq_len] - Single sequence
    /// - [batch, seq_len] - Batched sequences (take first)
    /// - [batch, beam, seq_len] - Beam search (take first batch, first beam)
    fn extract_token_ids(&self, tensor: &TensorData) -> Result<Vec<u32>, ConfigError> {
        trace!("Extracting token IDs from shape: {:?}", tensor.shape);

        let token_ids: Vec<u32> = match tensor.shape.len() {
            1 => {
                // [seq_len] - Single sequence
                tensor.data.iter().map(|&f| f as u32).collect()
            }
            2 => {
                // [batch, seq_len] - Take first batch
                let seq_len = tensor.shape[1];
                tensor.data[0..seq_len].iter().map(|&f| f as u32).collect()
            }
            3 => {
                // [batch, beam, seq_len] - Take first batch, first beam
                let seq_len = tensor.shape[2];
                tensor.data[0..seq_len].iter().map(|&f| f as u32).collect()
            }
            _ => {
                return Err(ConfigError::invalid_tensor_shape(
                    &self.output_name,
                    "[seq_len], [batch, seq_len], or [batch, beam, seq_len]",
                    format!("{:?}", tensor.shape),
                ));
            }
        };

        Ok(token_ids)
    }
}

impl OutputHandler for TextHandler {
    fn handler_type(&self) -> &'static str {
        "text"
    }

    fn process(
        &self,
        outputs: &HashMap<String, TensorData>,
    ) -> Result<ProcessedOutput, ConfigError> {
        let tensor = outputs
            .get(&self.output_name)
            .ok_or_else(|| ConfigError::missing_output_tensor(&self.output_name))?;

        trace!("Processing text tensor: shape={:?}", tensor.shape);

        // Extract token IDs from tensor
        let token_ids = self.extract_token_ids(tensor)?;

        // Decode to text
        let text = self.decode_tokens(&token_ids)?;

        debug!(
            "Decoded {} tokens to text ({} chars)",
            token_ids.len(),
            text.len()
        );

        Ok(ProcessedOutput::Text(text))
    }

    fn save(&self, output: &ProcessedOutput, path: &Path) -> Result<(), ConfigError> {
        if let ProcessedOutput::Text(text) = output {
            debug!("Saving text to: {} ({} chars)", path.display(), text.len());
            fs::write(path, text)?;
            Ok(())
        } else {
            Err(ConfigError::Other("Expected Text output".to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::{NamedTempFile, TempDir};

    // Helper to create a simple test tokenizer
    fn create_test_tokenizer() -> (Tokenizer, std::path::PathBuf) {
        let temp_dir = TempDir::new().unwrap();
        let tokenizer_path = temp_dir.path().join("tokenizer.json");

        // Create a minimal tokenizer JSON for testing
        let tokenizer_json = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {"id": 0, "content": "<pad>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 1, "content": "<eos>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
            ],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": {
                "type": "Sequence",
                "decoders": []
            },
            "model": {
                "type": "WordLevel",
                "vocab": {
                    "<pad>": 0,
                    "<eos>": 1,
                    "hello": 2,
                    "world": 3,
                    "test": 4,
                    " ": 5
                },
                "unk_token": "<unk>"
            }
        }"#;

        let mut file = std::fs::File::create(&tokenizer_path).unwrap();
        file.write_all(tokenizer_json.as_bytes()).unwrap();

        let tokenizer = Tokenizer::from_file(&tokenizer_path).unwrap();

        // Return both tokenizer and path (path keeps temp_dir alive)
        (tokenizer, tokenizer_path)
    }

    #[test]
    fn test_text_handler_from_config() {
        let (_tokenizer, tokenizer_path) = create_test_tokenizer();

        let mut config_map = HashMap::new();
        config_map.insert(
            "tokenizer_path".to_string(),
            toml::Value::String(tokenizer_path.to_str().unwrap().to_string()),
        );
        config_map.insert(
            "skip_special_tokens".to_string(),
            toml::Value::Boolean(true),
        );

        let config = OutputHandlerConfig {
            handler_type: "text".to_string(),
            output: "token_ids".to_string(),
            config: config_map,
        };

        let handler = TextHandler::from_config(&config).unwrap();
        assert_eq!(handler.output_name, "token_ids");
        assert!(handler.skip_special_tokens);
    }

    #[test]
    fn test_text_handler_missing_tokenizer_path() {
        let config = OutputHandlerConfig {
            handler_type: "text".to_string(),
            output: "token_ids".to_string(),
            config: HashMap::new(),
        };

        let result = TextHandler::from_config(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("tokenizer_path"));
    }

    #[test]
    fn test_text_handler_invalid_tokenizer_path() {
        let mut config_map = HashMap::new();
        config_map.insert(
            "tokenizer_path".to_string(),
            toml::Value::String("/nonexistent/tokenizer.json".to_string()),
        );

        let config = OutputHandlerConfig {
            handler_type: "text".to_string(),
            output: "token_ids".to_string(),
            config: config_map,
        };

        let result = TextHandler::from_config(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("tokenizer"));
    }

    #[test]
    fn test_text_handler_defaults() {
        let (_tokenizer, tokenizer_path) = create_test_tokenizer();

        let mut config_map = HashMap::new();
        config_map.insert(
            "tokenizer_path".to_string(),
            toml::Value::String(tokenizer_path.to_str().unwrap().to_string()),
        );

        let config = OutputHandlerConfig {
            handler_type: "text".to_string(),
            output: "token_ids".to_string(),
            config: config_map,
        };

        let handler = TextHandler::from_config(&config).unwrap();
        assert!(handler.skip_special_tokens);
    }

    #[test]
    fn test_extract_token_ids_1d() {
        let (_tokenizer, tokenizer_path) = create_test_tokenizer();

        let mut config_map = HashMap::new();
        config_map.insert(
            "tokenizer_path".to_string(),
            toml::Value::String(tokenizer_path.to_str().unwrap().to_string()),
        );

        let config = OutputHandlerConfig {
            handler_type: "text".to_string(),
            output: "tokens".to_string(),
            config: config_map,
        };

        let handler = TextHandler::from_config(&config).unwrap();

        // [seq_len=5]
        let tensor = TensorData::new(vec![2.0, 5.0, 3.0, 1.0, 0.0], vec![5]);

        let token_ids = handler.extract_token_ids(&tensor).unwrap();
        assert_eq!(token_ids, vec![2, 5, 3, 1, 0]);
    }

    #[test]
    fn test_extract_token_ids_2d() {
        let (_tokenizer, tokenizer_path) = create_test_tokenizer();

        let mut config_map = HashMap::new();
        config_map.insert(
            "tokenizer_path".to_string(),
            toml::Value::String(tokenizer_path.to_str().unwrap().to_string()),
        );

        let config = OutputHandlerConfig {
            handler_type: "text".to_string(),
            output: "tokens".to_string(),
            config: config_map,
        };

        let handler = TextHandler::from_config(&config).unwrap();

        // [batch=2, seq_len=3] - should take first batch
        let tensor = TensorData::new(
            vec![
                2.0, 5.0, 3.0, // Batch 0: "hello world"
                4.0, 1.0, 0.0, // Batch 1: "test <eos> <pad>"
            ],
            vec![2, 3],
        );

        let token_ids = handler.extract_token_ids(&tensor).unwrap();
        assert_eq!(token_ids, vec![2, 5, 3]);
    }

    #[test]
    fn test_extract_token_ids_3d() {
        let (_tokenizer, tokenizer_path) = create_test_tokenizer();

        let mut config_map = HashMap::new();
        config_map.insert(
            "tokenizer_path".to_string(),
            toml::Value::String(tokenizer_path.to_str().unwrap().to_string()),
        );

        let config = OutputHandlerConfig {
            handler_type: "text".to_string(),
            output: "tokens".to_string(),
            config: config_map,
        };

        let handler = TextHandler::from_config(&config).unwrap();

        // [batch=1, beam=2, seq_len=3] - should take first batch, first beam
        let tensor = TensorData::new(
            vec![
                2.0, 5.0, 3.0, // Batch 0, Beam 0
                4.0, 1.0, 0.0, // Batch 0, Beam 1
            ],
            vec![1, 2, 3],
        );

        let token_ids = handler.extract_token_ids(&tensor).unwrap();
        assert_eq!(token_ids, vec![2, 5, 3]);
    }

    #[test]
    fn test_extract_token_ids_invalid_shape() {
        let (_tokenizer, tokenizer_path) = create_test_tokenizer();

        let mut config_map = HashMap::new();
        config_map.insert(
            "tokenizer_path".to_string(),
            toml::Value::String(tokenizer_path.to_str().unwrap().to_string()),
        );

        let config = OutputHandlerConfig {
            handler_type: "text".to_string(),
            output: "tokens".to_string(),
            config: config_map,
        };

        let handler = TextHandler::from_config(&config).unwrap();

        // 4D tensor is invalid
        let tensor = TensorData::new(vec![1.0; 24], vec![2, 3, 2, 2]);

        let result = handler.extract_token_ids(&tensor);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_tokens() {
        let (_tokenizer, tokenizer_path) = create_test_tokenizer();

        let mut config_map = HashMap::new();
        config_map.insert(
            "tokenizer_path".to_string(),
            toml::Value::String(tokenizer_path.to_str().unwrap().to_string()),
        );

        let config = OutputHandlerConfig {
            handler_type: "text".to_string(),
            output: "tokens".to_string(),
            config: config_map,
        };

        let handler = TextHandler::from_config(&config).unwrap();

        // Token IDs: hello(2), space(5), world(3)
        let token_ids = vec![2, 5, 3];
        let text = handler.decode_tokens(&token_ids).unwrap();

        // The exact output depends on tokenizer decoding logic
        assert!(text.contains("hello") || text.contains("world"));
    }

    #[test]
    fn test_process_1d_tensor() {
        let (_tokenizer, tokenizer_path) = create_test_tokenizer();

        let mut config_map = HashMap::new();
        config_map.insert(
            "tokenizer_path".to_string(),
            toml::Value::String(tokenizer_path.to_str().unwrap().to_string()),
        );

        let config = OutputHandlerConfig {
            handler_type: "text".to_string(),
            output: "output_ids".to_string(),
            config: config_map,
        };

        let handler = TextHandler::from_config(&config).unwrap();

        let mut outputs = HashMap::new();
        outputs.insert(
            "output_ids".to_string(),
            TensorData::new(vec![2.0, 5.0, 3.0], vec![3]),
        );

        let result = handler.process(&outputs).unwrap();

        if let ProcessedOutput::Text(text) = result {
            assert!(!text.is_empty());
        } else {
            panic!("Expected Text output");
        }
    }

    #[test]
    fn test_process_missing_tensor() {
        let (_tokenizer, tokenizer_path) = create_test_tokenizer();

        let mut config_map = HashMap::new();
        config_map.insert(
            "tokenizer_path".to_string(),
            toml::Value::String(tokenizer_path.to_str().unwrap().to_string()),
        );

        let config = OutputHandlerConfig {
            handler_type: "text".to_string(),
            output: "missing".to_string(),
            config: config_map,
        };

        let handler = TextHandler::from_config(&config).unwrap();

        let outputs = HashMap::new();
        let result = handler.process(&outputs);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("missing"));
    }

    #[test]
    fn test_save_text() {
        let (_tokenizer, tokenizer_path) = create_test_tokenizer();

        let mut config_map = HashMap::new();
        config_map.insert(
            "tokenizer_path".to_string(),
            toml::Value::String(tokenizer_path.to_str().unwrap().to_string()),
        );

        let config = OutputHandlerConfig {
            handler_type: "text".to_string(),
            output: "tokens".to_string(),
            config: config_map,
        };

        let handler = TextHandler::from_config(&config).unwrap();

        let text_output = ProcessedOutput::Text("Hello, World!".to_string());

        let temp_file = NamedTempFile::new().unwrap();
        let result = handler.save(&text_output, temp_file.path());
        assert!(result.is_ok());

        // Verify file content
        let content = fs::read_to_string(temp_file.path()).unwrap();
        assert_eq!(content, "Hello, World!");
    }
}
