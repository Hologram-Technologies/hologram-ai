//! Tokenizer implementations that compile to hologram IR.
//!
//! This module provides tokenizers as first-class hologram operations,
//! allowing them to be compiled to .holo files and executed on the
//! hologram backend with SIMD optimizations.
//!
//! Supported tokenizer types:
//! - SentencePiece (Unigram) - used by T5, ALBERT, XLNet

pub mod compiler;
pub mod sentencepiece;

use anyhow::Result;
use std::path::Path;

/// Tokenizer configuration from pipeline config.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct TokenizerConfig {
    /// Tokenizer type (sentencepiece)
    #[serde(rename = "type")]
    pub tokenizer_type: String,

    /// Path to vocabulary file (usually tokenizer.json)
    pub vocab_path: String,

    /// Path to merges file (BPE only)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub merges_path: Option<String>,

    /// Maximum sequence length
    pub max_length: usize,

    /// Padding token ID
    pub pad_token_id: u32,

    /// End-of-sequence token ID
    pub eos_token_id: u32,

    /// Unknown token ID
    pub unk_token_id: u32,

    /// Beginning-of-sequence token ID (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bos_token_id: Option<u32>,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            tokenizer_type: "sentencepiece".to_string(),
            vocab_path: String::new(),
            merges_path: None,
            max_length: 512,
            pad_token_id: 0,
            eos_token_id: 1,
            unk_token_id: 2,
            bos_token_id: None,
        }
    }
}

/// Tokenizer trait - all tokenizers implement this.
pub trait Tokenizer: Send + Sync {
    /// Tokenize text into token IDs.
    fn encode(&self, text: &str, max_length: usize) -> Result<Vec<u32>>;

    /// Decode token IDs back to text.
    fn decode(&self, token_ids: &[u32]) -> Result<String>;

    /// Get vocabulary size.
    fn vocab_size(&self) -> usize;

    /// Get tokenizer type name.
    fn tokenizer_type(&self) -> &str;

    /// Get the padding token ID.
    fn pad_token_id(&self) -> u32;

    /// Get the end-of-sequence token ID.
    fn eos_token_id(&self) -> u32;

    /// Get the unknown token ID.
    fn unk_token_id(&self) -> u32;

    /// Get the list of special token IDs (e.g., <extra_id_0> through <extra_id_99> for T5).
    fn special_token_ids(&self) -> &[u32];
}

/// Load a tokenizer from configuration.
pub fn load_tokenizer(config: &TokenizerConfig) -> Result<Box<dyn Tokenizer>> {
    match config.tokenizer_type.as_str() {
        "sentencepiece" => {
            let tokenizer =
                sentencepiece::SentencePieceTokenizer::from_file(Path::new(&config.vocab_path))?;
            Ok(Box::new(tokenizer))
        }
        _ => Err(anyhow::anyhow!(
            "Unsupported tokenizer type: {} (supported: sentencepiece)",
            config.tokenizer_type
        )),
    }
}

/// Compile a tokenizer to a .holo file for fast execution.
///
/// This creates a hologram IR graph that performs tokenization using
/// lookup tables and optimized operations, then compiles it to a .holo file.
pub fn compile_tokenizer(config: &TokenizerConfig, output_path: &Path) -> Result<()> {
    compiler::compile_tokenizer_to_holo(config, output_path)
}

/// Compile a tokenizer to a HOLB bundle for pipeline bundling.
///
/// This creates a unified bundle (.holo file with HOLB format) that can be
/// combined with other HOLB bundles into a HOLM pipeline bundle.
pub fn compile_tokenizer_to_bundle(config: &TokenizerConfig, output_path: &Path) -> Result<()> {
    compiler::compile_tokenizer_to_bundle(config, output_path)
}
