//! Native tokenizer for hologram-ai.
//!
//! Provides BPE tokenization with support for HuggingFace `tokenizer.json`
//! format. Tokenizer data can be embedded in `.holo` archives (Phase 2).

mod bpe;
mod config;
mod native;
mod unigram;
mod vocab;
mod wordpiece;

pub use config::{
    NormalizationConfig, NormStep, PreTokenizerConfig, SpecialTokens, TokenizerAlgorithm,
    TokenizerConfig,
};
pub use native::NativeTokenizer;
pub use vocab::{MergeRules, VocabTable};

/// Tokenizer trait for text ↔ token ID conversion.
///
/// Object-safe: `Box<dyn Tokenizer>` and `Arc<dyn Tokenizer>` work.
pub trait Tokenizer: Send + Sync {
    /// Encode text into token IDs.
    fn encode(&self, text: &str) -> Vec<u32>;

    /// Decode token IDs back to text.
    fn decode(&self, tokens: &[u32]) -> String;

    /// End-of-sequence token ID.
    fn eos_token_id(&self) -> u32;

    /// Beginning-of-sequence token ID, if the model uses one.
    fn bos_token_id(&self) -> Option<u32>;

    /// Total vocabulary size.
    fn vocab_size(&self) -> usize;

    /// Look up the string representation of a token ID.
    fn id_to_token(&self, id: u32) -> Option<&str>;

    /// Look up the token ID for a string token.
    fn token_to_id(&self, token: &str) -> Option<u32>;
}
