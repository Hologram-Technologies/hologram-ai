//! Native tokenizer for hologram-ai.
//!
//! Provides BPE / Unigram / WordPiece tokenization. Split the way the rest of
//! hologram-ai is (architecture §2/§3):
//!
//! - **Runtime core** (`no_std` + `alloc`, on-device) — the `Tokenizer` trait
//!   and the encode/decode path (`NativeTokenizer` + the algorithm encoders,
//!   `VocabTable`, `MergeRules`). Builds on wasm / embedded. Byte-level
//!   pre-tokenization uses `regex-automata` (no_std).
//! - **Host shell** (`std` feature) — loading a HuggingFace `tokenizer.json`
//!   and embedding/reading `.holo` archive sections (serde_json, rkyv, file
//!   I/O). Never linked into a no_std build.
#![no_std]

#[macro_use]
extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

use alloc::string::String;
use alloc::vec::Vec;

#[cfg(feature = "std")]
pub mod archive;
mod bpe;
mod config;
mod native;
mod sentencepiece;
mod streaming;
mod unigram;
mod vocab;
mod wordpiece;

pub use config::{
    NormStep, NormalizationConfig, PreTokenizerConfig, SpecialTokens, TokenizerAlgorithm,
    TokenizerConfig,
};
pub use native::NativeTokenizer;
pub use streaming::StreamingDecoder;
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
