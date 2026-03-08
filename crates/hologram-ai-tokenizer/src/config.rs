//! Tokenizer configuration types.

use crate::vocab::{MergeRules, VocabTable};
use std::collections::HashMap;

/// Top-level tokenizer configuration.
pub struct TokenizerConfig {
    pub algorithm: TokenizerAlgorithm,
    pub special_tokens: SpecialTokens,
    pub normalization: NormalizationConfig,
    pub pre_tokenizer: PreTokenizerConfig,
    pub byte_fallback: bool,
    pub add_bos: bool,
    pub add_eos: bool,
}

/// Tokenizer algorithm variant.
pub enum TokenizerAlgorithm {
    Bpe {
        vocab: VocabTable,
        merges: MergeRules,
    },
    /// SentencePiece Unigram (Viterbi segmentation).
    Unigram {
        vocab: VocabTable,
        /// (token_bytes, log_probability) pairs for Viterbi scoring.
        scores: Vec<f32>,
    },
    /// WordPiece (greedy longest-prefix match).
    WordPiece {
        vocab: VocabTable,
        /// Prefix for continuing subword tokens (typically "##").
        continuing_subword_prefix: String,
        /// Maximum characters per word before falling back to unk.
        max_input_chars_per_word: usize,
    },
}

/// Special token IDs.
pub struct SpecialTokens {
    pub bos_id: Option<u32>,
    pub eos_id: u32,
    pub pad_id: Option<u32>,
    pub unk_id: Option<u32>,
    /// Additional special tokens, e.g. `<|im_start|>` → id.
    pub additional: HashMap<String, u32>,
}

/// Pre-tokenization configuration.
#[derive(Clone)]
pub enum PreTokenizerConfig {
    /// No pre-tokenization — input is used as-is.
    None,
    /// Metaspace pre-tokenizer (SentencePiece convention).
    /// Replaces spaces with `replacement` char and optionally prepends it.
    Metaspace { replacement: char, prepend: bool },
    /// Regex-based pre-tokenization (GPT-2 / LLaMA-3 style).
    Regex(String),
}

/// Pre-tokenization text normalization.
pub enum NormalizationConfig {
    None,
    Nfc,
    Nfkc,
    /// Prepend a space to the input (SentencePiece convention).
    PrependSpace,
    /// Custom sequence of normalization steps.
    Sequence(Vec<NormStep>),
}

/// Individual normalization step.
pub enum NormStep {
    Nfc,
    Nfkc,
    Lowercase,
    StripAccents,
    PrependSpace,
    Replace {
        pattern: String,
        replacement: String,
    },
}
