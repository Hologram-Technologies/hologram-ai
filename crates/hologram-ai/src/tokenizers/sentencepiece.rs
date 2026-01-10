//! SentencePiece tokenizer implementation.
//!
//! Implements the Unigram language model algorithm used by T5, ALBERT, XLNet, and other models.
//! This is a pure Rust implementation with no external dependencies, following the hologram
//! principle of "everything runs through hologram."
//!
//! # Algorithm
//!
//! SentencePiece Unigram uses a probabilistic model to find the optimal tokenization:
//! 1. Each token has a score (log probability)
//! 2. Use Viterbi algorithm to find the highest-scoring tokenization
//! 3. Dynamic programming: best_score[i] = max(best_score[j] + token_score(text[j..i]))
//!
//! # Bridge Implementation
//!
//! This is a pure Rust runtime implementation that serves as a bridge until hologram_ir
//! gains the necessary operations (string/byte manipulation, Gather for vocab lookups).
//! The tokenizer still compiles to .holo format for the architecture, but runtime execution
//! uses this Rust implementation.

use super::Tokenizer;
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::Path;

/// SentencePiece tokenizer using Unigram language model.
#[derive(Debug, Clone)]
pub struct SentencePieceTokenizer {
    /// Token to ID mapping
    vocab: HashMap<String, u32>,

    /// ID to token mapping
    id_to_token: HashMap<u32, String>,

    /// Token scores (log probabilities) for Unigram algorithm
    token_scores: HashMap<String, f32>,

    /// Prefix trie for efficient token matching
    trie: PrefixTrie,

    /// Special tokens
    pad_token_id: u32,
    eos_token_id: u32,
    unk_token_id: u32,
}

/// Prefix trie for efficient token lookup.
///
/// Stores vocabulary tokens in a trie structure to quickly find all tokens
/// that start with a given prefix. This is essential for the Unigram algorithm
/// which needs to try all possible tokens at each position.
#[derive(Debug, Clone)]
struct PrefixTrie {
    children: HashMap<char, PrefixTrie>,
    /// Token ID if this node represents a complete token
    token_id: Option<u32>,
}

impl PrefixTrie {
    /// Create a new empty trie.
    fn new() -> Self {
        Self {
            children: HashMap::new(),
            token_id: None,
        }
    }

    /// Insert a token into the trie.
    fn insert(&mut self, token: &str, token_id: u32) {
        let mut node = self;
        for ch in token.chars() {
            node = node.children.entry(ch).or_insert_with(PrefixTrie::new);
        }
        node.token_id = Some(token_id);
    }

    /// Find all tokens that start at the given position in the text.
    ///
    /// Returns a vector of (token_length, token_id) pairs.
    fn find_matches(&self, text: &str, start_pos: usize) -> Vec<(usize, u32)> {
        let mut matches = Vec::new();
        let mut node = self;
        let chars: Vec<char> = text[start_pos..].chars().collect();

        for (i, &ch) in chars.iter().enumerate() {
            // Check if current position is a complete token
            if let Some(token_id) = node.token_id {
                matches.push((i, token_id));
            }

            // Try to continue matching
            if let Some(next_node) = node.children.get(&ch) {
                node = next_node;
            } else {
                break;
            }
        }

        // Check final position
        if let Some(token_id) = node.token_id {
            matches.push((chars.len(), token_id));
        }

        matches
    }
}

/// Viterbi path node for tracking best tokenization.
#[derive(Debug, Clone)]
struct ViterbiNode {
    /// Best score to reach this position
    score: f32,
    /// Token ID that ends at this position
    token_id: u32,
    /// Position where the token started
    start_pos: usize,
}

impl SentencePieceTokenizer {
    /// Load SentencePiece tokenizer from tokenizer.json file.
    ///
    /// Parses the vocabulary and scores from a Hugging Face tokenizer.json file
    /// and builds internal data structures for efficient tokenization.
    pub fn from_file(path: &Path) -> Result<Self> {
        // Parse vocabulary with scores
        let vocab_data = super::compiler::parse_tokenizer_vocab(path)?;

        // Load scores from tokenizer.json
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read tokenizer file: {}", path.display()))?;

        let json: serde_json::Value = serde_json::from_str(&content)
            .with_context(|| "Failed to parse tokenizer.json")?;

        // Extract scores from model.vocab array
        let mut token_scores = HashMap::new();
        if let Some(vocab_array) = json
            .get("model")
            .and_then(|m| m.get("vocab"))
            .and_then(|v| v.as_array())
        {
            for entry in vocab_array {
                if let Some(pair) = entry.as_array()
                    && pair.len() >= 2
                    && let (Some(token), Some(score)) =
                        (pair[0].as_str(), pair[1].as_f64())
                {
                    token_scores.insert(token.to_string(), score as f32);
                }
            }
        }

        // Build prefix trie for efficient matching
        let mut trie = PrefixTrie::new();
        for (token, &token_id) in &vocab_data.token_to_id {
            trie.insert(token, token_id);
        }

        // Detect special token IDs
        let pad_token_id = vocab_data.token_to_id.get("<pad>").copied().unwrap_or(0);
        let eos_token_id = vocab_data.token_to_id.get("</s>").copied().unwrap_or(1);
        let unk_token_id = vocab_data.token_to_id.get("<unk>").copied().unwrap_or(2);

        Ok(Self {
            vocab: vocab_data.token_to_id,
            id_to_token: vocab_data.id_to_token,
            token_scores,
            trie,
            pad_token_id,
            eos_token_id,
            unk_token_id,
        })
    }

    /// Tokenize text using Unigram language model with Viterbi algorithm.
    ///
    /// Finds the optimal tokenization by maximizing the sum of token scores (log probabilities).
    /// Uses dynamic programming to efficiently compute the best path.
    ///
    /// # Algorithm
    ///
    /// 1. Normalize text (add ▁ for spaces in SentencePiece)
    /// 2. For each position, try all possible tokens
    /// 3. Use Viterbi to find highest-scoring path
    /// 4. Backtrack to recover token sequence
    fn tokenize_unigram(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return vec![];
        }

        // Normalize text: replace spaces with ▁ (SentencePiece marker)
        let normalized = format!("▁{}", text.replace(' ', "▁"));
        let char_count = normalized.chars().count();

        if char_count == 0 {
            return vec![];
        }

        // Initialize Viterbi lattice
        // best[i] = best path to reach position i
        let mut best: Vec<Option<ViterbiNode>> = vec![None; char_count + 1];
        best[0] = Some(ViterbiNode {
            score: 0.0,
            token_id: 0, // Dummy
            start_pos: 0,
        });

        // Forward pass: compute best score for each position
        let normalized_chars: Vec<char> = normalized.chars().collect();

        for i in 0..char_count {
            if best[i].is_none() {
                continue;
            }

            let current_score = best[i].as_ref().unwrap().score;

            // Try all tokens that start at position i
            let byte_offset = normalized_chars[..i].iter().collect::<String>().len();
            let matches = self.trie.find_matches(&normalized, byte_offset);

            if matches.is_empty() {
                // No matches: use unknown token
                let unk_score = self.token_scores.get("<unk>").copied().unwrap_or(-10.0);
                let new_score = current_score + unk_score;

                if i < char_count
                    && (best[i + 1].is_none() || new_score > best[i + 1].as_ref().unwrap().score)
                {
                    best[i + 1] = Some(ViterbiNode {
                        score: new_score,
                        token_id: self.unk_token_id,
                        start_pos: i,
                    });
                }
            } else {
                // Try each matching token
                for (_token_len, token_id) in matches {
                    if let Some(token_str) = self.id_to_token.get(&token_id) {
                        let token_score = self.token_scores.get(token_str).copied().unwrap_or(-10.0);
                        let token_char_len = token_str.chars().count();
                        let next_pos = i + token_char_len;

                        if next_pos <= char_count {
                            let new_score = current_score + token_score;

                            if best[next_pos].is_none()
                                || new_score > best[next_pos].as_ref().unwrap().score
                            {
                                best[next_pos] = Some(ViterbiNode {
                                    score: new_score,
                                    token_id,
                                    start_pos: i,
                                });
                            }
                        }
                    }
                }
            }
        }

        // Backward pass: reconstruct best path
        let mut tokens = Vec::new();
        let mut pos = char_count;

        while pos > 0 {
            if let Some(node) = &best[pos] {
                if node.token_id != 0 {
                    // Not dummy
                    tokens.push(node.token_id);
                }
                pos = node.start_pos;
            } else {
                // Fallback: use unknown token and step back
                tokens.push(self.unk_token_id);
                pos = pos.saturating_sub(1);
            }
        }

        tokens.reverse();
        tokens
    }
}

impl Tokenizer for SentencePieceTokenizer {
    fn encode(&self, text: &str, max_length: usize) -> Result<Vec<u32>> {
        let mut tokens = self.tokenize_unigram(text);

        // Truncate to max_length
        tokens.truncate(max_length);

        // Pad to max_length
        while tokens.len() < max_length {
            tokens.push(self.pad_token_id);
        }

        Ok(tokens)
    }

    fn decode(&self, token_ids: &[u32]) -> Result<String> {
        let mut text = String::new();

        for &token_id in token_ids {
            // Skip padding tokens
            if token_id == self.pad_token_id {
                continue;
            }

            // Stop at EOS tokens
            if token_id == self.eos_token_id {
                break;
            }

            if let Some(token) = self.id_to_token.get(&token_id) {
                // Remove SentencePiece underscore prefix (▁ = space)
                let decoded = token.replace('▁', " ");
                text.push_str(&decoded);
            }
        }

        // Clean up extra spaces
        let text = text.trim().to_string();

        Ok(text)
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn tokenizer_type(&self) -> &str {
        "sentencepiece"
    }

    fn pad_token_id(&self) -> u32 {
        self.pad_token_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefix_trie() {
        let mut trie = PrefixTrie::new();
        trie.insert("hello", 1);
        trie.insert("hell", 2);
        trie.insert("he", 3);

        // Test finding matches
        let text = "hello world";
        let matches = trie.find_matches(text, 0);

        // Should find "he", "hell", "hello" at position 0
        assert!(matches.len() >= 2);
        assert!(matches.iter().any(|(_, id)| *id == 3)); // "he"
        assert!(matches.iter().any(|(_, id)| *id == 1)); // "hello"
    }

    #[test]
    fn test_prefix_trie_no_match() {
        let mut trie = PrefixTrie::new();
        trie.insert("hello", 1);

        let text = "world";
        let matches = trie.find_matches(text, 0);

        // Should find no matches
        assert_eq!(matches.len(), 0);
    }

    #[test]
    #[ignore = "Requires external tokenizer.json fixture"]
    fn test_sentencepiece_encode_decode() {
        let tokenizer =
            SentencePieceTokenizer::from_file(Path::new("models/t5-small/tokenizer.json"))
                .unwrap();

        let text = "Hello world";
        let tokens = tokenizer.encode(text, 20).unwrap();
        let decoded = tokenizer.decode(&tokens).unwrap();

        println!("Original: {}", text);
        println!("Tokens: {:?}", tokens);
        println!("Decoded: {}", decoded);

        // Decoded should be similar (may have slight differences due to tokenization)
        assert!(!decoded.is_empty());
    }

    #[test]
    #[ignore = "Requires external tokenizer.json fixture"]
    fn test_sentencepiece_unigram() {
        let tokenizer =
            SentencePieceTokenizer::from_file(Path::new("models/t5-small/tokenizer.json"))
                .unwrap();

        let text = "Tell me a joke about programming";
        let tokens = tokenizer.tokenize_unigram(text);

        println!("Text: {}", text);
        println!("Tokens: {:?}", tokens);
        println!("Token count: {}", tokens.len());

        // Should produce reasonable number of tokens (not all unknown)
        assert!(!tokens.is_empty());
        let unk_count = tokens.iter().filter(|&&t| t == tokenizer.unk_token_id).count();
        let unk_ratio = unk_count as f32 / tokens.len() as f32;

        // Less than 50% unknown tokens indicates working tokenization
        assert!(
            unk_ratio < 0.5,
            "Too many unknown tokens: {}/{}",
            unk_count,
            tokens.len()
        );
    }

    #[test]
    #[ignore = "Requires external tokenizer.json fixture"]
    fn test_vocab_loading() {
        let tokenizer =
            SentencePieceTokenizer::from_file(Path::new("models/t5-small/tokenizer.json"))
                .unwrap();

        // Check special tokens
        assert_eq!(tokenizer.pad_token_id, 0);
        assert_eq!(tokenizer.eos_token_id, 1);
        assert_eq!(tokenizer.unk_token_id, 2);

        // Check vocab size
        assert_eq!(tokenizer.vocab_size(), 32100);

        // Check that scores were loaded
        assert!(!tokenizer.token_scores.is_empty());
        assert!(tokenizer.token_scores.contains_key("<pad>"));
        assert!(tokenizer.token_scores.contains_key("▁"));
    }
}
