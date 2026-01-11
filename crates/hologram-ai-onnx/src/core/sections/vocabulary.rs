//! Vocabulary section for WordPiece/BPE tokenizers.
//!
//! This module provides the [`VocabularySection`] type for embedding
//! token vocabularies in .holo bundles.

use super::error::{EmbedError, EmbedResult};
use super::traits::{EmbeddableSection, FromEmbeddedSection};
use std::collections::HashMap;

/// Vocabulary format type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VocabFormat {
    /// One token per line (BERT vocab.txt style).
    LineBased,
    /// JSON object mapping token -> id.
    Json,
}

/// Vocabulary section containing token-to-id mappings.
///
/// Supports both line-based format (one token per line, like BERT's vocab.txt)
/// and JSON format (token -> id mapping, like GPT-2's vocab.json).
///
/// # Example
///
/// ```rust,ignore
/// use hologram_ai_onnx::core::sections::VocabularySection;
///
/// // Create from line-based vocabulary
/// let tokens = vec!["[PAD]".to_string(), "[UNK]".to_string(), "hello".to_string()];
/// let vocab = VocabularySection::from_lines(tokens);
///
/// assert_eq!(vocab.len(), 3);
/// assert_eq!(vocab.get_token(0), Some("[PAD]"));
/// assert_eq!(vocab.get_id("hello"), Some(2));
/// ```
#[derive(Debug, Clone)]
pub struct VocabularySection {
    /// List of tokens in vocabulary order (index = token ID).
    tokens: Vec<String>,
    /// Format to use for serialization.
    format: VocabFormat,
}

impl VocabularySection {
    /// Create from line-based vocabulary (one token per line).
    ///
    /// This is the format used by BERT's vocab.txt files.
    /// The index of each token in the vector becomes its token ID.
    ///
    /// # Arguments
    /// * `tokens` - Vector of tokens in ID order
    ///
    /// # Example
    /// ```rust,ignore
    /// let vocab = VocabularySection::from_lines(vec![
    ///     "[PAD]".to_string(),
    ///     "[UNK]".to_string(),
    ///     "[CLS]".to_string(),
    ///     "[SEP]".to_string(),
    ///     "[MASK]".to_string(),
    /// ]);
    /// ```
    pub fn from_lines(tokens: Vec<String>) -> Self {
        Self {
            tokens,
            format: VocabFormat::LineBased,
        }
    }

    /// Create from JSON vocabulary string.
    ///
    /// Parses a JSON object where keys are tokens and values are token IDs.
    /// This is the format used by GPT-2's vocab.json files.
    ///
    /// # Arguments
    /// * `json_str` - JSON string containing token -> id mapping
    ///
    /// # Errors
    /// Returns an error if the JSON is invalid or cannot be parsed.
    ///
    /// # Example
    /// ```rust,ignore
    /// let json = r#"{"hello": 0, "world": 1, "!": 2}"#;
    /// let vocab = VocabularySection::from_json(json)?;
    /// ```
    pub fn from_json(json_str: &str) -> EmbedResult<Self> {
        let map: HashMap<String, usize> = serde_json::from_str(json_str)?;

        // Find the maximum ID to determine vector size
        let max_id = map.values().copied().max().unwrap_or(0);
        let mut tokens = vec![String::new(); max_id + 1];

        for (token, id) in map {
            if id < tokens.len() {
                tokens[id] = token;
            }
        }

        Ok(Self {
            tokens,
            format: VocabFormat::Json,
        })
    }

    /// Create from a HashMap of token -> id mappings.
    ///
    /// # Arguments
    /// * `map` - HashMap mapping tokens to their IDs
    pub fn from_map(map: HashMap<String, usize>) -> Self {
        let max_id = map.values().copied().max().unwrap_or(0);
        let mut tokens = vec![String::new(); max_id + 1];

        for (token, id) in map {
            if id < tokens.len() {
                tokens[id] = token;
            }
        }

        Self {
            tokens,
            format: VocabFormat::Json,
        }
    }

    /// Get token by ID.
    ///
    /// # Arguments
    /// * `id` - Token ID
    ///
    /// # Returns
    /// The token string if the ID exists, None otherwise.
    pub fn get_token(&self, id: usize) -> Option<&str> {
        self.tokens.get(id).map(|s| s.as_str())
    }

    /// Get ID by token.
    ///
    /// Performs a linear search through the vocabulary.
    /// For frequent lookups, consider building a reverse index.
    ///
    /// # Arguments
    /// * `token` - Token string to look up
    ///
    /// # Returns
    /// The token ID if found, None otherwise.
    pub fn get_id(&self, token: &str) -> Option<usize> {
        self.tokens.iter().position(|t| t == token)
    }

    /// Get vocabulary size.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Check if vocabulary is empty.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Get all tokens as a slice.
    pub fn tokens(&self) -> &[String] {
        &self.tokens
    }

    /// Build a reverse lookup map (token -> id).
    ///
    /// Useful for efficient token-to-id lookups.
    pub fn build_reverse_map(&self) -> HashMap<&str, usize> {
        self.tokens
            .iter()
            .enumerate()
            .map(|(id, token)| (token.as_str(), id))
            .collect()
    }

    /// Check if vocabulary contains a token.
    pub fn contains(&self, token: &str) -> bool {
        self.tokens.iter().any(|t| t == token)
    }

    /// Get the format used for serialization.
    pub fn format(&self) -> &'static str {
        match self.format {
            VocabFormat::LineBased => "line-based",
            VocabFormat::Json => "json",
        }
    }
}

impl EmbeddableSection for VocabularySection {
    fn section_id(&self) -> &'static str {
        "vocabulary"
    }

    fn to_bytes(&self) -> Vec<u8> {
        match self.format {
            VocabFormat::LineBased => self.tokens.join("\n").into_bytes(),
            VocabFormat::Json => {
                let map: HashMap<&str, usize> = self
                    .tokens
                    .iter()
                    .enumerate()
                    .map(|(i, t)| (t.as_str(), i))
                    .collect();
                serde_json::to_vec(&map).unwrap_or_default()
            }
        }
    }

    fn content_type(&self) -> &'static str {
        match self.format {
            VocabFormat::LineBased => "text/plain",
            VocabFormat::Json => "application/json",
        }
    }
}

impl FromEmbeddedSection for VocabularySection {
    const SECTION_ID: &'static str = "vocabulary";

    fn from_bytes(bytes: &[u8]) -> EmbedResult<Self> {
        let text = String::from_utf8(bytes.to_vec())?;

        // Try JSON first (starts with '{')
        if text.trim_start().starts_with('{') {
            return Self::from_json(&text);
        }

        // Fall back to line-based format
        let tokens: Vec<String> = text.lines().map(|s| s.to_string()).collect();
        Ok(Self::from_lines(tokens))
    }
}

/// JSON vocabulary section (explicit JSON format).
///
/// This is a variant of [`VocabularySection`] that always uses JSON format
/// for serialization, regardless of how it was created.
#[derive(Debug, Clone)]
pub struct VocabularyJsonSection {
    inner: VocabularySection,
}

impl VocabularyJsonSection {
    /// Create from a HashMap of token -> id mappings.
    pub fn from_map(map: HashMap<String, usize>) -> Self {
        Self {
            inner: VocabularySection::from_map(map),
        }
    }

    /// Create from JSON string.
    pub fn from_json(json_str: &str) -> EmbedResult<Self> {
        Ok(Self {
            inner: VocabularySection::from_json(json_str)?,
        })
    }

    /// Get the inner vocabulary section.
    pub fn inner(&self) -> &VocabularySection {
        &self.inner
    }
}

impl EmbeddableSection for VocabularyJsonSection {
    fn section_id(&self) -> &'static str {
        "vocabulary_json"
    }

    fn to_bytes(&self) -> Vec<u8> {
        let map: HashMap<&str, usize> = self
            .inner
            .tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.as_str(), i))
            .collect();
        serde_json::to_vec(&map).unwrap_or_default()
    }

    fn content_type(&self) -> &'static str {
        "application/json"
    }
}

impl FromEmbeddedSection for VocabularyJsonSection {
    const SECTION_ID: &'static str = "vocabulary_json";

    fn from_bytes(bytes: &[u8]) -> EmbedResult<Self> {
        let text = String::from_utf8(bytes.to_vec())?;
        Self::from_json(&text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_lines() {
        let tokens = vec![
            "[PAD]".to_string(),
            "[UNK]".to_string(),
            "[CLS]".to_string(),
            "[SEP]".to_string(),
            "hello".to_string(),
            "world".to_string(),
        ];
        let vocab = VocabularySection::from_lines(tokens);

        assert_eq!(vocab.len(), 6);
        assert_eq!(vocab.get_token(0), Some("[PAD]"));
        assert_eq!(vocab.get_token(4), Some("hello"));
        assert_eq!(vocab.get_id("[UNK]"), Some(1));
        assert_eq!(vocab.get_id("world"), Some(5));
        assert_eq!(vocab.get_id("missing"), None);
    }

    #[test]
    fn test_from_json() {
        let json = r#"{"hello": 0, "world": 1, "foo": 2}"#;
        let vocab = VocabularySection::from_json(json).unwrap();

        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get_token(0), Some("hello"));
        assert_eq!(vocab.get_token(1), Some("world"));
        assert_eq!(vocab.get_token(2), Some("foo"));
    }

    #[test]
    fn test_from_json_sparse_ids() {
        let json = r#"{"a": 0, "b": 5, "c": 10}"#;
        let vocab = VocabularySection::from_json(json).unwrap();

        assert_eq!(vocab.len(), 11);
        assert_eq!(vocab.get_token(0), Some("a"));
        assert_eq!(vocab.get_token(5), Some("b"));
        assert_eq!(vocab.get_token(10), Some("c"));
        // Gaps should be empty strings
        assert_eq!(vocab.get_token(1), Some(""));
    }

    #[test]
    fn test_from_map() {
        let mut map = HashMap::new();
        map.insert("token_a".to_string(), 0);
        map.insert("token_b".to_string(), 1);
        map.insert("token_c".to_string(), 2);

        let vocab = VocabularySection::from_map(map);

        assert_eq!(vocab.len(), 3);
        assert!(vocab.contains("token_a"));
        assert!(vocab.contains("token_b"));
        assert!(vocab.contains("token_c"));
    }

    #[test]
    fn test_serialization_line_based() {
        let tokens = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let vocab = VocabularySection::from_lines(tokens);

        let bytes = vocab.to_bytes();
        assert_eq!(bytes, b"a\nb\nc");
        assert_eq!(vocab.content_type(), "text/plain");
    }

    #[test]
    fn test_serialization_json() {
        let json = r#"{"x": 0, "y": 1}"#;
        let vocab = VocabularySection::from_json(json).unwrap();

        let bytes = vocab.to_bytes();
        // JSON output, parse it back
        let parsed: HashMap<String, usize> = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(parsed.get("x"), Some(&0));
        assert_eq!(parsed.get("y"), Some(&1));
        assert_eq!(vocab.content_type(), "application/json");
    }

    #[test]
    fn test_roundtrip_line_based() {
        let original = VocabularySection::from_lines(vec![
            "token1".to_string(),
            "token2".to_string(),
            "token3".to_string(),
        ]);

        let bytes = original.to_bytes();
        let restored = VocabularySection::from_bytes(&bytes).unwrap();

        assert_eq!(original.len(), restored.len());
        for i in 0..original.len() {
            assert_eq!(original.get_token(i), restored.get_token(i));
        }
    }

    #[test]
    fn test_roundtrip_json() {
        let original = VocabularySection::from_json(r#"{"a": 0, "b": 1, "c": 2}"#).unwrap();

        let bytes = original.to_bytes();
        let restored = VocabularySection::from_bytes(&bytes).unwrap();

        assert_eq!(original.len(), restored.len());
        for i in 0..original.len() {
            assert_eq!(original.get_token(i), restored.get_token(i));
        }
    }

    #[test]
    fn test_empty_vocabulary() {
        let vocab = VocabularySection::from_lines(vec![]);
        assert!(vocab.is_empty());
        assert_eq!(vocab.len(), 0);
        assert_eq!(vocab.get_token(0), None);
    }

    #[test]
    fn test_build_reverse_map() {
        let vocab = VocabularySection::from_lines(vec![
            "foo".to_string(),
            "bar".to_string(),
            "baz".to_string(),
        ]);

        let reverse = vocab.build_reverse_map();
        assert_eq!(reverse.get("foo"), Some(&0));
        assert_eq!(reverse.get("bar"), Some(&1));
        assert_eq!(reverse.get("baz"), Some(&2));
    }

    #[test]
    fn test_section_id() {
        let vocab = VocabularySection::from_lines(vec!["test".to_string()]);
        assert_eq!(vocab.section_id(), "vocabulary");
        assert_eq!(VocabularySection::SECTION_ID, "vocabulary");
    }

    #[test]
    fn test_contains() {
        let vocab = VocabularySection::from_lines(vec![
            "hello".to_string(),
            "world".to_string(),
        ]);

        assert!(vocab.contains("hello"));
        assert!(vocab.contains("world"));
        assert!(!vocab.contains("missing"));
    }

    #[test]
    fn test_tokens_accessor() {
        let tokens = vec!["a".to_string(), "b".to_string()];
        let vocab = VocabularySection::from_lines(tokens.clone());

        assert_eq!(vocab.tokens(), &tokens);
    }

    #[test]
    fn test_invalid_json() {
        let result = VocabularySection::from_json("not valid json");
        assert!(result.is_err());
    }

    #[test]
    fn test_vocabulary_json_section() {
        let json = r#"{"token1": 0, "token2": 1}"#;
        let vocab = VocabularyJsonSection::from_json(json).unwrap();

        assert_eq!(vocab.section_id(), "vocabulary_json");
        assert_eq!(vocab.content_type(), "application/json");
        assert_eq!(vocab.inner().len(), 2);
    }

    #[test]
    fn test_large_vocabulary() {
        // Test with a vocabulary similar to BERT (30k+ tokens)
        let tokens: Vec<String> = (0..30000).map(|i| format!("token_{}", i)).collect();
        let vocab = VocabularySection::from_lines(tokens);

        assert_eq!(vocab.len(), 30000);
        assert_eq!(vocab.get_token(0), Some("token_0"));
        assert_eq!(vocab.get_token(29999), Some("token_29999"));
        assert_eq!(vocab.get_id("token_15000"), Some(15000));
    }
}
