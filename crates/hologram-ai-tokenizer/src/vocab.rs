//! Vocabulary table and merge rules for BPE tokenization.

use alloc::string::String;
use alloc::vec::Vec;
use hashbrown::HashMap;

/// Bidirectional mapping between token IDs and token byte sequences.
///
/// Tokens are stored as raw bytes because BPE byte-fallback tokens
/// (e.g. `<0xNN>`) are arbitrary byte sequences that may not be valid UTF-8.
pub struct VocabTable {
    /// Token bytes indexed by token ID.
    pub id_to_token: Vec<Vec<u8>>,
    /// Reverse mapping from token bytes to ID.
    pub token_to_id: HashMap<Vec<u8>, u32>,
    /// Cached UTF-8 string for each token (None if not valid UTF-8).
    id_to_str: Vec<Option<String>>,
}

impl VocabTable {
    /// Build a vocab table from an ordered list of token byte sequences.
    pub fn new(tokens: Vec<Vec<u8>>) -> Self {
        let token_to_id: HashMap<Vec<u8>, u32> = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
        let id_to_str: Vec<Option<String>> = tokens
            .iter()
            .map(|t| String::from_utf8(t.clone()).ok())
            .collect();
        Self {
            id_to_token: tokens,
            token_to_id,
            id_to_str,
        }
    }

    /// Build from a string→id map (HuggingFace tokenizer.json `model.vocab`).
    pub fn from_vocab_map(vocab: &HashMap<String, u32>) -> Self {
        let max_id = vocab.values().copied().max().unwrap_or(0);
        let mut tokens = vec![Vec::new(); (max_id + 1) as usize];
        for (token, &id) in vocab {
            tokens[id as usize] = token.as_bytes().to_vec();
        }
        Self::new(tokens)
    }

    pub fn len(&self) -> usize {
        self.id_to_token.len()
    }

    pub fn is_empty(&self) -> bool {
        self.id_to_token.is_empty()
    }

    /// Get the string representation of a token ID (returns None if
    /// the ID is out of range or the token is not valid UTF-8).
    pub fn id_to_str(&self, id: u32) -> Option<&str> {
        self.id_to_str.get(id as usize)?.as_deref()
    }

    /// Look up a token ID by its string representation.
    pub fn str_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token.as_bytes()).copied()
    }
}

/// Ordered list of BPE merge pairs.
///
/// Lower index = higher priority. During encoding, the pair with the
/// lowest index (rank) is merged first.
pub struct MergeRules {
    pub merges: Vec<(Vec<u8>, Vec<u8>)>,
}

impl MergeRules {
    /// Parse merge rules from HuggingFace format (list of `"token1 token2"` strings).
    pub fn from_hf_merges(merges: &[String]) -> Self {
        let parsed = merges
            .iter()
            .filter_map(|m| {
                let (a, b) = m.split_once(' ')?;
                Some((a.as_bytes().to_vec(), b.as_bytes().to_vec()))
            })
            .collect();
        Self { merges: parsed }
    }

    /// Parse merges from JSON values — handles both `["a", "b"]` arrays
    /// and `"a b"` strings (both occur in HuggingFace tokenizer.json files).
    #[cfg(feature = "std")]
    pub fn from_json_merges(merges: &[serde_json::Value]) -> Self {
        let parsed = merges
            .iter()
            .filter_map(|v| {
                if let Some(arr) = v.as_array() {
                    // ["a", "b"] format
                    let a = arr.first()?.as_str()?;
                    let b = arr.get(1)?.as_str()?;
                    Some((a.as_bytes().to_vec(), b.as_bytes().to_vec()))
                } else if let Some(s) = v.as_str() {
                    // "a b" format
                    let (a, b) = s.split_once(' ')?;
                    Some((a.as_bytes().to_vec(), b.as_bytes().to_vec()))
                } else {
                    None
                }
            })
            .collect();
        Self { merges: parsed }
    }
}
