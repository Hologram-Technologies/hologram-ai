//! Tokenizer archive section for `.holo` archives.
//!
//! Packs vocabulary, merge rules, and scores into an rkyv-serialized
//! section for zero-copy access from memory-mapped archives.

use crate::tokenizer_json::infer_model_type;
use anyhow::Context;
use hologram_ai_common::exec_context::{Section, SECTION_CUSTOM_BASE};
use std::collections::HashMap;
use std::path::Path;
use std::string::{String, ToString};
use std::vec::Vec;

/// Section kind for tokenizer data.
pub const SECTION_TOKENIZER: u32 = SECTION_CUSTOM_BASE + 0x01;

/// Tokenizer data embedded in a `.holo` archive.
///
/// Zero-copy deserializable via rkyv — can be read directly from a
/// memory-mapped archive without allocation.
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct TokenizerSectionData {
    /// Vocabulary tokens as UTF-8 strings (indexed by token ID).
    pub vocab: Vec<String>,
    /// BPE merge rules as `"token1 token2"` strings.
    pub merges: Vec<String>,
    /// Unigram/SentencePiece scores per token (empty if BPE-only).
    pub scores: Vec<f32>,
    /// Special token mappings (e.g. "eos" → token_id).
    pub special_tokens: Vec<(String, u32)>,
}

impl TokenizerSectionData {
    /// Build from vocabulary table and merge rules.
    pub fn new(
        vocab: Vec<String>,
        merges: Vec<String>,
        scores: Vec<f32>,
        special_tokens: HashMap<String, u32>,
    ) -> Self {
        let mut special: Vec<(String, u32)> = special_tokens.into_iter().collect();
        special.sort_by_key(|(_, id)| *id);
        Self {
            vocab,
            merges,
            scores,
            special_tokens: special,
        }
    }

    /// Build from a HuggingFace `tokenizer.json` file.
    ///
    /// Extracts vocab, merges, scores, and special tokens directly from the
    /// JSON without constructing a full `NativeTokenizer`.
    pub fn from_tokenizer_json(path: &Path) -> anyhow::Result<Self> {
        let data = std::fs::read_to_string(path)
            .with_context(|| format!("reading tokenizer file: {}", path.display()))?;
        let json: serde_json::Value =
            serde_json::from_str(&data).context("parsing tokenizer JSON")?;

        let model = &json["model"];
        let model_type = infer_model_type(model)?;

        // Extract vocab as ordered Vec<String> indexed by token ID.
        let vocab: Vec<String> = match model_type {
            "BPE" | "WordPiece" => {
                let vocab_obj = model["vocab"].as_object().context("missing model.vocab")?;
                let mut pairs: Vec<(u32, String)> = vocab_obj
                    .iter()
                    .map(|(k, v)| (v.as_u64().unwrap_or(0) as u32, k.clone()))
                    .collect();
                pairs.sort_by_key(|(id, _)| *id);
                pairs.into_iter().map(|(_, tok)| tok).collect()
            }
            "Unigram" => {
                let arr = model["vocab"]
                    .as_array()
                    .context("missing model.vocab for Unigram")?;
                arr.iter()
                    .map(|e| e[0].as_str().unwrap_or("").to_string())
                    .collect()
            }
            _ => vec![],
        };

        // Extract merges (BPE only).
        let merges: Vec<String> = model
            .get("merges")
            .and_then(|m| m.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| {
                        if let Some(s) = v.as_str() {
                            Some(s.to_string())
                        } else if let Some(a) = v.as_array() {
                            Some(format!("{} {}", a.first()?.as_str()?, a.get(1)?.as_str()?))
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();

        // Extract scores (Unigram only).
        let scores: Vec<f32> = if model_type == "Unigram" {
            model["vocab"]
                .as_array()
                .unwrap_or(&vec![])
                .iter()
                .map(|e| e[1].as_f64().unwrap_or(0.0) as f32)
                .collect()
        } else {
            vec![]
        };

        // Extract special tokens from added_tokens.
        let mut special = HashMap::new();
        if let Some(added) = json.get("added_tokens").and_then(|v| v.as_array()) {
            for t in added {
                if t["special"].as_bool().unwrap_or(false) {
                    let content = t["content"].as_str().unwrap_or("");
                    let id = t["id"].as_u64().unwrap_or(0) as u32;
                    if !content.is_empty() {
                        special.insert(content.to_string(), id);
                    }
                }
            }
        }

        Ok(Self::new(vocab, merges, scores, special))
    }

    /// Zero-copy access from raw bytes (e.g. memory-mapped archive section).
    pub fn from_bytes(bytes: &[u8]) -> Result<&ArchivedTokenizerSectionData, rkyv::rancor::Error> {
        rkyv::access::<ArchivedTokenizerSectionData, rkyv::rancor::Error>(bytes)
    }

    /// Deserialize from raw bytes into an owned `TokenizerSectionData`.
    pub fn deserialize_from(bytes: &[u8]) -> Result<Self, rkyv::rancor::Error> {
        rkyv::from_bytes::<Self, rkyv::rancor::Error>(bytes)
    }
}

impl Section for TokenizerSectionData {
    fn section_kind(&self) -> u32 {
        SECTION_TOKENIZER
    }

    fn to_bytes(&self) -> Vec<u8> {
        rkyv::to_bytes::<rkyv::rancor::Error>(self)
            .expect("TokenizerSectionData serialization")
            .to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_rkyv() {
        let mut special = HashMap::new();
        special.insert("eos".into(), 2u32);
        special.insert("bos".into(), 1);

        let section = TokenizerSectionData::new(
            vec!["<unk>".into(), "<s>".into(), "</s>".into(), "hello".into()],
            vec!["h e".into(), "l l".into()],
            vec![0.0, 0.0, 0.0, -1.5],
            special,
        );
        assert_eq!(section.section_kind(), SECTION_TOKENIZER);

        let bytes = section.to_bytes();

        // Zero-copy access.
        let archived = TokenizerSectionData::from_bytes(&bytes).unwrap();
        assert_eq!(archived.vocab.len(), 4);
        assert_eq!(archived.vocab[3].as_str(), "hello");
        assert_eq!(archived.merges.len(), 2);
        assert_eq!(archived.special_tokens.len(), 2);

        // Full deserialization.
        let deserialized = TokenizerSectionData::deserialize_from(&bytes).unwrap();
        assert_eq!(deserialized.vocab.len(), 4);
        assert_eq!(deserialized.scores[3], -1.5);
    }
}
