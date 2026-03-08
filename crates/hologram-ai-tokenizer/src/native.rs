//! `NativeTokenizer` — concrete tokenizer implementation.

use crate::bpe::BpeEncoder;
use crate::config::{
    NormalizationConfig, PreTokenizerConfig, SpecialTokens, TokenizerAlgorithm, TokenizerConfig,
};
use crate::unigram::UnigramEncoder;
use crate::vocab::{MergeRules, VocabTable};
use crate::wordpiece::WordPieceEncoder;
use crate::Tokenizer;
use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::path::Path;

/// Encoder backend — dispatches to the algorithm-specific encoder.
enum EncoderBackend {
    Bpe(BpeEncoder),
    Unigram {
        vocab: VocabTable,
        scores: Vec<f32>,
    },
    WordPiece {
        vocab: VocabTable,
        continuing_prefix: String,
        max_input_chars_per_word: usize,
    },
}

/// Native tokenizer backed by hologram data structures.
///
/// Supports BPE, Unigram (SentencePiece), and WordPiece algorithms.
pub struct NativeTokenizer {
    config: TokenizerConfig,
    backend: EncoderBackend,
}

impl NativeTokenizer {
    /// Construct from a HuggingFace `tokenizer.json` file.
    pub fn from_tokenizer_json(path: &Path) -> Result<Self> {
        let data = std::fs::read_to_string(path)
            .with_context(|| format!("reading tokenizer file: {}", path.display()))?;
        let json: serde_json::Value =
            serde_json::from_str(&data).context("parsing tokenizer JSON")?;

        let model = &json["model"];
        let model_type = model["type"]
            .as_str()
            .context("missing model.type")?;

        match model_type {
            "BPE" => Self::from_bpe_json(&json),
            "Unigram" => Self::from_unigram_json(&json),
            "WordPiece" => Self::from_wordpiece_json(&json),
            other => bail!("unsupported tokenizer model type: {other:?}"),
        }
    }

    fn from_bpe_json(json: &serde_json::Value) -> Result<Self> {
        let model = &json["model"];

        // Parse vocab: string → id map
        let vocab_obj = model["vocab"]
            .as_object()
            .context("missing model.vocab")?;
        let vocab_map: HashMap<String, u32> = vocab_obj
            .iter()
            .map(|(k, v)| {
                let id = v.as_u64().unwrap_or(0) as u32;
                (k.clone(), id)
            })
            .collect();
        let vocab = VocabTable::from_vocab_map(&vocab_map);

        // Parse merges — can be either ["a", "b"] arrays or "a b" strings
        let merges_arr = model["merges"]
            .as_array()
            .context("missing model.merges")?;
        let merges = MergeRules::from_json_merges(merges_arr);

        // byte_fallback
        let byte_fallback = model["byte_fallback"].as_bool().unwrap_or(false);

        // Parse special tokens from added_tokens
        let special = parse_special_tokens(json)?;

        // Parse pre-tokenizer
        let pre_tokenizer = parse_pre_tokenizer(json);

        // Parse normalization
        let normalization = parse_normalization(json);

        // Check post_processor for add_bos behavior
        let add_bos = json.get("post_processor").is_some();
        let add_eos = false;

        let config = TokenizerConfig {
            algorithm: TokenizerAlgorithm::Bpe {
                vocab: VocabTable::new(vec![]), // placeholder — actual vocab is in encoder
                merges: MergeRules { merges: vec![] }, // placeholder
            },
            special_tokens: special,
            normalization,
            pre_tokenizer: pre_tokenizer.clone(),
            byte_fallback,
            add_bos,
            add_eos,
        };

        let encoder = BpeEncoder::new(vocab, merges, byte_fallback, pre_tokenizer);

        Ok(Self { config, backend: EncoderBackend::Bpe(encoder) })
    }

    fn from_unigram_json(json: &serde_json::Value) -> Result<Self> {
        let model = &json["model"];
        let vocab_arr = model["vocab"]
            .as_array()
            .context("missing model.vocab for Unigram")?;

        let mut tokens = Vec::with_capacity(vocab_arr.len());
        let mut scores = Vec::with_capacity(vocab_arr.len());
        for entry in vocab_arr {
            let arr = entry.as_array().context("vocab entry must be [token, score]")?;
            let token = arr[0].as_str().context("vocab token must be string")?;
            let score = arr[1].as_f64().unwrap_or(0.0) as f32;
            tokens.push(token.as_bytes().to_vec());
            scores.push(score);
        }
        let vocab = VocabTable::new(tokens);

        let byte_fallback = model["byte_fallback"].as_bool().unwrap_or(false);
        let special = parse_special_tokens(json)?;
        let pre_tokenizer = parse_pre_tokenizer(json);
        let normalization = parse_normalization(json);
        let add_bos = json.get("post_processor").is_some();

        let config = TokenizerConfig {
            algorithm: TokenizerAlgorithm::Unigram {
                vocab: VocabTable::new(vec![]),
                scores: vec![],
            },
            special_tokens: special,
            normalization,
            pre_tokenizer,
            byte_fallback,
            add_bos,
            add_eos: false,
        };

        Ok(Self {
            config,
            backend: EncoderBackend::Unigram { vocab, scores },
        })
    }

    fn from_wordpiece_json(json: &serde_json::Value) -> Result<Self> {
        let model = &json["model"];
        let vocab_obj = model["vocab"]
            .as_object()
            .context("missing model.vocab for WordPiece")?;
        let vocab_map: HashMap<String, u32> = vocab_obj
            .iter()
            .map(|(k, v)| (k.clone(), v.as_u64().unwrap_or(0) as u32))
            .collect();
        let vocab = VocabTable::from_vocab_map(&vocab_map);

        let continuing_prefix = model["continuing_subword_prefix"]
            .as_str()
            .unwrap_or("##")
            .to_string();
        let max_chars = model["max_input_chars_per_word"]
            .as_u64()
            .unwrap_or(200) as usize;

        let special = parse_special_tokens(json)?;
        let pre_tokenizer = parse_pre_tokenizer(json);
        let normalization = parse_normalization(json);

        let config = TokenizerConfig {
            algorithm: TokenizerAlgorithm::WordPiece {
                vocab: VocabTable::new(vec![]),
                continuing_subword_prefix: String::new(),
                max_input_chars_per_word: 0,
            },
            special_tokens: special,
            normalization,
            pre_tokenizer,
            byte_fallback: false,
            add_bos: false,
            add_eos: false,
        };

        Ok(Self {
            config,
            backend: EncoderBackend::WordPiece {
                vocab,
                continuing_prefix,
                max_input_chars_per_word: max_chars,
            },
        })
    }
}

impl NativeTokenizer {
    fn vocab_table(&self) -> &VocabTable {
        match &self.backend {
            EncoderBackend::Bpe(enc) => enc.vocab(),
            EncoderBackend::Unigram { vocab, .. } => vocab,
            EncoderBackend::WordPiece { vocab, .. } => vocab,
        }
    }

    fn encode_raw(&self, text: &str) -> Vec<u32> {
        match &self.backend {
            EncoderBackend::Bpe(enc) => enc.encode(text),
            EncoderBackend::Unigram { vocab, scores } => {
                let enc = UnigramEncoder::new(vocab, scores);
                enc.encode(text)
            }
            EncoderBackend::WordPiece {
                vocab,
                continuing_prefix,
                max_input_chars_per_word,
            } => {
                // Simple whitespace split, then encode each word.
                text.split_whitespace()
                    .flat_map(|word| {
                        let enc = WordPieceEncoder::new(vocab, continuing_prefix, *max_input_chars_per_word);
                        enc.encode_word(word)
                    })
                    .collect()
            }
        }
    }

    fn decode_raw(&self, tokens: &[u32]) -> String {
        match &self.backend {
            EncoderBackend::Bpe(enc) => enc.decode(tokens),
            EncoderBackend::Unigram { vocab, .. } | EncoderBackend::WordPiece { vocab, .. } => {
                tokens
                    .iter()
                    .filter_map(|&id| vocab.id_to_str(id))
                    .collect::<Vec<_>>()
                    .join("")
            }
        }
    }
}

impl Tokenizer for NativeTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();

        if self.config.add_bos {
            if let Some(bos) = self.config.special_tokens.bos_id {
                ids.push(bos);
            }
        }

        ids.extend(self.encode_raw(text));

        if self.config.add_eos {
            ids.push(self.config.special_tokens.eos_id);
        }

        ids
    }

    fn decode(&self, tokens: &[u32]) -> String {
        let bos = self.config.special_tokens.bos_id;
        let eos = self.config.special_tokens.eos_id;
        let filtered: Vec<u32> = tokens
            .iter()
            .copied()
            .filter(|&id| Some(id) != bos && id != eos)
            .collect();
        self.decode_raw(&filtered)
    }

    fn eos_token_id(&self) -> u32 {
        self.config.special_tokens.eos_id
    }

    fn bos_token_id(&self) -> Option<u32> {
        self.config.special_tokens.bos_id
    }

    fn vocab_size(&self) -> usize {
        self.vocab_table().len()
    }

    fn id_to_token(&self, id: u32) -> Option<&str> {
        self.vocab_table().id_to_str(id)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab_table().str_to_id(token)
    }
}

// ── JSON parsing helpers ────────────────────────────────────────────────

fn parse_special_tokens(json: &serde_json::Value) -> Result<SpecialTokens> {
    let added = json.get("added_tokens").and_then(|v| v.as_array());

    let mut bos_id = None;
    let mut eos_id = None;
    let mut unk_id = None;
    let mut pad_id = None;
    let mut additional = HashMap::new();

    if let Some(tokens) = added {
        for t in tokens {
            let id = t["id"].as_u64().unwrap_or(0) as u32;
            let content = t["content"].as_str().unwrap_or("");
            let special = t["special"].as_bool().unwrap_or(false);

            match content {
                "<s>" => bos_id = Some(id),
                "</s>" => eos_id = Some(id),
                "<unk>" => unk_id = Some(id),
                "<pad>" => pad_id = Some(id),
                _ if special => {
                    additional.insert(content.to_string(), id);
                }
                _ => {}
            }
        }
    }

    Ok(SpecialTokens {
        bos_id,
        eos_id: eos_id.unwrap_or(2), // default EOS
        pad_id,
        unk_id,
        additional,
    })
}

fn parse_pre_tokenizer(json: &serde_json::Value) -> PreTokenizerConfig {
    let pt = match json.get("pre_tokenizer") {
        Some(v) if !v.is_null() => v,
        _ => return PreTokenizerConfig::None,
    };

    match pt["type"].as_str() {
        Some("Metaspace") => {
            let replacement = pt["replacement"]
                .as_str()
                .and_then(|s| s.chars().next())
                .unwrap_or('\u{2581}');
            let prepend = match pt["prepend_scheme"].as_str() {
                Some("first") | Some("always") => true,
                _ => pt["add_prefix_space"].as_bool().unwrap_or(true),
            };
            PreTokenizerConfig::Metaspace {
                replacement,
                prepend,
            }
        }
        Some("Split") => {
            if let Some(pattern) = pt["pattern"].as_object().and_then(|p| p.get("Regex")).and_then(|r| r.as_str()) {
                PreTokenizerConfig::Regex(pattern.to_string())
            } else {
                PreTokenizerConfig::None
            }
        }
        _ => PreTokenizerConfig::None,
    }
}

fn parse_normalization(json: &serde_json::Value) -> NormalizationConfig {
    let norm = match json.get("normalizer") {
        Some(v) if !v.is_null() => v,
        _ => return NormalizationConfig::None,
    };

    match norm["type"].as_str() {
        Some("NFC") => NormalizationConfig::Nfc,
        Some("NFKC") => NormalizationConfig::Nfkc,
        Some("Prepend") => NormalizationConfig::PrependSpace,
        _ => NormalizationConfig::None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn tokenizer_json_path() -> PathBuf {
        // Walk up from crate root to workspace root
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.pop(); // crates/
        p.pop(); // workspace root
        p.push("models/TinyLlama-1.1B-Chat-v1.0/tokenizer.json");
        p
    }

    #[test]
    fn load_tinyllama_tokenizer() {
        let path = tokenizer_json_path();
        if !path.exists() {
            eprintln!("skipping: tokenizer.json not found at {}", path.display());
            return;
        }
        let tok = NativeTokenizer::from_tokenizer_json(&path).unwrap();
        assert_eq!(tok.vocab_size(), 32000);
        assert_eq!(tok.eos_token_id(), 2);
        assert_eq!(tok.bos_token_id(), Some(1));
    }

    #[test]
    fn encode_hello() {
        let path = tokenizer_json_path();
        if !path.exists() {
            return;
        }
        let tok = NativeTokenizer::from_tokenizer_json(&path).unwrap();
        // With BOS prepended, "Hello" → [1, 15043]
        let ids = tok.encode("Hello");
        assert_eq!(ids[0], 1, "should start with BOS");
        assert_eq!(ids[1], 15043, "▁Hello should be token 15043");
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn encode_sentence() {
        let path = tokenizer_json_path();
        if !path.exists() {
            return;
        }
        let tok = NativeTokenizer::from_tokenizer_json(&path).unwrap();
        // "tell me a joke" → BOS + [2649, 592, 263, 2958, 446]
        let ids = tok.encode("tell me a joke");
        assert_eq!(ids[0], 1, "BOS");
        assert_eq!(&ids[1..], &[2649, 592, 263, 2958, 446]);
    }

    #[test]
    fn decode_roundtrip() {
        let path = tokenizer_json_path();
        if !path.exists() {
            return;
        }
        let tok = NativeTokenizer::from_tokenizer_json(&path).unwrap();
        let texts = ["Hello", "tell me a joke", "Hello, world!"];
        for text in texts {
            let ids = tok.encode(text);
            let decoded = tok.decode(&ids);
            assert_eq!(decoded, text, "round-trip failed for {text:?}");
        }
    }

    #[test]
    fn token_lookups() {
        let path = tokenizer_json_path();
        if !path.exists() {
            return;
        }
        let tok = NativeTokenizer::from_tokenizer_json(&path).unwrap();
        assert_eq!(tok.id_to_token(1), Some("<s>"));
        assert_eq!(tok.id_to_token(2), Some("</s>"));
        assert_eq!(tok.token_to_id("<s>"), Some(1));
        assert_eq!(tok.token_to_id("</s>"), Some(2));
    }
}
