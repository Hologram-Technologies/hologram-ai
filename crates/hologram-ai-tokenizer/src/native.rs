//! `NativeTokenizer` — concrete tokenizer implementation.

use crate::bpe::BpeEncoder;
use crate::config::{
    NormStep, NormalizationConfig, PreTokenizerConfig, TokenizerAlgorithm, TokenizerConfig,
};
use crate::unigram::UnigramEncoder;
use crate::vocab::VocabTable;
use crate::wordpiece::WordPieceEncoder;
use crate::Tokenizer;
use alloc::borrow::Cow;
use alloc::string::String;
use alloc::vec::Vec;

// ── Host-shell-only imports (JSON loading) ───────────────────────────────────
#[cfg(feature = "std")]
use crate::config::SpecialTokens;
#[cfg(feature = "std")]
use crate::vocab::MergeRules;
#[cfg(feature = "std")]
use alloc::string::ToString;
#[cfg(feature = "std")]
use anyhow::{bail, Context, Result};
#[cfg(feature = "std")]
use hashbrown::HashMap;
#[cfg(feature = "std")]
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
    /// Construct from an already-parsed [`TokenizerConfig`] — the no_std /
    /// on-device entry point. The encode/decode path needs no `std`: a
    /// pre-built vocab + merges (e.g. decoded from a `.holo` tokenizer
    /// section) is turned into the algorithm-specific encoder here. The
    /// host-shell `from_tokenizer_json` (behind the `std` feature) is just a
    /// JSON front-end onto the same config.
    pub fn from_config(mut config: TokenizerConfig) -> Self {
        let pre_tokenizer = config.pre_tokenizer.clone();
        let byte_fallback = config.byte_fallback;
        // Move the (vocab-bearing) algorithm out into the backend; the stored
        // config keeps only the lightweight metadata (special tokens, flags),
        // so the large vocab is never duplicated.
        let algorithm = core::mem::replace(
            &mut config.algorithm,
            TokenizerAlgorithm::Unigram {
                vocab: VocabTable::new(Vec::new()),
                scores: Vec::new(),
            },
        );
        let backend = match algorithm {
            TokenizerAlgorithm::Bpe { vocab, merges } => {
                EncoderBackend::Bpe(BpeEncoder::new(vocab, merges, byte_fallback, pre_tokenizer))
            }
            TokenizerAlgorithm::Unigram { vocab, scores } => {
                EncoderBackend::Unigram { vocab, scores }
            }
            TokenizerAlgorithm::WordPiece {
                vocab,
                continuing_subword_prefix,
                max_input_chars_per_word,
            } => EncoderBackend::WordPiece {
                vocab,
                continuing_prefix: continuing_subword_prefix,
                max_input_chars_per_word,
            },
        };
        Self { config, backend }
    }
}

#[cfg(feature = "std")]
impl NativeTokenizer {
    /// Construct from a tokenizer file. The conventional name is
    /// `tokenizer.json`, but some HF repos ship only the SentencePiece
    /// `tokenizer.model`; when the `.json` path is absent its `.model` sibling
    /// is read instead (`from_tokenizer_json_bytes` sniffs either format by
    /// content, never by name).
    pub fn from_tokenizer_json(path: &Path) -> Result<Self> {
        let data = match std::fs::read(path) {
            Ok(data) => data,
            Err(e)
                if e.kind() == std::io::ErrorKind::NotFound
                    && path.file_name().is_some_and(|n| n == "tokenizer.json") =>
            {
                let sibling = path.with_file_name("tokenizer.model");
                std::fs::read(&sibling).with_context(|| {
                    format!(
                        "reading tokenizer file: {} (and its SentencePiece sibling {})",
                        path.display(),
                        sibling.display()
                    )
                })?
            }
            Err(e) => {
                return Err(e)
                    .with_context(|| format!("reading tokenizer file: {}", path.display()))
            }
        };
        let mut tok = Self::from_tokenizer_json_bytes(&data)?;
        // The eos/bos IDENTITY lives in the sibling `tokenizer_config.json`, not
        // in `tokenizer.json` (which only lists tokens, never which one ends a
        // turn). Resolve it so a non-`</s>` model (ChatML `<|im_end|>`, Llama-3
        // `<|eot_id|>`, GPT-2/Qwen `<|endoftext|>`, …) stops on its OWN eos
        // rather than the Llama default id 2. Best-effort: a bare tokenizer dir
        // with no config leaves the parsed ids untouched.
        if let Some(dir) = path.parent() {
            tok.resolve_special_tokens_from_config(&dir.join("tokenizer_config.json"));
        }
        Ok(tok)
    }

    /// Override eos/bos ids from a `tokenizer_config.json`'s `eos_token` /
    /// `bos_token` (a bare string or an `{ "content": … }` AddedToken), mapping
    /// each to its id through the loaded vocab. A missing file, missing field,
    /// or unmapped token leaves the existing id in place — enrichment, never a
    /// hard failure (the caller may pass `--eos` to be explicit).
    fn resolve_special_tokens_from_config(&mut self, config_path: &Path) {
        let Ok(bytes) = std::fs::read(config_path) else {
            return;
        };
        let Ok(cfg) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
            return;
        };
        if let Some(token) = special_token_content(&cfg["eos_token"]) {
            if let Some(id) = self.vocab_table().str_to_id(token) {
                self.config.special_tokens.eos_id = id;
            }
        }
        if let Some(token) = special_token_content(&cfg["bos_token"]) {
            if let Some(id) = self.vocab_table().str_to_id(token) {
                self.config.special_tokens.bos_id = Some(id);
            }
        }
        // Whether a BOS is auto-prepended is the model's OWN declaration
        // (`add_bos_token`), not a guess from the mere presence of a
        // post_processor: a ChatML model (Qwen) ships a ByteLevel post_processor
        // yet sets `add_bos_token: false`, and prepending one would corrupt its
        // prompt. When the config states it, it overrides the coarse heuristic.
        if let Some(add_bos) = cfg["add_bos_token"].as_bool() {
            self.config.add_bos = add_bos;
        }
    }

    /// Construct from in-memory tokenizer bytes — a HuggingFace
    /// `tokenizer.json` or a SentencePiece `tokenizer.model` (ModelProto),
    /// told apart by content, never by file name. Also loads the tokenizer
    /// baked into a `.holo` archive extension (no file needed); the `_json`
    /// name is kept because the wasm ABI calls it.
    ///
    /// `.tiktoken` rank files are refused by name: they underdetermine the
    /// tokenizer (see the error text), so loading one would be silently wrong.
    pub fn from_tokenizer_json_bytes(bytes: &[u8]) -> Result<Self> {
        let body = strip_bom_and_ws(bytes);
        if body.first() == Some(&b'{') {
            let json: serde_json::Value =
                serde_json::from_slice(body).context("parsing tokenizer JSON")?;

            let model = &json["model"];
            let model_type = model["type"].as_str().context("missing model.type")?;

            return match model_type {
                "BPE" => Self::from_bpe_json(&json),
                "Unigram" => Self::from_unigram_json(&json),
                "WordPiece" => Self::from_wordpiece_json(&json),
                other => bail!("unsupported tokenizer model type: {other:?}"),
            };
        }
        if looks_like_tiktoken_ranks(body) {
            bail!(
                "refusing `.tiktoken` rank data: the file only carries `<base64 token> <rank>` \
                 lines — the pre-tokenization regex that defines token boundaries lives in the \
                 reference code, not in the file, so loading it here would tokenize silently \
                 wrong; use the model's tokenizer.json instead"
            );
        }
        match crate::sentencepiece::config_from_model_proto(bytes) {
            Ok(config) => Ok(Self::from_config(config)),
            Err(crate::sentencepiece::SpError::Refused(why)) => {
                bail!("refusing SentencePiece tokenizer.model: {why}")
            }
            Err(crate::sentencepiece::SpError::NotModelProto(why)) => bail!(
                "unrecognized tokenizer bytes (first byte {:?}): not tokenizer.json (no leading \
                 '{{'), not `.tiktoken` rank lines, and not a SentencePiece ModelProto ({why})",
                body.first().copied()
            ),
        }
    }

    fn from_bpe_json(json: &serde_json::Value) -> Result<Self> {
        let model = &json["model"];

        // Parse vocab: string → id map
        let vocab_obj = model["vocab"].as_object().context("missing model.vocab")?;
        let vocab_map: HashMap<String, u32> = vocab_obj
            .iter()
            .map(|(k, v)| {
                let id = v.as_u64().unwrap_or(0) as u32;
                (k.clone(), id)
            })
            .collect();
        let vocab = VocabTable::from_vocab_map(&vocab_map);

        // Parse merges — can be either ["a", "b"] arrays or "a b" strings
        let merges_arr = model["merges"].as_array().context("missing model.merges")?;
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

        Ok(Self {
            config,
            backend: EncoderBackend::Bpe(encoder),
        })
    }

    fn from_unigram_json(json: &serde_json::Value) -> Result<Self> {
        let model = &json["model"];
        let vocab_arr = model["vocab"]
            .as_array()
            .context("missing model.vocab for Unigram")?;

        let mut tokens = Vec::with_capacity(vocab_arr.len());
        let mut scores = Vec::with_capacity(vocab_arr.len());
        for entry in vocab_arr {
            let arr = entry
                .as_array()
                .context("vocab entry must be [token, score]")?;
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
        let max_chars = model["max_input_chars_per_word"].as_u64().unwrap_or(200) as usize;

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

    /// Apply the normalization steps the runtime implements (`Sequence` — the
    /// SentencePiece loader emits it). NFC/NFKC remain stored-but-unapplied,
    /// as before.
    fn normalize<'t>(&self, text: &'t str) -> Cow<'t, str> {
        let NormalizationConfig::Sequence(steps) = &self.config.normalization else {
            return Cow::Borrowed(text);
        };
        let mut out = Cow::Borrowed(text);
        for step in steps {
            match step {
                NormStep::RemoveExtraWhitespace => {
                    let trimmed = out.trim_matches(' ');
                    let mut collapsed = String::with_capacity(trimmed.len());
                    let mut prev_space = false;
                    for ch in trimmed.chars() {
                        if ch == ' ' && prev_space {
                            continue;
                        }
                        prev_space = ch == ' ';
                        collapsed.push(ch);
                    }
                    out = Cow::Owned(collapsed);
                }
                NormStep::PrependSpace => out = Cow::Owned(format!(" {out}")),
                _ => {}
            }
        }
        out
    }

    fn encode_raw(&self, text: &str) -> Vec<u32> {
        let normalized = self.normalize(text);
        let text = normalized.as_ref();
        match &self.backend {
            EncoderBackend::Bpe(enc) => enc.encode(text),
            EncoderBackend::Unigram { vocab, scores } => {
                let enc = UnigramEncoder::new(vocab, scores);
                // SentencePiece Viterbi segments the whole normalized
                // sentence, so the Metaspace ▁-escaping + dummy prefix are
                // applied to the text directly (no word splitting needed).
                if let PreTokenizerConfig::Metaspace {
                    replacement,
                    prepend,
                } = &self.config.pre_tokenizer
                {
                    let mut t = text.replace(' ', &String::from(*replacement));
                    if *prepend {
                        t.insert(0, *replacement);
                    }
                    enc.encode(&t)
                } else {
                    enc.encode(text)
                }
            }
            EncoderBackend::WordPiece {
                vocab,
                continuing_prefix,
                max_input_chars_per_word,
            } => {
                // Simple whitespace split, then encode each word.
                text.split_whitespace()
                    .flat_map(|word| {
                        let enc = WordPieceEncoder::new(
                            vocab,
                            continuing_prefix,
                            *max_input_chars_per_word,
                        );
                        enc.encode_word(word)
                    })
                    .collect()
            }
        }
    }

    fn decode_raw(&self, tokens: &[u32]) -> String {
        match &self.backend {
            EncoderBackend::Bpe(enc) => enc.decode(tokens),
            EncoderBackend::Unigram { vocab, .. } => {
                // SentencePiece decode: byte pieces (`<0xNN>`) reassemble to
                // raw bytes before UTF-8 recovery; ▁ renders as space; the
                // dummy prefix that encode prepended is stripped once.
                let mut bytes = Vec::new();
                for &id in tokens {
                    if self.config.byte_fallback {
                        if let Some(b) = vocab
                            .id_to_str(id)
                            .and_then(crate::bpe::parse_byte_fallback)
                        {
                            bytes.push(b);
                            continue;
                        }
                    }
                    if let Some(tok) = vocab.id_to_token.get(id as usize) {
                        bytes.extend_from_slice(tok);
                    }
                }
                let text = String::from_utf8_lossy(&bytes);
                match &self.config.pre_tokenizer {
                    PreTokenizerConfig::Metaspace {
                        replacement,
                        prepend,
                    } => {
                        let text = text.replace(*replacement, " ");
                        if *prepend {
                            String::from(text.strip_prefix(' ').unwrap_or(&text))
                        } else {
                            text
                        }
                    }
                    _ => {
                        // With ▁-escaping off, the dummy prefix is a plain
                        // leading space.
                        let prepended = matches!(
                            &self.config.normalization,
                            NormalizationConfig::Sequence(steps)
                                if steps.iter().any(|s| matches!(s, NormStep::PrependSpace))
                        );
                        if prepended {
                            String::from(text.strip_prefix(' ').unwrap_or(&text))
                        } else {
                            text.into_owned()
                        }
                    }
                }
            }
            EncoderBackend::WordPiece { vocab, .. } => tokens
                .iter()
                .filter_map(|&id| vocab.id_to_str(id))
                .collect::<Vec<_>>()
                .join(""),
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

// ── Format sniffing helpers (host shell) ────────────────────────────────

/// UTF-8 BOM + leading ASCII whitespace, stripped for format sniffing.
#[cfg(feature = "std")]
fn strip_bom_and_ws(bytes: &[u8]) -> &[u8] {
    let b = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(bytes);
    let start = b
        .iter()
        .position(|c| !c.is_ascii_whitespace())
        .unwrap_or(b.len());
    &b[start..]
}

/// A `.tiktoken` rank file: `<base64 token> <decimal rank>` per line.
/// Detected so the refusal can name the format instead of surfacing a
/// generic parse failure.
#[cfg(feature = "std")]
fn looks_like_tiktoken_ranks(body: &[u8]) -> bool {
    let line = body.split(|&b| b == b'\n').next().unwrap_or(&[]);
    let line = line.strip_suffix(b"\r").unwrap_or(line);
    let Some(space) = line.iter().position(|&b| b == b' ') else {
        return false;
    };
    let (token, rank) = (&line[..space], &line[space + 1..]);
    !token.is_empty()
        && !rank.is_empty()
        && token
            .iter()
            .all(|&b| b.is_ascii_alphanumeric() || b == b'+' || b == b'/' || b == b'=')
        && rank.iter().all(u8::is_ascii_digit)
}

// ── JSON parsing helpers (host shell) ───────────────────────────────────

/// The content of a `tokenizer_config.json` special-token field: either a bare
/// string (`"eos_token": "</s>"`) or an `AddedToken` object with a `content`
/// key (`"eos_token": { "content": "<|im_end|>", … }`).
#[cfg(feature = "std")]
fn special_token_content(v: &serde_json::Value) -> Option<&str> {
    v.as_str()
        .or_else(|| v.get("content").and_then(|c| c.as_str()))
}

#[cfg(feature = "std")]
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

#[cfg(feature = "std")]
fn parse_pre_tokenizer(json: &serde_json::Value) -> PreTokenizerConfig {
    let pt = match json.get("pre_tokenizer") {
        Some(v) if !v.is_null() => v,
        _ => return PreTokenizerConfig::None,
    };

    parse_pre_tokenizer_value(pt)
}

#[cfg(feature = "std")]
fn parse_pre_tokenizer_value(pt: &serde_json::Value) -> PreTokenizerConfig {
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
            if let Some(pattern) = pt["pattern"]
                .as_object()
                .and_then(|p| p.get("Regex"))
                .and_then(|r| r.as_str())
            {
                PreTokenizerConfig::Regex(pattern.to_string())
            } else {
                PreTokenizerConfig::None
            }
        }
        Some("ByteLevel") => PreTokenizerConfig::ByteLevel { regex: None },
        Some("Sequence") => {
            // A Sequence contains an ordered list of sub-tokenizers.
            // For byte-level BPE (Qwen, GPT-2), the pattern is:
            //   [Split(regex), ByteLevel]
            // We extract the regex from Split and combine with ByteLevel.
            let subs = match pt["pretokenizers"].as_array() {
                Some(arr) => arr,
                None => return PreTokenizerConfig::None,
            };

            let mut regex: Option<String> = None;
            let mut has_byte_level = false;

            for sub in subs {
                match sub["type"].as_str() {
                    Some("Split") => {
                        regex = sub["pattern"]
                            .as_object()
                            .and_then(|p| p.get("Regex"))
                            .and_then(|r| r.as_str())
                            .map(|s| s.to_string());
                    }
                    Some("ByteLevel") => {
                        has_byte_level = true;
                    }
                    Some("Metaspace") => {
                        // If Sequence contains Metaspace (not ByteLevel),
                        // use the Metaspace config.
                        return parse_pre_tokenizer_value(sub);
                    }
                    _ => {}
                }
            }

            if has_byte_level {
                PreTokenizerConfig::ByteLevel { regex }
            } else if let Some(pattern) = regex {
                PreTokenizerConfig::Regex(pattern)
            } else {
                PreTokenizerConfig::None
            }
        }
        _ => PreTokenizerConfig::None,
    }
}

#[cfg(feature = "std")]
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

/// Core (no_std-buildable) tests: exercise `from_config` + encode/decode
/// without any JSON loading, so they run on the no_std build too.
#[cfg(test)]
mod core_tests {
    use super::*;
    use crate::config::{NormalizationConfig, PreTokenizerConfig, SpecialTokens};
    use alloc::vec;

    fn bpe_config() -> TokenizerConfig {
        // Vocab covers single bytes h,e,l,o plus the merge "ll".
        let vocab = VocabTable::new(vec![
            b"h".to_vec(),
            b"e".to_vec(),
            b"l".to_vec(),
            b"o".to_vec(),
            b"ll".to_vec(),
        ]);
        let merges = MergeRules {
            merges: vec![(b"l".to_vec(), b"l".to_vec())],
        };
        TokenizerConfig {
            algorithm: TokenizerAlgorithm::Bpe { vocab, merges },
            special_tokens: SpecialTokens {
                bos_id: None,
                // Out of the content-token range (vocab ids 0..5) so decode's
                // eos filtering never drops a real token.
                eos_id: 100,
                pad_id: None,
                unk_id: None,
                additional: Default::default(),
            },
            normalization: NormalizationConfig::None,
            pre_tokenizer: PreTokenizerConfig::None,
            byte_fallback: false,
            add_bos: false,
            add_eos: false,
        }
    }

    #[test]
    fn from_config_bpe_encodes_with_merges() {
        let tok = NativeTokenizer::from_config(bpe_config());
        // "hello" → h, e, ll (merged), o  → ids [0, 1, 4, 3]
        assert_eq!(tok.encode("hello"), vec![0, 1, 4, 3]);
        assert_eq!(tok.vocab_size(), 5);
    }

    #[test]
    fn from_config_decode_roundtrips() {
        let tok = NativeTokenizer::from_config(bpe_config());
        let ids = tok.encode("hello");
        assert_eq!(tok.decode(&ids), "hello");
    }
}

#[cfg(all(test, feature = "std"))]
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

    /// A non-`</s>` chat model (ChatML `<|im_end|>`) loaded from a path resolves
    /// its OWN eos from the sibling `tokenizer_config.json` — never the Llama
    /// default id 2, which would be a wrong (or never-firing) stop token.
    #[test]
    fn eos_resolves_from_sibling_tokenizer_config() {
        let dir = tempfile::tempdir().unwrap();
        // A minimal BPE tokenizer.json: vocab has `<|im_end|>` at id 5, and its
        // `added_tokens` deliberately contain NO `</s>` — so `parse_special_tokens`
        // alone would fall back to the default eos id 2.
        // A `post_processor` is present, so the coarse heuristic sets
        // `add_bos = true` — the sibling config must be able to override it.
        let tokenizer_json = r#"{
            "model": {
                "type": "BPE",
                "vocab": {"h": 0, "e": 1, "l": 2, "o": 3, "ll": 4, "<|im_end|>": 5},
                "merges": ["l l"],
                "byte_fallback": false
            },
            "post_processor": {"type": "ByteLevel"},
            "added_tokens": [
                {"id": 5, "content": "<|im_end|>", "special": true}
            ]
        }"#;
        std::fs::write(dir.path().join("tokenizer.json"), tokenizer_json).unwrap();

        // Without the config: eos falls back to the default 2 (the bug), and the
        // presence of a post_processor coarsely implies add_bos.
        let bare =
            NativeTokenizer::from_tokenizer_json(&dir.path().join("tokenizer.json")).unwrap();
        assert_eq!(bare.eos_token_id(), 2, "no config ⇒ documented default");
        assert!(bare.config.add_bos, "post_processor ⇒ coarse add_bos");

        // With the sibling config naming `<|im_end|>`: eos resolves to id 5.
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"eos_token": "<|im_end|>", "add_bos_token": false}"#,
        )
        .unwrap();
        let tok = NativeTokenizer::from_tokenizer_json(&dir.path().join("tokenizer.json")).unwrap();
        assert_eq!(
            tok.eos_token_id(),
            5,
            "eos must come from the model's own config, not the Llama default"
        );
        assert!(
            !tok.config.add_bos,
            "the model's own add_bos_token: false must override the post_processor heuristic"
        );

        // The AddedToken object form (`{content}`) resolves identically.
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"eos_token": {"content": "<|im_end|>", "special": true}}"#,
        )
        .unwrap();
        let tok = NativeTokenizer::from_tokenizer_json(&dir.path().join("tokenizer.json")).unwrap();
        assert_eq!(tok.eos_token_id(), 5);
    }

    #[test]
    fn load_tinyllama_tokenizer() {
        let path = tokenizer_json_path();
        if !path.exists() {
            std::eprintln!("skipping: tokenizer.json not found at {}", path.display());
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
