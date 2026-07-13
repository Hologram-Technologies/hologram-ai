//! SentencePiece `tokenizer.model` (ModelProto) loader.
//!
//! The protobuf wire decoding is hand-rolled: the runtime core is `no_std`
//! and the needed subset of the (frozen) `sentencepiece_model.proto` schema
//! is tiny, so a protobuf dependency would buy nothing. Unknown fields are
//! skipped by wire type; truncated or malformed input is a hard error naming
//! the byte offset — never a best-effort parse.

// The only caller is the std host shell (`from_tokenizer_json_bytes`); the
// parser itself is core + alloc, so it still compiles — unused — in no_std
// builds.
#![cfg_attr(not(feature = "std"), allow(dead_code))]

use crate::config::{
    NormStep, NormalizationConfig, PreTokenizerConfig, SpecialTokens, TokenizerAlgorithm,
    TokenizerConfig,
};
use crate::vocab::{MergeRules, VocabTable};
use alloc::string::String;
use alloc::vec::Vec;
use hashbrown::{HashMap, HashSet};

// `SentencePiece.Type` values (sentencepiece_model.proto).
const PIECE_NORMAL: u64 = 1;
const PIECE_CONTROL: u64 = 3;

// `TrainerSpec.ModelType` values.
const MODEL_UNIGRAM: u64 = 1;
const MODEL_BPE: u64 = 2;
const MODEL_WORD: u64 = 3;
const MODEL_CHAR: u64 = 4;

/// Loader failure, split so the format sniffer can tell "these bytes are not
/// a ModelProto at all" (fall through to the unrecognized-format error) from
/// "this IS a SentencePiece model, refused by name".
pub(crate) enum SpError {
    /// The bytes don't decode as a ModelProto.
    NotModelProto(String),
    /// A recognized SentencePiece model that cannot be loaded faithfully;
    /// the message names exactly what was refused and why.
    Refused(String),
}

/// Build a [`TokenizerConfig`] from raw `tokenizer.model` bytes.
///
/// Refused loudly (naming the field): a non-empty `precompiled_charsmap`
/// (applying it needs the Darts trie — skipping it would normalize, and so
/// tokenize, silently wrong) and the WORD / CHAR model types (no runtime
/// encoder maps them).
pub(crate) fn config_from_model_proto(bytes: &[u8]) -> Result<TokenizerConfig, SpError> {
    let (pieces, trainer, norm) = parse_model_proto(bytes).map_err(SpError::NotModelProto)?;
    if pieces.is_empty() {
        return Err(SpError::NotModelProto(
            "ModelProto carries no pieces (field 1)".into(),
        ));
    }
    if norm.charsmap_len != 0 {
        return Err(SpError::Refused(format!(
            "normalizer_spec.precompiled_charsmap is non-empty ({} bytes, normalizer {:?}): \
             applying it requires the Darts trie, and skipping it would normalize (and so \
             tokenize) silently wrong",
            norm.charsmap_len, norm.name
        )));
    }

    let n = pieces.len();
    let resolve = |what: &str, id: i32| -> Result<Option<u32>, SpError> {
        match usize::try_from(id) {
            // Negative ids mean "disabled" in trainer_spec.
            Err(_) => Ok(None),
            Ok(id) if id < n => Ok(Some(id as u32)),
            Ok(id) => Err(SpError::Refused(format!(
                "trainer_spec.{what} = {id} is out of range ({n} pieces)"
            ))),
        }
    };
    let unk_id = resolve("unk_id", trainer.unk_id)?;
    let bos_id = resolve("bos_id", trainer.bos_id)?;
    // `eos_id: -1` declares "no eos". u32::MAX lies outside any vocab, so the
    // eos filter in decode and generation's stop check can never eat a real
    // piece (mapping to the JSON-path default 2 would).
    let eos_id = resolve("eos_id", trainer.eos_id)?.unwrap_or(u32::MAX);

    let vocab_bytes: Vec<Vec<u8>> = pieces.iter().map(|p| p.text.as_bytes().to_vec()).collect();
    let algorithm = match trainer.model_type {
        MODEL_UNIGRAM => TokenizerAlgorithm::Unigram {
            vocab: VocabTable::new(vocab_bytes),
            scores: pieces.iter().map(|p| p.score).collect(),
        },
        MODEL_BPE => TokenizerAlgorithm::Bpe {
            merges: derive_bpe_merges(&pieces),
            vocab: VocabTable::new(vocab_bytes),
        },
        MODEL_WORD => {
            return Err(SpError::Refused(
                "unsupported SentencePiece model_type WORD (3): only UNIGRAM and BPE map onto \
                 the runtime encoders"
                    .into(),
            ))
        }
        MODEL_CHAR => {
            return Err(SpError::Refused(
                "unsupported SentencePiece model_type CHAR (4): only UNIGRAM and BPE map onto \
                 the runtime encoders"
                    .into(),
            ))
        }
        other => {
            return Err(SpError::Refused(format!(
                "unknown SentencePiece model_type {other}: only UNIGRAM (1) and BPE (2) are \
                 supported"
            )))
        }
    };

    // CONTROL pieces are special by construction (SentencePiece never produces
    // them from text); expose the non-bos/eos ones the way the JSON path
    // exposes special `added_tokens`.
    let mut additional = HashMap::new();
    for (id, p) in pieces.iter().enumerate() {
        let id = id as u32;
        if p.ty == PIECE_CONTROL && bos_id != Some(id) && eos_id != id {
            additional.insert(p.text.clone(), id);
        }
    }

    // SP normalizer pipeline (identity charsmap): whitespace collapse first;
    // the dummy prefix and ▁-escaping ride on the Metaspace pre-tokenizer.
    // With escaping off, the dummy prefix is a plain-space normalization step.
    let mut steps = Vec::new();
    if norm.remove_extra_whitespaces {
        steps.push(NormStep::RemoveExtraWhitespace);
    }
    let pre_tokenizer = if norm.escape_whitespaces {
        PreTokenizerConfig::Metaspace {
            replacement: '\u{2581}',
            prepend: norm.add_dummy_prefix,
        }
    } else {
        if norm.add_dummy_prefix {
            steps.push(NormStep::PrependSpace);
        }
        PreTokenizerConfig::None
    };
    let normalization = if steps.is_empty() {
        NormalizationConfig::None
    } else {
        NormalizationConfig::Sequence(steps)
    };

    Ok(TokenizerConfig {
        algorithm,
        special_tokens: SpecialTokens {
            bos_id,
            eos_id,
            pad_id: None,
            unk_id,
            additional,
        },
        normalization,
        pre_tokenizer,
        byte_fallback: trainer.byte_fallback,
        add_bos: bos_id.is_some(),
        add_eos: false,
    })
}

/// SP-BPE merges, at every step, the adjacent pair whose concatenation is the
/// vocab piece with the highest score; the rank engine merges the pair with
/// the lowest rank. Ranking every (left, right) split of every NORMAL piece
/// by descending score therefore reproduces the same choice at each step
/// (ties broken by piece id then split offset — deterministic).
fn derive_bpe_merges(pieces: &[Piece]) -> MergeRules {
    let normal: HashSet<&str> = pieces
        .iter()
        .filter(|p| p.ty == PIECE_NORMAL)
        .map(|p| p.text.as_str())
        .collect();
    // (score, piece id, split offset, left, right)
    let mut candidates: Vec<(f32, usize, usize, &str, &str)> = Vec::new();
    for (id, p) in pieces.iter().enumerate() {
        if p.ty != PIECE_NORMAL {
            continue;
        }
        for (split, _) in p.text.char_indices().skip(1) {
            let (l, r) = p.text.split_at(split);
            if normal.contains(l) && normal.contains(r) {
                candidates.push((p.score, id, split, l, r));
            }
        }
    }
    candidates.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
            .then(a.2.cmp(&b.2))
    });
    MergeRules {
        merges: candidates
            .into_iter()
            .map(|(_, _, _, l, r)| (l.as_bytes().to_vec(), r.as_bytes().to_vec()))
            .collect(),
    }
}

// ── protobuf wire parsing ────────────────────────────────────────────────────

struct Piece {
    text: String,
    score: f32,
    ty: u64,
}

struct Trainer {
    model_type: u64,
    byte_fallback: bool,
    unk_id: i32,
    bos_id: i32,
    eos_id: i32,
}

impl Default for Trainer {
    // proto2 field defaults: model_type UNIGRAM, unk/bos/eos = 0/1/2.
    fn default() -> Self {
        Self {
            model_type: MODEL_UNIGRAM,
            byte_fallback: false,
            unk_id: 0,
            bos_id: 1,
            eos_id: 2,
        }
    }
}

struct Normalizer {
    name: String,
    charsmap_len: usize,
    add_dummy_prefix: bool,
    remove_extra_whitespaces: bool,
    escape_whitespaces: bool,
}

impl Default for Normalizer {
    // proto2 field defaults: all three flags true.
    fn default() -> Self {
        Self {
            name: String::new(),
            charsmap_len: 0,
            add_dummy_prefix: true,
            remove_extra_whitespaces: true,
            escape_whitespaces: true,
        }
    }
}

fn parse_model_proto(bytes: &[u8]) -> Result<(Vec<Piece>, Trainer, Normalizer), String> {
    let mut r = Reader::new(bytes);
    let mut pieces = Vec::new();
    let mut trainer = Trainer::default();
    let mut normalizer = Normalizer::default();
    while !r.done() {
        let (field, wire) = r.key()?;
        match (field, wire) {
            // pieces = 1, trainer_spec = 2, normalizer_spec = 3
            (1, 2) => {
                let idx = pieces.len();
                pieces.push(parse_piece(r.len_delimited()?, idx)?);
            }
            (2, 2) => trainer = parse_trainer(r.len_delimited()?)?,
            (3, 2) => normalizer = parse_normalizer(r.len_delimited()?)?,
            (f, w) => r.skip(f, w)?,
        }
    }
    Ok((pieces, trainer, normalizer))
}

fn parse_piece(bytes: &[u8], index: usize) -> Result<Piece, String> {
    let mut r = Reader::new(bytes);
    let mut text = None;
    let mut score = 0.0f32;
    let mut ty = PIECE_NORMAL;
    while !r.done() {
        let (field, wire) = r.key()?;
        match (field, wire) {
            // piece = 1, score = 2, type = 3
            (1, 2) => {
                text = Some(
                    String::from_utf8(r.len_delimited()?.to_vec())
                        .map_err(|_| format!("piece {index} is not valid UTF-8"))?,
                );
            }
            (2, 5) => score = f32::from_le_bytes(r.fixed32()?),
            (3, 0) => ty = r.varint()?,
            (f, w) => r.skip(f, w)?,
        }
    }
    let text = text.ok_or_else(|| format!("piece {index} has no piece string (field 1)"))?;
    Ok(Piece { text, score, ty })
}

fn parse_trainer(bytes: &[u8]) -> Result<Trainer, String> {
    let mut r = Reader::new(bytes);
    let mut t = Trainer::default();
    while !r.done() {
        let (field, wire) = r.key()?;
        match (field, wire) {
            // model_type = 3, byte_fallback = 35, unk/bos/eos_id = 40/41/42
            (3, 0) => t.model_type = r.varint()?,
            (35, 0) => t.byte_fallback = r.varint()? != 0,
            (40, 0) => t.unk_id = r.varint()? as i32,
            (41, 0) => t.bos_id = r.varint()? as i32,
            (42, 0) => t.eos_id = r.varint()? as i32,
            (f, w) => r.skip(f, w)?,
        }
    }
    Ok(t)
}

fn parse_normalizer(bytes: &[u8]) -> Result<Normalizer, String> {
    let mut r = Reader::new(bytes);
    let mut n = Normalizer::default();
    while !r.done() {
        let (field, wire) = r.key()?;
        match (field, wire) {
            // name = 1, precompiled_charsmap = 2, add_dummy_prefix = 3,
            // remove_extra_whitespaces = 4, escape_whitespaces = 5
            (1, 2) => {
                n.name = String::from_utf8(r.len_delimited()?.to_vec())
                    .map_err(|_| String::from("normalizer_spec.name is not valid UTF-8"))?;
            }
            (2, 2) => n.charsmap_len = r.len_delimited()?.len(),
            (3, 0) => n.add_dummy_prefix = r.varint()? != 0,
            (4, 0) => n.remove_extra_whitespaces = r.varint()? != 0,
            (5, 0) => n.escape_whitespaces = r.varint()? != 0,
            (f, w) => r.skip(f, w)?,
        }
    }
    Ok(n)
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn done(&self) -> bool {
        self.pos >= self.buf.len()
    }

    fn varint(&mut self) -> Result<u64, String> {
        let mut v: u64 = 0;
        for shift in 0..10u32 {
            let Some(&b) = self.buf.get(self.pos) else {
                return Err(format!("truncated varint at byte {}", self.pos));
            };
            self.pos += 1;
            // The 10th byte of a u64 varint carries at most bit 63.
            if shift == 9 && b > 1 {
                return Err(format!("varint overflows u64 at byte {}", self.pos - 1));
            }
            v |= u64::from(b & 0x7F) << (shift * 7);
            if b & 0x80 == 0 {
                return Ok(v);
            }
        }
        Err(format!("unterminated varint at byte {}", self.pos))
    }

    fn key(&mut self) -> Result<(u32, u8), String> {
        let at = self.pos;
        let k = self.varint()?;
        // Field number 0 is invalid protobuf — reject rather than "skip".
        if k >> 3 == 0 {
            return Err(format!("invalid field number 0 at byte {at}"));
        }
        Ok(((k >> 3) as u32, (k & 7) as u8))
    }

    fn len_delimited(&mut self) -> Result<&'a [u8], String> {
        let at = self.pos;
        let len = self.varint()? as usize;
        let end = self
            .pos
            .checked_add(len)
            .filter(|&e| e <= self.buf.len())
            .ok_or_else(|| {
                format!("length-delimited field of {len} bytes at byte {at} overruns the input")
            })?;
        let out = &self.buf[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn fixed32(&mut self) -> Result<[u8; 4], String> {
        let end = self.pos + 4;
        let slice = self
            .buf
            .get(self.pos..end)
            .ok_or_else(|| format!("truncated fixed32 at byte {}", self.pos))?;
        self.pos = end;
        Ok(slice.try_into().expect("4-byte slice"))
    }

    fn skip(&mut self, field: u32, wire: u8) -> Result<(), String> {
        match wire {
            0 => {
                self.varint()?;
            }
            1 => {
                if self.buf.len() - self.pos < 8 {
                    return Err(format!("truncated fixed64 at byte {}", self.pos));
                }
                self.pos += 8;
            }
            2 => {
                self.len_delimited()?;
            }
            5 => {
                self.fixed32()?;
            }
            w => {
                return Err(format!(
                    "field {field} has unsupported wire type {w} at byte {}",
                    self.pos
                ))
            }
        }
        Ok(())
    }
}
