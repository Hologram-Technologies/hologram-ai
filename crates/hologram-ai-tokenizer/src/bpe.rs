//! BPE (Byte-Pair Encoding) tokenizer implementation.

use crate::config::PreTokenizerConfig;
use crate::vocab::{MergeRules, VocabTable};
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use hashbrown::HashMap;

/// BPE encoder/decoder.
pub struct BpeEncoder {
    vocab: VocabTable,
    /// Maps (left_bytes, right_bytes) → merge rank (lower = higher priority).
    merge_ranks: HashMap<(Vec<u8>, Vec<u8>), u32>,
    byte_fallback: bool,
    pre_tokenizer: PreTokenizerConfig,
    /// Byte-level / regex pre-tokenizer pattern, compiled once at
    /// construction (recompiling per `encode` would be wasteful). `None` when
    /// the pre-tokenizer takes no pattern or the pattern failed to compile.
    split_re: Option<regex_automata::meta::Regex>,
}

impl BpeEncoder {
    pub fn new(
        vocab: VocabTable,
        merges: MergeRules,
        byte_fallback: bool,
        pre_tokenizer: PreTokenizerConfig,
    ) -> Self {
        let merge_ranks: HashMap<(Vec<u8>, Vec<u8>), u32> = merges
            .merges
            .into_iter()
            .enumerate()
            .map(|(rank, pair)| (pair, rank as u32))
            .collect();
        let pattern = match &pre_tokenizer {
            PreTokenizerConfig::Regex(p) => Some(p.as_str()),
            PreTokenizerConfig::ByteLevel { regex: Some(p) } => Some(p.as_str()),
            _ => None,
        };
        let split_re = pattern.and_then(|p| regex_automata::meta::Regex::new(p).ok());
        Self {
            vocab,
            merge_ranks,
            byte_fallback,
            pre_tokenizer,
            split_re,
        }
    }

    /// Encode text into token IDs.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }
        let pre_tokens = self.pre_tokenize(text);
        let mut output = Vec::new();
        for word in &pre_tokens {
            let ids = self.encode_word(word.as_bytes());
            output.extend(ids);
        }
        output
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, token_ids: &[u32]) -> String {
        let is_byte_level = matches!(self.pre_tokenizer, PreTokenizerConfig::ByteLevel { .. });

        let mut bytes = Vec::new();
        for &id in token_ids {
            if let Some(token_bytes) = self.vocab.id_to_token.get(id as usize) {
                let token_str = self.vocab.id_to_str(id);
                // Check for byte fallback tokens like <0xNN>
                if self.byte_fallback {
                    if let Some(s) = token_str {
                        if let Some(byte_val) = parse_byte_fallback(s) {
                            bytes.push(byte_val);
                            continue;
                        }
                    }
                }
                bytes.extend_from_slice(token_bytes);
            }
        }

        if is_byte_level {
            // Byte-level BPE (GPT-2 / Qwen): the vocab stores Unicode-mapped
            // characters. Reverse the byte_to_unicode mapping to recover the
            // original bytes, then decode as UTF-8.
            let unicode_to_byte = unicode_to_byte_table();
            let text = String::from_utf8_lossy(&bytes);
            let raw_bytes: Vec<u8> = text
                .chars()
                .filter_map(|c| unicode_to_byte.get(&c).copied())
                .collect();
            String::from_utf8_lossy(&raw_bytes).into_owned()
        } else {
            // Metaspace (SentencePiece) encoding: replace ▁ (U+2581) with space
            let text = String::from_utf8_lossy(&bytes);
            let text = text.replace('\u{2581}', " ");
            // Strip leading space from Metaspace prepend
            let text = text.strip_prefix(' ').unwrap_or(&text);
            text.to_string()
        }
    }

    pub fn vocab(&self) -> &VocabTable {
        &self.vocab
    }

    // ── Pre-tokenization ────────────────────────────────────────────────

    fn pre_tokenize(&self, text: &str) -> Vec<String> {
        match &self.pre_tokenizer {
            PreTokenizerConfig::None => vec![text.to_string()],
            PreTokenizerConfig::Metaspace {
                replacement,
                prepend,
            } => self.pre_tokenize_metaspace(text, *replacement, *prepend),
            PreTokenizerConfig::Regex(_) => {
                // GPT-2 / LLaMA-3 regex split, no byte→unicode remapping.
                self.split_fragments(text)
                    .into_iter()
                    .map(|f| f.to_string())
                    .collect()
            }
            PreTokenizerConfig::ByteLevel { .. } => self.pre_tokenize_byte_level(text),
        }
    }

    /// Split `text` into pre-token fragments using the compiled regex
    /// (`self.split_re`), falling back to the whole text when no pattern is
    /// configured or it failed to compile.
    fn split_fragments<'t>(&self, text: &'t str) -> Vec<&'t str> {
        match &self.split_re {
            Some(re) => re.find_iter(text).map(|m| &text[m.range()]).collect(),
            None => vec![text],
        }
    }

    /// Byte-level pre-tokenization (GPT-2 / Qwen style):
    /// 1. Optionally split text using the compiled regex pattern
    /// 2. Map each byte to a Unicode character via the GPT-2 byte-to-unicode table
    fn pre_tokenize_byte_level(&self, text: &str) -> Vec<String> {
        let table = byte_to_unicode_table();
        // Map each fragment's bytes through the byte-to-unicode table
        self.split_fragments(text)
            .into_iter()
            .map(|frag| frag.bytes().map(|b| table[b as usize]).collect::<String>())
            .collect()
    }

    /// Metaspace pre-tokenization (SentencePiece style):
    /// 1. Replace all spaces with the replacement character
    /// 2. Optionally prepend the replacement character at the start
    /// 3. Split into individual "words" (each starting with the replacement char)
    fn pre_tokenize_metaspace(&self, text: &str, replacement: char, prepend: bool) -> Vec<String> {
        if text.is_empty() {
            return Vec::new();
        }

        // Replace spaces with replacement char and prepend if needed
        let replaced = text.replace(' ', &replacement.to_string());
        let full = if prepend {
            format!("{replacement}{replaced}")
        } else {
            replaced
        };

        // Split into words at replacement boundaries, keeping the replacement
        // char attached to the following word.
        let mut words = Vec::new();
        let mut current = String::new();

        for ch in full.chars() {
            if ch == replacement && !current.is_empty() {
                words.push(current);
                current = String::new();
            }
            current.push(ch);
        }
        if !current.is_empty() {
            words.push(current);
        }
        words
    }

    // ── BPE merging ─────────────────────────────────────────────────────

    /// Encode a single pre-tokenized word to token IDs.
    fn encode_word(&self, word: &[u8]) -> Vec<u32> {
        if word.is_empty() {
            return Vec::new();
        }

        // Start with character-level tokens. For each character in the word,
        // look up the character as a token. If byte_fallback is enabled and
        // the character token isn't in vocab, use byte tokens.
        let mut pieces: Vec<Vec<u8>> = Vec::new();

        // For Metaspace/SentencePiece style tokenizers, the vocab uses
        // full UTF-8 chars (including multi-byte), not individual bytes.
        // Start with character-level splitting.
        let word_str = String::from_utf8_lossy(word);
        for ch in word_str.chars() {
            let ch_bytes = ch.to_string().into_bytes();
            if self.vocab.token_to_id.contains_key(&ch_bytes) {
                pieces.push(ch_bytes);
            } else if self.byte_fallback {
                // Fall back to individual byte tokens <0xNN>
                for &b in ch.to_string().as_bytes() {
                    let byte_token = format!("<0x{b:02X}>").into_bytes();
                    pieces.push(byte_token);
                }
            } else {
                // Unknown token — use as-is, will fail lookup later
                pieces.push(ch_bytes);
            }
        }

        // Iteratively merge: find the pair with the lowest rank, merge it.
        loop {
            if pieces.len() < 2 {
                break;
            }

            // Find the pair with the lowest merge rank.
            let mut best_rank = u32::MAX;
            let mut best_idx = usize::MAX;

            for i in 0..pieces.len() - 1 {
                let pair = (pieces[i].clone(), pieces[i + 1].clone());
                if let Some(&rank) = self.merge_ranks.get(&pair) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_idx = i;
                    }
                }
            }

            if best_idx == usize::MAX {
                break; // No more merges apply
            }

            // Merge the pair at best_idx
            let merged = [pieces[best_idx].as_slice(), pieces[best_idx + 1].as_slice()].concat();
            pieces[best_idx] = merged;
            pieces.remove(best_idx + 1);
        }

        // Map final pieces to token IDs.
        pieces
            .iter()
            .map(|piece| {
                self.vocab
                    .token_to_id
                    .get(piece)
                    .copied()
                    .unwrap_or_else(|| {
                        // Unknown token
                        self.vocab
                            .token_to_id
                            .get("<unk>".as_bytes())
                            .copied()
                            .unwrap_or(0)
                    })
            })
            .collect()
    }
}

/// GPT-2 byte-to-unicode mapping.
///
/// Maps each byte value (0–255) to a unique Unicode character. Printable ASCII
/// characters map to themselves; non-printable bytes map to characters starting
/// at U+0100 (Ā, ā, Ă, ...). This is the standard mapping used by GPT-2, Qwen,
/// and other byte-level BPE tokenizers.
fn byte_to_unicode_table() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut n: u32 = 0;
    for b in 0u8..=255 {
        let ch = match b {
            // Printable ranges that map to themselves:
            // '!' (33) through '~' (126), and '¡' (161) through '¬' (172),
            // and '®' (174) through 'ÿ' (255)
            33..=126 | 161..=172 | 174..=255 => b as u32,
            // Non-printable bytes map to U+0100 + offset
            _ => {
                let ch = 256 + n;
                n += 1;
                ch
            }
        };
        table[b as usize] = char::from_u32(ch).unwrap_or('?');
    }
    table
}

/// Reverse mapping: Unicode char → original byte value.
///
/// Inverts `byte_to_unicode_table()` for decoding byte-level BPE tokens
/// back to raw bytes.
fn unicode_to_byte_table() -> HashMap<char, u8> {
    let forward = byte_to_unicode_table();
    let mut reverse = HashMap::with_capacity(256);
    for (byte_val, &ch) in forward.iter().enumerate() {
        reverse.insert(ch, byte_val as u8);
    }
    reverse
}

/// Parse a byte-fallback token like `<0x41>` → Some(0x41).
fn parse_byte_fallback(token: &str) -> Option<u8> {
    let inner = token.strip_prefix("<0x")?.strip_suffix('>')?;
    u8::from_str_radix(inner, 16).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_byte_fallback_valid() {
        assert_eq!(parse_byte_fallback("<0x41>"), Some(0x41));
        assert_eq!(parse_byte_fallback("<0xFF>"), Some(0xFF));
        assert_eq!(parse_byte_fallback("<0x00>"), Some(0x00));
    }

    #[test]
    fn parse_byte_fallback_invalid() {
        assert_eq!(parse_byte_fallback("hello"), None);
        assert_eq!(parse_byte_fallback("<0x>"), None);
        assert_eq!(parse_byte_fallback("<0xGG>"), None);
    }

    #[test]
    fn metaspace_pre_tokenize() {
        let vocab = VocabTable::new(vec![]);
        let merges = MergeRules { merges: vec![] };
        let enc = BpeEncoder::new(
            vocab,
            merges,
            false,
            PreTokenizerConfig::Metaspace {
                replacement: '\u{2581}',
                prepend: true,
            },
        );

        let words = enc.pre_tokenize("hello world");
        assert_eq!(words, vec!["▁hello", "▁world"]);
    }

    #[test]
    fn metaspace_pre_tokenize_no_prepend() {
        let vocab = VocabTable::new(vec![]);
        let merges = MergeRules { merges: vec![] };
        let enc = BpeEncoder::new(
            vocab,
            merges,
            false,
            PreTokenizerConfig::Metaspace {
                replacement: '\u{2581}',
                prepend: false,
            },
        );

        let words = enc.pre_tokenize("hello world");
        assert_eq!(words, vec!["hello", "▁world"]);
    }

    #[test]
    fn encode_empty() {
        let vocab = VocabTable::new(vec![]);
        let merges = MergeRules { merges: vec![] };
        let enc = BpeEncoder::new(vocab, merges, false, PreTokenizerConfig::None);
        assert!(enc.encode("").is_empty());
    }
}
