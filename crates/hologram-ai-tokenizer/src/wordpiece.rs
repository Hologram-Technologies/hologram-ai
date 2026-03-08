//! WordPiece encoder (greedy longest-prefix match).

use crate::vocab::VocabTable;

/// WordPiece encoder using greedy longest-prefix matching.
pub struct WordPieceEncoder<'a> {
    vocab: &'a VocabTable,
    continuing_prefix: &'a str,
    max_input_chars_per_word: usize,
}

impl<'a> WordPieceEncoder<'a> {
    pub fn new(
        vocab: &'a VocabTable,
        continuing_prefix: &'a str,
        max_input_chars_per_word: usize,
    ) -> Self {
        Self {
            vocab,
            continuing_prefix,
            max_input_chars_per_word,
        }
    }

    /// Encode a single word into token IDs using greedy longest-prefix match.
    ///
    /// Words longer than `max_input_chars_per_word` are mapped to `[UNK]`.
    pub fn encode_word(&self, word: &str) -> Vec<u32> {
        if word.chars().count() > self.max_input_chars_per_word {
            return if let Some(unk) = self.vocab.str_to_id("[UNK]") {
                vec![unk]
            } else {
                vec![]
            };
        }

        let mut tokens = Vec::new();
        let mut start = 0;
        let bytes = word.as_bytes();

        while start < bytes.len() {
            let mut end = bytes.len();
            let mut matched = false;

            while end > start {
                let substr = &bytes[start..end];
                if let Ok(s) = std::str::from_utf8(substr) {
                    let candidate = if start > 0 {
                        format!("{}{}", self.continuing_prefix, s)
                    } else {
                        s.to_string()
                    };
                    if let Some(id) = self.vocab.str_to_id(&candidate) {
                        tokens.push(id);
                        matched = true;
                        break;
                    }
                }
                // Back off by one character (respect UTF-8 boundaries).
                end = prev_char_boundary(bytes, end);
            }

            if !matched {
                // No match found — emit UNK and advance one character.
                if let Some(unk) = self.vocab.str_to_id("[UNK]") {
                    tokens.push(unk);
                }
                end = next_char_boundary(bytes, start + 1);
            }
            start = end;
        }

        tokens
    }
}

/// Find the previous UTF-8 character boundary before `pos`.
fn prev_char_boundary(bytes: &[u8], mut pos: usize) -> usize {
    if pos == 0 {
        return 0;
    }
    pos -= 1;
    while pos > 0 && (bytes[pos] & 0xC0) == 0x80 {
        pos -= 1;
    }
    pos
}

/// Find the next UTF-8 character boundary at or after `pos`.
fn next_char_boundary(bytes: &[u8], mut pos: usize) -> usize {
    while pos < bytes.len() && (bytes[pos] & 0xC0) == 0x80 {
        pos += 1;
    }
    pos
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_known_word() {
        let vocab = VocabTable::new(vec![
            b"un".to_vec(),
            b"##aff".to_vec(),
            b"##able".to_vec(),
            b"[UNK]".to_vec(),
        ]);
        let enc = WordPieceEncoder::new(&vocab, "##", 200);
        let ids = enc.encode_word("unaffable");
        // "un" + "##aff" + "##able"
        assert_eq!(ids, vec![0, 1, 2]);
    }

    #[test]
    fn too_long_word_gives_unk() {
        let vocab = VocabTable::new(vec![b"[UNK]".to_vec()]);
        let enc = WordPieceEncoder::new(&vocab, "##", 3);
        let ids = enc.encode_word("toolong");
        assert_eq!(ids, vec![0]); // [UNK]
    }
}
