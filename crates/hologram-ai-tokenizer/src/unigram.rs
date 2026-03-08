//! SentencePiece Unigram encoder (Viterbi segmentation).

use crate::vocab::VocabTable;

/// Unigram encoder using Viterbi dynamic programming.
pub struct UnigramEncoder<'a> {
    vocab: &'a VocabTable,
    scores: &'a [f32],
}

impl<'a> UnigramEncoder<'a> {
    pub fn new(vocab: &'a VocabTable, scores: &'a [f32]) -> Self {
        Self { vocab, scores }
    }

    /// Encode text into token IDs using Viterbi segmentation.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }
        let bytes = text.as_bytes();
        let n = bytes.len();

        // best_score[i] = best log-probability for segmenting bytes[0..i]
        let mut best_score = vec![f64::NEG_INFINITY; n + 1];
        // back_ptr[i] = (start_pos, token_id) of the best token ending at position i
        let mut back_ptr: Vec<Option<(usize, u32)>> = vec![None; n + 1];
        best_score[0] = 0.0;

        for i in 0..n {
            if best_score[i] == f64::NEG_INFINITY {
                continue;
            }
            // Try all substrings starting at position i.
            let max_len = (n - i).min(128); // cap max token length
            for len in 1..=max_len {
                let end = i + len;
                let substr = &bytes[i..end];
                if let Ok(s) = std::str::from_utf8(substr) {
                    if let Some(id) = self.vocab.str_to_id(s) {
                        let score = if (id as usize) < self.scores.len() {
                            self.scores[id as usize] as f64
                        } else {
                            -10.0 // default penalty for unknown score
                        };
                        let total = best_score[i] + score;
                        if total > best_score[end] {
                            best_score[end] = total;
                            back_ptr[end] = Some((i, id));
                        }
                    }
                }
            }

            // Byte fallback: if no token matched from position i, treat single byte as unk.
            if i < n && back_ptr[i + 1].is_none() && best_score[i + 1] == f64::NEG_INFINITY {
                let byte_token = format!("<0x{:02X}>", bytes[i]);
                if let Some(id) = self.vocab.str_to_id(&byte_token) {
                    let total = best_score[i] + (-100.0);
                    if total > best_score[i + 1] {
                        best_score[i + 1] = total;
                        back_ptr[i + 1] = Some((i, id));
                    }
                }
            }
        }

        // Backtrack to recover the best segmentation.
        let mut tokens = Vec::new();
        let mut pos = n;
        while pos > 0 {
            if let Some((start, id)) = back_ptr[pos] {
                tokens.push(id);
                pos = start;
            } else {
                // No valid segmentation — shouldn't happen with byte fallback.
                break;
            }
        }
        tokens.reverse();
        tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_empty() {
        let vocab = VocabTable::new(Vec::new());
        let scores = Vec::new();
        let enc = UnigramEncoder::new(&vocab, &scores);
        assert!(enc.encode("").is_empty());
    }

    #[test]
    fn encode_known_tokens() {
        let vocab = VocabTable::new(vec![
            b"h".to_vec(),
            b"e".to_vec(),
            b"l".to_vec(),
            b"he".to_vec(),
            b"ll".to_vec(),
            b"o".to_vec(),
        ]);
        // Scores: longer matches should have better (higher) scores
        let scores = vec![-2.0, -2.0, -2.0, -1.0, -1.0, -2.0];
        let enc = UnigramEncoder::new(&vocab, &scores);
        let ids = enc.encode("hello");
        // Should prefer "he" + "ll" + "o" (score -4.0) over "h"+"e"+"l"+"l"+"o" (score -10.0)
        assert_eq!(ids, vec![3, 4, 5]); // "he", "ll", "o"
    }
}
