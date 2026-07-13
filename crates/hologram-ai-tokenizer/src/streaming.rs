//! Incremental streaming detokenizer.
//!
//! Re-decoding the whole generated sequence to stream each new token makes a
//! generation loop O(N²) in decoded tokens. [`StreamingDecoder`] makes the
//! running text O(N) total: each fed id re-decodes only a short window — the
//! tokens flushed at the last emission (the *overlap*) plus the tokens not yet
//! emitted (the *held* tail) — and returns the new byte delta. The window
//! never grows with the sequence, only with how long a character stays
//! incomplete, so per-token work is O(1) amortized, not O(position).
//!
//! Two properties of [`Tokenizer::decode`] make a naive per-token decode of
//! just the new id wrong, and this decoder handles both:
//!
//! - **Segment decodes are not concatenative.** `decode` applies per-call
//!   artifacts at the segment start (the Metaspace path replaces ▁ with a
//!   space and strips ONE leading space per call), so
//!   `decode(a) + decode(b) ≠ decode(a ++ b)`. The overlap cancels the
//!   artifact: every delta is `decode(overlap ++ held)` minus its
//!   `decode(overlap)` prefix, and both decodes start at the same token, so
//!   the artifact is byte-identical in both and vanishes in the diff.
//! - **A character may span several tokens.** Byte-fallback pieces
//!   (`<0xE2><0x96><0x81>`) and byte-level BPE pieces cut UTF-8 sequences
//!   mid-character; `decode` of the incomplete tail ends in U+FFFD. Emission
//!   holds while the window text ends with U+FFFD — the next token may extend
//!   the tail into the real character, and streaming the replacement char
//!   would emit bytes the final decode does not contain. Because a flush
//!   therefore only ever happens on a clean tail, the overlap's bytes begin
//!   and end on real character boundaries, which is what keeps the diff
//!   prefix exact.
//!
//! The law (witnessed in `tests/streaming_detok.rs`): the concatenation of
//! every [`feed`](StreamingDecoder::feed) delta plus
//! [`finish`](StreamingDecoder::finish) is byte-identical to
//! `decode(all_ids)`.

use alloc::string::String;
use alloc::vec::Vec;

use crate::Tokenizer;

/// Streaming detokenizer over any [`Tokenizer`] — feed token ids one at a
/// time, get back the newly stable text delta. See the module docs for the
/// mechanism and the byte-identity law.
pub struct StreamingDecoder<'a> {
    tokenizer: &'a dyn Tokenizer,
    /// Tokens flushed at the last emission — already emitted, kept as left
    /// context so segment-start decode artifacts cancel in the diff.
    overlap: Vec<u32>,
    /// Tokens fed since the last emission — their text is not yet emitted.
    held: Vec<u32>,
    /// Scratch for `overlap ++ held` (avoids a per-feed allocation).
    window: Vec<u32>,
    /// `decode(overlap)` — the already-emitted prefix of `window_text`.
    prefix_text: String,
    /// `decode(overlap ++ held)`.
    window_text: String,
}

impl<'a> StreamingDecoder<'a> {
    pub fn new(tokenizer: &'a dyn Tokenizer) -> Self {
        Self {
            tokenizer,
            overlap: Vec::new(),
            held: Vec::new(),
            window: Vec::new(),
            prefix_text: String::new(),
            window_text: String::new(),
        }
    }

    /// Feed one token id; returns the newly stable text (empty while the tail
    /// is still an incomplete character or contributes nothing visible yet).
    pub fn feed(&mut self, id: u32) -> String {
        self.held.push(id);
        self.window.clear();
        self.window.extend_from_slice(&self.overlap);
        self.window.extend_from_slice(&self.held);
        self.window_text = self.tokenizer.decode(&self.window);
        // Hold while nothing new is visible, or while the tail is the lossy
        // replacement for a byte sequence that is not yet complete.
        if self.window_text.len() <= self.prefix_text.len()
            || self.window_text.ends_with('\u{FFFD}')
        {
            return String::new();
        }
        let delta = String::from(self.unemitted());
        // Flush: the held tokens become the next overlap — emitted text whose
        // decode anchors the next diff at the same segment start.
        self.overlap.clear();
        self.overlap.append(&mut self.held);
        self.prefix_text = self.tokenizer.decode(&self.overlap);
        self.window_text.clone_from(&self.prefix_text);
        delta
    }

    /// Text decoded but not yet emitted (the unstable tail). At every point
    /// `emitted deltas + pending()` equals `decode(all ids fed)` — this is
    /// what an exact stop-string scan reads alongside the deltas.
    pub fn pending(&self) -> &str {
        self.unemitted()
    }

    /// Flush whatever is still held back (an unfinished multi-token character
    /// at the very end of a generation decodes here, U+FFFD and all — exactly
    /// as the one-shot decode of the full sequence would render it).
    pub fn finish(&mut self) -> String {
        let rest = String::from(self.unemitted());
        self.overlap.append(&mut self.held);
        self.prefix_text.clone_from(&self.window_text);
        rest
    }

    /// `window_text` minus its emitted `prefix_text` prefix. The prefix
    /// relation is the decoder's soundness invariant: if `decode` were not
    /// left-stable the already-emitted bytes would be wrong, so a violation
    /// fails loud instead of streaming silently corrupt text.
    fn unemitted(&self) -> &str {
        assert!(
            self.window_text
                .as_bytes()
                .starts_with(self.prefix_text.as_bytes()),
            "streaming detokenizer invariant broken: decode(overlap ++ held) does not extend \
             decode(overlap) byte-for-byte — this Tokenizer's decode is not left-stable, so \
             already-emitted text cannot be trusted"
        );
        &self.window_text[self.prefix_text.len()..]
    }
}
