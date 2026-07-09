//! Speculative decoding (dictionary row `speculative-decode`): the parametric,
//! zero-weight drafter.
//!
//! A decode step is the substrate matmul kernel's worst shape (`M = 1`); the
//! `decode_shape` bench measures an `M = K` pass at a fraction of the wall-clock
//! of `K` single steps. Speculative decoding exploits that: DRAFT `K`
//! continuation tokens cheaply, VERIFY them all in one `M = K` pass (the verify
//! head, [`crate::decode::DecodeSession::verify`]), and ACCEPT the longest
//! prefix the model would itself have produced. For greedy (temperature 0) the
//! result is byte-identical to single-step decode — a pure speedup whose size is
//! the draft's acceptance rate.
//!
//! Two drafters share the ONE verify/accept/commit loop, because that loop is
//! drafter-AGNOSTIC — only the TARGET's own token rule decides what is emitted,
//! so a drafter changes the acceptance rate, never the output:
//!
//! - **prompt-lookup** ([`PromptLookupDrafter`]): the next tokens are guessed by
//!   finding the most recent earlier occurrence of the current suffix in the
//!   realized sequence and copying what followed it. No draft model, no
//!   training, no weights — parametric over any model and any input. It shines
//!   on structured/repetitive text (code, JSON, retrieval, format echoing) and
//!   returns nothing on novel text, so the caller falls back to one plain step.
//! - **draft model** ([`ModelDrafter`]): a small second model (sharing the
//!   target's tokenizer/vocab) proposes the continuation from its own cheaper
//!   forward — the general throughput lever for novel text, where prompt-lookup
//!   finds no recurrence.
//!
//! Either way every drafted token is verified and only the target's own tokens
//! are accepted, so drafting is never worse than not drafting.

use anyhow::Result;

use crate::decode::DecodeSession;
use crate::engine::LmSession;

/// A source of draft tokens for speculative decode. The verify/accept/commit
/// loop is drafter-agnostic, so any drafter plugs into it; a drafter only
/// affects the acceptance rate, never the tokens emitted (those are the
/// target's own, byte for byte).
pub trait Drafter {
    /// Prime the drafter to the target's prompt before the loop (a draft model
    /// prefills its own K/V; prompt-lookup is stateless, the default no-op).
    fn prefill(&mut self, prompt_tokens: &[i64]) -> Result<()> {
        let _ = prompt_tokens;
        Ok(())
    }
    /// Propose up to `cap` continuation tokens given the target's realized
    /// sequence. An empty proposal signals a plain step (never worse).
    fn propose(&mut self, realized: &[i64], cap: usize) -> Result<Vec<i64>>;
    /// Sync the drafter to the committed sequence: the target accepted
    /// `accepted` of the last proposal, then committed `bonus`.
    fn commit(&mut self, accepted: usize, bonus: i64) -> Result<()>;
}

/// The zero-weight prompt-lookup drafter (the shipped default): stateless — it
/// reads the realized sequence and needs no prefill or commit.
pub struct PromptLookupDrafter {
    pub ngram_max: usize,
}

impl Drafter for PromptLookupDrafter {
    fn propose(&mut self, realized: &[i64], cap: usize) -> Result<Vec<i64>> {
        Ok(prompt_lookup_draft(realized, self.ngram_max, cap))
    }
    fn commit(&mut self, _accepted: usize, _bonus: i64) -> Result<()> {
        Ok(())
    }
}

/// A small DRAFT MODEL as the drafter: a second [`DecodeSession`] (sharing the
/// target's tokenizer/vocab) proposes `cap` continuation tokens GREEDILY from
/// its own — cheaper — forward. The target verifies them in one `M = K` pass and
/// accepts the longest prefix IT would produce; on `commit` the drafter rewinds
/// to the accepted length (keeping the accepted tokens' K/V) and steps the
/// target's bonus, so both models carry the identical realized sequence. The
/// output is the TARGET's, byte for byte — the draft model only proposes, so
/// acceptance (speedup) tracks how well the small model predicts the large one.
pub struct ModelDrafter<S: LmSession> {
    session: DecodeSession<S>,
    /// The draft's next-token logits (empty until `prefill`).
    row: Vec<f32>,
    /// The realized length at the start of the last `propose`.
    start: usize,
}

impl<S: LmSession> ModelDrafter<S> {
    /// Wrap a draft decode session (not yet prefilled — [`Drafter::prefill`]
    /// primes it to the target's prompt).
    pub fn new(session: DecodeSession<S>) -> Self {
        Self {
            session,
            row: Vec::new(),
            start: 0,
        }
    }

    /// Reclaim the wrapped draft session after a turn. The drafter is
    /// constructed per turn (it borrows the target's realized sequence through
    /// `propose`/`commit`), but the draft session it owns — a compiled decode
    /// pipeline with resident stages — is expensive to rebuild. A warm caller
    /// (the browser's `DecodeChatSession`) `take`s the session into a fresh
    /// drafter each turn and returns it here, so the draft's residency survives
    /// across turns exactly as the target's does (row `speculative-draft-pairing`).
    pub fn into_session(self) -> DecodeSession<S> {
        self.session
    }
}

impl<S: LmSession> Drafter for ModelDrafter<S> {
    fn prefill(&mut self, prompt_tokens: &[i64]) -> Result<()> {
        // Fresh prefill of the whole prompt (a warm-turn common-prefix rewind is
        // a later optimization): the draft ends at the same realized sequence as
        // the target, which is all the loop's sync relies on.
        self.session.reset();
        self.row = self.session.feed(prompt_tokens)?;
        Ok(())
    }

    fn propose(&mut self, _realized: &[i64], cap: usize) -> Result<Vec<i64>> {
        self.start = self.session.realized_len();
        let mut draft = Vec::with_capacity(cap);
        for _ in 0..cap {
            let next = argmax(&self.row) as i64;
            draft.push(next);
            self.row = self.session.step(next)?;
        }
        Ok(draft)
    }

    fn commit(&mut self, accepted: usize, bonus: i64) -> Result<()> {
        // The draft over-generated `cap` tokens from `start`; keep the `accepted`
        // that the target confirmed (their K/V is the draft's own, correct), drop
        // the rejected tail, and step the target's bonus so both align.
        self.session.rewind_to(self.start + accepted);
        self.row = self.session.step(bonus)?;
        Ok(())
    }
}

/// Greedy argmax over a logit row (the draft model's proposal rule).
fn argmax(row: &[f32]) -> u32 {
    row.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Draft up to `max_draft` continuation tokens for `seq` by prompt-lookup.
///
/// For the largest suffix length `g ∈ [1, ngram_max]` that occurs earlier in
/// `seq`, take the most recent such earlier occurrence and copy the up-to-
/// `max_draft` tokens that followed it. A longer matched suffix is a more
/// specific context, so it is tried first (higher acceptance). Returns an empty
/// draft when nothing matches — the signal to fall back to a single decode step.
///
/// This never affects correctness, only speed: every drafted token is verified
/// and a wrong one is rejected. It is a pure function of the realized sequence.
pub fn prompt_lookup_draft(seq: &[i64], ngram_max: usize, max_draft: usize) -> Vec<i64> {
    let n = seq.len();
    if n < 2 || max_draft == 0 || ngram_max == 0 {
        return Vec::new();
    }
    // Longest matching suffix first: a more specific context accepts better.
    let g_max = ngram_max.min(n - 1);
    for g in (1..=g_max).rev() {
        let needle = &seq[n - g..];
        // Most recent earlier occurrence first (skip the trailing suffix itself).
        for start in (0..n - g).rev() {
            if &seq[start..start + g] == needle {
                let from = start + g;
                let take = max_draft.min(n - from);
                if take > 0 {
                    return seq[from..from + take].to_vec();
                }
            }
        }
    }
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_or_trivial_sequence_drafts_nothing() {
        assert!(prompt_lookup_draft(&[], 3, 4).is_empty());
        assert!(prompt_lookup_draft(&[7], 3, 4).is_empty());
        // A budget or ngram of zero disables drafting.
        assert!(prompt_lookup_draft(&[1, 2, 3], 0, 4).is_empty());
        assert!(prompt_lookup_draft(&[1, 2, 3], 3, 0).is_empty());
    }

    #[test]
    fn repeated_suffix_drafts_its_earlier_continuation() {
        // "a b c d ... a b" → the suffix `a b` last continued with `c d e`.
        let seq = [10, 20, 30, 40, 99, 10, 20];
        let draft = prompt_lookup_draft(&seq, 3, 3);
        assert_eq!(
            draft,
            vec![30, 40, 99],
            "drafts what followed the earlier `10,20`"
        );
    }

    #[test]
    fn longest_suffix_wins_over_a_shorter_ambiguous_one() {
        // The 2-gram `2 3` occurred once (→ 4); the 1-gram `3` also occurred
        // earlier (→ 4 as well). The longer, more specific match is used.
        let seq = [1, 2, 3, 4, 5, 2, 3];
        let draft = prompt_lookup_draft(&seq, 3, 2);
        assert_eq!(draft, vec![4, 5], "the 2-gram `2,3` context drafts `4,5`");
    }

    #[test]
    fn most_recent_occurrence_is_preferred() {
        // `7` appears after position 0 (→ 1) and position 3 (→ 8). The most
        // recent earlier occurrence (index 3) drafts its follower.
        let seq = [7, 1, 2, 7, 8, 9, 7];
        let draft = prompt_lookup_draft(&seq, 1, 1);
        assert_eq!(draft, vec![8], "the freshest match for `7` drafts `8`");
    }

    #[test]
    fn max_draft_caps_the_length() {
        let seq = [5, 6, 7, 8, 9, 5, 6];
        assert_eq!(prompt_lookup_draft(&seq, 3, 2), vec![7, 8]);
        assert_eq!(prompt_lookup_draft(&seq, 3, 10), vec![7, 8, 9, 5, 6]);
    }

    #[test]
    fn novel_suffix_drafts_nothing() {
        // The trailing `42` never occurred earlier → no draft, fall back to a step.
        let seq = [1, 2, 3, 4, 42];
        assert!(prompt_lookup_draft(&seq, 3, 4).is_empty());
    }
}
