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
//!   finding the most recent earlier occurrence of the LONGEST recurring suffix
//!   of the realized sequence and copying what followed it. No draft model, no
//!   training, no weights, and no n-gram cap — parametric over any model and any
//!   input, at O(n) per draft. It shines
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

/// Why a TARGET↔DRAFT pairing is refused, or `None` when the draft is
/// compatible (row `speculative-draft-pairing`). The draft consumes the
/// TARGET's token ids and carries the target's realized sequence, so it must
/// cover BOTH the target's vocabulary (every target id indexes the draft's
/// embedding) and its context (a shorter-context draft would abort its own
/// forward the moment the sequence crossed its window). A refusal is not an
/// error — the caller falls back to prompt-lookup — and because the output is
/// the target's regardless of the drafter, this policy only avoids an
/// out-of-range Gather or a mid-turn window abort, never affects correctness.
///
/// Pure and parametric (no model identity, no size constant): the single source
/// of the pairing rule, called by the browser's `attach_draft` and tested here.
pub fn draft_pairing_refusal(
    target_vocab: u64,
    target_context: u64,
    draft_vocab: u64,
    draft_context: u64,
) -> Option<String> {
    if target_vocab == 0 {
        return Some(
            "the target declares no vocabulary size — the draft's coverage cannot be verified"
                .to_string(),
        );
    }
    if draft_vocab < target_vocab {
        return Some(format!(
            "draft vocabulary ({draft_vocab}) does not cover the target's ({target_vocab}) — the \
             draft would index its embedding out of range"
        ));
    }
    if draft_context < target_context {
        return Some(format!(
            "draft context ({draft_context}) is shorter than the target's ({target_context}) — \
             the target's realized sequence would exceed the draft's window and abort its forward"
        ));
    }
    None
}

/// The zero-weight prompt-lookup drafter (the shipped default): stateless — it
/// reads the realized sequence and needs no prefill or commit. It carries no
/// tuning parameter: the context it drafts from is the LONGEST recurrence the
/// sequence itself contains.
pub struct PromptLookupDrafter;

impl Drafter for PromptLookupDrafter {
    fn propose(&mut self, realized: &[i64], cap: usize) -> Result<Vec<i64>> {
        Ok(prompt_lookup_draft(realized, cap))
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
/// The drafting context is the LONGEST suffix of `seq` that occurs earlier in
/// `seq` at all — there is no n-gram cap, because a cap is a guess about how
/// much context an arbitrary input recurs over. Among earlier occurrences of
/// that longest suffix the MOST RECENT is used (the freshest continuation), and
/// the up-to-`max_draft` tokens that followed it are the draft. Returns an empty
/// draft when the suffix never recurs — the signal to fall back to a single
/// decode step.
///
/// Found in O(n) time and space with the Z-function of the REVERSED sequence:
/// `z[i]` is the longest common prefix of `rev` and `rev[i..]`, i.e. the longest
/// common SUFFIX of `seq` and `seq[..n-i]` — an earlier occurrence ending at
/// `n - i`. Scanning `i` ascending and keeping strictly longer matches yields the
/// longest suffix, tie-broken toward the most recent occurrence. (The former
/// capped scan was `O(n² · g)`, which an arbitrarily long realized sequence
/// cannot afford.)
///
/// This never affects correctness, only speed: every drafted token is verified
/// and a wrong one is rejected. It is a pure function of the realized sequence.
pub fn prompt_lookup_draft(seq: &[i64], max_draft: usize) -> Vec<i64> {
    let n = seq.len();
    if n < 2 || max_draft == 0 {
        return Vec::new();
    }
    let rev: Vec<i64> = seq.iter().rev().copied().collect();
    let z = z_function(&rev);

    // `best_end` is the EXCLUSIVE end of the earlier occurrence of the longest
    // recurring suffix; the draft is what followed it.
    let (mut best_len, mut best_end) = (0usize, 0usize);
    for (i, &zi) in z.iter().enumerate().skip(1) {
        // The occurrence must fit entirely before the position it ends at.
        let len = zi.min(n - i);
        if len > best_len {
            best_len = len;
            best_end = n - i;
        }
    }
    if best_len == 0 {
        return Vec::new();
    }
    let take = max_draft.min(n - best_end);
    if take == 0 {
        return Vec::new();
    }
    seq[best_end..best_end + take].to_vec()
}

/// `z[i]` = length of the longest common prefix of `s` and `s[i..]` (`z[0] = n`).
/// Linear time, the standard two-pointer construction.
fn z_function(s: &[i64]) -> Vec<usize> {
    let n = s.len();
    let mut z = vec![0usize; n];
    if n == 0 {
        return z;
    }
    z[0] = n;
    let (mut l, mut r) = (0usize, 0usize);
    for i in 1..n {
        if i < r {
            z[i] = (r - i).min(z[i - l]);
        }
        while i + z[i] < n && s[z[i]] == s[i + z[i]] {
            z[i] += 1;
        }
        if i + z[i] > r {
            l = i;
            r = i + z[i];
        }
    }
    z
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_or_trivial_sequence_drafts_nothing() {
        assert!(prompt_lookup_draft(&[], 4).is_empty());
        assert!(prompt_lookup_draft(&[7], 4).is_empty());
        // A zero draft budget disables drafting.
        assert!(prompt_lookup_draft(&[1, 2, 3], 0).is_empty());
    }

    #[test]
    fn repeated_suffix_drafts_its_earlier_continuation() {
        // "a b c d ... a b" → the suffix `a b` last continued with `c d e`.
        let seq = [10, 20, 30, 40, 99, 10, 20];
        assert_eq!(
            prompt_lookup_draft(&seq, 3),
            vec![30, 40, 99],
            "drafts what followed the earlier `10,20`"
        );
    }

    #[test]
    fn longest_suffix_wins_over_a_shorter_ambiguous_one() {
        // The 2-gram `2 3` occurred once (→ 4); the 1-gram `3` also occurred
        // earlier. The longer, more specific match is used.
        let seq = [1, 2, 3, 4, 5, 2, 3];
        assert_eq!(prompt_lookup_draft(&seq, 2), vec![4, 5]);
    }

    #[test]
    fn most_recent_occurrence_is_preferred() {
        // `7` appears after position 0 (→ 1) and position 3 (→ 8). Among equally
        // long matches the freshest is used.
        let seq = [7, 1, 2, 7, 8, 9, 7];
        assert_eq!(prompt_lookup_draft(&seq, 1), vec![8]);
    }

    #[test]
    fn max_draft_caps_the_length() {
        let seq = [5, 6, 7, 8, 9, 5, 6];
        assert_eq!(prompt_lookup_draft(&seq, 2), vec![7, 8]);
        assert_eq!(prompt_lookup_draft(&seq, 10), vec![7, 8, 9, 5, 6]);
    }

    #[test]
    fn novel_suffix_drafts_nothing() {
        // The trailing `42` never occurred earlier → no draft, fall back to a step.
        let seq = [1, 2, 3, 4, 42];
        assert!(prompt_lookup_draft(&seq, 4).is_empty());
    }

    #[test]
    fn a_longer_context_beats_a_more_recent_shorter_one() {
        // The suffix `1,2,3` recurs (ending at 3) and continued with `7,8`.
        // The SHORTER suffix `2,3` has a MORE RECENT earlier occurrence (ending
        // at 7) that continued with `9`. Specificity wins: the longest recurring
        // context is the right one. A 2-token n-gram cap would have drafted `9`.
        let seq = [1, 2, 3, 7, 8, 2, 3, 9, 1, 2, 3];
        assert_eq!(prompt_lookup_draft(&seq, 2), vec![7, 8]);
    }

    #[test]
    fn the_drafter_has_no_context_cap_and_stays_linear() {
        // An arbitrarily long recurrence is matched in full — no n-gram ceiling.
        // 4096 tokens is far past any cap a tuning constant would have imposed,
        // and the O(n) search returns immediately.
        let period: Vec<i64> = (0..4096).collect();
        let mut seq = period.clone();
        seq.push(-1); // a separator so the recurrence is the whole period
        seq.extend_from_slice(&period);
        // The longest recurring suffix is the entire 4096-token period, whose
        // earlier occurrence ended just before the separator.
        assert_eq!(prompt_lookup_draft(&seq, 1), vec![-1]);
    }

    #[test]
    fn z_function_matches_a_naive_reference() {
        let s: Vec<i64> = [1, 2, 1, 2, 1, 3, 1, 2, 1].to_vec();
        let z = z_function(&s);
        for i in 0..s.len() {
            let naive = (0..s.len() - i).take_while(|&k| s[k] == s[i + k]).count();
            assert_eq!(z[i], naive, "z[{i}]");
        }
    }

    /// Simulate GREEDY speculative decode over a fixed token sequence `seq` (the
    /// output the target would greedily produce): at each realized prefix,
    /// prompt-lookup proposes up to `k`, the model "accepts" the longest prefix
    /// matching `seq` (exact for greedy — the sequence IS the model's output),
    /// and the step advances `accepted + 1` (the bonus is the model's own token).
    /// Returns (mean tokens advanced per forward pass, fraction of passes that
    /// drafted). Mean-tokens-per-pass IS the speculative speedup in forward passes.
    fn simulate_greedy_speculation(seq: &[i64], k: usize) -> (f64, f64) {
        let mut i = 1usize; // the first token is the prefill's last row
        let (mut passes, mut drafted) = (0usize, 0usize);
        while i < seq.len() {
            passes += 1;
            let draft = prompt_lookup_draft(&seq[..i], k);
            if draft.is_empty() {
                i += 1; // plain step — never worse
                continue;
            }
            drafted += 1;
            let mut acc = 0usize;
            while acc < draft.len() && i + acc < seq.len() && draft[acc] == seq[i + acc] {
                acc += 1;
            }
            i += acc + 1; // accepted tokens + the model's own bonus token
        }
        let tokens = (seq.len() - 1) as f64;
        (tokens / passes as f64, drafted as f64 / passes as f64)
    }

    #[test]
    fn speculation_wins_on_recurrence_and_never_loses_on_novel_text() {
        // WITNESS for defaulting speculative on: on text with recurrence it
        // advances several tokens per pass; on novel text the drafter proposes
        // nothing, so it falls back to plain decode (exactly 1 token/pass — never
        // worse). This is the property that makes prompt-lookup safe by default.

        // Structured/echoing output (a chatbot repeating a list/format/quote):
        // a block that recurs verbatim — the case prompt-lookup targets. Once the
        // recurrence dominates the sequence the per-pass advance climbs well above 1.
        let block: Vec<i64> = (100..150).collect();
        let mut structured = vec![1i64, 2];
        for _ in 0..4 {
            structured.extend_from_slice(&block); // the model re-emits the block
        }
        let (spd, drafted) = simulate_greedy_speculation(&structured, 8);
        assert!(
            spd >= 1.5,
            "recurrence must give a real speculative speedup, got {spd:.2} tokens/pass"
        );
        assert!(drafted > 0.0);

        // Novel output (no recurrence): every token distinct. The drafter returns
        // empty every step → plain decode → exactly 1 token/pass (never worse).
        let novel: Vec<i64> = (0..200).collect();
        let (spd_novel, drafted_novel) = simulate_greedy_speculation(&novel, 8);
        assert_eq!(
            spd_novel, 1.0,
            "novel text must fall back to plain decode (1 token/pass), got {spd_novel:.2}"
        );
        assert_eq!(drafted_novel, 0.0, "novel text must never draft");
    }
}
